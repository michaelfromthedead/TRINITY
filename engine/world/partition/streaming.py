"""
Streaming manager for world partition.

Provides streaming sources, configuration, budget management, and the
main WorldStreaming manager for orchestrating cell loading/unloading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.core.math.vec import Vec3
from engine.core.math.geometry import AABB
from engine.world.partition.cell import CellCoord, CellState, StreamingCell
from engine.world.partition.grid import WorldGrid
from engine.world.partition.constants import (
    DEFAULT_PLAYER_LOAD_RADIUS,
    DEFAULT_CAMERA_LOAD_RADIUS,
    DEFAULT_CAMERA_FORWARD_BIAS,
    DEFAULT_CUSTOM_LOAD_RADIUS,
    DEFAULT_STREAMING_LOAD_DISTANCE,
    DEFAULT_STREAMING_UNLOAD_DISTANCE,
    DEFAULT_STREAMING_HYSTERESIS,
    DEFAULT_PRIORITY_SCALE,
    DEFAULT_MAX_CONCURRENT_LOADS,
    DEFAULT_MAX_CONCURRENT_UNLOADS,
    DEFAULT_PRELOAD_DISTANCE,
    DEFAULT_MEMORY_BUDGET_MB,
    DEFAULT_IO_BANDWIDTH_MBPS,
    DEFAULT_FRAME_TIME_BUDGET_MS,
    DEFAULT_VOLUME_SIZE,
    PRIORITY_DISTANCE_BASE,
    PRIORITY_DISTANCE_DIVISOR,
    UNLOAD_PRIORITY_DIVISOR,
    HALF_MULTIPLIER,
)


class StreamingVolumeType(Enum):
    """Types of streaming volumes."""
    TRIGGER = auto()      # Load when player enters
    PRELOAD = auto()      # Load in advance when nearby
    BLOCKING = auto()     # Must be loaded before entering
    UNLOAD = auto()       # Force unload when entered


class StreamingPriority(Enum):
    """Streaming priority levels."""
    CRITICAL = 100  # Must load immediately (e.g., player spawn)
    HIGH = 75       # Important gameplay areas
    NORMAL = 50     # Standard streaming
    LOW = 25        # Background/distant areas
    BACKGROUND = 0  # Load when nothing else is pending


class StreamingSource(ABC):
    """
    Abstract base class for streaming sources.

    A streaming source is any entity that drives what content
    should be loaded (e.g., player, camera, prefetch point).
    """

    @property
    @abstractmethod
    def position(self) -> Vec3:
        """Get the current position of the streaming source."""
        pass

    @property
    def priority(self) -> int:
        """Get the priority of this source (higher = more important)."""
        return StreamingPriority.NORMAL.value

    @property
    def load_radius(self) -> float:
        """Get the load radius for this source."""
        return DEFAULT_PLAYER_LOAD_RADIUS

    @property
    def is_active(self) -> bool:
        """Check if this source is currently active."""
        return True


class PlayerStreamingSource(StreamingSource):
    """Streaming source that follows a player position."""

    def __init__(
        self,
        position: Optional[Vec3] = None,
        load_radius: float = DEFAULT_PLAYER_LOAD_RADIUS,
        priority: int = StreamingPriority.CRITICAL.value,
    ) -> None:
        """
        Initialize a player streaming source.

        Args:
            position: Initial position.
            load_radius: Radius to load around player.
            priority: Streaming priority.
        """
        self._position = position or Vec3(0, 0, 0)
        self._load_radius = load_radius
        self._priority = priority
        self._active = True

    @property
    def position(self) -> Vec3:
        """Get the player position."""
        return self._position

    @position.setter
    def position(self, value: Vec3) -> None:
        """Set the player position."""
        self._position = value

    @property
    def priority(self) -> int:
        """Get priority."""
        return self._priority

    @property
    def load_radius(self) -> float:
        """Get load radius."""
        return self._load_radius

    @load_radius.setter
    def load_radius(self, value: float) -> None:
        """Set load radius."""
        self._load_radius = value

    @property
    def is_active(self) -> bool:
        """Check if active."""
        return self._active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        """Set active state."""
        self._active = value


class CameraStreamingSource(StreamingSource):
    """Streaming source that follows a camera position."""

    def __init__(
        self,
        position: Optional[Vec3] = None,
        forward: Optional[Vec3] = None,
        load_radius: float = DEFAULT_CAMERA_LOAD_RADIUS,
        forward_bias: float = DEFAULT_CAMERA_FORWARD_BIAS,
    ) -> None:
        """
        Initialize a camera streaming source.

        Args:
            position: Initial position.
            forward: Forward direction vector.
            load_radius: Base load radius.
            forward_bias: Multiplier for load distance in view direction.
        """
        self._position = position or Vec3(0, 0, 0)
        self._forward = forward or Vec3(0, 0, -1)
        self._load_radius = load_radius
        self._forward_bias = forward_bias
        self._active = True

    @property
    def position(self) -> Vec3:
        """Get the camera position."""
        return self._position

    @position.setter
    def position(self, value: Vec3) -> None:
        """Set the camera position."""
        self._position = value

    @property
    def forward(self) -> Vec3:
        """Get the camera forward direction."""
        return self._forward

    @forward.setter
    def forward(self, value: Vec3) -> None:
        """Set the camera forward direction."""
        self._forward = value.normalized()

    @property
    def priority(self) -> int:
        """Get priority."""
        return StreamingPriority.HIGH.value

    @property
    def load_radius(self) -> float:
        """Get load radius."""
        return self._load_radius

    @property
    def forward_bias(self) -> float:
        """Get forward bias multiplier."""
        return self._forward_bias

    @property
    def is_active(self) -> bool:
        """Check if active."""
        return self._active


class CustomStreamingSource(StreamingSource):
    """
    Custom streaming source with configurable position callback.

    Useful for prefetch points, cinematic cameras, etc.
    """

    def __init__(
        self,
        position_getter: Callable[[], Vec3],
        load_radius: float = DEFAULT_CUSTOM_LOAD_RADIUS,
        priority: int = StreamingPriority.NORMAL.value,
        name: str = "",
    ) -> None:
        """
        Initialize a custom streaming source.

        Args:
            position_getter: Callback to get current position.
            load_radius: Load radius.
            priority: Streaming priority.
            name: Optional name for identification.
        """
        self._position_getter = position_getter
        self._load_radius = load_radius
        self._priority = priority
        self._active = True
        self.name = name

    @property
    def position(self) -> Vec3:
        """Get position from callback."""
        return self._position_getter()

    @property
    def priority(self) -> int:
        """Get priority."""
        return self._priority

    @property
    def load_radius(self) -> float:
        """Get load radius."""
        return self._load_radius

    @property
    def is_active(self) -> bool:
        """Check if active."""
        return self._active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        """Set active state."""
        self._active = value


@dataclass
class StreamingConfig:
    """Configuration for the streaming system."""
    load_distance: float = DEFAULT_STREAMING_LOAD_DISTANCE
    unload_distance: float = DEFAULT_STREAMING_UNLOAD_DISTANCE
    hysteresis: float = DEFAULT_STREAMING_HYSTERESIS
    priority_scale: float = DEFAULT_PRIORITY_SCALE
    max_concurrent_loads: int = DEFAULT_MAX_CONCURRENT_LOADS
    max_concurrent_unloads: int = DEFAULT_MAX_CONCURRENT_UNLOADS
    enable_preloading: bool = True
    preload_distance: float = DEFAULT_PRELOAD_DISTANCE

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.unload_distance <= self.load_distance:
            self.unload_distance = self.load_distance + self.hysteresis


@dataclass
class StreamingBudget:
    """Budget constraints for streaming operations."""
    memory_mb: float = DEFAULT_MEMORY_BUDGET_MB
    io_mbps: float = DEFAULT_IO_BANDWIDTH_MBPS
    frame_ms: float = DEFAULT_FRAME_TIME_BUDGET_MS

    # Current usage tracking
    current_memory_mb: float = 0.0
    current_io_mbps: float = 0.0
    current_frame_ms: float = 0.0

    @property
    def memory_available(self) -> float:
        """Get available memory budget."""
        return max(0, self.memory_mb - self.current_memory_mb)

    @property
    def io_available(self) -> float:
        """Get available IO bandwidth."""
        return max(0, self.io_mbps - self.current_io_mbps)

    @property
    def frame_available(self) -> float:
        """Get available frame time budget."""
        return max(0, self.frame_ms - self.current_frame_ms)

    def can_load(self, estimated_mb: float = 0, estimated_io: float = 0) -> bool:
        """Check if a load operation can fit within budget."""
        return (
            self.memory_available >= estimated_mb
            and self.io_available >= estimated_io
        )

    def reserve_memory(self, mb: float) -> bool:
        """Reserve memory from the budget."""
        if self.current_memory_mb + mb > self.memory_mb:
            return False
        self.current_memory_mb += mb
        return True

    def release_memory(self, mb: float) -> None:
        """Release memory back to the budget."""
        self.current_memory_mb = max(0, self.current_memory_mb - mb)

    def reset_frame_budget(self) -> None:
        """Reset per-frame budgets."""
        self.current_frame_ms = 0.0
        self.current_io_mbps = 0.0


@dataclass
class StreamingVolume:
    """
    A volume that affects streaming behavior.

    Can trigger loading, force unloading, or provide preload hints.
    """
    volume_type: StreamingVolumeType = StreamingVolumeType.TRIGGER
    bounds_min: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    bounds_max: Vec3 = field(default_factory=lambda: Vec3(
        DEFAULT_VOLUME_SIZE, DEFAULT_VOLUME_SIZE, DEFAULT_VOLUME_SIZE
    ))
    priority: int = StreamingPriority.NORMAL.value
    cells_to_load: List[CellCoord] = field(default_factory=list)
    enabled: bool = True
    name: str = ""

    @property
    def bounds(self) -> AABB:
        """Get the volume bounds."""
        return AABB(self.bounds_min, self.bounds_max)

    @property
    def center(self) -> Vec3:
        """Get the volume center."""
        return (self.bounds_min + self.bounds_max) * HALF_MULTIPLIER

    def contains_point(self, point: Vec3) -> bool:
        """Check if a point is inside the volume."""
        return (
            self.bounds_min.x <= point.x <= self.bounds_max.x
            and self.bounds_min.y <= point.y <= self.bounds_max.y
            and self.bounds_min.z <= point.z <= self.bounds_max.z
        )

    def overlaps(self, other: "StreamingVolume") -> bool:
        """Check if this volume overlaps another."""
        return (
            self.bounds_min.x <= other.bounds_max.x and self.bounds_max.x >= other.bounds_min.x
            and self.bounds_min.y <= other.bounds_max.y and self.bounds_max.y >= other.bounds_min.y
            and self.bounds_min.z <= other.bounds_max.z and self.bounds_max.z >= other.bounds_min.z
        )


@dataclass
class StreamingRequest:
    """A pending streaming operation request."""
    cell: StreamingCell
    priority: int = 0
    is_load: bool = True  # True for load, False for unload
    source: Optional[StreamingSource] = None
    timestamp: float = 0.0


class WorldStreaming:
    """
    Main streaming manager for the world.

    Coordinates streaming sources, budget management, and cell loading/unloading
    decisions based on distance, priority, and system constraints.
    """

    def __init__(
        self,
        grid: WorldGrid,
        config: StreamingConfig = None,
        budget: StreamingBudget = None,
    ) -> None:
        """
        Initialize the world streaming manager.

        Args:
            grid: The world grid to manage.
            config: Streaming configuration.
            budget: Resource budget constraints.
        """
        self.grid = grid
        self.config = config or StreamingConfig()
        self.budget = budget or StreamingBudget()

        self.sources: List[StreamingSource] = []
        self.volumes: List[StreamingVolume] = []

        # Pending operations
        self._load_queue: List[StreamingRequest] = []
        self._unload_queue: List[StreamingRequest] = []

        # Active operations
        self._loading_cells: Set[Tuple[int, int]] = set()
        self._unloading_cells: Set[Tuple[int, int]] = set()

        # Callbacks
        self._on_cell_loaded: List[Callable[[StreamingCell], None]] = []
        self._on_cell_unloaded: List[Callable[[StreamingCell], None]] = []

    def add_source(self, source: StreamingSource) -> None:
        """Add a streaming source."""
        self.sources.append(source)

    def remove_source(self, source: StreamingSource) -> bool:
        """Remove a streaming source."""
        if source in self.sources:
            self.sources.remove(source)
            return True
        return False

    def add_volume(self, volume: StreamingVolume) -> None:
        """Add a streaming volume."""
        self.volumes.append(volume)

    def remove_volume(self, volume: StreamingVolume) -> bool:
        """Remove a streaming volume."""
        if volume in self.volumes:
            self.volumes.remove(volume)
            return True
        return False

    def update(self, dt: float) -> Tuple[int, int]:
        """
        Update the streaming system.

        Args:
            dt: Delta time in seconds.

        Returns:
            Tuple of (cells_loaded, cells_unloaded) this frame.
        """
        # Reset per-frame budget
        self.budget.reset_frame_budget()

        # Gather cells to load/unload
        to_load, to_unload = self._gather_streaming_decisions()

        # Queue new requests
        self._queue_load_requests(to_load)
        self._queue_unload_requests(to_unload)

        # Process queues within budget
        cells_loaded = self._process_load_queue()
        cells_unloaded = self._process_unload_queue()

        return cells_loaded, cells_unloaded

    def _gather_streaming_decisions(self) -> Tuple[List[StreamingCell], List[StreamingCell]]:
        """
        Determine which cells should be loaded/unloaded.

        Returns:
            Tuple of (cells_to_load, cells_to_unload).
        """
        cells_to_load: Set[Tuple[int, int]] = set()
        cells_to_unload: Set[Tuple[int, int]] = set()

        # Get cells needed by all active sources
        for source in self.sources:
            if not source.is_active:
                continue

            cells_in_range = self.grid.get_cells_in_radius(
                source.position,
                min(source.load_radius, self.config.load_distance),
            )

            for cell in cells_in_range:
                if cell.state == CellState.UNLOADED:
                    cells_to_load.add((cell.coord.x, cell.coord.y))

        # Check streaming volumes
        for volume in self.volumes:
            if not volume.enabled:
                continue

            for source in self.sources:
                if source.is_active and volume.contains_point(source.position):
                    for coord in volume.cells_to_load:
                        cell = self.grid.get_cell(coord.x, coord.y)
                        if cell and cell.state == CellState.UNLOADED:
                            cells_to_load.add((coord.x, coord.y))

        # Determine cells to unload (outside all source ranges)
        for cell in self.grid.get_loaded_cells():
            key = (cell.coord.x, cell.coord.y)

            # Skip if queued for load
            if key in cells_to_load:
                continue

            should_unload = True
            for source in self.sources:
                if not source.is_active:
                    continue

                distance = cell.distance_to_point(source.position)
                if distance < self.config.unload_distance:
                    should_unload = False
                    break

            if should_unload:
                cells_to_unload.add(key)

        # Convert to cell lists
        to_load = [
            self.grid.get_or_create_cell(x, y)
            for x, y in cells_to_load
            if (x, y) not in self._loading_cells
        ]
        to_unload = [
            self.grid.get_cell(x, y)
            for x, y in cells_to_unload
            if (x, y) not in self._unloading_cells
        ]

        return to_load, [c for c in to_unload if c is not None]

    def _queue_load_requests(self, cells: List[StreamingCell]) -> None:
        """Queue cells for loading with priority."""
        for cell in cells:
            # Calculate priority based on distance to nearest source
            priority = 0
            min_distance = float('inf')

            for source in self.sources:
                if source.is_active:
                    dist = cell.distance_to_point(source.position)
                    if dist < min_distance:
                        min_distance = dist
                        priority = source.priority

            # Closer cells get higher priority
            priority = int(priority + (PRIORITY_DISTANCE_BASE - min_distance / PRIORITY_DISTANCE_DIVISOR))

            request = StreamingRequest(
                cell=cell,
                priority=priority,
                is_load=True,
            )
            self._load_queue.append(request)

        # Sort by priority (highest first)
        self._load_queue.sort(key=lambda r: r.priority, reverse=True)

    def _queue_unload_requests(self, cells: List[StreamingCell]) -> None:
        """Queue cells for unloading."""
        for cell in cells:
            # Lower priority for closer cells (unload furthest first)
            min_distance = float('inf')

            for source in self.sources:
                if source.is_active:
                    dist = cell.distance_to_point(source.position)
                    if dist < min_distance:
                        min_distance = dist

            priority = int(min_distance / UNLOAD_PRIORITY_DIVISOR)

            request = StreamingRequest(
                cell=cell,
                priority=priority,
                is_load=False,
            )
            self._unload_queue.append(request)

        # Sort by priority (highest = furthest = unload first)
        self._unload_queue.sort(key=lambda r: r.priority, reverse=True)

    def _process_load_queue(self) -> int:
        """
        Process pending load requests.

        Returns:
            Number of cells loaded this frame.
        """
        cells_loaded = 0
        requests_to_remove = []

        for request in self._load_queue:
            if len(self._loading_cells) >= self.config.max_concurrent_loads:
                break

            cell = request.cell

            # Estimate memory requirement
            estimated_mb = cell.get_memory_estimate() / (1024 * 1024)
            if not self.budget.can_load(estimated_mb):
                continue

            # Start loading
            if cell.load():
                self._loading_cells.add((cell.coord.x, cell.coord.y))
                self.budget.reserve_memory(estimated_mb)
                requests_to_remove.append(request)

                # Simulate immediate load completion (actual implementation would be async)
                cell.complete_load()
                self._loading_cells.discard((cell.coord.x, cell.coord.y))
                cells_loaded += 1

                for callback in self._on_cell_loaded:
                    callback(cell)

        # Remove processed requests
        for request in requests_to_remove:
            self._load_queue.remove(request)

        return cells_loaded

    def _process_unload_queue(self) -> int:
        """
        Process pending unload requests.

        Returns:
            Number of cells unloaded this frame.
        """
        cells_unloaded = 0
        requests_to_remove = []

        for request in self._unload_queue:
            if len(self._unloading_cells) >= self.config.max_concurrent_unloads:
                break

            cell = request.cell

            # Start unloading
            if cell.unload():
                self._unloading_cells.add((cell.coord.x, cell.coord.y))
                requests_to_remove.append(request)

                # Estimate memory to release
                estimated_mb = cell.get_memory_estimate() / (1024 * 1024)

                # Simulate immediate unload completion
                cell.complete_unload()
                self._unloading_cells.discard((cell.coord.x, cell.coord.y))
                self.budget.release_memory(estimated_mb)
                cells_unloaded += 1

                for callback in self._on_cell_unloaded:
                    callback(cell)

        # Remove processed requests
        for request in requests_to_remove:
            self._unload_queue.remove(request)

        return cells_unloaded

    def get_cells_to_load(self) -> List[StreamingCell]:
        """Get list of cells pending load."""
        return [r.cell for r in self._load_queue]

    def get_cells_to_unload(self) -> List[StreamingCell]:
        """Get list of cells pending unload."""
        return [r.cell for r in self._unload_queue]

    def get_loading_cells(self) -> List[StreamingCell]:
        """Get cells currently loading."""
        return [
            self.grid.get_cell(x, y)
            for x, y in self._loading_cells
            if self.grid.get_cell(x, y) is not None
        ]

    def get_unloading_cells(self) -> List[StreamingCell]:
        """Get cells currently unloading."""
        return [
            self.grid.get_cell(x, y)
            for x, y in self._unloading_cells
            if self.grid.get_cell(x, y) is not None
        ]

    def force_load_cell(self, coord: CellCoord) -> bool:
        """
        Force immediate load of a cell, bypassing queue.

        Args:
            coord: Cell coordinates to load.

        Returns:
            True if load succeeded.
        """
        cell = self.grid.get_or_create_cell(coord.x, coord.y)

        if cell.state != CellState.UNLOADED:
            return False

        if cell.load():
            cell.complete_load()
            for callback in self._on_cell_loaded:
                callback(cell)
            return True

        return False

    def force_unload_cell(self, coord: CellCoord) -> bool:
        """
        Force immediate unload of a cell, bypassing queue.

        Args:
            coord: Cell coordinates to unload.

        Returns:
            True if unload succeeded.
        """
        cell = self.grid.get_cell(coord.x, coord.y)

        if cell is None or cell.state == CellState.UNLOADED:
            return False

        if cell.unload():
            cell.complete_unload()
            for callback in self._on_cell_unloaded:
                callback(cell)
            return True

        return False

    def on_cell_loaded(self, callback: Callable[[StreamingCell], None]) -> None:
        """Register a callback for when a cell is loaded."""
        self._on_cell_loaded.append(callback)

    def on_cell_unloaded(self, callback: Callable[[StreamingCell], None]) -> None:
        """Register a callback for when a cell is unloaded."""
        self._on_cell_unloaded.append(callback)

    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        loaded = len(self.grid.get_loaded_cells())
        active = len(self.grid.get_active_cells())
        total = self.grid.get_cell_count()

        return {
            "total_cells": total,
            "loaded_cells": loaded,
            "active_cells": active,
            "pending_loads": len(self._load_queue),
            "pending_unloads": len(self._unload_queue),
            "loading_cells": len(self._loading_cells),
            "unloading_cells": len(self._unloading_cells),
            "source_count": len(self.sources),
            "volume_count": len(self.volumes),
            "memory_used_mb": self.budget.current_memory_mb,
            "memory_budget_mb": self.budget.memory_mb,
        }

    def clear_queues(self) -> None:
        """Clear all pending streaming requests."""
        self._load_queue.clear()
        self._unload_queue.clear()
