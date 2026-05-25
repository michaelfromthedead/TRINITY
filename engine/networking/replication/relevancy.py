"""Relevancy and interest management for network replication.

Determines which entities should be replicated to which clients based on
spatial proximity, ownership, and custom rules.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol, TypeVar

from ..config import get_config

_logger = logging.getLogger(__name__)

# Get config instance
_config = get_config()


class RelevancyType(Enum):
    """Types of relevancy checks for network replication."""
    ALWAYS = auto()    # Always relevant to all viewers
    OWNER = auto()     # Only relevant to owner
    DISTANCE = auto()  # Distance-based relevancy
    AREA = auto()      # Area/region-based relevancy
    CUSTOM = auto()    # Custom predicate function


# Default relevancy radius in world units - from config
DEFAULT_RELEVANCY_RADIUS = _config.DEFAULT_RELEVANCY_RADIUS

# Grid cell size for spatial hashing (default) - from config
DEFAULT_GRID_CELL_SIZE = _config.DEFAULT_GRID_CELL_SIZE


class HasPosition(Protocol):
    """Protocol for objects with a position."""
    @property
    def position(self) -> tuple[float, float, float]: ...


class HasOwner(Protocol):
    """Protocol for objects with an owner."""
    @property
    def owner_id(self) -> Optional[int]: ...


T = TypeVar('T')


@dataclass(slots=True)
class RelevancyResult:
    """Result of a relevancy check.

    Attributes:
        is_relevant: Whether entity is relevant to viewer
        priority: Priority modifier for bandwidth allocation
        reason: Human-readable reason for the result
    """
    is_relevant: bool
    priority: float = 1.0
    reason: str = ""


class InterestArea(ABC):
    """Base class for interest area implementations.

    Interest areas define the spatial or logical boundaries within which
    entities are considered relevant to a viewer.
    """

    @abstractmethod
    def check_relevant(self, entity: Any, viewer: Any) -> RelevancyResult:
        """Check if an entity is relevant to a viewer.

        Args:
            entity: The entity to check
            viewer: The viewer (player/connection) checking relevancy

        Returns:
            RelevancyResult with relevancy status and priority
        """
        pass

    @abstractmethod
    def update(self, viewer: Any) -> None:
        """Update the interest area for a viewer.

        Called when viewer's position or state changes.

        Args:
            viewer: The viewer to update
        """
        pass


class AlwaysRelevant(InterestArea):
    """Interest area where all entities are always relevant.

    Use for globally-relevant entities like game state, weather, etc.
    """

    def check_relevant(self, entity: Any, viewer: Any) -> RelevancyResult:
        return RelevancyResult(is_relevant=True, priority=1.0, reason="always_relevant")

    def update(self, viewer: Any) -> None:
        pass


class OwnerRelevant(InterestArea):
    """Interest area where entities are only relevant to their owner.

    Use for private player data like inventory, stats, etc.
    """

    def check_relevant(self, entity: Any, viewer: Any) -> RelevancyResult:
        # Get owner IDs
        entity_owner = getattr(entity, 'owner_id', None)
        viewer_id = getattr(viewer, 'player_id', None) or id(viewer)

        if entity_owner is not None and entity_owner == viewer_id:
            return RelevancyResult(is_relevant=True, priority=1.0, reason="is_owner")

        return RelevancyResult(is_relevant=False, priority=0.0, reason="not_owner")

    def update(self, viewer: Any) -> None:
        pass


@dataclass
class RadiusRelevancy(InterestArea):
    """Distance-based relevancy using spherical radius.

    Entities within the specified radius are relevant, with priority
    decreasing based on distance.

    Attributes:
        radius: Maximum relevancy distance in world units
        falloff_start: Distance at which priority begins decreasing
        always_relevant_to_owner: Whether owner always sees their entities
    """
    radius: float = DEFAULT_RELEVANCY_RADIUS
    falloff_start: float = field(default_factory=lambda: _config.DEFAULT_RELEVANCY_RADIUS * _config.DEFAULT_FALLOFF_START_MULTIPLIER)
    always_relevant_to_owner: bool = True

    def check_relevant(self, entity: Any, viewer: Any) -> RelevancyResult:
        # Check owner relevancy first
        if self.always_relevant_to_owner:
            entity_owner = getattr(entity, 'owner_id', None)
            viewer_id = getattr(viewer, 'player_id', None) or id(viewer)
            if entity_owner is not None and entity_owner == viewer_id:
                return RelevancyResult(
                    is_relevant=True,
                    priority=1.0,
                    reason="owner"
                )

        # Get positions
        entity_pos = self._get_position(entity)
        viewer_pos = self._get_position(viewer)

        if entity_pos is None or viewer_pos is None:
            # If no position, assume relevant (fail-open)
            return RelevancyResult(
                is_relevant=True,
                priority=_config.NO_POSITION_DEFAULT_PRIORITY,
                reason="no_position"
            )

        # Calculate distance
        distance = self._distance(entity_pos, viewer_pos)

        if distance > self.radius:
            return RelevancyResult(
                is_relevant=False,
                priority=0.0,
                reason=f"too_far:{distance:.0f}"
            )

        # Calculate priority based on distance
        if distance <= self.falloff_start:
            priority = 1.0
        else:
            # Linear falloff from falloff_start to radius
            falloff_range = self.radius - self.falloff_start
            if falloff_range > 0:
                priority = 1.0 - (distance - self.falloff_start) / falloff_range
            else:
                priority = 1.0

        return RelevancyResult(
            is_relevant=True,
            priority=max(_config.MIN_RELEVANCY_PRIORITY, priority),
            reason=f"distance:{distance:.0f}"
        )

    def update(self, viewer: Any) -> None:
        # No state to update for simple radius check
        pass

    def _get_position(self, obj: Any) -> Optional[tuple[float, float, float]]:
        """Extract position from object."""
        if hasattr(obj, 'position'):
            pos = obj.position
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                return (float(pos[0]), float(pos[1]), float(pos[2]))
        if hasattr(obj, 'x') and hasattr(obj, 'y') and hasattr(obj, 'z'):
            return (float(obj.x), float(obj.y), float(obj.z))
        return None

    def _distance(
        self,
        a: tuple[float, float, float],
        b: tuple[float, float, float]
    ) -> float:
        """Calculate 3D Euclidean distance."""
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)


@dataclass
class GridRelevancy(InterestArea):
    """Spatial hash grid-based relevancy.

    Uses a grid to efficiently find relevant entities in large worlds.
    More efficient than radius checks for many entities.

    Attributes:
        cell_size: Size of each grid cell in world units
        view_distance: Number of cells visible in each direction
        always_relevant_to_owner: Whether owner always sees their entities
    """
    cell_size: float = DEFAULT_GRID_CELL_SIZE
    view_distance: int = _config.DEFAULT_GRID_VIEW_DISTANCE
    always_relevant_to_owner: bool = True

    # Grid data - stores entity IDs, not objects directly (for hashability)
    _grid: dict[tuple[int, int, int], dict[int, Any]] = field(default_factory=dict)
    _entity_cells: dict[int, tuple[int, int, int]] = field(default_factory=dict)

    def check_relevant(self, entity: Any, viewer: Any) -> RelevancyResult:
        # Check owner relevancy first
        if self.always_relevant_to_owner:
            entity_owner = getattr(entity, 'owner_id', None)
            viewer_id = getattr(viewer, 'player_id', None) or id(viewer)
            if entity_owner is not None and entity_owner == viewer_id:
                return RelevancyResult(
                    is_relevant=True,
                    priority=1.0,
                    reason="owner"
                )

        # Get cell coordinates
        entity_pos = self._get_position(entity)
        viewer_pos = self._get_position(viewer)

        if entity_pos is None or viewer_pos is None:
            return RelevancyResult(
                is_relevant=True,
                priority=_config.NO_POSITION_DEFAULT_PRIORITY,
                reason="no_position"
            )

        entity_cell = self._get_cell(entity_pos)
        viewer_cell = self._get_cell(viewer_pos)

        # Check if within view distance
        dx = abs(entity_cell[0] - viewer_cell[0])
        dy = abs(entity_cell[1] - viewer_cell[1])
        dz = abs(entity_cell[2] - viewer_cell[2])

        max_dist = max(dx, dy, dz)

        if max_dist > self.view_distance:
            return RelevancyResult(
                is_relevant=False,
                priority=0.0,
                reason=f"grid_far:{max_dist}"
            )

        # Calculate priority based on cell distance
        priority = 1.0 - (max_dist / (self.view_distance + 1))

        return RelevancyResult(
            is_relevant=True,
            priority=max(_config.MIN_RELEVANCY_PRIORITY, priority),
            reason=f"grid:{dx},{dy},{dz}"
        )

    def update(self, viewer: Any) -> None:
        # Could precompute visible cells here for optimization
        pass

    def register_entity(self, entity: Any) -> None:
        """Register an entity in the spatial grid.

        Args:
            entity: Entity to register
        """
        pos = self._get_position(entity)
        if pos is None:
            return

        cell = self._get_cell(pos)
        entity_id = id(entity)

        # Remove from old cell if exists
        old_cell = self._entity_cells.get(entity_id)
        if old_cell is not None and old_cell != cell:
            self._grid.get(old_cell, {}).pop(entity_id, None)

        # Add to new cell
        if cell not in self._grid:
            self._grid[cell] = {}
        self._grid[cell][entity_id] = entity
        self._entity_cells[entity_id] = cell

    def unregister_entity(self, entity: Any) -> None:
        """Remove an entity from the spatial grid.

        Args:
            entity: Entity to remove
        """
        entity_id = id(entity)
        cell = self._entity_cells.pop(entity_id, None)
        if cell is not None:
            self._grid.get(cell, {}).pop(entity_id, None)

    def update_entity(self, entity: Any) -> None:
        """Update an entity's grid position.

        Args:
            entity: Entity to update
        """
        self.register_entity(entity)

    def get_entities_near(self, position: tuple[float, float, float]) -> set[Any]:
        """Get all entities near a position.

        Args:
            position: World position

        Returns:
            Set of entities in nearby cells
        """
        center_cell = self._get_cell(position)
        result: set[Any] = set()

        for dx in range(-self.view_distance, self.view_distance + 1):
            for dy in range(-self.view_distance, self.view_distance + 1):
                for dz in range(-self.view_distance, self.view_distance + 1):
                    cell = (
                        center_cell[0] + dx,
                        center_cell[1] + dy,
                        center_cell[2] + dz
                    )
                    entities_dict = self._grid.get(cell)
                    if entities_dict:
                        result.update(entities_dict.values())

        return result

    def _get_cell(self, pos: tuple[float, float, float]) -> tuple[int, int, int]:
        """Convert world position to grid cell."""
        return (
            int(pos[0] // self.cell_size),
            int(pos[1] // self.cell_size),
            int(pos[2] // self.cell_size)
        )

    def _get_position(self, obj: Any) -> Optional[tuple[float, float, float]]:
        """Extract position from object."""
        if hasattr(obj, 'position'):
            pos = obj.position
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                return (float(pos[0]), float(pos[1]), float(pos[2]))
        if hasattr(obj, 'x') and hasattr(obj, 'y') and hasattr(obj, 'z'):
            return (float(obj.x), float(obj.y), float(obj.z))
        return None


@dataclass
class CustomRelevancy(InterestArea):
    """Custom relevancy using a user-defined predicate function.

    Attributes:
        predicate: Function (entity, viewer) -> RelevancyResult
    """
    predicate: Callable[[Any, Any], RelevancyResult] = field(
        default=lambda e, v: RelevancyResult(is_relevant=True)
    )

    def check_relevant(self, entity: Any, viewer: Any) -> RelevancyResult:
        return self.predicate(entity, viewer)

    def update(self, viewer: Any) -> None:
        pass


@dataclass
class CompositeRelevancy(InterestArea):
    """Combines multiple relevancy checks with AND/OR logic.

    Attributes:
        areas: List of interest areas to check
        require_all: If True, all areas must be relevant (AND)
                    If False, any area being relevant is sufficient (OR)
    """
    areas: list[InterestArea] = field(default_factory=list)
    require_all: bool = False

    def check_relevant(self, entity: Any, viewer: Any) -> RelevancyResult:
        if not self.areas:
            return RelevancyResult(is_relevant=True, reason="no_areas")

        results = [area.check_relevant(entity, viewer) for area in self.areas]

        if self.require_all:
            # AND logic - all must be relevant
            if all(r.is_relevant for r in results):
                avg_priority = sum(r.priority for r in results) / len(results)
                return RelevancyResult(
                    is_relevant=True,
                    priority=avg_priority,
                    reason="all_relevant"
                )
            return RelevancyResult(is_relevant=False, reason="not_all_relevant")
        else:
            # OR logic - any relevant is sufficient
            relevant_results = [r for r in results if r.is_relevant]
            if relevant_results:
                max_priority = max(r.priority for r in relevant_results)
                return RelevancyResult(
                    is_relevant=True,
                    priority=max_priority,
                    reason="some_relevant"
                )
            return RelevancyResult(is_relevant=False, reason="none_relevant")

    def update(self, viewer: Any) -> None:
        for area in self.areas:
            area.update(viewer)

    def add_area(self, area: InterestArea) -> None:
        """Add an interest area to the composite."""
        self.areas.append(area)

    def remove_area(self, area: InterestArea) -> None:
        """Remove an interest area from the composite."""
        if area in self.areas:
            self.areas.remove(area)


class RelevancyManager:
    """Manages relevancy checking for all entities and viewers.

    Central coordinator for interest management in the networking layer.
    """
    __slots__ = ('_default_area', '_entity_areas', '_viewer_areas')

    def __init__(self, default_area: Optional[InterestArea] = None):
        """Initialize the relevancy manager.

        Args:
            default_area: Default interest area for entities without specific config
        """
        self._default_area = default_area or RadiusRelevancy()
        self._entity_areas: dict[int, InterestArea] = {}  # entity id -> area
        self._viewer_areas: dict[int, InterestArea] = {}  # viewer id -> area

    def set_entity_area(self, entity: Any, area: InterestArea) -> None:
        """Set a custom interest area for an entity.

        Args:
            entity: The entity
            area: Interest area to use
        """
        self._entity_areas[id(entity)] = area

    def set_viewer_area(self, viewer: Any, area: InterestArea) -> None:
        """Set a custom interest area for a viewer.

        Args:
            viewer: The viewer
            area: Interest area to use
        """
        self._viewer_areas[id(viewer)] = area

    def check_relevant(self, entity: Any, viewer: Any) -> RelevancyResult:
        """Check if an entity is relevant to a viewer.

        Uses entity-specific area if set, otherwise viewer-specific,
        otherwise default.

        Args:
            entity: The entity to check
            viewer: The viewer

        Returns:
            RelevancyResult
        """
        # Try entity-specific area first
        area = self._entity_areas.get(id(entity))
        if area is None:
            # Try viewer-specific area
            area = self._viewer_areas.get(id(viewer))
        if area is None:
            # Use default
            area = self._default_area

        return area.check_relevant(entity, viewer)

    def get_relevant_entities(
        self,
        entities: list[Any],
        viewer: Any
    ) -> list[tuple[Any, float]]:
        """Filter entities by relevancy to a viewer.

        Args:
            entities: List of entities to filter
            viewer: The viewer

        Returns:
            List of (entity, priority) tuples for relevant entities
        """
        results = []
        for entity in entities:
            result = self.check_relevant(entity, viewer)
            if result.is_relevant:
                results.append((entity, result.priority))

        # Sort by priority (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def update_viewer(self, viewer: Any) -> None:
        """Update interest areas for a viewer.

        Args:
            viewer: The viewer to update
        """
        # Update viewer-specific area
        area = self._viewer_areas.get(id(viewer))
        if area:
            area.update(viewer)

        # Update default area
        self._default_area.update(viewer)

    def remove_entity(self, entity: Any) -> None:
        """Remove an entity from tracking.

        Args:
            entity: The entity to remove
        """
        self._entity_areas.pop(id(entity), None)

    def remove_viewer(self, viewer: Any) -> None:
        """Remove a viewer from tracking.

        Args:
            viewer: The viewer to remove
        """
        self._viewer_areas.pop(id(viewer), None)
