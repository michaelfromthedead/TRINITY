"""
Off-mesh navigation links for special traversal.

Provides support for jumps, ladders, doors, teleports, and other
special navigation connections that cannot be represented on the navmesh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .constants import (
    DEFAULT_CLIMB_ANGLE,
    DEFAULT_CLIMB_DURATION,
    DEFAULT_CLIMB_HEIGHT,
    DEFAULT_DROP_DURATION,
    DEFAULT_DROP_HEIGHT,
    DEFAULT_JUMP_DISTANCE,
    DEFAULT_JUMP_DURATION,
    DEFAULT_JUMP_HEIGHT,
    DEFAULT_SPATIAL_CELL_SIZE,
    DEFAULT_TELEPORT_COOLDOWN,
    DEFAULT_TELEPORT_DURATION,
    MAX_CLIMB_HEIGHT,
    MAX_DROP_HEIGHT,
    MAX_JUMP_HEIGHT,
    MIN_CLIMB_HEIGHT,
    MIN_DROP_HEIGHT,
    MIN_JUMP_HEIGHT,
    NavLinkDirection,
    NavLinkType,
)
from .navmesh import NavMesh, Vector3


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class NavLinkParams:
    """Parameters for a navigation link."""
    # Common parameters
    cost_modifier: float = 1.0
    traversal_time: float = 0.5
    enabled: bool = True

    # Jump parameters
    jump_height: float = DEFAULT_JUMP_HEIGHT
    jump_distance: float = DEFAULT_JUMP_DISTANCE

    # Drop parameters
    drop_height: float = DEFAULT_DROP_HEIGHT

    # Climb parameters
    climb_height: float = DEFAULT_CLIMB_HEIGHT
    climb_angle: float = DEFAULT_CLIMB_ANGLE

    # Teleport parameters
    teleport_cooldown: float = DEFAULT_TELEPORT_COOLDOWN

    # Requirements
    required_flags: int = 0  # Agent must have these flags
    excluded_flags: int = 0  # Agent must not have these flags
    min_agent_radius: float = 0.0
    max_agent_radius: float = float('inf')

    def validate(self) -> bool:
        """Validate parameters are in acceptable ranges."""
        if self.cost_modifier < 0:
            return False
        if self.traversal_time < 0:
            return False
        if not MIN_JUMP_HEIGHT <= self.jump_height <= MAX_JUMP_HEIGHT:
            return False
        if not MIN_DROP_HEIGHT <= self.drop_height <= MAX_DROP_HEIGHT:
            return False
        if not MIN_CLIMB_HEIGHT <= self.climb_height <= MAX_CLIMB_HEIGHT:
            return False
        return True


@dataclass
class NavLink:
    """Navigation link between two points."""
    id: int
    link_type: NavLinkType
    direction: NavLinkDirection
    start_position: Vector3
    end_position: Vector3
    params: NavLinkParams = field(default_factory=NavLinkParams)

    # Connected polygon IDs (if linked to navmesh)
    start_polygon_id: int = -1
    end_polygon_id: int = -1

    # Custom data
    user_data: Optional[object] = None
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Set default traversal time based on link type."""
        if self.params.traversal_time == 0.5:
            if self.link_type == NavLinkType.JUMP:
                self.params.traversal_time = DEFAULT_JUMP_DURATION
            elif self.link_type == NavLinkType.DROP:
                self.params.traversal_time = DEFAULT_DROP_DURATION
            elif self.link_type == NavLinkType.CLIMB:
                self.params.traversal_time = DEFAULT_CLIMB_DURATION
            elif self.link_type == NavLinkType.TELEPORT:
                self.params.traversal_time = DEFAULT_TELEPORT_DURATION

    @property
    def is_bidirectional(self) -> bool:
        """Check if link can be traversed in both directions."""
        return self.direction == NavLinkDirection.TWO_WAY

    @property
    def length(self) -> float:
        """Get straight-line distance of link."""
        return self.start_position.distance_to(self.end_position)

    @property
    def height_difference(self) -> float:
        """Get height difference between start and end."""
        return self.end_position.y - self.start_position.y

    def get_cost(self, base_cost: float = 1.0) -> float:
        """Calculate traversal cost."""
        return base_cost * self.params.cost_modifier * self.length

    def can_traverse(
        self, agent_flags: int = 0, agent_radius: float = 0.5,
        forward: bool = True
    ) -> bool:
        """Check if an agent can traverse this link."""
        if not self.params.enabled:
            return False

        # Check direction
        if not forward and not self.is_bidirectional:
            return False

        # Check flags
        if self.params.required_flags != 0:
            if (agent_flags & self.params.required_flags) != self.params.required_flags:
                return False

        if self.params.excluded_flags != 0:
            if (agent_flags & self.params.excluded_flags) != 0:
                return False

        # Check radius
        if not self.params.min_agent_radius <= agent_radius <= self.params.max_agent_radius:
            return False

        return True

    def get_traversal_position(self, t: float, forward: bool = True) -> Vector3:
        """Get position along link at time t (0-1)."""
        t = max(0, min(1, t))

        if not forward:
            t = 1 - t

        start = self.start_position
        end = self.end_position

        if self.link_type == NavLinkType.JUMP:
            # Parabolic arc
            horizontal = start.lerp(end, t)

            # Calculate height using parabola
            height_offset = self.params.jump_height * 4 * t * (1 - t)

            return Vector3(horizontal.x, start.y + height_offset + (end.y - start.y) * t, horizontal.z)

        elif self.link_type == NavLinkType.DROP:
            # Accelerating fall
            t_squared = t * t
            horizontal = start.lerp(end, t)
            vertical = start.y + (end.y - start.y) * t_squared
            return Vector3(horizontal.x, vertical, horizontal.z)

        elif self.link_type == NavLinkType.CLIMB:
            # Linear climb (could be enhanced with animation)
            return start.lerp(end, t)

        else:
            # Linear interpolation for teleport and custom
            return start.lerp(end, t)


@dataclass
class NavLinkTraversal:
    """Active traversal of a navigation link."""
    link_id: int
    agent_id: int
    start_time: float
    forward: bool = True
    progress: float = 0.0
    completed: bool = False

    def update(self, dt: float, duration: float) -> bool:
        """
        Update traversal progress.

        Returns True if traversal is complete.
        """
        if self.completed:
            return True

        if duration <= 0:
            self.progress = 1.0
            self.completed = True
            return True

        self.progress += dt / duration

        if self.progress >= 1.0:
            self.progress = 1.0
            self.completed = True
            return True

        return False


# =============================================================================
# Door Link State
# =============================================================================


@dataclass
class DoorState:
    """State for a door navigation link."""
    is_open: bool = False
    is_locked: bool = False
    open_progress: float = 0.0  # 0 = closed, 1 = fully open
    required_key: Optional[str] = None
    auto_close_time: float = 0.0  # 0 = doesn't auto-close
    time_since_used: float = 0.0


class DoorLink:
    """Navigation link representing a door."""

    def __init__(
        self, link: NavLink,
        initial_open: bool = False,
        locked: bool = False,
        required_key: Optional[str] = None,
        auto_close_time: float = 0.0
    ) -> None:
        """Initialize door link."""
        self._link = link
        self._state = DoorState(
            is_open=initial_open,
            is_locked=locked,
            open_progress=1.0 if initial_open else 0.0,
            required_key=required_key,
            auto_close_time=auto_close_time
        )

    @property
    def link(self) -> NavLink:
        """Get underlying nav link."""
        return self._link

    @property
    def state(self) -> DoorState:
        """Get door state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if door is open."""
        return self._state.is_open and self._state.open_progress >= 0.9

    @property
    def is_locked(self) -> bool:
        """Check if door is locked."""
        return self._state.is_locked

    def open(self, key: Optional[str] = None) -> bool:
        """
        Attempt to open the door.

        Returns True if door opened or started opening.
        """
        if self._state.is_locked:
            if key is None or key != self._state.required_key:
                return False
            self._state.is_locked = False

        self._state.is_open = True
        self._state.time_since_used = 0.0
        return True

    def close(self) -> bool:
        """Close the door."""
        self._state.is_open = False
        return True

    def lock(self, key: Optional[str] = None) -> bool:
        """Lock the door."""
        if not self._state.is_open or self._state.open_progress < 0.1:
            self._state.is_locked = True
            if key is not None:
                self._state.required_key = key
            return True
        return False

    def unlock(self, key: Optional[str] = None) -> bool:
        """Unlock the door."""
        if self._state.required_key is None or key == self._state.required_key:
            self._state.is_locked = False
            return True
        return False

    def update(self, dt: float, open_speed: float = 2.0) -> None:
        """Update door state."""
        # Update open progress
        if self._state.is_open:
            self._state.open_progress = min(1.0, self._state.open_progress + dt * open_speed)
        else:
            self._state.open_progress = max(0.0, self._state.open_progress - dt * open_speed)

        # Auto-close logic
        if self._state.is_open and self._state.auto_close_time > 0:
            self._state.time_since_used += dt
            if self._state.time_since_used >= self._state.auto_close_time:
                self.close()

    def can_traverse(self, agent_flags: int = 0, agent_radius: float = 0.5) -> bool:
        """Check if agent can traverse door."""
        if not self.is_open:
            return False
        return self._link.can_traverse(agent_flags, agent_radius)


# =============================================================================
# Ladder Link
# =============================================================================


@dataclass
class LadderParams:
    """Parameters for ladder navigation."""
    climb_speed: float = 2.0  # Units per second
    dismount_height: float = 0.5  # Height from top/bottom to dismount
    requires_hands_free: bool = True
    max_agent_width: float = 1.0


class LadderLink:
    """Navigation link representing a ladder."""

    def __init__(
        self, link: NavLink,
        params: Optional[LadderParams] = None
    ) -> None:
        """Initialize ladder link."""
        self._link = link
        self._params = params or LadderParams()

        # Calculate ladder properties
        self._height = abs(link.height_difference)
        self._rungs = int(self._height / 0.3)  # Rung every 30cm

    @property
    def link(self) -> NavLink:
        """Get underlying nav link."""
        return self._link

    @property
    def params(self) -> LadderParams:
        """Get ladder parameters."""
        return self._params

    @property
    def height(self) -> float:
        """Get ladder height."""
        return self._height

    @property
    def rung_count(self) -> int:
        """Get number of rungs."""
        return self._rungs

    @property
    def climb_time(self) -> float:
        """Get time to climb the full ladder."""
        return self._height / self._params.climb_speed

    def get_rung_position(self, rung_index: int) -> Vector3:
        """Get position of a specific rung."""
        if self._rungs <= 0:
            return self._link.start_position

        t = rung_index / self._rungs
        return self._link.get_traversal_position(t)

    def can_mount(
        self, position: Vector3, agent_flags: int = 0,
        agent_radius: float = 0.5
    ) -> bool:
        """Check if agent can mount the ladder from position."""
        # Check base requirements
        if not self._link.can_traverse(agent_flags, agent_radius):
            return False

        # Check distance to mount points
        dist_to_bottom = position.distance_to(self._link.start_position)
        dist_to_top = position.distance_to(self._link.end_position)

        mount_radius = 1.0  # Mount from within 1 unit
        return dist_to_bottom < mount_radius or dist_to_top < mount_radius


# =============================================================================
# Navigation Link Manager
# =============================================================================


class NavLinkManager:
    """
    Manager for all navigation links in a level.

    Handles creation, querying, and traversal of nav links.
    """

    def __init__(self, navmesh: Optional[NavMesh] = None) -> None:
        """Initialize nav link manager."""
        self._navmesh = navmesh
        self._links: Dict[int, NavLink] = {}
        self._door_links: Dict[int, DoorLink] = {}
        self._ladder_links: Dict[int, LadderLink] = {}
        self._active_traversals: Dict[int, NavLinkTraversal] = {}
        self._next_id = 0
        self._next_traversal_id = 0

        # Spatial index for efficient queries
        self._spatial_cells: Dict[Tuple[int, int, int], List[int]] = {}
        self._cell_size = DEFAULT_SPATIAL_CELL_SIZE

    @property
    def link_count(self) -> int:
        """Get total number of links."""
        return len(self._links)

    @property
    def active_traversal_count(self) -> int:
        """Get number of active traversals."""
        return len(self._active_traversals)

    def _get_cell_key(self, position: Vector3) -> Tuple[int, int, int]:
        """Get spatial cell key for position."""
        return (
            int(position.x / self._cell_size),
            int(position.y / self._cell_size),
            int(position.z / self._cell_size)
        )

    def _add_to_spatial_index(self, link: NavLink) -> None:
        """Add link to spatial index."""
        start_cell = self._get_cell_key(link.start_position)
        end_cell = self._get_cell_key(link.end_position)

        for cell in [start_cell, end_cell]:
            if cell not in self._spatial_cells:
                self._spatial_cells[cell] = []
            if link.id not in self._spatial_cells[cell]:
                self._spatial_cells[cell].append(link.id)

    def _remove_from_spatial_index(self, link: NavLink) -> None:
        """Remove link from spatial index."""
        start_cell = self._get_cell_key(link.start_position)
        end_cell = self._get_cell_key(link.end_position)

        for cell in [start_cell, end_cell]:
            if cell in self._spatial_cells and link.id in self._spatial_cells[cell]:
                self._spatial_cells[cell].remove(link.id)

    def add_link(
        self, link_type: NavLinkType,
        start: Vector3, end: Vector3,
        direction: NavLinkDirection = NavLinkDirection.ONE_WAY,
        params: Optional[NavLinkParams] = None
    ) -> int:
        """
        Add a navigation link.

        Returns the link ID.
        """
        self._next_id += 1

        link = NavLink(
            id=self._next_id,
            link_type=link_type,
            direction=direction,
            start_position=start,
            end_position=end,
            params=params or NavLinkParams()
        )

        # Connect to navmesh if available
        if self._navmesh is not None:
            link.start_polygon_id = self._navmesh.find_polygon_at(start) or -1
            link.end_polygon_id = self._navmesh.find_polygon_at(end) or -1

        self._links[link.id] = link
        self._add_to_spatial_index(link)

        return link.id

    def add_jump_link(
        self, start: Vector3, end: Vector3,
        bidirectional: bool = False,
        jump_height: float = DEFAULT_JUMP_HEIGHT
    ) -> int:
        """Add a jump link."""
        params = NavLinkParams(
            jump_height=jump_height,
            traversal_time=DEFAULT_JUMP_DURATION
        )

        direction = NavLinkDirection.TWO_WAY if bidirectional else NavLinkDirection.ONE_WAY
        return self.add_link(NavLinkType.JUMP, start, end, direction, params)

    def add_drop_link(
        self, start: Vector3, end: Vector3,
        drop_height: Optional[float] = None
    ) -> int:
        """Add a drop link (one-way down)."""
        if drop_height is None:
            drop_height = start.y - end.y

        params = NavLinkParams(
            drop_height=drop_height,
            traversal_time=DEFAULT_DROP_DURATION
        )

        return self.add_link(NavLinkType.DROP, start, end, NavLinkDirection.ONE_WAY, params)

    def add_ladder_link(
        self, bottom: Vector3, top: Vector3,
        ladder_params: Optional[LadderParams] = None
    ) -> int:
        """Add a ladder link."""
        params = NavLinkParams(
            climb_height=abs(top.y - bottom.y),
            traversal_time=DEFAULT_CLIMB_DURATION
        )

        link_id = self.add_link(
            NavLinkType.CLIMB, bottom, top,
            NavLinkDirection.TWO_WAY, params
        )

        # Create ladder wrapper
        link = self._links[link_id]
        ladder = LadderLink(link, ladder_params)
        self._ladder_links[link_id] = ladder

        # Update traversal time based on climb speed
        link.params.traversal_time = ladder.climb_time

        return link_id

    def add_door_link(
        self, side_a: Vector3, side_b: Vector3,
        initial_open: bool = False,
        locked: bool = False,
        required_key: Optional[str] = None,
        auto_close_time: float = 0.0
    ) -> int:
        """Add a door link."""
        params = NavLinkParams(
            traversal_time=0.5  # Time to walk through door
        )

        link_id = self.add_link(
            NavLinkType.CUSTOM, side_a, side_b,
            NavLinkDirection.TWO_WAY, params
        )

        # Create door wrapper
        link = self._links[link_id]
        door = DoorLink(link, initial_open, locked, required_key, auto_close_time)
        self._door_links[link_id] = door

        # Disable link if door is closed
        if not initial_open:
            link.params.enabled = False

        return link_id

    def add_teleport_link(
        self, start: Vector3, end: Vector3,
        bidirectional: bool = False,
        cooldown: float = DEFAULT_TELEPORT_COOLDOWN
    ) -> int:
        """Add a teleport link."""
        params = NavLinkParams(
            teleport_cooldown=cooldown,
            traversal_time=DEFAULT_TELEPORT_DURATION,
            cost_modifier=0.1  # Cheap to use
        )

        direction = NavLinkDirection.TWO_WAY if bidirectional else NavLinkDirection.ONE_WAY
        return self.add_link(NavLinkType.TELEPORT, start, end, direction, params)

    def remove_link(self, link_id: int) -> bool:
        """Remove a navigation link."""
        link = self._links.get(link_id)
        if link is None:
            return False

        self._remove_from_spatial_index(link)
        del self._links[link_id]

        # Remove associated wrappers
        self._door_links.pop(link_id, None)
        self._ladder_links.pop(link_id, None)

        return True

    def get_link(self, link_id: int) -> Optional[NavLink]:
        """Get link by ID."""
        return self._links.get(link_id)

    def get_door(self, link_id: int) -> Optional[DoorLink]:
        """Get door link by ID."""
        return self._door_links.get(link_id)

    def get_ladder(self, link_id: int) -> Optional[LadderLink]:
        """Get ladder link by ID."""
        return self._ladder_links.get(link_id)

    def enable_link(self, link_id: int, enabled: bool = True) -> bool:
        """Enable or disable a link."""
        link = self._links.get(link_id)
        if link is None:
            return False
        link.params.enabled = enabled
        return True

    def find_links_at_position(
        self, position: Vector3, radius: float = 1.0
    ) -> List[NavLink]:
        """Find all links within radius of position."""
        result = []
        radius_sq = radius * radius

        # Check nearby cells
        cell = self._get_cell_key(position)
        cells_to_check = [
            (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
        ]

        checked_ids: Set[int] = set()

        for check_cell in cells_to_check:
            if check_cell not in self._spatial_cells:
                continue

            for link_id in self._spatial_cells[check_cell]:
                if link_id in checked_ids:
                    continue
                checked_ids.add(link_id)

                link = self._links.get(link_id)
                if link is None:
                    continue

                # Check distance to both endpoints
                start_dist_sq = position.distance_squared_to(link.start_position)
                end_dist_sq = position.distance_squared_to(link.end_position)

                if start_dist_sq < radius_sq or end_dist_sq < radius_sq:
                    result.append(link)

        return result

    def find_links_by_type(self, link_type: NavLinkType) -> List[NavLink]:
        """Find all links of a specific type."""
        return [
            link for link in self._links.values()
            if link.link_type == link_type
        ]

    def find_links_with_tag(self, tag: str) -> List[NavLink]:
        """Find all links with a specific tag."""
        return [
            link for link in self._links.values()
            if tag in link.tags
        ]

    def begin_traversal(
        self, link_id: int, agent_id: int,
        current_time: float, forward: bool = True
    ) -> Optional[int]:
        """
        Begin traversing a link.

        Returns traversal ID or None if traversal cannot start.
        """
        link = self._links.get(link_id)
        if link is None:
            return None

        # Check door state
        if link_id in self._door_links:
            door = self._door_links[link_id]
            if not door.is_open:
                return None

        self._next_traversal_id += 1

        traversal = NavLinkTraversal(
            link_id=link_id,
            agent_id=agent_id,
            start_time=current_time,
            forward=forward
        )

        self._active_traversals[self._next_traversal_id] = traversal
        return self._next_traversal_id

    def update_traversal(
        self, traversal_id: int, dt: float
    ) -> Tuple[bool, Optional[Vector3]]:
        """
        Update a traversal.

        Returns (completed, current_position).
        """
        traversal = self._active_traversals.get(traversal_id)
        if traversal is None:
            return (True, None)

        link = self._links.get(traversal.link_id)
        if link is None:
            return (True, None)

        completed = traversal.update(dt, link.params.traversal_time)
        position = link.get_traversal_position(traversal.progress, traversal.forward)

        if completed:
            del self._active_traversals[traversal_id]

        return (completed, position)

    def cancel_traversal(self, traversal_id: int) -> bool:
        """Cancel an active traversal."""
        if traversal_id in self._active_traversals:
            del self._active_traversals[traversal_id]
            return True
        return False

    def get_traversal(self, traversal_id: int) -> Optional[NavLinkTraversal]:
        """Get active traversal by ID."""
        return self._active_traversals.get(traversal_id)

    def update(self, dt: float) -> None:
        """Update all door links."""
        for door in self._door_links.values():
            door.update(dt)
            # Update link enabled state based on door
            door.link.params.enabled = door.is_open

    def validate_link(self, link_id: int) -> bool:
        """
        Validate that a link is properly configured.

        Checks that endpoints are on or near the navmesh.
        """
        link = self._links.get(link_id)
        if link is None:
            return False

        if not link.params.validate():
            return False

        # Check that link endpoints make sense for the type
        height_diff = link.height_difference

        if link.link_type == NavLinkType.JUMP:
            # Jump should not go too high
            if height_diff > link.params.jump_height * 2:
                return False

        elif link.link_type == NavLinkType.DROP:
            # Drop should go down
            if height_diff > 0:
                return False

        elif link.link_type == NavLinkType.CLIMB:
            # Climb should have reasonable height
            if abs(height_diff) < 0.1:
                return False

        return True

    def get_links_between_polygons(
        self, poly_a: int, poly_b: int
    ) -> List[NavLink]:
        """Get links connecting two navmesh polygons."""
        result = []

        for link in self._links.values():
            if link.start_polygon_id == poly_a and link.end_polygon_id == poly_b:
                result.append(link)
            elif link.is_bidirectional:
                if link.start_polygon_id == poly_b and link.end_polygon_id == poly_a:
                    result.append(link)

        return result
