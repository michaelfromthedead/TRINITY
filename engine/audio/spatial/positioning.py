"""Spatial Audio Source Positioning.

Implements different source positioning types:
- Point source (single location)
- Area source (extended 2D region)
- Line source (along a path)
- Volume source (3D region)

Also handles multi-listener support for split-screen.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from engine.audio.spatial.config import (
    MAX_ATTENUATION_DISTANCE,
    MAX_LISTENERS,
    MIN_ATTENUATION_DISTANCE,
    SourceType,
)
from engine.core.math.vec import Vec3

if TYPE_CHECKING:
    from engine.audio.spatial.attenuation import AttenuationCurve


@dataclass
class ListenerState:
    """State of an audio listener."""

    position: Vec3 = field(default_factory=Vec3.zero)
    forward: Vec3 = field(default_factory=Vec3.forward)
    up: Vec3 = field(default_factory=Vec3.up)
    velocity: Vec3 = field(default_factory=Vec3.zero)
    active: bool = True
    gain: float = 1.0
    listener_id: int = 0

    def get_right(self) -> Vec3:
        """Calculate right vector from forward and up."""
        return self.forward.cross(self.up).normalized()

    def world_to_listener(self, world_pos: Vec3) -> Vec3:
        """Transform world position to listener-local coordinates."""
        # Translate to listener origin
        relative = world_pos - self.position

        # Get basis vectors
        right = self.get_right()
        up = self.up
        forward = self.forward

        # Project onto listener basis (listener looks down -Z)
        return Vec3(
            relative.dot(right),
            relative.dot(up),
            -relative.dot(forward)  # Negative because forward is -Z
        )


@dataclass
class ListenerManager:
    """Manages multiple listeners for split-screen support."""

    listeners: List[ListenerState] = field(default_factory=list)
    primary_listener_id: int = 0

    def __post_init__(self) -> None:
        if not self.listeners:
            # Create default listener
            self.listeners.append(ListenerState(listener_id=0))

    def add_listener(self) -> Optional[ListenerState]:
        """Add a new listener. Returns None if max listeners reached."""
        if len(self.listeners) >= MAX_LISTENERS:
            return None

        new_id = len(self.listeners)
        listener = ListenerState(listener_id=new_id)
        self.listeners.append(listener)
        return listener

    def remove_listener(self, listener_id: int) -> bool:
        """Remove a listener by ID."""
        if listener_id == 0:
            return False  # Cannot remove primary listener

        for i, listener in enumerate(self.listeners):
            if listener.listener_id == listener_id:
                self.listeners.pop(i)
                return True
        return False

    def get_listener(self, listener_id: int) -> Optional[ListenerState]:
        """Get listener by ID."""
        for listener in self.listeners:
            if listener.listener_id == listener_id:
                return listener
        return None

    def get_primary_listener(self) -> ListenerState:
        """Get the primary listener."""
        return self.listeners[0]

    def get_active_listeners(self) -> List[ListenerState]:
        """Get all active listeners."""
        return [l for l in self.listeners if l.active]

    def update_listener(
        self,
        listener_id: int,
        position: Optional[Vec3] = None,
        forward: Optional[Vec3] = None,
        up: Optional[Vec3] = None,
        velocity: Optional[Vec3] = None
    ) -> bool:
        """Update listener state."""
        listener = self.get_listener(listener_id)
        if listener is None:
            return False

        if position is not None:
            listener.position = position
        if forward is not None:
            listener.forward = forward.normalized()
        if up is not None:
            listener.up = up.normalized()
        if velocity is not None:
            listener.velocity = velocity

        return True


class SpatialSource(ABC):
    """Abstract base class for spatial audio sources."""

    def __init__(
        self,
        source_type: SourceType,
        min_distance: float = MIN_ATTENUATION_DISTANCE,
        max_distance: float = MAX_ATTENUATION_DISTANCE
    ) -> None:
        self._source_type = source_type
        self._min_distance = max(0.001, min_distance)
        self._max_distance = max(self._min_distance, max_distance)
        self._velocity = Vec3.zero()

    @property
    def source_type(self) -> SourceType:
        """Get the source type."""
        return self._source_type

    @property
    def min_distance(self) -> float:
        """Get minimum attenuation distance."""
        return self._min_distance

    @min_distance.setter
    def min_distance(self, value: float) -> None:
        self._min_distance = max(0.001, value)

    @property
    def max_distance(self) -> float:
        """Get maximum attenuation distance."""
        return self._max_distance

    @max_distance.setter
    def max_distance(self, value: float) -> None:
        self._max_distance = max(self._min_distance, value)

    @property
    def velocity(self) -> Vec3:
        """Get source velocity for Doppler effect."""
        return self._velocity

    @velocity.setter
    def velocity(self, value: Vec3) -> None:
        self._velocity = value

    @abstractmethod
    def get_closest_point(self, listener_pos: Vec3) -> Vec3:
        """Get the closest point on the source to the listener."""
        pass

    @abstractmethod
    def get_distance(self, listener_pos: Vec3) -> float:
        """Get the distance from listener to the source."""
        pass

    @abstractmethod
    def get_direction(self, listener_pos: Vec3) -> Vec3:
        """Get normalized direction from listener to source."""
        pass

    def is_in_range(self, listener_pos: Vec3) -> bool:
        """Check if listener is within audible range."""
        return self.get_distance(listener_pos) <= self._max_distance

    def get_normalized_distance(self, listener_pos: Vec3) -> float:
        """Get distance normalized to 0-1 range between min and max."""
        distance = self.get_distance(listener_pos)
        if distance <= self._min_distance:
            return 0.0
        if distance >= self._max_distance:
            return 1.0
        return (distance - self._min_distance) / (self._max_distance - self._min_distance)


class PointSource(SpatialSource):
    """Point source at a single location."""

    def __init__(
        self,
        position: Vec3,
        min_distance: float = MIN_ATTENUATION_DISTANCE,
        max_distance: float = MAX_ATTENUATION_DISTANCE
    ) -> None:
        super().__init__(SourceType.POINT, min_distance, max_distance)
        self._position = position

    @property
    def position(self) -> Vec3:
        """Get source position."""
        return self._position

    @position.setter
    def position(self, value: Vec3) -> None:
        self._position = value

    def get_closest_point(self, listener_pos: Vec3) -> Vec3:
        """Point source closest point is always its position."""
        return self._position

    def get_distance(self, listener_pos: Vec3) -> float:
        """Get distance from listener to point."""
        return self._position.distance(listener_pos)

    def get_direction(self, listener_pos: Vec3) -> Vec3:
        """Get direction from listener to point."""
        diff = self._position - listener_pos
        length = diff.length()
        if length < 0.0001:
            return Vec3.forward()
        return diff / length


class AreaSource(SpatialSource):
    """Area source covering a 2D rectangular region."""

    def __init__(
        self,
        center: Vec3,
        half_extents: Vec3,
        normal: Vec3 = Vec3.up(),
        min_distance: float = MIN_ATTENUATION_DISTANCE,
        max_distance: float = MAX_ATTENUATION_DISTANCE
    ) -> None:
        super().__init__(SourceType.AREA, min_distance, max_distance)
        self._center = center
        self._half_extents = half_extents
        self._normal = normal.normalized()
        self._update_basis()

    def _update_basis(self) -> None:
        """Update local basis vectors from normal."""
        # Find a vector not parallel to normal
        if abs(self._normal.dot(Vec3.up())) < 0.99:
            self._tangent = self._normal.cross(Vec3.up()).normalized()
        else:
            self._tangent = self._normal.cross(Vec3.right()).normalized()
        self._bitangent = self._normal.cross(self._tangent).normalized()

    @property
    def center(self) -> Vec3:
        """Get area center position."""
        return self._center

    @center.setter
    def center(self, value: Vec3) -> None:
        self._center = value

    @property
    def half_extents(self) -> Vec3:
        """Get half extents (width/2, height/2, depth/2)."""
        return self._half_extents

    @half_extents.setter
    def half_extents(self, value: Vec3) -> None:
        self._half_extents = Vec3(abs(value.x), abs(value.y), abs(value.z))

    @property
    def normal(self) -> Vec3:
        """Get area normal vector."""
        return self._normal

    @normal.setter
    def normal(self, value: Vec3) -> None:
        self._normal = value.normalized()
        self._update_basis()

    def get_closest_point(self, listener_pos: Vec3) -> Vec3:
        """Get closest point on the area to the listener."""
        # Project listener position onto the area plane
        to_listener = listener_pos - self._center

        # Project onto local coordinates
        local_x = to_listener.dot(self._tangent)
        local_y = to_listener.dot(self._bitangent)

        # Clamp to area bounds
        clamped_x = max(-self._half_extents.x, min(self._half_extents.x, local_x))
        clamped_y = max(-self._half_extents.z, min(self._half_extents.z, local_y))

        # Convert back to world space
        return (
            self._center +
            self._tangent * clamped_x +
            self._bitangent * clamped_y
        )

    def get_distance(self, listener_pos: Vec3) -> float:
        """Get distance from listener to closest point on area."""
        closest = self.get_closest_point(listener_pos)
        return closest.distance(listener_pos)

    def get_direction(self, listener_pos: Vec3) -> Vec3:
        """Get direction from listener to closest point on area."""
        closest = self.get_closest_point(listener_pos)
        diff = closest - listener_pos
        length = diff.length()
        if length < 0.0001:
            return self._normal
        return diff / length

    def contains(self, point: Vec3) -> bool:
        """Check if a point is within the area volume."""
        to_point = point - self._center

        local_x = abs(to_point.dot(self._tangent))
        local_y = abs(to_point.dot(self._bitangent))
        local_z = abs(to_point.dot(self._normal))

        return (
            local_x <= self._half_extents.x and
            local_y <= self._half_extents.z and
            local_z <= self._half_extents.y
        )


class LineSource(SpatialSource):
    """Line source along a path between two points."""

    def __init__(
        self,
        start: Vec3,
        end: Vec3,
        min_distance: float = MIN_ATTENUATION_DISTANCE,
        max_distance: float = MAX_ATTENUATION_DISTANCE
    ) -> None:
        super().__init__(SourceType.LINE, min_distance, max_distance)
        self._start = start
        self._end = end

    @property
    def start(self) -> Vec3:
        """Get line start point."""
        return self._start

    @start.setter
    def start(self, value: Vec3) -> None:
        self._start = value

    @property
    def end(self) -> Vec3:
        """Get line end point."""
        return self._end

    @end.setter
    def end(self, value: Vec3) -> None:
        self._end = value

    @property
    def length(self) -> float:
        """Get line length."""
        return self._start.distance(self._end)

    @property
    def direction(self) -> Vec3:
        """Get normalized direction from start to end."""
        diff = self._end - self._start
        length = diff.length()
        if length < 0.0001:
            return Vec3.forward()
        return diff / length

    def get_closest_point(self, listener_pos: Vec3) -> Vec3:
        """Get closest point on the line to the listener."""
        line_vec = self._end - self._start
        line_len_sq = line_vec.length_squared()

        if line_len_sq < 0.0001:
            return self._start

        # Project listener onto line
        t = (listener_pos - self._start).dot(line_vec) / line_len_sq
        t = max(0.0, min(1.0, t))

        return self._start + line_vec * t

    def get_distance(self, listener_pos: Vec3) -> float:
        """Get distance from listener to closest point on line."""
        closest = self.get_closest_point(listener_pos)
        return closest.distance(listener_pos)

    def get_direction(self, listener_pos: Vec3) -> Vec3:
        """Get direction from listener to closest point on line."""
        closest = self.get_closest_point(listener_pos)
        diff = closest - listener_pos
        length = diff.length()
        if length < 0.0001:
            return self.direction
        return diff / length

    def get_point_on_line(self, t: float) -> Vec3:
        """Get point on line at parameter t (0=start, 1=end)."""
        t = max(0.0, min(1.0, t))
        return self._start.lerp(self._end, t)


class VolumeSource(SpatialSource):
    """Volume source covering a 3D box region."""

    def __init__(
        self,
        center: Vec3,
        half_extents: Vec3,
        min_distance: float = MIN_ATTENUATION_DISTANCE,
        max_distance: float = MAX_ATTENUATION_DISTANCE
    ) -> None:
        super().__init__(SourceType.VOLUME, min_distance, max_distance)
        self._center = center
        self._half_extents = Vec3(abs(half_extents.x), abs(half_extents.y), abs(half_extents.z))

    @property
    def center(self) -> Vec3:
        """Get volume center position."""
        return self._center

    @center.setter
    def center(self, value: Vec3) -> None:
        self._center = value

    @property
    def half_extents(self) -> Vec3:
        """Get volume half extents."""
        return self._half_extents

    @half_extents.setter
    def half_extents(self, value: Vec3) -> None:
        self._half_extents = Vec3(abs(value.x), abs(value.y), abs(value.z))

    @property
    def min_corner(self) -> Vec3:
        """Get minimum corner of the volume."""
        return self._center - self._half_extents

    @property
    def max_corner(self) -> Vec3:
        """Get maximum corner of the volume."""
        return self._center + self._half_extents

    def get_closest_point(self, listener_pos: Vec3) -> Vec3:
        """Get closest point on/in the volume to the listener."""
        # If inside, return listener position
        if self.contains(listener_pos):
            return listener_pos

        # Clamp to box bounds
        min_c = self.min_corner
        max_c = self.max_corner

        return Vec3(
            max(min_c.x, min(max_c.x, listener_pos.x)),
            max(min_c.y, min(max_c.y, listener_pos.y)),
            max(min_c.z, min(max_c.z, listener_pos.z))
        )

    def get_distance(self, listener_pos: Vec3) -> float:
        """Get distance from listener to volume (0 if inside)."""
        if self.contains(listener_pos):
            return 0.0
        closest = self.get_closest_point(listener_pos)
        return closest.distance(listener_pos)

    def get_direction(self, listener_pos: Vec3) -> Vec3:
        """Get direction from listener to closest point on volume."""
        if self.contains(listener_pos):
            # Return direction to center if inside
            diff = self._center - listener_pos
            length = diff.length()
            if length < 0.0001:
                return Vec3.forward()
            return diff / length

        closest = self.get_closest_point(listener_pos)
        diff = closest - listener_pos
        length = diff.length()
        if length < 0.0001:
            return Vec3.forward()
        return diff / length

    def contains(self, point: Vec3) -> bool:
        """Check if a point is inside the volume."""
        min_c = self.min_corner
        max_c = self.max_corner

        return (
            min_c.x <= point.x <= max_c.x and
            min_c.y <= point.y <= max_c.y and
            min_c.z <= point.z <= max_c.z
        )

    def get_blend_factor(self, listener_pos: Vec3, fade_distance: float) -> float:
        """Get blend factor (0-1) based on position within volume.

        Returns 1.0 when fully inside (beyond fade distance from edges),
        and interpolates to 0.0 at the edge.
        """
        if not self.contains(listener_pos):
            return 0.0

        if fade_distance <= 0.0:
            return 1.0

        # Calculate distance to nearest edge
        min_c = self.min_corner
        max_c = self.max_corner

        dist_to_edges = [
            listener_pos.x - min_c.x,
            max_c.x - listener_pos.x,
            listener_pos.y - min_c.y,
            max_c.y - listener_pos.y,
            listener_pos.z - min_c.z,
            max_c.z - listener_pos.z,
        ]

        min_dist = min(dist_to_edges)

        if min_dist >= fade_distance:
            return 1.0

        return min_dist / fade_distance


@dataclass
class SpatialSourceState:
    """Complete state of a spatial source for processing."""

    source: SpatialSource
    gain: float = 1.0
    pitch: float = 1.0
    spread: float = 0.0
    focus: float = 1.0
    cone_inner_angle: float = 360.0
    cone_outer_angle: float = 360.0
    cone_outer_gain: float = 1.0
    direction: Vec3 = field(default_factory=Vec3.forward)
    active: bool = True

    def get_cone_attenuation(self, to_listener: Vec3) -> float:
        """Calculate attenuation based on cone angles.

        Args:
            to_listener: Normalized direction from source to listener.

        Returns:
            Attenuation factor (0.0 to 1.0).
        """
        if self.cone_inner_angle >= 360.0:
            return 1.0

        # Calculate angle between source direction and to-listener
        cos_angle = self.direction.dot(to_listener)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))

        half_inner = self.cone_inner_angle / 2.0
        half_outer = self.cone_outer_angle / 2.0

        if angle <= half_inner:
            return 1.0
        elif angle >= half_outer:
            return self.cone_outer_gain
        else:
            # Interpolate between inner and outer
            # Guard against division by zero if inner == outer
            denominator = half_outer - half_inner
            if denominator < 0.0001:
                return self.cone_outer_gain
            t = (angle - half_inner) / denominator
            return 1.0 - t * (1.0 - self.cone_outer_gain)


def create_source(
    source_type: SourceType,
    position: Vec3,
    **kwargs
) -> SpatialSource:
    """Factory function to create spatial sources.

    Args:
        source_type: Type of source to create.
        position: Primary position (or center for area/volume).
        **kwargs: Additional arguments based on source type:
            - POINT: No additional args needed.
            - AREA: half_extents (Vec3), normal (Vec3, optional)
            - LINE: end (Vec3)
            - VOLUME: half_extents (Vec3)

    Returns:
        The created spatial source.

    Raises:
        ValueError: If required arguments are missing.
    """
    min_dist = kwargs.get("min_distance", MIN_ATTENUATION_DISTANCE)
    max_dist = kwargs.get("max_distance", MAX_ATTENUATION_DISTANCE)

    if source_type == SourceType.POINT:
        return PointSource(position, min_dist, max_dist)

    elif source_type == SourceType.AREA:
        half_extents = kwargs.get("half_extents")
        if half_extents is None:
            raise ValueError("AREA source requires 'half_extents'")
        normal = kwargs.get("normal", Vec3.up())
        return AreaSource(position, half_extents, normal, min_dist, max_dist)

    elif source_type == SourceType.LINE:
        end = kwargs.get("end")
        if end is None:
            raise ValueError("LINE source requires 'end' position")
        return LineSource(position, end, min_dist, max_dist)

    elif source_type == SourceType.VOLUME:
        half_extents = kwargs.get("half_extents")
        if half_extents is None:
            raise ValueError("VOLUME source requires 'half_extents'")
        return VolumeSource(position, half_extents, min_dist, max_dist)

    else:
        raise ValueError(f"Unknown source type: {source_type}")
