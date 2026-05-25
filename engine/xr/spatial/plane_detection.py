"""Plane detection for AR spatial understanding.

Detects and tracks real-world surfaces including floors, walls, ceilings,
tables, and seats. Provides geometry data for placing virtual content.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec2, Vec3
from engine.core.math.quat import Quat
from engine.xr.config import XR_CONFIG


class PlaneType(Enum):
    """Classification of detected planes."""
    UNKNOWN = auto()      # Unclassified plane
    FLOOR = auto()        # Horizontal floor surface
    CEILING = auto()      # Horizontal ceiling surface
    WALL = auto()         # Vertical wall surface
    TABLE = auto()        # Horizontal table/desk surface
    SEAT = auto()         # Seating surface (chair, couch)
    DOOR = auto()         # Door surface
    WINDOW = auto()       # Window surface


class PlaneOrientation(Enum):
    """Orientation of a plane."""
    HORIZONTAL_UP = auto()    # Facing upward (floor, table)
    HORIZONTAL_DOWN = auto()  # Facing downward (ceiling)
    VERTICAL = auto()         # Vertical (wall)
    ARBITRARY = auto()        # Non-axis-aligned


class PlaneTrackingState(Enum):
    """Tracking state of a detected plane."""
    NONE = auto()          # No plane detected
    DETECTING = auto()     # Detection in progress
    TRACKED = auto()       # Actively tracked
    LIMITED = auto()       # Tracking with reduced accuracy
    PAUSED = auto()        # Temporarily not tracking
    STOPPED = auto()       # Tracking stopped


class PlaneAlignment(Enum):
    """Alignment constraint for plane detection."""
    ANY = auto()           # Detect any orientation
    HORIZONTAL = auto()    # Only horizontal planes
    VERTICAL = auto()      # Only vertical planes


@dataclass(slots=True)
class PlaneBounds:
    """Boundary polygon for a detected plane."""
    vertices: list[Vec2] = field(default_factory=list)
    center_local: Vec2 = field(default_factory=lambda: Vec2(0, 0))

    @property
    def vertex_count(self) -> int:
        """Get the number of vertices in the boundary."""
        return len(self.vertices)

    def contains_point(self, point: Vec2) -> bool:
        """Check if a 2D point is inside the boundary polygon.

        Uses ray casting algorithm.

        Args:
            point: Point to test

        Returns:
            True if point is inside polygon
        """
        if len(self.vertices) < 3:
            return False

        inside = False
        n = len(self.vertices)
        j = n - 1

        for i in range(n):
            vi = self.vertices[i]
            vj = self.vertices[j]

            if ((vi.y > point.y) != (vj.y > point.y)) and \
               (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x):
                inside = not inside
            j = i

        return inside

    def compute_area(self) -> float:
        """Compute the area of the boundary polygon.

        Uses the shoelace formula.

        Returns:
            Area in square meters
        """
        if len(self.vertices) < 3:
            return 0.0

        area = 0.0
        n = len(self.vertices)

        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y

        return abs(area) / 2.0


@dataclass(slots=True)
class PlaneGeometry:
    """Geometry data for a detected plane."""
    center: Vec3 = field(default_factory=Vec3.zero)
    normal: Vec3 = field(default_factory=Vec3.up)
    orientation: Quat = field(default_factory=Quat.identity)
    width: float = 0.0
    height: float = 0.0
    bounds: PlaneBounds = field(default_factory=PlaneBounds)

    @property
    def area(self) -> float:
        """Get the approximate area of the plane."""
        return self.bounds.compute_area() if self.bounds.vertices else self.width * self.height

    @property
    def extents(self) -> Vec2:
        """Get the plane extents (half-size)."""
        return Vec2(self.width / 2.0, self.height / 2.0)


class DetectedPlane:
    """A detected real-world surface.

    Represents a plane detected by the AR system with classification,
    geometry data, and tracking state.

    Attributes:
        plane_id: Unique identifier
        plane_type: Classification (floor, wall, table, etc.)
        geometry: Position, orientation, and boundary data
        tracking_state: Current tracking status
    """
    __slots__ = (
        '_plane_id',
        '_plane_type',
        '_plane_orientation',
        '_geometry',
        '_tracking_state',
        '_confidence',
        '_is_valid',
        '_native_handle',
        '_subsumed_by',
        '_last_updated',
        '_created_timestamp',
        '_callbacks',
    )

    def __init__(
        self,
        plane_type: PlaneType = PlaneType.UNKNOWN,
        center: Optional[Vec3] = None,
        normal: Optional[Vec3] = None,
    ) -> None:
        """Initialize a detected plane.

        Args:
            plane_type: Classification of the plane
            center: World-space center position
            normal: Surface normal direction
        """
        self._plane_id: str = str(uuid.uuid4())
        self._plane_type: PlaneType = plane_type
        self._plane_orientation: PlaneOrientation = self._compute_orientation(
            normal or Vec3.up()
        )
        self._geometry: PlaneGeometry = PlaneGeometry(
            center=center or Vec3.zero(),
            normal=normal or Vec3.up(),
        )
        self._tracking_state: PlaneTrackingState = PlaneTrackingState.NONE
        self._confidence: float = 0.0
        self._is_valid: bool = False
        self._native_handle: Optional[int] = None
        self._subsumed_by: Optional[str] = None
        self._last_updated: float = 0.0
        self._created_timestamp: float = 0.0
        self._callbacks: dict[str, list[Callable]] = {
            "geometry_updated": [],
            "tracking_changed": [],
            "subsumed": [],
        }

    def _compute_orientation(self, normal: Vec3) -> PlaneOrientation:
        """Determine plane orientation from normal.

        Args:
            normal: Surface normal

        Returns:
            Plane orientation classification
        """
        up_dot = abs(normal.dot(Vec3.up()))

        if up_dot > XR_CONFIG.spatial.HORIZONTAL_PLANE_THRESHOLD:
            if normal.y > 0:
                return PlaneOrientation.HORIZONTAL_UP
            else:
                return PlaneOrientation.HORIZONTAL_DOWN
        elif up_dot < XR_CONFIG.spatial.VERTICAL_PLANE_THRESHOLD:
            return PlaneOrientation.VERTICAL
        else:
            return PlaneOrientation.ARBITRARY

    @property
    def plane_id(self) -> str:
        """Get the unique plane identifier."""
        return self._plane_id

    @property
    def plane_type(self) -> PlaneType:
        """Get the plane classification."""
        return self._plane_type

    @property
    def plane_orientation(self) -> PlaneOrientation:
        """Get the plane orientation."""
        return self._plane_orientation

    @property
    def center(self) -> Vec3:
        """Get the world-space center position."""
        return self._geometry.center

    @property
    def normal(self) -> Vec3:
        """Get the surface normal."""
        return self._geometry.normal

    @property
    def orientation(self) -> Quat:
        """Get the plane orientation quaternion."""
        return self._geometry.orientation

    @property
    def width(self) -> float:
        """Get the plane width in meters."""
        return self._geometry.width

    @property
    def height(self) -> float:
        """Get the plane height in meters."""
        return self._geometry.height

    @property
    def area(self) -> float:
        """Get the plane area in square meters."""
        return self._geometry.area

    @property
    def bounds(self) -> PlaneBounds:
        """Get the boundary polygon."""
        return self._geometry.bounds

    @property
    def geometry(self) -> PlaneGeometry:
        """Get the full geometry data."""
        return self._geometry

    @property
    def tracking_state(self) -> PlaneTrackingState:
        """Get the current tracking state."""
        return self._tracking_state

    @property
    def is_tracked(self) -> bool:
        """Check if the plane is actively tracked."""
        return self._tracking_state == PlaneTrackingState.TRACKED

    @property
    def confidence(self) -> float:
        """Get tracking confidence (0.0 to 1.0)."""
        return self._confidence

    @property
    def is_valid(self) -> bool:
        """Check if the plane data is valid."""
        return self._is_valid

    @property
    def is_horizontal(self) -> bool:
        """Check if this is a horizontal plane."""
        return self._plane_orientation in (
            PlaneOrientation.HORIZONTAL_UP,
            PlaneOrientation.HORIZONTAL_DOWN,
        )

    @property
    def is_vertical(self) -> bool:
        """Check if this is a vertical plane."""
        return self._plane_orientation == PlaneOrientation.VERTICAL

    @property
    def subsumed_by(self) -> Optional[str]:
        """Get the ID of the plane that subsumed this one."""
        return self._subsumed_by

    @property
    def is_subsumed(self) -> bool:
        """Check if this plane was merged into another."""
        return self._subsumed_by is not None

    def update_geometry(
        self,
        center: Vec3,
        normal: Vec3,
        orientation: Quat,
        width: float,
        height: float,
        boundary_vertices: list[Vec2],
        timestamp: float,
    ) -> None:
        """Update the plane geometry.

        Args:
            center: New center position
            normal: New surface normal
            orientation: New orientation quaternion
            width: New width
            height: New height
            boundary_vertices: New boundary polygon vertices
            timestamp: Update timestamp
        """
        self._geometry.center = center
        self._geometry.normal = normal
        self._geometry.orientation = orientation
        self._geometry.width = width
        self._geometry.height = height
        self._geometry.bounds.vertices = boundary_vertices

        # Compute local center
        if boundary_vertices:
            sum_x = sum(v.x for v in boundary_vertices)
            sum_y = sum(v.y for v in boundary_vertices)
            n = len(boundary_vertices)
            self._geometry.bounds.center_local = Vec2(sum_x / n, sum_y / n)

        self._plane_orientation = self._compute_orientation(normal)
        self._last_updated = timestamp
        self._notify_callbacks("geometry_updated")

    def update_classification(self, plane_type: PlaneType) -> None:
        """Update the plane classification.

        Args:
            plane_type: New classification
        """
        self._plane_type = plane_type

    def update_tracking_state(
        self,
        state: PlaneTrackingState,
        confidence: float = 0.0,
    ) -> None:
        """Update the tracking state.

        Args:
            state: New tracking state
            confidence: Optional confidence update
        """
        old_state = self._tracking_state
        self._tracking_state = state
        if confidence > 0.0:
            self._confidence = max(0.0, min(1.0, confidence))

        self._is_valid = state == PlaneTrackingState.TRACKED

        if old_state != state:
            self._notify_callbacks("tracking_changed")

    def mark_subsumed(self, by_plane_id: str) -> None:
        """Mark this plane as merged into another.

        Args:
            by_plane_id: ID of the plane that absorbed this one
        """
        self._subsumed_by = by_plane_id
        self._tracking_state = PlaneTrackingState.STOPPED
        self._is_valid = False
        self._notify_callbacks("subsumed")

    def world_to_local(self, world_point: Vec3) -> Vec2:
        """Convert a world point to local plane coordinates.

        Args:
            world_point: Point in world space

        Returns:
            Point in local 2D plane coordinates
        """
        # Transform to plane-local space
        local = world_point - self._geometry.center
        inv_rot = self._geometry.orientation.inverse()
        rotated = inv_rot.rotate_vector(local)
        return Vec2(rotated.x, rotated.z)

    def local_to_world(self, local_point: Vec2) -> Vec3:
        """Convert a local plane point to world space.

        Args:
            local_point: Point in local 2D plane coordinates

        Returns:
            Point in world space
        """
        local_3d = Vec3(local_point.x, 0.0, local_point.y)
        rotated = self._geometry.orientation.rotate_vector(local_3d)
        return rotated + self._geometry.center

    def project_point(self, world_point: Vec3) -> Vec3:
        """Project a world point onto the plane surface.

        Args:
            world_point: Point to project

        Returns:
            Projected point on plane surface
        """
        to_point = world_point - self._geometry.center
        distance = to_point.dot(self._geometry.normal)
        return world_point - self._geometry.normal * distance

    def distance_to_point(self, world_point: Vec3) -> float:
        """Get the signed distance from a point to the plane.

        Args:
            world_point: Point to measure from

        Returns:
            Signed distance (positive = in front of plane)
        """
        to_point = world_point - self._geometry.center
        return to_point.dot(self._geometry.normal)

    def contains_point(self, world_point: Vec3, tolerance: float = XR_CONFIG.spatial.DEFAULT_PLANE_TOLERANCE) -> bool:
        """Check if a world point is on this plane's surface.

        Args:
            world_point: Point to test
            tolerance: Distance tolerance in meters

        Returns:
            True if point is on the plane surface
        """
        distance = abs(self.distance_to_point(world_point))
        if distance > tolerance:
            return False

        local = self.world_to_local(world_point)
        return self._geometry.bounds.contains_point(local)

    def add_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for plane events.

        Args:
            event: Event name
            callback: Function to call
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def remove_callback(self, event: str, callback: Callable) -> None:
        """Remove a registered callback.

        Args:
            event: Event name
            callback: Callback to remove
        """
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def _notify_callbacks(self, event: str) -> None:
        """Notify callbacks for an event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                callback(self)

    def __repr__(self) -> str:
        return (
            f"DetectedPlane(id={self._plane_id[:8]}..., "
            f"type={self._plane_type.name}, "
            f"area={self.area:.2f}m2, "
            f"tracking={self._tracking_state.name})"
        )


@dataclass(slots=True)
class PlaneDetectionConfig:
    """Configuration for plane detection."""
    alignment: PlaneAlignment = PlaneAlignment.ANY
    min_area: float = XR_CONFIG.spatial.PLANE_MIN_AREA_M2  # Minimum plane area in m2
    max_planes: int = XR_CONFIG.spatial.MAX_ANCHORS_PER_SESSION  # Maximum planes to track
    merge_planes: bool = True
    classification_enabled: bool = True
    update_rate: float = 30.0  # Updates per second


class PlaneDetector:
    """Manages plane detection for AR.

    Handles detection, tracking, merging, and classification of
    real-world surfaces.

    Attributes:
        config: Detection configuration
        planes: Currently detected planes
    """
    __slots__ = (
        '_config',
        '_planes',
        '_is_running',
        '_callbacks',
        '_last_update',
    )

    def __init__(self, config: Optional[PlaneDetectionConfig] = None) -> None:
        """Initialize the plane detector.

        Args:
            config: Detection configuration
        """
        self._config: PlaneDetectionConfig = config or PlaneDetectionConfig()
        self._planes: dict[str, DetectedPlane] = {}
        self._is_running: bool = False
        self._callbacks: dict[str, list[Callable]] = {
            "plane_added": [],
            "plane_removed": [],
            "plane_updated": [],
        }
        self._last_update: float = 0.0

    @property
    def config(self) -> PlaneDetectionConfig:
        """Get the detection configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if detection is active."""
        return self._is_running

    @property
    def plane_count(self) -> int:
        """Get the number of detected planes."""
        return len(self._planes)

    def start(self) -> bool:
        """Start plane detection.

        Returns:
            True if started successfully
        """
        if self._is_running:
            return False
        self._is_running = True
        return True

    def stop(self) -> bool:
        """Stop plane detection.

        Returns:
            True if stopped successfully
        """
        if not self._is_running:
            return False
        self._is_running = False
        return True

    def update(self, timestamp: float) -> None:
        """Update plane detection.

        Args:
            timestamp: Current time
        """
        if not self._is_running:
            return

        self._last_update = timestamp

        # Remove subsumed planes
        subsumed = [
            pid for pid, plane in self._planes.items()
            if plane.is_subsumed
        ]
        for pid in subsumed:
            self._remove_plane(pid)

    def get_plane(self, plane_id: str) -> Optional[DetectedPlane]:
        """Get a plane by ID.

        Args:
            plane_id: Plane identifier

        Returns:
            Plane if found, None otherwise
        """
        return self._planes.get(plane_id)

    def get_all_planes(self) -> list[DetectedPlane]:
        """Get all detected planes.

        Returns:
            List of all planes
        """
        return list(self._planes.values())

    def get_planes_by_type(self, plane_type: PlaneType) -> list[DetectedPlane]:
        """Get planes by classification.

        Args:
            plane_type: Type to filter by

        Returns:
            List of matching planes
        """
        return [p for p in self._planes.values() if p.plane_type == plane_type]

    def get_horizontal_planes(self) -> list[DetectedPlane]:
        """Get all horizontal planes (floors, tables, etc.).

        Returns:
            List of horizontal planes
        """
        return [p for p in self._planes.values() if p.is_horizontal]

    def get_vertical_planes(self) -> list[DetectedPlane]:
        """Get all vertical planes (walls).

        Returns:
            List of vertical planes
        """
        return [p for p in self._planes.values() if p.is_vertical]

    def get_floor_planes(self) -> list[DetectedPlane]:
        """Get detected floor planes.

        Returns:
            List of floor planes
        """
        return self.get_planes_by_type(PlaneType.FLOOR)

    def get_wall_planes(self) -> list[DetectedPlane]:
        """Get detected wall planes.

        Returns:
            List of wall planes
        """
        return self.get_planes_by_type(PlaneType.WALL)

    def get_table_planes(self) -> list[DetectedPlane]:
        """Get detected table/surface planes.

        Returns:
            List of table planes
        """
        return self.get_planes_by_type(PlaneType.TABLE)

    def raycast(
        self,
        origin: Vec3,
        direction: Vec3,
        max_distance: float = XR_CONFIG.spatial.MAX_RAYCAST_DISTANCE,
        filter_types: Optional[list[PlaneType]] = None,
    ) -> Optional[tuple[DetectedPlane, Vec3, float]]:
        """Cast a ray and find the first plane intersection.

        Args:
            origin: Ray origin
            direction: Ray direction (normalized)
            max_distance: Maximum ray distance
            filter_types: Optional plane types to consider

        Returns:
            Tuple of (plane, hit_point, distance) or None
        """
        closest_hit: Optional[tuple[DetectedPlane, Vec3, float]] = None
        closest_distance = max_distance

        for plane in self._planes.values():
            if not plane.is_tracked:
                continue

            if filter_types and plane.plane_type not in filter_types:
                continue

            # Ray-plane intersection
            denom = direction.dot(plane.normal)
            if abs(denom) < XR_CONFIG.spatial.RAY_EPSILON:
                continue  # Parallel to plane

            t = (plane.center - origin).dot(plane.normal) / denom
            if t < 0 or t > closest_distance:
                continue  # Behind ray or too far

            hit_point = origin + direction * t

            # Check if hit is within plane bounds
            if plane.contains_point(hit_point):
                closest_hit = (plane, hit_point, t)
                closest_distance = t

        return closest_hit

    def find_placement_surface(
        self,
        position: Vec3,
        prefer_horizontal: bool = True,
        min_area: float = XR_CONFIG.spatial.MIN_PLACEMENT_AREA_M2,
    ) -> Optional[tuple[DetectedPlane, Vec3]]:
        """Find a suitable surface for placing content.

        Args:
            position: Desired placement position
            prefer_horizontal: Prefer horizontal surfaces
            min_area: Minimum required area

        Returns:
            Tuple of (plane, snap_point) or None
        """
        best_plane: Optional[DetectedPlane] = None
        best_distance = float('inf')
        best_point = Vec3.zero()

        for plane in self._planes.values():
            if not plane.is_tracked:
                continue

            if plane.area < min_area:
                continue

            if prefer_horizontal and not plane.is_horizontal:
                continue

            # Project position onto plane
            projected = plane.project_point(position)
            local = plane.world_to_local(projected)

            # Check if within bounds
            if not plane.bounds.contains_point(local):
                continue

            distance = position.distance(projected)
            if distance < best_distance:
                best_distance = distance
                best_plane = plane
                best_point = projected

        if best_plane:
            return (best_plane, best_point)
        return None

    def add_plane(self, plane: DetectedPlane) -> None:
        """Add a detected plane.

        Args:
            plane: Plane to add
        """
        self._planes[plane.plane_id] = plane
        self._notify_callbacks("plane_added", plane)

    def _remove_plane(self, plane_id: str) -> None:
        """Remove a plane by ID.

        Args:
            plane_id: Plane to remove
        """
        plane = self._planes.pop(plane_id, None)
        if plane:
            self._notify_callbacks("plane_removed", plane)

    def clear_all(self) -> None:
        """Clear all detected planes."""
        for plane_id in list(self._planes.keys()):
            self._remove_plane(plane_id)

    def add_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for detection events.

        Args:
            event: Event name
            callback: Function to call
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def remove_callback(self, event: str, callback: Callable) -> None:
        """Remove a registered callback.

        Args:
            event: Event name
            callback: Callback to remove
        """
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def _notify_callbacks(self, event: str, plane: DetectedPlane) -> None:
        """Notify callbacks for an event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                callback(plane)
