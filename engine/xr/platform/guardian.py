"""XR Guardian/Boundary System - Play area safety management.

This module provides guardian/chaperone/boundary functionality for XR systems:
- Play area bounds detection and visualization
- Proximity warning system
- Passthrough trigger on boundary approach
- Boundary shape management (stationary, room-scale, custom)
- Multi-user boundary coordination
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Callable, Any
import math
import logging

from engine.xr.config import XR_CONFIG

logger = logging.getLogger(__name__)


class GuardianMode(Enum):
    """Guardian system operating modes."""

    DISABLED = auto()
    STATIONARY = auto()  # Standing/seated only
    ROOM_SCALE = auto()  # Full room boundaries
    CUSTOM = auto()  # User-defined boundaries
    PASS_THROUGH = auto()  # AR passthrough mode


class BoundaryType(Enum):
    """Types of boundary geometry."""

    RECTANGLE = auto()
    POLYGON = auto()
    CYLINDER = auto()
    CUSTOM_MESH = auto()


class ProximityLevel(Enum):
    """Distance to boundary classification."""

    SAFE = auto()  # Far from boundary
    APPROACHING = auto()  # Getting close
    NEAR = auto()  # Close to boundary
    AT_BOUNDARY = auto()  # At or past boundary
    OUTSIDE = auto()  # Past boundary


@dataclass
class BoundaryVertex:
    """A vertex in the boundary polygon."""

    x: float = 0.0
    y: float = 0.0  # Height (usually floor level)
    z: float = 0.0


@dataclass
class PlayAreaBounds:
    """Play area boundary definition."""

    # Boundary vertices (floor polygon)
    vertices: List[BoundaryVertex] = field(default_factory=list)

    # Boundary type
    boundary_type: BoundaryType = BoundaryType.POLYGON

    # Size metrics
    width: float = 2.0  # X dimension in meters
    depth: float = 2.0  # Z dimension in meters
    height: float = 2.5  # Ceiling height in meters

    # Center position (world space)
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0

    # Orientation (yaw rotation in radians)
    rotation: float = 0.0

    # Floor height
    floor_height: float = 0.0

    def get_area(self) -> float:
        """Calculate play area in square meters.

        Returns:
            Area in square meters.
        """
        if self.boundary_type == BoundaryType.RECTANGLE:
            return self.width * self.depth
        elif self.boundary_type == BoundaryType.CYLINDER:
            return math.pi * (self.width / 2) ** 2

        # Calculate polygon area using shoelace formula
        if len(self.vertices) < 3:
            return 0.0

        area = 0.0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].z
            area -= self.vertices[j].x * self.vertices[i].z
        return abs(area) / 2.0

    def contains_point(self, x: float, z: float) -> bool:
        """Check if a point is inside the boundary.

        Args:
            x: X coordinate.
            z: Z coordinate.

        Returns:
            True if point is inside boundary.
        """
        if self.boundary_type == BoundaryType.RECTANGLE:
            half_w = self.width / 2
            half_d = self.depth / 2
            return (
                self.center_x - half_w <= x <= self.center_x + half_w and
                self.center_z - half_d <= z <= self.center_z + half_d
            )
        elif self.boundary_type == BoundaryType.CYLINDER:
            radius = self.width / 2
            dist = math.sqrt(
                (x - self.center_x) ** 2 +
                (z - self.center_z) ** 2
            )
            return dist <= radius

        # Ray casting for polygon
        if len(self.vertices) < 3:
            return False

        inside = False
        n = len(self.vertices)
        j = n - 1

        for i in range(n):
            vi = self.vertices[i]
            vj = self.vertices[j]

            if ((vi.z > z) != (vj.z > z)) and \
               (x < (vj.x - vi.x) * (z - vi.z) / (vj.z - vi.z) + vi.x):
                inside = not inside
            j = i

        return inside


@dataclass
class GuardianConfig:
    """Guardian system configuration."""

    # Operating mode
    mode: GuardianMode = GuardianMode.ROOM_SCALE

    # Distance thresholds (meters)
    approaching_distance: float = XR_CONFIG.platform.GUARDIAN_FADE_START_DISTANCE_M
    near_distance: float = XR_CONFIG.platform.GUARDIAN_WARNING_DISTANCE_M
    boundary_distance: float = XR_CONFIG.platform.GUARDIAN_FADE_END_DISTANCE_M

    # Visual settings
    wall_color: Tuple[float, float, float, float] = (0.0, 0.7, 1.0, 0.5)
    grid_color: Tuple[float, float, float, float] = (0.0, 0.5, 0.8, 0.3)
    warning_color: Tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.7)
    danger_color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.9)

    # Wall height
    wall_height: float = XR_CONFIG.platform.GUARDIAN_HEIGHT_M

    # Grid settings
    show_floor_grid: bool = True
    grid_cell_size: float = XR_CONFIG.platform.GUARDIAN_GRID_SIZE_M

    # Passthrough settings
    passthrough_on_proximity: bool = True
    passthrough_blend_distance: float = 0.3
    passthrough_trigger_distance: float = 0.2

    # Audio feedback
    audio_warning_enabled: bool = True
    haptic_warning_enabled: bool = True

    # Fade settings
    fade_on_exit: bool = True
    fade_distance: float = 0.1


@dataclass
class ProximityInfo:
    """Information about proximity to boundary."""

    level: ProximityLevel = ProximityLevel.SAFE
    distance: float = float("inf")
    nearest_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)


class GuardianSystem(ABC):
    """Abstract guardian/boundary system.

    Manages play area boundaries, proximity detection, and
    safety visualizations for XR experiences.
    """

    def __init__(self, config: Optional[GuardianConfig] = None) -> None:
        self._config = config or GuardianConfig()
        self._bounds: Optional[PlayAreaBounds] = None
        self._enabled = True
        self._visible = False
        self._current_proximity = ProximityInfo()

        # Event callbacks
        self._on_proximity_changed: List[Callable[[ProximityInfo], None]] = []
        self._on_boundary_crossed: List[Callable[[bool], None]] = []
        self._on_bounds_changed: List[Callable[[PlayAreaBounds], None]] = []

    @property
    def config(self) -> GuardianConfig:
        """Get guardian configuration."""
        return self._config

    @property
    def bounds(self) -> Optional[PlayAreaBounds]:
        """Get current play area bounds."""
        return self._bounds

    @property
    def enabled(self) -> bool:
        """Check if guardian is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable guardian."""
        self._enabled = value
        if not value:
            self._visible = False

    @property
    def visible(self) -> bool:
        """Check if guardian visualization is currently visible."""
        return self._visible

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the guardian system.

        Returns:
            True if initialization succeeded.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the guardian system."""
        pass

    @abstractmethod
    def request_bounds(self) -> Optional[PlayAreaBounds]:
        """Request play area bounds from the runtime.

        Returns:
            Play area bounds or None if unavailable.
        """
        pass

    @abstractmethod
    def set_custom_bounds(self, bounds: PlayAreaBounds) -> bool:
        """Set custom play area bounds.

        Args:
            bounds: Custom boundary definition.

        Returns:
            True if bounds were accepted.
        """
        pass

    @abstractmethod
    def recenter(self) -> bool:
        """Recenter the play area to current HMD position.

        Returns:
            True if recentering succeeded.
        """
        pass

    def update(
        self,
        head_position: Tuple[float, float, float],
        delta_time: float
    ) -> ProximityInfo:
        """Update guardian system state.

        Args:
            head_position: Current HMD position (x, y, z).
            delta_time: Time since last update.

        Returns:
            Current proximity information.
        """
        if not self._enabled or not self._bounds:
            return ProximityInfo()

        # Calculate distance to boundary
        proximity = self._calculate_proximity(head_position)
        old_level = self._current_proximity.level
        self._current_proximity = proximity

        # Update visibility based on proximity
        should_show = (
            proximity.level in [
                ProximityLevel.APPROACHING,
                ProximityLevel.NEAR,
                ProximityLevel.AT_BOUNDARY,
                ProximityLevel.OUTSIDE
            ]
        )

        if should_show != self._visible:
            self._visible = should_show
            self._on_visibility_changed(should_show)

        # Fire events
        if proximity.level != old_level:
            for callback in self._on_proximity_changed:
                try:
                    callback(proximity)
                except Exception as e:
                    logger.error(f"Error in proximity callback: {e}")

            # Check for boundary crossing
            crossed_out = (
                old_level != ProximityLevel.OUTSIDE and
                proximity.level == ProximityLevel.OUTSIDE
            )
            crossed_in = (
                old_level == ProximityLevel.OUTSIDE and
                proximity.level != ProximityLevel.OUTSIDE
            )

            if crossed_out or crossed_in:
                for callback in self._on_boundary_crossed:
                    try:
                        callback(crossed_out)
                    except Exception as e:
                        logger.error(f"Error in boundary callback: {e}")

        return proximity

    def _calculate_proximity(
        self,
        position: Tuple[float, float, float]
    ) -> ProximityInfo:
        """Calculate proximity to boundary.

        Args:
            position: Current position (x, y, z).

        Returns:
            Proximity information.
        """
        if not self._bounds:
            return ProximityInfo()

        x, y, z = position
        bounds = self._bounds

        # Find nearest point on boundary
        min_dist = float("inf")
        nearest_point = (0.0, 0.0, 0.0)
        normal = (0.0, 0.0, 1.0)

        if bounds.boundary_type == BoundaryType.RECTANGLE:
            half_w = bounds.width / 2
            half_d = bounds.depth / 2
            cx, cz = bounds.center_x, bounds.center_z

            # Clamp to boundary edges
            clamped_x = max(cx - half_w, min(cx + half_w, x))
            clamped_z = max(cz - half_d, min(cz + half_d, z))

            # Check each edge
            edges = [
                ((cx - half_w, cz - half_d), (cx - half_w, cz + half_d), (-1, 0, 0)),
                ((cx + half_w, cz - half_d), (cx + half_w, cz + half_d), (1, 0, 0)),
                ((cx - half_w, cz - half_d), (cx + half_w, cz - half_d), (0, 0, -1)),
                ((cx - half_w, cz + half_d), (cx + half_w, cz + half_d), (0, 0, 1)),
            ]

            for (ax, az), (bx, bz), n in edges:
                # Project point onto edge
                edge_dist = self._point_to_segment_distance(
                    (x, z), (ax, az), (bx, bz)
                )
                if edge_dist < min_dist:
                    min_dist = edge_dist
                    normal = n
                    # Find nearest point on segment
                    nearest_point = self._nearest_point_on_segment(
                        (x, z), (ax, az), (bx, bz)
                    )
                    nearest_point = (nearest_point[0], y, nearest_point[1])

        elif bounds.boundary_type == BoundaryType.POLYGON:
            # Check distance to each edge
            n_verts = len(bounds.vertices)
            for i in range(n_verts):
                v1 = bounds.vertices[i]
                v2 = bounds.vertices[(i + 1) % n_verts]

                edge_dist = self._point_to_segment_distance(
                    (x, z), (v1.x, v1.z), (v2.x, v2.z)
                )

                if edge_dist < min_dist:
                    min_dist = edge_dist
                    nearest_point_2d = self._nearest_point_on_segment(
                        (x, z), (v1.x, v1.z), (v2.x, v2.z)
                    )
                    nearest_point = (nearest_point_2d[0], y, nearest_point_2d[1])

                    # Calculate outward normal
                    dx = v2.x - v1.x
                    dz = v2.z - v1.z
                    length = math.sqrt(dx * dx + dz * dz)
                    if length > 0.001:
                        normal = (dz / length, 0, -dx / length)

        elif bounds.boundary_type == BoundaryType.CYLINDER:
            radius = bounds.width / 2
            dx = x - bounds.center_x
            dz = z - bounds.center_z
            dist_to_center = math.sqrt(dx * dx + dz * dz)

            if dist_to_center > 0.001:
                # Nearest point on cylinder surface
                scale = radius / dist_to_center
                nearest_point = (
                    bounds.center_x + dx * scale,
                    y,
                    bounds.center_z + dz * scale
                )
                normal = (dx / dist_to_center, 0, dz / dist_to_center)
                min_dist = abs(dist_to_center - radius)
            else:
                nearest_point = (bounds.center_x + radius, y, bounds.center_z)
                normal = (1, 0, 0)
                min_dist = radius

        # Determine proximity level
        inside = bounds.contains_point(x, z)
        if not inside:
            level = ProximityLevel.OUTSIDE
        elif min_dist < self._config.boundary_distance:
            level = ProximityLevel.AT_BOUNDARY
        elif min_dist < self._config.near_distance:
            level = ProximityLevel.NEAR
        elif min_dist < self._config.approaching_distance:
            level = ProximityLevel.APPROACHING
        else:
            level = ProximityLevel.SAFE

        return ProximityInfo(
            level=level,
            distance=min_dist,
            nearest_point=nearest_point,
            normal=normal
        )

    def _point_to_segment_distance(
        self,
        point: Tuple[float, float],
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float]
    ) -> float:
        """Calculate distance from point to line segment.

        Args:
            point: The point (x, z).
            seg_start: Segment start point.
            seg_end: Segment end point.

        Returns:
            Distance to segment.
        """
        px, pz = point
        ax, az = seg_start
        bx, bz = seg_end

        dx = bx - ax
        dz = bz - az
        length_sq = dx * dx + dz * dz

        if length_sq < 0.0001:
            # Degenerate segment
            return math.sqrt((px - ax) ** 2 + (pz - az) ** 2)

        t = max(0, min(1, ((px - ax) * dx + (pz - az) * dz) / length_sq))

        proj_x = ax + t * dx
        proj_z = az + t * dz

        return math.sqrt((px - proj_x) ** 2 + (pz - proj_z) ** 2)

    def _nearest_point_on_segment(
        self,
        point: Tuple[float, float],
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Find nearest point on line segment to given point.

        Args:
            point: The point (x, z).
            seg_start: Segment start point.
            seg_end: Segment end point.

        Returns:
            Nearest point on segment.
        """
        px, pz = point
        ax, az = seg_start
        bx, bz = seg_end

        dx = bx - ax
        dz = bz - az
        length_sq = dx * dx + dz * dz

        if length_sq < 0.0001:
            return seg_start

        t = max(0, min(1, ((px - ax) * dx + (pz - az) * dz) / length_sq))

        return (ax + t * dx, az + t * dz)

    def _on_visibility_changed(self, visible: bool) -> None:
        """Handle visibility change.

        Args:
            visible: Whether guardian is now visible.
        """
        logger.debug(f"Guardian visibility: {visible}")

    def on_proximity_changed(
        self,
        callback: Callable[[ProximityInfo], None]
    ) -> None:
        """Register proximity change callback.

        Args:
            callback: Function to call when proximity changes.
        """
        self._on_proximity_changed.append(callback)

    def on_boundary_crossed(
        self,
        callback: Callable[[bool], None]
    ) -> None:
        """Register boundary crossing callback.

        Args:
            callback: Function to call when boundary is crossed.
                     Argument is True if exiting, False if entering.
        """
        self._on_boundary_crossed.append(callback)

    def on_bounds_changed(
        self,
        callback: Callable[[PlayAreaBounds], None]
    ) -> None:
        """Register bounds change callback.

        Args:
            callback: Function to call when bounds change.
        """
        self._on_bounds_changed.append(callback)

    def get_passthrough_blend(self) -> float:
        """Get passthrough blend factor based on proximity.

        Returns:
            Blend factor (0 = no passthrough, 1 = full passthrough).
        """
        if not self._config.passthrough_on_proximity:
            return 0.0

        if not self._enabled or not self._bounds:
            return 0.0

        distance = self._current_proximity.distance
        trigger = self._config.passthrough_trigger_distance
        blend_range = self._config.passthrough_blend_distance

        if distance > trigger + blend_range:
            return 0.0
        elif distance < trigger:
            return 1.0
        else:
            return 1.0 - (distance - trigger) / blend_range

    def get_warning_intensity(self) -> float:
        """Get visual warning intensity based on proximity.

        Returns:
            Warning intensity (0 = none, 1 = maximum).
        """
        if not self._enabled or not self._bounds:
            return 0.0

        level = self._current_proximity.level

        if level == ProximityLevel.OUTSIDE:
            return 1.0
        elif level == ProximityLevel.AT_BOUNDARY:
            return 0.9
        elif level == ProximityLevel.NEAR:
            dist = self._current_proximity.distance
            near = self._config.near_distance
            boundary = self._config.boundary_distance
            return 0.5 + 0.4 * (1 - (dist - boundary) / (near - boundary))
        elif level == ProximityLevel.APPROACHING:
            dist = self._current_proximity.distance
            approaching = self._config.approaching_distance
            near = self._config.near_distance
            return 0.2 + 0.3 * (1 - (dist - near) / (approaching - near))
        else:
            return 0.0


class OpenXRGuardian(GuardianSystem):
    """OpenXR-based guardian implementation."""

    def initialize(self) -> bool:
        """Initialize OpenXR reference space bounds."""
        try:
            self._bounds = self.request_bounds()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenXR guardian: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown guardian system."""
        self._bounds = None
        self._enabled = False

    def request_bounds(self) -> Optional[PlayAreaBounds]:
        """Request bounds from OpenXR runtime."""
        # TODO: Query xrGetReferenceSpaceBoundsRect
        # This would return the stage bounds

        # Default room-scale bounds for development
        bounds = PlayAreaBounds(
            boundary_type=BoundaryType.RECTANGLE,
            width=2.5,
            depth=2.5,
            height=2.5,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
        )

        return bounds

    def set_custom_bounds(self, bounds: PlayAreaBounds) -> bool:
        """Set custom bounds (not supported in OpenXR)."""
        # OpenXR doesn't support custom bounds
        # We can only use the configured play area
        logger.warning("Custom bounds not supported in OpenXR")
        return False

    def recenter(self) -> bool:
        """Recenter via OpenXR."""
        # TODO: Recenter the reference space
        return True


class SteamVRGuardian(GuardianSystem):
    """SteamVR Chaperone implementation."""

    def initialize(self) -> bool:
        """Initialize SteamVR chaperone."""
        try:
            self._bounds = self.request_bounds()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SteamVR chaperone: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown chaperone system."""
        self._bounds = None
        self._enabled = False

    def request_bounds(self) -> Optional[PlayAreaBounds]:
        """Request bounds from SteamVR chaperone."""
        # TODO: Query IVRChaperone
        # GetPlayAreaRect returns the rectangle
        # GetPlayAreaSize returns dimensions

        bounds = PlayAreaBounds(
            boundary_type=BoundaryType.RECTANGLE,
            width=2.5,
            depth=2.5,
            height=2.5,
        )

        return bounds

    def set_custom_bounds(self, bounds: PlayAreaBounds) -> bool:
        """Set custom bounds via SteamVR."""
        # SteamVR allows setting bounds via the overlay
        self._bounds = bounds
        return True

    def recenter(self) -> bool:
        """Recenter via SteamVR."""
        # TODO: ResetZeroPose
        return True


class QuestGuardian(GuardianSystem):
    """Meta Quest Guardian implementation."""

    def initialize(self) -> bool:
        """Initialize Quest guardian."""
        try:
            self._bounds = self.request_bounds()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Quest guardian: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown guardian system."""
        self._bounds = None
        self._enabled = False

    def request_bounds(self) -> Optional[PlayAreaBounds]:
        """Request bounds from Quest guardian."""
        # TODO: Query OVR_Guardian
        # OvrBoundary_GetGeometry returns polygon vertices

        # Default Quest play area
        bounds = PlayAreaBounds(
            boundary_type=BoundaryType.POLYGON,
            width=3.0,
            depth=3.0,
            height=2.5,
            vertices=[
                BoundaryVertex(-1.5, 0, -1.5),
                BoundaryVertex(1.5, 0, -1.5),
                BoundaryVertex(1.5, 0, 1.5),
                BoundaryVertex(-1.5, 0, 1.5),
            ]
        )

        return bounds

    def set_custom_bounds(self, bounds: PlayAreaBounds) -> bool:
        """Set custom bounds (requires Guardian setup)."""
        logger.warning("Custom bounds require Quest Guardian setup")
        return False

    def recenter(self) -> bool:
        """Recenter via Quest."""
        # TODO: OvrBoundary_RequestVisible
        return True


def create_guardian_system(runtime: str = "openxr") -> GuardianSystem:
    """Create appropriate guardian system for runtime.

    Args:
        runtime: XR runtime name (openxr, steamvr, quest).

    Returns:
        Guardian system instance.
    """
    runtime_lower = runtime.lower()

    if runtime_lower == "steamvr":
        return SteamVRGuardian()
    elif runtime_lower == "quest":
        return QuestGuardian()
    else:
        return OpenXRGuardian()


__all__ = [
    # Enums
    "GuardianMode",
    "BoundaryType",
    "ProximityLevel",
    # Data classes
    "BoundaryVertex",
    "PlayAreaBounds",
    "GuardianConfig",
    "ProximityInfo",
    # Systems
    "GuardianSystem",
    "OpenXRGuardian",
    "SteamVRGuardian",
    "QuestGuardian",
    # Factory
    "create_guardian_system",
]
