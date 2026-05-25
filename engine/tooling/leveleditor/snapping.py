"""
Snapping System - Precision snapping tools for object placement.

Provides multiple snapping modes:
- Grid: Snap to configurable grid
- Surface: Snap to surfaces with normal alignment
- Vertex: Snap to mesh vertices
- Edge: Snap to mesh edges
- Pivot: Snap to object pivots

All snap operations integrate with the Foundation Tracker for undo/redo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Any, Optional, Protocol

from .placement import Vector3, Quaternion, Transform, editor, track_changes
from foundation.tracker import tracker


# =============================================================================
# Enums and Flags
# =============================================================================

class SnapMode(Flag):
    """Snap modes that can be combined."""
    NONE = 0
    GRID = auto()
    SURFACE = auto()
    VERTEX = auto()
    EDGE = auto()
    PIVOT = auto()
    ALL = GRID | SURFACE | VERTEX | EDGE | PIVOT


class SnapPriority(Enum):
    """Priority order for snap resolution when multiple snaps are active."""
    VERTEX_FIRST = auto()  # Vertex > Edge > Surface > Grid
    SURFACE_FIRST = auto()  # Surface > Vertex > Edge > Grid
    GRID_FIRST = auto()  # Grid > Surface > Vertex > Edge
    NEAREST = auto()  # Use whatever snap point is closest


class GridType(Enum):
    """Type of grid snap."""
    WORLD = auto()  # Fixed world-space grid
    LOCAL = auto()  # Grid relative to object
    CUSTOM = auto()  # Custom origin and orientation


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(slots=True)
class SnapSettings:
    """Global snap settings."""
    enabled: bool = True
    mode: SnapMode = SnapMode.GRID
    priority: SnapPriority = SnapPriority.NEAREST
    snap_radius: float = 10.0  # Max distance for non-grid snaps
    visual_feedback: bool = True
    show_snap_points: bool = False


@dataclass(slots=True)
class GridSnapSettings:
    """Settings for grid snapping."""
    enabled: bool = True
    grid_type: GridType = GridType.WORLD
    size_x: float = 1.0
    size_y: float = 1.0
    size_z: float = 1.0
    origin: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    snap_to_integer: bool = False
    subdivisions: int = 1


@dataclass(slots=True)
class SurfaceSnapSettings:
    """Settings for surface snapping."""
    enabled: bool = True
    align_to_normal: bool = True
    normal_offset: float = 0.0
    layer_mask: int = 0xFFFFFFFF
    max_distance: float = 1000.0
    back_face_culling: bool = True


@dataclass(slots=True)
class VertexSnapSettings:
    """Settings for vertex snapping."""
    enabled: bool = True
    snap_radius: float = 5.0
    highlight_color: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0)
    layer_mask: int = 0xFFFFFFFF


@dataclass(slots=True)
class EdgeSnapSettings:
    """Settings for edge snapping."""
    enabled: bool = True
    snap_radius: float = 5.0
    snap_to_midpoint: bool = True
    snap_to_perpendicular: bool = True
    highlight_color: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0)
    layer_mask: int = 0xFFFFFFFF


@dataclass(slots=True)
class PivotSnapSettings:
    """Settings for pivot snapping."""
    enabled: bool = True
    snap_radius: float = 10.0
    include_bounds_centers: bool = True
    highlight_color: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 1.0)


@dataclass(slots=True)
class SnapResult:
    """Result of a snap operation."""
    snapped: bool
    position: Vector3
    normal: Optional[Vector3] = None
    rotation: Optional[Quaternion] = None
    snap_type: SnapMode = SnapMode.NONE
    snap_target_id: Optional[str] = None
    distance: float = 0.0


@dataclass(slots=True)
class VertexInfo:
    """Information about a vertex for snapping."""
    position: Vector3
    normal: Optional[Vector3] = None
    object_id: str = ""
    vertex_index: int = 0


@dataclass(slots=True)
class EdgeInfo:
    """Information about an edge for snapping."""
    start: Vector3
    end: Vector3
    object_id: str = ""
    edge_index: int = 0

    @property
    def midpoint(self) -> Vector3:
        return Vector3(
            (self.start.x + self.end.x) / 2,
            (self.start.y + self.end.y) / 2,
            (self.start.z + self.end.z) / 2,
        )

    @property
    def direction(self) -> Vector3:
        return (self.end - self.start).normalized()

    @property
    def length(self) -> float:
        return (self.end - self.start).length()


@dataclass(slots=True)
class PivotInfo:
    """Information about a pivot point."""
    position: Vector3
    object_id: str = ""
    is_bounds_center: bool = False


# =============================================================================
# Protocols
# =============================================================================

class MeshProvider(Protocol):
    """Protocol for mesh data access."""
    def get_vertices(self, object_id: str) -> list[VertexInfo]: ...
    def get_edges(self, object_id: str) -> list[EdgeInfo]: ...


class SceneProvider(Protocol):
    """Protocol for scene object access."""
    def get_object_ids(self, layer_mask: int = 0xFFFFFFFF) -> list[str]: ...
    def get_object_transform(self, object_id: str) -> Transform: ...
    def get_object_bounds(self, object_id: str) -> tuple[Vector3, Vector3]: ...
    def raycast(
        self,
        origin: Vector3,
        direction: Vector3,
        max_distance: float,
        layer_mask: int
    ) -> Optional[tuple[str, Vector3, Vector3, float]]: ...


# =============================================================================
# Snap Implementations
# =============================================================================

@editor
class GridSnap:
    """Grid-based snapping implementation."""

    __slots__ = ("_settings",)

    def __init__(self, settings: Optional[GridSnapSettings] = None):
        self._settings = settings or GridSnapSettings()

    @property
    def settings(self) -> GridSnapSettings:
        return self._settings

    @settings.setter
    def settings(self, value: GridSnapSettings) -> None:
        self._settings = value

    def snap(self, position: Vector3) -> SnapResult:
        """
        Snap position to grid.

        Args:
            position: World position to snap

        Returns:
            SnapResult with snapped position
        """
        if not self._settings.enabled:
            return SnapResult(snapped=False, position=position)

        settings = self._settings

        # Transform to grid space if needed
        if settings.grid_type == GridType.LOCAL:
            # Would apply inverse rotation here
            pass
        elif settings.grid_type == GridType.CUSTOM:
            position = position - settings.origin

        # Calculate subdivision size
        sub_size_x = settings.size_x / max(1, settings.subdivisions)
        sub_size_y = settings.size_y / max(1, settings.subdivisions)
        sub_size_z = settings.size_z / max(1, settings.subdivisions)

        # Snap each axis
        if settings.snap_to_integer:
            snapped_x = round(position.x)
            snapped_y = round(position.y)
            snapped_z = round(position.z)
        else:
            snapped_x = round(position.x / sub_size_x) * sub_size_x
            snapped_y = round(position.y / sub_size_y) * sub_size_y
            snapped_z = round(position.z / sub_size_z) * sub_size_z

        snapped = Vector3(snapped_x, snapped_y, snapped_z)

        # Transform back from grid space
        if settings.grid_type == GridType.CUSTOM:
            snapped = snapped + settings.origin

        distance = (snapped - position).length()

        return SnapResult(
            snapped=True,
            position=snapped,
            snap_type=SnapMode.GRID,
            distance=distance,
        )

    def get_grid_lines(
        self,
        center: Vector3,
        extent: float
    ) -> list[tuple[Vector3, Vector3]]:
        """Get grid lines for visualization."""
        lines = []
        settings = self._settings
        size = settings.size_x  # Assume uniform grid for simplicity

        half_extent = extent / 2
        start = -half_extent
        end = half_extent

        # Snap start/end to grid
        start = math.floor(start / size) * size
        end = math.ceil(end / size) * size

        # X-axis lines
        x = start
        while x <= end:
            lines.append((
                Vector3(center.x + x, center.y, center.z + start),
                Vector3(center.x + x, center.y, center.z + end),
            ))
            x += size

        # Z-axis lines
        z = start
        while z <= end:
            lines.append((
                Vector3(center.x + start, center.y, center.z + z),
                Vector3(center.x + end, center.y, center.z + z),
            ))
            z += size

        return lines


@editor
class SurfaceSnap:
    """Surface-based snapping with normal alignment."""

    __slots__ = ("_settings", "_scene")

    def __init__(
        self,
        settings: Optional[SurfaceSnapSettings] = None,
        scene: Optional[SceneProvider] = None
    ):
        self._settings = settings or SurfaceSnapSettings()
        self._scene = scene

    @property
    def settings(self) -> SurfaceSnapSettings:
        return self._settings

    @settings.setter
    def settings(self, value: SurfaceSnapSettings) -> None:
        self._settings = value

    def set_scene(self, scene: SceneProvider) -> None:
        """Set the scene provider for raycasting."""
        self._scene = scene

    def snap(
        self,
        position: Vector3,
        ray_origin: Vector3,
        ray_direction: Vector3
    ) -> SnapResult:
        """
        Snap to surface via raycast.

        Args:
            position: Current position (fallback)
            ray_origin: Origin of snap ray
            ray_direction: Direction of snap ray

        Returns:
            SnapResult with surface hit information
        """
        if not self._settings.enabled or not self._scene:
            return SnapResult(snapped=False, position=position)

        settings = self._settings

        hit = self._scene.raycast(
            ray_origin,
            ray_direction,
            settings.max_distance,
            settings.layer_mask,
        )

        if hit is None:
            return SnapResult(snapped=False, position=position)

        object_id, hit_point, hit_normal, distance = hit

        # Apply normal offset
        final_position = hit_point + hit_normal * settings.normal_offset

        # Calculate rotation to align with normal
        rotation = None
        if settings.align_to_normal:
            up = Vector3(0, 1, 0)
            if abs(hit_normal.dot(up)) < 0.999:
                axis = up.cross(hit_normal)
                angle = math.acos(max(-1, min(1, up.dot(hit_normal))))
                rotation = Quaternion.from_axis_angle(axis, angle)
            else:
                rotation = Quaternion.identity()

        return SnapResult(
            snapped=True,
            position=final_position,
            normal=hit_normal,
            rotation=rotation,
            snap_type=SnapMode.SURFACE,
            snap_target_id=object_id,
            distance=distance,
        )


@editor
class VertexSnap:
    """Vertex-based snapping to mesh vertices."""

    __slots__ = ("_settings", "_mesh_provider", "_scene", "_cached_vertices")

    def __init__(
        self,
        settings: Optional[VertexSnapSettings] = None,
        mesh_provider: Optional[MeshProvider] = None,
        scene: Optional[SceneProvider] = None
    ):
        self._settings = settings or VertexSnapSettings()
        self._mesh_provider = mesh_provider
        self._scene = scene
        self._cached_vertices: list[VertexInfo] = []

    @property
    def settings(self) -> VertexSnapSettings:
        return self._settings

    @settings.setter
    def settings(self, value: VertexSnapSettings) -> None:
        self._settings = value

    def set_providers(
        self,
        mesh_provider: MeshProvider,
        scene: SceneProvider
    ) -> None:
        """Set the data providers."""
        self._mesh_provider = mesh_provider
        self._scene = scene

    def update_cache(self) -> None:
        """Update the vertex cache from scene objects."""
        self._cached_vertices.clear()

        if not self._mesh_provider or not self._scene:
            return

        object_ids = self._scene.get_object_ids(self._settings.layer_mask)
        for obj_id in object_ids:
            vertices = self._mesh_provider.get_vertices(obj_id)
            self._cached_vertices.extend(vertices)

    def snap(self, position: Vector3) -> SnapResult:
        """
        Snap to nearest vertex within radius.

        Args:
            position: Position to snap from

        Returns:
            SnapResult with vertex position
        """
        if not self._settings.enabled:
            return SnapResult(snapped=False, position=position)

        nearest: Optional[VertexInfo] = None
        nearest_dist = float('inf')

        for vertex in self._cached_vertices:
            dist = (vertex.position - position).length()
            if dist < self._settings.snap_radius and dist < nearest_dist:
                nearest = vertex
                nearest_dist = dist

        if nearest is None:
            return SnapResult(snapped=False, position=position)

        return SnapResult(
            snapped=True,
            position=nearest.position,
            normal=nearest.normal,
            snap_type=SnapMode.VERTEX,
            snap_target_id=nearest.object_id,
            distance=nearest_dist,
        )

    def get_snap_candidates(
        self,
        position: Vector3,
        radius: Optional[float] = None
    ) -> list[VertexInfo]:
        """Get all vertices within radius for visualization."""
        search_radius = radius or self._settings.snap_radius
        candidates = []

        for vertex in self._cached_vertices:
            dist = (vertex.position - position).length()
            if dist < search_radius:
                candidates.append(vertex)

        return candidates


@editor
class EdgeSnap:
    """Edge-based snapping to mesh edges."""

    __slots__ = ("_settings", "_mesh_provider", "_scene", "_cached_edges")

    def __init__(
        self,
        settings: Optional[EdgeSnapSettings] = None,
        mesh_provider: Optional[MeshProvider] = None,
        scene: Optional[SceneProvider] = None
    ):
        self._settings = settings or EdgeSnapSettings()
        self._mesh_provider = mesh_provider
        self._scene = scene
        self._cached_edges: list[EdgeInfo] = []

    @property
    def settings(self) -> EdgeSnapSettings:
        return self._settings

    @settings.setter
    def settings(self, value: EdgeSnapSettings) -> None:
        self._settings = value

    def set_providers(
        self,
        mesh_provider: MeshProvider,
        scene: SceneProvider
    ) -> None:
        """Set the data providers."""
        self._mesh_provider = mesh_provider
        self._scene = scene

    def update_cache(self) -> None:
        """Update the edge cache from scene objects."""
        self._cached_edges.clear()

        if not self._mesh_provider or not self._scene:
            return

        object_ids = self._scene.get_object_ids(self._settings.layer_mask)
        for obj_id in object_ids:
            edges = self._mesh_provider.get_edges(obj_id)
            self._cached_edges.extend(edges)

    def _point_to_line_distance(
        self,
        point: Vector3,
        line_start: Vector3,
        line_end: Vector3
    ) -> tuple[float, Vector3]:
        """Calculate distance from point to line segment and closest point."""
        line_vec = line_end - line_start
        line_len = line_vec.length()

        if line_len < 0.0001:
            return (point - line_start).length(), line_start

        line_unit = line_vec * (1.0 / line_len)
        point_vec = point - line_start
        proj_length = point_vec.dot(line_unit)

        # Clamp to line segment
        proj_length = max(0, min(line_len, proj_length))

        closest = line_start + line_unit * proj_length
        distance = (point - closest).length()

        return distance, closest

    def snap(self, position: Vector3) -> SnapResult:
        """
        Snap to nearest edge within radius.

        Args:
            position: Position to snap from

        Returns:
            SnapResult with edge position
        """
        if not self._settings.enabled:
            return SnapResult(snapped=False, position=position)

        settings = self._settings
        nearest_edge: Optional[EdgeInfo] = None
        nearest_point = position
        nearest_dist = float('inf')

        for edge in self._cached_edges:
            dist, closest = self._point_to_line_distance(
                position, edge.start, edge.end
            )

            if dist < settings.snap_radius and dist < nearest_dist:
                nearest_edge = edge
                nearest_point = closest
                nearest_dist = dist

        if nearest_edge is None:
            return SnapResult(snapped=False, position=position)

        # Check if we should snap to midpoint
        if settings.snap_to_midpoint:
            midpoint = nearest_edge.midpoint
            mid_dist = (position - midpoint).length()
            if mid_dist < settings.snap_radius * 0.5:
                nearest_point = midpoint

        return SnapResult(
            snapped=True,
            position=nearest_point,
            snap_type=SnapMode.EDGE,
            snap_target_id=nearest_edge.object_id,
            distance=nearest_dist,
        )

    def get_snap_candidates(
        self,
        position: Vector3,
        radius: Optional[float] = None
    ) -> list[EdgeInfo]:
        """Get all edges within radius for visualization."""
        search_radius = radius or self._settings.snap_radius
        candidates = []

        for edge in self._cached_edges:
            dist, _ = self._point_to_line_distance(
                position, edge.start, edge.end
            )
            if dist < search_radius:
                candidates.append(edge)

        return candidates


@editor
class PivotSnap:
    """Pivot-based snapping to object centers and pivots."""

    __slots__ = ("_settings", "_scene", "_cached_pivots")

    def __init__(
        self,
        settings: Optional[PivotSnapSettings] = None,
        scene: Optional[SceneProvider] = None
    ):
        self._settings = settings or PivotSnapSettings()
        self._scene = scene
        self._cached_pivots: list[PivotInfo] = []

    @property
    def settings(self) -> PivotSnapSettings:
        return self._settings

    @settings.setter
    def settings(self, value: PivotSnapSettings) -> None:
        self._settings = value

    def set_scene(self, scene: SceneProvider) -> None:
        """Set the scene provider."""
        self._scene = scene

    def update_cache(self) -> None:
        """Update the pivot cache from scene objects."""
        self._cached_pivots.clear()

        if not self._scene:
            return

        object_ids = self._scene.get_object_ids()
        for obj_id in object_ids:
            transform = self._scene.get_object_transform(obj_id)
            self._cached_pivots.append(PivotInfo(
                position=transform.position,
                object_id=obj_id,
                is_bounds_center=False,
            ))

            if self._settings.include_bounds_centers:
                bounds_min, bounds_max = self._scene.get_object_bounds(obj_id)
                center = Vector3(
                    (bounds_min.x + bounds_max.x) / 2,
                    (bounds_min.y + bounds_max.y) / 2,
                    (bounds_min.z + bounds_max.z) / 2,
                )
                if (center - transform.position).length() > 0.01:
                    self._cached_pivots.append(PivotInfo(
                        position=center,
                        object_id=obj_id,
                        is_bounds_center=True,
                    ))

    def snap(self, position: Vector3) -> SnapResult:
        """
        Snap to nearest pivot within radius.

        Args:
            position: Position to snap from

        Returns:
            SnapResult with pivot position
        """
        if not self._settings.enabled:
            return SnapResult(snapped=False, position=position)

        nearest: Optional[PivotInfo] = None
        nearest_dist = float('inf')

        for pivot in self._cached_pivots:
            dist = (pivot.position - position).length()
            if dist < self._settings.snap_radius and dist < nearest_dist:
                nearest = pivot
                nearest_dist = dist

        if nearest is None:
            return SnapResult(snapped=False, position=position)

        return SnapResult(
            snapped=True,
            position=nearest.position,
            snap_type=SnapMode.PIVOT,
            snap_target_id=nearest.object_id,
            distance=nearest_dist,
        )

    def get_snap_candidates(
        self,
        position: Vector3,
        radius: Optional[float] = None
    ) -> list[PivotInfo]:
        """Get all pivots within radius for visualization."""
        search_radius = radius or self._settings.snap_radius
        candidates = []

        for pivot in self._cached_pivots:
            dist = (pivot.position - position).length()
            if dist < search_radius:
                candidates.append(pivot)

        return candidates


# =============================================================================
# Snap Manager
# =============================================================================

@editor
class SnapManager:
    """
    Central manager for all snapping operations.

    Coordinates multiple snap types and resolves conflicts based on priority.
    """

    __slots__ = (
        "_settings",
        "_grid_snap",
        "_surface_snap",
        "_vertex_snap",
        "_edge_snap",
        "_pivot_snap",
        "_last_result",
        "__weakref__",
    )

    def __init__(self):
        """Initialize snap manager with default settings."""
        self._settings = SnapSettings()
        self._grid_snap = GridSnap()
        self._surface_snap = SurfaceSnap()
        self._vertex_snap = VertexSnap()
        self._edge_snap = EdgeSnap()
        self._pivot_snap = PivotSnap()
        self._last_result: Optional[SnapResult] = None

    @property
    def settings(self) -> SnapSettings:
        return self._settings

    @settings.setter
    def settings(self, value: SnapSettings) -> None:
        self._settings = value

    @property
    def grid(self) -> GridSnap:
        return self._grid_snap

    @property
    def surface(self) -> SurfaceSnap:
        return self._surface_snap

    @property
    def vertex(self) -> VertexSnap:
        return self._vertex_snap

    @property
    def edge(self) -> EdgeSnap:
        return self._edge_snap

    @property
    def pivot(self) -> PivotSnap:
        return self._pivot_snap

    @property
    def last_result(self) -> Optional[SnapResult]:
        return self._last_result

    def set_providers(
        self,
        scene: SceneProvider,
        mesh_provider: Optional[MeshProvider] = None
    ) -> None:
        """Set providers for all snap types."""
        self._surface_snap.set_scene(scene)
        self._pivot_snap.set_scene(scene)
        if mesh_provider:
            self._vertex_snap.set_providers(mesh_provider, scene)
            self._edge_snap.set_providers(mesh_provider, scene)

    def update_caches(self) -> None:
        """Update all snap caches."""
        self._vertex_snap.update_cache()
        self._edge_snap.update_cache()
        self._pivot_snap.update_cache()

    @track_changes
    def snap(
        self,
        position: Vector3,
        ray_origin: Optional[Vector3] = None,
        ray_direction: Optional[Vector3] = None
    ) -> SnapResult:
        """
        Perform snap operation based on current settings.

        Args:
            position: Position to snap
            ray_origin: Optional ray origin for surface snap
            ray_direction: Optional ray direction for surface snap

        Returns:
            SnapResult with best snap position
        """
        if not self._settings.enabled:
            result = SnapResult(snapped=False, position=position)
            self._last_result = result
            return result

        candidates: list[SnapResult] = []
        mode = self._settings.mode

        # Gather snap results from enabled modes
        if SnapMode.VERTEX in mode and self._vertex_snap.settings.enabled:
            result = self._vertex_snap.snap(position)
            if result.snapped:
                candidates.append(result)

        if SnapMode.EDGE in mode and self._edge_snap.settings.enabled:
            result = self._edge_snap.snap(position)
            if result.snapped:
                candidates.append(result)

        if SnapMode.PIVOT in mode and self._pivot_snap.settings.enabled:
            result = self._pivot_snap.snap(position)
            if result.snapped:
                candidates.append(result)

        if SnapMode.SURFACE in mode and self._surface_snap.settings.enabled:
            if ray_origin and ray_direction:
                result = self._surface_snap.snap(position, ray_origin, ray_direction)
                if result.snapped:
                    candidates.append(result)

        if SnapMode.GRID in mode and self._grid_snap.settings.enabled:
            result = self._grid_snap.snap(position)
            if result.snapped:
                candidates.append(result)

        # Resolve based on priority
        final_result = self._resolve_candidates(candidates, position)
        self._last_result = final_result
        return final_result

    def _resolve_candidates(
        self,
        candidates: list[SnapResult],
        original_position: Vector3
    ) -> SnapResult:
        """Resolve multiple snap candidates based on priority."""
        if not candidates:
            return SnapResult(snapped=False, position=original_position)

        if len(candidates) == 1:
            return candidates[0]

        priority = self._settings.priority

        if priority == SnapPriority.NEAREST:
            return min(candidates, key=lambda r: r.distance)

        # Build priority order
        order: list[SnapMode]
        if priority == SnapPriority.VERTEX_FIRST:
            order = [SnapMode.VERTEX, SnapMode.EDGE, SnapMode.SURFACE, SnapMode.PIVOT, SnapMode.GRID]
        elif priority == SnapPriority.SURFACE_FIRST:
            order = [SnapMode.SURFACE, SnapMode.VERTEX, SnapMode.EDGE, SnapMode.PIVOT, SnapMode.GRID]
        elif priority == SnapPriority.GRID_FIRST:
            order = [SnapMode.GRID, SnapMode.SURFACE, SnapMode.VERTEX, SnapMode.EDGE, SnapMode.PIVOT]
        else:
            order = [SnapMode.VERTEX, SnapMode.EDGE, SnapMode.SURFACE, SnapMode.PIVOT, SnapMode.GRID]

        for snap_type in order:
            for candidate in candidates:
                if candidate.snap_type == snap_type:
                    return candidate

        return candidates[0]

    def toggle_mode(self, mode: SnapMode) -> None:
        """Toggle a snap mode on or off."""
        if mode in self._settings.mode:
            self._settings.mode = self._settings.mode & ~mode
        else:
            self._settings.mode = self._settings.mode | mode

    def set_grid_size(self, size: float) -> None:
        """Convenience method to set uniform grid size."""
        self._grid_snap.settings.size_x = size
        self._grid_snap.settings.size_y = size
        self._grid_snap.settings.size_z = size


__all__ = [
    "SnapMode",
    "SnapSettings",
    "SnapResult",
    "GridSnap",
    "GridSnapSettings",
    "SurfaceSnap",
    "SurfaceSnapSettings",
    "VertexSnap",
    "VertexSnapSettings",
    "VertexInfo",
    "EdgeSnap",
    "EdgeSnapSettings",
    "EdgeInfo",
    "PivotSnap",
    "PivotSnapSettings",
    "PivotInfo",
    "SnapManager",
    "SnapPriority",
    "GridType",
]
