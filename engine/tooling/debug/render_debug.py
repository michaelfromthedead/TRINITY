"""
Render Debug - Wireframe, bounding boxes, LOD visualization, and overdraw heatmap.

Provides rendering debug tools for analyzing visual performance and correctness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, ClassVar, Optional, Any
import threading
import time


class WireframeMode(Enum):
    """Wireframe rendering modes."""
    OFF = auto()
    OVERLAY = auto()       # Wireframe on top of solid
    ONLY = auto()          # Wireframe only
    XRAY = auto()          # See-through wireframe


class BoundingBoxType(Enum):
    """Types of bounding boxes."""
    AABB = auto()          # Axis-aligned bounding box
    OBB = auto()           # Oriented bounding box
    SPHERE = auto()        # Bounding sphere
    CAPSULE = auto()       # Bounding capsule


class LODLevel(Enum):
    """LOD levels."""
    LOD0 = 0  # Highest detail
    LOD1 = 1
    LOD2 = 2
    LOD3 = 3
    LOD4 = 4  # Lowest detail
    CULLED = -1


@dataclass(slots=True)
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(slots=True)
class Quaternion:
    """Quaternion for rotations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(0, 0, 0, 1)


@dataclass
class BoundingBox:
    """Represents a bounding box."""
    box_id: str
    box_type: BoundingBoxType
    center: Vector3 = field(default_factory=Vector3)
    extents: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    radius: float = 1.0  # For sphere type
    object_name: str = ""


@dataclass
class LODObject:
    """Represents an object with LOD information."""
    object_id: str
    position: Vector3
    current_lod: LODLevel = LODLevel.LOD0
    lod_distances: list[float] = field(default_factory=lambda: [10, 25, 50, 100])
    triangle_counts: list[int] = field(default_factory=lambda: [10000, 5000, 2000, 500, 100])
    is_culled: bool = False
    screen_size: float = 0.0  # Percentage of screen


@dataclass
class OverdrawPixel:
    """Represents overdraw data for a screen region."""
    x: int
    y: int
    overdraw_count: int
    objects_rendered: list[str] = field(default_factory=list)


class BoundingBoxDisplay:
    """Visualizes bounding boxes."""

    __slots__ = (
        '_boxes',
        '_enabled',
        '_show_aabb',
        '_show_obb',
        '_show_sphere',
        '_aabb_color',
        '_obb_color',
        '_sphere_color',
    )

    def __init__(self):
        self._boxes: dict[str, BoundingBox] = {}
        self._enabled = True
        self._show_aabb = True
        self._show_obb = True
        self._show_sphere = True
        self._aabb_color = (0.0, 1.0, 0.0, 0.5)
        self._obb_color = (1.0, 1.0, 0.0, 0.5)
        self._sphere_color = (0.0, 0.5, 1.0, 0.5)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def add_box(self, box: BoundingBox) -> None:
        """Add a bounding box."""
        self._boxes[box.box_id] = box

    def remove_box(self, box_id: str) -> Optional[BoundingBox]:
        """Remove a box."""
        return self._boxes.pop(box_id, None)

    def get_box(self, box_id: str) -> Optional[BoundingBox]:
        """Get a box by ID."""
        return self._boxes.get(box_id)

    def update_box(
        self,
        box_id: str,
        center: Optional[Vector3] = None,
        extents: Optional[Vector3] = None,
        rotation: Optional[Quaternion] = None,
    ) -> bool:
        """Update box properties."""
        box = self._boxes.get(box_id)
        if not box:
            return False

        if center is not None:
            box.center = center
        if extents is not None:
            box.extents = extents
        if rotation is not None:
            box.rotation = rotation

        return True

    def set_show_aabb(self, show: bool) -> None:
        self._show_aabb = show

    def set_show_obb(self, show: bool) -> None:
        self._show_obb = show

    def set_show_sphere(self, show: bool) -> None:
        self._show_sphere = show

    def generate_draw_commands(self) -> list[dict[str, Any]]:
        """Generate draw commands for bounding boxes."""
        if not self._enabled:
            return []

        commands = []
        for box in self._boxes.values():
            if box.box_type == BoundingBoxType.AABB and self._show_aabb:
                commands.append({
                    "type": "box",
                    "center": box.center.to_tuple(),
                    "extents": box.extents.to_tuple(),
                    "color": self._aabb_color,
                    "wireframe": True,
                })
            elif box.box_type == BoundingBoxType.OBB and self._show_obb:
                commands.append({
                    "type": "box",
                    "center": box.center.to_tuple(),
                    "extents": box.extents.to_tuple(),
                    "rotation": box.rotation.to_tuple(),
                    "color": self._obb_color,
                    "wireframe": True,
                })
            elif box.box_type == BoundingBoxType.SPHERE and self._show_sphere:
                commands.append({
                    "type": "sphere",
                    "center": box.center.to_tuple(),
                    "radius": box.radius,
                    "color": self._sphere_color,
                    "wireframe": True,
                })

        return commands

    @property
    def box_count(self) -> int:
        return len(self._boxes)

    def clear_all_boxes(self) -> None:
        """Remove all boxes."""
        self._boxes.clear()


class LODVisualization:
    """Visualizes LOD levels and transitions."""

    __slots__ = (
        '_objects',
        '_enabled',
        '_show_lod_levels',
        '_show_triangle_counts',
        '_show_screen_size',
        '_lod_colors',
        '_highlight_transitions',
    )

    def __init__(self):
        self._objects: dict[str, LODObject] = {}
        self._enabled = True
        self._show_lod_levels = True
        self._show_triangle_counts = False
        self._show_screen_size = False
        self._highlight_transitions = True

        # Colors for LOD levels
        self._lod_colors = {
            LODLevel.LOD0: (0.0, 1.0, 0.0, 1.0),
            LODLevel.LOD1: (0.5, 1.0, 0.0, 1.0),
            LODLevel.LOD2: (1.0, 1.0, 0.0, 1.0),
            LODLevel.LOD3: (1.0, 0.5, 0.0, 1.0),
            LODLevel.LOD4: (1.0, 0.0, 0.0, 1.0),
            LODLevel.CULLED: (0.5, 0.5, 0.5, 0.5),
        }

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def add_object(self, obj: LODObject) -> None:
        """Add a LOD object."""
        self._objects[obj.object_id] = obj

    def remove_object(self, object_id: str) -> Optional[LODObject]:
        """Remove an object."""
        return self._objects.pop(object_id, None)

    def get_object(self, object_id: str) -> Optional[LODObject]:
        """Get an object by ID."""
        return self._objects.get(object_id)

    def update_object(
        self,
        object_id: str,
        position: Optional[Vector3] = None,
        current_lod: Optional[LODLevel] = None,
        is_culled: Optional[bool] = None,
        screen_size: Optional[float] = None,
    ) -> bool:
        """Update object state."""
        obj = self._objects.get(object_id)
        if not obj:
            return False

        if position is not None:
            obj.position = position
        if current_lod is not None:
            obj.current_lod = current_lod
        if is_culled is not None:
            obj.is_culled = is_culled
        if screen_size is not None:
            obj.screen_size = screen_size

        return True

    def set_show_lod_levels(self, show: bool) -> None:
        self._show_lod_levels = show

    def set_show_triangle_counts(self, show: bool) -> None:
        self._show_triangle_counts = show

    def set_show_screen_size(self, show: bool) -> None:
        self._show_screen_size = show

    def get_lod_color(self, lod: LODLevel) -> tuple[float, float, float, float]:
        """Get color for a LOD level."""
        return self._lod_colors.get(lod, (1.0, 1.0, 1.0, 1.0))

    def generate_draw_commands(self) -> list[dict[str, Any]]:
        """Generate draw commands for LOD visualization."""
        if not self._enabled:
            return []

        commands = []
        for obj in self._objects.values():
            commands.extend(self._generate_object_draws(obj))

        return commands

    def _generate_object_draws(self, obj: LODObject) -> list[dict[str, Any]]:
        """Generate draw commands for a single object."""
        commands = []

        lod_level = LODLevel.CULLED if obj.is_culled else obj.current_lod
        color = self.get_lod_color(lod_level)

        # Object indicator
        commands.append({
            "type": "sphere",
            "center": obj.position.to_tuple(),
            "radius": 0.5,
            "color": color,
        })

        # LOD level text
        if self._show_lod_levels:
            text_pos = Vector3(obj.position.x, obj.position.y + 1.5, obj.position.z)
            lod_text = "CULLED" if obj.is_culled else f"LOD{obj.current_lod.value}"
            commands.append({
                "type": "text",
                "position": text_pos.to_tuple(),
                "text": lod_text,
                "color": color,
            })

        # Triangle count
        if self._show_triangle_counts and not obj.is_culled:
            lod_idx = obj.current_lod.value
            if 0 <= lod_idx < len(obj.triangle_counts):
                tri_count = obj.triangle_counts[lod_idx]
                text_pos = Vector3(obj.position.x, obj.position.y + 2.0, obj.position.z)
                commands.append({
                    "type": "text",
                    "position": text_pos.to_tuple(),
                    "text": f"{tri_count:,} tris",
                    "color": (1.0, 1.0, 1.0, 0.8),
                    "scale": 0.8,
                })

        # Screen size
        if self._show_screen_size:
            text_pos = Vector3(obj.position.x, obj.position.y + 2.5, obj.position.z)
            commands.append({
                "type": "text",
                "position": text_pos.to_tuple(),
                "text": f"{obj.screen_size:.1f}%",
                "color": (0.8, 0.8, 0.8, 0.8),
                "scale": 0.7,
            })

        return commands

    @property
    def object_count(self) -> int:
        return len(self._objects)

    def get_objects_by_lod(self, lod: LODLevel) -> list[LODObject]:
        """Get all objects at a specific LOD level."""
        return [o for o in self._objects.values() if o.current_lod == lod]

    def get_culled_objects(self) -> list[LODObject]:
        """Get all culled objects."""
        return [o for o in self._objects.values() if o.is_culled]

    def get_stats(self) -> dict[str, Any]:
        """Get LOD statistics."""
        stats = {
            "total_objects": len(self._objects),
            "culled_objects": sum(1 for o in self._objects.values() if o.is_culled),
            "total_triangles": 0,
        }

        for lod in LODLevel:
            if lod != LODLevel.CULLED:
                stats[f"lod{lod.value}_count"] = sum(
                    1 for o in self._objects.values()
                    if o.current_lod == lod and not o.is_culled
                )

        # Calculate total triangles
        for obj in self._objects.values():
            if not obj.is_culled:
                lod_idx = obj.current_lod.value
                if 0 <= lod_idx < len(obj.triangle_counts):
                    stats["total_triangles"] += obj.triangle_counts[lod_idx]

        return stats

    def clear_all_objects(self) -> None:
        """Remove all objects."""
        self._objects.clear()


class OverdrawHeatmap:
    """Visualizes overdraw as a heatmap."""

    __slots__ = (
        '_enabled',
        '_overdraw_data',
        '_width',
        '_height',
        '_max_overdraw',
        '_heatmap_colors',
        '_show_legend',
    )

    def __init__(self, width: int = 1920, height: int = 1080):
        self._enabled = True
        self._overdraw_data: list[list[int]] = []
        self._width = width
        self._height = height
        self._max_overdraw = 0
        self._show_legend = True

        # Heatmap colors (overdraw count -> color)
        self._heatmap_colors = [
            (0.0, 0.0, 0.0, 0.0),     # 0: Transparent
            (0.0, 0.0, 1.0, 0.5),     # 1: Blue
            (0.0, 1.0, 0.0, 0.5),     # 2: Green
            (1.0, 1.0, 0.0, 0.5),     # 3: Yellow
            (1.0, 0.5, 0.0, 0.5),     # 4: Orange
            (1.0, 0.0, 0.0, 0.5),     # 5+: Red
        ]

        self._init_data()

    def _init_data(self) -> None:
        """Initialize overdraw data array."""
        self._overdraw_data = [[0 for _ in range(self._width)] for _ in range(self._height)]

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def resize(self, width: int, height: int) -> None:
        """Resize the heatmap."""
        self._width = width
        self._height = height
        self._init_data()

    def record_overdraw(self, x: int, y: int, count: int = 1) -> None:
        """Record overdraw at a pixel."""
        if 0 <= x < self._width and 0 <= y < self._height:
            self._overdraw_data[y][x] += count
            self._max_overdraw = max(self._max_overdraw, self._overdraw_data[y][x])

    def set_overdraw(self, x: int, y: int, count: int) -> None:
        """Set overdraw count at a pixel."""
        if 0 <= x < self._width and 0 <= y < self._height:
            self._overdraw_data[y][x] = count
            self._max_overdraw = max(self._max_overdraw, count)

    def get_overdraw(self, x: int, y: int) -> int:
        """Get overdraw count at a pixel."""
        if 0 <= x < self._width and 0 <= y < self._height:
            return self._overdraw_data[y][x]
        return 0

    def clear(self) -> None:
        """Clear all overdraw data."""
        self._init_data()
        self._max_overdraw = 0

    def get_color_for_overdraw(self, count: int) -> tuple[float, float, float, float]:
        """Get heatmap color for overdraw count."""
        if count <= 0:
            return self._heatmap_colors[0]
        elif count >= len(self._heatmap_colors) - 1:
            return self._heatmap_colors[-1]
        else:
            return self._heatmap_colors[count]

    def set_show_legend(self, show: bool) -> None:
        self._show_legend = show

    def generate_render_data(self) -> dict[str, Any]:
        """Generate render data for the heatmap."""
        if not self._enabled:
            return {}

        return {
            "type": "overdraw_heatmap",
            "width": self._width,
            "height": self._height,
            "data": self._overdraw_data,
            "max_overdraw": self._max_overdraw,
            "colors": self._heatmap_colors,
            "show_legend": self._show_legend,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get overdraw statistics."""
        total_pixels = self._width * self._height
        overdraw_sum = sum(sum(row) for row in self._overdraw_data)
        non_zero_pixels = sum(1 for row in self._overdraw_data for val in row if val > 0)

        return {
            "total_pixels": total_pixels,
            "max_overdraw": self._max_overdraw,
            "average_overdraw": overdraw_sum / total_pixels if total_pixels > 0 else 0,
            "non_zero_pixels": non_zero_pixels,
            "coverage_percent": (non_zero_pixels / total_pixels * 100) if total_pixels > 0 else 0,
        }

    def get_overdraw_histogram(self) -> dict[int, int]:
        """Get histogram of overdraw counts."""
        histogram: dict[int, int] = {}
        for row in self._overdraw_data:
            for count in row:
                histogram[count] = histogram.get(count, 0) + 1
        return histogram

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height


class RenderDebugger:
    """Central render debugging system."""

    _instance: ClassVar[Optional["RenderDebugger"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    __slots__ = (
        '_enabled',
        '_wireframe_mode',
        '_bbox_display',
        '_lod_viz',
        '_overdraw_heatmap',
        '_show_normals',
        '_show_tangents',
        '_show_uvs',
        '_show_vertex_colors',
        '_force_lod_level',
    )

    def __init__(self):
        self._enabled = True
        self._wireframe_mode = WireframeMode.OFF
        self._bbox_display = BoundingBoxDisplay()
        self._lod_viz = LODVisualization()
        self._overdraw_heatmap = OverdrawHeatmap()
        self._show_normals = False
        self._show_tangents = False
        self._show_uvs = False
        self._show_vertex_colors = False
        self._force_lod_level: Optional[LODLevel] = None

    @classmethod
    def get_instance(cls) -> "RenderDebugger":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def wireframe_mode(self) -> WireframeMode:
        return self._wireframe_mode

    @wireframe_mode.setter
    def wireframe_mode(self, mode: WireframeMode) -> None:
        self._wireframe_mode = mode

    def cycle_wireframe_mode(self) -> WireframeMode:
        """Cycle through wireframe modes."""
        modes = list(WireframeMode)
        current_idx = modes.index(self._wireframe_mode)
        next_idx = (current_idx + 1) % len(modes)
        self._wireframe_mode = modes[next_idx]
        return self._wireframe_mode

    @property
    def bounding_box_display(self) -> BoundingBoxDisplay:
        return self._bbox_display

    @property
    def lod_visualization(self) -> LODVisualization:
        return self._lod_viz

    @property
    def overdraw_heatmap(self) -> OverdrawHeatmap:
        return self._overdraw_heatmap

    def set_show_normals(self, show: bool) -> None:
        self._show_normals = show

    def set_show_tangents(self, show: bool) -> None:
        self._show_tangents = show

    def set_show_uvs(self, show: bool) -> None:
        self._show_uvs = show

    def set_show_vertex_colors(self, show: bool) -> None:
        self._show_vertex_colors = show

    def force_lod(self, level: Optional[LODLevel]) -> None:
        """Force all objects to a specific LOD level."""
        self._force_lod_level = level

    @property
    def forced_lod_level(self) -> Optional[LODLevel]:
        return self._force_lod_level

    def generate_all_draw_commands(self) -> list[dict[str, Any]]:
        """Generate draw commands from all subsystems."""
        if not self._enabled:
            return []

        commands = []
        commands.extend(self._bbox_display.generate_draw_commands())
        commands.extend(self._lod_viz.generate_draw_commands())
        return commands

    def get_render_settings(self) -> dict[str, Any]:
        """Get current render debug settings."""
        return {
            "wireframe_mode": self._wireframe_mode.name,
            "show_normals": self._show_normals,
            "show_tangents": self._show_tangents,
            "show_uvs": self._show_uvs,
            "show_vertex_colors": self._show_vertex_colors,
            "force_lod": self._force_lod_level.name if self._force_lod_level else None,
        }

    def clear_all(self) -> None:
        """Clear all debug visualizations."""
        self._bbox_display.clear_all_boxes()
        self._lod_viz.clear_all_objects()
        self._overdraw_heatmap.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get combined stats."""
        return {
            "bounding_boxes": self._bbox_display.box_count,
            "lod_objects": self._lod_viz.object_count,
            "lod_stats": self._lod_viz.get_stats(),
            "overdraw_stats": self._overdraw_heatmap.get_stats(),
        }
