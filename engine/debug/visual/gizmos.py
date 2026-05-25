"""
Editor gizmos for interactive debug visualization.

Provides gizmos for transform manipulation (translate, rotate, scale),
bounding box visualization, and light source visualization. Gizmos
can be used for both display and interactive manipulation.

Usage:
    from engine.debug.visual import TransformGizmo, BoundsGizmo, LightGizmo, GizmoType

    # Create transform gizmo
    gizmo = TransformGizmo()
    gizmo.set_mode(GizmoType.TRANSLATE)
    gizmo.set_target(entity_transform)

    # Render gizmo
    gizmo.render(camera)

    # Handle interaction
    if gizmo.handle_input(mouse_pos, mouse_down):
        entity.transform = gizmo.get_result()
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .config import GIZMO_CONFIG
from .draw import Color, DebugDraw, Vec3, Quat

# Callback type for gizmo value changes
GizmoChangeCallback = Callable[[Any], None]


class GizmoType(Enum):
    """Types of gizmo operations."""
    TRANSLATE = auto()
    ROTATE = auto()
    SCALE = auto()
    UNIVERSAL = auto()  # Combined translate/rotate/scale


class GizmoSpace(Enum):
    """Coordinate space for gizmo operations."""
    WORLD = auto()
    LOCAL = auto()
    VIEW = auto()


class GizmoAxis(Enum):
    """Axis selection for gizmo operations."""
    NONE = auto()
    X = auto()
    Y = auto()
    Z = auto()
    XY = auto()
    XZ = auto()
    YZ = auto()
    XYZ = auto()  # Uniform scaling or free transform
    VIEW = auto()  # View-aligned plane


@dataclass(slots=True)
class GizmoStyle:
    """
    Visual style configuration for gizmos.

    Attributes:
        x_color: Color for X axis (default: red)
        y_color: Color for Y axis (default: green)
        z_color: Color for Z axis (default: blue)
        highlight_color: Color when axis is hovered/selected
        inactive_opacity: Opacity when gizmo is not active
        axis_thickness: Thickness of axis lines
        handle_size: Size of axis handles
        plane_opacity: Opacity of plane handles
    """
    x_color: Color = field(default_factory=lambda: Color.RED)
    y_color: Color = field(default_factory=lambda: Color.GREEN)
    z_color: Color = field(default_factory=lambda: Color.BLUE)
    highlight_color: Color = field(default_factory=lambda: Color.YELLOW)
    inactive_opacity: float = 0.5
    axis_thickness: float = 2.0
    handle_size: float = 0.15
    plane_opacity: float = 0.25


@dataclass(slots=True)
class GizmoState:
    """
    Current state of a gizmo.

    Attributes:
        active: Whether the gizmo is currently being manipulated
        hovered_axis: Which axis is currently hovered
        dragging: Whether the user is dragging
        start_position: Position when drag started
        current_position: Current manipulation position
        delta: Change since last update
    """
    active: bool = False
    hovered_axis: GizmoAxis = GizmoAxis.NONE
    dragging: bool = False
    start_position: Vec3 = (0.0, 0.0, 0.0)
    current_position: Vec3 = (0.0, 0.0, 0.0)
    delta: Vec3 = (0.0, 0.0, 0.0)


def _vec3_add(a: Vec3, b: Vec3) -> Vec3:
    """Add two Vec3 tuples."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    """Subtract two Vec3 tuples."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec3_scale(v: Vec3, s: float) -> Vec3:
    """Scale a Vec3 by a scalar."""
    return (v[0] * s, v[1] * s, v[2] * s)


def _vec3_length(v: Vec3) -> float:
    """Compute length of a Vec3."""
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def _vec3_normalize(v: Vec3) -> Vec3:
    """Normalize a Vec3 to unit length."""
    length = _vec3_length(v)
    if length < GIZMO_CONFIG.vector_normalize_epsilon:
        return (0.0, 1.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _vec3_dot(a: Vec3, b: Vec3) -> float:
    """Compute dot product of two Vec3 tuples."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec3_cross(a: Vec3, b: Vec3) -> Vec3:
    """Compute cross product of two Vec3 tuples."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )


class BaseGizmo(ABC):
    """
    Abstract base class for all gizmos.

    Provides common functionality for gizmo rendering, hit testing,
    and interaction handling.
    """

    def __init__(
        self,
        style: Optional[GizmoStyle] = None,
        enabled: bool = True
    ) -> None:
        """
        Initialize the base gizmo.

        Args:
            style: Visual style configuration
            enabled: Whether the gizmo is enabled
        """
        self._style = style or GizmoStyle()
        self._enabled = enabled
        self._visible = True
        self._state = GizmoState()
        self._callbacks: List[GizmoChangeCallback] = []

    @property
    def enabled(self) -> bool:
        """Return whether the gizmo is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set whether the gizmo is enabled."""
        self._enabled = value
        if not value:
            self._state = GizmoState()

    @property
    def visible(self) -> bool:
        """Return whether the gizmo is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set whether the gizmo is visible."""
        self._visible = value

    @property
    def style(self) -> GizmoStyle:
        """Return the gizmo style."""
        return self._style

    @style.setter
    def style(self, value: GizmoStyle) -> None:
        """Set the gizmo style."""
        self._style = value

    @property
    def state(self) -> GizmoState:
        """Return the current gizmo state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Return whether the gizmo is currently being manipulated."""
        return self._state.active

    @property
    def hovered_axis(self) -> GizmoAxis:
        """Return the currently hovered axis."""
        return self._state.hovered_axis

    def register_callback(self, callback: GizmoChangeCallback) -> None:
        """
        Register a callback for value changes.

        Args:
            callback: Callback function called with new value
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: GizmoChangeCallback) -> bool:
        """
        Unregister a change callback.

        Args:
            callback: Callback to remove

        Returns:
            True if callback was found and removed
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def _notify_callbacks(self, value: Any) -> None:
        """Notify all registered callbacks of a value change."""
        for callback in self._callbacks:
            try:
                callback(value)
            except Exception as e:
                print(f"Error in gizmo callback: {e}")

    def _get_axis_color(self, axis: GizmoAxis) -> Color:
        """Get the color for an axis, considering hover state."""
        if self._state.hovered_axis == axis:
            return self._style.highlight_color

        if axis in (GizmoAxis.X, GizmoAxis.XY, GizmoAxis.XZ):
            color = self._style.x_color
        elif axis in (GizmoAxis.Y, GizmoAxis.YZ):
            color = self._style.y_color
        elif axis == GizmoAxis.Z:
            color = self._style.z_color
        else:
            color = Color.WHITE

        if not self._state.active:
            return color.with_alpha(color.a * self._style.inactive_opacity)
        return color

    @abstractmethod
    def render(self, camera: Any) -> None:
        """
        Render the gizmo.

        Args:
            camera: Camera for view-dependent calculations
        """
        pass

    @abstractmethod
    def hit_test(self, ray_origin: Vec3, ray_direction: Vec3) -> GizmoAxis:
        """
        Test if a ray hits the gizmo.

        Args:
            ray_origin: Ray origin in world space
            ray_direction: Ray direction (normalized)

        Returns:
            The axis that was hit, or GizmoAxis.NONE
        """
        pass

    @abstractmethod
    def begin_drag(self, axis: GizmoAxis, ray_origin: Vec3, ray_direction: Vec3) -> None:
        """
        Begin a drag operation.

        Args:
            axis: The axis to constrain to
            ray_origin: Initial ray origin
            ray_direction: Initial ray direction
        """
        pass

    @abstractmethod
    def update_drag(self, ray_origin: Vec3, ray_direction: Vec3) -> None:
        """
        Update the drag operation.

        Args:
            ray_origin: Current ray origin
            ray_direction: Current ray direction
        """
        pass

    @abstractmethod
    def end_drag(self) -> None:
        """End the current drag operation."""
        pass


class TransformGizmo(BaseGizmo):
    """
    Gizmo for interactive transform manipulation.

    Supports translation, rotation, and scaling operations
    with axis constraints and snapping.
    """

    def __init__(
        self,
        mode: GizmoType = GizmoType.TRANSLATE,
        space: GizmoSpace = GizmoSpace.WORLD,
        style: Optional[GizmoStyle] = None,
        enabled: bool = True
    ) -> None:
        """
        Initialize the transform gizmo.

        Args:
            mode: Initial operation mode
            space: Coordinate space for operations
            style: Visual style configuration
            enabled: Whether the gizmo is enabled
        """
        super().__init__(style, enabled)
        self._mode = mode
        self._space = space
        self._position: Vec3 = (0.0, 0.0, 0.0)
        self._rotation: Quat = (0.0, 0.0, 0.0, 1.0)
        self._scale: Vec3 = (1.0, 1.0, 1.0)
        self._size: float = 1.0
        self._snap_translation: float = 0.0
        self._snap_rotation: float = 0.0
        self._snap_scale: float = 0.0
        self._drag_start_value: Union[Vec3, Quat] = (0.0, 0.0, 0.0)
        self._drag_plane_normal: Vec3 = (0.0, 1.0, 0.0)
        self._drag_plane_origin: Vec3 = (0.0, 0.0, 0.0)

    @property
    def mode(self) -> GizmoType:
        """Return the current operation mode."""
        return self._mode

    @mode.setter
    def mode(self, value: GizmoType) -> None:
        """Set the operation mode."""
        self._mode = value
        self._state = GizmoState()

    @property
    def space(self) -> GizmoSpace:
        """Return the coordinate space."""
        return self._space

    @space.setter
    def space(self, value: GizmoSpace) -> None:
        """Set the coordinate space."""
        self._space = value

    @property
    def position(self) -> Vec3:
        """Return the gizmo position."""
        return self._position

    @position.setter
    def position(self, value: Vec3) -> None:
        """Set the gizmo position."""
        self._position = value

    @property
    def rotation(self) -> Quat:
        """Return the gizmo rotation."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: Quat) -> None:
        """Set the gizmo rotation."""
        self._rotation = value

    @property
    def scale(self) -> Vec3:
        """Return the gizmo scale."""
        return self._scale

    @scale.setter
    def scale(self, value: Vec3) -> None:
        """Set the gizmo scale."""
        self._scale = value

    @property
    def size(self) -> float:
        """Return the visual size of the gizmo."""
        return self._size

    @size.setter
    def size(self, value: float) -> None:
        """Set the visual size of the gizmo."""
        if value <= 0:
            raise ValueError(f"Size must be positive, got {value}")
        self._size = value

    def set_snap(
        self,
        translation: float = 0.0,
        rotation: float = 0.0,
        scale: float = 0.0
    ) -> None:
        """
        Set snap values for operations.

        Args:
            translation: Translation snap increment (0 = disabled)
            rotation: Rotation snap increment in degrees (0 = disabled)
            scale: Scale snap increment (0 = disabled)
        """
        self._snap_translation = translation
        self._snap_rotation = rotation
        self._snap_scale = scale

    def set_target(
        self,
        position: Vec3,
        rotation: Optional[Quat] = None,
        scale: Optional[Vec3] = None
    ) -> None:
        """
        Set the target transform for the gizmo.

        Args:
            position: Target position
            rotation: Target rotation (optional)
            scale: Target scale (optional)
        """
        self._position = position
        if rotation is not None:
            self._rotation = rotation
        if scale is not None:
            self._scale = scale

    def get_result(self) -> Tuple[Vec3, Quat, Vec3]:
        """
        Get the current transform result.

        Returns:
            Tuple of (position, rotation, scale)
        """
        return (self._position, self._rotation, self._scale)

    def render(self, camera: Any) -> None:
        """
        Render the transform gizmo.

        Args:
            camera: Camera for view-dependent calculations
        """
        if not self._enabled or not self._visible:
            return

        if self._mode == GizmoType.TRANSLATE:
            self._render_translate_gizmo()
        elif self._mode == GizmoType.ROTATE:
            self._render_rotate_gizmo()
        elif self._mode == GizmoType.SCALE:
            self._render_scale_gizmo()
        elif self._mode == GizmoType.UNIVERSAL:
            self._render_translate_gizmo()
            self._render_rotate_gizmo()

    def _render_translate_gizmo(self) -> None:
        """Render translation arrows."""
        pos = self._position
        size = self._size

        # X axis arrow
        DebugDraw.arrow(
            pos, (1.0, 0.0, 0.0), self._get_axis_color(GizmoAxis.X),
            size, thickness=self._style.axis_thickness, head_size=self._style.handle_size
        )

        # Y axis arrow
        DebugDraw.arrow(
            pos, (0.0, 1.0, 0.0), self._get_axis_color(GizmoAxis.Y),
            size, thickness=self._style.axis_thickness, head_size=self._style.handle_size
        )

        # Z axis arrow
        DebugDraw.arrow(
            pos, (0.0, 0.0, 1.0), self._get_axis_color(GizmoAxis.Z),
            size, thickness=self._style.axis_thickness, head_size=self._style.handle_size
        )

        # Plane handles (small squares)
        plane_size = size * GIZMO_CONFIG.translate_plane_size_ratio
        plane_offset = size * GIZMO_CONFIG.translate_plane_offset_ratio
        plane_thickness = GIZMO_CONFIG.translate_plane_thickness

        # XY plane
        xy_center = _vec3_add(pos, (plane_offset, plane_offset, 0.0))
        DebugDraw.box(
            xy_center,
            (plane_size * 0.5, plane_size * 0.5, plane_thickness),
            self._get_axis_color(GizmoAxis.XY).with_alpha(self._style.plane_opacity),
            wireframe=False
        )

        # XZ plane
        xz_center = _vec3_add(pos, (plane_offset, 0.0, plane_offset))
        DebugDraw.box(
            xz_center,
            (plane_size * 0.5, plane_thickness, plane_size * 0.5),
            self._get_axis_color(GizmoAxis.XZ).with_alpha(self._style.plane_opacity),
            wireframe=False
        )

        # YZ plane
        yz_center = _vec3_add(pos, (0.0, plane_offset, plane_offset))
        DebugDraw.box(
            yz_center,
            (plane_thickness, plane_size * 0.5, plane_size * 0.5),
            self._get_axis_color(GizmoAxis.YZ).with_alpha(self._style.plane_opacity),
            wireframe=False
        )

    def _render_rotate_gizmo(self) -> None:
        """Render rotation circles."""
        pos = self._position
        size = self._size

        # X rotation (around X axis - YZ plane)
        DebugDraw.circle(
            pos, (1.0, 0.0, 0.0), size,
            self._get_axis_color(GizmoAxis.X),
            thickness=self._style.axis_thickness
        )

        # Y rotation (around Y axis - XZ plane)
        DebugDraw.circle(
            pos, (0.0, 1.0, 0.0), size,
            self._get_axis_color(GizmoAxis.Y),
            thickness=self._style.axis_thickness
        )

        # Z rotation (around Z axis - XY plane)
        DebugDraw.circle(
            pos, (0.0, 0.0, 1.0), size,
            self._get_axis_color(GizmoAxis.Z),
            thickness=self._style.axis_thickness
        )

    def _render_scale_gizmo(self) -> None:
        """Render scale handles (boxes at ends)."""
        pos = self._position
        size = self._size
        box_size = size * GIZMO_CONFIG.scale_handle_box_ratio

        # X axis
        x_end = _vec3_add(pos, (size, 0.0, 0.0))
        DebugDraw.line(pos, x_end, self._get_axis_color(GizmoAxis.X),
                       thickness=self._style.axis_thickness)
        DebugDraw.box(x_end, (box_size, box_size, box_size),
                      self._get_axis_color(GizmoAxis.X), wireframe=False)

        # Y axis
        y_end = _vec3_add(pos, (0.0, size, 0.0))
        DebugDraw.line(pos, y_end, self._get_axis_color(GizmoAxis.Y),
                       thickness=self._style.axis_thickness)
        DebugDraw.box(y_end, (box_size, box_size, box_size),
                      self._get_axis_color(GizmoAxis.Y), wireframe=False)

        # Z axis
        z_end = _vec3_add(pos, (0.0, 0.0, size))
        DebugDraw.line(pos, z_end, self._get_axis_color(GizmoAxis.Z),
                       thickness=self._style.axis_thickness)
        DebugDraw.box(z_end, (box_size, box_size, box_size),
                      self._get_axis_color(GizmoAxis.Z), wireframe=False)

        # Center cube for uniform scale
        center_size = box_size * GIZMO_CONFIG.scale_center_box_multiplier
        DebugDraw.box(pos, (center_size, center_size, center_size),
                      self._get_axis_color(GizmoAxis.XYZ), wireframe=False)

    def hit_test(self, ray_origin: Vec3, ray_direction: Vec3) -> GizmoAxis:
        """
        Test if a ray hits the gizmo.

        This is a simplified hit test - production code would use
        more precise geometric intersection tests.

        Args:
            ray_origin: Ray origin in world space
            ray_direction: Ray direction (normalized)

        Returns:
            The axis that was hit, or GizmoAxis.NONE
        """
        if not self._enabled:
            return GizmoAxis.NONE

        # Calculate distance from ray to gizmo center
        to_gizmo = _vec3_sub(self._position, ray_origin)
        closest_dist = _vec3_length(to_gizmo)

        # Simple proximity-based hit test (simplified for example)
        hit_radius = self._size * GIZMO_CONFIG.hit_test_radius_ratio

        # Test each axis
        for axis, direction in [
            (GizmoAxis.X, (1.0, 0.0, 0.0)),
            (GizmoAxis.Y, (0.0, 1.0, 0.0)),
            (GizmoAxis.Z, (0.0, 0.0, 1.0)),
        ]:
            axis_end = _vec3_add(self._position, _vec3_scale(direction, self._size))
            # Simplified: check if ray is near the axis line
            # Production code would compute line-line distance
            axis_dir = _vec3_normalize(_vec3_sub(axis_end, self._position))
            proj = _vec3_dot(to_gizmo, axis_dir)
            if 0 < proj < self._size:
                # Within axis range - more detailed test needed in production
                return axis

        return GizmoAxis.NONE

    def begin_drag(self, axis: GizmoAxis, ray_origin: Vec3, ray_direction: Vec3) -> None:
        """Begin a drag operation on the specified axis."""
        self._state.active = True
        self._state.dragging = True
        self._state.hovered_axis = axis
        self._state.start_position = self._position

        if self._mode == GizmoType.TRANSLATE:
            self._drag_start_value = self._position
        elif self._mode == GizmoType.SCALE:
            self._drag_start_value = self._scale
        elif self._mode == GizmoType.ROTATE:
            self._drag_start_value = self._rotation

        # Set up drag plane based on axis
        if axis == GizmoAxis.X:
            self._drag_plane_normal = (0.0, 1.0, 0.0)
        elif axis == GizmoAxis.Y:
            self._drag_plane_normal = (0.0, 0.0, 1.0)
        elif axis == GizmoAxis.Z:
            self._drag_plane_normal = (0.0, 1.0, 0.0)
        else:
            self._drag_plane_normal = (0.0, 1.0, 0.0)

        self._drag_plane_origin = self._position

    def update_drag(self, ray_origin: Vec3, ray_direction: Vec3) -> None:
        """Update the drag operation based on current ray."""
        if not self._state.dragging:
            return

        # Simplified drag calculation
        # Production code would do proper ray-plane intersection
        axis = self._state.hovered_axis

        if self._mode == GizmoType.TRANSLATE:
            # Calculate movement along constraint axis
            if axis == GizmoAxis.X:
                constraint = (1.0, 0.0, 0.0)
            elif axis == GizmoAxis.Y:
                constraint = (0.0, 1.0, 0.0)
            elif axis == GizmoAxis.Z:
                constraint = (0.0, 0.0, 1.0)
            else:
                constraint = (1.0, 1.0, 1.0)

            # Project ray movement onto constraint
            ray_delta = _vec3_sub(ray_origin, self._state.start_position)
            movement = _vec3_dot(ray_delta, constraint)

            if self._snap_translation > 0:
                movement = round(movement / self._snap_translation) * self._snap_translation

            delta = _vec3_scale(constraint, movement)
            start_pos = self._drag_start_value
            if isinstance(start_pos, tuple) and len(start_pos) == 3:
                self._position = _vec3_add(start_pos, delta)
                self._state.delta = delta
                self._notify_callbacks(self._position)

    def end_drag(self) -> None:
        """End the current drag operation."""
        self._state.active = False
        self._state.dragging = False
        self._state.hovered_axis = GizmoAxis.NONE


class BoundsGizmo(BaseGizmo):
    """
    Gizmo for visualizing bounding boxes (AABB/OBB).

    Displays axis-aligned or oriented bounding boxes with
    optional labels and size information.
    """

    def __init__(
        self,
        style: Optional[GizmoStyle] = None,
        enabled: bool = True
    ) -> None:
        """
        Initialize the bounds gizmo.

        Args:
            style: Visual style configuration
            enabled: Whether the gizmo is enabled
        """
        super().__init__(style, enabled)
        self._center: Vec3 = (0.0, 0.0, 0.0)
        self._extent: Vec3 = (1.0, 1.0, 1.0)
        self._rotation: Optional[Quat] = None
        self._show_axes: bool = False
        self._show_size_labels: bool = False
        self._color: Color = Color.YELLOW

    @property
    def center(self) -> Vec3:
        """Return the bounds center."""
        return self._center

    @center.setter
    def center(self, value: Vec3) -> None:
        """Set the bounds center."""
        self._center = value

    @property
    def extent(self) -> Vec3:
        """Return the bounds half-extents."""
        return self._extent

    @extent.setter
    def extent(self, value: Vec3) -> None:
        """Set the bounds half-extents."""
        self._extent = value

    @property
    def rotation(self) -> Optional[Quat]:
        """Return the bounds rotation (None for AABB)."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: Optional[Quat]) -> None:
        """Set the bounds rotation (None for AABB)."""
        self._rotation = value

    @property
    def color(self) -> Color:
        """Return the bounds color."""
        return self._color

    @color.setter
    def color(self, value: Color) -> None:
        """Set the bounds color."""
        self._color = value

    @property
    def show_axes(self) -> bool:
        """Return whether local axes are shown."""
        return self._show_axes

    @show_axes.setter
    def show_axes(self, value: bool) -> None:
        """Set whether local axes are shown."""
        self._show_axes = value

    @property
    def show_size_labels(self) -> bool:
        """Return whether size labels are shown."""
        return self._show_size_labels

    @show_size_labels.setter
    def show_size_labels(self, value: bool) -> None:
        """Set whether size labels are shown."""
        self._show_size_labels = value

    def set_bounds(
        self,
        center: Vec3,
        extent: Vec3,
        rotation: Optional[Quat] = None
    ) -> None:
        """
        Set the bounds to display.

        Args:
            center: Bounds center
            extent: Bounds half-extents
            rotation: Rotation for OBB (None for AABB)
        """
        self._center = center
        self._extent = extent
        self._rotation = rotation

    def set_from_min_max(self, min_point: Vec3, max_point: Vec3) -> None:
        """
        Set bounds from min/max corners.

        Args:
            min_point: Minimum corner
            max_point: Maximum corner
        """
        self._center = (
            (min_point[0] + max_point[0]) / 2,
            (min_point[1] + max_point[1]) / 2,
            (min_point[2] + max_point[2]) / 2
        )
        self._extent = (
            (max_point[0] - min_point[0]) / 2,
            (max_point[1] - min_point[1]) / 2,
            (max_point[2] - min_point[2]) / 2
        )
        self._rotation = None

    def render(self, camera: Any) -> None:
        """Render the bounds gizmo."""
        if not self._enabled or not self._visible:
            return

        DebugDraw.box(
            self._center,
            self._extent,
            self._color,
            rotation=self._rotation,
            wireframe=True
        )

        if self._show_axes:
            axis_size = min(self._extent) * GIZMO_CONFIG.bounds_axis_size_ratio
            DebugDraw.coordinate_axes(self._center, axis_size)

        if self._show_size_labels:
            size = _vec3_scale(self._extent, 2.0)  # Full size
            label = f"Size: ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})"
            label_offset = self._extent[1] + GIZMO_CONFIG.bounds_label_offset
            label_pos = _vec3_add(self._center, (0.0, label_offset, 0.0))
            DebugDraw.world_text(label, label_pos, Color.WHITE, scale=GIZMO_CONFIG.bounds_label_scale)

    def hit_test(self, ray_origin: Vec3, ray_direction: Vec3) -> GizmoAxis:
        """Hit test is not interactive for bounds gizmo."""
        return GizmoAxis.NONE

    def begin_drag(self, axis: GizmoAxis, ray_origin: Vec3, ray_direction: Vec3) -> None:
        """Bounds gizmo does not support dragging."""
        pass

    def update_drag(self, ray_origin: Vec3, ray_direction: Vec3) -> None:
        """Bounds gizmo does not support dragging."""
        pass

    def end_drag(self) -> None:
        """Bounds gizmo does not support dragging."""
        pass


class LightGizmo(BaseGizmo):
    """
    Gizmo for visualizing light sources.

    Displays light radius (point/spot), cone angle (spot),
    and direction (directional/spot).
    """

    class LightType(Enum):
        """Types of lights that can be visualized."""
        POINT = auto()
        SPOT = auto()
        DIRECTIONAL = auto()
        AREA = auto()

    def __init__(
        self,
        light_type: LightType = LightType.POINT,
        style: Optional[GizmoStyle] = None,
        enabled: bool = True
    ) -> None:
        """
        Initialize the light gizmo.

        Args:
            light_type: Type of light to visualize
            style: Visual style configuration
            enabled: Whether the gizmo is enabled
        """
        super().__init__(style, enabled)
        self._light_type = light_type
        self._position: Vec3 = (0.0, 0.0, 0.0)
        self._direction: Vec3 = (0.0, -1.0, 0.0)
        self._radius: float = 5.0
        self._inner_angle: float = 0.5  # radians
        self._outer_angle: float = 0.8  # radians
        self._color: Color = Color.YELLOW
        self._show_attenuation: bool = True
        self._show_direction: bool = True

    @property
    def light_type(self) -> LightType:
        """Return the light type."""
        return self._light_type

    @light_type.setter
    def light_type(self, value: LightType) -> None:
        """Set the light type."""
        self._light_type = value

    @property
    def position(self) -> Vec3:
        """Return the light position."""
        return self._position

    @position.setter
    def position(self, value: Vec3) -> None:
        """Set the light position."""
        self._position = value

    @property
    def direction(self) -> Vec3:
        """Return the light direction."""
        return self._direction

    @direction.setter
    def direction(self, value: Vec3) -> None:
        """Set the light direction."""
        self._direction = _vec3_normalize(value)

    @property
    def radius(self) -> float:
        """Return the light radius."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        """Set the light radius."""
        if value <= 0:
            raise ValueError(f"Radius must be positive, got {value}")
        self._radius = value

    @property
    def inner_angle(self) -> float:
        """Return the spotlight inner cone angle (radians)."""
        return self._inner_angle

    @inner_angle.setter
    def inner_angle(self, value: float) -> None:
        """Set the spotlight inner cone angle (radians)."""
        self._inner_angle = value

    @property
    def outer_angle(self) -> float:
        """Return the spotlight outer cone angle (radians)."""
        return self._outer_angle

    @outer_angle.setter
    def outer_angle(self, value: float) -> None:
        """Set the spotlight outer cone angle (radians)."""
        self._outer_angle = value

    @property
    def color(self) -> Color:
        """Return the gizmo color."""
        return self._color

    @color.setter
    def color(self, value: Color) -> None:
        """Set the gizmo color."""
        self._color = value

    def set_point_light(self, position: Vec3, radius: float) -> None:
        """
        Configure as a point light.

        Args:
            position: Light position
            radius: Attenuation radius
        """
        self._light_type = self.LightType.POINT
        self._position = position
        self._radius = radius

    def set_spot_light(
        self,
        position: Vec3,
        direction: Vec3,
        radius: float,
        inner_angle: float,
        outer_angle: float
    ) -> None:
        """
        Configure as a spot light.

        Args:
            position: Light position
            direction: Light direction
            radius: Attenuation radius
            inner_angle: Inner cone angle (radians)
            outer_angle: Outer cone angle (radians)
        """
        self._light_type = self.LightType.SPOT
        self._position = position
        self._direction = _vec3_normalize(direction)
        self._radius = radius
        self._inner_angle = inner_angle
        self._outer_angle = outer_angle

    def set_directional_light(self, direction: Vec3) -> None:
        """
        Configure as a directional light.

        Args:
            direction: Light direction
        """
        self._light_type = self.LightType.DIRECTIONAL
        self._direction = _vec3_normalize(direction)

    def render(self, camera: Any) -> None:
        """Render the light gizmo."""
        if not self._enabled or not self._visible:
            return

        if self._light_type == self.LightType.POINT:
            self._render_point_light()
        elif self._light_type == self.LightType.SPOT:
            self._render_spot_light()
        elif self._light_type == self.LightType.DIRECTIONAL:
            self._render_directional_light()

    def _render_point_light(self) -> None:
        """Render a point light gizmo."""
        # Central sphere
        DebugDraw.sphere(
            self._position,
            GIZMO_CONFIG.light_center_sphere_radius,
            self._color,
            segments=8
        )

        if self._show_attenuation:
            # Attenuation radius sphere
            DebugDraw.sphere(
                self._position,
                self._radius,
                self._color.with_alpha(GIZMO_CONFIG.light_attenuation_alpha),
                segments=16
            )

            # Cross lines through center
            for direction in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                start = _vec3_sub(self._position, _vec3_scale(direction, self._radius))
                end = _vec3_add(self._position, _vec3_scale(direction, self._radius))
                DebugDraw.line(start, end, self._color.with_alpha(GIZMO_CONFIG.light_attenuation_line_alpha))

    def _render_spot_light(self) -> None:
        """Render a spot light gizmo."""
        # Central point
        DebugDraw.sphere(self._position, GIZMO_CONFIG.light_center_sphere_radius, self._color, segments=8)

        if self._show_direction:
            # Direction arrow
            DebugDraw.arrow(
                self._position,
                self._direction,
                self._color,
                length=self._radius * GIZMO_CONFIG.light_arrow_length_ratio,
                head_size=GIZMO_CONFIG.light_arrow_head_size
            )

        # Inner and outer cones
        inner_radius = math.tan(self._inner_angle) * self._radius
        outer_radius = math.tan(self._outer_angle) * self._radius

        cone_end = _vec3_add(
            self._position,
            _vec3_scale(self._direction, self._radius)
        )

        # Inner cone
        DebugDraw.cone(
            self._position,
            self._direction,
            self._radius,
            self._inner_angle,
            self._color,
            segments=16
        )

        # Outer cone (semi-transparent)
        DebugDraw.cone(
            self._position,
            self._direction,
            self._radius,
            self._outer_angle,
            self._color.with_alpha(GIZMO_CONFIG.light_attenuation_alpha),
            segments=16
        )

    def _render_directional_light(self) -> None:
        """Render a directional light gizmo."""
        # Draw parallel arrows indicating direction
        arrow_length = GIZMO_CONFIG.directional_arrow_length
        arrow_spacing = GIZMO_CONFIG.directional_arrow_spacing
        arrow_head_size = GIZMO_CONFIG.directional_arrow_head_size

        # Find perpendicular vectors
        up = (0.0, 1.0, 0.0) if abs(self._direction[1]) < 0.9 else (1.0, 0.0, 0.0)
        right = _vec3_normalize(_vec3_cross(self._direction, up))
        actual_up = _vec3_cross(right, self._direction)

        # Draw grid of arrows
        for i in range(-1, 2):
            for j in range(-1, 2):
                offset = _vec3_add(
                    _vec3_scale(right, i * arrow_spacing),
                    _vec3_scale(actual_up, j * arrow_spacing)
                )
                arrow_pos = _vec3_add(self._position, offset)
                DebugDraw.arrow(
                    arrow_pos,
                    self._direction,
                    self._color,
                    length=arrow_length,
                    head_size=arrow_head_size
                )

    def hit_test(self, ray_origin: Vec3, ray_direction: Vec3) -> GizmoAxis:
        """Hit test is not interactive for light gizmo."""
        return GizmoAxis.NONE

    def begin_drag(self, axis: GizmoAxis, ray_origin: Vec3, ray_direction: Vec3) -> None:
        """Light gizmo does not support dragging."""
        pass

    def update_drag(self, ray_origin: Vec3, ray_direction: Vec3) -> None:
        """Light gizmo does not support dragging."""
        pass

    def end_drag(self) -> None:
        """Light gizmo does not support dragging."""
        pass


class CameraGizmo(BaseGizmo):
    """
    Gizmo for visualizing camera frustums.

    Displays the camera's view frustum with near/far planes
    and optional field of view indicators.
    """

    def __init__(
        self,
        style: Optional[GizmoStyle] = None,
        enabled: bool = True
    ) -> None:
        """
        Initialize the camera gizmo.

        Args:
            style: Visual style configuration
            enabled: Whether the gizmo is enabled
        """
        super().__init__(style, enabled)
        self._position: Vec3 = (0.0, 0.0, 0.0)
        self._direction: Vec3 = (0.0, 0.0, -1.0)
        self._up: Vec3 = (0.0, 1.0, 0.0)
        self._fov_y: float = math.radians(60)
        self._aspect: float = GIZMO_CONFIG.camera_default_aspect_ratio
        self._near: float = 0.1
        self._far: float = 100.0
        self._color: Color = Color.CYAN

    def set_camera(
        self,
        position: Vec3,
        direction: Vec3,
        up: Vec3,
        fov_y: float,
        aspect: float,
        near: float,
        far: float
    ) -> None:
        """
        Set camera parameters.

        Args:
            position: Camera position
            direction: View direction
            up: Up vector
            fov_y: Vertical field of view (radians)
            aspect: Aspect ratio
            near: Near plane distance
            far: Far plane distance
        """
        self._position = position
        self._direction = _vec3_normalize(direction)
        self._up = _vec3_normalize(up)
        self._fov_y = fov_y
        self._aspect = aspect
        self._near = near
        self._far = far

    def render(self, camera: Any) -> None:
        """Render the camera frustum gizmo."""
        if not self._enabled or not self._visible:
            return

        DebugDraw.frustum(
            self._position,
            self._direction,
            self._up,
            self._fov_y,
            self._aspect,
            self._near,
            self._far,
            self._color
        )

        # Camera icon at position
        DebugDraw.sphere(self._position, GIZMO_CONFIG.camera_icon_sphere_radius, self._color, segments=8)

    def hit_test(self, ray_origin: Vec3, ray_direction: Vec3) -> GizmoAxis:
        """Camera gizmo does not support hit testing."""
        return GizmoAxis.NONE

    def begin_drag(self, axis: GizmoAxis, ray_origin: Vec3, ray_direction: Vec3) -> None:
        """Camera gizmo does not support dragging."""
        pass

    def update_drag(self, ray_origin: Vec3, ray_direction: Vec3) -> None:
        """Camera gizmo does not support dragging."""
        pass

    def end_drag(self) -> None:
        """Camera gizmo does not support dragging."""
        pass


# Module-level exports
__all__ = [
    'GizmoType',
    'GizmoSpace',
    'GizmoAxis',
    'GizmoStyle',
    'GizmoState',
    'GizmoChangeCallback',
    'BaseGizmo',
    'TransformGizmo',
    'BoundsGizmo',
    'LightGizmo',
    'CameraGizmo',
]
