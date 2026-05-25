"""
Gizmos - Transform gizmos for translate, rotate, scale operations.

Provides:
- Translation gizmo with axis/plane handles
- Rotation gizmo with axis circles
- Scale gizmo with axis/uniform handles
- Universal gizmo combining all transform types
- Transform space modes (world, local, view, parent)
- Axis constraints for precision transforms
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Any, Callable, Optional, Tuple

from engine.tooling.editor.app_shell import editor, reloadable


class GizmoType(Enum):
    """Types of transform gizmos."""
    NONE = auto()
    TRANSLATE = auto()
    ROTATE = auto()
    SCALE = auto()
    UNIVERSAL = auto()


class GizmoSpace(Enum):
    """Transform space for gizmo operations."""
    WORLD = auto()    # World-aligned axes
    LOCAL = auto()    # Object-aligned axes
    VIEW = auto()     # View-aligned (screen space)
    PARENT = auto()   # Parent object space


class GizmoAxis(Flag):
    """Axis flags for gizmo operations."""
    NONE = 0
    X = auto()
    Y = auto()
    Z = auto()
    XY = X | Y
    XZ = X | Z
    YZ = Y | Z
    XYZ = X | Y | Z


@editor(category="Gizmos")
@reloadable()
class GizmoConstraint:
    """Constraint for gizmo operations."""
    __slots__ = ("axis", "snap_enabled", "snap_value", "limit_enabled",
                 "limit_min", "limit_max")

    def __init__(self, axis: GizmoAxis = GizmoAxis.XYZ):
        self.axis = axis
        self.snap_enabled = False
        self.snap_value = 1.0
        self.limit_enabled = False
        self.limit_min = float('-inf')
        self.limit_max = float('inf')

    def apply_snap(self, value: float) -> float:
        """Apply snap to a value."""
        if self.snap_enabled and self.snap_value > 0:
            return round(value / self.snap_value) * self.snap_value
        return value

    def apply_limit(self, value: float) -> float:
        """Apply limits to a value."""
        if self.limit_enabled:
            return max(self.limit_min, min(self.limit_max, value))
        return value

    def apply(self, value: float) -> float:
        """Apply both snap and limit."""
        value = self.apply_snap(value)
        value = self.apply_limit(value)
        return value

    def is_axis_enabled(self, axis: GizmoAxis) -> bool:
        """Check if an axis is enabled."""
        return axis in self.axis


@editor(category="Gizmos")
@reloadable()
class Gizmo:
    """Base class for transform gizmos."""
    __slots__ = ("gizmo_type", "position", "rotation", "scale", "visible",
                 "enabled", "selected_axis", "space", "constraint",
                 "size", "color_x", "color_y", "color_z", "color_selected",
                 "_dragging", "_drag_start", "_drag_axis", "on_transform_start",
                 "on_transform", "on_transform_end")

    def __init__(self, gizmo_type: GizmoType = GizmoType.NONE):
        self.gizmo_type = gizmo_type
        self.position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.visible: bool = True
        self.enabled: bool = True
        self.selected_axis: GizmoAxis = GizmoAxis.NONE
        self.space: GizmoSpace = GizmoSpace.WORLD
        self.constraint = GizmoConstraint()
        self.size: float = 1.0

        # Colors (RGBA)
        self.color_x: Tuple[float, float, float, float] = (0.9, 0.2, 0.2, 1.0)
        self.color_y: Tuple[float, float, float, float] = (0.2, 0.9, 0.2, 1.0)
        self.color_z: Tuple[float, float, float, float] = (0.2, 0.2, 0.9, 1.0)
        self.color_selected: Tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0)

        self._dragging: bool = False
        self._drag_start: Optional[Tuple[float, float]] = None
        self._drag_axis: GizmoAxis = GizmoAxis.NONE

        # Callbacks
        self.on_transform_start: Optional[Callable[[GizmoAxis], None]] = None
        self.on_transform: Optional[Callable[[Tuple[float, float, float]], None]] = None
        self.on_transform_end: Optional[Callable[[], None]] = None

    def set_position(self, x: float, y: float, z: float) -> None:
        """Set gizmo position."""
        self.position = (x, y, z)

    def set_rotation(self, pitch: float, yaw: float, roll: float) -> None:
        """Set gizmo rotation (for local space visualization)."""
        self.rotation = (pitch, yaw, roll)

    def set_space(self, space: GizmoSpace) -> None:
        """Set transform space."""
        self.space = space

    def enable_snap(self, enabled: bool, value: float = 1.0) -> None:
        """Enable/disable snapping."""
        self.constraint.snap_enabled = enabled
        self.constraint.snap_value = value

    def hit_test(self, screen_x: int, screen_y: int,
                 camera_pos: Tuple[float, float, float]) -> GizmoAxis:
        """Test if screen position hits a gizmo axis. Override in subclasses."""
        return GizmoAxis.NONE

    def begin_drag(self, axis: GizmoAxis, screen_x: int, screen_y: int) -> None:
        """Begin a drag operation on an axis."""
        if not self.enabled or not self.visible:
            return
        self._dragging = True
        self._drag_start = (screen_x, screen_y)
        self._drag_axis = axis
        self.selected_axis = axis
        if self.on_transform_start:
            self.on_transform_start(axis)

    def update_drag(self, screen_x: int, screen_y: int,
                    camera_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Update drag operation. Returns delta transform or None."""
        if not self._dragging or self._drag_start is None:
            return None

        # Calculate screen delta
        delta_x = screen_x - self._drag_start[0]
        delta_y = screen_y - self._drag_start[1]

        # Convert to transform delta (override in subclasses for specific behavior)
        delta = self._calculate_delta(delta_x, delta_y, camera_pos)

        if delta and self.on_transform:
            self.on_transform(delta)

        return delta

    def end_drag(self) -> None:
        """End the drag operation."""
        self._dragging = False
        self._drag_start = None
        self._drag_axis = GizmoAxis.NONE
        self.selected_axis = GizmoAxis.NONE
        if self.on_transform_end:
            self.on_transform_end()

    def cancel_drag(self) -> None:
        """Cancel the drag operation without applying."""
        self._dragging = False
        self._drag_start = None
        self._drag_axis = GizmoAxis.NONE
        self.selected_axis = GizmoAxis.NONE

    def _calculate_delta(self, delta_x: float, delta_y: float,
                         camera_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Calculate transform delta from screen movement. Override in subclasses."""
        return None

    @property
    def is_dragging(self) -> bool:
        """Check if currently dragging."""
        return self._dragging


@editor(category="Gizmos")
@reloadable()
class TranslateGizmo(Gizmo):
    """Translation gizmo with axis arrows and plane handles."""
    __slots__ = ("arrow_length", "arrow_head_size", "plane_size",
                 "plane_offset", "sensitivity")

    def __init__(self):
        super().__init__(GizmoType.TRANSLATE)
        self.arrow_length: float = 1.0
        self.arrow_head_size: float = 0.1
        self.plane_size: float = 0.3
        self.plane_offset: float = 0.4
        self.sensitivity: float = 0.01

    def hit_test(self, screen_x: int, screen_y: int,
                 camera_pos: Tuple[float, float, float]) -> GizmoAxis:
        """Test which axis/plane is hit."""
        # Simplified hit test - in practice would do proper ray-gizmo intersection
        # Return based on screen position relative to gizmo center
        return GizmoAxis.NONE

    def _calculate_delta(self, delta_x: float, delta_y: float,
                         camera_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Calculate translation delta."""
        if self._drag_axis == GizmoAxis.NONE:
            return None

        # Sensitivity-adjusted movement
        move = (delta_x - delta_y) * self.sensitivity * self.size

        dx = dy = dz = 0.0

        if GizmoAxis.X in self._drag_axis:
            dx = self.constraint.apply(move)
        if GizmoAxis.Y in self._drag_axis:
            dy = self.constraint.apply(-delta_y * self.sensitivity * self.size)
        if GizmoAxis.Z in self._drag_axis:
            dz = self.constraint.apply(move)

        return (dx, dy, dz)


@editor(category="Gizmos")
@reloadable()
class RotateGizmo(Gizmo):
    """Rotation gizmo with axis circles."""
    __slots__ = ("radius", "thickness", "arc_segments", "sensitivity",
                 "show_angle_display")

    def __init__(self):
        super().__init__(GizmoType.ROTATE)
        self.radius: float = 1.0
        self.thickness: float = 0.05
        self.arc_segments: int = 64
        self.sensitivity: float = 0.5
        self.show_angle_display: bool = True

    def hit_test(self, screen_x: int, screen_y: int,
                 camera_pos: Tuple[float, float, float]) -> GizmoAxis:
        """Test which rotation axis is hit."""
        return GizmoAxis.NONE

    def _calculate_delta(self, delta_x: float, delta_y: float,
                         camera_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Calculate rotation delta (in degrees)."""
        if self._drag_axis == GizmoAxis.NONE:
            return None

        # Calculate rotation amount from mouse movement
        rotation_amount = (delta_x + delta_y) * self.sensitivity

        rx = ry = rz = 0.0

        if GizmoAxis.X in self._drag_axis:
            rx = self.constraint.apply(rotation_amount)
        if GizmoAxis.Y in self._drag_axis:
            ry = self.constraint.apply(rotation_amount)
        if GizmoAxis.Z in self._drag_axis:
            rz = self.constraint.apply(rotation_amount)

        return (rx, ry, rz)


@editor(category="Gizmos")
@reloadable()
class ScaleGizmo(Gizmo):
    """Scale gizmo with axis handles and uniform scale center."""
    __slots__ = ("handle_length", "handle_size", "uniform_handle_size",
                 "sensitivity", "preserve_volume")

    def __init__(self):
        super().__init__(GizmoType.SCALE)
        self.handle_length: float = 1.0
        self.handle_size: float = 0.1
        self.uniform_handle_size: float = 0.2
        self.sensitivity: float = 0.01
        self.preserve_volume: bool = False

    def hit_test(self, screen_x: int, screen_y: int,
                 camera_pos: Tuple[float, float, float]) -> GizmoAxis:
        """Test which scale axis is hit."""
        return GizmoAxis.NONE

    def _calculate_delta(self, delta_x: float, delta_y: float,
                         camera_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Calculate scale delta."""
        if self._drag_axis == GizmoAxis.NONE:
            return None

        # Scale factor from mouse movement
        scale_factor = 1.0 + (delta_x - delta_y) * self.sensitivity

        sx = sy = sz = 1.0

        if self._drag_axis == GizmoAxis.XYZ:
            # Uniform scale
            sx = sy = sz = scale_factor
        else:
            if GizmoAxis.X in self._drag_axis:
                sx = scale_factor
            if GizmoAxis.Y in self._drag_axis:
                sy = scale_factor
            if GizmoAxis.Z in self._drag_axis:
                sz = scale_factor

            # Volume preservation
            if self.preserve_volume:
                volume = sx * sy * sz
                if abs(volume) > 0.001:
                    correction = 1.0 / math.pow(abs(volume), 1.0/3.0)
                    sx *= correction
                    sy *= correction
                    sz *= correction

        return (sx, sy, sz)


@editor(category="Gizmos")
@reloadable()
class UniversalGizmo(Gizmo):
    """Universal gizmo combining translate, rotate, and scale."""
    __slots__ = ("_translate", "_rotate", "_scale", "active_mode",
                 "show_all_handles")

    def __init__(self):
        super().__init__(GizmoType.UNIVERSAL)
        self._translate = TranslateGizmo()
        self._rotate = RotateGizmo()
        self._scale = ScaleGizmo()
        self.active_mode: GizmoType = GizmoType.TRANSLATE
        self.show_all_handles: bool = False

    @property
    def translate(self) -> TranslateGizmo:
        """Get translate gizmo."""
        return self._translate

    @property
    def rotate(self) -> RotateGizmo:
        """Get rotate gizmo."""
        return self._rotate

    @property
    def scale_gizmo(self) -> ScaleGizmo:
        """Get scale gizmo."""
        return self._scale

    def set_mode(self, mode: GizmoType) -> None:
        """Set active transform mode."""
        if mode in (GizmoType.TRANSLATE, GizmoType.ROTATE, GizmoType.SCALE):
            self.active_mode = mode

    def cycle_mode(self) -> GizmoType:
        """Cycle to next transform mode."""
        modes = [GizmoType.TRANSLATE, GizmoType.ROTATE, GizmoType.SCALE]
        current_idx = modes.index(self.active_mode) if self.active_mode in modes else 0
        next_idx = (current_idx + 1) % len(modes)
        self.active_mode = modes[next_idx]
        return self.active_mode

    def set_position(self, x: float, y: float, z: float) -> None:
        """Set position for all sub-gizmos."""
        super().set_position(x, y, z)
        self._translate.set_position(x, y, z)
        self._rotate.set_position(x, y, z)
        self._scale.set_position(x, y, z)

    def set_rotation(self, pitch: float, yaw: float, roll: float) -> None:
        """Set rotation for all sub-gizmos."""
        super().set_rotation(pitch, yaw, roll)
        self._translate.set_rotation(pitch, yaw, roll)
        self._rotate.set_rotation(pitch, yaw, roll)
        self._scale.set_rotation(pitch, yaw, roll)

    def set_space(self, space: GizmoSpace) -> None:
        """Set space for all sub-gizmos."""
        super().set_space(space)
        self._translate.set_space(space)
        self._rotate.set_space(space)
        self._scale.set_space(space)

    def hit_test(self, screen_x: int, screen_y: int,
                 camera_pos: Tuple[float, float, float]) -> GizmoAxis:
        """Test hit on active gizmo."""
        if self.active_mode == GizmoType.TRANSLATE:
            return self._translate.hit_test(screen_x, screen_y, camera_pos)
        elif self.active_mode == GizmoType.ROTATE:
            return self._rotate.hit_test(screen_x, screen_y, camera_pos)
        elif self.active_mode == GizmoType.SCALE:
            return self._scale.hit_test(screen_x, screen_y, camera_pos)
        return GizmoAxis.NONE

    def _get_active_gizmo(self) -> Gizmo:
        """Get the currently active sub-gizmo."""
        if self.active_mode == GizmoType.TRANSLATE:
            return self._translate
        elif self.active_mode == GizmoType.ROTATE:
            return self._rotate
        elif self.active_mode == GizmoType.SCALE:
            return self._scale
        return self._translate

    def begin_drag(self, axis: GizmoAxis, screen_x: int, screen_y: int) -> None:
        """Begin drag on active gizmo."""
        super().begin_drag(axis, screen_x, screen_y)
        self._get_active_gizmo().begin_drag(axis, screen_x, screen_y)

    def update_drag(self, screen_x: int, screen_y: int,
                    camera_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Update drag on active gizmo."""
        return self._get_active_gizmo().update_drag(screen_x, screen_y, camera_pos)

    def end_drag(self) -> None:
        """End drag on active gizmo."""
        super().end_drag()
        self._get_active_gizmo().end_drag()


@editor(category="Gizmos")
@reloadable(preserve=["_gizmos", "_active_gizmo"])
class GizmoManager:
    """Manages gizmos for the editor."""
    __slots__ = ("_gizmos", "_active_gizmo", "_active_type", "space",
                 "snap_translate", "snap_rotate", "snap_scale",
                 "snap_translate_value", "snap_rotate_value", "snap_scale_value",
                 "on_gizmo_changed", "_tracker_ref", "_selection_ref")

    def __init__(self):
        self._gizmos: dict[GizmoType, Gizmo] = {
            GizmoType.TRANSLATE: TranslateGizmo(),
            GizmoType.ROTATE: RotateGizmo(),
            GizmoType.SCALE: ScaleGizmo(),
            GizmoType.UNIVERSAL: UniversalGizmo(),
        }
        self._active_gizmo: Optional[Gizmo] = self._gizmos[GizmoType.TRANSLATE]
        self._active_type: GizmoType = GizmoType.TRANSLATE
        self.space: GizmoSpace = GizmoSpace.WORLD

        # Snap settings
        self.snap_translate: bool = False
        self.snap_rotate: bool = False
        self.snap_scale: bool = False
        self.snap_translate_value: float = 1.0
        self.snap_rotate_value: float = 15.0
        self.snap_scale_value: float = 0.1

        self.on_gizmo_changed: Optional[Callable[[GizmoType], None]] = None
        self._tracker_ref: Any = None
        self._selection_ref: Any = None

    def set_tracker(self, tracker: Any) -> None:
        """Set the Foundation Tracker for undo integration."""
        self._tracker_ref = tracker

    def set_selection(self, selection: Any) -> None:
        """Set the selection manager reference."""
        self._selection_ref = selection

    @property
    def active_gizmo(self) -> Optional[Gizmo]:
        """Get the active gizmo."""
        return self._active_gizmo

    @property
    def active_type(self) -> GizmoType:
        """Get the active gizmo type."""
        return self._active_type

    def set_gizmo_type(self, gizmo_type: GizmoType) -> None:
        """Set the active gizmo type."""
        if gizmo_type == GizmoType.NONE:
            self._active_gizmo = None
        else:
            self._active_gizmo = self._gizmos.get(gizmo_type)
            if self._active_gizmo:
                self._apply_snap_settings(self._active_gizmo)

        self._active_type = gizmo_type
        if self.on_gizmo_changed:
            self.on_gizmo_changed(gizmo_type)

    def cycle_gizmo_type(self) -> GizmoType:
        """Cycle through gizmo types."""
        types = [GizmoType.TRANSLATE, GizmoType.ROTATE, GizmoType.SCALE]
        if self._active_type in types:
            current_idx = types.index(self._active_type)
            next_idx = (current_idx + 1) % len(types)
            self.set_gizmo_type(types[next_idx])
        else:
            self.set_gizmo_type(GizmoType.TRANSLATE)
        return self._active_type

    def set_space(self, space: GizmoSpace) -> None:
        """Set transform space for all gizmos."""
        self.space = space
        for gizmo in self._gizmos.values():
            gizmo.set_space(space)

    def cycle_space(self) -> GizmoSpace:
        """Cycle through transform spaces."""
        spaces = [GizmoSpace.WORLD, GizmoSpace.LOCAL, GizmoSpace.VIEW]
        current_idx = spaces.index(self.space) if self.space in spaces else 0
        next_idx = (current_idx + 1) % len(spaces)
        self.set_space(spaces[next_idx])
        return self.space

    def _apply_snap_settings(self, gizmo: Gizmo) -> None:
        """Apply snap settings to a gizmo."""
        if gizmo.gizmo_type == GizmoType.TRANSLATE:
            gizmo.enable_snap(self.snap_translate, self.snap_translate_value)
        elif gizmo.gizmo_type == GizmoType.ROTATE:
            gizmo.enable_snap(self.snap_rotate, self.snap_rotate_value)
        elif gizmo.gizmo_type == GizmoType.SCALE:
            gizmo.enable_snap(self.snap_scale, self.snap_scale_value)
        elif gizmo.gizmo_type == GizmoType.UNIVERSAL:
            universal = gizmo
            if isinstance(universal, UniversalGizmo):
                universal.translate.enable_snap(self.snap_translate, self.snap_translate_value)
                universal.rotate.enable_snap(self.snap_rotate, self.snap_rotate_value)
                universal.scale_gizmo.enable_snap(self.snap_scale, self.snap_scale_value)

    def update_gizmo_transform(self, position: Tuple[float, float, float],
                               rotation: Tuple[float, float, float] = (0, 0, 0)) -> None:
        """Update gizmo position/rotation to match selection."""
        if self._active_gizmo:
            self._active_gizmo.set_position(*position)
            if self.space == GizmoSpace.LOCAL:
                self._active_gizmo.set_rotation(*rotation)
            else:
                self._active_gizmo.set_rotation(0, 0, 0)

    def begin_transform(self, axis: GizmoAxis, screen_x: int, screen_y: int) -> bool:
        """Begin a transform operation. Returns True if started."""
        if self._active_gizmo and self._active_gizmo.enabled:
            self._active_gizmo.begin_drag(axis, screen_x, screen_y)
            return True
        return False

    def update_transform(self, screen_x: int, screen_y: int,
                         camera_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Update the current transform. Returns delta or None."""
        if self._active_gizmo and self._active_gizmo.is_dragging:
            return self._active_gizmo.update_drag(screen_x, screen_y, camera_pos)
        return None

    def end_transform(self) -> None:
        """End the current transform."""
        if self._active_gizmo:
            self._active_gizmo.end_drag()

    def cancel_transform(self) -> None:
        """Cancel the current transform."""
        if self._active_gizmo:
            self._active_gizmo.cancel_drag()

    @property
    def is_transforming(self) -> bool:
        """Check if a transform is in progress."""
        return self._active_gizmo is not None and self._active_gizmo.is_dragging

    def hit_test(self, screen_x: int, screen_y: int,
                 camera_pos: Tuple[float, float, float]) -> GizmoAxis:
        """Test if screen position hits the active gizmo."""
        if self._active_gizmo:
            return self._active_gizmo.hit_test(screen_x, screen_y, camera_pos)
        return GizmoAxis.NONE

    def get_gizmo(self, gizmo_type: GizmoType) -> Optional[Gizmo]:
        """Get a specific gizmo by type."""
        return self._gizmos.get(gizmo_type)
