"""
Viewport - 3D/2D viewport rendering with camera controls and render modes.

Provides:
- Camera with multiple control modes (orbit, fly, pan/zoom)
- Render modes (Lit, Unlit, Wireframe, Normals, Overdraw, LOD Coloring, Collision, Navmesh)
- Viewport overlays (grid, gizmos, icons)
- Input handling for viewport navigation
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Any, Callable, Optional, Tuple

from engine.tooling.editor.app_shell import editor, reloadable


class CameraMode(Enum):
    """Camera control modes."""
    ORBIT = auto()      # Orbit around focus point
    FLY = auto()        # Free-fly first person
    PAN = auto()        # 2D pan mode
    ZOOM = auto()       # Zoom only


class RenderMode(Enum):
    """Viewport render modes."""
    LIT = auto()
    UNLIT = auto()
    WIREFRAME = auto()
    NORMALS = auto()
    OVERDRAW = auto()
    LOD_COLORING = auto()
    COLLISION = auto()
    NAVMESH = auto()
    DEPTH = auto()
    MOTION_VECTORS = auto()
    LIGHTMAP_DENSITY = auto()
    SHADER_COMPLEXITY = auto()


class ViewportOverlayFlags(Flag):
    """Flags for viewport overlays."""
    NONE = 0
    GRID = auto()
    GIZMOS = auto()
    ICONS = auto()
    SELECTION_OUTLINE = auto()
    BOUNDS = auto()
    NAMES = auto()
    STATS = auto()
    SAFE_FRAMES = auto()
    CAMERA_INFO = auto()


@editor(category="Viewport")
@reloadable(preserve=["position", "rotation", "fov"])
class Camera:
    """A viewport camera with various projection modes."""
    __slots__ = ("position", "rotation", "fov", "near_clip", "far_clip",
                 "orthographic", "ortho_size", "focus_point", "focus_distance",
                 "_view_matrix", "_proj_matrix", "_dirty")

    def __init__(self, position: Tuple[float, float, float] = (0.0, 5.0, 10.0),
                 rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 fov: float = 60.0, near_clip: float = 0.1, far_clip: float = 10000.0):
        self.position = position
        self.rotation = rotation
        self.fov = fov
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.orthographic = False
        self.ortho_size = 10.0
        self.focus_point = (0.0, 0.0, 0.0)
        self.focus_distance = math.sqrt(sum(p*p for p in position))
        self._view_matrix = None
        self._proj_matrix = None
        self._dirty = True

    def set_position(self, x: float, y: float, z: float) -> None:
        """Set camera position."""
        self.position = (x, y, z)
        self._dirty = True

    def set_rotation(self, pitch: float, yaw: float, roll: float = 0.0) -> None:
        """Set camera rotation (Euler angles in degrees)."""
        self.rotation = (pitch, yaw, roll)
        self._dirty = True

    def look_at(self, target: Tuple[float, float, float]) -> None:
        """Point camera at a target position."""
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        dz = target[2] - self.position[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > 0.0001:
            pitch = math.degrees(math.asin(-dy / dist))
            yaw = math.degrees(math.atan2(dx, dz))
            self.rotation = (pitch, yaw, 0.0)
            self.focus_point = target
            self.focus_distance = dist
        self._dirty = True

    def orbit(self, delta_pitch: float, delta_yaw: float) -> None:
        """Orbit camera around focus point."""
        pitch, yaw, roll = self.rotation
        pitch = max(-89.0, min(89.0, pitch + delta_pitch))
        yaw = (yaw + delta_yaw) % 360.0
        self.rotation = (pitch, yaw, roll)

        # Update position based on orbit
        rad_pitch = math.radians(pitch)
        rad_yaw = math.radians(yaw)
        x = self.focus_point[0] + self.focus_distance * math.cos(rad_pitch) * math.sin(rad_yaw)
        y = self.focus_point[1] + self.focus_distance * math.sin(rad_pitch)
        z = self.focus_point[2] + self.focus_distance * math.cos(rad_pitch) * math.cos(rad_yaw)
        self.position = (x, y, z)
        self._dirty = True

    def pan(self, delta_x: float, delta_y: float) -> None:
        """Pan camera (move focus point)."""
        rad_yaw = math.radians(self.rotation[1])
        right = (math.cos(rad_yaw), 0.0, -math.sin(rad_yaw))
        up = (0.0, 1.0, 0.0)

        fx = self.focus_point[0] + right[0] * delta_x + up[0] * delta_y
        fy = self.focus_point[1] + right[1] * delta_x + up[1] * delta_y
        fz = self.focus_point[2] + right[2] * delta_x + up[2] * delta_y
        self.focus_point = (fx, fy, fz)

        px = self.position[0] + right[0] * delta_x + up[0] * delta_y
        py = self.position[1] + right[1] * delta_x + up[1] * delta_y
        pz = self.position[2] + right[2] * delta_x + up[2] * delta_y
        self.position = (px, py, pz)
        self._dirty = True

    def zoom(self, delta: float) -> None:
        """Zoom camera (change focus distance)."""
        self.focus_distance = max(0.1, self.focus_distance * (1.0 - delta * 0.1))
        self.orbit(0, 0)  # Recalculate position

    def fly_forward(self, distance: float) -> None:
        """Move camera forward in fly mode."""
        rad_pitch = math.radians(self.rotation[0])
        rad_yaw = math.radians(self.rotation[1])

        forward = (
            math.cos(rad_pitch) * math.sin(rad_yaw),
            -math.sin(rad_pitch),
            math.cos(rad_pitch) * math.cos(rad_yaw)
        )
        self.position = (
            self.position[0] + forward[0] * distance,
            self.position[1] + forward[1] * distance,
            self.position[2] + forward[2] * distance
        )
        self._dirty = True

    def fly_strafe(self, distance: float) -> None:
        """Strafe camera left/right in fly mode."""
        rad_yaw = math.radians(self.rotation[1])
        right = (math.cos(rad_yaw), 0.0, -math.sin(rad_yaw))
        self.position = (
            self.position[0] + right[0] * distance,
            self.position[1] + right[1] * distance,
            self.position[2] + right[2] * distance
        )
        self._dirty = True

    def fly_up(self, distance: float) -> None:
        """Move camera up/down in fly mode."""
        self.position = (
            self.position[0],
            self.position[1] + distance,
            self.position[2]
        )
        self._dirty = True

    def frame_bounds(self, min_bounds: Tuple[float, float, float],
                     max_bounds: Tuple[float, float, float]) -> None:
        """Frame camera to show bounds."""
        center = (
            (min_bounds[0] + max_bounds[0]) / 2,
            (min_bounds[1] + max_bounds[1]) / 2,
            (min_bounds[2] + max_bounds[2]) / 2
        )
        size = max(
            max_bounds[0] - min_bounds[0],
            max_bounds[1] - min_bounds[1],
            max_bounds[2] - min_bounds[2]
        )
        self.focus_point = center
        self.focus_distance = size * 1.5
        self.orbit(0, 0)

    def set_orthographic(self, ortho: bool, size: float = 10.0) -> None:
        """Set orthographic projection mode."""
        self.orthographic = ortho
        self.ortho_size = size
        self._dirty = True


@editor(category="Viewport")
@reloadable()
class GridSettings:
    """Settings for viewport grid display."""
    __slots__ = ("enabled", "size", "divisions", "primary_color",
                 "secondary_color", "axis_x_color", "axis_y_color",
                 "axis_z_color", "fade_distance", "snap_enabled",
                 "snap_size")

    def __init__(self, enabled: bool = True, size: float = 100.0, divisions: int = 10):
        self.enabled = enabled
        self.size = size
        self.divisions = divisions
        self.primary_color = (0.3, 0.3, 0.3, 1.0)
        self.secondary_color = (0.2, 0.2, 0.2, 1.0)
        self.axis_x_color = (0.8, 0.2, 0.2, 1.0)
        self.axis_y_color = (0.2, 0.8, 0.2, 1.0)
        self.axis_z_color = (0.2, 0.2, 0.8, 1.0)
        self.fade_distance = 50.0
        self.snap_enabled = False
        self.snap_size = 1.0


@editor(category="Viewport")
@reloadable()
class ViewportOverlay:
    """Overlay configuration for a viewport."""
    __slots__ = ("flags", "selection_color", "hover_color", "icon_scale",
                 "text_size", "show_fps", "show_triangle_count",
                 "show_draw_calls")

    def __init__(self):
        self.flags = ViewportOverlayFlags.GRID | ViewportOverlayFlags.GIZMOS
        self.selection_color = (1.0, 0.5, 0.0, 1.0)
        self.hover_color = (0.8, 0.8, 0.0, 1.0)
        self.icon_scale = 1.0
        self.text_size = 12.0
        self.show_fps = True
        self.show_triangle_count = False
        self.show_draw_calls = False

    def toggle_flag(self, flag: ViewportOverlayFlags) -> None:
        """Toggle an overlay flag."""
        if flag in self.flags:
            self.flags &= ~flag
        else:
            self.flags |= flag

    def has_flag(self, flag: ViewportOverlayFlags) -> bool:
        """Check if an overlay flag is set."""
        return flag in self.flags


@editor(category="Viewport")
@reloadable()
class ViewportInput:
    """Handles input for viewport navigation."""
    __slots__ = ("_viewport_ref", "orbit_sensitivity", "pan_sensitivity",
                 "zoom_sensitivity", "fly_speed", "invert_y",
                 "_is_orbiting", "_is_panning", "_last_mouse_pos")

    def __init__(self, viewport: "Viewport"):
        self._viewport_ref = viewport
        self.orbit_sensitivity: float = 0.5
        self.pan_sensitivity: float = 0.01
        self.zoom_sensitivity: float = 1.0
        self.fly_speed: float = 10.0
        self.invert_y: bool = False
        self._is_orbiting: bool = False
        self._is_panning: bool = False
        self._last_mouse_pos: Optional[Tuple[int, int]] = None

    def on_mouse_down(self, x: int, y: int, button: int) -> bool:
        """Handle mouse button down. Returns True if handled."""
        self._last_mouse_pos = (x, y)
        if button == 1:  # Middle button
            self._is_orbiting = True
            return True
        elif button == 2:  # Right button
            self._is_panning = True
            return True
        return False

    def on_mouse_up(self, x: int, y: int, button: int) -> bool:
        """Handle mouse button up. Returns True if handled."""
        if button == 1:
            self._is_orbiting = False
            return True
        elif button == 2:
            self._is_panning = False
            return True
        return False

    def on_mouse_move(self, x: int, y: int) -> bool:
        """Handle mouse movement. Returns True if handled."""
        if self._last_mouse_pos is None:
            self._last_mouse_pos = (x, y)
            return False

        dx = x - self._last_mouse_pos[0]
        dy = y - self._last_mouse_pos[1]
        self._last_mouse_pos = (x, y)

        if self.invert_y:
            dy = -dy

        camera = self._viewport_ref.camera
        if self._is_orbiting:
            camera.orbit(-dy * self.orbit_sensitivity, -dx * self.orbit_sensitivity)
            return True
        elif self._is_panning:
            camera.pan(-dx * self.pan_sensitivity * camera.focus_distance,
                      dy * self.pan_sensitivity * camera.focus_distance)
            return True
        return False

    def on_mouse_wheel(self, delta: float) -> bool:
        """Handle mouse wheel. Returns True if handled."""
        self._viewport_ref.camera.zoom(delta * self.zoom_sensitivity)
        return True

    def on_key_down(self, key: str) -> bool:
        """Handle key down for fly mode. Returns True if handled."""
        if self._viewport_ref.camera_mode != CameraMode.FLY:
            return False

        camera = self._viewport_ref.camera
        if key == "w":
            camera.fly_forward(self.fly_speed)
            return True
        elif key == "s":
            camera.fly_forward(-self.fly_speed)
            return True
        elif key == "a":
            camera.fly_strafe(-self.fly_speed)
            return True
        elif key == "d":
            camera.fly_strafe(self.fly_speed)
            return True
        elif key == "q":
            camera.fly_up(-self.fly_speed)
            return True
        elif key == "e":
            camera.fly_up(self.fly_speed)
            return True
        return False


@editor(category="Viewport")
@reloadable(preserve=["camera", "render_mode"])
class Viewport:
    """A 3D/2D viewport for scene rendering."""
    __slots__ = ("id", "name", "width", "height", "camera", "camera_mode",
                 "render_mode", "grid", "overlay", "input", "_scene_ref",
                 "on_render", "on_resize", "background_color", "active")

    def __init__(self, id: str, name: str = "", width: int = 800, height: int = 600):
        self.id = id
        self.name = name or id
        self.width = width
        self.height = height
        self.camera = Camera()
        self.camera_mode = CameraMode.ORBIT
        self.render_mode = RenderMode.LIT
        self.grid = GridSettings()
        self.overlay = ViewportOverlay()
        self.input = ViewportInput(self)
        self._scene_ref: Any = None
        self.on_render: Optional[Callable[[], None]] = None
        self.on_resize: Optional[Callable[[int, int], None]] = None
        self.background_color: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)
        self.active = True

    @property
    def aspect_ratio(self) -> float:
        """Get viewport aspect ratio."""
        return self.width / max(self.height, 1)

    def set_size(self, width: int, height: int) -> None:
        """Set viewport size."""
        self.width = max(1, width)
        self.height = max(1, height)
        if self.on_resize:
            self.on_resize(self.width, self.height)

    def set_camera_mode(self, mode: CameraMode) -> None:
        """Set camera control mode."""
        self.camera_mode = mode

    def set_render_mode(self, mode: RenderMode) -> None:
        """Set render mode."""
        self.render_mode = mode

    def cycle_render_mode(self) -> RenderMode:
        """Cycle to next render mode."""
        modes = list(RenderMode)
        current_idx = modes.index(self.render_mode)
        next_idx = (current_idx + 1) % len(modes)
        self.render_mode = modes[next_idx]
        return self.render_mode

    def focus_on_selection(self, selection: Any) -> None:
        """Focus camera on selection."""
        if selection and hasattr(selection, "get_bounds"):
            min_b, max_b = selection.get_bounds()
            self.camera.frame_bounds(min_b, max_b)

    def screen_to_world(self, screen_x: int, screen_y: int,
                        depth: float = 1.0) -> Tuple[float, float, float]:
        """Convert screen coordinates to world coordinates."""
        # Normalize screen coordinates to [-1, 1]
        ndc_x = (2.0 * screen_x / self.width) - 1.0
        ndc_y = 1.0 - (2.0 * screen_y / self.height)

        # Basic unprojection (simplified)
        fov_rad = math.radians(self.camera.fov / 2)
        tan_fov = math.tan(fov_rad)
        aspect = self.aspect_ratio

        # Camera space direction
        dir_x = ndc_x * tan_fov * aspect
        dir_y = ndc_y * tan_fov
        dir_z = -1.0

        # Rotate by camera orientation (simplified)
        rad_pitch = math.radians(self.camera.rotation[0])
        rad_yaw = math.radians(self.camera.rotation[1])

        # Apply yaw
        cos_yaw = math.cos(rad_yaw)
        sin_yaw = math.sin(rad_yaw)
        rx = dir_x * cos_yaw + dir_z * sin_yaw
        rz = -dir_x * sin_yaw + dir_z * cos_yaw

        # Apply pitch
        cos_pitch = math.cos(rad_pitch)
        sin_pitch = math.sin(rad_pitch)
        ry = dir_y * cos_pitch - rz * sin_pitch
        rz2 = dir_y * sin_pitch + rz * cos_pitch

        # World position at depth
        return (
            self.camera.position[0] + rx * depth,
            self.camera.position[1] + ry * depth,
            self.camera.position[2] + rz2 * depth
        )

    def world_to_screen(self, world_x: float, world_y: float,
                        world_z: float) -> Optional[Tuple[int, int]]:
        """Convert world coordinates to screen coordinates."""
        # Vector from camera to point
        dx = world_x - self.camera.position[0]
        dy = world_y - self.camera.position[1]
        dz = world_z - self.camera.position[2]

        # Rotate to camera space (inverse camera rotation)
        rad_yaw = math.radians(-self.camera.rotation[1])
        rad_pitch = math.radians(-self.camera.rotation[0])

        cos_yaw = math.cos(rad_yaw)
        sin_yaw = math.sin(rad_yaw)
        rx = dx * cos_yaw + dz * sin_yaw
        rz = -dx * sin_yaw + dz * cos_yaw

        cos_pitch = math.cos(rad_pitch)
        sin_pitch = math.sin(rad_pitch)
        ry = dy * cos_pitch - rz * sin_pitch
        rz2 = dy * sin_pitch + rz * cos_pitch

        # Behind camera check
        if rz2 >= 0:
            return None

        # Project to screen
        fov_rad = math.radians(self.camera.fov / 2)
        tan_fov = math.tan(fov_rad)
        aspect = self.aspect_ratio

        ndc_x = rx / (-rz2 * tan_fov * aspect)
        ndc_y = ry / (-rz2 * tan_fov)

        # NDC to screen
        screen_x = int((ndc_x + 1.0) * 0.5 * self.width)
        screen_y = int((1.0 - ndc_y) * 0.5 * self.height)

        return (screen_x, screen_y)

    def pick_at(self, screen_x: int, screen_y: int) -> Optional[Any]:
        """Pick an object at screen coordinates."""
        # This would integrate with the scene's picking system
        if self._scene_ref and hasattr(self._scene_ref, "pick"):
            ray_origin = self.camera.position
            ray_dir = self.screen_to_world(screen_x, screen_y, 1.0)
            ray_dir = (
                ray_dir[0] - ray_origin[0],
                ray_dir[1] - ray_origin[1],
                ray_dir[2] - ray_origin[2]
            )
            length = math.sqrt(sum(d*d for d in ray_dir))
            ray_dir = (ray_dir[0]/length, ray_dir[1]/length, ray_dir[2]/length)
            return self._scene_ref.pick(ray_origin, ray_dir)
        return None

    def render(self) -> None:
        """Render the viewport."""
        if not self.active:
            return
        if self.on_render:
            self.on_render()

    def save_state(self) -> dict:
        """Save viewport state."""
        return {
            "camera": {
                "position": self.camera.position,
                "rotation": self.camera.rotation,
                "fov": self.camera.fov,
                "focus_point": self.camera.focus_point,
                "focus_distance": self.camera.focus_distance,
                "orthographic": self.camera.orthographic,
                "ortho_size": self.camera.ortho_size,
            },
            "camera_mode": self.camera_mode.name,
            "render_mode": self.render_mode.name,
            "grid_enabled": self.grid.enabled,
            "overlay_flags": self.overlay.flags.value,
        }

    def load_state(self, state: dict) -> None:
        """Load viewport state."""
        if "camera" in state:
            cam = state["camera"]
            self.camera.position = tuple(cam.get("position", (0, 5, 10)))
            self.camera.rotation = tuple(cam.get("rotation", (0, 0, 0)))
            self.camera.fov = cam.get("fov", 60.0)
            self.camera.focus_point = tuple(cam.get("focus_point", (0, 0, 0)))
            self.camera.focus_distance = cam.get("focus_distance", 10.0)
            self.camera.orthographic = cam.get("orthographic", False)
            self.camera.ortho_size = cam.get("ortho_size", 10.0)
        if "camera_mode" in state:
            self.camera_mode = CameraMode[state["camera_mode"]]
        if "render_mode" in state:
            self.render_mode = RenderMode[state["render_mode"]]
        if "grid_enabled" in state:
            self.grid.enabled = state["grid_enabled"]
        if "overlay_flags" in state:
            self.overlay.flags = ViewportOverlayFlags(state["overlay_flags"])
