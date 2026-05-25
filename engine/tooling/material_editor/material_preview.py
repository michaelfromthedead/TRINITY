"""Material preview - Real-time material preview with lighting setups."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import math


class PreviewShape(Enum):
    """Preview mesh shape."""
    SPHERE = auto()
    CUBE = auto()
    PLANE = auto()
    CYLINDER = auto()
    TORUS = auto()
    CUSTOM = auto()


class LightType(Enum):
    """Type of light in preview scene."""
    DIRECTIONAL = auto()
    POINT = auto()
    SPOT = auto()
    AREA = auto()
    ENVIRONMENT = auto()


@dataclass
class PreviewLight:
    """Light in preview scene."""
    light_type: LightType
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    position: Tuple[float, float, float] = (0.0, 5.0, 0.0)
    direction: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    radius: float = 10.0
    spot_angle: float = 45.0
    cast_shadows: bool = True
    enabled: bool = True


@dataclass
class PreviewCamera:
    """Camera for preview scene."""
    position: Tuple[float, float, float] = (0.0, 0.0, 5.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 45.0
    near_plane: float = 0.1
    far_plane: float = 100.0
    orbit_distance: float = 5.0
    orbit_yaw: float = 0.0
    orbit_pitch: float = 0.0

    def orbit(self, delta_yaw: float, delta_pitch: float) -> None:
        """Orbit camera around target."""
        self.orbit_yaw += delta_yaw
        self.orbit_pitch = max(-89.0, min(89.0, self.orbit_pitch + delta_pitch))
        self._update_position()

    def zoom(self, delta: float) -> None:
        """Zoom camera in/out."""
        self.orbit_distance = max(0.5, self.orbit_distance - delta)
        self._update_position()

    def pan(self, delta_x: float, delta_y: float) -> None:
        """Pan camera."""
        # Calculate right and up vectors
        forward = self._normalize(self._subtract(self.target, self.position))
        right = self._normalize(self._cross(forward, self.up))
        up = self._cross(right, forward)

        # Move target
        self.target = (
            self.target[0] + right[0] * delta_x + up[0] * delta_y,
            self.target[1] + right[1] * delta_x + up[1] * delta_y,
            self.target[2] + right[2] * delta_x + up[2] * delta_y
        )
        self._update_position()

    def _update_position(self) -> None:
        """Update position from orbit parameters."""
        yaw_rad = math.radians(self.orbit_yaw)
        pitch_rad = math.radians(self.orbit_pitch)

        x = self.orbit_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = self.orbit_distance * math.sin(pitch_rad)
        z = self.orbit_distance * math.cos(pitch_rad) * math.cos(yaw_rad)

        self.position = (
            self.target[0] + x,
            self.target[1] + y,
            self.target[2] + z
        )

    @staticmethod
    def _subtract(a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, ...]:
        return tuple(x - y for x, y in zip(a, b))

    @staticmethod
    def _cross(a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        )

    @staticmethod
    def _normalize(v: Tuple[float, ...]) -> Tuple[float, ...]:
        length = math.sqrt(sum(x * x for x in v))
        if length > 0:
            return tuple(x / length for x in v)
        return v


@dataclass
class LightingPreset:
    """Preset lighting configuration."""
    name: str
    description: str
    lights: List[PreviewLight]
    environment_map: Optional[str] = None
    ambient_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)


@dataclass
class PreviewSettings:
    """Settings for material preview."""
    shape: PreviewShape = PreviewShape.SPHERE
    custom_mesh_path: str = ""
    wireframe: bool = False
    show_uv_grid: bool = False
    show_normal_vectors: bool = False
    show_tangent_vectors: bool = False
    rotation_speed: float = 0.0  # Auto-rotation
    background_color: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)
    grid_visible: bool = True
    grid_size: float = 10.0
    grid_divisions: int = 10
    exposure: float = 1.0
    gamma: float = 2.2
    tonemap: str = "aces"  # none, reinhard, aces


class PreviewRenderer(ABC):
    """Abstract renderer for material preview."""

    @abstractmethod
    def initialize(self, width: int, height: int) -> bool:
        """Initialize the renderer."""
        pass

    @abstractmethod
    def resize(self, width: int, height: int) -> None:
        """Resize the render target."""
        pass

    @abstractmethod
    def render(
        self,
        camera: PreviewCamera,
        lights: List[PreviewLight],
        settings: PreviewSettings,
        material_data: Dict[str, Any]
    ) -> None:
        """Render a frame."""
        pass

    @abstractmethod
    def get_framebuffer(self) -> Any:
        """Get the rendered framebuffer."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the renderer."""
        pass


class NullPreviewRenderer(PreviewRenderer):
    """Null implementation of preview renderer for testing."""

    def __init__(self):
        self._width = 0
        self._height = 0
        self._initialized = False
        self._render_count = 0

    def initialize(self, width: int, height: int) -> bool:
        self._width = width
        self._height = height
        self._initialized = True
        return True

    def resize(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def render(
        self,
        camera: PreviewCamera,
        lights: List[PreviewLight],
        settings: PreviewSettings,
        material_data: Dict[str, Any]
    ) -> None:
        self._render_count += 1

    def get_framebuffer(self) -> Any:
        return None

    def shutdown(self) -> None:
        self._initialized = False

    @property
    def render_count(self) -> int:
        return self._render_count

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class MaterialPreview:
    """
    Real-time material preview with lighting setups.

    Provides interactive preview of materials with various lighting
    configurations and preview meshes.
    """

    # Default lighting presets
    DEFAULT_PRESETS = {
        "studio": LightingPreset(
            name="Studio",
            description="Soft studio lighting with key, fill, and rim lights",
            lights=[
                PreviewLight(LightType.DIRECTIONAL, (1.0, 0.98, 0.95), 1.0,
                            direction=(-0.5, -1.0, -0.3)),
                PreviewLight(LightType.DIRECTIONAL, (0.8, 0.85, 1.0), 0.5,
                            direction=(0.5, -0.5, 0.5)),
                PreviewLight(LightType.DIRECTIONAL, (1.0, 1.0, 1.0), 0.3,
                            direction=(0.0, 0.5, -1.0)),
            ],
            ambient_color=(0.15, 0.15, 0.18)
        ),
        "outdoor": LightingPreset(
            name="Outdoor",
            description="Bright outdoor sunlight",
            lights=[
                PreviewLight(LightType.DIRECTIONAL, (1.0, 0.95, 0.85), 2.0,
                            direction=(-0.3, -1.0, -0.5), cast_shadows=True),
            ],
            ambient_color=(0.3, 0.35, 0.4)
        ),
        "indoor": LightingPreset(
            name="Indoor",
            description="Warm indoor lighting",
            lights=[
                PreviewLight(LightType.POINT, (1.0, 0.9, 0.7), 1.5,
                            position=(2.0, 3.0, 2.0), radius=8.0),
                PreviewLight(LightType.POINT, (0.8, 0.85, 1.0), 0.8,
                            position=(-2.0, 2.0, -1.0), radius=6.0),
            ],
            ambient_color=(0.1, 0.08, 0.06)
        ),
        "dramatic": LightingPreset(
            name="Dramatic",
            description="High contrast dramatic lighting",
            lights=[
                PreviewLight(LightType.SPOT, (1.0, 0.9, 0.8), 3.0,
                            position=(3.0, 4.0, 3.0),
                            direction=(-0.5, -0.7, -0.5),
                            spot_angle=30.0),
            ],
            ambient_color=(0.02, 0.02, 0.03)
        ),
        "neutral": LightingPreset(
            name="Neutral",
            description="Even, neutral lighting for material evaluation",
            lights=[
                PreviewLight(LightType.DIRECTIONAL, (1.0, 1.0, 1.0), 1.0,
                            direction=(0.0, -1.0, 0.0)),
                PreviewLight(LightType.DIRECTIONAL, (0.5, 0.5, 0.5), 0.5,
                            direction=(0.0, 1.0, 0.0)),
            ],
            ambient_color=(0.2, 0.2, 0.2)
        ),
        "rim": LightingPreset(
            name="Rim Light",
            description="Strong rim lighting for edge visualization",
            lights=[
                PreviewLight(LightType.DIRECTIONAL, (0.3, 0.3, 0.3), 0.5,
                            direction=(0.0, -1.0, 1.0)),
                PreviewLight(LightType.DIRECTIONAL, (1.0, 1.0, 1.0), 2.0,
                            direction=(0.0, 0.0, -1.0)),
            ],
            ambient_color=(0.05, 0.05, 0.05)
        ),
    }

    def __init__(self, renderer: Optional[PreviewRenderer] = None):
        self._renderer = renderer or NullPreviewRenderer()
        self._camera = PreviewCamera()
        self._lights: List[PreviewLight] = []
        self._settings = PreviewSettings()
        self._presets = dict(self.DEFAULT_PRESETS)
        self._current_preset: Optional[str] = None
        self._material_data: Dict[str, Any] = {}
        self._width = 512
        self._height = 512
        self._dirty = True
        self._auto_update = True

        # Callbacks
        self._on_render_complete: List[Callable[[], None]] = []

        # Apply default preset
        self.apply_preset("studio")

    @property
    def camera(self) -> PreviewCamera:
        return self._camera

    @property
    def settings(self) -> PreviewSettings:
        return self._settings

    @property
    def lights(self) -> List[PreviewLight]:
        return self._lights.copy()

    @property
    def current_preset(self) -> Optional[str]:
        return self._current_preset

    @property
    def auto_update(self) -> bool:
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value: bool) -> None:
        self._auto_update = value
        if value and self._dirty:
            self.render()

    def initialize(self, width: int = 512, height: int = 512) -> bool:
        """Initialize the preview renderer."""
        self._width = width
        self._height = height
        return self._renderer.initialize(width, height)

    def resize(self, width: int, height: int) -> None:
        """Resize the preview."""
        self._width = width
        self._height = height
        self._renderer.resize(width, height)
        self.mark_dirty()

    def shutdown(self) -> None:
        """Shutdown the preview renderer."""
        self._renderer.shutdown()

    def mark_dirty(self) -> None:
        """Mark preview as needing update."""
        self._dirty = True
        if self._auto_update:
            self.render()

    def set_material_data(self, data: Dict[str, Any]) -> None:
        """Set material data for preview."""
        self._material_data = data
        self.mark_dirty()

    def render(self) -> None:
        """Render the preview."""
        self._renderer.render(
            self._camera,
            self._lights,
            self._settings,
            self._material_data
        )
        self._dirty = False

        for callback in self._on_render_complete:
            callback()

    def get_framebuffer(self) -> Any:
        """Get the rendered framebuffer."""
        return self._renderer.get_framebuffer()

    # ========================================================================
    # Lighting
    # ========================================================================

    def add_light(self, light: PreviewLight) -> int:
        """Add a light and return its index."""
        self._lights.append(light)
        self._current_preset = None
        self.mark_dirty()
        return len(self._lights) - 1

    def remove_light(self, index: int) -> bool:
        """Remove a light by index."""
        if 0 <= index < len(self._lights):
            self._lights.pop(index)
            self._current_preset = None
            self.mark_dirty()
            return True
        return False

    def get_light(self, index: int) -> Optional[PreviewLight]:
        """Get a light by index."""
        if 0 <= index < len(self._lights):
            return self._lights[index]
        return None

    def update_light(self, index: int, light: PreviewLight) -> bool:
        """Update a light at index."""
        if 0 <= index < len(self._lights):
            self._lights[index] = light
            self._current_preset = None
            self.mark_dirty()
            return True
        return False

    def clear_lights(self) -> None:
        """Remove all lights."""
        self._lights.clear()
        self._current_preset = None
        self.mark_dirty()

    # ========================================================================
    # Presets
    # ========================================================================

    def get_preset_names(self) -> List[str]:
        """Get list of available preset names."""
        return list(self._presets.keys())

    def get_preset(self, name: str) -> Optional[LightingPreset]:
        """Get a lighting preset by name."""
        return self._presets.get(name)

    def apply_preset(self, name: str) -> bool:
        """Apply a lighting preset."""
        preset = self._presets.get(name)
        if preset is None:
            return False

        self._lights = [PreviewLight(
            light_type=light.light_type,
            color=light.color,
            intensity=light.intensity,
            position=light.position,
            direction=light.direction,
            radius=light.radius,
            spot_angle=light.spot_angle,
            cast_shadows=light.cast_shadows,
            enabled=light.enabled
        ) for light in preset.lights]

        self._current_preset = name
        self.mark_dirty()
        return True

    def register_preset(self, preset: LightingPreset) -> None:
        """Register a custom lighting preset."""
        self._presets[preset.name.lower()] = preset

    def unregister_preset(self, name: str) -> bool:
        """Unregister a lighting preset."""
        if name.lower() in self._presets and name.lower() not in self.DEFAULT_PRESETS:
            del self._presets[name.lower()]
            return True
        return False

    # ========================================================================
    # Preview Shape
    # ========================================================================

    def set_shape(self, shape: PreviewShape, custom_path: str = "") -> None:
        """Set the preview mesh shape."""
        self._settings.shape = shape
        if shape == PreviewShape.CUSTOM:
            self._settings.custom_mesh_path = custom_path
        self.mark_dirty()

    # ========================================================================
    # Camera Controls
    # ========================================================================

    def orbit_camera(self, delta_yaw: float, delta_pitch: float) -> None:
        """Orbit camera around target."""
        self._camera.orbit(delta_yaw, delta_pitch)
        self.mark_dirty()

    def zoom_camera(self, delta: float) -> None:
        """Zoom camera in/out."""
        self._camera.zoom(delta)
        self.mark_dirty()

    def pan_camera(self, delta_x: float, delta_y: float) -> None:
        """Pan camera."""
        self._camera.pan(delta_x, delta_y)
        self.mark_dirty()

    def reset_camera(self) -> None:
        """Reset camera to default position."""
        self._camera = PreviewCamera()
        self.mark_dirty()

    def frame_object(self) -> None:
        """Frame the camera to fit the preview object."""
        # Reset to default framing
        self._camera.orbit_distance = 5.0
        self._camera.orbit_yaw = 45.0
        self._camera.orbit_pitch = 30.0
        self._camera.target = (0.0, 0.0, 0.0)
        self._camera._update_position()
        self.mark_dirty()

    # ========================================================================
    # Settings
    # ========================================================================

    def set_wireframe(self, enabled: bool) -> None:
        """Enable/disable wireframe mode."""
        self._settings.wireframe = enabled
        self.mark_dirty()

    def set_uv_grid(self, enabled: bool) -> None:
        """Enable/disable UV grid overlay."""
        self._settings.show_uv_grid = enabled
        self.mark_dirty()

    def set_normal_vectors(self, enabled: bool) -> None:
        """Enable/disable normal vector visualization."""
        self._settings.show_normal_vectors = enabled
        self.mark_dirty()

    def set_tangent_vectors(self, enabled: bool) -> None:
        """Enable/disable tangent vector visualization."""
        self._settings.show_tangent_vectors = enabled
        self.mark_dirty()

    def set_auto_rotation(self, speed: float) -> None:
        """Set auto-rotation speed (0 to disable)."""
        self._settings.rotation_speed = speed
        self.mark_dirty()

    def set_background(self, r: float, g: float, b: float, a: float = 1.0) -> None:
        """Set background color."""
        self._settings.background_color = (r, g, b, a)
        self.mark_dirty()

    def set_grid(self, visible: bool, size: float = 10.0, divisions: int = 10) -> None:
        """Configure grid display."""
        self._settings.grid_visible = visible
        self._settings.grid_size = size
        self._settings.grid_divisions = divisions
        self.mark_dirty()

    def set_exposure(self, exposure: float) -> None:
        """Set exposure value."""
        self._settings.exposure = max(0.01, exposure)
        self.mark_dirty()

    def set_gamma(self, gamma: float) -> None:
        """Set gamma value."""
        self._settings.gamma = max(0.1, gamma)
        self.mark_dirty()

    def set_tonemap(self, method: str) -> None:
        """Set tonemapping method."""
        if method in ("none", "reinhard", "aces"):
            self._settings.tonemap = method
            self.mark_dirty()

    # ========================================================================
    # Callbacks
    # ========================================================================

    def on_render_complete(self, callback: Callable[[], None]) -> None:
        """Register callback for render completion."""
        self._on_render_complete.append(callback)

    # ========================================================================
    # Update Loop
    # ========================================================================

    def update(self, delta_time: float) -> None:
        """Update preview (call every frame for auto-rotation)."""
        if self._settings.rotation_speed != 0:
            self._camera.orbit(
                self._settings.rotation_speed * delta_time * 60,
                0
            )
            # Don't mark dirty here, just update camera
            if self._auto_update:
                self.render()
