"""Animation preview with ground, lighting, and props.

Provides a preview scene for visualizing animations with configurable
ground, lighting, camera, and props.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.core.math import Quat, Transform, Vec3


# =============================================================================
# ENUMS
# =============================================================================


class PropType(Enum):
    """Types of preview props."""

    STATIC_MESH = auto()
    SKELETAL_MESH = auto()
    PRIMITIVE = auto()
    LIGHT = auto()
    PARTICLE = auto()


# =============================================================================
# SETTINGS
# =============================================================================


@dataclass
class GroundSettings:
    """Settings for preview ground plane.

    Attributes:
        visible: Whether ground is visible
        grid_visible: Whether grid is visible
        color: Ground color (RGB)
        grid_color: Grid line color (RGB)
        size: Ground plane size
        grid_divisions: Number of grid divisions
        reflection_enabled: Whether ground reflects
        reflection_strength: Reflection strength (0-1)
    """

    visible: bool = True
    grid_visible: bool = True
    color: Tuple[int, int, int] = (50, 50, 50)
    grid_color: Tuple[int, int, int] = (80, 80, 80)
    size: float = 10.0
    grid_divisions: int = 10
    reflection_enabled: bool = False
    reflection_strength: float = 0.5


@dataclass
class LightingSettings:
    """Settings for preview lighting.

    Attributes:
        directional_enabled: Whether main directional light is enabled
        directional_color: Directional light color
        directional_intensity: Directional light intensity
        directional_rotation: Directional light rotation (euler)
        ambient_enabled: Whether ambient light is enabled
        ambient_color: Ambient light color
        ambient_intensity: Ambient light intensity
        sky_light_enabled: Whether sky light is enabled
        sky_color_top: Sky color at top
        sky_color_bottom: Sky color at bottom
    """

    directional_enabled: bool = True
    directional_color: Tuple[int, int, int] = (255, 250, 240)
    directional_intensity: float = 1.0
    directional_rotation: Tuple[float, float, float] = (-45.0, 30.0, 0.0)

    ambient_enabled: bool = True
    ambient_color: Tuple[int, int, int] = (100, 100, 120)
    ambient_intensity: float = 0.3

    sky_light_enabled: bool = True
    sky_color_top: Tuple[int, int, int] = (150, 180, 220)
    sky_color_bottom: Tuple[int, int, int] = (80, 80, 80)


@dataclass
class CameraSettings:
    """Settings for preview camera.

    Attributes:
        fov: Field of view in degrees
        near_clip: Near clip distance
        far_clip: Far clip distance
        orbit_enabled: Whether orbit mode is enabled
        orbit_target: Orbit target position
        orbit_distance: Distance from target
        orbit_yaw: Horizontal angle
        orbit_pitch: Vertical angle
        pan_speed: Pan speed multiplier
        zoom_speed: Zoom speed multiplier
    """

    fov: float = 60.0
    near_clip: float = 0.1
    far_clip: float = 1000.0

    orbit_enabled: bool = True
    orbit_target: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    orbit_distance: float = 3.0
    orbit_yaw: float = 0.0
    orbit_pitch: float = 15.0

    pan_speed: float = 1.0
    zoom_speed: float = 1.0


@dataclass
class PreviewSettings:
    """Combined preview settings.

    Attributes:
        ground: Ground settings
        lighting: Lighting settings
        camera: Camera settings
        show_skeleton: Whether to show skeleton
        show_mesh: Whether to show mesh
        show_bounds: Whether to show bounds
        show_floor_contact: Whether to show floor contact
        background_color: Background color
    """

    ground: GroundSettings = field(default_factory=GroundSettings)
    lighting: LightingSettings = field(default_factory=LightingSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)

    show_skeleton: bool = False
    show_mesh: bool = True
    show_bounds: bool = False
    show_floor_contact: bool = False
    show_origin: bool = False
    show_bone_names: bool = False
    background_color: Tuple[int, int, int] = (40, 40, 45)

    wireframe_mode: bool = False
    unlit_mode: bool = False


# =============================================================================
# PREVIEW PROP
# =============================================================================


@dataclass
class PreviewProp:
    """A prop in the preview scene.

    Attributes:
        name: Prop name
        prop_type: Type of prop
        asset_path: Path to prop asset
        transform: World transform
        visible: Whether prop is visible
        attached_to_bone: Bone to attach to (if any)
        socket_name: Socket to attach to (if any)
    """

    name: str
    prop_type: PropType
    asset_path: str = ""
    transform: Transform = field(default_factory=Transform.identity)
    visible: bool = True
    attached_to_bone: Optional[str] = None
    socket_name: Optional[str] = None
    color: Tuple[int, int, int] = (200, 200, 200)
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Prop name cannot be empty")

    @property
    def is_attached(self) -> bool:
        """Check if prop is attached to something."""
        return self.attached_to_bone is not None or self.socket_name is not None

    def attach_to_bone(self, bone_name: str) -> None:
        """Attach to a bone."""
        self.attached_to_bone = bone_name
        self.socket_name = None

    def attach_to_socket(self, socket_name: str) -> None:
        """Attach to a socket."""
        self.socket_name = socket_name
        self.attached_to_bone = None

    def detach(self) -> None:
        """Detach from bone/socket."""
        self.attached_to_bone = None
        self.socket_name = None

    def copy(self) -> PreviewProp:
        """Create a copy."""
        return PreviewProp(
            name=self.name,
            prop_type=self.prop_type,
            asset_path=self.asset_path,
            transform=Transform(
                translation=Vec3(
                    self.transform.translation.x,
                    self.transform.translation.y,
                    self.transform.translation.z,
                ),
                rotation=Quat(
                    self.transform.rotation.x,
                    self.transform.rotation.y,
                    self.transform.rotation.z,
                    self.transform.rotation.w,
                ),
                scale=Vec3(
                    self.transform.scale.x,
                    self.transform.scale.y,
                    self.transform.scale.z,
                ),
            ),
            visible=self.visible,
            attached_to_bone=self.attached_to_bone,
            socket_name=self.socket_name,
            color=self.color,
            scale=self.scale,
        )


# =============================================================================
# PREVIEW PLAYBACK
# =============================================================================


@dataclass
class PreviewPlayback:
    """Playback state for preview.

    Attributes:
        is_playing: Whether playback is active
        current_time: Current playback time
        playback_speed: Speed multiplier
        loop: Whether to loop
        animation_duration: Duration of animation
    """

    is_playing: bool = False
    current_time: float = 0.0
    playback_speed: float = 1.0
    loop: bool = True
    animation_duration: float = 0.0

    def play(self) -> None:
        """Start playback."""
        self.is_playing = True

    def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False

    def stop(self) -> None:
        """Stop and reset playback."""
        self.is_playing = False
        self.current_time = 0.0

    def toggle(self) -> None:
        """Toggle playback."""
        self.is_playing = not self.is_playing

    def seek(self, time: float) -> None:
        """Seek to a time."""
        self.current_time = max(0.0, min(time, self.animation_duration))

    def seek_normalized(self, normalized: float) -> None:
        """Seek to a normalized time (0-1)."""
        self.current_time = normalized * self.animation_duration

    def update(self, dt: float) -> bool:
        """Update playback.

        Returns:
            True if looped this frame
        """
        if not self.is_playing:
            return False

        self.current_time += dt * self.playback_speed
        looped = False

        if self.current_time >= self.animation_duration:
            if self.loop:
                self.current_time = self.current_time % self.animation_duration
                looped = True
            else:
                self.current_time = self.animation_duration
                self.is_playing = False

        elif self.current_time < 0:
            if self.loop:
                self.current_time = self.animation_duration + self.current_time
                looped = True
            else:
                self.current_time = 0.0
                self.is_playing = False

        return looped

    @property
    def normalized_time(self) -> float:
        """Get normalized time (0-1)."""
        if self.animation_duration <= 0:
            return 0.0
        return self.current_time / self.animation_duration


# =============================================================================
# PREVIEW VIEWPORT
# =============================================================================


class PreviewViewport:
    """Viewport settings for the preview.

    Manages viewport dimensions, aspect ratio, and rendering settings.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
    ) -> None:
        self._width = width
        self._height = height
        self._scale_factor = 1.0
        self._render_mode = "default"  # default, wireframe, unlit, normals

    @property
    def width(self) -> int:
        """Get viewport width."""
        return self._width

    @width.setter
    def width(self, value: int) -> None:
        """Set viewport width."""
        self._width = max(1, value)

    @property
    def height(self) -> int:
        """Get viewport height."""
        return self._height

    @height.setter
    def height(self, value: int) -> None:
        """Set viewport height."""
        self._height = max(1, value)

    @property
    def aspect_ratio(self) -> float:
        """Get viewport aspect ratio."""
        return self._width / self._height if self._height > 0 else 1.0

    @property
    def scale_factor(self) -> float:
        """Get render scale factor."""
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value: float) -> None:
        """Set render scale factor."""
        self._scale_factor = max(0.25, min(4.0, value))

    @property
    def render_mode(self) -> str:
        """Get render mode."""
        return self._render_mode

    @render_mode.setter
    def render_mode(self, value: str) -> None:
        """Set render mode."""
        valid_modes = ("default", "wireframe", "unlit", "normals", "uv")
        if value not in valid_modes:
            raise ValueError(f"Invalid render mode: {value}")
        self._render_mode = value

    @property
    def render_width(self) -> int:
        """Get actual render width."""
        return int(self._width * self._scale_factor)

    @property
    def render_height(self) -> int:
        """Get actual render height."""
        return int(self._height * self._scale_factor)

    def resize(self, width: int, height: int) -> None:
        """Resize viewport."""
        self._width = max(1, width)
        self._height = max(1, height)


# =============================================================================
# PREVIEW SCENE
# =============================================================================


class PreviewScene:
    """Animation preview scene.

    Provides a complete preview environment for animations with ground,
    lighting, props, and camera controls.
    """

    def __init__(self) -> None:
        self._settings = PreviewSettings()
        self._viewport = PreviewViewport()
        self._playback = PreviewPlayback()
        self._props: Dict[str, PreviewProp] = {}
        self._animation_path: Optional[str] = None
        self._skeleton_path: Optional[str] = None
        self._mesh_path: Optional[str] = None
        self._selected_prop: Optional[str] = None
        self._on_change_callbacks: List[Callable[[], None]] = []

    @property
    def settings(self) -> PreviewSettings:
        """Get preview settings."""
        return self._settings

    @property
    def viewport(self) -> PreviewViewport:
        """Get viewport."""
        return self._viewport

    @property
    def playback(self) -> PreviewPlayback:
        """Get playback state."""
        return self._playback

    @property
    def props(self) -> List[PreviewProp]:
        """Get all props."""
        return list(self._props.values())

    @property
    def prop_count(self) -> int:
        """Get number of props."""
        return len(self._props)

    @property
    def animation_path(self) -> Optional[str]:
        """Get animation asset path."""
        return self._animation_path

    @property
    def skeleton_path(self) -> Optional[str]:
        """Get skeleton asset path."""
        return self._skeleton_path

    @property
    def mesh_path(self) -> Optional[str]:
        """Get mesh asset path."""
        return self._mesh_path

    @property
    def selected_prop(self) -> Optional[str]:
        """Get selected prop name."""
        return self._selected_prop

    def set_animation(self, path: str, duration: float = 0.0) -> None:
        """Set the animation to preview."""
        self._animation_path = path
        self._playback.animation_duration = duration
        self._playback.stop()
        self._notify_change()

    def set_skeleton(self, path: str) -> None:
        """Set the skeleton to use."""
        self._skeleton_path = path
        self._notify_change()

    def set_mesh(self, path: str) -> None:
        """Set the preview mesh."""
        self._mesh_path = path
        self._notify_change()

    def add_prop(self, prop: PreviewProp) -> bool:
        """Add a prop to the scene."""
        if prop.name in self._props:
            return False
        self._props[prop.name] = prop
        self._notify_change()
        return True

    def remove_prop(self, name: str) -> bool:
        """Remove a prop from the scene."""
        if name not in self._props:
            return False
        del self._props[name]
        if self._selected_prop == name:
            self._selected_prop = None
        self._notify_change()
        return True

    def get_prop(self, name: str) -> Optional[PreviewProp]:
        """Get a prop by name."""
        return self._props.get(name)

    def select_prop(self, name: Optional[str]) -> None:
        """Select a prop."""
        if name is None or name in self._props:
            self._selected_prop = name

    def rename_prop(self, old_name: str, new_name: str) -> bool:
        """Rename a prop."""
        if old_name not in self._props:
            return False
        if new_name in self._props:
            return False

        prop = self._props.pop(old_name)
        prop.name = new_name
        self._props[new_name] = prop

        if self._selected_prop == old_name:
            self._selected_prop = new_name

        self._notify_change()
        return True

    def add_static_mesh_prop(
        self,
        name: str,
        asset_path: str,
        position: Optional[Vec3] = None,
    ) -> PreviewProp:
        """Add a static mesh prop."""
        transform = Transform.identity()
        if position:
            transform.translation = position

        prop = PreviewProp(
            name=name,
            prop_type=PropType.STATIC_MESH,
            asset_path=asset_path,
            transform=transform,
        )
        self.add_prop(prop)
        return prop

    def add_primitive_prop(
        self,
        name: str,
        primitive_type: str = "cube",  # cube, sphere, cylinder, plane
        position: Optional[Vec3] = None,
        scale: float = 1.0,
    ) -> PreviewProp:
        """Add a primitive prop."""
        transform = Transform.identity()
        if position:
            transform.translation = position

        prop = PreviewProp(
            name=name,
            prop_type=PropType.PRIMITIVE,
            asset_path=primitive_type,
            transform=transform,
            scale=scale,
        )
        self.add_prop(prop)
        return prop

    def add_attached_prop(
        self,
        name: str,
        asset_path: str,
        socket_name: str,
    ) -> PreviewProp:
        """Add a prop attached to a socket."""
        prop = PreviewProp(
            name=name,
            prop_type=PropType.STATIC_MESH,
            asset_path=asset_path,
            socket_name=socket_name,
        )
        self.add_prop(prop)
        return prop

    def clear_props(self) -> None:
        """Remove all props."""
        self._props.clear()
        self._selected_prop = None
        self._notify_change()

    # Camera controls

    def orbit_camera(self, delta_yaw: float, delta_pitch: float) -> None:
        """Orbit camera around target."""
        self._settings.camera.orbit_yaw += delta_yaw
        self._settings.camera.orbit_pitch += delta_pitch
        # Clamp pitch to prevent flipping
        self._settings.camera.orbit_pitch = max(
            -89.0,
            min(89.0, self._settings.camera.orbit_pitch),
        )

    def pan_camera(self, delta_x: float, delta_y: float) -> None:
        """Pan camera target."""
        speed = self._settings.camera.pan_speed * 0.01
        target = self._settings.camera.orbit_target
        # Simple pan in world XZ plane
        target.x += delta_x * speed
        target.z += delta_y * speed

    def zoom_camera(self, delta: float) -> None:
        """Zoom camera (change distance)."""
        speed = self._settings.camera.zoom_speed * 0.1
        self._settings.camera.orbit_distance -= delta * speed
        self._settings.camera.orbit_distance = max(
            0.1,
            min(100.0, self._settings.camera.orbit_distance),
        )

    def reset_camera(self) -> None:
        """Reset camera to default position."""
        self._settings.camera.orbit_target = Vec3(0, 1, 0)
        self._settings.camera.orbit_distance = 3.0
        self._settings.camera.orbit_yaw = 0.0
        self._settings.camera.orbit_pitch = 15.0

    def frame_character(self) -> None:
        """Frame camera on character."""
        self._settings.camera.orbit_target = Vec3(0, 1, 0)
        self._settings.camera.orbit_distance = 3.0

    def get_camera_position(self) -> Vec3:
        """Get camera world position from orbit settings."""
        import math

        yaw_rad = math.radians(self._settings.camera.orbit_yaw)
        pitch_rad = math.radians(self._settings.camera.orbit_pitch)
        dist = self._settings.camera.orbit_distance
        target = self._settings.camera.orbit_target

        x = target.x + dist * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = target.y + dist * math.sin(pitch_rad)
        z = target.z + dist * math.cos(pitch_rad) * math.cos(yaw_rad)

        return Vec3(x, y, z)

    # Preset configurations

    def apply_preset_default(self) -> None:
        """Apply default preview preset."""
        self._settings = PreviewSettings()
        self._notify_change()

    def apply_preset_dark(self) -> None:
        """Apply dark preview preset."""
        self._settings = PreviewSettings(
            ground=GroundSettings(
                color=(30, 30, 30),
                grid_color=(50, 50, 50),
            ),
            lighting=LightingSettings(
                directional_intensity=0.7,
                ambient_intensity=0.2,
            ),
            background_color=(20, 20, 25),
        )
        self._notify_change()

    def apply_preset_bright(self) -> None:
        """Apply bright preview preset."""
        self._settings = PreviewSettings(
            ground=GroundSettings(
                color=(100, 100, 100),
                grid_color=(130, 130, 130),
            ),
            lighting=LightingSettings(
                directional_intensity=1.2,
                ambient_intensity=0.5,
            ),
            background_color=(150, 150, 160),
        )
        self._notify_change()

    def apply_preset_studio(self) -> None:
        """Apply studio lighting preset."""
        self._settings = PreviewSettings(
            ground=GroundSettings(
                visible=True,
                grid_visible=False,
                color=(60, 60, 60),
                reflection_enabled=True,
                reflection_strength=0.3,
            ),
            lighting=LightingSettings(
                directional_intensity=0.8,
                ambient_intensity=0.4,
                sky_light_enabled=True,
            ),
            background_color=(70, 70, 75),
        )
        self._notify_change()

    def update(self, dt: float) -> None:
        """Update preview scene."""
        self._playback.update(dt)

    def add_on_change(self, callback: Callable[[], None]) -> None:
        """Register change callback."""
        self._on_change_callbacks.append(callback)

    def remove_on_change(self, callback: Callable[[], None]) -> None:
        """Remove change callback."""
        if callback in self._on_change_callbacks:
            self._on_change_callbacks.remove(callback)

    def _notify_change(self) -> None:
        """Notify change callbacks."""
        for callback in self._on_change_callbacks:
            callback()


__all__ = [
    "PropType",
    "GroundSettings",
    "LightingSettings",
    "CameraSettings",
    "PreviewSettings",
    "PreviewProp",
    "PreviewPlayback",
    "PreviewViewport",
    "PreviewScene",
]
