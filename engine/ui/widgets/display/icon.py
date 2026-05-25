"""
Icon widget - display icons with tinting, sizing, and animation support.

An Icon widget displays graphical icons from an atlas or sprite system.
It supports:
    - Size presets (small=16, medium=24, large=32, custom)
    - Color tinting
    - Animated icons (frame-based)
    - Icon atlas management
    - Rotation and flipping

Example:
    icon = Icon(name="sword")
    icon = Icon(name="heart", size=IconSize.LARGE, tint="#ff0000")
    icon = Icon(name="loading", animated=True, frame_count=8, fps=12)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, TYPE_CHECKING
import time


class IconSize(Enum):
    """Predefined icon size presets."""
    SMALL = 16       # 16x16 pixels
    MEDIUM = 24      # 24x24 pixels
    LARGE = 32       # 32x32 pixels
    XLARGE = 48      # 48x48 pixels
    CUSTOM = 0       # Custom size specified by width/height


class IconFlip(Enum):
    """Icon flip/mirror options."""
    NONE = auto()
    HORIZONTAL = auto()
    VERTICAL = auto()
    BOTH = auto()


@dataclass(slots=True)
class IconAnimation:
    """Configuration for animated icons."""
    frame_count: int = 1           # Number of frames
    fps: float = 12.0              # Frames per second
    loop: bool = True              # Whether to loop
    ping_pong: bool = False        # Reverse direction at end
    start_frame: int = 0           # Starting frame index

    # Runtime state (managed by Icon)
    current_frame: int = field(default=0, repr=False)
    elapsed_time: float = field(default=0.0, repr=False)
    is_playing: bool = field(default=True, repr=False)
    is_reversed: bool = field(default=False, repr=False)  # For ping-pong

    @property
    def frame_duration(self) -> float:
        """Get duration of each frame in seconds."""
        return 1.0 / self.fps if self.fps > 0 else 0.0


@dataclass(slots=True)
class IconAtlasEntry:
    """Entry for an icon in an atlas."""
    name: str                      # Icon name/identifier
    atlas_name: str                # Atlas texture name
    x: int                         # X position in atlas (pixels)
    y: int                         # Y position in atlas (pixels)
    width: int                     # Width in atlas (pixels)
    height: int                    # Height in atlas (pixels)
    # For animated icons, this is the first frame
    frame_offset_x: int = 0        # X offset between frames
    frame_offset_y: int = 0        # Y offset between frames


class IconAtlasManager:
    """
    Singleton manager for icon atlases.

    Manages loading and lookup of icons from texture atlases.
    """

    _instance: Optional["IconAtlasManager"] = None

    def __new__(cls) -> "IconAtlasManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._atlases: dict[str, dict[str, IconAtlasEntry]] = {}
            cls._instance._default_atlas: Optional[str] = None
        return cls._instance

    @classmethod
    def get_instance(cls) -> "IconAtlasManager":
        """Get the singleton instance."""
        return cls()

    def register_atlas(self, atlas_name: str, entries: list[IconAtlasEntry]) -> None:
        """
        Register an icon atlas.

        Args:
            atlas_name: Name of the atlas texture.
            entries: List of icon entries in the atlas.
        """
        self._atlases[atlas_name] = {entry.name: entry for entry in entries}
        if self._default_atlas is None:
            self._default_atlas = atlas_name

    def unregister_atlas(self, atlas_name: str) -> None:
        """Remove an atlas from the registry."""
        self._atlases.pop(atlas_name, None)
        if self._default_atlas == atlas_name:
            self._default_atlas = next(iter(self._atlases.keys()), None)

    def get_icon(self, name: str, atlas_name: Optional[str] = None) -> Optional[IconAtlasEntry]:
        """
        Look up an icon by name.

        Args:
            name: Icon name.
            atlas_name: Specific atlas to search (None = search all).

        Returns:
            IconAtlasEntry if found, None otherwise.
        """
        if atlas_name:
            atlas = self._atlases.get(atlas_name, {})
            return atlas.get(name)

        # Search all atlases
        for atlas in self._atlases.values():
            if name in atlas:
                return atlas[name]
        return None

    def has_icon(self, name: str, atlas_name: Optional[str] = None) -> bool:
        """Check if an icon exists."""
        return self.get_icon(name, atlas_name) is not None

    def list_icons(self, atlas_name: Optional[str] = None) -> list[str]:
        """
        List all registered icon names.

        Args:
            atlas_name: Specific atlas (None = all atlases).

        Returns:
            List of icon names.
        """
        if atlas_name:
            return list(self._atlases.get(atlas_name, {}).keys())

        icons = []
        for atlas in self._atlases.values():
            icons.extend(atlas.keys())
        return icons

    def clear(self) -> None:
        """Clear all registered atlases."""
        self._atlases.clear()
        self._default_atlas = None

    @property
    def default_atlas(self) -> Optional[str]:
        """Get the default atlas name."""
        return self._default_atlas

    @default_atlas.setter
    def default_atlas(self, atlas_name: str) -> None:
        """Set the default atlas name."""
        if atlas_name in self._atlases:
            self._default_atlas = atlas_name


class Icon:
    """
    Icon widget for displaying graphical icons.

    Features:
    - Size presets or custom dimensions
    - Color tinting
    - Frame-based animation
    - Rotation and flipping
    - Atlas-based icon lookup
    """

    __slots__ = (
        '_name', '_atlas_name', '_atlas_entry',
        '_size_preset', '_width', '_height',
        '_tint', '_opacity',
        '_rotation', '_flip',
        '_animation', '_is_animated',
        '_x', '_y',
        '_visible', '_enabled',
        '_on_click', '_on_animation_complete',
        '_dirty_fields', '_id',
        '_parent', '_children',
    )

    _next_id: int = 0

    def __init__(
        self,
        name: str = "",
        atlas_name: Optional[str] = None,
        size: IconSize = IconSize.MEDIUM,
        width: Optional[float] = None,
        height: Optional[float] = None,
        tint: Optional[str] = None,
        animated: bool = False,
        frame_count: int = 1,
        fps: float = 12.0,
        x: float = 0.0,
        y: float = 0.0,
    ):
        """
        Initialize the icon widget.

        Args:
            name: Icon name in the atlas.
            atlas_name: Specific atlas to use (None = default).
            size: Size preset.
            width: Custom width (overrides preset if specified).
            height: Custom height (overrides preset if specified).
            tint: Color tint as hex string (None = no tint).
            animated: Whether this is an animated icon.
            frame_count: Number of animation frames.
            fps: Animation frames per second.
            x: X position.
            y: Y position.
        """
        self._id = Icon._next_id
        Icon._next_id += 1

        self._name = name
        self._atlas_name = atlas_name
        self._atlas_entry: Optional[IconAtlasEntry] = None

        self._size_preset = size

        # Calculate dimensions
        if width is not None or height is not None:
            self._size_preset = IconSize.CUSTOM
            self._width = width if width is not None else size.value
            self._height = height if height is not None else size.value
        else:
            self._width = float(size.value)
            self._height = float(size.value)

        self._tint = tint
        self._opacity = 1.0

        self._rotation = 0.0  # Degrees
        self._flip = IconFlip.NONE

        # Animation
        self._is_animated = animated
        if animated:
            self._animation = IconAnimation(
                frame_count=frame_count,
                fps=fps,
            )
        else:
            self._animation = None

        self._x = x
        self._y = y

        self._visible = True
        self._enabled = True

        self._on_click: Optional[Callable[[], None]] = None
        self._on_animation_complete: Optional[Callable[[], None]] = None

        self._dirty_fields: set[str] = set()

        self._parent = None
        self._children: list = []

        # Lookup atlas entry
        self._resolve_atlas_entry()

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def id(self) -> int:
        """Get the widget ID."""
        return self._id

    @property
    def name(self) -> str:
        """Get the icon name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the icon name."""
        if self._name != value:
            self._name = value
            self._resolve_atlas_entry()
            self._mark_dirty("name")

    @property
    def atlas_name(self) -> Optional[str]:
        """Get the atlas name."""
        return self._atlas_name

    @atlas_name.setter
    def atlas_name(self, value: Optional[str]) -> None:
        """Set the atlas name."""
        if self._atlas_name != value:
            self._atlas_name = value
            self._resolve_atlas_entry()
            self._mark_dirty("atlas_name")

    @property
    def size_preset(self) -> IconSize:
        """Get the size preset."""
        return self._size_preset

    @size_preset.setter
    def size_preset(self, value: IconSize) -> None:
        """Set the size preset."""
        if self._size_preset != value:
            self._size_preset = value
            if value != IconSize.CUSTOM:
                self._width = float(value.value)
                self._height = float(value.value)
            self._mark_dirty("size_preset")

    @property
    def width(self) -> float:
        """Get the width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set the width."""
        value = max(1.0, value)
        if self._width != value:
            self._width = value
            self._size_preset = IconSize.CUSTOM
            self._mark_dirty("width")

    @property
    def height(self) -> float:
        """Get the height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set the height."""
        value = max(1.0, value)
        if self._height != value:
            self._height = value
            self._size_preset = IconSize.CUSTOM
            self._mark_dirty("height")

    @property
    def tint(self) -> Optional[str]:
        """Get the color tint."""
        return self._tint

    @tint.setter
    def tint(self, value: Optional[str]) -> None:
        """Set the color tint."""
        if self._tint != value:
            self._tint = value
            self._mark_dirty("tint")

    @property
    def opacity(self) -> float:
        """Get the opacity."""
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set the opacity."""
        value = max(0.0, min(1.0, value))
        if self._opacity != value:
            self._opacity = value
            self._mark_dirty("opacity")

    @property
    def rotation(self) -> float:
        """Get the rotation in degrees."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: float) -> None:
        """Set the rotation in degrees."""
        # Normalize to 0-360
        value = value % 360.0
        if self._rotation != value:
            self._rotation = value
            self._mark_dirty("rotation")

    @property
    def flip(self) -> IconFlip:
        """Get the flip mode."""
        return self._flip

    @flip.setter
    def flip(self, value: IconFlip) -> None:
        """Set the flip mode."""
        if self._flip != value:
            self._flip = value
            self._mark_dirty("flip")

    @property
    def x(self) -> float:
        """Get the X position."""
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        """Set the X position."""
        if self._x != value:
            self._x = value
            self._mark_dirty("x")

    @property
    def y(self) -> float:
        """Get the Y position."""
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        """Set the Y position."""
        if self._y != value:
            self._y = value
            self._mark_dirty("y")

    @property
    def visible(self) -> bool:
        """Get visibility."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set visibility."""
        if self._visible != value:
            self._visible = value
            self._mark_dirty("visible")

    @property
    def enabled(self) -> bool:
        """Get enabled state."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        if self._enabled != value:
            self._enabled = value
            self._mark_dirty("enabled")

    @property
    def is_animated(self) -> bool:
        """Check if icon is animated."""
        return self._is_animated and self._animation is not None

    @property
    def animation(self) -> Optional[IconAnimation]:
        """Get the animation configuration."""
        return self._animation

    @property
    def current_frame(self) -> int:
        """Get the current animation frame."""
        if self._animation:
            return self._animation.current_frame
        return 0

    @property
    def atlas_entry(self) -> Optional[IconAtlasEntry]:
        """Get the resolved atlas entry."""
        return self._atlas_entry

    # =========================================================================
    # ANIMATION METHODS
    # =========================================================================

    def update(self, delta_time: float) -> None:
        """
        Update animation state.

        Args:
            delta_time: Time since last update in seconds.
        """
        if not self._is_animated or not self._animation or not self._animation.is_playing:
            return

        anim = self._animation
        anim.elapsed_time += delta_time

        if anim.elapsed_time >= anim.frame_duration:
            anim.elapsed_time -= anim.frame_duration

            # Advance frame
            if anim.ping_pong and anim.is_reversed:
                anim.current_frame -= 1
                if anim.current_frame <= 0:
                    anim.current_frame = 0
                    anim.is_reversed = False
                    if not anim.loop:
                        anim.is_playing = False
                        self._trigger_animation_complete()
            else:
                anim.current_frame += 1
                if anim.current_frame >= anim.frame_count:
                    if anim.ping_pong:
                        anim.current_frame = anim.frame_count - 1
                        anim.is_reversed = True
                    elif anim.loop:
                        anim.current_frame = 0
                    else:
                        anim.current_frame = anim.frame_count - 1
                        anim.is_playing = False
                        self._trigger_animation_complete()

            self._mark_dirty("frame")

    def play(self) -> None:
        """Start or resume animation."""
        if self._animation:
            self._animation.is_playing = True

    def pause(self) -> None:
        """Pause animation."""
        if self._animation:
            self._animation.is_playing = False

    def stop(self) -> None:
        """Stop animation and reset to first frame."""
        if self._animation:
            self._animation.is_playing = False
            self._animation.current_frame = self._animation.start_frame
            self._animation.elapsed_time = 0.0
            self._animation.is_reversed = False
            self._mark_dirty("frame")

    def set_frame(self, frame: int) -> None:
        """
        Set the current frame directly.

        Args:
            frame: Frame index (clamped to valid range).
        """
        if self._animation:
            frame = max(0, min(self._animation.frame_count - 1, frame))
            if self._animation.current_frame != frame:
                self._animation.current_frame = frame
                self._mark_dirty("frame")

    def set_animation(
        self,
        frame_count: int,
        fps: float = 12.0,
        loop: bool = True,
        ping_pong: bool = False,
    ) -> None:
        """
        Configure animation parameters.

        Args:
            frame_count: Number of frames.
            fps: Frames per second.
            loop: Whether to loop.
            ping_pong: Whether to reverse at end.
        """
        self._is_animated = frame_count > 1
        self._animation = IconAnimation(
            frame_count=frame_count,
            fps=fps,
            loop=loop,
            ping_pong=ping_pong,
        )
        self._mark_dirty("animation")

    def remove_animation(self) -> None:
        """Remove animation from this icon."""
        self._is_animated = False
        self._animation = None
        self._mark_dirty("animation")

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_click(self, callback: Optional[Callable[[], None]]) -> None:
        """Set click callback."""
        self._on_click = callback

    def on_animation_complete(self, callback: Optional[Callable[[], None]]) -> None:
        """Set animation complete callback."""
        self._on_animation_complete = callback

    def _trigger_animation_complete(self) -> None:
        """Trigger the animation complete callback."""
        if self._on_animation_complete:
            self._on_animation_complete()

    def handle_click(self) -> bool:
        """
        Handle a click event.

        Returns:
            True if the click was handled.
        """
        if not self._visible or not self._enabled:
            return False

        if self._on_click:
            self._on_click()
            return True
        return False

    # =========================================================================
    # RENDERING HELPERS
    # =========================================================================

    def get_source_rect(self) -> Optional[tuple[int, int, int, int]]:
        """
        Get the source rectangle in the atlas for the current frame.

        Returns:
            Tuple of (x, y, width, height) in atlas pixels, or None if no entry.
        """
        if not self._atlas_entry:
            return None

        entry = self._atlas_entry
        x = entry.x
        y = entry.y

        # Offset for animation frame
        if self._animation and self._animation.frame_count > 1:
            frame = self._animation.current_frame
            x += frame * entry.frame_offset_x
            y += frame * entry.frame_offset_y

        return (x, y, entry.width, entry.height)

    def get_dest_rect(self) -> tuple[float, float, float, float]:
        """
        Get the destination rectangle for rendering.

        Returns:
            Tuple of (x, y, width, height).
        """
        return (self._x, self._y, self._width, self._height)

    def get_center(self) -> tuple[float, float]:
        """Get the center point of the icon."""
        return (self._x + self._width / 2, self._y + self._height / 2)

    def contains_point(self, px: float, py: float) -> bool:
        """
        Check if a point is inside the icon bounds.

        Args:
            px: Point X coordinate.
            py: Point Y coordinate.

        Returns:
            True if point is inside bounds.
        """
        return (
            self._x <= px <= self._x + self._width and
            self._y <= py <= self._y + self._height
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _resolve_atlas_entry(self) -> None:
        """Look up the atlas entry for the current icon name."""
        if self._name:
            manager = IconAtlasManager.get_instance()
            self._atlas_entry = manager.get_icon(self._name, self._atlas_name)
        else:
            self._atlas_entry = None

    def _mark_dirty(self, field_name: str) -> None:
        """Mark a field as dirty."""
        self._dirty_fields.add(field_name)

    def is_dirty(self, field_name: Optional[str] = None) -> bool:
        """Check if a field or any field is dirty."""
        if field_name is None:
            return len(self._dirty_fields) > 0
        return field_name in self._dirty_fields

    def clear_dirty(self) -> None:
        """Clear all dirty flags."""
        self._dirty_fields.clear()

    def clone(self) -> "Icon":
        """Create a copy of this icon."""
        icon = Icon(
            name=self._name,
            atlas_name=self._atlas_name,
            size=self._size_preset,
            width=self._width if self._size_preset == IconSize.CUSTOM else None,
            height=self._height if self._size_preset == IconSize.CUSTOM else None,
            tint=self._tint,
            x=self._x,
            y=self._y,
        )
        icon._opacity = self._opacity
        icon._rotation = self._rotation
        icon._flip = self._flip

        if self._animation:
            icon.set_animation(
                frame_count=self._animation.frame_count,
                fps=self._animation.fps,
                loop=self._animation.loop,
                ping_pong=self._animation.ping_pong,
            )

        return icon

    # =========================================================================
    # CONVENIENCE FACTORY METHODS
    # =========================================================================

    @classmethod
    def small(cls, name: str, **kwargs) -> "Icon":
        """Create a small (16x16) icon."""
        return cls(name=name, size=IconSize.SMALL, **kwargs)

    @classmethod
    def medium(cls, name: str, **kwargs) -> "Icon":
        """Create a medium (24x24) icon."""
        return cls(name=name, size=IconSize.MEDIUM, **kwargs)

    @classmethod
    def large(cls, name: str, **kwargs) -> "Icon":
        """Create a large (32x32) icon."""
        return cls(name=name, size=IconSize.LARGE, **kwargs)

    @classmethod
    def xlarge(cls, name: str, **kwargs) -> "Icon":
        """Create an extra-large (48x48) icon."""
        return cls(name=name, size=IconSize.XLARGE, **kwargs)

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize icon to dictionary."""
        data = {
            "name": self._name,
            "atlas_name": self._atlas_name,
            "size_preset": self._size_preset.name,
            "width": self._width,
            "height": self._height,
            "tint": self._tint,
            "opacity": self._opacity,
            "rotation": self._rotation,
            "flip": self._flip.name,
            "x": self._x,
            "y": self._y,
            "visible": self._visible,
            "enabled": self._enabled,
        }

        if self._animation:
            data["animation"] = {
                "frame_count": self._animation.frame_count,
                "fps": self._animation.fps,
                "loop": self._animation.loop,
                "ping_pong": self._animation.ping_pong,
                "start_frame": self._animation.start_frame,
            }

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Icon":
        """Deserialize icon from dictionary."""
        anim_data = data.get("animation")

        icon = cls(
            name=data.get("name", ""),
            atlas_name=data.get("atlas_name"),
            size=IconSize[data.get("size_preset", "MEDIUM")],
            width=data.get("width"),
            height=data.get("height"),
            tint=data.get("tint"),
            animated=anim_data is not None,
            frame_count=anim_data.get("frame_count", 1) if anim_data else 1,
            fps=anim_data.get("fps", 12.0) if anim_data else 12.0,
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
        )

        icon._opacity = data.get("opacity", 1.0)
        icon._rotation = data.get("rotation", 0.0)
        icon._flip = IconFlip[data.get("flip", "NONE")]
        icon._visible = data.get("visible", True)
        icon._enabled = data.get("enabled", True)

        if anim_data and icon._animation:
            icon._animation.loop = anim_data.get("loop", True)
            icon._animation.ping_pong = anim_data.get("ping_pong", False)
            icon._animation.start_frame = anim_data.get("start_frame", 0)

        return icon

    def __repr__(self) -> str:
        """String representation."""
        size_str = f"{self._width}x{self._height}"
        anim_str = f", animated={self._animation.frame_count}f" if self._animation else ""
        tint_str = f", tint={self._tint}" if self._tint else ""
        return f"Icon(name={self._name!r}, size={size_str}{tint_str}{anim_str})"
