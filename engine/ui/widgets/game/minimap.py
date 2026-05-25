"""
Minimap widget for world overview and navigation.

Provides a minimap widget supporting:
- Map texture rendering
- Player position marker
- Entity markers (enemies, allies, objectives)
- Zoom level control
- Click-to-navigate functionality
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


class MarkerType(Enum):
    """Types of markers on the minimap."""
    PLAYER = auto()
    ALLY = auto()
    ENEMY = auto()
    NEUTRAL = auto()
    OBJECTIVE = auto()
    WAYPOINT = auto()
    QUEST = auto()
    POINT_OF_INTEREST = auto()
    CUSTOM = auto()


@dataclass(slots=True)
class MinimapMarker:
    """A marker on the minimap."""
    id: int
    marker_type: MarkerType
    world_x: float
    world_y: float
    rotation: float = 0.0  # Rotation in degrees
    icon: Optional[str] = None  # Icon path or identifier
    color: str = "#ffffff"
    size: float = 8.0
    label: Optional[str] = None
    is_visible: bool = True
    is_pulsing: bool = False
    custom_data: dict = field(default_factory=dict)

    # Computed minimap position (set by Minimap)
    map_x: float = 0.0
    map_y: float = 0.0


@dataclass(slots=True)
class MinimapConfig:
    """Configuration for minimap appearance and behavior."""
    # Appearance
    background_color: str = "#1f2937"
    border_color: str = "#374151"
    border_width: float = 2.0
    corner_radius: float = 0.0  # 0 for square, >0 for rounded, -1 for circle
    mask_shape: str = "rectangle"  # "rectangle", "circle", "rounded"

    # Player marker
    player_color: str = "#22c55e"
    player_size: float = 12.0
    player_icon: Optional[str] = None
    show_player_direction: bool = True

    # Default marker colors by type
    ally_color: str = "#3b82f6"
    enemy_color: str = "#ef4444"
    neutral_color: str = "#9ca3af"
    objective_color: str = "#f59e0b"
    quest_color: str = "#8b5cf6"
    poi_color: str = "#06b6d4"

    # Zoom
    min_zoom: float = 0.5
    max_zoom: float = 4.0
    default_zoom: float = 1.0
    zoom_step: float = 0.25

    # Interaction
    allow_click_navigation: bool = True
    allow_zoom: bool = True
    allow_pan: bool = False
    show_coordinates: bool = False

    # Markers
    max_visible_markers: int = 100
    marker_fade_distance: float = 0.0  # 0 = no fade
    cluster_nearby_markers: bool = False
    cluster_distance: float = 10.0


class Minimap:
    """Minimap widget for world overview.

    Features:
    - Map texture rendering with pan/zoom
    - Player position tracking with rotation
    - Multiple entity markers with customization
    - Click-to-navigate functionality
    - Configurable appearance and behavior
    """

    __slots__ = (
        '_id', '_x', '_y', '_width', '_height',
        '_world_width', '_world_height',
        '_center_x', '_center_y', '_zoom',
        '_rotation', '_follow_player',
        '_map_texture', '_map_texture_path',
        '_markers', '_next_marker_id',
        '_player_marker_id',
        '_config',
        '_is_visible', '_is_interactive',
        '_on_click', '_on_navigate', '_on_marker_click',
        '_parent', '_children',
        '_is_dragging', '_drag_start_x', '_drag_start_y',
    )

    _next_id: int = 0

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 200.0,
        height: float = 200.0,
        world_width: float = 1000.0,
        world_height: float = 1000.0,
        config: Optional[MinimapConfig] = None,
    ):
        """Initialize the minimap.

        Args:
            x: X position
            y: Y position
            width: Widget width in pixels
            height: Widget height in pixels
            world_width: World size X
            world_height: World size Y
            config: Minimap configuration
        """
        self._id = Minimap._next_id
        Minimap._next_id += 1

        self._x = x
        self._y = y
        self._width = max(1.0, width)
        self._height = max(1.0, height)

        self._world_width = max(1.0, world_width)
        self._world_height = max(1.0, world_height)

        self._center_x = world_width / 2
        self._center_y = world_height / 2
        self._zoom = config.default_zoom if config else 1.0
        self._rotation = 0.0
        self._follow_player = True

        self._map_texture = None
        self._map_texture_path: Optional[str] = None

        self._markers: dict[int, MinimapMarker] = {}
        self._next_marker_id = 0
        self._player_marker_id: Optional[int] = None

        self._config = config or MinimapConfig()

        self._is_visible = True
        self._is_interactive = True

        self._on_click: Optional[Callable[[float, float], None]] = None
        self._on_navigate: Optional[Callable[[float, float], None]] = None
        self._on_marker_click: Optional[Callable[[MinimapMarker], None]] = None

        self._parent = None
        self._children: list = []

        self._is_dragging = False
        self._drag_start_x = 0.0
        self._drag_start_y = 0.0

    # Properties
    @property
    def id(self) -> int:
        """Get widget ID."""
        return self._id

    @property
    def x(self) -> float:
        """Get X position."""
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        """Set X position."""
        self._x = value

    @property
    def y(self) -> float:
        """Get Y position."""
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        """Set Y position."""
        self._y = value

    @property
    def width(self) -> float:
        """Get width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set width."""
        self._width = max(1.0, value)

    @property
    def height(self) -> float:
        """Get height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set height."""
        self._height = max(1.0, value)

    @property
    def world_width(self) -> float:
        """Get world width."""
        return self._world_width

    @property
    def world_height(self) -> float:
        """Get world height."""
        return self._world_height

    @property
    def center_x(self) -> float:
        """Get map center X in world coordinates."""
        return self._center_x

    @property
    def center_y(self) -> float:
        """Get map center Y in world coordinates."""
        return self._center_y

    @property
    def zoom(self) -> float:
        """Get current zoom level."""
        return self._zoom

    @zoom.setter
    def zoom(self, value: float) -> None:
        """Set zoom level."""
        self._zoom = max(
            self._config.min_zoom,
            min(self._config.max_zoom, value)
        )
        self._update_marker_positions()

    @property
    def rotation(self) -> float:
        """Get map rotation in degrees."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: float) -> None:
        """Set map rotation."""
        self._rotation = value % 360
        self._update_marker_positions()

    @property
    def follow_player(self) -> bool:
        """Check if map follows player."""
        return self._follow_player

    @follow_player.setter
    def follow_player(self, value: bool) -> None:
        """Set player follow mode."""
        self._follow_player = value

    @property
    def config(self) -> MinimapConfig:
        """Get configuration."""
        return self._config

    @property
    def is_visible(self) -> bool:
        """Check if minimap is visible."""
        return self._is_visible

    @is_visible.setter
    def is_visible(self, value: bool) -> None:
        """Set visibility."""
        self._is_visible = value

    @property
    def is_interactive(self) -> bool:
        """Check if minimap is interactive."""
        return self._is_interactive

    @is_interactive.setter
    def is_interactive(self, value: bool) -> None:
        """Set interactivity."""
        self._is_interactive = value

    @property
    def marker_count(self) -> int:
        """Get number of markers."""
        return len(self._markers)

    @property
    def visible_markers(self) -> list[MinimapMarker]:
        """Get list of visible markers."""
        return [m for m in self._markers.values() if m.is_visible]

    # Map texture
    def set_map_texture(self, texture_path: str) -> None:
        """Set the map background texture.

        Args:
            texture_path: Path to texture file
        """
        self._map_texture_path = texture_path
        # Actual texture loading handled by renderer

    def get_map_texture_path(self) -> Optional[str]:
        """Get the map texture path."""
        return self._map_texture_path

    # World bounds
    def set_world_bounds(self, width: float, height: float) -> None:
        """Set world dimensions.

        Args:
            width: World width
            height: World height
        """
        self._world_width = max(1.0, width)
        self._world_height = max(1.0, height)
        self._update_marker_positions()

    # View control
    def set_center(self, world_x: float, world_y: float) -> None:
        """Set map center position.

        Args:
            world_x: World X coordinate
            world_y: World Y coordinate
        """
        self._center_x = max(0, min(self._world_width, world_x))
        self._center_y = max(0, min(self._world_height, world_y))
        self._update_marker_positions()

    def zoom_in(self) -> None:
        """Zoom in by one step."""
        self.zoom = self._zoom + self._config.zoom_step

    def zoom_out(self) -> None:
        """Zoom out by one step."""
        self.zoom = self._zoom - self._config.zoom_step

    def reset_zoom(self) -> None:
        """Reset zoom to default."""
        self.zoom = self._config.default_zoom

    def pan(self, dx: float, dy: float) -> None:
        """Pan the map view.

        Args:
            dx: Delta X in world units
            dy: Delta Y in world units
        """
        if not self._config.allow_pan:
            return

        self._center_x = max(0, min(self._world_width, self._center_x + dx))
        self._center_y = max(0, min(self._world_height, self._center_y + dy))
        self._update_marker_positions()

    # Marker management
    def add_marker(
        self,
        marker_type: MarkerType,
        world_x: float,
        world_y: float,
        **kwargs
    ) -> int:
        """Add a marker to the minimap.

        Args:
            marker_type: Type of marker
            world_x: World X position
            world_y: World Y position
            **kwargs: Additional marker properties

        Returns:
            Marker ID
        """
        marker_id = self._next_marker_id
        self._next_marker_id += 1

        # Set default color based on type
        if "color" not in kwargs:
            kwargs["color"] = self._get_default_marker_color(marker_type)

        marker = MinimapMarker(
            id=marker_id,
            marker_type=marker_type,
            world_x=world_x,
            world_y=world_y,
            **kwargs
        )

        self._markers[marker_id] = marker
        self._update_marker_position(marker)

        return marker_id

    def remove_marker(self, marker_id: int) -> bool:
        """Remove a marker.

        Args:
            marker_id: Marker ID

        Returns:
            True if marker was removed
        """
        if marker_id in self._markers:
            del self._markers[marker_id]
            return True
        return False

    def get_marker(self, marker_id: int) -> Optional[MinimapMarker]:
        """Get a marker by ID.

        Args:
            marker_id: Marker ID

        Returns:
            Marker if found
        """
        return self._markers.get(marker_id)

    def update_marker(
        self,
        marker_id: int,
        world_x: Optional[float] = None,
        world_y: Optional[float] = None,
        rotation: Optional[float] = None,
        **kwargs
    ) -> bool:
        """Update marker properties.

        Args:
            marker_id: Marker ID
            world_x: New X position
            world_y: New Y position
            rotation: New rotation
            **kwargs: Other properties to update

        Returns:
            True if marker was updated
        """
        marker = self._markers.get(marker_id)
        if not marker:
            return False

        if world_x is not None:
            marker.world_x = world_x
        if world_y is not None:
            marker.world_y = world_y
        if rotation is not None:
            marker.rotation = rotation

        for key, value in kwargs.items():
            if hasattr(marker, key):
                setattr(marker, key, value)

        self._update_marker_position(marker)
        return True

    def clear_markers(self, marker_type: Optional[MarkerType] = None) -> int:
        """Clear markers.

        Args:
            marker_type: Type to clear, or None for all

        Returns:
            Number of markers removed
        """
        if marker_type is None:
            count = len(self._markers)
            self._markers.clear()
            self._player_marker_id = None
            return count

        to_remove = [
            mid for mid, m in self._markers.items()
            if m.marker_type == marker_type
        ]

        for mid in to_remove:
            if mid == self._player_marker_id:
                self._player_marker_id = None
            del self._markers[mid]

        return len(to_remove)

    # Player tracking
    def set_player_position(
        self,
        world_x: float,
        world_y: float,
        rotation: float = 0.0
    ) -> None:
        """Update player position.

        Args:
            world_x: World X position
            world_y: World Y position
            rotation: Player rotation in degrees
        """
        if self._player_marker_id is None:
            self._player_marker_id = self.add_marker(
                MarkerType.PLAYER,
                world_x,
                world_y,
                rotation=rotation,
                color=self._config.player_color,
                size=self._config.player_size,
                icon=self._config.player_icon,
            )
        else:
            self.update_marker(
                self._player_marker_id,
                world_x=world_x,
                world_y=world_y,
                rotation=rotation,
            )

        if self._follow_player:
            self.set_center(world_x, world_y)

    def get_player_position(self) -> Optional[tuple[float, float]]:
        """Get player world position.

        Returns:
            (x, y) tuple or None
        """
        if self._player_marker_id is not None:
            marker = self._markers.get(self._player_marker_id)
            if marker:
                return (marker.world_x, marker.world_y)
        return None

    # Coordinate conversion
    def world_to_map(self, world_x: float, world_y: float) -> tuple[float, float]:
        """Convert world coordinates to minimap coordinates.

        Args:
            world_x: World X position
            world_y: World Y position

        Returns:
            (map_x, map_y) in minimap widget space
        """
        # Calculate visible world area based on zoom
        visible_width = self._world_width / self._zoom
        visible_height = self._world_height / self._zoom

        # Calculate offset from center
        offset_x = world_x - self._center_x
        offset_y = world_y - self._center_y

        # Apply rotation if needed
        if self._rotation != 0:
            rad = math.radians(-self._rotation)
            cos_r = math.cos(rad)
            sin_r = math.sin(rad)
            new_x = offset_x * cos_r - offset_y * sin_r
            new_y = offset_x * sin_r + offset_y * cos_r
            offset_x, offset_y = new_x, new_y

        # Convert to minimap space
        map_x = self._x + self._width / 2 + (offset_x / visible_width) * self._width
        map_y = self._y + self._height / 2 + (offset_y / visible_height) * self._height

        return (map_x, map_y)

    def map_to_world(self, map_x: float, map_y: float) -> tuple[float, float]:
        """Convert minimap coordinates to world coordinates.

        Args:
            map_x: Minimap X position
            map_y: Minimap Y position

        Returns:
            (world_x, world_y) in world space
        """
        # Calculate visible world area based on zoom
        visible_width = self._world_width / self._zoom
        visible_height = self._world_height / self._zoom

        # Convert from minimap space to offset
        offset_x = ((map_x - self._x) / self._width - 0.5) * visible_width
        offset_y = ((map_y - self._y) / self._height - 0.5) * visible_height

        # Reverse rotation if needed
        if self._rotation != 0:
            rad = math.radians(self._rotation)
            cos_r = math.cos(rad)
            sin_r = math.sin(rad)
            new_x = offset_x * cos_r - offset_y * sin_r
            new_y = offset_x * sin_r + offset_y * cos_r
            offset_x, offset_y = new_x, new_y

        # Add center offset
        world_x = self._center_x + offset_x
        world_y = self._center_y + offset_y

        return (world_x, world_y)

    def is_world_position_visible(self, world_x: float, world_y: float) -> bool:
        """Check if world position is visible on minimap.

        Args:
            world_x: World X position
            world_y: World Y position

        Returns:
            True if position is within visible area
        """
        map_x, map_y = self.world_to_map(world_x, world_y)
        return (
            self._x <= map_x <= self._x + self._width and
            self._y <= map_y <= self._y + self._height
        )

    # Input handling
    def handle_click(self, map_x: float, map_y: float) -> bool:
        """Handle click on minimap.

        Args:
            map_x: Click X in widget space
            map_y: Click Y in widget space

        Returns:
            True if click was handled
        """
        if not self._is_interactive:
            return False

        # Check bounds
        if not self._point_in_bounds(map_x, map_y):
            return False

        # Check for marker clicks first
        clicked_marker = self._get_marker_at_position(map_x, map_y)
        if clicked_marker and self._on_marker_click:
            self._on_marker_click(clicked_marker)
            return True

        # Convert to world coordinates
        world_x, world_y = self.map_to_world(map_x, map_y)

        # Fire click callback
        if self._on_click:
            self._on_click(world_x, world_y)

        # Fire navigate callback if enabled
        if self._config.allow_click_navigation and self._on_navigate:
            self._on_navigate(world_x, world_y)

        return True

    def handle_scroll(self, delta: float) -> bool:
        """Handle scroll input for zoom.

        Args:
            delta: Scroll delta (positive = zoom in)

        Returns:
            True if scroll was handled
        """
        if not self._is_interactive or not self._config.allow_zoom:
            return False

        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

        return True

    def handle_drag_start(self, map_x: float, map_y: float) -> bool:
        """Handle drag start for panning.

        Args:
            map_x: Start X position
            map_y: Start Y position

        Returns:
            True if drag started
        """
        if not self._is_interactive or not self._config.allow_pan:
            return False

        if not self._point_in_bounds(map_x, map_y):
            return False

        self._is_dragging = True
        self._drag_start_x = map_x
        self._drag_start_y = map_y
        self._follow_player = False  # Disable follow when manually panning

        return True

    def handle_drag(self, map_x: float, map_y: float) -> bool:
        """Handle drag movement.

        Args:
            map_x: Current X position
            map_y: Current Y position

        Returns:
            True if drag was handled
        """
        if not self._is_dragging:
            return False

        # Calculate world delta
        visible_width = self._world_width / self._zoom
        visible_height = self._world_height / self._zoom

        dx = ((self._drag_start_x - map_x) / self._width) * visible_width
        dy = ((self._drag_start_y - map_y) / self._height) * visible_height

        self.pan(dx, dy)

        self._drag_start_x = map_x
        self._drag_start_y = map_y

        return True

    def handle_drag_end(self) -> None:
        """Handle drag end."""
        self._is_dragging = False

    # Callbacks
    def on_click(self, callback: Callable[[float, float], None]) -> None:
        """Set click callback.

        Args:
            callback: Function(world_x, world_y)
        """
        self._on_click = callback

    def on_navigate(self, callback: Callable[[float, float], None]) -> None:
        """Set navigation callback.

        Args:
            callback: Function(world_x, world_y)
        """
        self._on_navigate = callback

    def on_marker_click(self, callback: Callable[[MinimapMarker], None]) -> None:
        """Set marker click callback.

        Args:
            callback: Function(marker)
        """
        self._on_marker_click = callback

    # Update
    def update(self, delta_time: float) -> None:
        """Update minimap state.

        Args:
            delta_time: Time since last update
        """
        # Update marker positions if following player
        if self._follow_player and self._player_marker_id is not None:
            self._update_marker_positions()

    # Rendering helpers
    def get_visible_bounds(self) -> tuple[float, float, float, float]:
        """Get visible world bounds.

        Returns:
            (min_x, min_y, max_x, max_y) in world space
        """
        visible_width = self._world_width / self._zoom
        visible_height = self._world_height / self._zoom

        min_x = self._center_x - visible_width / 2
        min_y = self._center_y - visible_height / 2
        max_x = self._center_x + visible_width / 2
        max_y = self._center_y + visible_height / 2

        return (min_x, min_y, max_x, max_y)

    def get_map_uv_bounds(self) -> tuple[float, float, float, float]:
        """Get UV coordinates for map texture rendering.

        Returns:
            (u_min, v_min, u_max, v_max)
        """
        min_x, min_y, max_x, max_y = self.get_visible_bounds()

        u_min = max(0, min_x / self._world_width)
        v_min = max(0, min_y / self._world_height)
        u_max = min(1, max_x / self._world_width)
        v_max = min(1, max_y / self._world_height)

        return (u_min, v_min, u_max, v_max)

    # Private methods
    def _point_in_bounds(self, map_x: float, map_y: float) -> bool:
        """Check if point is within minimap bounds."""
        return (
            self._x <= map_x <= self._x + self._width and
            self._y <= map_y <= self._y + self._height
        )

    def _get_marker_at_position(
        self,
        map_x: float,
        map_y: float
    ) -> Optional[MinimapMarker]:
        """Get marker at minimap position."""
        for marker in self._markers.values():
            if not marker.is_visible:
                continue

            # Check distance from marker center
            dx = map_x - marker.map_x
            dy = map_y - marker.map_y
            dist_sq = dx * dx + dy * dy
            hit_radius = marker.size / 2 + 2  # Small tolerance

            if dist_sq <= hit_radius * hit_radius:
                return marker

        return None

    def _update_marker_positions(self) -> None:
        """Update all marker minimap positions."""
        for marker in self._markers.values():
            self._update_marker_position(marker)

    def _update_marker_position(self, marker: MinimapMarker) -> None:
        """Update a single marker's minimap position."""
        marker.map_x, marker.map_y = self.world_to_map(
            marker.world_x,
            marker.world_y
        )

    def _get_default_marker_color(self, marker_type: MarkerType) -> str:
        """Get default color for marker type."""
        colors = {
            MarkerType.PLAYER: self._config.player_color,
            MarkerType.ALLY: self._config.ally_color,
            MarkerType.ENEMY: self._config.enemy_color,
            MarkerType.NEUTRAL: self._config.neutral_color,
            MarkerType.OBJECTIVE: self._config.objective_color,
            MarkerType.WAYPOINT: self._config.objective_color,
            MarkerType.QUEST: self._config.quest_color,
            MarkerType.POINT_OF_INTEREST: self._config.poi_color,
            MarkerType.CUSTOM: "#ffffff",
        }
        return colors.get(marker_type, "#ffffff")

    def __repr__(self) -> str:
        return (
            f"Minimap(id={self._id}, "
            f"size={self._width}x{self._height}, "
            f"world={self._world_width}x{self._world_height}, "
            f"markers={len(self._markers)})"
        )
