"""
Debug Overlays - Screen overlays with categories, filtering, and persistence.

Provides HUD-style debug information display with customizable positioning,
filtering by category, and configurable visibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, ClassVar, Optional, Any
import threading
import time


class OverlayPosition(Enum):
    """Screen position for overlays."""
    TOP_LEFT = auto()
    TOP_CENTER = auto()
    TOP_RIGHT = auto()
    CENTER_LEFT = auto()
    CENTER = auto()
    CENTER_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_CENTER = auto()
    BOTTOM_RIGHT = auto()
    CUSTOM = auto()


class OverlayVisibility(Enum):
    """Visibility modes for overlays."""
    ALWAYS = auto()        # Always visible
    TOGGLE = auto()        # Toggle with key
    HOVER = auto()         # Show on hover
    CONDITIONAL = auto()   # Show when condition is met
    HIDDEN = auto()        # Never show


@dataclass
class OverlayStyle:
    """Visual style for overlays."""
    background_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.7)
    text_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    border_color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    font_size: float = 12.0
    padding: float = 8.0
    border_width: float = 1.0
    corner_radius: float = 4.0


@dataclass
class OverlayEntry:
    """A single entry in an overlay."""
    key: str
    value: Any
    format_string: str = "{key}: {value}"
    color: Optional[tuple[float, float, float, float]] = None
    priority: int = 0

    def format(self) -> str:
        """Format the entry as a string."""
        return self.format_string.format(key=self.key, value=self.value)


class DebugOverlay(ABC):
    """Base class for debug overlays."""

    __slots__ = (
        '_id',
        '_title',
        '_position',
        '_visibility',
        '_style',
        '_enabled',
        '_category',
        '_entries',
        '_custom_x',
        '_custom_y',
        '_width',
        '_height',
        '_last_update',
        '_update_interval',
        '_condition',
    )

    def __init__(
        self,
        overlay_id: str,
        title: str = "",
        position: OverlayPosition = OverlayPosition.TOP_LEFT,
        visibility: OverlayVisibility = OverlayVisibility.ALWAYS,
        style: Optional[OverlayStyle] = None,
        category: Optional[str] = None,
        update_interval: float = 0.0,
    ):
        self._id = overlay_id
        self._title = title
        self._position = position
        self._visibility = visibility
        self._style = style or OverlayStyle()
        self._enabled = True
        self._category = category
        self._entries: list[OverlayEntry] = []
        self._custom_x = 0.0
        self._custom_y = 0.0
        self._width = 0.0
        self._height = 0.0
        self._last_update = 0.0
        self._update_interval = update_interval
        self._condition: Optional[Callable[[], bool]] = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        self._title = value

    @property
    def position(self) -> OverlayPosition:
        return self._position

    @position.setter
    def position(self, value: OverlayPosition) -> None:
        self._position = value

    @property
    def visibility(self) -> OverlayVisibility:
        return self._visibility

    @visibility.setter
    def visibility(self, value: OverlayVisibility) -> None:
        self._visibility = value

    @property
    def style(self) -> OverlayStyle:
        return self._style

    @style.setter
    def style(self, value: OverlayStyle) -> None:
        self._style = value

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Enable the overlay."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the overlay."""
        self._enabled = False

    def toggle(self) -> None:
        """Toggle the overlay enabled state."""
        self._enabled = not self._enabled

    @property
    def category(self) -> Optional[str]:
        return self._category

    @category.setter
    def category(self, value: Optional[str]) -> None:
        self._category = value

    def set_custom_position(self, x: float, y: float) -> None:
        """Set custom screen position."""
        self._position = OverlayPosition.CUSTOM
        self._custom_x = x
        self._custom_y = y

    def get_custom_position(self) -> tuple[float, float]:
        """Get custom position."""
        return (self._custom_x, self._custom_y)

    def set_condition(self, condition: Callable[[], bool]) -> None:
        """Set visibility condition."""
        self._condition = condition
        self._visibility = OverlayVisibility.CONDITIONAL

    def is_visible(self) -> bool:
        """Check if overlay should be visible."""
        if not self._enabled:
            return False

        if self._visibility == OverlayVisibility.HIDDEN:
            return False
        elif self._visibility == OverlayVisibility.ALWAYS:
            return True
        elif self._visibility == OverlayVisibility.TOGGLE:
            return True  # Managed by toggle state
        elif self._visibility == OverlayVisibility.CONDITIONAL:
            if self._condition:
                return self._condition()
            return False
        elif self._visibility == OverlayVisibility.HOVER:
            return False  # Managed by hover detection

        return True

    def add_entry(
        self,
        key: str,
        value: Any,
        format_string: str = "{key}: {value}",
        color: Optional[tuple[float, float, float, float]] = None,
        priority: int = 0,
    ) -> OverlayEntry:
        """Add an entry to the overlay."""
        entry = OverlayEntry(
            key=key,
            value=value,
            format_string=format_string,
            color=color,
            priority=priority,
        )
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e.priority, reverse=True)
        return entry

    def update_entry(self, key: str, value: Any) -> bool:
        """Update an existing entry's value."""
        for entry in self._entries:
            if entry.key == key:
                entry.value = value
                return True
        return False

    def remove_entry(self, key: str) -> bool:
        """Remove an entry by key."""
        for i, entry in enumerate(self._entries):
            if entry.key == key:
                self._entries.pop(i)
                return True
        return False

    def clear_entries(self) -> None:
        """Clear all entries."""
        self._entries.clear()

    def get_entry(self, key: str) -> Optional[OverlayEntry]:
        """Get an entry by key."""
        for entry in self._entries:
            if entry.key == key:
                return entry
        return None

    @property
    def entries(self) -> list[OverlayEntry]:
        return self._entries.copy()

    def should_update(self, current_time: float) -> bool:
        """Check if overlay should update based on interval."""
        if self._update_interval <= 0:
            return True
        if (current_time - self._last_update) >= self._update_interval:
            self._last_update = current_time
            return True
        return False

    @abstractmethod
    def update(self) -> None:
        """Update the overlay content."""
        pass

    @abstractmethod
    def render(self) -> dict[str, Any]:
        """Render the overlay and return render data."""
        pass


class TextOverlay(DebugOverlay):
    """Simple text-based overlay."""

    def update(self) -> None:
        """Update is a no-op for static text overlays."""
        pass

    def render(self) -> dict[str, Any]:
        """Render text overlay."""
        lines = []
        if self._title:
            lines.append(self._title)
            lines.append("-" * len(self._title))

        for entry in self._entries:
            lines.append(entry.format())

        return {
            "type": "text",
            "id": self._id,
            "position": self._position.name,
            "custom_position": (self._custom_x, self._custom_y),
            "style": {
                "background": self._style.background_color,
                "text_color": self._style.text_color,
                "border_color": self._style.border_color,
                "font_size": self._style.font_size,
                "padding": self._style.padding,
            },
            "lines": lines,
            "visible": self.is_visible(),
        }


class StatsOverlay(DebugOverlay):
    """Overlay showing statistics with automatic updates."""

    __slots__ = ('_stat_providers',)

    def __init__(
        self,
        overlay_id: str,
        title: str = "Stats",
        position: OverlayPosition = OverlayPosition.TOP_LEFT,
        update_interval: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            overlay_id=overlay_id,
            title=title,
            position=position,
            update_interval=update_interval,
            **kwargs,
        )
        self._stat_providers: dict[str, Callable[[], Any]] = {}

    def register_stat(
        self,
        key: str,
        provider: Callable[[], Any],
        format_string: str = "{key}: {value}",
        priority: int = 0,
    ) -> None:
        """Register a stat provider function."""
        self._stat_providers[key] = provider
        self.add_entry(key, None, format_string=format_string, priority=priority)

    def unregister_stat(self, key: str) -> None:
        """Unregister a stat provider."""
        self._stat_providers.pop(key, None)
        self.remove_entry(key)

    def update(self) -> None:
        """Update all stats from providers."""
        for key, provider in self._stat_providers.items():
            try:
                value = provider()
                self.update_entry(key, value)
            except Exception:
                self.update_entry(key, "N/A")

    def render(self) -> dict[str, Any]:
        """Render stats overlay."""
        lines = []
        if self._title:
            lines.append(self._title)
            lines.append("-" * len(self._title))

        for entry in self._entries:
            lines.append(entry.format())

        return {
            "type": "stats",
            "id": self._id,
            "position": self._position.name,
            "custom_position": (self._custom_x, self._custom_y),
            "style": {
                "background": self._style.background_color,
                "text_color": self._style.text_color,
                "border_color": self._style.border_color,
                "font_size": self._style.font_size,
                "padding": self._style.padding,
            },
            "lines": lines,
            "visible": self.is_visible(),
        }


class GraphOverlay(DebugOverlay):
    """Overlay showing a graph of values over time."""

    __slots__ = ('_data_points', '_max_points', '_min_value', '_max_value', '_auto_scale')

    def __init__(
        self,
        overlay_id: str,
        title: str = "Graph",
        position: OverlayPosition = OverlayPosition.TOP_RIGHT,
        max_points: int = 100,
        min_value: float = 0.0,
        max_value: float = 100.0,
        auto_scale: bool = True,
        **kwargs,
    ):
        super().__init__(
            overlay_id=overlay_id,
            title=title,
            position=position,
            **kwargs,
        )
        self._data_points: list[float] = []
        self._max_points = max_points
        self._min_value = min_value
        self._max_value = max_value
        self._auto_scale = auto_scale

    def add_data_point(self, value: float) -> None:
        """Add a data point to the graph."""
        self._data_points.append(value)
        if len(self._data_points) > self._max_points:
            self._data_points.pop(0)

        if self._auto_scale and self._data_points:
            self._min_value = min(self._data_points)
            self._max_value = max(self._data_points)
            # Prevent division by zero
            if self._min_value == self._max_value:
                self._max_value = self._min_value + 1.0

    def clear_data(self) -> None:
        """Clear all data points."""
        self._data_points.clear()

    @property
    def data_points(self) -> list[float]:
        return self._data_points.copy()

    @property
    def current_value(self) -> Optional[float]:
        """Get the most recent value."""
        if self._data_points:
            return self._data_points[-1]
        return None

    @property
    def average_value(self) -> Optional[float]:
        """Get the average value."""
        if self._data_points:
            return sum(self._data_points) / len(self._data_points)
        return None

    def update(self) -> None:
        """Update is handled by add_data_point."""
        pass

    def render(self) -> dict[str, Any]:
        """Render graph overlay."""
        return {
            "type": "graph",
            "id": self._id,
            "title": self._title,
            "position": self._position.name,
            "custom_position": (self._custom_x, self._custom_y),
            "style": {
                "background": self._style.background_color,
                "text_color": self._style.text_color,
                "border_color": self._style.border_color,
            },
            "data": self._data_points,
            "min_value": self._min_value,
            "max_value": self._max_value,
            "visible": self.is_visible(),
        }


class OverlayManager:
    """Manages multiple debug overlays."""

    _instance: ClassVar[Optional["OverlayManager"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    __slots__ = (
        '_overlays',
        '_enabled',
        '_enabled_categories',
        '_current_time',
    )

    def __init__(self):
        self._overlays: dict[str, DebugOverlay] = {}
        self._enabled = True
        self._enabled_categories: set[str] = set()
        self._current_time = time.time()

    @classmethod
    def get_instance(cls) -> "OverlayManager":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def enable(self) -> None:
        """Enable all overlays."""
        self._enabled = True

    def disable(self) -> None:
        """Disable all overlays."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def register_overlay(self, overlay: DebugOverlay) -> None:
        """Register an overlay."""
        self._overlays[overlay.id] = overlay
        if overlay.category:
            self._enabled_categories.add(overlay.category)

    def unregister_overlay(self, overlay_id: str) -> Optional[DebugOverlay]:
        """Unregister and return an overlay."""
        return self._overlays.pop(overlay_id, None)

    def get_overlay(self, overlay_id: str) -> Optional[DebugOverlay]:
        """Get an overlay by ID."""
        return self._overlays.get(overlay_id)

    def get_overlays_by_category(self, category: str) -> list[DebugOverlay]:
        """Get all overlays in a category."""
        return [o for o in self._overlays.values() if o.category == category]

    def enable_category(self, category: str) -> None:
        """Enable a category of overlays."""
        self._enabled_categories.add(category)
        for overlay in self._overlays.values():
            if overlay.category == category:
                overlay.enable()

    def disable_category(self, category: str) -> None:
        """Disable a category of overlays."""
        self._enabled_categories.discard(category)
        for overlay in self._overlays.values():
            if overlay.category == category:
                overlay.disable()

    def is_category_enabled(self, category: str) -> bool:
        """Check if a category is enabled."""
        return category in self._enabled_categories

    def toggle_overlay(self, overlay_id: str) -> bool:
        """Toggle an overlay's visibility. Returns new state."""
        overlay = self._overlays.get(overlay_id)
        if overlay:
            overlay.toggle()
            return overlay.enabled
        return False

    def update(self) -> None:
        """Update all overlays."""
        self._current_time = time.time()
        if not self._enabled:
            return

        for overlay in self._overlays.values():
            if overlay.enabled and overlay.should_update(self._current_time):
                overlay.update()

    def render_all(self) -> list[dict[str, Any]]:
        """Render all visible overlays."""
        if not self._enabled:
            return []

        render_data = []
        for overlay in self._overlays.values():
            if overlay.is_visible():
                render_data.append(overlay.render())

        return render_data

    def clear_all(self) -> None:
        """Remove all overlays."""
        self._overlays.clear()

    @property
    def overlay_count(self) -> int:
        return len(self._overlays)

    @property
    def visible_count(self) -> int:
        return sum(1 for o in self._overlays.values() if o.is_visible())

    def create_text_overlay(
        self,
        overlay_id: str,
        title: str = "",
        position: OverlayPosition = OverlayPosition.TOP_LEFT,
        **kwargs,
    ) -> TextOverlay:
        """Create and register a text overlay."""
        overlay = TextOverlay(
            overlay_id=overlay_id,
            title=title,
            position=position,
            **kwargs,
        )
        self.register_overlay(overlay)
        return overlay

    def create_stats_overlay(
        self,
        overlay_id: str,
        title: str = "Stats",
        position: OverlayPosition = OverlayPosition.TOP_LEFT,
        **kwargs,
    ) -> StatsOverlay:
        """Create and register a stats overlay."""
        overlay = StatsOverlay(
            overlay_id=overlay_id,
            title=title,
            position=position,
            **kwargs,
        )
        self.register_overlay(overlay)
        return overlay

    def create_graph_overlay(
        self,
        overlay_id: str,
        title: str = "Graph",
        position: OverlayPosition = OverlayPosition.TOP_RIGHT,
        **kwargs,
    ) -> GraphOverlay:
        """Create and register a graph overlay."""
        overlay = GraphOverlay(
            overlay_id=overlay_id,
            title=title,
            position=position,
            **kwargs,
        )
        self.register_overlay(overlay)
        return overlay


# Built-in overlays
class FPSOverlay(StatsOverlay):
    """FPS counter overlay."""

    __slots__ = ('_frame_times', '_last_frame_time')

    def __init__(
        self,
        overlay_id: str = "fps_overlay",
        position: OverlayPosition = OverlayPosition.TOP_RIGHT,
        **kwargs,
    ):
        super().__init__(
            overlay_id=overlay_id,
            title="FPS",
            position=position,
            update_interval=0.5,
            **kwargs,
        )
        self._frame_times: list[float] = []
        self._last_frame_time = time.time()

        self.register_stat("FPS", self._get_fps, "{key}: {value:.1f}")
        self.register_stat("Frame Time", self._get_frame_time, "{key}: {value:.2f}ms")

    def record_frame(self) -> None:
        """Record a frame time."""
        current_time = time.time()
        dt = current_time - self._last_frame_time
        self._last_frame_time = current_time

        self._frame_times.append(dt)
        if len(self._frame_times) > 60:
            self._frame_times.pop(0)

    def _get_fps(self) -> float:
        if not self._frame_times:
            return 0.0
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    def _get_frame_time(self) -> float:
        if not self._frame_times:
            return 0.0
        return (sum(self._frame_times) / len(self._frame_times)) * 1000.0


class MemoryOverlay(StatsOverlay):
    """Memory usage overlay."""

    def __init__(
        self,
        overlay_id: str = "memory_overlay",
        position: OverlayPosition = OverlayPosition.TOP_LEFT,
        **kwargs,
    ):
        super().__init__(
            overlay_id=overlay_id,
            title="Memory",
            position=position,
            update_interval=1.0,
            **kwargs,
        )

        self.register_stat("Used", self._get_used_memory, "{key}: {value}")
        self.register_stat("Available", self._get_available_memory, "{key}: {value}")

    def _get_used_memory(self) -> str:
        try:
            import psutil
            mem = psutil.Process().memory_info()
            return f"{mem.rss / (1024 * 1024):.1f} MB"
        except ImportError:
            return "N/A"

    def _get_available_memory(self) -> str:
        try:
            import psutil
            mem = psutil.virtual_memory()
            return f"{mem.available / (1024 * 1024 * 1024):.1f} GB"
        except ImportError:
            return "N/A"
