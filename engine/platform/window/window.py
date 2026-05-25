"""
Window management for the game engine.

Provides window creation, lifecycle management, and event handling
with a headless backend for testing without a display server.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar
import threading

from ..constants import (
    DEFAULT_WINDOW_X, DEFAULT_WINDOW_Y,
    DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT,
    DEFAULT_DPI_SCALE
)


class WindowStyle(Enum):
    """Window display style."""
    WINDOWED = auto()
    BORDERLESS = auto()
    FULLSCREEN_EXCLUSIVE = auto()


class FullscreenMode(Enum):
    """Fullscreen mode options."""
    WINDOWED = auto()
    BORDERLESS = auto()
    EXCLUSIVE = auto()


class WindowState(Enum):
    """Window state."""
    NORMAL = auto()
    MINIMIZED = auto()
    MAXIMIZED = auto()
    HIDDEN = auto()


class WindowEventType(Enum):
    """Window event types."""
    CLOSE = auto()
    RESIZE = auto()
    MOVE = auto()
    FOCUS = auto()
    BLUR = auto()
    MINIMIZE = auto()
    MAXIMIZE = auto()
    RESTORE = auto()
    KEY_DOWN = auto()
    KEY_UP = auto()
    MOUSE_MOVE = auto()
    MOUSE_BUTTON = auto()
    MOUSE_SCROLL = auto()


@dataclass(slots=True)
class Rect:
    """Rectangle representing position and size."""
    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class WindowConfig:
    """Configuration for window creation."""
    title: str = "Game Window"
    x: int = DEFAULT_WINDOW_X
    y: int = DEFAULT_WINDOW_Y
    width: int = DEFAULT_WINDOW_WIDTH
    height: int = DEFAULT_WINDOW_HEIGHT
    resizable: bool = True
    fullscreen: bool = False
    style: WindowStyle = WindowStyle.WINDOWED


@dataclass(slots=True)
class WindowEvent:
    """Window event data."""
    type: WindowEventType
    data: dict[str, Any] = field(default_factory=dict)


class Window:
    """
    Window management class with headless backend.

    This implementation provides a full window API surface without
    requiring an actual display server, suitable for headless testing.
    """

    __slots__ = (
        "_config",
        "_state",
        "_is_open",
        "_handle",
        "_events",
        "_dpi_scale",
        "_client_rect",
    )

    _next_handle: ClassVar[int] = 1
    _handle_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config: WindowConfig):
        """Initialize window with configuration."""
        self._config = config
        self._state = WindowState.HIDDEN if not config.fullscreen else WindowState.NORMAL
        self._is_open = True
        # Thread-safe handle assignment
        with Window._handle_lock:
            self._handle = Window._next_handle
            Window._next_handle += 1
        self._events: list[WindowEvent] = []
        self._dpi_scale = DEFAULT_DPI_SCALE
        self._client_rect = Rect(
            config.x,
            config.y,
            config.width,
            config.height
        )

    @classmethod
    def create(cls, config: WindowConfig) -> "Window":
        """
        Create a new window with the given configuration.

        Args:
            config: Window configuration

        Returns:
            New Window instance
        """
        return cls(config)

    def show(self) -> None:
        """Show the window."""
        if self._state == WindowState.HIDDEN:
            self._state = WindowState.NORMAL
            self._events.append(WindowEvent(WindowEventType.RESTORE))

    def hide(self) -> None:
        """Hide the window."""
        if self._state != WindowState.HIDDEN:
            old_state = self._state
            self._state = WindowState.HIDDEN
            if old_state != WindowState.HIDDEN:
                self._events.append(WindowEvent(WindowEventType.BLUR))

    def minimize(self) -> None:
        """Minimize the window."""
        if self._state != WindowState.MINIMIZED:
            self._state = WindowState.MINIMIZED
            self._events.append(WindowEvent(WindowEventType.MINIMIZE))

    def maximize(self) -> None:
        """Maximize the window."""
        if self._state != WindowState.MAXIMIZED:
            old_state = self._state
            self._state = WindowState.MAXIMIZED
            if old_state != WindowState.HIDDEN:
                self._events.append(WindowEvent(WindowEventType.MAXIMIZE))

    def restore(self) -> None:
        """Restore window to normal state."""
        if self._state != WindowState.NORMAL:
            old_state = self._state
            self._state = WindowState.NORMAL
            if old_state == WindowState.MINIMIZED or old_state == WindowState.MAXIMIZED:
                self._events.append(WindowEvent(WindowEventType.RESTORE))

    def close(self) -> None:
        """Close the window."""
        if self._is_open:
            self._is_open = False
            self._events.append(WindowEvent(WindowEventType.CLOSE))

    def set_title(self, title: str) -> None:
        """
        Set window title.

        Args:
            title: New window title
        """
        self._config.title = title

    def set_size(self, width: int, height: int) -> None:
        """
        Set window size.

        Args:
            width: New width in pixels
            height: New height in pixels
        """
        if self._config.width != width or self._config.height != height:
            self._config.width = width
            self._config.height = height
            self._client_rect.width = width
            self._client_rect.height = height
            self._events.append(WindowEvent(
                WindowEventType.RESIZE,
                {"width": width, "height": height}
            ))

    def set_position(self, x: int, y: int) -> None:
        """
        Set window position.

        Args:
            x: X coordinate in screen space
            y: Y coordinate in screen space
        """
        if self._config.x != x or self._config.y != y:
            self._config.x = x
            self._config.y = y
            self._client_rect.x = x
            self._client_rect.y = y
            self._events.append(WindowEvent(
                WindowEventType.MOVE,
                {"x": x, "y": y}
            ))

    def set_fullscreen(self, fullscreen: bool, mode: FullscreenMode = FullscreenMode.EXCLUSIVE) -> None:
        """
        Set fullscreen mode.

        Args:
            fullscreen: Enable or disable fullscreen
            mode: Fullscreen mode to use
        """
        self._config.fullscreen = fullscreen
        if fullscreen:
            if mode == FullscreenMode.EXCLUSIVE:
                self._config.style = WindowStyle.FULLSCREEN_EXCLUSIVE
            elif mode == FullscreenMode.BORDERLESS:
                self._config.style = WindowStyle.BORDERLESS
            self._events.append(WindowEvent(
                WindowEventType.RESIZE,
                {"width": self._config.width, "height": self._config.height, "fullscreen": True}
            ))
        else:
            self._config.style = WindowStyle.WINDOWED
            self._events.append(WindowEvent(
                WindowEventType.RESIZE,
                {"width": self._config.width, "height": self._config.height, "fullscreen": False}
            ))

    def client_rect(self) -> Rect:
        """
        Get the client area rectangle.

        Returns:
            Rectangle representing the client area
        """
        return Rect(
            self._client_rect.x,
            self._client_rect.y,
            self._client_rect.width,
            self._client_rect.height
        )

    def dpi_scale(self) -> float:
        """
        Get the DPI scale factor.

        Returns:
            DPI scale factor (1.0 = 96 DPI)
        """
        return self._dpi_scale

    def native_handle(self) -> int:
        """
        Get the native window handle.

        Returns:
            Opaque native handle (simulated in headless mode)
        """
        return self._handle

    def poll_events(self) -> list[WindowEvent]:
        """
        Poll for window events.

        Returns:
            List of events since last poll
        """
        events = self._events.copy()
        self._events.clear()
        return events

    @property
    def is_open(self) -> bool:
        """Check if window is open."""
        return self._is_open

    @property
    def state(self) -> WindowState:
        """Get current window state."""
        return self._state

    @property
    def config(self) -> WindowConfig:
        """Get window configuration."""
        return self._config
