"""
Cursor management for the game engine.

Provides cursor type selection, visibility control, and confinement
with a headless backend for testing.
"""

from enum import Enum, auto
from typing import Optional


class CursorType(Enum):
    """System cursor types."""
    ARROW = auto()
    HAND = auto()
    IBEAM = auto()
    CROSSHAIR = auto()
    RESIZE_NS = auto()
    RESIZE_EW = auto()
    RESIZE_NESW = auto()
    RESIZE_NWSE = auto()
    RESIZE_ALL = auto()
    NOT_ALLOWED = auto()
    WAIT = auto()
    WAIT_ARROW = auto()


class CursorManager:
    """
    Cursor management class.

    Manages cursor appearance, visibility, and confinement with a
    headless backend for testing without a display server.
    """

    __slots__ = (
        "_current_type",
        "_visible",
        "_confined",
        "_custom_cursor_data",
    )

    def __init__(self):
        """Initialize cursor manager with defaults."""
        self._current_type = CursorType.ARROW
        self._visible = True
        self._confined = False
        self._custom_cursor_data: Optional[bytes] = None

    def set_cursor(self, cursor_type: CursorType) -> None:
        """
        Set the current cursor type.

        Args:
            cursor_type: System cursor type to display
        """
        self._current_type = cursor_type
        self._custom_cursor_data = None  # Clear custom cursor

    def set_visible(self, visible: bool) -> None:
        """
        Set cursor visibility.

        Args:
            visible: True to show cursor, False to hide
        """
        self._visible = visible

    def confine(self, confined: bool) -> None:
        """
        Confine cursor to window area.

        Args:
            confined: True to confine cursor, False to allow free movement
        """
        self._confined = confined

    def set_custom_cursor(self, image_data: bytes, hot_x: int, hot_y: int) -> None:
        """
        Set a custom cursor image.

        Args:
            image_data: Raw image data for cursor
            hot_x: X coordinate of cursor hotspot
            hot_y: Y coordinate of cursor hotspot
        """
        self._custom_cursor_data = image_data
        self._current_type = CursorType.ARROW  # Reset type when using custom

    @property
    def current_type(self) -> CursorType:
        """Get current cursor type."""
        return self._current_type

    @property
    def visible(self) -> bool:
        """Check if cursor is visible."""
        return self._visible

    @property
    def confined(self) -> bool:
        """Check if cursor is confined to window."""
        return self._confined

    @property
    def has_custom_cursor(self) -> bool:
        """Check if a custom cursor is set."""
        return self._custom_cursor_data is not None
