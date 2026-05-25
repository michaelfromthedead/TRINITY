"""
Display enumeration and management.

Provides display detection, mode enumeration, and DPI information
with sensible defaults for headless operation.
"""

from dataclasses import dataclass
from typing import ClassVar

from .window import Rect
from ..constants import (
    STANDARD_RESOLUTIONS, STANDARD_REFRESH_RATES,
    DEFAULT_TASKBAR_HEIGHT, DEFAULT_DPI_SCALE
)


@dataclass(slots=True)
class DisplayMode:
    """Display mode information."""
    width: int
    height: int
    refresh_rate: int
    format: str = "RGBA8888"


@dataclass(slots=True)
class DisplayInfo:
    """Display information."""
    name: str
    bounds: Rect
    work_area: Rect
    dpi_scale: float
    is_primary: bool


class Display:
    """
    Display management class.

    Provides display enumeration and mode information with a headless
    backend that returns sensible defaults for testing.
    """

    __slots__ = ("_info", "_modes", "_current_mode")

    _displays: ClassVar[list["Display"]] = []
    _initialized: ClassVar[bool] = False

    def __init__(self, info: DisplayInfo, modes: list[DisplayMode], current_mode: DisplayMode):
        """Initialize display with information."""
        self._info = info
        self._modes = modes
        self._current_mode = current_mode

    @classmethod
    def _initialize_displays(cls) -> None:
        """Initialize default display configuration."""
        if cls._initialized:
            return

        # Create a default primary display
        primary_modes = []
        for width, height in STANDARD_RESOLUTIONS:
            for refresh_rate in STANDARD_REFRESH_RATES:
                primary_modes.append(DisplayMode(width, height, refresh_rate))

        primary_width, primary_height = STANDARD_RESOLUTIONS[0]
        primary_info = DisplayInfo(
            name="Primary Display",
            bounds=Rect(0, 0, primary_width, primary_height),
            work_area=Rect(0, 0, primary_width, primary_height - DEFAULT_TASKBAR_HEIGHT),
            dpi_scale=DEFAULT_DPI_SCALE,
            is_primary=True
        )

        primary = cls(primary_info, primary_modes, primary_modes[0])
        cls._displays.append(primary)

        # Create a secondary display
        secondary_modes = [
            DisplayMode(primary_width, primary_height, STANDARD_REFRESH_RATES[0]),
            DisplayMode(primary_width, primary_height, 75),
        ]

        secondary_info = DisplayInfo(
            name="Secondary Display",
            bounds=Rect(primary_width, 0, primary_width, primary_height),
            work_area=Rect(primary_width, 0, primary_width, primary_height),
            dpi_scale=DEFAULT_DPI_SCALE,
            is_primary=False
        )

        secondary = cls(secondary_info, secondary_modes, secondary_modes[0])
        cls._displays.append(secondary)

        cls._initialized = True

    @classmethod
    def enumerate(cls) -> list["Display"]:
        """
        Enumerate all available displays.

        Returns:
            List of Display objects
        """
        cls._initialize_displays()
        return cls._displays.copy()

    @classmethod
    def primary(cls) -> "Display":
        """
        Get the primary display.

        Returns:
            Primary Display object
        """
        cls._initialize_displays()
        for display in cls._displays:
            if display._info.is_primary:
                return display
        return cls._displays[0]

    @property
    def name(self) -> str:
        """Get display name."""
        return self._info.name

    @property
    def bounds(self) -> Rect:
        """Get display bounds."""
        return self._info.bounds

    @property
    def work_area(self) -> Rect:
        """Get display work area (excluding taskbars, etc.)."""
        return self._info.work_area

    @property
    def dpi_scale(self) -> float:
        """Get DPI scale factor."""
        return self._info.dpi_scale

    @property
    def is_primary(self) -> bool:
        """Check if this is the primary display."""
        return self._info.is_primary

    def supported_modes(self) -> list[DisplayMode]:
        """
        Get list of supported display modes.

        Returns:
            List of supported DisplayMode objects
        """
        return self._modes.copy()

    def current_mode(self) -> DisplayMode:
        """
        Get current display mode.

        Returns:
            Current DisplayMode
        """
        return self._current_mode
