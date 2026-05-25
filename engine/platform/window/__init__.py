"""
Window and display management subsystem.

This module provides cross-platform window creation, display enumeration,
cursor management, HDR support, and variable refresh rate capabilities.
"""

from .window import (
    Window,
    WindowConfig,
    WindowStyle,
    FullscreenMode,
    WindowState,
    WindowEvent,
    WindowEventType,
    Rect,
)
from .display import (
    Display,
    DisplayMode,
    DisplayInfo,
)
from .cursor import (
    CursorManager,
    CursorType,
)
from .hdr import (
    DisplayHDR,
    HDRCapabilities,
    ColorSpace,
)
from .vrr import (
    VariableRefresh,
    VRRType,
    RefreshRange,
)

__all__ = [
    # Window
    "Window",
    "WindowConfig",
    "WindowStyle",
    "FullscreenMode",
    "WindowState",
    "WindowEvent",
    "WindowEventType",
    "Rect",
    # Display
    "Display",
    "DisplayMode",
    "DisplayInfo",
    # Cursor
    "CursorManager",
    "CursorType",
    # HDR
    "DisplayHDR",
    "HDRCapabilities",
    "ColorSpace",
    # VRR
    "VariableRefresh",
    "VRRType",
    "RefreshRange",
]
