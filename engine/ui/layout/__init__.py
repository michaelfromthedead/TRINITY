"""
Layout system for UI widgets.

Provides various layout containers for arranging child widgets:
- Canvas: Absolute positioning with anchors and pivots
- HBox: Horizontal arrangement with flex support
- VBox: Vertical arrangement with flex support
- Grid: Row/column grid with spans and auto-sizing
- Flex: Full flexbox-style layout with wrapping
- Responsive: Breakpoint-based responsive utilities
"""

from engine.ui.layout.canvas import (
    Anchor,
    AnchorPoint,
    Canvas,
    CanvasChild,
    CanvasSlot,
    Pivot,
    Rect,
)
from engine.ui.layout.flex import (
    AlignContent,
    FlexChild,
    FlexContainer,
    FlexDirection,
    FlexSlot,
    FlexWrap,
)
from engine.ui.layout.grid import (
    Grid,
    GridChild,
    GridSlot,
    TrackSize,
    TrackSizeType,
)
from engine.ui.layout.hbox import (
    Alignment,
    HBox,
    HBoxChild,
    HBoxSlot,
    Justify,
)
from engine.ui.layout.responsive import (
    Breakpoint,
    BreakpointManager,
    Orientation,
    ResponsiveContainer,
    ResponsiveRule,
    ResponsiveValue,
    SafeAreaInsets,
    Visibility,
)
from engine.ui.layout.vbox import (
    VBox,
    VBoxChild,
    VBoxSlot,
)

__all__ = [
    # Canvas
    "Anchor",
    "AnchorPoint",
    "Canvas",
    "CanvasChild",
    "CanvasSlot",
    "Pivot",
    "Rect",
    # HBox
    "Alignment",
    "HBox",
    "HBoxChild",
    "HBoxSlot",
    "Justify",
    # VBox
    "VBox",
    "VBoxChild",
    "VBoxSlot",
    # Grid
    "Grid",
    "GridChild",
    "GridSlot",
    "TrackSize",
    "TrackSizeType",
    # Flex
    "AlignContent",
    "FlexChild",
    "FlexContainer",
    "FlexDirection",
    "FlexSlot",
    "FlexWrap",
    # Responsive
    "Breakpoint",
    "BreakpointManager",
    "Orientation",
    "ResponsiveContainer",
    "ResponsiveRule",
    "ResponsiveValue",
    "SafeAreaInsets",
    "Visibility",
]
