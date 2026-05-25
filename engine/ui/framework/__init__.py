"""
UI Framework module for AI Game Engine.

Provides the foundational components for building user interfaces:
- Widget: Base class for all UI elements
- Container: Layout containers for organizing widgets
- Events: Event system with capture/bubble phases
- Focus: Focus management and tab navigation
- Coordinate: Coordinate systems and transforms

Core Classes:
    - Widget: Base widget with hierarchy, events, and dirty tracking
    - Container: Container with automatic child layout
    - FocusManager: Singleton for managing UI focus state

Example:
    from engine.ui.framework import Widget, Container, FocusManager

    # Create a simple widget hierarchy
    root = Container(name="root", width=800, height=600)
    button = Widget(name="button", x=10, y=10, width=100, height=40, focusable=True)
    root.add_child(button)

    # Set up focus management
    focus = FocusManager.get_instance()
    focus.set_root(root)
    focus.set_focus(button)
"""

# Coordinate system utilities
from engine.ui.framework.coordinate import (
    # Enums
    Anchor,
    CoordinateSpace,
    StretchMode,
    # Data classes
    Margins,
    Point,
    Rect,
    Size,
    Transform2D,
    # Converter
    CoordinateConverter,
    # Functions
    calculate_anchor_position,
)

# Event system
from engine.ui.framework.events import (
    # Enums
    EventPhase,
    EventType,
    KeyModifier,
    MouseButton,
    # Event classes
    DragEvent,
    FocusEvent,
    KeyboardEvent,
    MouseEvent,
    UIEvent,
    # Types
    EventHandler,
    # Dispatcher
    EventDispatcher,
)

# Base widget
from engine.ui.framework.widget import (
    # Descriptor
    TrackedDescriptor,
    # Data classes
    LayoutConstraints,
    WidgetStyle,
    # Main class
    Widget,
)

# Container widgets
from engine.ui.framework.container import (
    # Enums
    Alignment,
    CrossAlignment,
    LayoutDirection,
    # Data classes
    LayoutConfig,
    # Container classes
    Container,
    HBox,
    ScrollContainer,
    Stack,
    VBox,
)

# Focus management
from engine.ui.framework.focus import (
    # Enums
    FocusDirection,
    FocusReason,
    # Data classes
    FocusGroup,
    FocusTrap,
    # Manager
    FocusManager,
    # Convenience
    get_focus_manager,
)

__all__ = [
    # ============================================
    # Coordinate System
    # ============================================
    # Enums
    "Anchor",
    "CoordinateSpace",
    "StretchMode",
    # Data classes
    "Margins",
    "Point",
    "Rect",
    "Size",
    "Transform2D",
    # Converter
    "CoordinateConverter",
    # Functions
    "calculate_anchor_position",
    # ============================================
    # Events
    # ============================================
    # Enums
    "EventPhase",
    "EventType",
    "KeyModifier",
    "MouseButton",
    # Event classes
    "DragEvent",
    "FocusEvent",
    "KeyboardEvent",
    "MouseEvent",
    "UIEvent",
    # Types
    "EventHandler",
    # Dispatcher
    "EventDispatcher",
    # ============================================
    # Widget
    # ============================================
    # Descriptor
    "TrackedDescriptor",
    # Data classes
    "LayoutConstraints",
    "WidgetStyle",
    # Main class
    "Widget",
    # ============================================
    # Container
    # ============================================
    # Enums
    "Alignment",
    "CrossAlignment",
    "LayoutDirection",
    # Data classes
    "LayoutConfig",
    # Container classes
    "Container",
    "HBox",
    "ScrollContainer",
    "Stack",
    "VBox",
    # ============================================
    # Focus
    # ============================================
    # Enums
    "FocusDirection",
    "FocusReason",
    # Data classes
    "FocusGroup",
    "FocusTrap",
    # Manager
    "FocusManager",
    # Convenience
    "get_focus_manager",
]
