"""
Responsive layout utilities - breakpoint-based adaptive layouts.

Provides utilities for creating responsive UIs that adapt to different
screen sizes, orientations, and safe area insets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, Optional, TypeVar

from engine.ui.layout.canvas import Rect

T = TypeVar("T")


class Breakpoint(Enum):
    """Standard breakpoint categories."""

    MOBILE = auto()  # < 600px
    TABLET = auto()  # 600px - 1023px
    DESKTOP = auto()  # >= 1024px


class Orientation(Enum):
    """Screen orientation."""

    PORTRAIT = auto()
    LANDSCAPE = auto()


class Visibility(Enum):
    """Visibility states for responsive elements."""

    VISIBLE = auto()
    HIDDEN = auto()
    COLLAPSED = auto()  # Hidden and takes no space


# Default breakpoint thresholds (in pixels)
# These follow common responsive design conventions:
# - Mobile: 0-599px (phones, small screens)
# - Tablet: 600-1023px (tablets, medium screens)
# - Desktop: 1024px+ (laptops, desktops, large screens)
BREAKPOINT_MOBILE_MIN: float = 0
BREAKPOINT_TABLET_MIN: float = 600
BREAKPOINT_DESKTOP_MIN: float = 1024

DEFAULT_BREAKPOINTS: dict[Breakpoint, float] = {
    Breakpoint.MOBILE: BREAKPOINT_MOBILE_MIN,
    Breakpoint.TABLET: BREAKPOINT_TABLET_MIN,
    Breakpoint.DESKTOP: BREAKPOINT_DESKTOP_MIN,
}


@dataclass
class SafeAreaInsets:
    """
    Safe area insets for devices with notches, rounded corners, etc.

    Values represent the inset from each edge that should be avoided.
    """

    top: float = 0.0
    right: float = 0.0
    bottom: float = 0.0
    left: float = 0.0

    def __post_init__(self) -> None:
        if self.top < 0:
            raise ValueError(f"top cannot be negative, got {self.top}")
        if self.right < 0:
            raise ValueError(f"right cannot be negative, got {self.right}")
        if self.bottom < 0:
            raise ValueError(f"bottom cannot be negative, got {self.bottom}")
        if self.left < 0:
            raise ValueError(f"left cannot be negative, got {self.left}")

    @property
    def horizontal(self) -> float:
        """Total horizontal inset."""
        return self.left + self.right

    @property
    def vertical(self) -> float:
        """Total vertical inset."""
        return self.top + self.bottom

    def with_top(self, value: float) -> "SafeAreaInsets":
        """Return a new SafeAreaInsets with updated top."""
        return SafeAreaInsets(top=value, right=self.right, bottom=self.bottom, left=self.left)

    def with_right(self, value: float) -> "SafeAreaInsets":
        """Return a new SafeAreaInsets with updated right."""
        return SafeAreaInsets(top=self.top, right=value, bottom=self.bottom, left=self.left)

    def with_bottom(self, value: float) -> "SafeAreaInsets":
        """Return a new SafeAreaInsets with updated bottom."""
        return SafeAreaInsets(top=self.top, right=self.right, bottom=value, left=self.left)

    def with_left(self, value: float) -> "SafeAreaInsets":
        """Return a new SafeAreaInsets with updated left."""
        return SafeAreaInsets(top=self.top, right=self.right, bottom=self.bottom, left=value)

    @classmethod
    def uniform(cls, value: float) -> "SafeAreaInsets":
        """Create insets with the same value on all sides."""
        return cls(top=value, right=value, bottom=value, left=value)

    @classmethod
    def symmetric(cls, horizontal: float = 0.0, vertical: float = 0.0) -> "SafeAreaInsets":
        """Create symmetric insets."""
        return cls(top=vertical, right=horizontal, bottom=vertical, left=horizontal)


@dataclass
class ResponsiveValue(Generic[T]):
    """
    A value that can vary based on breakpoint.

    Provides different values for mobile, tablet, and desktop breakpoints.
    """

    mobile: T
    tablet: Optional[T] = None
    desktop: Optional[T] = None

    def get(self, breakpoint: Breakpoint) -> T:
        """Get the value for a specific breakpoint."""
        if breakpoint == Breakpoint.DESKTOP and self.desktop is not None:
            return self.desktop
        if breakpoint == Breakpoint.TABLET and self.tablet is not None:
            return self.tablet
        if breakpoint == Breakpoint.TABLET and self.desktop is not None:
            # Fall back to desktop if no tablet value
            return self.desktop
        return self.mobile

    @classmethod
    def constant(cls, value: T) -> "ResponsiveValue[T]":
        """Create a responsive value that's the same for all breakpoints."""
        return cls(mobile=value)


@dataclass
class ResponsiveRule:
    """
    A rule for responsive behavior at a specific breakpoint.

    Can specify visibility, spacing, and custom property overrides.
    """

    breakpoint: Breakpoint
    visibility: Visibility = Visibility.VISIBLE
    padding_scale: float = 1.0
    margin_scale: float = 1.0
    gap_scale: float = 1.0
    font_scale: float = 1.0
    custom_properties: dict[str, Any] = field(default_factory=dict)


class BreakpointManager:
    """
    Manages breakpoint detection and responsive calculations.

    Tracks current screen size and provides utilities for breakpoint-aware layouts.
    """

    __slots__ = (
        "_width",
        "_height",
        "_safe_area",
        "_breakpoints",
        "_current_breakpoint",
        "_orientation",
        "_on_breakpoint_changed",
        "_on_orientation_changed",
    )

    def __init__(
        self,
        width: float = 0.0,
        height: float = 0.0,
        safe_area: Optional[SafeAreaInsets] = None,
        breakpoints: Optional[dict[Breakpoint, float]] = None,
    ) -> None:
        """
        Initialize the breakpoint manager.

        Args:
            width: Initial screen width.
            height: Initial screen height.
            safe_area: Safe area insets.
            breakpoints: Custom breakpoint thresholds.
        """
        if width < 0:
            raise ValueError(f"Width cannot be negative, got {width}")
        if height < 0:
            raise ValueError(f"Height cannot be negative, got {height}")

        self._width = width
        self._height = height
        self._safe_area = safe_area or SafeAreaInsets()
        self._breakpoints = breakpoints or dict(DEFAULT_BREAKPOINTS)
        self._current_breakpoint = self._calculate_breakpoint()
        self._orientation = self._calculate_orientation()
        self._on_breakpoint_changed: Optional[Callable[[Breakpoint], None]] = None
        self._on_orientation_changed: Optional[Callable[[Orientation], None]] = None

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    @property
    def safe_area(self) -> SafeAreaInsets:
        return self._safe_area

    @property
    def safe_width(self) -> float:
        """Width minus safe area insets."""
        return max(0, self._width - self._safe_area.horizontal)

    @property
    def safe_height(self) -> float:
        """Height minus safe area insets."""
        return max(0, self._height - self._safe_area.vertical)

    @property
    def safe_rect(self) -> Rect:
        """Rectangle representing the safe area."""
        return Rect(
            x=self._safe_area.left,
            y=self._safe_area.top,
            width=self.safe_width,
            height=self.safe_height,
        )

    @property
    def breakpoint(self) -> Breakpoint:
        return self._current_breakpoint

    @property
    def orientation(self) -> Orientation:
        return self._orientation

    @property
    def is_mobile(self) -> bool:
        return self._current_breakpoint == Breakpoint.MOBILE

    @property
    def is_tablet(self) -> bool:
        return self._current_breakpoint == Breakpoint.TABLET

    @property
    def is_desktop(self) -> bool:
        return self._current_breakpoint == Breakpoint.DESKTOP

    @property
    def is_portrait(self) -> bool:
        return self._orientation == Orientation.PORTRAIT

    @property
    def is_landscape(self) -> bool:
        return self._orientation == Orientation.LANDSCAPE

    def set_on_breakpoint_changed(
        self, callback: Optional[Callable[[Breakpoint], None]]
    ) -> None:
        """Set a callback for breakpoint changes."""
        self._on_breakpoint_changed = callback

    def set_on_orientation_changed(
        self, callback: Optional[Callable[[Orientation], None]]
    ) -> None:
        """Set a callback for orientation changes."""
        self._on_orientation_changed = callback

    def _calculate_breakpoint(self) -> Breakpoint:
        """Determine current breakpoint based on width."""
        # Sort breakpoints by threshold descending
        sorted_bp = sorted(
            self._breakpoints.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for bp, threshold in sorted_bp:
            if self._width >= threshold:
                return bp

        return Breakpoint.MOBILE

    def _calculate_orientation(self) -> Orientation:
        """Determine current orientation."""
        if self._height > self._width:
            return Orientation.PORTRAIT
        return Orientation.LANDSCAPE

    def update_size(
        self,
        width: float,
        height: float,
        safe_area: Optional[SafeAreaInsets] = None,
    ) -> None:
        """
        Update the screen size and safe area.

        Triggers callbacks if breakpoint or orientation changed.
        """
        if width < 0:
            raise ValueError(f"Width cannot be negative, got {width}")
        if height < 0:
            raise ValueError(f"Height cannot be negative, got {height}")

        self._width = width
        self._height = height
        if safe_area is not None:
            self._safe_area = safe_area

        new_breakpoint = self._calculate_breakpoint()
        new_orientation = self._calculate_orientation()

        if new_breakpoint != self._current_breakpoint:
            self._current_breakpoint = new_breakpoint
            if self._on_breakpoint_changed:
                self._on_breakpoint_changed(new_breakpoint)

        if new_orientation != self._orientation:
            self._orientation = new_orientation
            if self._on_orientation_changed:
                self._on_orientation_changed(new_orientation)

    def get_value(self, responsive_value: ResponsiveValue[T]) -> T:
        """Get the current value from a responsive value."""
        return responsive_value.get(self._current_breakpoint)

    def get_spacing(
        self,
        base_value: float,
        mobile_scale: float = 1.0,
        tablet_scale: float = 1.0,
        desktop_scale: float = 1.0,
    ) -> float:
        """Get a spacing value scaled for the current breakpoint."""
        if self.is_mobile:
            return base_value * mobile_scale
        elif self.is_tablet:
            return base_value * tablet_scale
        else:
            return base_value * desktop_scale

    def get_columns(
        self,
        mobile: int = 1,
        tablet: int = 2,
        desktop: int = 3,
    ) -> int:
        """Get the number of columns for the current breakpoint."""
        if self.is_mobile:
            return mobile
        elif self.is_tablet:
            return tablet
        else:
            return desktop


class ResponsiveContainer:
    """
    A container that adapts its layout based on breakpoints.

    Wraps another layout and applies responsive rules.
    """

    __slots__ = (
        "_layout",
        "_breakpoint_manager",
        "_rules",
        "_base_padding",
        "_base_gap",
        "_visibility_rules",
        "_on_layout_changed",
    )

    def __init__(
        self,
        layout: Any,
        breakpoint_manager: BreakpointManager,
        rules: Optional[list[ResponsiveRule]] = None,
    ) -> None:
        """
        Initialize a responsive container.

        Args:
            layout: The underlying layout to wrap.
            breakpoint_manager: The breakpoint manager to use.
            rules: Responsive rules to apply.
        """
        self._layout = layout
        self._breakpoint_manager = breakpoint_manager
        self._rules: dict[Breakpoint, ResponsiveRule] = {}
        self._visibility_rules: dict[Any, dict[Breakpoint, Visibility]] = {}
        self._on_layout_changed: Optional[Callable[[], None]] = None

        if rules:
            for rule in rules:
                self._rules[rule.breakpoint] = rule

        # Connect to breakpoint manager
        breakpoint_manager.set_on_breakpoint_changed(self._on_breakpoint_change)

    @property
    def layout(self) -> Any:
        """The underlying layout."""
        return self._layout

    @property
    def breakpoint_manager(self) -> BreakpointManager:
        """The breakpoint manager."""
        return self._breakpoint_manager

    @property
    def current_breakpoint(self) -> Breakpoint:
        """Current active breakpoint."""
        return self._breakpoint_manager.breakpoint

    @property
    def current_rule(self) -> Optional[ResponsiveRule]:
        """Get the current active rule."""
        return self._rules.get(self.current_breakpoint)

    def set_on_layout_changed(self, callback: Optional[Callable[[], None]]) -> None:
        """Set a callback for layout changes."""
        self._on_layout_changed = callback

    def add_rule(self, rule: ResponsiveRule) -> None:
        """Add a responsive rule."""
        self._rules[rule.breakpoint] = rule
        if rule.breakpoint == self.current_breakpoint:
            self._apply_current_rule()

    def remove_rule(self, breakpoint: Breakpoint) -> bool:
        """Remove a rule for a breakpoint."""
        if breakpoint in self._rules:
            del self._rules[breakpoint]
            return True
        return False

    def set_visibility_rule(
        self,
        widget: Any,
        mobile: Visibility = Visibility.VISIBLE,
        tablet: Visibility = Visibility.VISIBLE,
        desktop: Visibility = Visibility.VISIBLE,
    ) -> None:
        """
        Set visibility rules for a specific widget.

        Args:
            widget: The widget to configure.
            mobile: Visibility on mobile.
            tablet: Visibility on tablet.
            desktop: Visibility on desktop.
        """
        self._visibility_rules[id(widget)] = {
            Breakpoint.MOBILE: mobile,
            Breakpoint.TABLET: tablet,
            Breakpoint.DESKTOP: desktop,
        }
        self._apply_visibility()

    def get_widget_visibility(self, widget: Any) -> Visibility:
        """Get the current visibility for a widget."""
        rules = self._visibility_rules.get(id(widget))
        if rules:
            return rules.get(self.current_breakpoint, Visibility.VISIBLE)
        return Visibility.VISIBLE

    def is_widget_visible(self, widget: Any) -> bool:
        """Check if a widget should be visible at the current breakpoint."""
        visibility = self.get_widget_visibility(widget)
        return visibility == Visibility.VISIBLE

    def _on_breakpoint_change(self, new_breakpoint: Breakpoint) -> None:
        """Handle breakpoint changes."""
        self._apply_current_rule()
        self._apply_visibility()

        if self._on_layout_changed:
            self._on_layout_changed()

    def _apply_current_rule(self) -> None:
        """Apply the current breakpoint's rule to the layout."""
        rule = self.current_rule
        if not rule:
            return

        # Read base values fresh from layout to avoid stale caches
        base_padding = getattr(self._layout, "padding_left", 0.0)
        base_gap = getattr(self._layout, "gap", 0.0)

        # Apply padding scale
        if hasattr(self._layout, "set_padding"):
            scaled_padding = base_padding * rule.padding_scale
            self._layout.set_padding(all=scaled_padding)

        # Apply gap scale
        if hasattr(self._layout, "gap"):
            self._layout.gap = base_gap * rule.gap_scale

        # Apply custom properties
        for prop, value in rule.custom_properties.items():
            if hasattr(self._layout, prop):
                setattr(self._layout, prop, value)

    def _apply_visibility(self) -> None:
        """Apply visibility rules to children."""
        if not hasattr(self._layout, "_children"):
            return

        for child in self._layout._children:
            widget = getattr(child, "widget", None)
            if widget is None:
                continue

            visibility = self.get_widget_visibility(widget)
            slot = getattr(child, "slot", None)

            if slot and hasattr(slot, "visible"):
                if visibility == Visibility.VISIBLE:
                    child.slot.visible = True
                else:
                    child.slot.visible = False

    def calculate_layout(self) -> dict[int, Rect]:
        """Calculate the layout, delegating to the underlying layout."""
        self._apply_visibility()
        if hasattr(self._layout, "calculate_layout"):
            return self._layout.calculate_layout()
        return {}

    def get_child_rect(self, widget: Any) -> Optional[Rect]:
        """Get the computed rectangle for a child widget."""
        if hasattr(self._layout, "get_child_rect"):
            return self._layout.get_child_rect(widget)
        return None


def responsive_spacing(
    base: float,
    mobile_scale: float = 0.75,
    tablet_scale: float = 1.0,
    desktop_scale: float = 1.25,
) -> ResponsiveValue[float]:
    """
    Create a responsive spacing value.

    Args:
        base: Base spacing value.
        mobile_scale: Scale factor for mobile.
        tablet_scale: Scale factor for tablet.
        desktop_scale: Scale factor for desktop.

    Returns:
        A ResponsiveValue with scaled spacing.
    """
    return ResponsiveValue(
        mobile=base * mobile_scale,
        tablet=base * tablet_scale,
        desktop=base * desktop_scale,
    )


def responsive_font_size(
    base: float,
    mobile_scale: float = 0.875,
    tablet_scale: float = 1.0,
    desktop_scale: float = 1.125,
) -> ResponsiveValue[float]:
    """
    Create a responsive font size value.

    Args:
        base: Base font size.
        mobile_scale: Scale factor for mobile.
        tablet_scale: Scale factor for tablet.
        desktop_scale: Scale factor for desktop.

    Returns:
        A ResponsiveValue with scaled font sizes.
    """
    return ResponsiveValue(
        mobile=base * mobile_scale,
        tablet=base * tablet_scale,
        desktop=base * desktop_scale,
    )


def hide_on_mobile(widget: Any, container: ResponsiveContainer) -> None:
    """Configure a widget to be hidden on mobile devices."""
    container.set_visibility_rule(
        widget,
        mobile=Visibility.HIDDEN,
        tablet=Visibility.VISIBLE,
        desktop=Visibility.VISIBLE,
    )


def show_only_on_mobile(widget: Any, container: ResponsiveContainer) -> None:
    """Configure a widget to be visible only on mobile devices."""
    container.set_visibility_rule(
        widget,
        mobile=Visibility.VISIBLE,
        tablet=Visibility.HIDDEN,
        desktop=Visibility.HIDDEN,
    )


def hide_on_desktop(widget: Any, container: ResponsiveContainer) -> None:
    """Configure a widget to be hidden on desktop devices."""
    container.set_visibility_rule(
        widget,
        mobile=Visibility.VISIBLE,
        tablet=Visibility.VISIBLE,
        desktop=Visibility.HIDDEN,
    )


__all__ = [
    "Breakpoint",
    "Orientation",
    "Visibility",
    "SafeAreaInsets",
    "ResponsiveValue",
    "ResponsiveRule",
    "BreakpointManager",
    "ResponsiveContainer",
    "responsive_spacing",
    "responsive_font_size",
    "hide_on_mobile",
    "show_only_on_mobile",
    "hide_on_desktop",
    "BREAKPOINT_MOBILE_MIN",
    "BREAKPOINT_TABLET_MIN",
    "BREAKPOINT_DESKTOP_MIN",
    "DEFAULT_BREAKPOINTS",
]
