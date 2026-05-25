"""
Progress Bar Widget Implementation.

A progress bar widget for visualizing progress and loading states:
- Horizontal, vertical, and circular styles
- Determinate (known progress) and indeterminate (loading) modes
- Animated fill transitions with easing
- Segmented display option
- Custom value ranges
- Accessibility support (screen reader compatible)

Follows the Standalone Pattern with explicit state management
and event subscriptions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Optional


class ProgressBarStyle(Enum):
    """Visual style of the progress bar."""
    HORIZONTAL = auto()  # Left-to-right fill
    VERTICAL = auto()    # Bottom-to-top fill
    CIRCULAR = auto()    # Circular/radial fill


class ProgressBarMode(Enum):
    """Progress bar mode."""
    DETERMINATE = auto()    # Known progress value
    INDETERMINATE = auto()  # Unknown/loading state


class ProgressBarDirection(Enum):
    """Fill direction for horizontal/vertical styles."""
    FORWARD = auto()   # Left-to-right or bottom-to-top
    REVERSE = auto()   # Right-to-left or top-to-bottom


class ProgressBarState(Enum):
    """Visual states for the progress bar widget."""
    NORMAL = auto()
    COMPLETE = auto()
    DISABLED = auto()


@dataclass(slots=True)
class ProgressBarAppearance:
    """Style configuration for progress bar appearance.

    Attributes:
        fill_color: Color of the filled portion
        background_color: Color of the unfilled portion
        border_color: Border color
        border_width: Border width in pixels
        corner_radius: Corner radius for rounded corners
        complete_color: Fill color when progress is complete
        disabled_color: Color when disabled
        show_value: Whether to display the value as text
        value_format: Format string for value display
        value_color: Color of the value text
        value_font_size: Font size for value text
    """
    fill_color: str = "#4CAF50"
    background_color: str = "#E0E0E0"
    border_color: str = "#BDBDBD"
    border_width: float = 0.0
    corner_radius: float = 4.0
    complete_color: str = "#2E7D32"
    disabled_color: str = "#CCCCCC"
    show_value: bool = False
    value_format: str = "{:.0%}"
    value_color: str = "#000000"
    value_font_size: float = 12.0


@dataclass(slots=True)
class ProgressChangeEvent:
    """Event emitted when progress value changes.

    Attributes:
        progress_bar: Reference to the progress bar widget
        timestamp: Time of the change
        new_value: New progress value
        previous_value: Previous progress value
        normalized_value: Value normalized to 0-1 range
        is_complete: Whether progress reached maximum
    """
    progress_bar: "ProgressBar"
    timestamp: float
    new_value: float
    previous_value: float
    normalized_value: float
    is_complete: bool


class ProgressBar:
    """Progress bar widget for value visualization.

    Displays a value as a visual progress indicator. Supports multiple
    styles (horizontal, vertical, circular), determinate and indeterminate
    modes, and animated transitions.

    Attributes:
        value: Current value (clamped to range)
        min_value: Minimum value of the range
        max_value: Maximum value of the range
        style: Visual style (horizontal, vertical, circular)
        mode: Determinate or indeterminate mode
        direction: Fill direction (forward or reverse)
        animated: Whether to animate value changes
        animation_duration: Duration of value transition in seconds
        segments: Number of segments (0 = smooth fill)
        segment_gap: Gap between segments
        enabled: Whether the widget is enabled
        visible: Whether the widget is rendered

    Events:
        on_value_change: Fired when value changes
        on_complete: Fired when progress reaches maximum

    Example:
        # Simple progress bar
        progress = ProgressBar(value=0.5)

        # Custom range (e.g., health bar)
        health = ProgressBar(value=75, min_value=0, max_value=100)

        # Circular loading indicator
        loading = ProgressBar(
            style=ProgressBarStyle.CIRCULAR,
            mode=ProgressBarMode.INDETERMINATE
        )
    """

    __slots__ = (
        '_id', '_value', '_min_value', '_max_value',
        '_style', '_mode', '_direction', '_state',
        '_animated', '_animation_duration',
        '_animation_start_time', '_animation_start_value', '_animation_target_value',
        '_is_animating',
        '_indeterminate_position', '_indeterminate_speed',
        '_segments', '_segment_gap',
        '_appearance',
        '_x', '_y', '_width', '_height',
        '_visible', '_enabled', '_opacity',
        '_on_value_change_handlers', '_on_complete_handlers',
        '_dirty', '_cached_mesh'
    )

    # Class-level ID counter
    _next_id: int = 0

    def __init__(
        self,
        value: float = 0.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        style: ProgressBarStyle = ProgressBarStyle.HORIZONTAL,
        mode: ProgressBarMode = ProgressBarMode.DETERMINATE,
        direction: ProgressBarDirection = ProgressBarDirection.FORWARD,
        animated: bool = True,
        animation_duration: float = 0.2,
        segments: int = 0,
        segment_gap: float = 2.0,
        indeterminate_speed: float = 1.0,
        appearance: Optional[ProgressBarAppearance] = None,
        enabled: bool = True,
        visible: bool = True,
        opacity: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 200.0,
        height: float = 20.0,
    ):
        """Initialize a progress bar widget.

        Args:
            value: Initial value
            min_value: Minimum value
            max_value: Maximum value
            style: Visual style (HORIZONTAL, VERTICAL, CIRCULAR)
            mode: Progress mode (DETERMINATE, INDETERMINATE)
            direction: Fill direction (FORWARD, REVERSE)
            animated: Whether to animate value changes
            animation_duration: Animation duration in seconds
            segments: Number of segments (0 for smooth)
            segment_gap: Gap between segments in pixels
            indeterminate_speed: Speed of indeterminate animation (cycles/sec)
            appearance: Style configuration
            enabled: Initial enabled state
            visible: Initial visibility
            opacity: Opacity from 0.0 to 1.0
            x: X position
            y: Y position
            width: Widget width
            height: Widget height

        Raises:
            ValueError: If min_value >= max_value or invalid parameters
        """
        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")
        if animation_duration < 0:
            raise ValueError("animation_duration must be >= 0")
        if segments < 0:
            raise ValueError("segments must be >= 0")

        self._id = ProgressBar._next_id
        ProgressBar._next_id += 1

        self._min_value = min_value
        self._max_value = max_value
        self._value = self._clamp(value)
        self._style = style
        self._mode = mode
        self._direction = direction
        self._state = ProgressBarState.NORMAL if enabled else ProgressBarState.DISABLED

        self._animated = animated
        self._animation_duration = animation_duration
        self._animation_start_time = 0.0
        self._animation_start_value = self._value
        self._animation_target_value = self._value
        self._is_animating = False

        self._indeterminate_position = 0.0
        self._indeterminate_speed = max(0.1, indeterminate_speed)

        self._segments = segments
        self._segment_gap = max(0.0, segment_gap)

        self._appearance = appearance or ProgressBarAppearance()

        self._x = x
        self._y = y
        self._width = max(0.0, width)
        self._height = max(0.0, height)

        self._visible = visible
        self._enabled = enabled
        self._opacity = max(0.0, min(1.0, opacity))

        self._on_value_change_handlers: list[Callable[[ProgressChangeEvent], None]] = []
        self._on_complete_handlers: list[Callable[["ProgressBar"], None]] = []

        self._dirty = True
        self._cached_mesh: Any = None

        # Update state if already complete
        if self._value >= self._max_value:
            self._state = ProgressBarState.COMPLETE

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the ID counter. Used for testing."""
        cls._next_id = 0

    def _clamp(self, value: float) -> float:
        """Clamp value to the valid range.

        Args:
            value: Value to clamp

        Returns:
            Clamped value
        """
        return max(self._min_value, min(self._max_value, value))

    def _update_state(self) -> None:
        """Update visual state based on current conditions."""
        if not self._enabled:
            self._state = ProgressBarState.DISABLED
        elif self._value >= self._max_value:
            self._state = ProgressBarState.COMPLETE
        else:
            self._state = ProgressBarState.NORMAL

    def _emit_value_change(self, previous: float) -> None:
        """Emit value change event to all handlers."""
        is_complete = self._value >= self._max_value
        event = ProgressChangeEvent(
            progress_bar=self,
            timestamp=time.time(),
            new_value=self._value,
            previous_value=previous,
            normalized_value=self.normalized_value,
            is_complete=is_complete,
        )
        for handler in self._on_value_change_handlers:
            handler(event)

        # Fire completion handlers if just completed
        if is_complete and previous < self._max_value:
            for handler in self._on_complete_handlers:
                handler(self)

    # =========================================================================
    # CORE PROPERTIES
    # =========================================================================

    @property
    def id(self) -> int:
        """Get the unique widget ID."""
        return self._id

    @property
    def value(self) -> float:
        """Get the current value."""
        return self._value

    @value.setter
    def value(self, new_value: float) -> None:
        """Set the current value with optional animation."""
        new_value = self._clamp(new_value)
        if self._value == new_value:
            return

        previous = self._value

        if self._animated and self._mode == ProgressBarMode.DETERMINATE:
            # Start animation
            self._animation_start_value = self._value
            self._animation_target_value = new_value
            self._animation_start_time = time.time()
            self._is_animating = True
            # Update underlying value for state checks
            self._value = new_value
        else:
            self._value = new_value

        self._update_state()
        self._dirty = True
        self._emit_value_change(previous)

    @property
    def min_value(self) -> float:
        """Get the minimum value."""
        return self._min_value

    @min_value.setter
    def min_value(self, value: float) -> None:
        """Set the minimum value."""
        if value >= self._max_value:
            raise ValueError("min_value must be less than max_value")
        if self._min_value != value:
            self._min_value = value
            # Re-clamp current value
            new_val = self._clamp(self._value)
            if new_val != self._value:
                previous = self._value
                self._value = new_val
                self._emit_value_change(previous)
            self._update_state()
            self._dirty = True

    @property
    def max_value(self) -> float:
        """Get the maximum value."""
        return self._max_value

    @max_value.setter
    def max_value(self, value: float) -> None:
        """Set the maximum value."""
        if value <= self._min_value:
            raise ValueError("max_value must be greater than min_value")
        if self._max_value != value:
            self._max_value = value
            # Re-clamp current value
            new_val = self._clamp(self._value)
            if new_val != self._value:
                previous = self._value
                self._value = new_val
                self._emit_value_change(previous)
            self._update_state()
            self._dirty = True

    @property
    def range(self) -> float:
        """Get the value range (max - min)."""
        return self._max_value - self._min_value

    @property
    def normalized_value(self) -> float:
        """Get the value normalized to 0-1 range."""
        if self.range == 0:
            return 0.0
        return (self._value - self._min_value) / self.range

    @property
    def percentage(self) -> float:
        """Get the value as a percentage (0-100)."""
        return self.normalized_value * 100.0

    @property
    def percent(self) -> float:
        """Alias for percentage. Get the value as a percentage (0-100)."""
        return self.percentage

    # =========================================================================
    # STYLE PROPERTIES
    # =========================================================================

    @property
    def style(self) -> ProgressBarStyle:
        """Get the visual style."""
        return self._style

    @style.setter
    def style(self, value: ProgressBarStyle) -> None:
        """Set the visual style."""
        if self._style != value:
            self._style = value
            self._dirty = True

    @property
    def mode(self) -> ProgressBarMode:
        """Get the progress mode."""
        return self._mode

    @mode.setter
    def mode(self, value: ProgressBarMode) -> None:
        """Set the progress mode."""
        if self._mode != value:
            self._mode = value
            if value == ProgressBarMode.INDETERMINATE:
                self._indeterminate_position = 0.0
            self._dirty = True

    @property
    def direction(self) -> ProgressBarDirection:
        """Get the fill direction."""
        return self._direction

    @direction.setter
    def direction(self, value: ProgressBarDirection) -> None:
        """Set the fill direction."""
        if self._direction != value:
            self._direction = value
            self._dirty = True

    @property
    def state(self) -> ProgressBarState:
        """Get current visual state."""
        return self._state

    @property
    def appearance(self) -> ProgressBarAppearance:
        """Get the appearance configuration."""
        return self._appearance

    @appearance.setter
    def appearance(self, value: ProgressBarAppearance) -> None:
        """Set the appearance configuration."""
        self._appearance = value
        self._dirty = True

    # =========================================================================
    # ANIMATION PROPERTIES
    # =========================================================================

    @property
    def animated(self) -> bool:
        """Get whether animation is enabled."""
        return self._animated

    @animated.setter
    def animated(self, value: bool) -> None:
        """Set whether animation is enabled."""
        if self._animated != value:
            self._animated = value
            if not value:
                self._is_animating = False

    @property
    def animation_duration(self) -> float:
        """Get the animation duration in seconds."""
        return self._animation_duration

    @animation_duration.setter
    def animation_duration(self, value: float) -> None:
        """Set the animation duration."""
        value = max(0.0, value)
        if self._animation_duration != value:
            self._animation_duration = value

    @property
    def indeterminate_speed(self) -> float:
        """Get the indeterminate animation speed (cycles per second)."""
        return self._indeterminate_speed

    @indeterminate_speed.setter
    def indeterminate_speed(self, value: float) -> None:
        """Set the indeterminate animation speed."""
        value = max(0.1, value)
        if self._indeterminate_speed != value:
            self._indeterminate_speed = value

    @property
    def is_animating(self) -> bool:
        """Check if currently animating a value change."""
        return self._is_animating

    # =========================================================================
    # SEGMENT PROPERTIES
    # =========================================================================

    @property
    def segments(self) -> int:
        """Get the number of segments (0 = smooth fill)."""
        return self._segments

    @segments.setter
    def segments(self, value: int) -> None:
        """Set the number of segments."""
        value = max(0, value)
        if self._segments != value:
            self._segments = value
            self._dirty = True

    @property
    def segment_gap(self) -> float:
        """Get the gap between segments in pixels."""
        return self._segment_gap

    @segment_gap.setter
    def segment_gap(self, value: float) -> None:
        """Set the gap between segments."""
        value = max(0.0, value)
        if self._segment_gap != value:
            self._segment_gap = value
            self._dirty = True

    # =========================================================================
    # TRANSFORM PROPERTIES
    # =========================================================================

    @property
    def x(self) -> float:
        """Get X position."""
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        """Set X position."""
        if self._x != value:
            self._x = value
            self._dirty = True

    @property
    def y(self) -> float:
        """Get Y position."""
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        """Set Y position."""
        if self._y != value:
            self._y = value
            self._dirty = True

    @property
    def width(self) -> float:
        """Get widget width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set widget width."""
        value = max(0.0, value)
        if self._width != value:
            self._width = value
            self._dirty = True

    @property
    def height(self) -> float:
        """Get widget height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set widget height."""
        value = max(0.0, value)
        if self._height != value:
            self._height = value
            self._dirty = True

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get widget bounds (x, y, width, height)."""
        return (self._x, self._y, self._width, self._height)

    # =========================================================================
    # VISIBILITY PROPERTIES
    # =========================================================================

    @property
    def visible(self) -> bool:
        """Check if widget is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set visibility."""
        if self._visible != value:
            self._visible = value
            self._dirty = True

    @property
    def enabled(self) -> bool:
        """Check if widget is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        if self._enabled != value:
            self._enabled = value
            self._update_state()
            self._dirty = True

    @property
    def opacity(self) -> float:
        """Get opacity (0.0 to 1.0)."""
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set opacity."""
        value = max(0.0, min(1.0, value))
        if self._opacity != value:
            self._opacity = value
            self._dirty = True

    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================

    @property
    def display_value(self) -> float:
        """Get the current display value (considering animation)."""
        if self._is_animating and self._animated:
            elapsed = time.time() - self._animation_start_time
            if elapsed >= self._animation_duration:
                self._is_animating = False
                return self._value

            if self._animation_duration > 0:
                t = elapsed / self._animation_duration
                # Ease out quad
                t = 1 - (1 - t) * (1 - t)
            else:
                t = 1.0

            return self._animation_start_value + t * (
                self._animation_target_value - self._animation_start_value
            )
        return self._value

    @property
    def display_normalized(self) -> float:
        """Get the normalized display value (0-1)."""
        if self.range == 0:
            return 0.0
        return (self.display_value - self._min_value) / self.range

    @property
    def is_complete(self) -> bool:
        """Check if progress is complete."""
        return self._value >= self._max_value

    @property
    def formatted_value(self) -> str:
        """Get the formatted value string for display."""
        try:
            return self._appearance.value_format.format(self.normalized_value)
        except (ValueError, KeyError):
            return str(self._value)

    @property
    def is_dirty(self) -> bool:
        """Check if widget needs re-rendering."""
        return self._dirty

    # =========================================================================
    # EVENT SUBSCRIPTION
    # =========================================================================

    def on_value_change(
        self, handler: Callable[[ProgressChangeEvent], None]
    ) -> Callable[[], None]:
        """Subscribe to value change events.

        Args:
            handler: Callback function receiving ProgressChangeEvent

        Returns:
            Unsubscribe function
        """
        self._on_value_change_handlers.append(handler)
        return lambda: self._on_value_change_handlers.remove(handler)

    def on_complete(self, handler: Callable[["ProgressBar"], None]) -> Callable[[], None]:
        """Subscribe to completion events.

        Args:
            handler: Callback function called when progress reaches max

        Returns:
            Unsubscribe function
        """
        self._on_complete_handlers.append(handler)
        return lambda: self._on_complete_handlers.remove(handler)

    # =========================================================================
    # METHODS
    # =========================================================================

    def update(self, delta_time: float) -> None:
        """Update the progress bar animation.

        Args:
            delta_time: Time since last update in seconds
        """
        if self._mode == ProgressBarMode.INDETERMINATE:
            # Update indeterminate animation position
            self._indeterminate_position += delta_time * self._indeterminate_speed
            self._indeterminate_position %= 1.0
            self._dirty = True

        if self._is_animating:
            # Check if animation is complete
            elapsed = time.time() - self._animation_start_time
            if elapsed >= self._animation_duration:
                self._is_animating = False
            self._dirty = True

    def set_value_immediate(self, value: float) -> None:
        """Set the value without animation.

        Args:
            value: New value to set
        """
        value = self._clamp(value)
        if self._value != value:
            previous = self._value
            self._value = value
            self._animation_target_value = value
            self._is_animating = False
            self._update_state()
            self._dirty = True
            self._emit_value_change(previous)

    def reset(self) -> None:
        """Reset progress to minimum value."""
        self.set_value_immediate(self._min_value)

    def complete(self) -> None:
        """Set progress to maximum value."""
        self.value = self._max_value

    def increment(self, amount: float = 0.1) -> None:
        """Increment the progress value.

        Args:
            amount: Amount to increment by
        """
        self.value = self._value + amount

    def decrement(self, amount: float = 0.1) -> None:
        """Decrement the progress value.

        Args:
            amount: Amount to decrement by
        """
        self.value = self._value - amount

    def mark_clean(self) -> None:
        """Mark the widget as rendered."""
        self._dirty = False

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within the widget bounds.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside bounds
        """
        return (
            self._x <= x <= self._x + self._width and
            self._y <= y <= self._y + self._height
        )

    # =========================================================================
    # RENDERING HELPERS
    # =========================================================================

    def get_current_fill_color(self) -> str:
        """Get the fill color for current state.

        Returns:
            Color string for current state
        """
        if self._state == ProgressBarState.DISABLED:
            return self._appearance.disabled_color
        elif self._state == ProgressBarState.COMPLETE:
            return self._appearance.complete_color
        else:
            return self._appearance.fill_color

    def get_fill_rect(self) -> tuple[float, float, float, float]:
        """Get the filled portion rectangle.

        Returns:
            Tuple of (x, y, width, height) for the filled area
        """
        fill = self.display_normalized

        if self._style == ProgressBarStyle.HORIZONTAL:
            if self._direction == ProgressBarDirection.FORWARD:
                return (self._x, self._y, self._width * fill, self._height)
            else:
                return (
                    self._x + self._width * (1 - fill),
                    self._y,
                    self._width * fill,
                    self._height
                )
        elif self._style == ProgressBarStyle.VERTICAL:
            if self._direction == ProgressBarDirection.FORWARD:
                return (
                    self._x,
                    self._y + self._height * (1 - fill),
                    self._width,
                    self._height * fill
                )
            else:
                return (self._x, self._y, self._width, self._height * fill)
        else:  # CIRCULAR
            return (self._x, self._y, self._width, self._height)

    def get_segment_rects(self) -> list[tuple[float, float, float, float, bool]]:
        """Get rectangles for segmented display.

        Returns:
            List of (x, y, width, height, filled) tuples
        """
        if self._segments <= 0:
            return []

        segments = []
        fill_count = int(self.display_normalized * self._segments)

        if self._style == ProgressBarStyle.HORIZONTAL:
            segment_width = (
                self._width - self._segment_gap * (self._segments - 1)
            ) / self._segments
            for i in range(self._segments):
                seg_x = self._x + i * (segment_width + self._segment_gap)
                filled = i < fill_count
                segments.append((seg_x, self._y, segment_width, self._height, filled))
        elif self._style == ProgressBarStyle.VERTICAL:
            segment_height = (
                self._height - self._segment_gap * (self._segments - 1)
            ) / self._segments
            for i in range(self._segments):
                seg_y = (
                    self._y + self._height -
                    (i + 1) * segment_height -
                    i * self._segment_gap
                )
                filled = i < fill_count
                segments.append((self._x, seg_y, self._width, segment_height, filled))

        return segments

    def get_circular_arc(self) -> tuple[float, float, float, float, float]:
        """Get the arc parameters for circular style.

        Returns:
            Tuple of (center_x, center_y, radius, start_angle, end_angle)
        """
        cx = self._x + self._width / 2
        cy = self._y + self._height / 2
        radius = min(self._width, self._height) / 2

        start_angle = -math.pi / 2  # Start from top
        end_angle = start_angle + 2 * math.pi * self.display_normalized

        return (cx, cy, radius, start_angle, end_angle)

    def get_indeterminate_position(self) -> float:
        """Get current position for indeterminate animation (0-1).

        Returns:
            Current animation position
        """
        return self._indeterminate_position

    # =========================================================================
    # ACCESSIBILITY
    # =========================================================================

    def get_accessible_text(self) -> str:
        """Get text for screen readers.

        Returns:
            Human-readable description of current state
        """
        if self._mode == ProgressBarMode.INDETERMINATE:
            return "Loading"
        return f"Progress: {self.percentage:.0f}%"

    def get_accessible_role(self) -> str:
        """Get the accessibility role.

        Returns:
            ARIA role string
        """
        return "progressbar"

    def get_accessible_value(self) -> dict[str, float]:
        """Get accessibility value information.

        Returns:
            Dictionary with min, max, and current values
        """
        return {
            "min": self._min_value,
            "max": self._max_value,
            "now": self._value,
        }

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize progress bar to dictionary.

        Returns:
            Dictionary representation of the widget
        """
        return {
            "id": self._id,
            "value": self._value,
            "min_value": self._min_value,
            "max_value": self._max_value,
            "style": self._style.name,
            "mode": self._mode.name,
            "direction": self._direction.name,
            "animated": self._animated,
            "animation_duration": self._animation_duration,
            "indeterminate_speed": self._indeterminate_speed,
            "segments": self._segments,
            "segment_gap": self._segment_gap,
            "appearance": {
                "fill_color": self._appearance.fill_color,
                "background_color": self._appearance.background_color,
                "border_color": self._appearance.border_color,
                "border_width": self._appearance.border_width,
                "corner_radius": self._appearance.corner_radius,
                "complete_color": self._appearance.complete_color,
                "disabled_color": self._appearance.disabled_color,
                "show_value": self._appearance.show_value,
                "value_format": self._appearance.value_format,
                "value_color": self._appearance.value_color,
                "value_font_size": self._appearance.value_font_size,
            },
            "x": self._x,
            "y": self._y,
            "width": self._width,
            "height": self._height,
            "visible": self._visible,
            "enabled": self._enabled,
            "opacity": self._opacity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProgressBar":
        """Deserialize progress bar from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            New ProgressBar instance
        """
        appearance_data = data.get("appearance", {})
        appearance = ProgressBarAppearance(
            fill_color=appearance_data.get("fill_color", "#4CAF50"),
            background_color=appearance_data.get("background_color", "#E0E0E0"),
            border_color=appearance_data.get("border_color", "#BDBDBD"),
            border_width=appearance_data.get("border_width", 0.0),
            corner_radius=appearance_data.get("corner_radius", 4.0),
            complete_color=appearance_data.get("complete_color", "#2E7D32"),
            disabled_color=appearance_data.get("disabled_color", "#CCCCCC"),
            show_value=appearance_data.get("show_value", False),
            value_format=appearance_data.get("value_format", "{:.0%}"),
            value_color=appearance_data.get("value_color", "#000000"),
            value_font_size=appearance_data.get("value_font_size", 12.0),
        )

        return cls(
            value=data.get("value", 0.0),
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 1.0),
            style=ProgressBarStyle[data.get("style", "HORIZONTAL")],
            mode=ProgressBarMode[data.get("mode", "DETERMINATE")],
            direction=ProgressBarDirection[data.get("direction", "FORWARD")],
            animated=data.get("animated", True),
            animation_duration=data.get("animation_duration", 0.2),
            indeterminate_speed=data.get("indeterminate_speed", 1.0),
            segments=data.get("segments", 0),
            segment_gap=data.get("segment_gap", 2.0),
            appearance=appearance,
            enabled=data.get("enabled", True),
            visible=data.get("visible", True),
            opacity=data.get("opacity", 1.0),
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            width=data.get("width", 200.0),
            height=data.get("height", 20.0),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ProgressBar(id={self._id}, value={self._value}, "
            f"range=[{self._min_value}, {self._max_value}], "
            f"style={self._style.name}, mode={self._mode.name})"
        )
