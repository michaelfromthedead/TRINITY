"""
Slider Widget Implementation.

A slider widget for selecting numeric values within a range:
- Configurable min/max/step values
- Horizontal and vertical orientation
- Thumb dragging and track clicking
- Value change events
- Keyboard accessibility (arrow keys)

Follows the Trinity Pattern with TrackedDescriptor for state changes
and ObservableDescriptor for event subscriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from time import time
from typing import Any, Callable, Optional


class SliderOrientation(Enum):
    """Orientation of the slider."""
    HORIZONTAL = auto()
    VERTICAL = auto()


class SliderState(Enum):
    """Visual states for the slider widget."""
    NORMAL = auto()
    HOVERED = auto()
    DRAGGING = auto()
    FOCUSED = auto()
    DISABLED = auto()


@dataclass(slots=True)
class SliderStyle:
    """Style configuration for slider appearance.

    Attributes:
        track_color: Track background color
        track_fill_color: Color of filled portion of track
        thumb_color: Thumb (handle) color
        thumb_hover_color: Thumb color when hovered
        thumb_active_color: Thumb color when dragging
        disabled_color: Color when disabled
        track_height: Height of the track (or width if vertical)
        thumb_size: Diameter of the thumb
        corner_radius: Track corner rounding
        show_fill: Whether to show filled portion of track
        show_ticks: Whether to show tick marks
        tick_count: Number of tick marks to display
        tick_color: Color of tick marks
    """
    track_color: str = "#E0E0E0"
    track_fill_color: str = "#4A90D9"
    thumb_color: str = "#4A90D9"
    thumb_hover_color: str = "#5BA0E9"
    thumb_active_color: str = "#3A80C9"
    disabled_color: str = "#CCCCCC"
    track_height: float = 6.0
    thumb_size: float = 20.0
    corner_radius: float = 3.0
    show_fill: bool = True
    show_ticks: bool = False
    tick_count: int = 5
    tick_color: str = "#AAAAAA"


@dataclass(slots=True)
class ValueChangeEvent:
    """Event emitted when slider value changes.

    Attributes:
        slider: Reference to the slider widget
        timestamp: Time of the change
        new_value: New slider value
        previous_value: Previous slider value
        is_user_action: True if triggered by user interaction
        is_dragging: True if change occurred during drag
    """
    slider: "Slider"
    timestamp: float
    new_value: float
    previous_value: float
    is_user_action: bool = True
    is_dragging: bool = False


class Slider:
    """Interactive slider widget for numeric input.

    A slider allows users to select a value within a range by dragging
    a thumb along a track or clicking on the track.

    Attributes:
        value: Current slider value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        step: Value increment step (0 for continuous)
        orientation: HORIZONTAL or VERTICAL
        enabled: Whether the slider is interactive
        visible: Whether the slider is rendered

    Events:
        on_value_change: Fired when value changes
        on_drag_start: Fired when drag begins
        on_drag_end: Fired when drag ends

    Example:
        slider = Slider(min_value=0, max_value=100, value=50)
        slider.on_value_change(lambda e: print(f"Value: {e.new_value}"))
    """

    __slots__ = (
        '_id', '_value', '_min_value', '_max_value', '_step',
        '_orientation', '_enabled', '_visible', '_focusable',
        '_state', '_style',
        '_x', '_y', '_width', '_height',
        '_on_value_change_handlers', '_on_drag_start_handlers', '_on_drag_end_handlers',
        '_is_hovered', '_is_focused', '_is_dragging',
        '_drag_start_value', '_drag_start_pos',
        '_dirty', '_cached_mesh'
    )

    # Class-level ID counter
    _next_id: int = 0

    def __init__(
        self,
        value: float = 0.0,
        min_value: float = 0.0,
        max_value: float = 100.0,
        step: float = 0.0,
        orientation: SliderOrientation = SliderOrientation.HORIZONTAL,
        enabled: bool = True,
        visible: bool = True,
        style: Optional[SliderStyle] = None,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 200.0,
        height: float = 30.0,
    ):
        """Initialize a slider widget.

        Args:
            value: Initial value
            min_value: Minimum value
            max_value: Maximum value
            step: Value step (0 for continuous)
            orientation: Slider orientation
            enabled: Initial enabled state
            visible: Initial visibility
            style: Style configuration
            x: X position
            y: Y position
            width: Slider width
            height: Slider height

        Raises:
            ValueError: If min_value >= max_value or value out of range
        """
        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")
        if step < 0:
            raise ValueError("step must be >= 0")

        self._id = Slider._next_id
        Slider._next_id += 1

        self._min_value = min_value
        self._max_value = max_value
        self._step = step
        self._value = self._clamp_and_step(value)
        self._orientation = orientation
        self._enabled = enabled
        self._visible = visible
        self._focusable = True
        self._state = SliderState.NORMAL if enabled else SliderState.DISABLED
        self._style = style or SliderStyle()

        self._x = x
        self._y = y
        self._width = width
        self._height = height

        self._on_value_change_handlers: list[Callable[[ValueChangeEvent], None]] = []
        self._on_drag_start_handlers: list[Callable[["Slider"], None]] = []
        self._on_drag_end_handlers: list[Callable[["Slider"], None]] = []

        self._is_hovered = False
        self._is_focused = False
        self._is_dragging = False
        self._drag_start_value = 0.0
        self._drag_start_pos = (0.0, 0.0)

        self._dirty = True
        self._cached_mesh: Any = None

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the ID counter. Used for testing."""
        cls._next_id = 0

    def _clamp_and_step(self, value: float) -> float:
        """Clamp value to range and apply stepping.

        Args:
            value: Value to process

        Returns:
            Clamped and stepped value
        """
        # Clamp to range
        clamped = max(self._min_value, min(self._max_value, value))

        # Apply stepping if step > 0
        if self._step > 0:
            steps = round((clamped - self._min_value) / self._step)
            stepped = self._min_value + steps * self._step
            # Ensure we don't exceed max due to rounding
            stepped = min(stepped, self._max_value)
            return stepped

        return clamped

    @property
    def id(self) -> int:
        """Get the unique widget ID."""
        return self._id

    @property
    def value(self) -> float:
        """Get the current slider value."""
        return self._value

    @value.setter
    def value(self, val: float) -> None:
        """Set the slider value programmatically."""
        new_val = self._clamp_and_step(val)
        if self._value != new_val:
            previous = self._value
            self._value = new_val
            self._dirty = True
            self._emit_value_change(previous, is_user_action=False, is_dragging=False)

    @property
    def min_value(self) -> float:
        """Get the minimum value."""
        return self._min_value

    @min_value.setter
    def min_value(self, val: float) -> None:
        """Set the minimum value."""
        if val >= self._max_value:
            raise ValueError("min_value must be less than max_value")
        self._min_value = val
        # Re-clamp current value
        new_val = self._clamp_and_step(self._value)
        if new_val != self._value:
            previous = self._value
            self._value = new_val
            self._emit_value_change(previous, is_user_action=False, is_dragging=False)
        self._dirty = True

    @property
    def max_value(self) -> float:
        """Get the maximum value."""
        return self._max_value

    @max_value.setter
    def max_value(self, val: float) -> None:
        """Set the maximum value."""
        if val <= self._min_value:
            raise ValueError("max_value must be greater than min_value")
        self._max_value = val
        # Re-clamp current value
        new_val = self._clamp_and_step(self._value)
        if new_val != self._value:
            previous = self._value
            self._value = new_val
            self._emit_value_change(previous, is_user_action=False, is_dragging=False)
        self._dirty = True

    @property
    def step(self) -> float:
        """Get the step value."""
        return self._step

    @step.setter
    def step(self, val: float) -> None:
        """Set the step value."""
        if val < 0:
            raise ValueError("step must be >= 0")
        self._step = val
        # Re-step current value
        new_val = self._clamp_and_step(self._value)
        if new_val != self._value:
            previous = self._value
            self._value = new_val
            self._emit_value_change(previous, is_user_action=False, is_dragging=False)
        self._dirty = True

    @property
    def range(self) -> float:
        """Get the value range (max - min)."""
        return self._max_value - self._min_value

    @property
    def normalized_value(self) -> float:
        """Get the value as a 0-1 normalized value."""
        if self.range == 0:
            return 0.0
        return (self._value - self._min_value) / self.range

    @property
    def percentage(self) -> float:
        """Get the value as a percentage (0-100)."""
        return self.normalized_value * 100.0

    @property
    def orientation(self) -> SliderOrientation:
        """Get slider orientation."""
        return self._orientation

    @orientation.setter
    def orientation(self, val: SliderOrientation) -> None:
        """Set slider orientation."""
        if self._orientation != val:
            self._orientation = val
            self._dirty = True

    @property
    def enabled(self) -> bool:
        """Check if slider is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        if self._enabled != value:
            self._enabled = value
            if self._is_dragging:
                self._is_dragging = False
                for handler in self._on_drag_end_handlers:
                    handler(self)
            self._update_visual_state()
            self._dirty = True

    @property
    def visible(self) -> bool:
        """Check if slider is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set visibility."""
        if self._visible != value:
            self._visible = value
            self._dirty = True

    @property
    def focusable(self) -> bool:
        """Check if slider can receive focus."""
        return self._focusable and self._enabled

    @focusable.setter
    def focusable(self, value: bool) -> None:
        """Set focusable state."""
        self._focusable = value

    @property
    def state(self) -> SliderState:
        """Get current visual state."""
        return self._state

    @property
    def style(self) -> SliderStyle:
        """Get slider style."""
        return self._style

    @style.setter
    def style(self, value: SliderStyle) -> None:
        """Set slider style."""
        self._style = value
        self._dirty = True

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
        """Get slider width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set slider width."""
        if value < 0:
            raise ValueError("width must be >= 0")
        if self._width != value:
            self._width = value
            self._dirty = True

    @property
    def height(self) -> float:
        """Get slider height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set slider height."""
        if value < 0:
            raise ValueError("height must be >= 0")
        if self._height != value:
            self._height = value
            self._dirty = True

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get widget bounds (x, y, width, height)."""
        return (self._x, self._y, self._width, self._height)

    @property
    def is_dragging(self) -> bool:
        """Check if slider is currently being dragged."""
        return self._is_dragging

    @property
    def is_dirty(self) -> bool:
        """Check if slider needs re-rendering."""
        return self._dirty

    def mark_clean(self) -> None:
        """Mark the slider as rendered."""
        self._dirty = False

    def _update_visual_state(self) -> None:
        """Update visual state based on current conditions."""
        if not self._enabled:
            self._state = SliderState.DISABLED
        elif self._is_dragging:
            self._state = SliderState.DRAGGING
        elif self._is_hovered:
            self._state = SliderState.HOVERED
        elif self._is_focused:
            self._state = SliderState.FOCUSED
        else:
            self._state = SliderState.NORMAL

    def _emit_value_change(self, previous: float, is_user_action: bool, is_dragging: bool) -> None:
        """Emit value change event to all handlers."""
        event = ValueChangeEvent(
            slider=self,
            timestamp=time(),
            new_value=self._value,
            previous_value=previous,
            is_user_action=is_user_action,
            is_dragging=is_dragging,
        )
        for handler in self._on_value_change_handlers:
            handler(event)

    def get_thumb_position(self) -> tuple[float, float]:
        """Get the current thumb center position.

        Returns:
            (x, y) position of thumb center
        """
        half_thumb = self._style.thumb_size / 2

        if self._orientation == SliderOrientation.HORIZONTAL:
            track_start = self._x + half_thumb
            track_length = self._width - self._style.thumb_size
            thumb_x = track_start + self.normalized_value * track_length
            thumb_y = self._y + self._height / 2
        else:
            track_start = self._y + self._height - half_thumb
            track_length = self._height - self._style.thumb_size
            thumb_x = self._x + self._width / 2
            thumb_y = track_start - self.normalized_value * track_length

        return (thumb_x, thumb_y)

    def get_track_bounds(self) -> tuple[float, float, float, float]:
        """Get the track rectangle bounds.

        Returns:
            (x, y, width, height) of track
        """
        if self._orientation == SliderOrientation.HORIZONTAL:
            track_y = self._y + (self._height - self._style.track_height) / 2
            return (self._x, track_y, self._width, self._style.track_height)
        else:
            track_x = self._x + (self._width - self._style.track_height) / 2
            return (track_x, self._y, self._style.track_height, self._height)

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

    def _is_point_on_thumb(self, x: float, y: float) -> bool:
        """Check if a point is on the thumb.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is on the thumb
        """
        thumb_x, thumb_y = self.get_thumb_position()
        half_thumb = self._style.thumb_size / 2

        # Check circular thumb bounds
        dx = x - thumb_x
        dy = y - thumb_y
        return (dx * dx + dy * dy) <= (half_thumb * half_thumb)

    def _position_to_value(self, x: float, y: float) -> float:
        """Convert a screen position to a slider value.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Slider value at that position
        """
        half_thumb = self._style.thumb_size / 2

        if self._orientation == SliderOrientation.HORIZONTAL:
            track_start = self._x + half_thumb
            track_length = self._width - self._style.thumb_size
            if track_length <= 0:
                return self._min_value
            normalized = (x - track_start) / track_length
        else:
            track_start = self._y + self._height - half_thumb
            track_length = self._height - self._style.thumb_size
            if track_length <= 0:
                return self._min_value
            normalized = (track_start - y) / track_length

        normalized = max(0.0, min(1.0, normalized))
        return self._min_value + normalized * self.range

    # Event subscription methods
    def on_value_change(self, handler: Callable[[ValueChangeEvent], None]) -> Callable[[], None]:
        """Subscribe to value change events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_value_change_handlers.append(handler)
        return lambda: self._on_value_change_handlers.remove(handler)

    def on_drag_start(self, handler: Callable[["Slider"], None]) -> Callable[[], None]:
        """Subscribe to drag start events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_drag_start_handlers.append(handler)
        return lambda: self._on_drag_start_handlers.remove(handler)

    def on_drag_end(self, handler: Callable[["Slider"], None]) -> Callable[[], None]:
        """Subscribe to drag end events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_drag_end_handlers.append(handler)
        return lambda: self._on_drag_end_handlers.remove(handler)

    # Input event handlers
    def handle_mouse_enter(self) -> None:
        """Handle mouse entering the slider area."""
        if not self._enabled:
            return
        self._is_hovered = True
        self._update_visual_state()
        self._dirty = True

    def handle_mouse_leave(self) -> None:
        """Handle mouse leaving the slider area."""
        self._is_hovered = False
        if not self._is_dragging:
            self._update_visual_state()
        self._dirty = True

    def handle_mouse_down(self, x: float, y: float) -> bool:
        """Handle mouse button press.

        Args:
            x: Mouse X position
            y: Mouse Y position

        Returns:
            True if event was consumed
        """
        if not self._enabled or not self.contains_point(x, y):
            return False

        # Start dragging
        self._is_dragging = True
        self._drag_start_value = self._value
        self._drag_start_pos = (x, y)
        self._update_visual_state()
        self._dirty = True

        for handler in self._on_drag_start_handlers:
            handler(self)

        # If clicked on track (not thumb), jump to that position
        if not self._is_point_on_thumb(x, y):
            new_value = self._clamp_and_step(self._position_to_value(x, y))
            if new_value != self._value:
                previous = self._value
                self._value = new_value
                self._emit_value_change(previous, is_user_action=True, is_dragging=True)

        return True

    def handle_mouse_move(self, x: float, y: float) -> bool:
        """Handle mouse movement (drag).

        Args:
            x: Mouse X position
            y: Mouse Y position

        Returns:
            True if event was consumed
        """
        if not self._is_dragging:
            return False

        new_value = self._clamp_and_step(self._position_to_value(x, y))
        if new_value != self._value:
            previous = self._value
            self._value = new_value
            self._dirty = True
            self._emit_value_change(previous, is_user_action=True, is_dragging=True)

        return True

    def handle_mouse_up(self, x: float, y: float) -> bool:
        """Handle mouse button release.

        Args:
            x: Mouse X position
            y: Mouse Y position

        Returns:
            True if event was consumed
        """
        if not self._is_dragging:
            return False

        self._is_dragging = False
        self._update_visual_state()
        self._dirty = True

        for handler in self._on_drag_end_handlers:
            handler(self)

        return True

    def handle_focus_gained(self) -> None:
        """Handle receiving keyboard focus."""
        if not self._enabled:
            return
        self._is_focused = True
        self._update_visual_state()
        self._dirty = True

    def handle_focus_lost(self) -> None:
        """Handle losing keyboard focus."""
        self._is_focused = False
        self._update_visual_state()
        self._dirty = True

    def handle_key_down(self, key: str, shift: bool = False, ctrl: bool = False, alt: bool = False) -> bool:
        """Handle keyboard key press.

        Args:
            key: Key identifier
            shift: Whether shift is held (for larger steps)
            ctrl: Ctrl modifier state (reserved for future use)
            alt: Alt modifier state (reserved for future use)

        Returns:
            True if event was consumed
        """
        if not self._enabled or not self._is_focused:
            return False

        step_multiplier = 10 if shift else 1
        effective_step = self._step if self._step > 0 else self.range / 100
        step_amount = effective_step * step_multiplier

        previous = self._value
        new_value = self._value

        if self._orientation == SliderOrientation.HORIZONTAL:
            if key in ("right", "up"):
                new_value = self._clamp_and_step(self._value + step_amount)
            elif key in ("left", "down"):
                new_value = self._clamp_and_step(self._value - step_amount)
            elif key == "home":
                new_value = self._min_value
            elif key == "end":
                new_value = self._max_value
            else:
                return False
        else:  # VERTICAL
            if key in ("up", "right"):
                new_value = self._clamp_and_step(self._value + step_amount)
            elif key in ("down", "left"):
                new_value = self._clamp_and_step(self._value - step_amount)
            elif key == "home":
                new_value = self._min_value
            elif key == "end":
                new_value = self._max_value
            else:
                return False

        if new_value != previous:
            self._value = new_value
            self._dirty = True
            self._emit_value_change(previous, is_user_action=True, is_dragging=False)

        return True

    def get_current_thumb_color(self) -> str:
        """Get the thumb color for current state.

        Returns:
            Color string for current state
        """
        if self._state == SliderState.DISABLED:
            return self._style.disabled_color
        elif self._state == SliderState.DRAGGING:
            return self._style.thumb_active_color
        elif self._state == SliderState.HOVERED:
            return self._style.thumb_hover_color
        else:
            return self._style.thumb_color

    def get_tick_values(self) -> list[float]:
        """Get the values where tick marks should be displayed.

        Returns:
            List of tick mark values
        """
        if not self._style.show_ticks or self._style.tick_count < 2:
            return []

        ticks = []
        for i in range(self._style.tick_count):
            t = i / (self._style.tick_count - 1)
            ticks.append(self._min_value + t * self.range)
        return ticks
