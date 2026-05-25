"""
Checkbox Widget Implementation.

A checkbox widget with support for:
- Three states: checked, unchecked, indeterminate
- Label text with configurable position
- Focus and hover states
- Keyboard accessibility (Space to toggle)
- Value change events

Follows the Trinity Pattern with TrackedDescriptor for state changes
and ObservableDescriptor for event subscriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from time import time
from typing import Any, Callable, Optional


class CheckState(Enum):
    """Possible states for a checkbox."""
    UNCHECKED = auto()
    CHECKED = auto()
    INDETERMINATE = auto()


class CheckboxState(Enum):
    """Visual interaction states for the checkbox."""
    NORMAL = auto()
    HOVERED = auto()
    FOCUSED = auto()
    DISABLED = auto()


@dataclass(slots=True)
class CheckboxStyle:
    """Style configuration for checkbox appearance.

    Attributes:
        box_size: Size of the checkbox box in pixels
        box_color: Background color of unchecked box
        box_hover_color: Background color when hovered
        checked_color: Background color when checked
        check_mark_color: Color of the check mark
        border_color: Border color
        border_width: Border thickness
        corner_radius: Corner rounding
        label_color: Text color for label
        disabled_color: Color when disabled
        label_spacing: Gap between box and label
        font_size: Label font size
        font_weight: Label font weight
    """
    box_size: float = 20.0
    box_color: str = "#FFFFFF"
    box_hover_color: str = "#F0F0F0"
    checked_color: str = "#4A90D9"
    check_mark_color: str = "#FFFFFF"
    border_color: str = "#CCCCCC"
    border_width: float = 2.0
    corner_radius: float = 3.0
    label_color: str = "#333333"
    disabled_color: str = "#CCCCCC"
    label_spacing: float = 8.0
    font_size: float = 14.0
    font_weight: str = "normal"


@dataclass(slots=True)
class CheckStateChangeEvent:
    """Event emitted when checkbox state changes.

    Attributes:
        checkbox: Reference to the checkbox widget
        timestamp: Time of the change
        new_state: New check state
        previous_state: Previous check state
        is_user_action: True if triggered by user interaction
    """
    checkbox: "Checkbox"
    timestamp: float
    new_state: CheckState
    previous_state: CheckState
    is_user_action: bool = True


class Checkbox:
    """Interactive checkbox widget.

    A checkbox displays a toggleable box with optional label text.
    Supports three states: unchecked, checked, and indeterminate.

    Attributes:
        label: Text label displayed next to the checkbox
        check_state: Current check state (UNCHECKED, CHECKED, INDETERMINATE)
        enabled: Whether the checkbox is interactive
        visible: Whether the checkbox is rendered
        allow_indeterminate: Whether indeterminate state is allowed

    Events:
        on_change: Fired when check state changes

    Example:
        checkbox = Checkbox(label="Accept terms")
        checkbox.on_change(lambda e: print(f"Checked: {e.new_state}"))
    """

    __slots__ = (
        '_id', '_label', '_check_state', '_enabled', '_visible', '_focusable',
        '_allow_indeterminate', '_visual_state', '_style',
        '_x', '_y', '_width', '_height',
        '_on_change_handlers',
        '_is_hovered', '_is_focused',
        '_dirty', '_cached_mesh'
    )

    # Class-level ID counter
    _next_id: int = 0

    def __init__(
        self,
        label: str = "",
        checked: bool = False,
        enabled: bool = True,
        visible: bool = True,
        allow_indeterminate: bool = False,
        style: Optional[CheckboxStyle] = None,
        x: float = 0.0,
        y: float = 0.0,
        width: Optional[float] = None,
        height: Optional[float] = None,
    ):
        """Initialize a checkbox widget.

        Args:
            label: Text label to display
            checked: Initial checked state
            enabled: Initial enabled state
            visible: Initial visibility
            allow_indeterminate: Allow third indeterminate state
            style: Style configuration
            x: X position
            y: Y position
            width: Widget width (auto-calculated if None)
            height: Widget height (defaults to box size)
        """
        self._id = Checkbox._next_id
        Checkbox._next_id += 1

        self._label = label
        self._check_state = CheckState.CHECKED if checked else CheckState.UNCHECKED
        self._enabled = enabled
        self._visible = visible
        self._focusable = True
        self._allow_indeterminate = allow_indeterminate
        self._visual_state = CheckboxState.NORMAL if enabled else CheckboxState.DISABLED
        self._style = style or CheckboxStyle()

        self._x = x
        self._y = y
        # Auto-calculate width based on label if not provided
        self._width = width if width is not None else self._calculate_auto_width()
        self._height = height if height is not None else self._style.box_size

        self._on_change_handlers: list[Callable[[CheckStateChangeEvent], None]] = []

        self._is_hovered = False
        self._is_focused = False

        self._dirty = True
        self._cached_mesh: Any = None

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the ID counter. Used for testing."""
        cls._next_id = 0

    def _calculate_auto_width(self) -> float:
        """Calculate width based on label length (rough estimate)."""
        # Rough approximation: box + spacing + ~7 pixels per character
        label_width = len(self._label) * 7 if self._label else 0
        return self._style.box_size + self._style.label_spacing + label_width

    @property
    def id(self) -> int:
        """Get the unique widget ID."""
        return self._id

    @property
    def label(self) -> str:
        """Get the checkbox label."""
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        """Set the checkbox label."""
        if self._label != value:
            self._label = value
            self._dirty = True

    @property
    def check_state(self) -> CheckState:
        """Get the current check state."""
        return self._check_state

    @check_state.setter
    def check_state(self, value: CheckState) -> None:
        """Set the check state programmatically."""
        if not isinstance(value, CheckState):
            raise TypeError("check_state must be a CheckState enum value")
        if value == CheckState.INDETERMINATE and not self._allow_indeterminate:
            raise ValueError("Indeterminate state not allowed for this checkbox")
        if self._check_state != value:
            previous = self._check_state
            self._check_state = value
            self._dirty = True
            self._emit_change(previous, is_user_action=False)

    @property
    def checked(self) -> bool:
        """Get boolean checked state (True if CHECKED)."""
        return self._check_state == CheckState.CHECKED

    @checked.setter
    def checked(self, value: bool) -> None:
        """Set boolean checked state."""
        new_state = CheckState.CHECKED if value else CheckState.UNCHECKED
        if self._check_state != new_state:
            previous = self._check_state
            self._check_state = new_state
            self._dirty = True
            self._emit_change(previous, is_user_action=False)

    @property
    def is_indeterminate(self) -> bool:
        """Check if checkbox is in indeterminate state."""
        return self._check_state == CheckState.INDETERMINATE

    @property
    def enabled(self) -> bool:
        """Check if checkbox is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        if self._enabled != value:
            self._enabled = value
            self._update_visual_state()
            self._dirty = True

    @property
    def visible(self) -> bool:
        """Check if checkbox is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set visibility."""
        if self._visible != value:
            self._visible = value
            self._dirty = True

    @property
    def focusable(self) -> bool:
        """Check if checkbox can receive focus."""
        return self._focusable and self._enabled

    @focusable.setter
    def focusable(self, value: bool) -> None:
        """Set focusable state."""
        self._focusable = value

    @property
    def allow_indeterminate(self) -> bool:
        """Check if indeterminate state is allowed."""
        return self._allow_indeterminate

    @allow_indeterminate.setter
    def allow_indeterminate(self, value: bool) -> None:
        """Set whether indeterminate state is allowed."""
        self._allow_indeterminate = value
        # If currently indeterminate and disallowing, switch to unchecked
        if not value and self._check_state == CheckState.INDETERMINATE:
            self.check_state = CheckState.UNCHECKED

    @property
    def visual_state(self) -> CheckboxState:
        """Get current visual state."""
        return self._visual_state

    @property
    def style(self) -> CheckboxStyle:
        """Get checkbox style."""
        return self._style

    @style.setter
    def style(self, value: CheckboxStyle) -> None:
        """Set checkbox style."""
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
        """Get widget width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set widget width."""
        if value < 0:
            raise ValueError("width must be >= 0")
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
    def box_bounds(self) -> tuple[float, float, float, float]:
        """Get the checkbox box bounds (not including label)."""
        return (self._x, self._y, self._style.box_size, self._style.box_size)

    @property
    def is_dirty(self) -> bool:
        """Check if checkbox needs re-rendering."""
        return self._dirty

    def mark_clean(self) -> None:
        """Mark the checkbox as rendered."""
        self._dirty = False

    def _update_visual_state(self) -> None:
        """Update visual state based on current conditions."""
        if not self._enabled:
            self._visual_state = CheckboxState.DISABLED
        elif self._is_hovered:
            self._visual_state = CheckboxState.HOVERED
        elif self._is_focused:
            self._visual_state = CheckboxState.FOCUSED
        else:
            self._visual_state = CheckboxState.NORMAL

    def _emit_change(self, previous: CheckState, is_user_action: bool = True) -> None:
        """Emit state change event to all handlers."""
        event = CheckStateChangeEvent(
            checkbox=self,
            timestamp=time(),
            new_state=self._check_state,
            previous_state=previous,
            is_user_action=is_user_action,
        )
        for handler in self._on_change_handlers:
            handler(event)

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

    def on_change(self, handler: Callable[[CheckStateChangeEvent], None]) -> Callable[[], None]:
        """Subscribe to state change events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_change_handlers.append(handler)
        return lambda: self._on_change_handlers.remove(handler)

    # Input event handlers
    def handle_mouse_enter(self) -> None:
        """Handle mouse entering the checkbox area."""
        if not self._enabled:
            return
        self._is_hovered = True
        self._update_visual_state()
        self._dirty = True

    def handle_mouse_leave(self) -> None:
        """Handle mouse leaving the checkbox area."""
        self._is_hovered = False
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
        return True  # Consume but wait for mouse up to toggle

    def handle_mouse_up(self, x: float, y: float) -> bool:
        """Handle mouse button release.

        Args:
            x: Mouse X position
            y: Mouse Y position

        Returns:
            True if event was consumed
        """
        if not self._enabled or not self.contains_point(x, y):
            return False

        self.toggle()
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
            shift: Shift modifier state
            ctrl: Ctrl modifier state
            alt: Alt modifier state

        Returns:
            True if event was consumed
        """
        if not self._enabled or not self._is_focused:
            return False

        # Space toggles the checkbox
        if key == "space":
            self.toggle()
            return True

        return False

    def toggle(self) -> None:
        """Toggle the checkbox state.

        Cycles through: UNCHECKED -> CHECKED -> (INDETERMINATE ->) UNCHECKED
        """
        if not self._enabled:
            return

        previous = self._check_state

        if self._allow_indeterminate:
            # Three-state cycling
            if self._check_state == CheckState.UNCHECKED:
                self._check_state = CheckState.CHECKED
            elif self._check_state == CheckState.CHECKED:
                self._check_state = CheckState.INDETERMINATE
            else:
                self._check_state = CheckState.UNCHECKED
        else:
            # Two-state toggle
            if self._check_state == CheckState.UNCHECKED:
                self._check_state = CheckState.CHECKED
            else:
                self._check_state = CheckState.UNCHECKED

        self._dirty = True
        self._emit_change(previous, is_user_action=True)

    def set_indeterminate(self) -> None:
        """Set the checkbox to indeterminate state.

        Raises:
            ValueError: If indeterminate state is not allowed
        """
        if not self._allow_indeterminate:
            raise ValueError("Indeterminate state not allowed for this checkbox")

        if self._check_state != CheckState.INDETERMINATE:
            previous = self._check_state
            self._check_state = CheckState.INDETERMINATE
            self._dirty = True
            self._emit_change(previous, is_user_action=False)

    def get_current_box_color(self) -> str:
        """Get the box background color for current state.

        Returns:
            Color string for current state
        """
        if self._visual_state == CheckboxState.DISABLED:
            return self._style.disabled_color
        elif self._check_state == CheckState.CHECKED or self._check_state == CheckState.INDETERMINATE:
            return self._style.checked_color
        elif self._visual_state == CheckboxState.HOVERED:
            return self._style.box_hover_color
        else:
            return self._style.box_color
