"""
Button Widget Implementation.

A clickable button widget with support for:
- Multiple visual states (normal, hovered, pressed, focused, disabled)
- Icon and text content
- Toggle mode for on/off buttons
- Click and press events
- Keyboard accessibility (Enter/Space activation)

Follows the Trinity Pattern with TrackedDescriptor for state changes
and ObservableDescriptor for event subscriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from time import time
from typing import Any, Callable, Optional


class ButtonState(Enum):
    """Visual states for a button widget."""
    NORMAL = auto()
    HOVERED = auto()
    PRESSED = auto()
    FOCUSED = auto()
    DISABLED = auto()


@dataclass(slots=True)
class ButtonStyle:
    """Style configuration for button appearance.

    Attributes:
        background_color: Background color in normal state
        hover_color: Background color when hovered
        pressed_color: Background color when pressed
        disabled_color: Background color when disabled
        text_color: Text foreground color
        disabled_text_color: Text color when disabled
        border_color: Border color
        border_width: Border thickness in pixels
        corner_radius: Corner rounding in pixels
        padding_horizontal: Horizontal padding in pixels
        padding_vertical: Vertical padding in pixels
        font_size: Text font size
        font_weight: Font weight (normal, bold, light)
        icon_size: Icon dimensions if present
        icon_spacing: Gap between icon and text
    """
    background_color: str = "#4A90D9"
    hover_color: str = "#5BA0E9"
    pressed_color: str = "#3A80C9"
    disabled_color: str = "#CCCCCC"
    text_color: str = "#FFFFFF"
    disabled_text_color: str = "#888888"
    border_color: str = "#2A70B9"
    border_width: float = 1.0
    corner_radius: float = 4.0
    padding_horizontal: float = 16.0
    padding_vertical: float = 8.0
    font_size: float = 14.0
    font_weight: str = "normal"
    icon_size: float = 16.0
    icon_spacing: float = 8.0


@dataclass(slots=True)
class ClickEvent:
    """Event emitted when button is clicked.

    Attributes:
        button: Reference to the button widget
        timestamp: Time of the click
        position: (x, y) position of the click within the button
        modifier_shift: Whether Shift was held
        modifier_ctrl: Whether Ctrl was held
        modifier_alt: Whether Alt was held
    """
    button: "Button"
    timestamp: float
    position: tuple[float, float] = (0.0, 0.0)
    modifier_shift: bool = False
    modifier_ctrl: bool = False
    modifier_alt: bool = False


@dataclass(slots=True)
class PressEvent:
    """Event emitted when button press begins or ends.

    Attributes:
        button: Reference to the button widget
        timestamp: Time of the event
        pressed: True if press started, False if released
    """
    button: "Button"
    timestamp: float
    pressed: bool


@dataclass(slots=True)
class ToggleEvent:
    """Event emitted when a toggle button changes state.

    Attributes:
        button: Reference to the button widget
        timestamp: Time of the toggle
        toggled_on: New toggle state
        previous_state: Previous toggle state
    """
    button: "Button"
    timestamp: float
    toggled_on: bool
    previous_state: bool


class Button:
    """Interactive button widget.

    A button can display text, an icon, or both. It supports multiple
    visual states and can operate in toggle mode for on/off functionality.

    Attributes:
        text: Button label text
        icon: Optional icon identifier or path
        enabled: Whether the button is interactive
        visible: Whether the button is rendered
        toggle_mode: If True, button acts as a toggle
        toggled_on: Current toggle state (only relevant if toggle_mode=True)
        state: Current visual state

    Events:
        on_click: Fired when button is clicked (and released)
        on_press: Fired when press begins/ends
        on_toggle: Fired when toggle state changes (toggle_mode only)
    """

    __slots__ = (
        '_id', '_text', '_icon', '_enabled', '_visible', '_focusable',
        '_toggle_mode', '_toggled_on', '_state', '_style',
        '_x', '_y', '_width', '_height',
        '_on_click_handlers', '_on_press_handlers', '_on_toggle_handlers',
        '_is_pressed', '_is_hovered', '_is_focused',
        '_dirty', '_cached_mesh'
    )

    # Class-level ID counter for unique widget IDs
    _next_id: int = 0

    def __init__(
        self,
        text: str = "",
        icon: Optional[str] = None,
        enabled: bool = True,
        visible: bool = True,
        toggle_mode: bool = False,
        toggled_on: bool = False,
        style: Optional[ButtonStyle] = None,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 100.0,
        height: float = 40.0,
    ):
        """Initialize a button widget.

        Args:
            text: Label text to display
            icon: Optional icon identifier
            enabled: Initial enabled state
            visible: Initial visibility
            toggle_mode: Enable toggle behavior
            toggled_on: Initial toggle state
            style: Style configuration
            x: X position
            y: Y position
            width: Button width
            height: Button height
        """
        self._id = Button._next_id
        Button._next_id += 1

        self._text = text
        self._icon = icon
        self._enabled = enabled
        self._visible = visible
        self._focusable = True
        self._toggle_mode = toggle_mode
        self._toggled_on = toggled_on
        self._state = ButtonState.NORMAL if enabled else ButtonState.DISABLED
        self._style = style or ButtonStyle()

        self._x = x
        self._y = y
        self._width = width
        self._height = height

        self._on_click_handlers: list[Callable[[ClickEvent], None]] = []
        self._on_press_handlers: list[Callable[[PressEvent], None]] = []
        self._on_toggle_handlers: list[Callable[[ToggleEvent], None]] = []

        self._is_pressed = False
        self._is_hovered = False
        self._is_focused = False

        self._dirty = True
        self._cached_mesh: Any = None

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the ID counter. Used for testing."""
        cls._next_id = 0

    @property
    def id(self) -> int:
        """Get the unique widget ID."""
        return self._id

    @property
    def text(self) -> str:
        """Get the button text."""
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """Set the button text."""
        if self._text != value:
            self._text = value
            self._dirty = True

    @property
    def icon(self) -> Optional[str]:
        """Get the button icon."""
        return self._icon

    @icon.setter
    def icon(self, value: Optional[str]) -> None:
        """Set the button icon."""
        if self._icon != value:
            self._icon = value
            self._dirty = True

    @property
    def enabled(self) -> bool:
        """Check if button is enabled."""
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
        """Check if button is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set visibility."""
        if self._visible != value:
            self._visible = value
            self._dirty = True

    @property
    def focusable(self) -> bool:
        """Check if button can receive focus."""
        return self._focusable and self._enabled

    @focusable.setter
    def focusable(self, value: bool) -> None:
        """Set focusable state."""
        self._focusable = value

    @property
    def toggle_mode(self) -> bool:
        """Check if button is in toggle mode."""
        return self._toggle_mode

    @toggle_mode.setter
    def toggle_mode(self, value: bool) -> None:
        """Set toggle mode."""
        self._toggle_mode = value

    @property
    def toggled_on(self) -> bool:
        """Get toggle state."""
        return self._toggled_on

    @toggled_on.setter
    def toggled_on(self, value: bool) -> None:
        """Set toggle state programmatically."""
        if self._toggled_on != value:
            previous = self._toggled_on
            self._toggled_on = value
            self._dirty = True
            self._emit_toggle(previous)

    @property
    def state(self) -> ButtonState:
        """Get current visual state."""
        return self._state

    @property
    def style(self) -> ButtonStyle:
        """Get button style."""
        return self._style

    @style.setter
    def style(self, value: ButtonStyle) -> None:
        """Set button style."""
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
        """Get button width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set button width."""
        if value < 0:
            raise ValueError("width must be >= 0")
        if self._width != value:
            self._width = value
            self._dirty = True

    @property
    def height(self) -> float:
        """Get button height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set button height."""
        if value < 0:
            raise ValueError("height must be >= 0")
        if self._height != value:
            self._height = value
            self._dirty = True

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get button bounds (x, y, width, height)."""
        return (self._x, self._y, self._width, self._height)

    @property
    def is_dirty(self) -> bool:
        """Check if button needs re-rendering."""
        return self._dirty

    def mark_clean(self) -> None:
        """Mark the button as rendered."""
        self._dirty = False

    def _update_visual_state(self) -> None:
        """Update the visual state based on current conditions."""
        if not self._enabled:
            self._state = ButtonState.DISABLED
        elif self._is_pressed:
            self._state = ButtonState.PRESSED
        elif self._is_hovered:
            self._state = ButtonState.HOVERED
        elif self._is_focused:
            self._state = ButtonState.FOCUSED
        else:
            self._state = ButtonState.NORMAL

    def _emit_click(self, pos: tuple[float, float], shift: bool, ctrl: bool, alt: bool) -> None:
        """Emit click event to all handlers."""
        event = ClickEvent(
            button=self,
            timestamp=time(),
            position=pos,
            modifier_shift=shift,
            modifier_ctrl=ctrl,
            modifier_alt=alt,
        )
        for handler in self._on_click_handlers:
            handler(event)

    def _emit_press(self, pressed: bool) -> None:
        """Emit press event to all handlers."""
        event = PressEvent(
            button=self,
            timestamp=time(),
            pressed=pressed,
        )
        for handler in self._on_press_handlers:
            handler(event)

    def _emit_toggle(self, previous: bool) -> None:
        """Emit toggle event to all handlers."""
        event = ToggleEvent(
            button=self,
            timestamp=time(),
            toggled_on=self._toggled_on,
            previous_state=previous,
        )
        for handler in self._on_toggle_handlers:
            handler(event)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within the button bounds.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside button bounds
        """
        return (
            self._x <= x <= self._x + self._width and
            self._y <= y <= self._y + self._height
        )

    # Event subscription methods
    def on_click(self, handler: Callable[[ClickEvent], None]) -> Callable[[], None]:
        """Subscribe to click events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_click_handlers.append(handler)
        return lambda: self._on_click_handlers.remove(handler)

    def on_press(self, handler: Callable[[PressEvent], None]) -> Callable[[], None]:
        """Subscribe to press events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_press_handlers.append(handler)
        return lambda: self._on_press_handlers.remove(handler)

    def on_toggle(self, handler: Callable[[ToggleEvent], None]) -> Callable[[], None]:
        """Subscribe to toggle events (only fires in toggle mode).

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_toggle_handlers.append(handler)
        return lambda: self._on_toggle_handlers.remove(handler)

    # Input event handlers
    def handle_mouse_enter(self) -> None:
        """Handle mouse entering the button area."""
        if not self._enabled:
            return
        self._is_hovered = True
        self._update_visual_state()
        self._dirty = True

    def handle_mouse_leave(self) -> None:
        """Handle mouse leaving the button area."""
        self._is_hovered = False
        if self._is_pressed:
            self._is_pressed = False
            self._emit_press(False)
        self._update_visual_state()
        self._dirty = True

    def handle_mouse_down(
        self,
        x: float,
        y: float,
        shift: bool = False,
        ctrl: bool = False,
        alt: bool = False,
    ) -> bool:
        """Handle mouse button press.

        Args:
            x: Mouse X position
            y: Mouse Y position
            shift: Shift modifier state
            ctrl: Ctrl modifier state
            alt: Alt modifier state

        Returns:
            True if event was consumed
        """
        if not self._enabled or not self.contains_point(x, y):
            return False

        self._is_pressed = True
        self._update_visual_state()
        self._dirty = True
        self._emit_press(True)
        return True

    def handle_mouse_up(
        self,
        x: float,
        y: float,
        shift: bool = False,
        ctrl: bool = False,
        alt: bool = False,
    ) -> bool:
        """Handle mouse button release.

        Args:
            x: Mouse X position
            y: Mouse Y position
            shift: Shift modifier state
            ctrl: Ctrl modifier state
            alt: Alt modifier state

        Returns:
            True if event was consumed
        """
        if not self._is_pressed:
            return False

        was_pressed = self._is_pressed
        self._is_pressed = False
        self._emit_press(False)

        # Only trigger click if released inside button
        if was_pressed and self.contains_point(x, y) and self._enabled:
            local_x = x - self._x
            local_y = y - self._y

            if self._toggle_mode:
                previous = self._toggled_on
                self._toggled_on = not self._toggled_on
                self._emit_toggle(previous)

            self._emit_click((local_x, local_y), shift, ctrl, alt)

        self._update_visual_state()
        self._dirty = True
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

        # Space and Enter activate the button
        if key in ("space", "enter", "return"):
            self._is_pressed = True
            self._update_visual_state()
            self._dirty = True
            self._emit_press(True)
            return True

        return False

    def handle_key_up(self, key: str, shift: bool = False, ctrl: bool = False, alt: bool = False) -> bool:
        """Handle keyboard key release.

        Args:
            key: Key identifier
            shift: Shift modifier state
            ctrl: Ctrl modifier state
            alt: Alt modifier state

        Returns:
            True if event was consumed
        """
        if not self._is_pressed:
            return False

        if key in ("space", "enter", "return"):
            self._is_pressed = False
            self._emit_press(False)

            if self._enabled:
                if self._toggle_mode:
                    previous = self._toggled_on
                    self._toggled_on = not self._toggled_on
                    self._emit_toggle(previous)

                self._emit_click((self._width / 2, self._height / 2), shift, ctrl, alt)

            self._update_visual_state()
            self._dirty = True
            return True

        return False

    def click(self) -> None:
        """Programmatically trigger a click."""
        if not self._enabled:
            return

        if self._toggle_mode:
            previous = self._toggled_on
            self._toggled_on = not self._toggled_on
            self._emit_toggle(previous)

        self._emit_click((self._width / 2, self._height / 2), False, False, False)

    def get_current_background_color(self) -> str:
        """Get the background color for the current state.

        Returns:
            Color string for current state
        """
        if self._state == ButtonState.DISABLED:
            return self._style.disabled_color
        elif self._state == ButtonState.PRESSED:
            return self._style.pressed_color
        elif self._state == ButtonState.HOVERED:
            return self._style.hover_color
        else:
            return self._style.background_color

    def get_current_text_color(self) -> str:
        """Get the text color for the current state.

        Returns:
            Text color string for current state
        """
        if self._state == ButtonState.DISABLED:
            return self._style.disabled_text_color
        return self._style.text_color
