"""
UI event system for the framework.

Provides event classes for mouse, keyboard, focus, and drag interactions.
Supports event bubbling and capture phases following W3C event model.

Event Phases:
    - CAPTURE: Event travels from root to target
    - TARGET: Event reaches the target widget
    - BUBBLE: Event travels from target back to root

Event Types:
    - UIEvent: Base class for all UI events
    - MouseEvent: Click, double-click, enter, leave, move, scroll
    - KeyboardEvent: Key down, key up, character input
    - FocusEvent: Focus in, focus out
    - DragEvent: Drag start, drag, drag end, drop
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto
from typing import TYPE_CHECKING, Any, Callable, Optional

from engine.ui.framework.coordinate import Point

if TYPE_CHECKING:
    from engine.ui.framework.widget import Widget


class EventPhase(Enum):
    """Event propagation phases."""

    NONE = auto()     # Event not dispatching
    CAPTURE = auto()  # Capturing from root to target
    TARGET = auto()   # At the target widget
    BUBBLE = auto()   # Bubbling from target to root


class MouseButton(IntFlag):
    """Mouse button flags."""

    NONE = 0
    LEFT = 1
    RIGHT = 2
    MIDDLE = 4
    BUTTON4 = 8
    BUTTON5 = 16

    @classmethod
    def from_index(cls, index: int) -> "MouseButton":
        """Convert button index (0-4) to MouseButton."""
        buttons = [cls.LEFT, cls.RIGHT, cls.MIDDLE, cls.BUTTON4, cls.BUTTON5]
        if 0 <= index < len(buttons):
            return buttons[index]
        return cls.NONE


class KeyModifier(IntFlag):
    """Keyboard modifier flags."""

    NONE = 0
    SHIFT = 1
    CTRL = 2
    ALT = 4
    META = 8  # Windows/Command key
    CAPS_LOCK = 16
    NUM_LOCK = 32

    @classmethod
    def from_bools(
        cls,
        shift: bool = False,
        ctrl: bool = False,
        alt: bool = False,
        meta: bool = False,
    ) -> "KeyModifier":
        """Create modifiers from boolean flags."""
        mods = cls.NONE
        if shift:
            mods |= cls.SHIFT
        if ctrl:
            mods |= cls.CTRL
        if alt:
            mods |= cls.ALT
        if meta:
            mods |= cls.META
        return mods


class EventType(Enum):
    """Types of UI events."""

    # Mouse events
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    MOUSE_ENTER = "mouse_enter"
    MOUSE_LEAVE = "mouse_leave"
    MOUSE_MOVE = "mouse_move"
    MOUSE_SCROLL = "mouse_scroll"

    # Keyboard events
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    CHAR_INPUT = "char_input"

    # Focus events
    FOCUS_IN = "focus_in"
    FOCUS_OUT = "focus_out"
    FOCUS_CHANGE = "focus_change"

    # Drag events
    DRAG_START = "drag_start"
    DRAG = "drag"
    DRAG_END = "drag_end"
    DRAG_ENTER = "drag_enter"
    DRAG_LEAVE = "drag_leave"
    DRAG_OVER = "drag_over"
    DROP = "drop"

    # Generic
    CUSTOM = "custom"


@dataclass
class UIEvent:
    """
    Base class for all UI events.

    Supports event bubbling and capture phases.
    Events can be stopped or have their default action prevented.
    """

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    target: Optional["Widget"] = None
    current_target: Optional["Widget"] = None
    phase: EventPhase = EventPhase.NONE
    bubbles: bool = True
    cancelable: bool = True

    _stopped: bool = field(default=False, init=False, repr=False)
    _stopped_immediate: bool = field(default=False, init=False, repr=False)
    _default_prevented: bool = field(default=False, init=False, repr=False)

    @property
    def is_stopped(self) -> bool:
        """Check if event propagation has been stopped."""
        return self._stopped

    @property
    def is_stopped_immediate(self) -> bool:
        """Check if immediate propagation has been stopped."""
        return self._stopped_immediate

    @property
    def is_default_prevented(self) -> bool:
        """Check if default action has been prevented."""
        return self._default_prevented

    def stop_propagation(self) -> None:
        """Stop event from propagating to other widgets."""
        self._stopped = True

    def stop_immediate_propagation(self) -> None:
        """Stop event from propagating to other handlers on current widget."""
        self._stopped = True
        self._stopped_immediate = True

    def prevent_default(self) -> None:
        """Prevent the default action for this event."""
        if self.cancelable:
            self._default_prevented = True

    def clone(self) -> "UIEvent":
        """Create a copy of this event for re-dispatch."""
        return UIEvent(
            event_type=self.event_type,
            timestamp=self.timestamp,
            target=self.target,
            bubbles=self.bubbles,
            cancelable=self.cancelable,
        )


@dataclass
class MouseEvent(UIEvent):
    """
    Mouse interaction event.

    Contains position, button state, and modifier information.
    """

    # Position in widget-local coordinates
    x: float = 0.0
    y: float = 0.0

    # Position in screen/viewport coordinates
    screen_x: float = 0.0
    screen_y: float = 0.0

    # Button state
    button: MouseButton = MouseButton.NONE  # Button that triggered the event
    buttons: MouseButton = MouseButton.NONE  # All currently pressed buttons

    # Modifiers
    modifiers: KeyModifier = KeyModifier.NONE

    # Scroll delta (for scroll events)
    delta_x: float = 0.0
    delta_y: float = 0.0

    # Click count (for detecting double/triple clicks)
    click_count: int = 1

    @property
    def position(self) -> Point:
        """Local position as Point."""
        return Point(self.x, self.y)

    @property
    def screen_position(self) -> Point:
        """Screen position as Point."""
        return Point(self.screen_x, self.screen_y)

    @property
    def is_left_button(self) -> bool:
        """Check if left button is involved."""
        return bool(self.button & MouseButton.LEFT)

    @property
    def is_right_button(self) -> bool:
        """Check if right button is involved."""
        return bool(self.button & MouseButton.RIGHT)

    @property
    def is_middle_button(self) -> bool:
        """Check if middle button is involved."""
        return bool(self.button & MouseButton.MIDDLE)

    def clone(self) -> "MouseEvent":
        """Create a copy of this event."""
        return MouseEvent(
            event_type=self.event_type,
            timestamp=self.timestamp,
            target=self.target,
            bubbles=self.bubbles,
            cancelable=self.cancelable,
            x=self.x,
            y=self.y,
            screen_x=self.screen_x,
            screen_y=self.screen_y,
            button=self.button,
            buttons=self.buttons,
            modifiers=self.modifiers,
            delta_x=self.delta_x,
            delta_y=self.delta_y,
            click_count=self.click_count,
        )

    @classmethod
    def click(
        cls,
        x: float,
        y: float,
        button: MouseButton = MouseButton.LEFT,
        modifiers: KeyModifier = KeyModifier.NONE,
    ) -> "MouseEvent":
        """Create a click event."""
        return cls(
            event_type=EventType.CLICK,
            x=x,
            y=y,
            button=button,
            modifiers=modifiers,
        )

    @classmethod
    def move(
        cls,
        x: float,
        y: float,
        buttons: MouseButton = MouseButton.NONE,
        modifiers: KeyModifier = KeyModifier.NONE,
    ) -> "MouseEvent":
        """Create a move event."""
        return cls(
            event_type=EventType.MOUSE_MOVE,
            x=x,
            y=y,
            buttons=buttons,
            modifiers=modifiers,
            bubbles=False,  # Move events don't bubble
        )

    @classmethod
    def scroll(
        cls,
        x: float,
        y: float,
        delta_x: float,
        delta_y: float,
        modifiers: KeyModifier = KeyModifier.NONE,
    ) -> "MouseEvent":
        """Create a scroll event."""
        return cls(
            event_type=EventType.MOUSE_SCROLL,
            x=x,
            y=y,
            delta_x=delta_x,
            delta_y=delta_y,
            modifiers=modifiers,
        )


@dataclass
class KeyboardEvent(UIEvent):
    """
    Keyboard interaction event.

    Contains key code, character data, and modifier information.
    """

    # Key identification
    key: str = ""        # Logical key (e.g., "Enter", "a", "ArrowLeft")
    key_code: int = 0    # Physical key code
    char: str = ""       # Character produced (for CHAR_INPUT)

    # Modifiers
    modifiers: KeyModifier = KeyModifier.NONE

    # Repeat state
    is_repeat: bool = False

    @property
    def is_shift(self) -> bool:
        """Check if Shift is pressed."""
        return bool(self.modifiers & KeyModifier.SHIFT)

    @property
    def is_ctrl(self) -> bool:
        """Check if Ctrl is pressed."""
        return bool(self.modifiers & KeyModifier.CTRL)

    @property
    def is_alt(self) -> bool:
        """Check if Alt is pressed."""
        return bool(self.modifiers & KeyModifier.ALT)

    @property
    def is_meta(self) -> bool:
        """Check if Meta (Windows/Command) is pressed."""
        return bool(self.modifiers & KeyModifier.META)

    def clone(self) -> "KeyboardEvent":
        """Create a copy of this event."""
        return KeyboardEvent(
            event_type=self.event_type,
            timestamp=self.timestamp,
            target=self.target,
            bubbles=self.bubbles,
            cancelable=self.cancelable,
            key=self.key,
            key_code=self.key_code,
            char=self.char,
            modifiers=self.modifiers,
            is_repeat=self.is_repeat,
        )

    @classmethod
    def key_down(
        cls,
        key: str,
        key_code: int = 0,
        modifiers: KeyModifier = KeyModifier.NONE,
        is_repeat: bool = False,
    ) -> "KeyboardEvent":
        """Create a key down event."""
        return cls(
            event_type=EventType.KEY_DOWN,
            key=key,
            key_code=key_code,
            modifiers=modifiers,
            is_repeat=is_repeat,
        )

    @classmethod
    def key_up(
        cls,
        key: str,
        key_code: int = 0,
        modifiers: KeyModifier = KeyModifier.NONE,
    ) -> "KeyboardEvent":
        """Create a key up event."""
        return cls(
            event_type=EventType.KEY_UP,
            key=key,
            key_code=key_code,
            modifiers=modifiers,
        )

    @classmethod
    def char_input(
        cls,
        char: str,
        modifiers: KeyModifier = KeyModifier.NONE,
    ) -> "KeyboardEvent":
        """Create a character input event."""
        return cls(
            event_type=EventType.CHAR_INPUT,
            char=char,
            key=char,
            modifiers=modifiers,
        )


@dataclass
class FocusEvent(UIEvent):
    """
    Focus change event.

    Contains information about focus transitions.
    """

    # The widget losing/gaining focus
    related_target: Optional["Widget"] = None

    def clone(self) -> "FocusEvent":
        """Create a copy of this event."""
        return FocusEvent(
            event_type=self.event_type,
            timestamp=self.timestamp,
            target=self.target,
            bubbles=self.bubbles,
            cancelable=self.cancelable,
            related_target=self.related_target,
        )

    @classmethod
    def focus_in(
        cls,
        target: Optional["Widget"] = None,
        related_target: Optional["Widget"] = None,
    ) -> "FocusEvent":
        """Create a focus in event."""
        return cls(
            event_type=EventType.FOCUS_IN,
            target=target,
            related_target=related_target,
            bubbles=False,  # Focus events don't bubble
            cancelable=False,
        )

    @classmethod
    def focus_out(
        cls,
        target: Optional["Widget"] = None,
        related_target: Optional["Widget"] = None,
    ) -> "FocusEvent":
        """Create a focus out event."""
        return cls(
            event_type=EventType.FOCUS_OUT,
            target=target,
            related_target=related_target,
            bubbles=False,
            cancelable=False,
        )


@dataclass
class DragEvent(UIEvent):
    """
    Drag and drop event.

    Contains drag data and position information.
    """

    # Position
    x: float = 0.0
    y: float = 0.0

    # Drag data
    data: Any = None
    data_type: str = ""

    # Source widget
    source: Optional["Widget"] = None

    # Modifiers
    modifiers: KeyModifier = KeyModifier.NONE

    @property
    def position(self) -> Point:
        """Position as Point."""
        return Point(self.x, self.y)

    def clone(self) -> "DragEvent":
        """Create a copy of this event."""
        return DragEvent(
            event_type=self.event_type,
            timestamp=self.timestamp,
            target=self.target,
            bubbles=self.bubbles,
            cancelable=self.cancelable,
            x=self.x,
            y=self.y,
            data=self.data,
            data_type=self.data_type,
            source=self.source,
            modifiers=self.modifiers,
        )

    @classmethod
    def drag_start(
        cls,
        x: float,
        y: float,
        source: Optional["Widget"] = None,
        data: Any = None,
        data_type: str = "",
    ) -> "DragEvent":
        """Create a drag start event."""
        return cls(
            event_type=EventType.DRAG_START,
            x=x,
            y=y,
            source=source,
            data=data,
            data_type=data_type,
        )

    @classmethod
    def drag(
        cls,
        x: float,
        y: float,
        source: Optional["Widget"] = None,
        data: Any = None,
        data_type: str = "",
    ) -> "DragEvent":
        """Create a drag event."""
        return cls(
            event_type=EventType.DRAG,
            x=x,
            y=y,
            source=source,
            data=data,
            data_type=data_type,
            bubbles=False,
        )

    @classmethod
    def drop(
        cls,
        x: float,
        y: float,
        source: Optional["Widget"] = None,
        data: Any = None,
        data_type: str = "",
    ) -> "DragEvent":
        """Create a drop event."""
        return cls(
            event_type=EventType.DROP,
            x=x,
            y=y,
            source=source,
            data=data,
            data_type=data_type,
        )


# Type alias for event handlers
EventHandler = Callable[[UIEvent], None]


class EventDispatcher:
    """
    Dispatches events through widget hierarchy.

    Supports capture and bubble phases.
    """

    __slots__ = ()

    @staticmethod
    def dispatch(event: UIEvent, target: "Widget") -> bool:
        """
        Dispatch an event to a target widget.

        Follows W3C event model:
        1. Capture phase: root -> target
        2. Target phase: at target
        3. Bubble phase: target -> root (if bubbles=True)

        Args:
            event: Event to dispatch.
            target: Target widget.

        Returns:
            True if event was not cancelled.
        """
        event.target = target

        # Build path from root to target
        path: list["Widget"] = []
        current: Optional["Widget"] = target
        while current is not None:
            path.insert(0, current)
            current = current.parent

        # Capture phase (root to target, excluding target)
        event.phase = EventPhase.CAPTURE
        for widget in path[:-1]:
            if event.is_stopped:
                break
            event.current_target = widget
            widget._dispatch_to_handlers(event, capture=True)

        # Target phase
        if not event.is_stopped:
            event.phase = EventPhase.TARGET
            event.current_target = target
            target._dispatch_to_handlers(event, capture=True)
            if not event.is_stopped_immediate:
                target._dispatch_to_handlers(event, capture=False)

        # Bubble phase (target to root, excluding target)
        if event.bubbles and not event.is_stopped:
            event.phase = EventPhase.BUBBLE
            for widget in reversed(path[:-1]):
                if event.is_stopped:
                    break
                event.current_target = widget
                widget._dispatch_to_handlers(event, capture=False)

        event.phase = EventPhase.NONE
        event.current_target = None

        return not event.is_default_prevented


__all__ = [
    # Enums
    "EventPhase",
    "MouseButton",
    "KeyModifier",
    "EventType",
    # Event classes
    "UIEvent",
    "MouseEvent",
    "KeyboardEvent",
    "FocusEvent",
    "DragEvent",
    # Types
    "EventHandler",
    # Dispatcher
    "EventDispatcher",
]
