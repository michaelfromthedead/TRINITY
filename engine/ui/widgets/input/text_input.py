"""
Text Input Widget Implementation.

A text input widget with support for:
- Single-line and multi-line modes
- Placeholder text
- Text selection (start, end)
- Cursor position and blinking
- Copy/paste support (clipboard integration)
- Input validation hook
- Max length constraint
- Password mode (masked characters)
- Input events (on_change, on_submit, on_focus)

Follows the Trinity Pattern with TrackedDescriptor for state changes
and ObservableDescriptor for event subscriptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from time import time
from typing import Any, Callable, Optional, Protocol, Union


class InputMode(Enum):
    """Input mode for the text input widget."""
    TEXT = auto()
    SINGLE_LINE = auto()
    MULTI_LINE = auto()
    PASSWORD = auto()
    NUMBER = auto()
    EMAIL = auto()


class TextInputState(Enum):
    """Visual interaction states for the text input."""
    NORMAL = auto()
    HOVERED = auto()
    FOCUSED = auto()
    DISABLED = auto()
    ERROR = auto()


@dataclass(slots=True)
class SelectionRange:
    """Represents a text selection range.

    Attributes:
        start: Start index of selection (inclusive)
        end: End index of selection (exclusive)
    """
    start: int
    end: int

    @property
    def length(self) -> int:
        """Get the length of the selection."""
        return abs(self.end - self.start)

    @property
    def is_empty(self) -> bool:
        """Check if selection is empty (cursor only)."""
        return self.start == self.end

    @property
    def is_collapsed(self) -> bool:
        """Check if selection is collapsed (cursor only, no selection)."""
        return self.start == self.end

    def normalized(self) -> "SelectionRange":
        """Get selection with start <= end."""
        if self.start <= self.end:
            return SelectionRange(self.start, self.end)
        return SelectionRange(self.end, self.start)


@dataclass(slots=True)
class ValidationResult:
    """Result of input validation.

    Attributes:
        is_valid: Whether the input is valid
        message: Optional error message if invalid
    """
    is_valid: bool
    message: Optional[str] = None

    @staticmethod
    def valid() -> "ValidationResult":
        """Create a valid result."""
        return ValidationResult(is_valid=True)

    @staticmethod
    def invalid(message: str) -> "ValidationResult":
        """Create an invalid result with error message."""
        return ValidationResult(is_valid=False, message=message)


class Validator(Protocol):
    """Protocol for text input validators."""

    def __call__(self, text: str) -> tuple[bool, Optional[str]]:
        """Validate the input text.

        Args:
            text: Text to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        ...


class ClipboardProvider(Protocol):
    """Protocol for clipboard operations."""

    def get_text(self) -> str:
        """Get text from clipboard."""
        ...

    def set_text(self, text: str) -> None:
        """Set text to clipboard."""
        ...


@dataclass(slots=True)
class TextInputStyle:
    """Style configuration for text input appearance.

    Attributes:
        background_color: Background color in normal state
        focused_background_color: Background color when focused
        error_background_color: Background color in error state
        text_color: Text foreground color
        placeholder_color: Placeholder text color
        selection_color: Selection highlight color
        cursor_color: Cursor color
        border_color: Border color in normal state
        focused_border_color: Border color when focused
        error_border_color: Border color in error state
        disabled_color: Background color when disabled
        border_width: Border thickness
        corner_radius: Corner rounding
        padding_horizontal: Horizontal padding
        padding_vertical: Vertical padding
        font_size: Text font size
        font_family: Font family name
        cursor_width: Cursor line width
        cursor_blink_rate: Cursor blink rate in seconds
        line_height: Line height multiplier for multi-line
        password_char: Character used for password masking
    """
    background_color: str = "#FFFFFF"
    focused_background_color: str = "#FFFFFF"
    error_background_color: str = "#FFF0F0"
    text_color: str = "#333333"
    placeholder_color: str = "#AAAAAA"
    selection_color: str = "#B4D7FF"
    cursor_color: str = "#333333"
    border_color: str = "#CCCCCC"
    focused_border_color: str = "#4A90D9"
    error_border_color: str = "#E53935"
    disabled_color: str = "#F5F5F5"
    border_width: float = 1.0
    corner_radius: float = 4.0
    padding_horizontal: float = 12.0
    padding_vertical: float = 8.0
    font_size: float = 14.0
    font_family: str = "default"
    cursor_width: float = 2.0
    cursor_blink_rate: float = 0.5
    line_height: float = 1.4
    password_char: str = "*"


@dataclass(slots=True)
class TextChangeEvent:
    """Event emitted when text content changes.

    Attributes:
        text_input: Reference to the text input widget
        timestamp: Time of the change
        new_text: New text content
        previous_text: Previous text content
        is_user_action: True if triggered by user interaction
    """
    text_input: "TextInput"
    timestamp: float
    new_text: str
    previous_text: str
    is_user_action: bool = True


@dataclass(slots=True)
class SelectionChangeEvent:
    """Event emitted when selection changes.

    Attributes:
        text_input: Reference to the text input widget
        timestamp: Time of the change
        selection: New selection range
    """
    text_input: "TextInput"
    timestamp: float
    selection: SelectionRange


@dataclass(slots=True)
class SubmitEvent:
    """Event emitted when text is submitted (Enter pressed in single-line).

    Attributes:
        text_input: Reference to the text input widget
        timestamp: Time of submission
        text: Submitted text content
    """
    text_input: "TextInput"
    timestamp: float
    text: str


@dataclass(slots=True)
class FocusEvent:
    """Event emitted when focus state changes.

    Attributes:
        text_input: Reference to the text input widget
        timestamp: Time of the event
        focused: True if focus gained, False if lost
    """
    text_input: "TextInput"
    timestamp: float
    focused: bool


class DefaultClipboard:
    """Default clipboard implementation (stores in memory)."""

    _text: str = ""

    @classmethod
    def get_text(cls) -> str:
        """Get text from clipboard."""
        return cls._text

    @classmethod
    def set_text(cls, text: str) -> None:
        """Set text to clipboard."""
        cls._text = text


class TextInput:
    """Interactive text input widget.

    A text input allows users to enter and edit text with support for
    selection, copy/paste, validation, and various input modes.

    Attributes:
        text: Current text content
        placeholder: Placeholder text shown when empty
        input_mode: Input mode (TEXT, PASSWORD, NUMBER, EMAIL)
        multiline: Whether multiline input is enabled
        max_length: Maximum allowed text length (0 for unlimited)
        enabled: Whether the input is interactive
        visible: Whether the input is rendered
        read_only: Whether text can be modified

    Events:
        on_text_change: Fired when text content changes
        on_selection_change: Fired when selection changes
        on_submit: Fired when Enter is pressed (single-line mode)
        on_focus: Fired when focus state changes

    Example:
        text_input = TextInput(placeholder="Enter name...")
        text_input.on_text_change(lambda e: print(f"Text: {e.new_text}"))
        text_input.on_submit(lambda e: print(f"Submitted: {e.text}"))
    """

    __slots__ = (
        '_id', '_text', '_placeholder', '_input_mode', '_multiline', '_max_length',
        '_enabled', '_visible', '_focusable', '_read_only',
        '_visual_state', '_style', '_validator', '_pattern', '_password_char',
        '_x', '_y', '_width', '_height',
        '_cursor_position', '_selection', '_selection_anchor',
        '_cursor_visible', '_cursor_blink_time',
        '_scroll_offset_x', '_scroll_offset_y',
        '_validation_result', '_clipboard',
        '_on_text_change_handlers', '_on_selection_change_handlers',
        '_on_submit_handlers', '_on_focus_handlers',
        '_is_hovered', '_is_focused', '_is_selecting',
        '_dirty', '_cached_mesh'
    )

    # Class-level ID counter
    _next_id: int = 0

    def __init__(
        self,
        text: str = "",
        placeholder: str = "",
        input_mode: InputMode = InputMode.TEXT,
        multiline: bool = False,
        max_length: int = 0,
        enabled: bool = True,
        visible: bool = True,
        read_only: bool = False,
        validator: Optional[Validator] = None,
        pattern: Optional[str] = None,
        password_char: str = "*",
        clipboard: Optional[ClipboardProvider] = None,
        style: Optional[TextInputStyle] = None,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 200.0,
        height: float = 40.0,
        # Compatibility aliases
        mode: Optional[InputMode] = None,
    ):
        """Initialize a text input widget.

        Args:
            text: Initial text content
            placeholder: Placeholder text
            input_mode: Input mode (TEXT, PASSWORD, NUMBER, EMAIL)
            multiline: Whether multiline input is enabled
            max_length: Maximum text length (0 for unlimited)
            enabled: Initial enabled state
            visible: Initial visibility
            read_only: Whether text is read-only
            validator: Optional validation function
            pattern: Optional regex pattern for validation
            password_char: Character used for password masking
            clipboard: Optional clipboard provider
            style: Style configuration
            x: X position
            y: Y position
            width: Widget width
            height: Widget height
            mode: Deprecated alias for input_mode
        """
        self._id = TextInput._next_id
        TextInput._next_id += 1

        self._placeholder = placeholder
        # Handle deprecated 'mode' parameter
        if mode is not None:
            self._input_mode = mode
        else:
            self._input_mode = input_mode
        self._multiline = multiline
        self._max_length = max_length
        self._enabled = enabled
        self._visible = visible
        self._focusable = True
        self._read_only = read_only
        self._visual_state = TextInputState.NORMAL if enabled else TextInputState.DISABLED
        self._style = style or TextInputStyle()
        self._validator = validator
        self._pattern = re.compile(pattern) if pattern else None
        self._password_char = password_char
        self._clipboard = clipboard or DefaultClipboard()

        self._x = x
        self._y = y
        self._width = width
        self._height = height

        # Set text after max_length is set
        self._text = self._constrain_text(text)
        self._cursor_position = len(self._text)
        self._selection = SelectionRange(self._cursor_position, self._cursor_position)
        self._selection_anchor: Optional[int] = None

        self._cursor_visible = True
        self._cursor_blink_time = 0.0

        self._scroll_offset_x = 0.0
        self._scroll_offset_y = 0.0

        self._on_text_change_handlers: list[Callable[[TextChangeEvent], None]] = []
        self._on_selection_change_handlers: list[Callable[[SelectionChangeEvent], None]] = []
        self._on_submit_handlers: list[Callable[[SubmitEvent], None]] = []
        self._on_focus_handlers: list[Callable[[FocusEvent], None]] = []

        self._is_hovered = False
        self._is_focused = False
        self._is_selecting = False

        # Run initial validation (after all state is initialized)
        self._validation_result = ValidationResult.valid()
        self._run_validation()

        self._dirty = True
        self._cached_mesh: Any = None

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the ID counter. Used for testing."""
        cls._next_id = 0

    def _constrain_text(self, text: str) -> str:
        """Constrain text to max_length and single-line if needed.

        Args:
            text: Text to constrain

        Returns:
            Constrained text
        """
        # Remove newlines in single-line and password modes
        if not self._multiline or self._input_mode == InputMode.PASSWORD:
            text = text.replace('\n', '').replace('\r', '')

        # Filter characters based on input mode
        if self._input_mode == InputMode.NUMBER:
            text = ''.join(c for c in text if c.isdigit() or c in '.-')

        # Apply max length
        if self._max_length > 0 and len(text) > self._max_length:
            text = text[:self._max_length]

        return text

    @property
    def id(self) -> int:
        """Get the unique widget ID."""
        return self._id

    @property
    def text(self) -> str:
        """Get the current text content."""
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """Set the text content programmatically."""
        new_text = self._constrain_text(value)
        if self._text != new_text:
            previous = self._text
            self._text = new_text
            self._cursor_position = min(self._cursor_position, len(new_text))
            self._selection = SelectionRange(self._cursor_position, self._cursor_position)
            self._run_validation()
            self._dirty = True
            self._emit_text_change(previous, is_user_action=False)

    @property
    def display_text(self) -> str:
        """Get the text to display (masked if password mode)."""
        if self._input_mode == InputMode.PASSWORD:
            return self._password_char * len(self._text)
        return self._text

    @property
    def placeholder(self) -> str:
        """Get the placeholder text."""
        return self._placeholder

    @placeholder.setter
    def placeholder(self, value: str) -> None:
        """Set the placeholder text."""
        if self._placeholder != value:
            self._placeholder = value
            self._dirty = True

    @property
    def input_mode(self) -> InputMode:
        """Get the input mode."""
        return self._input_mode

    @input_mode.setter
    def input_mode(self, value: InputMode) -> None:
        """Set the input mode."""
        if self._input_mode != value:
            self._input_mode = value
            # Constrain text for new mode
            self.text = self._text
            self._dirty = True

    @property
    def mode(self) -> InputMode:
        """Get the input mode (alias for input_mode)."""
        return self._input_mode

    @mode.setter
    def mode(self, value: InputMode) -> None:
        """Set the input mode (alias for input_mode)."""
        self.input_mode = value

    @property
    def multiline(self) -> bool:
        """Get whether multiline mode is enabled."""
        return self._multiline

    @multiline.setter
    def multiline(self, value: bool) -> None:
        """Set multiline mode."""
        if self._multiline != value:
            self._multiline = value
            # Constrain text for new mode
            self.text = self._text
            self._dirty = True

    @property
    def max_length(self) -> int:
        """Get the maximum text length."""
        return self._max_length

    @max_length.setter
    def max_length(self, value: int) -> None:
        """Set the maximum text length."""
        if value < 0:
            raise ValueError("max_length must be >= 0")
        if self._max_length != value:
            self._max_length = value
            # Constrain text to new max length
            self.text = self._text
            self._dirty = True

    @property
    def enabled(self) -> bool:
        """Check if input is enabled."""
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
        """Check if input is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set visibility."""
        if self._visible != value:
            self._visible = value
            self._dirty = True

    @property
    def focusable(self) -> bool:
        """Check if input can receive focus."""
        return self._focusable and self._enabled

    @focusable.setter
    def focusable(self, value: bool) -> None:
        """Set focusable state."""
        self._focusable = value

    @property
    def read_only(self) -> bool:
        """Check if input is read-only."""
        return self._read_only

    @read_only.setter
    def read_only(self, value: bool) -> None:
        """Set read-only state."""
        self._read_only = value

    @property
    def cursor_position(self) -> int:
        """Get the cursor position."""
        return self._cursor_position

    @cursor_position.setter
    def cursor_position(self, value: int) -> None:
        """Set the cursor position."""
        new_pos = max(0, min(len(self._text), value))
        if self._cursor_position != new_pos:
            self._cursor_position = new_pos
            self._selection = SelectionRange(new_pos, new_pos)
            self._reset_cursor_blink()
            self._dirty = True

    @property
    def selection(self) -> SelectionRange:
        """Get the current selection range."""
        return self._selection

    @selection.setter
    def selection(self, value: SelectionRange) -> None:
        """Set the selection range."""
        start = max(0, min(len(self._text), value.start))
        end = max(0, min(len(self._text), value.end))
        new_selection = SelectionRange(start, end)
        if self._selection.start != start or self._selection.end != end:
            self._selection = new_selection
            self._cursor_position = end
            self._dirty = True
            self._emit_selection_change()

    @property
    def selected_text(self) -> str:
        """Get the currently selected text."""
        sel = self._selection.normalized()
        return self._text[sel.start:sel.end]

    @property
    def has_selection(self) -> bool:
        """Check if there is a text selection."""
        return not self._selection.is_collapsed

    @property
    def is_valid(self) -> bool:
        """Check if current text passes validation."""
        return self._validation_result.is_valid

    @property
    def validation_error(self) -> Optional[str]:
        """Get the current validation error message."""
        return self._validation_result.message

    @property
    def visual_state(self) -> TextInputState:
        """Get current visual state."""
        return self._visual_state

    @property
    def style(self) -> TextInputStyle:
        """Get input style."""
        return self._style

    @style.setter
    def style(self, value: TextInputStyle) -> None:
        """Set input style."""
        self._style = value
        self._dirty = True

    @property
    def validator(self) -> Optional[Validator]:
        """Get the validator function."""
        return self._validator

    @validator.setter
    def validator(self, value: Optional[Validator]) -> None:
        """Set the validator function."""
        self._validator = value
        self._run_validation()
        self._dirty = True

    @property
    def line_count(self) -> int:
        """Get the number of lines in multiline mode."""
        if not self._multiline:
            return 1
        return self._text.count('\n') + 1

    @property
    def current_line(self) -> int:
        """Get the current line number (0-indexed)."""
        if not self._multiline:
            return 0
        text_before = self._text[:self._cursor_position]
        return text_before.count('\n')

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
    def text_area_bounds(self) -> tuple[float, float, float, float]:
        """Get the text area bounds (excluding padding)."""
        return (
            self._x + self._style.padding_horizontal,
            self._y + self._style.padding_vertical,
            self._width - 2 * self._style.padding_horizontal,
            self._height - 2 * self._style.padding_vertical,
        )

    @property
    def is_focused(self) -> bool:
        """Check if input has focus."""
        return self._is_focused

    @property
    def cursor_visible(self) -> bool:
        """Check if cursor should be visible (for blinking)."""
        return self._cursor_visible and self._is_focused

    @property
    def is_dirty(self) -> bool:
        """Check if input needs re-rendering."""
        return self._dirty

    def mark_clean(self) -> None:
        """Mark the input as rendered."""
        self._dirty = False

    def _update_visual_state(self) -> None:
        """Update visual state based on current conditions."""
        if not self._enabled:
            self._visual_state = TextInputState.DISABLED
        elif not self._validation_result.is_valid:
            self._visual_state = TextInputState.ERROR
        elif self._is_focused:
            self._visual_state = TextInputState.FOCUSED
        elif self._is_hovered:
            self._visual_state = TextInputState.HOVERED
        else:
            self._visual_state = TextInputState.NORMAL

    def _run_validation(self) -> None:
        """Run validation on current text."""
        # Check pattern first
        if self._pattern and self._text:
            if not self._pattern.fullmatch(self._text):
                self._validation_result = ValidationResult.invalid("Pattern mismatch")
                self._update_visual_state()
                return

        # Then check custom validator
        if self._validator:
            result = self._validator(self._text)
            if isinstance(result, tuple):
                is_valid, message = result
                self._validation_result = ValidationResult(is_valid=is_valid, message=message)
            else:
                self._validation_result = result
        else:
            self._validation_result = ValidationResult.valid()
        self._update_visual_state()

    def validate(self) -> ValidationResult:
        """Run validation and return the result."""
        self._run_validation()
        return self._validation_result

    def _reset_cursor_blink(self) -> None:
        """Reset cursor blink to visible state."""
        self._cursor_visible = True
        self._cursor_blink_time = 0.0

    def _emit_text_change(self, previous: str, is_user_action: bool = True) -> None:
        """Emit text change event to all handlers."""
        event = TextChangeEvent(
            text_input=self,
            timestamp=time(),
            new_text=self._text,
            previous_text=previous,
            is_user_action=is_user_action,
        )
        for handler in self._on_text_change_handlers:
            handler(event)

    def _emit_selection_change(self) -> None:
        """Emit selection change event to all handlers."""
        event = SelectionChangeEvent(
            text_input=self,
            timestamp=time(),
            selection=self._selection,
        )
        for handler in self._on_selection_change_handlers:
            handler(event)

    def _emit_submit(self) -> None:
        """Emit submit event to all handlers."""
        event = SubmitEvent(
            text_input=self,
            timestamp=time(),
            text=self._text,
        )
        for handler in self._on_submit_handlers:
            handler(event)

    def _emit_focus(self, focused: bool) -> None:
        """Emit focus event to all handlers."""
        event = FocusEvent(
            text_input=self,
            timestamp=time(),
            focused=focused,
        )
        for handler in self._on_focus_handlers:
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

    # Event subscription methods
    def on_text_change(self, handler: Callable[[TextChangeEvent], None]) -> Callable[[], None]:
        """Subscribe to text change events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_text_change_handlers.append(handler)
        return lambda: self._on_text_change_handlers.remove(handler)

    # Alias for compatibility
    def on_change(self, handler: Callable[[TextChangeEvent], None]) -> Callable[[], None]:
        """Subscribe to text change events (alias for on_text_change)."""
        return self.on_text_change(handler)

    def on_selection_change(self, handler: Callable[[SelectionChangeEvent], None]) -> Callable[[], None]:
        """Subscribe to selection change events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_selection_change_handlers.append(handler)
        return lambda: self._on_selection_change_handlers.remove(handler)

    def on_submit(self, handler: Callable[[SubmitEvent], None]) -> Callable[[], None]:
        """Subscribe to submit events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_submit_handlers.append(handler)
        return lambda: self._on_submit_handlers.remove(handler)

    def on_focus(self, handler: Callable[[FocusEvent], None]) -> Callable[[], None]:
        """Subscribe to focus events.

        Args:
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_focus_handlers.append(handler)
        return lambda: self._on_focus_handlers.remove(handler)

    def submit(self) -> None:
        """Trigger a submit event."""
        self._emit_submit()

    # Text manipulation methods
    def select_all(self) -> None:
        """Select all text."""
        self._selection = SelectionRange(0, len(self._text))
        self._cursor_position = len(self._text)
        self._dirty = True
        self._emit_selection_change()

    def select_range(self, start: int, end: int) -> None:
        """Select a range of text.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
        """
        start = max(0, min(len(self._text), start))
        end = max(0, min(len(self._text), end))
        self._selection = SelectionRange(start, end)
        self._cursor_position = end
        self._dirty = True
        self._emit_selection_change()

    def select_word(self) -> None:
        """Select the word at the cursor position."""
        pos = self._cursor_position
        length = len(self._text)

        # Find start of word
        start = pos
        while start > 0 and not self._text[start - 1].isspace():
            start -= 1

        # Find end of word
        end = pos
        while end < length and not self._text[end].isspace():
            end += 1

        self._selection = SelectionRange(start, end)
        self._cursor_position = end
        self._dirty = True
        self._emit_selection_change()

    def clear_selection(self) -> None:
        """Clear the current selection."""
        self._selection = SelectionRange(self._cursor_position, self._cursor_position)
        self._dirty = True
        self._emit_selection_change()

    def extend_selection(self, delta: int) -> None:
        """Extend the selection by delta characters.

        Args:
            delta: Number of characters to extend (negative for left)
        """
        # If no selection, start from cursor position
        if self._selection.is_collapsed:
            anchor = self._cursor_position
        else:
            # Use the start as anchor if extending left, end if extending right
            if delta < 0:
                anchor = self._selection.end
            else:
                anchor = self._selection.start

        new_cursor = max(0, min(len(self._text), self._cursor_position + delta))
        self._cursor_position = new_cursor
        self._selection = SelectionRange(anchor, new_cursor).normalized()
        self._dirty = True
        self._emit_selection_change()

    def delete_selection(self) -> str:
        """Delete selected text and return it.

        Returns:
            The deleted text
        """
        if self._selection.is_collapsed:
            return ""

        previous = self._text
        sel = self._selection.normalized()
        deleted = self._text[sel.start:sel.end]
        self._text = self._text[:sel.start] + self._text[sel.end:]
        self._cursor_position = sel.start
        self._selection = SelectionRange(sel.start, sel.start)
        self._run_validation()
        self._dirty = True
        self._emit_text_change(previous)
        return deleted

    def insert_text(self, text: str) -> None:
        """Insert text at cursor position, replacing selection.

        Args:
            text: Text to insert
        """
        if self._read_only:
            return

        previous = self._text

        # Delete selection first
        if self.has_selection:
            sel = self._selection.normalized()
            self._text = self._text[:sel.start] + self._text[sel.end:]
            self._cursor_position = sel.start
            self._selection = SelectionRange(sel.start, sel.start)

        # Constrain text for mode
        if not self._multiline or self._input_mode == InputMode.PASSWORD:
            text = text.replace('\n', '').replace('\r', '')

        # Filter characters based on input mode
        if self._input_mode == InputMode.NUMBER:
            text = ''.join(c for c in text if c.isdigit() or c in '.-')

        # Check max length
        if self._max_length > 0:
            available = self._max_length - len(self._text)
            text = text[:available]

        if not text:
            if previous != self._text:
                self._run_validation()
                self._dirty = True
                self._emit_text_change(previous)
            return

        # Insert text
        self._text = self._text[:self._cursor_position] + text + self._text[self._cursor_position:]
        self._cursor_position += len(text)
        self._selection = SelectionRange(self._cursor_position, self._cursor_position)

        self._run_validation()
        self._reset_cursor_blink()
        self._dirty = True
        self._emit_text_change(previous, is_user_action=True)

    def delete_backward(self) -> None:
        """Delete character before cursor (backspace)."""
        if self._read_only:
            return

        previous = self._text

        if self.has_selection:
            self.delete_selection()
            return
        elif self._cursor_position > 0:
            self._text = self._text[:self._cursor_position - 1] + self._text[self._cursor_position:]
            self._cursor_position -= 1
            self._selection = SelectionRange(self._cursor_position, self._cursor_position)

        if self._text != previous:
            self._run_validation()
            self._reset_cursor_blink()
            self._dirty = True
            self._emit_text_change(previous, is_user_action=True)

    # Alias for compatibility
    def delete_char_before(self) -> None:
        """Delete character before cursor (backspace). Alias for delete_backward."""
        self.delete_backward()

    def delete_forward(self) -> None:
        """Delete character after cursor (delete)."""
        if self._read_only:
            return

        previous = self._text

        if self.has_selection:
            self.delete_selection()
            return
        elif self._cursor_position < len(self._text):
            self._text = self._text[:self._cursor_position] + self._text[self._cursor_position + 1:]

        if self._text != previous:
            self._run_validation()
            self._reset_cursor_blink()
            self._dirty = True
            self._emit_text_change(previous, is_user_action=True)

    # Alias for compatibility
    def delete_char_after(self) -> None:
        """Delete character after cursor (delete). Alias for delete_forward."""
        self.delete_forward()

    def delete_word_backward(self) -> None:
        """Delete word before cursor."""
        if self._read_only:
            return

        previous = self._text

        if self.has_selection:
            self.delete_selection()
            return

        # Find start of word
        pos = self._cursor_position
        # Skip trailing whitespace
        while pos > 0 and self._text[pos - 1].isspace():
            pos -= 1
        # Skip word characters
        while pos > 0 and not self._text[pos - 1].isspace():
            pos -= 1

        if pos != self._cursor_position:
            self._text = self._text[:pos] + self._text[self._cursor_position:]
            self._cursor_position = pos
            self._selection = SelectionRange(pos, pos)
            self._run_validation()
            self._dirty = True
            self._emit_text_change(previous)

    def delete_word_forward(self) -> None:
        """Delete word after cursor."""
        if self._read_only:
            return

        previous = self._text

        if self.has_selection:
            self.delete_selection()
            return

        length = len(self._text)
        pos = self._cursor_position
        # Skip word characters
        while pos < length and not self._text[pos].isspace():
            pos += 1
        # Skip trailing whitespace
        while pos < length and self._text[pos].isspace():
            pos += 1

        if pos != self._cursor_position:
            self._text = self._text[:self._cursor_position] + self._text[pos:]
            self._run_validation()
            self._dirty = True
            self._emit_text_change(previous)

    def clear(self) -> None:
        """Clear all text."""
        if self._text:
            previous = self._text
            self._text = ""
            self._cursor_position = 0
            self._selection = SelectionRange(0, 0)
            self._run_validation()
            self._dirty = True
            self._emit_text_change(previous)

    def copy(self) -> Optional[str]:
        """Copy selected text to clipboard.

        Returns:
            The copied text, or empty string/None if in password mode
        """
        if self._input_mode == InputMode.PASSWORD:
            return "" if self.has_selection else None
        if self.has_selection:
            text = self.selected_text
            self._clipboard.set_text(text)
            return text
        return None

    def cut(self) -> Optional[str]:
        """Cut selected text to clipboard.

        Returns:
            The cut text, or None if nothing was cut
        """
        if self._read_only or self._input_mode == InputMode.PASSWORD:
            return None

        if self.has_selection:
            text = self.selected_text
            self._clipboard.set_text(text)
            self.delete_selection()
            return text
        return None

    def paste(self, text: Optional[str] = None) -> None:
        """Paste text at cursor position.

        Args:
            text: Text to paste, or None to use clipboard
        """
        if self._read_only:
            return

        if text is None:
            text = self._clipboard.get_text()
        if text:
            self.insert_text(text)

    # Cursor movement methods
    def move_cursor(self, delta: int) -> None:
        """Move cursor by delta characters.

        Args:
            delta: Number of characters to move (negative for left)
        """
        new_pos = max(0, min(len(self._text), self._cursor_position + delta))
        if new_pos != self._cursor_position:
            self._cursor_position = new_pos
            self._selection = SelectionRange(new_pos, new_pos)
            self._reset_cursor_blink()
            self._dirty = True

    def move_cursor_left(self, select: bool = False) -> None:
        """Move cursor one character left.

        Args:
            select: Whether to extend selection
        """
        if self._cursor_position > 0:
            self._cursor_position -= 1
            if select:
                self._selection = SelectionRange(self._selection.start, self._cursor_position)
            else:
                self._selection = SelectionRange(self._cursor_position, self._cursor_position)
            self._reset_cursor_blink()
            self._dirty = True

    def move_cursor_right(self, select: bool = False) -> None:
        """Move cursor one character right.

        Args:
            select: Whether to extend selection
        """
        if self._cursor_position < len(self._text):
            self._cursor_position += 1
            if select:
                self._selection = SelectionRange(self._selection.start, self._cursor_position)
            else:
                self._selection = SelectionRange(self._cursor_position, self._cursor_position)
            self._reset_cursor_blink()
            self._dirty = True

    def move_cursor_to_start(self, select: bool = False) -> None:
        """Move cursor to start of text.

        Args:
            select: Whether to extend selection
        """
        if self._cursor_position > 0:
            self._cursor_position = 0
            if select:
                self._selection = SelectionRange(self._selection.start, 0)
            else:
                self._selection = SelectionRange(0, 0)
            self._reset_cursor_blink()
            self._dirty = True

    def move_cursor_to_end(self, select: bool = False) -> None:
        """Move cursor to end of text.

        Args:
            select: Whether to extend selection
        """
        end = len(self._text)
        if self._cursor_position < end:
            self._cursor_position = end
            if select:
                self._selection = SelectionRange(self._selection.start, end)
            else:
                self._selection = SelectionRange(end, end)
            self._reset_cursor_blink()
            self._dirty = True

    def move_cursor_by_word(self, direction: int) -> None:
        """Move cursor by word.

        Args:
            direction: -1 for previous word, 1 for next word
        """
        if direction < 0:
            self.move_cursor_word_left()
        else:
            self.move_cursor_word_right()

    def move_cursor_word_left(self, select: bool = False) -> None:
        """Move cursor to start of previous word.

        Args:
            select: Whether to extend selection
        """
        pos = self._cursor_position
        # Skip whitespace
        while pos > 0 and self._text[pos - 1].isspace():
            pos -= 1
        # Skip word characters
        while pos > 0 and not self._text[pos - 1].isspace():
            pos -= 1

        if pos != self._cursor_position:
            self._cursor_position = pos
            if select:
                self._selection = SelectionRange(self._selection.start, pos)
            else:
                self._selection = SelectionRange(pos, pos)
            self._reset_cursor_blink()
            self._dirty = True

    def move_cursor_word_right(self, select: bool = False) -> None:
        """Move cursor to end of current word.

        Args:
            select: Whether to extend selection
        """
        pos = self._cursor_position
        length = len(self._text)
        # Skip current word characters to reach end of word
        while pos < length and not self._text[pos].isspace():
            pos += 1

        if pos != self._cursor_position:
            self._cursor_position = pos
            if select:
                self._selection = SelectionRange(self._selection.start, pos)
            else:
                self._selection = SelectionRange(pos, pos)
            self._reset_cursor_blink()
            self._dirty = True

    def move_cursor_line(self, direction: int) -> None:
        """Move cursor up or down a line in multiline mode.

        Args:
            direction: -1 for up, 1 for down
        """
        if not self._multiline:
            return

        lines = self._text.split('\n')
        current_line = self.current_line

        # Calculate column position
        text_before = self._text[:self._cursor_position]
        lines_before = text_before.split('\n')
        current_col = len(lines_before[-1])

        target_line = current_line + direction
        if target_line < 0 or target_line >= len(lines):
            return

        # Calculate new position
        new_pos = sum(len(lines[i]) + 1 for i in range(target_line))
        new_pos += min(current_col, len(lines[target_line]))
        new_pos = min(new_pos, len(self._text))

        if new_pos != self._cursor_position:
            self._cursor_position = new_pos
            self._selection = SelectionRange(new_pos, new_pos)
            self._reset_cursor_blink()
            self._dirty = True

    # Update method for cursor blinking
    def update(self, delta_time: float) -> None:
        """Update the widget (for cursor blinking).

        Args:
            delta_time: Time elapsed since last update in seconds
        """
        if self._is_focused:
            self._cursor_blink_time += delta_time
            if self._cursor_blink_time >= self._style.cursor_blink_rate:
                self._cursor_blink_time = 0.0
                self._cursor_visible = not self._cursor_visible
                self._dirty = True

    # Input event handlers
    def handle_mouse_enter(self) -> None:
        """Handle mouse entering the input area."""
        if not self._enabled:
            return
        self._is_hovered = True
        self._update_visual_state()
        self._dirty = True

    def handle_mouse_leave(self) -> None:
        """Handle mouse leaving the input area."""
        self._is_hovered = False
        self._update_visual_state()
        self._dirty = True

    def handle_mouse_down(self, x: float, y: float, shift: bool = False) -> bool:
        """Handle mouse button press.

        Args:
            x: Mouse X position
            y: Mouse Y position
            shift: Whether shift is held

        Returns:
            True if event was consumed
        """
        if not self._enabled or not self.contains_point(x, y):
            return False

        # Calculate cursor position from click
        pos = self._position_to_cursor(x, y)

        if shift and self._is_focused:
            # Extend selection
            self._selection = SelectionRange(self._selection.start, pos)
        else:
            # Start new selection
            self._selection_anchor = pos
            self._selection = SelectionRange(pos, pos)

        self._cursor_position = pos
        self._is_selecting = True
        self._reset_cursor_blink()
        self._dirty = True
        return True

    def handle_mouse_move(self, x: float, y: float) -> bool:
        """Handle mouse movement (drag for selection).

        Args:
            x: Mouse X position
            y: Mouse Y position

        Returns:
            True if event was consumed
        """
        if not self._is_selecting:
            return False

        pos = self._position_to_cursor(x, y)
        if self._selection_anchor is not None:
            self._selection = SelectionRange(self._selection_anchor, pos)
            self._cursor_position = pos
            self._dirty = True
            return True
        return False

    def handle_mouse_up(self, x: float, y: float) -> bool:
        """Handle mouse button release.

        Args:
            x: Mouse X position
            y: Mouse Y position

        Returns:
            True if event was consumed
        """
        self._is_selecting = False
        self._selection_anchor = None
        return True

    def handle_double_click(self, x: float, y: float) -> bool:
        """Handle double-click (select word).

        Args:
            x: Mouse X position
            y: Mouse Y position

        Returns:
            True if event was consumed
        """
        if not self._enabled or not self.contains_point(x, y):
            return False

        pos = self._position_to_cursor(x, y)

        # Find word boundaries
        start = pos
        end = pos
        length = len(self._text)

        # Find start of word
        while start > 0 and not self._text[start - 1].isspace():
            start -= 1

        # Find end of word
        while end < length and not self._text[end].isspace():
            end += 1

        self._selection = SelectionRange(start, end)
        self._cursor_position = end
        self._dirty = True
        return True

    def handle_triple_click(self, x: float, y: float) -> bool:
        """Handle triple-click (select all or line).

        Args:
            x: Mouse X position
            y: Mouse Y position

        Returns:
            True if event was consumed
        """
        if not self._enabled or not self.contains_point(x, y):
            return False

        if self._multiline:
            # Select current line
            pos = self._position_to_cursor(x, y)
            start = self._text.rfind('\n', 0, pos) + 1
            end = self._text.find('\n', pos)
            if end == -1:
                end = len(self._text)
            self._selection = SelectionRange(start, end)
            self._cursor_position = end
        else:
            # Select all
            self.select_all()

        self._dirty = True
        return True

    def _position_to_cursor(self, x: float, y: float) -> int:
        """Convert screen position to cursor position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Cursor position index
        """
        # Simple approximation - in real implementation this would use font metrics
        text_area = self.text_area_bounds
        char_width = self._style.font_size * 0.6  # Rough approximation

        if self._multiline:
            # Calculate line and column
            line_height = self._style.font_size * self._style.line_height
            relative_y = y - text_area[1] + self._scroll_offset_y
            line_index = int(relative_y / line_height)

            lines = self._text.split('\n')
            line_index = max(0, min(len(lines) - 1, line_index))

            relative_x = x - text_area[0] + self._scroll_offset_x
            col = int(relative_x / char_width)
            col = max(0, min(len(lines[line_index]), col))

            # Calculate position in full text
            pos = sum(len(lines[i]) + 1 for i in range(line_index)) + col
            return min(pos, len(self._text))
        else:
            relative_x = x - text_area[0] + self._scroll_offset_x
            pos = int(relative_x / char_width)
            return max(0, min(len(self._text), pos))

    def handle_focus_gained(self) -> None:
        """Handle receiving keyboard focus."""
        if not self._enabled:
            return
        self._is_focused = True
        self._reset_cursor_blink()
        self._update_visual_state()
        self._dirty = True
        self._emit_focus(True)

    def handle_focus_lost(self) -> None:
        """Handle losing keyboard focus."""
        self._is_focused = False
        self._is_selecting = False
        self._update_visual_state()
        self._dirty = True
        self._emit_focus(False)

    def handle_key_down(
        self,
        key: str,
        shift: bool = False,
        ctrl: bool = False,
        alt: bool = False,
    ) -> bool:
        """Handle keyboard key press.

        Args:
            key: Key identifier
            shift: Shift modifier state
            ctrl: Ctrl modifier state
            alt: Alt modifier state

        Returns:
            True if event was consumed
        """
        if not self._enabled:
            return False

        # Handle control shortcuts
        if ctrl:
            if key == "a":
                self.select_all()
                return True
            elif key == "c":
                self.copy()
                return True
            elif key == "x":
                self.cut()
                return True
            elif key == "v":
                self.paste()
                return True
            elif key == "left":
                self.move_cursor_word_left(select=shift)
                return True
            elif key == "right":
                self.move_cursor_word_right(select=shift)
                return True
            elif key == "home":
                self.move_cursor_to_start(select=shift)
                return True
            elif key == "end":
                self.move_cursor_to_end(select=shift)
                return True
            return False

        # Handle navigation keys
        if key == "left":
            if not shift and self.has_selection:
                # Move to start of selection
                self._cursor_position = self.selection.start
                self._selection = SelectionRange(self._cursor_position, self._cursor_position)
                self._dirty = True
            else:
                self.move_cursor_left(select=shift)
            return True
        elif key == "right":
            if not shift and self.has_selection:
                # Move to end of selection
                self._cursor_position = self.selection.end
                self._selection = SelectionRange(self._cursor_position, self._cursor_position)
                self._dirty = True
            else:
                self.move_cursor_right(select=shift)
            return True
        elif key == "home":
            self.move_cursor_to_start(select=shift)
            return True
        elif key == "end":
            self.move_cursor_to_end(select=shift)
            return True
        elif key == "up" and self._multiline:
            self.move_cursor_line(-1)
            return True
        elif key == "down" and self._multiline:
            self.move_cursor_line(1)
            return True
        elif key == "backspace":
            self.delete_backward()
            return True
        elif key == "delete":
            self.delete_forward()
            return True
        elif key in ("enter", "return"):
            if self._multiline:
                self.insert_text("\n")
            else:
                self._emit_submit()
            return True
        elif key == "tab":
            if self._multiline:
                self.insert_text("\t")
                return True
            return False
        elif key == "escape":
            self.clear_selection()
            return True

        return False

    def handle_text_input(self, text: str) -> bool:
        """Handle text input (printable characters).

        Args:
            text: Input text

        Returns:
            True if event was consumed
        """
        if not self._enabled or self._read_only:
            return False

        if text and text.isprintable():
            self.insert_text(text)
            return True
        return False

    # Visual state helpers
    def get_current_background_color(self) -> str:
        """Get the background color for current state.

        Returns:
            Color string for current state
        """
        if self._visual_state == TextInputState.DISABLED:
            return self._style.disabled_color
        elif self._visual_state == TextInputState.ERROR:
            return self._style.error_background_color
        elif self._visual_state == TextInputState.FOCUSED:
            return self._style.focused_background_color
        else:
            return self._style.background_color

    def get_current_border_color(self) -> str:
        """Get the border color for current state.

        Returns:
            Border color string for current state
        """
        if self._visual_state == TextInputState.DISABLED:
            return self._style.disabled_color
        elif self._visual_state == TextInputState.ERROR:
            return self._style.error_border_color
        elif self._visual_state == TextInputState.FOCUSED:
            return self._style.focused_border_color
        else:
            return self._style.border_color

    def get_cursor_position_px(self) -> tuple[float, float]:
        """Get the cursor position in pixels.

        Returns:
            (x, y) position of cursor
        """
        text_area = self.text_area_bounds
        char_width = self._style.font_size * 0.6

        if self._multiline:
            # Calculate line and column
            text_before = self._text[:self._cursor_position]
            lines = text_before.split('\n')
            line = len(lines) - 1
            col = len(lines[-1])

            line_height = self._style.font_size * self._style.line_height
            x = text_area[0] + col * char_width - self._scroll_offset_x
            y = text_area[1] + line * line_height - self._scroll_offset_y
        else:
            x = text_area[0] + self._cursor_position * char_width - self._scroll_offset_x
            y = text_area[1]

        return (x, y)
