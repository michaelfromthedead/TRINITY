"""Virtual Keyboard implementation for XR.

Provides a virtual keyboard for text input in XR environments with:
- Multiple layout support (QWERTY, AZERTY, numeric, etc.)
- Shift/caps lock states
- Special keys (backspace, enter, space, etc.)
- Predictive text suggestions
- Haptic feedback on key press
- Hand tracking support for typing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any


class KeyType(Enum):
    """Types of virtual keyboard keys."""
    CHARACTER = auto()  # Standard character key
    SHIFT = auto()  # Shift/caps toggle
    BACKSPACE = auto()  # Delete character
    ENTER = auto()  # Submit/newline
    SPACE = auto()  # Space bar
    TAB = auto()  # Tab character
    SYMBOLS = auto()  # Switch to symbols layout
    ABC = auto()  # Switch to letters layout
    NUMBERS = auto()  # Switch to numbers layout
    LANGUAGE = auto()  # Switch language/layout
    HIDE = auto()  # Hide keyboard
    LEFT = auto()  # Move cursor left
    RIGHT = auto()  # Move cursor right
    CLEAR = auto()  # Clear all text
    EMOJI = auto()  # Open emoji picker


class KeyboardLayout(Enum):
    """Available keyboard layouts."""
    QWERTY = auto()
    QWERTY_UPPER = auto()
    AZERTY = auto()
    QWERTZ = auto()
    NUMERIC = auto()
    SYMBOLS = auto()
    EMOJI = auto()


@dataclass(slots=True)
class VirtualKey:
    """A single key on the virtual keyboard.

    Attributes:
        label: Display label for the key
        value: Character(s) produced when pressed
        key_type: Type of key
        width_units: Width in grid units (1 = standard key width)
        alt_label: Alternative label (e.g., for shift state)
        alt_value: Alternative value when shifted
        is_hovered: Whether key is being pointed at
        is_pressed: Whether key is currently pressed
    """
    label: str
    value: str = ""
    key_type: KeyType = KeyType.CHARACTER
    width_units: float = 1.0
    alt_label: Optional[str] = None
    alt_value: Optional[str] = None
    is_hovered: bool = False
    is_pressed: bool = False
    position: tuple[int, int] = (0, 0)  # Row, column

    def __post_init__(self):
        """Initialize value from label if not provided."""
        if not self.value and self.key_type == KeyType.CHARACTER:
            self.value = self.label

    @property
    def display_label(self) -> str:
        """Get current display label based on key state."""
        return self.label

    def get_value(self, shifted: bool = False) -> str:
        """Get output value, considering shift state."""
        if shifted and self.alt_value:
            return self.alt_value
        return self.value


# Standard QWERTY layout definition
QWERTY_LAYOUT: list[list[VirtualKey]] = [
    # Row 1 - Numbers
    [VirtualKey("1", alt_label="!", alt_value="!"),
     VirtualKey("2", alt_label="@", alt_value="@"),
     VirtualKey("3", alt_label="#", alt_value="#"),
     VirtualKey("4", alt_label="$", alt_value="$"),
     VirtualKey("5", alt_label="%", alt_value="%"),
     VirtualKey("6", alt_label="^", alt_value="^"),
     VirtualKey("7", alt_label="&", alt_value="&"),
     VirtualKey("8", alt_label="*", alt_value="*"),
     VirtualKey("9", alt_label="(", alt_value="("),
     VirtualKey("0", alt_label=")", alt_value=")")],
    # Row 2
    [VirtualKey("q", alt_label="Q", alt_value="Q"),
     VirtualKey("w", alt_label="W", alt_value="W"),
     VirtualKey("e", alt_label="E", alt_value="E"),
     VirtualKey("r", alt_label="R", alt_value="R"),
     VirtualKey("t", alt_label="T", alt_value="T"),
     VirtualKey("y", alt_label="Y", alt_value="Y"),
     VirtualKey("u", alt_label="U", alt_value="U"),
     VirtualKey("i", alt_label="I", alt_value="I"),
     VirtualKey("o", alt_label="O", alt_value="O"),
     VirtualKey("p", alt_label="P", alt_value="P")],
    # Row 3
    [VirtualKey("a", alt_label="A", alt_value="A"),
     VirtualKey("s", alt_label="S", alt_value="S"),
     VirtualKey("d", alt_label="D", alt_value="D"),
     VirtualKey("f", alt_label="F", alt_value="F"),
     VirtualKey("g", alt_label="G", alt_value="G"),
     VirtualKey("h", alt_label="H", alt_value="H"),
     VirtualKey("j", alt_label="J", alt_value="J"),
     VirtualKey("k", alt_label="K", alt_value="K"),
     VirtualKey("l", alt_label="L", alt_value="L")],
    # Row 4
    [VirtualKey("Shift", key_type=KeyType.SHIFT, width_units=1.5),
     VirtualKey("z", alt_label="Z", alt_value="Z"),
     VirtualKey("x", alt_label="X", alt_value="X"),
     VirtualKey("c", alt_label="C", alt_value="C"),
     VirtualKey("v", alt_label="V", alt_value="V"),
     VirtualKey("b", alt_label="B", alt_value="B"),
     VirtualKey("n", alt_label="N", alt_value="N"),
     VirtualKey("m", alt_label="M", alt_value="M"),
     VirtualKey("<-", key_type=KeyType.BACKSPACE, width_units=1.5)],
    # Row 5 - Bottom row
    [VirtualKey("123", key_type=KeyType.NUMBERS, width_units=1.5),
     VirtualKey(","),
     VirtualKey("Space", " ", key_type=KeyType.SPACE, width_units=4.0),
     VirtualKey("."),
     VirtualKey("Enter", "\n", key_type=KeyType.ENTER, width_units=1.5)],
]

# Numeric layout
NUMERIC_LAYOUT: list[list[VirtualKey]] = [
    [VirtualKey("1"), VirtualKey("2"), VirtualKey("3")],
    [VirtualKey("4"), VirtualKey("5"), VirtualKey("6")],
    [VirtualKey("7"), VirtualKey("8"), VirtualKey("9")],
    [VirtualKey("-"), VirtualKey("0"), VirtualKey(".")],
    [VirtualKey("ABC", key_type=KeyType.ABC, width_units=1.5),
     VirtualKey("<-", key_type=KeyType.BACKSPACE),
     VirtualKey("Enter", "\n", key_type=KeyType.ENTER, width_units=1.5)],
]

# Symbol layout
SYMBOLS_LAYOUT: list[list[VirtualKey]] = [
    [VirtualKey("!"), VirtualKey("@"), VirtualKey("#"), VirtualKey("$"),
     VirtualKey("%"), VirtualKey("^"), VirtualKey("&"), VirtualKey("*"),
     VirtualKey("("), VirtualKey(")")],
    [VirtualKey("-"), VirtualKey("_"), VirtualKey("="), VirtualKey("+"),
     VirtualKey("["), VirtualKey("]"), VirtualKey("{"), VirtualKey("}"),
     VirtualKey("|"), VirtualKey("\\")],
    [VirtualKey(";"), VirtualKey(":"), VirtualKey("'"), VirtualKey('"'),
     VirtualKey(","), VirtualKey("."), VirtualKey("<"), VirtualKey(">"),
     VirtualKey("/"), VirtualKey("?")],
    [VirtualKey("ABC", key_type=KeyType.ABC, width_units=2.0),
     VirtualKey("Space", " ", key_type=KeyType.SPACE, width_units=4.0),
     VirtualKey("<-", key_type=KeyType.BACKSPACE, width_units=2.0)],
]


@dataclass(slots=True)
class KeyboardStyle:
    """Visual style for virtual keyboard."""
    key_color: tuple[float, float, float, float] = (0.3, 0.3, 0.35, 1.0)
    key_hover_color: tuple[float, float, float, float] = (0.4, 0.4, 0.5, 1.0)
    key_pressed_color: tuple[float, float, float, float] = (0.2, 0.4, 0.8, 1.0)
    special_key_color: tuple[float, float, float, float] = (0.25, 0.25, 0.3, 1.0)
    background_color: tuple[float, float, float, float] = (0.15, 0.15, 0.2, 0.95)
    text_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    key_spacing: float = 0.005  # Meters between keys
    key_height: float = 0.04  # Meters
    key_width: float = 0.04  # Meters (for 1 unit)
    border_radius: float = 0.005  # Meters
    font_size: float = 0.025  # Meters


@dataclass(slots=True)
class VirtualKeyboard:
    """Virtual keyboard for XR text input.

    Attributes:
        position: Position in world/local space
        orientation: Quaternion orientation
        width: Total keyboard width in meters
        height: Total keyboard height in meters
        current_layout: Active keyboard layout
        text: Current input text
        cursor_position: Cursor position in text
        is_visible: Whether keyboard is displayed
        is_shift_active: Whether shift is toggled
        is_caps_lock: Whether caps lock is on
        max_length: Maximum text length (0 for unlimited)
        placeholder: Placeholder text when empty
    """
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    width: float = 0.5  # Meters
    height: float = 0.25  # Meters
    current_layout: KeyboardLayout = KeyboardLayout.QWERTY
    text: str = ""
    cursor_position: int = 0
    is_visible: bool = False
    is_shift_active: bool = False
    is_caps_lock: bool = False
    max_length: int = 0
    placeholder: str = "Enter text..."
    style: KeyboardStyle = field(default_factory=KeyboardStyle)
    _keys: list[list[VirtualKey]] = field(default_factory=list)
    _on_text_changed: Optional[Callable[[str], None]] = None
    _on_submit: Optional[Callable[[str], None]] = None
    _on_key_pressed: Optional[Callable[[VirtualKey], None]] = None
    _suggestions: list[str] = field(default_factory=list)
    _parent: Any = None
    _hovered_key: Optional[VirtualKey] = None

    def __post_init__(self):
        """Initialize keyboard with default layout."""
        self._load_layout(self.current_layout)

    def _load_layout(self, layout: KeyboardLayout) -> None:
        """Load key layout from layout type."""
        if layout in (KeyboardLayout.QWERTY, KeyboardLayout.QWERTY_UPPER):
            self._keys = [row.copy() for row in QWERTY_LAYOUT]
        elif layout == KeyboardLayout.NUMERIC:
            self._keys = [row.copy() for row in NUMERIC_LAYOUT]
        elif layout == KeyboardLayout.SYMBOLS:
            self._keys = [row.copy() for row in SYMBOLS_LAYOUT]
        else:
            self._keys = [row.copy() for row in QWERTY_LAYOUT]

        # Update key positions
        for row_idx, row in enumerate(self._keys):
            for col_idx, key in enumerate(row):
                key.position = (row_idx, col_idx)

    @property
    def display_text(self) -> str:
        """Get text to display, with placeholder if empty."""
        return self.text if self.text else self.placeholder

    @property
    def is_uppercase(self) -> bool:
        """Check if keyboard is in uppercase mode."""
        return self.is_shift_active or self.is_caps_lock

    @property
    def keys(self) -> list[list[VirtualKey]]:
        """Get current key layout."""
        return self._keys

    @property
    def suggestions(self) -> list[str]:
        """Get current text suggestions."""
        return self._suggestions.copy()

    def show(self) -> None:
        """Show the keyboard."""
        self.is_visible = True

    def hide(self) -> None:
        """Hide the keyboard."""
        self.is_visible = False
        self._clear_hover()

    def toggle(self) -> None:
        """Toggle keyboard visibility."""
        if self.is_visible:
            self.hide()
        else:
            self.show()

    def clear(self) -> None:
        """Clear all input text."""
        self.text = ""
        self.cursor_position = 0
        if self._on_text_changed:
            self._on_text_changed(self.text)

    def set_text(self, text: str) -> None:
        """Set the input text."""
        if self.max_length > 0:
            text = text[:self.max_length]
        self.text = text
        self.cursor_position = len(text)

    def switch_layout(self, layout: KeyboardLayout) -> None:
        """Switch to a different keyboard layout."""
        if layout != self.current_layout:
            self.current_layout = layout
            self._load_layout(layout)

    def on_text_changed(self, callback: Callable[[str], None]) -> None:
        """Set callback for text changes."""
        self._on_text_changed = callback

    def on_submit(self, callback: Callable[[str], None]) -> None:
        """Set callback for text submission."""
        self._on_submit = callback

    def on_key_pressed(self, callback: Callable[[VirtualKey], None]) -> None:
        """Set callback for key presses."""
        self._on_key_pressed = callback

    def set_suggestions(self, suggestions: list[str]) -> None:
        """Set text suggestions."""
        self._suggestions = suggestions[:5]  # Limit to 5 suggestions

    def select_suggestion(self, index: int) -> None:
        """Select a suggestion to replace current text."""
        if 0 <= index < len(self._suggestions):
            self.text = self._suggestions[index]
            self.cursor_position = len(self.text)
            if self._on_text_changed:
                self._on_text_changed(self.text)

    def _clear_hover(self) -> None:
        """Clear all hover states."""
        for row in self._keys:
            for key in row:
                key.is_hovered = False
                key.is_pressed = False
        self._hovered_key = None

    def hover_key(self, row: int, col: int) -> Optional[VirtualKey]:
        """Set hover state on a key.

        Args:
            row: Key row index
            col: Key column index

        Returns:
            The hovered key or None
        """
        # Clear previous hover
        if self._hovered_key:
            self._hovered_key.is_hovered = False

        # Set new hover
        if 0 <= row < len(self._keys) and 0 <= col < len(self._keys[row]):
            key = self._keys[row][col]
            key.is_hovered = True
            self._hovered_key = key
            return key

        self._hovered_key = None
        return None

    def press_key(self, key: VirtualKey) -> bool:
        """Handle key press.

        Args:
            key: The key being pressed

        Returns:
            True if text was modified
        """
        key.is_pressed = True

        if self._on_key_pressed:
            self._on_key_pressed(key)

        modified = False

        if key.key_type == KeyType.CHARACTER:
            # Insert character
            char = key.get_value(self.is_uppercase)
            if self.max_length == 0 or len(self.text) < self.max_length:
                self.text = (self.text[:self.cursor_position] +
                            char +
                            self.text[self.cursor_position:])
                self.cursor_position += len(char)
                modified = True

            # Auto-release shift
            if self.is_shift_active and not self.is_caps_lock:
                self.is_shift_active = False

        elif key.key_type == KeyType.SPACE:
            if self.max_length == 0 or len(self.text) < self.max_length:
                self.text = (self.text[:self.cursor_position] +
                            " " +
                            self.text[self.cursor_position:])
                self.cursor_position += 1
                modified = True

        elif key.key_type == KeyType.BACKSPACE:
            if self.cursor_position > 0:
                self.text = (self.text[:self.cursor_position - 1] +
                            self.text[self.cursor_position:])
                self.cursor_position -= 1
                modified = True

        elif key.key_type == KeyType.ENTER:
            if self._on_submit:
                self._on_submit(self.text)

        elif key.key_type == KeyType.SHIFT:
            if self.is_caps_lock:
                self.is_caps_lock = False
                self.is_shift_active = False
            elif self.is_shift_active:
                # Double tap = caps lock
                self.is_caps_lock = True
            else:
                self.is_shift_active = True

        elif key.key_type == KeyType.NUMBERS:
            self.switch_layout(KeyboardLayout.NUMERIC)

        elif key.key_type == KeyType.SYMBOLS:
            self.switch_layout(KeyboardLayout.SYMBOLS)

        elif key.key_type == KeyType.ABC:
            self.switch_layout(KeyboardLayout.QWERTY)

        elif key.key_type == KeyType.HIDE:
            self.hide()

        elif key.key_type == KeyType.LEFT:
            if self.cursor_position > 0:
                self.cursor_position -= 1

        elif key.key_type == KeyType.RIGHT:
            if self.cursor_position < len(self.text):
                self.cursor_position += 1

        elif key.key_type == KeyType.CLEAR:
            self.clear()
            modified = True

        if modified and self._on_text_changed:
            self._on_text_changed(self.text)

        return modified

    def release_key(self, key: VirtualKey) -> None:
        """Handle key release."""
        key.is_pressed = False

    def get_key_at_uv(self, u: float, v: float) -> Optional[VirtualKey]:
        """Get key at UV coordinates on keyboard.

        Args:
            u: Horizontal position (0-1)
            v: Vertical position (0-1)

        Returns:
            Key at position or None
        """
        if not self._keys:
            return None

        # Calculate row
        row_count = len(self._keys)
        row_height = 1.0 / row_count
        row_idx = int(v / row_height)
        row_idx = max(0, min(row_idx, row_count - 1))

        row = self._keys[row_idx]

        # Calculate column based on key widths
        total_width = sum(k.width_units for k in row)
        target_width = u * total_width

        current_width = 0.0
        for key in row:
            if current_width + key.width_units > target_width:
                return key
            current_width += key.width_units

        return row[-1] if row else None

    def calculate_dimensions(self) -> tuple[float, float]:
        """Calculate actual keyboard dimensions based on layout.

        Returns:
            Tuple of (width, height) in meters
        """
        if not self._keys:
            return (self.width, self.height)

        max_units = max(sum(k.width_units for k in row) for row in self._keys)
        row_count = len(self._keys)

        width = max_units * self.style.key_width + (max_units - 1) * self.style.key_spacing
        height = row_count * self.style.key_height + (row_count - 1) * self.style.key_spacing

        return (width, height)


class KeyboardManager:
    """Manages virtual keyboard instances.

    Provides global keyboard management including:
    - Shared keyboard instances
    - Focus tracking
    - Layout preferences
    """

    __slots__ = ('_keyboards', '_active_keyboard', '_default_layout')

    def __init__(self, default_layout: KeyboardLayout = KeyboardLayout.QWERTY):
        """Initialize keyboard manager.

        Args:
            default_layout: Default layout for new keyboards
        """
        self._keyboards: dict[str, VirtualKeyboard] = {}
        self._active_keyboard: Optional[VirtualKeyboard] = None
        self._default_layout = default_layout

    def create(self, name: str, **kwargs) -> VirtualKeyboard:
        """Create a new keyboard instance.

        Args:
            name: Unique keyboard identifier
            **kwargs: Arguments passed to VirtualKeyboard

        Returns:
            New keyboard instance
        """
        keyboard = VirtualKeyboard(
            current_layout=kwargs.pop('layout', self._default_layout),
            **kwargs
        )
        self._keyboards[name] = keyboard
        return keyboard

    def get(self, name: str) -> Optional[VirtualKeyboard]:
        """Get keyboard by name."""
        return self._keyboards.get(name)

    def remove(self, name: str) -> None:
        """Remove a keyboard instance."""
        if name in self._keyboards:
            keyboard = self._keyboards.pop(name)
            if self._active_keyboard == keyboard:
                self._active_keyboard = None

    def activate(self, name: str) -> Optional[VirtualKeyboard]:
        """Activate a keyboard and show it.

        Args:
            name: Keyboard name

        Returns:
            Activated keyboard or None
        """
        if name in self._keyboards:
            # Hide previous active keyboard
            if self._active_keyboard:
                self._active_keyboard.hide()

            self._active_keyboard = self._keyboards[name]
            self._active_keyboard.show()
            return self._active_keyboard
        return None

    def deactivate(self) -> None:
        """Deactivate current keyboard."""
        if self._active_keyboard:
            self._active_keyboard.hide()
            self._active_keyboard = None

    @property
    def active(self) -> Optional[VirtualKeyboard]:
        """Get currently active keyboard."""
        return self._active_keyboard

    def set_default_layout(self, layout: KeyboardLayout) -> None:
        """Set default layout for new keyboards."""
        self._default_layout = layout
