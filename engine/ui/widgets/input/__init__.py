"""
UI Input Widgets Module.

This module exports all interactive input widgets for the UI system.
These widgets follow the Trinity Pattern with Foundation runtime integration.

Widgets:
    - Button: Clickable button with icon/text support and toggle mode
    - Checkbox: Boolean toggle with checked/unchecked/indeterminate states
    - Slider: Range value input with horizontal/vertical orientation
    - TextInput: Text entry with single/multi-line, validation, and selection
    - Dropdown: Select widget with searchable options list
"""

from .button import (
    Button,
    ButtonState,
    ButtonStyle,
    ClickEvent,
    PressEvent,
    ToggleEvent,
)
from .checkbox import (
    Checkbox,
    CheckboxState,
    CheckboxStyle,
    CheckState,
    CheckStateChangeEvent,
)
from .dropdown import (
    Dropdown,
    DropdownOption,
    DropdownState,
    DropdownStyle,
    SelectionChangeEvent,
)
from .slider import (
    Slider,
    SliderOrientation,
    SliderState,
    SliderStyle,
    ValueChangeEvent,
)
from .text_input import (
    InputMode,
    SelectionRange,
    TextInput,
    TextInputState,
    TextInputStyle,
    TextChangeEvent,
    ValidationResult,
)

__all__ = [
    # Button
    "Button",
    "ButtonState",
    "ButtonStyle",
    "ClickEvent",
    "PressEvent",
    "ToggleEvent",
    # Checkbox
    "Checkbox",
    "CheckboxState",
    "CheckboxStyle",
    "CheckState",
    "CheckStateChangeEvent",
    # Slider
    "Slider",
    "SliderOrientation",
    "SliderState",
    "SliderStyle",
    "ValueChangeEvent",
    # TextInput
    "TextInput",
    "TextInputState",
    "TextInputStyle",
    "TextChangeEvent",
    "InputMode",
    "SelectionRange",
    "ValidationResult",
    # Dropdown
    "Dropdown",
    "DropdownOption",
    "DropdownState",
    "DropdownStyle",
    "SelectionChangeEvent",
]
