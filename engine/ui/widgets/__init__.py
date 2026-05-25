"""
UI Widgets Module.

This module provides all UI widgets for the game engine, organized into categories:

- input: Interactive input widgets (Button, Checkbox, Slider, TextInput, Dropdown)
- display: Display-only widgets (Label, ProgressBar, Icon, Badge)
- game: Game-specific widgets (HealthBar, Minimap, InventorySlot, DamageNumbers, Tooltip)
- primitives: Basic building blocks (Text, Image, Border, Spacer)

All widgets follow consistent patterns for:
- State management via properties
- Event handling with callbacks
- Dirty tracking for efficient rendering
- Styling via dataclass configurations
"""

from __future__ import annotations

# Input widgets
from engine.ui.widgets.input import (
    Button,
    ButtonState,
    ButtonStyle,
    ClickEvent,
    PressEvent,
    ToggleEvent,
    Checkbox,
    CheckboxState,
    CheckboxStyle,
    CheckState,
    CheckStateChangeEvent,
    Slider,
    SliderOrientation,
    SliderState,
    SliderStyle,
    ValueChangeEvent,
    TextInput,
    TextInputState,
    TextInputStyle,
    TextChangeEvent,
    InputMode,
    SelectionRange,
    ValidationResult,
    Dropdown,
    DropdownOption,
    DropdownState,
    DropdownStyle,
    SelectionChangeEvent,
)

# Display widgets
from engine.ui.widgets.display import (
    Label,
    ProgressBar,
    ProgressBarMode,
    ProgressBarStyle,
    Icon,
    IconSize,
    IconAnimation,
    IconFlip,
    IconAtlasEntry,
    IconAtlasManager,
    Badge,
    BadgeMode,
    BadgePosition,
    BadgeVariant,
    BadgeStyle,
)

# Game widgets
from engine.ui.widgets.game import (
    HealthBar,
    HealthBarStyle,
    HealthBarSegment,
    ResourceType,
    Minimap,
    MinimapMarker,
    MarkerType,
    MinimapConfig,
    InventorySlot,
    ItemData,
    ItemRarity,
    DragPayload,
    DropResult,
    SlotState,
    DamageNumber,
    DamageNumberManager,
    DamageType,
    DamageNumberConfig,
    Tooltip,
    TooltipManager,
    TooltipContent,
    TooltipPosition,
    TooltipAnimation,
    TooltipStyle,
    RichTooltip,
)

# Primitive widgets
from engine.ui.widgets.primitives import (
    Image,
    ScaleMode,
    NineSliceConfig,
    UVCoordinates,
    Text,
    TextAlignment,
    VerticalAlignment,
    OverflowMode,
    TextStyle,
    RichTextParser,
    RichTextSpan,
    Border,
    BorderStyle,
    CornerRadius,
    Spacer,
    SpacerMode,
)

# Constants
from engine.ui.widgets.constants import (
    Colors,
    Dimensions,
    Typography,
    Animation,
    Thresholds,
    Limits,
)


__all__ = [
    # Input widgets
    "Button",
    "ButtonState",
    "ButtonStyle",
    "ClickEvent",
    "PressEvent",
    "ToggleEvent",
    "Checkbox",
    "CheckboxState",
    "CheckboxStyle",
    "CheckState",
    "CheckStateChangeEvent",
    "Slider",
    "SliderOrientation",
    "SliderState",
    "SliderStyle",
    "ValueChangeEvent",
    "TextInput",
    "TextInputState",
    "TextInputStyle",
    "TextChangeEvent",
    "InputMode",
    "SelectionRange",
    "ValidationResult",
    "Dropdown",
    "DropdownOption",
    "DropdownState",
    "DropdownStyle",
    "SelectionChangeEvent",
    # Display widgets
    "Label",
    "ProgressBar",
    "ProgressBarMode",
    "ProgressBarStyle",
    "Icon",
    "IconSize",
    "IconAnimation",
    "IconFlip",
    "IconAtlasEntry",
    "IconAtlasManager",
    "Badge",
    "BadgeMode",
    "BadgePosition",
    "BadgeVariant",
    "BadgeStyle",
    # Game widgets
    "HealthBar",
    "HealthBarStyle",
    "HealthBarSegment",
    "ResourceType",
    "Minimap",
    "MinimapMarker",
    "MarkerType",
    "MinimapConfig",
    "InventorySlot",
    "ItemData",
    "ItemRarity",
    "DragPayload",
    "DropResult",
    "SlotState",
    "DamageNumber",
    "DamageNumberManager",
    "DamageType",
    "DamageNumberConfig",
    "Tooltip",
    "TooltipManager",
    "TooltipContent",
    "TooltipPosition",
    "TooltipAnimation",
    "TooltipStyle",
    "RichTooltip",
    # Primitive widgets
    "Image",
    "ScaleMode",
    "NineSliceConfig",
    "UVCoordinates",
    "Text",
    "TextAlignment",
    "VerticalAlignment",
    "OverflowMode",
    "TextStyle",
    "RichTextParser",
    "RichTextSpan",
    "Border",
    "BorderStyle",
    "CornerRadius",
    "Spacer",
    "SpacerMode",
    # Constants
    "Colors",
    "Dimensions",
    "Typography",
    "Animation",
    "Thresholds",
    "Limits",
]
