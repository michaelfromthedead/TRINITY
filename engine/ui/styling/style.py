"""
Style system for UI widgets.

Provides Style class with visual states, style inheritance, merging,
computed styles, and style selectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from engine.ui.styling.color import Color
from engine.ui.styling.brush import Brush, SolidBrush, create_brush


# ========== Visual States ==========

class VisualState(Enum):
    """Visual states for widgets."""
    NORMAL = auto()
    HOVERED = auto()
    PRESSED = auto()
    FOCUSED = auto()
    DISABLED = auto()
    SELECTED = auto()


# ========== Style Property Descriptor ==========

T = TypeVar("T")


class StylePropertyDescriptor(Generic[T]):
    """
    Validated descriptor for style properties.

    Supports type validation, value constraints, and default values.
    Follows the Trinity Pattern's ValidatedDescriptor concept.
    """

    def __init__(
        self,
        name: str,
        property_type: Type[T],
        default: Optional[T] = None,
        validator: Optional[Callable[[T], bool]] = None,
        converter: Optional[Callable[[Any], T]] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        choices: Optional[Set[T]] = None,
    ) -> None:
        """
        Initialize style property descriptor.

        Args:
            name: Property name
            property_type: Expected type
            default: Default value
            validator: Custom validation function
            converter: Value converter function
            min_value: Minimum value for numeric types
            max_value: Maximum value for numeric types
            choices: Valid choices for the property
        """
        self.name = name
        self.property_type = property_type
        self.default = default
        self.validator = validator
        self.converter = converter
        self.min_value = min_value
        self.max_value = max_value
        self.choices = choices
        self._storage_name = f"_style_{name}"

    def __set_name__(self, owner: Type, name: str) -> None:
        """Called when descriptor is assigned to class."""
        self.name = name
        self._storage_name = f"_style_{name}"

    def __get__(self, obj: Any, objtype: Optional[Type] = None) -> Optional[T]:
        """Get the property value."""
        if obj is None:
            return self  # type: ignore
        return getattr(obj, self._storage_name, self.default)

    def __set__(self, obj: Any, value: Any) -> None:
        """Set the property value with validation."""
        if value is None:
            setattr(obj, self._storage_name, None)
            return

        # Convert value if converter is provided
        if self.converter is not None:
            try:
                value = self.converter(value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to convert '{self.name}': {e}") from e

        # Type check
        if not isinstance(value, self.property_type):
            raise TypeError(
                f"Property '{self.name}' expects {self.property_type.__name__}, "
                f"got {type(value).__name__}"
            )

        # Range check for numeric types
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                raise ValueError(
                    f"Property '{self.name}' must be >= {self.min_value}, got {value}"
                )
            if self.max_value is not None and value > self.max_value:
                raise ValueError(
                    f"Property '{self.name}' must be <= {self.max_value}, got {value}"
                )

        # Choices check
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"Property '{self.name}' must be one of {self.choices}, got {value}"
            )

        # Custom validation
        if self.validator is not None and not self.validator(value):
            raise ValueError(f"Property '{self.name}' failed validation: {value}")

        setattr(obj, self._storage_name, value)


def style_property(
    property_type: Type[T],
    default: Optional[T] = None,
    validator: Optional[Callable[[T], bool]] = None,
    converter: Optional[Callable[[Any], T]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    choices: Optional[Set[T]] = None,
) -> StylePropertyDescriptor[T]:
    """
    Factory function to create style property descriptors.

    Args:
        property_type: Expected type
        default: Default value
        validator: Custom validation function
        converter: Value converter function
        min_value: Minimum value for numeric types
        max_value: Maximum value for numeric types
        choices: Valid choices for the property

    Returns:
        StylePropertyDescriptor instance
    """
    return StylePropertyDescriptor(
        name="",  # Will be set by __set_name__
        property_type=property_type,
        default=default,
        validator=validator,
        converter=converter,
        min_value=min_value,
        max_value=max_value,
        choices=choices,
    )


# ========== Style Class ==========

@dataclass
class Style:
    """
    Style properties for a UI widget.

    Supports visual states, inheritance, and merging.
    """

    # Background
    background: Optional[Brush] = None
    background_color: Optional[Color] = None

    # Border
    border_brush: Optional[Brush] = None
    border_color: Optional[Color] = None
    border_width: float = 0.0
    border_radius: float = 0.0
    border_radius_top_left: Optional[float] = None
    border_radius_top_right: Optional[float] = None
    border_radius_bottom_left: Optional[float] = None
    border_radius_bottom_right: Optional[float] = None

    # Foreground
    foreground_color: Optional[Color] = None

    # Text
    font_family: Optional[str] = None
    font_size: Optional[float] = None
    font_weight: Optional[str] = None
    font_style: Optional[str] = None
    text_align: Optional[str] = None
    line_height: Optional[float] = None
    letter_spacing: Optional[float] = None

    # Opacity and visibility
    opacity: float = 1.0

    # Padding (internal spacing)
    padding_left: float = 0.0
    padding_right: float = 0.0
    padding_top: float = 0.0
    padding_bottom: float = 0.0

    # Margin (external spacing)
    margin_left: float = 0.0
    margin_right: float = 0.0
    margin_top: float = 0.0
    margin_bottom: float = 0.0

    # Shadow
    shadow_color: Optional[Color] = None
    shadow_offset_x: float = 0.0
    shadow_offset_y: float = 0.0
    shadow_blur: float = 0.0
    shadow_spread: float = 0.0

    # Transform
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    translate_x: float = 0.0
    translate_y: float = 0.0

    # Cursor
    cursor: Optional[str] = None

    # Transitions
    transition_duration: float = 0.0
    transition_easing: Optional[str] = None

    # Parent style for inheritance
    _parent: Optional["Style"] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate style properties."""
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"Opacity must be in [0.0, 1.0], got {self.opacity}")
        if self.border_width < 0:
            raise ValueError(f"Border width must be non-negative, got {self.border_width}")
        if self.border_radius < 0:
            raise ValueError(f"Border radius must be non-negative, got {self.border_radius}")

    # ========== Padding/Margin Shortcuts ==========

    @property
    def padding(self) -> Tuple[float, float, float, float]:
        """Get padding as (top, right, bottom, left)."""
        return (self.padding_top, self.padding_right, self.padding_bottom, self.padding_left)

    @padding.setter
    def padding(self, value: Union[float, Tuple[float, ...]]) -> None:
        """Set padding from single value or tuple."""
        if isinstance(value, (int, float)):
            self.padding_top = self.padding_right = self.padding_bottom = self.padding_left = float(value)
        elif len(value) == 1:
            self.padding_top = self.padding_right = self.padding_bottom = self.padding_left = value[0]
        elif len(value) == 2:
            self.padding_top = self.padding_bottom = value[0]
            self.padding_left = self.padding_right = value[1]
        elif len(value) == 4:
            self.padding_top, self.padding_right, self.padding_bottom, self.padding_left = value
        else:
            raise ValueError("Padding must be 1, 2, or 4 values")

    @property
    def margin(self) -> Tuple[float, float, float, float]:
        """Get margin as (top, right, bottom, left)."""
        return (self.margin_top, self.margin_right, self.margin_bottom, self.margin_left)

    @margin.setter
    def margin(self, value: Union[float, Tuple[float, ...]]) -> None:
        """Set margin from single value or tuple."""
        if isinstance(value, (int, float)):
            self.margin_top = self.margin_right = self.margin_bottom = self.margin_left = float(value)
        elif len(value) == 1:
            self.margin_top = self.margin_right = self.margin_bottom = self.margin_left = value[0]
        elif len(value) == 2:
            self.margin_top = self.margin_bottom = value[0]
            self.margin_left = self.margin_right = value[1]
        elif len(value) == 4:
            self.margin_top, self.margin_right, self.margin_bottom, self.margin_left = value
        else:
            raise ValueError("Margin must be 1, 2, or 4 values")

    # ========== Border Radius Shortcuts ==========

    def get_border_radii(self) -> Tuple[float, float, float, float]:
        """Get border radii as (top_left, top_right, bottom_right, bottom_left)."""
        return (
            self.border_radius_top_left if self.border_radius_top_left is not None else self.border_radius,
            self.border_radius_top_right if self.border_radius_top_right is not None else self.border_radius,
            self.border_radius_bottom_right if self.border_radius_bottom_right is not None else self.border_radius,
            self.border_radius_bottom_left if self.border_radius_bottom_left is not None else self.border_radius,
        )

    # ========== Style Inheritance ==========

    def inherit_from(self, parent: "Style") -> "Style":
        """
        Create a new style that inherits from parent.

        Non-None values in self take precedence over parent values.

        Args:
            parent: Parent style to inherit from

        Returns:
            New merged style
        """
        return self.merge(parent)

    def merge(self, other: "Style") -> "Style":
        """
        Merge this style with another.

        Values from self take precedence over other, but only when
        self's value is explicitly set (not the default value).

        Args:
            other: Style to merge with

        Returns:
            New merged style
        """
        merged_values: Dict[str, Any] = {}
        defaults = self._get_field_defaults()

        # Get all style fields (excluding private/internal)
        for field_name in self._get_style_fields():
            self_value = getattr(self, field_name)
            other_value = getattr(other, field_name)
            default_value = defaults.get(field_name)

            # Use self value if explicitly set (different from default)
            # Otherwise use other value
            if self_value != default_value:
                merged_values[field_name] = self_value
            elif other_value is not None:
                merged_values[field_name] = other_value
            else:
                merged_values[field_name] = self_value

        return Style(**merged_values)

    @staticmethod
    def _get_field_defaults() -> Dict[str, Any]:
        """Get default values for style fields."""
        return {
            "background": None, "background_color": None,
            "border_brush": None, "border_color": None,
            "border_width": 0.0, "border_radius": 0.0,
            "border_radius_top_left": None, "border_radius_top_right": None,
            "border_radius_bottom_left": None, "border_radius_bottom_right": None,
            "foreground_color": None,
            "font_family": None, "font_size": None, "font_weight": None, "font_style": None,
            "text_align": None, "line_height": None, "letter_spacing": None,
            "opacity": 1.0,
            "padding_left": 0.0, "padding_right": 0.0, "padding_top": 0.0, "padding_bottom": 0.0,
            "margin_left": 0.0, "margin_right": 0.0, "margin_top": 0.0, "margin_bottom": 0.0,
            "shadow_color": None, "shadow_offset_x": 0.0, "shadow_offset_y": 0.0,
            "shadow_blur": 0.0, "shadow_spread": 0.0,
            "min_width": None, "max_width": None, "min_height": None, "max_height": None,
            "flex_grow": 0.0, "flex_shrink": 1.0, "flex_basis": None,
            "align_self": None, "align_items": None, "justify_content": None,
            "z_index": 0, "visible": True,
            "scale_x": 1.0, "scale_y": 1.0, "rotation": 0.0,
            "translate_x": 0.0, "translate_y": 0.0,
            "cursor": None, "transition_duration": 0.0, "transition_easing": None,
        }

    @staticmethod
    def _get_style_fields() -> List[str]:
        """Get list of style field names."""
        return [
            "background", "background_color",
            "border_brush", "border_color", "border_width", "border_radius",
            "border_radius_top_left", "border_radius_top_right",
            "border_radius_bottom_left", "border_radius_bottom_right",
            "foreground_color",
            "font_family", "font_size", "font_weight", "font_style",
            "text_align", "line_height", "letter_spacing",
            "opacity",
            "padding_left", "padding_right", "padding_top", "padding_bottom",
            "margin_left", "margin_right", "margin_top", "margin_bottom",
            "shadow_color", "shadow_offset_x", "shadow_offset_y",
            "shadow_blur", "shadow_spread",
            "scale_x", "scale_y", "rotation", "translate_x", "translate_y",
            "cursor",
            "transition_duration", "transition_easing",
        ]

    # ========== Style Cloning ==========

    def clone(self) -> "Style":
        """Create a deep copy of this style."""
        return Style(
            background=self.background.clone() if self.background else None,
            background_color=self.background_color,
            border_brush=self.border_brush.clone() if self.border_brush else None,
            border_color=self.border_color,
            border_width=self.border_width,
            border_radius=self.border_radius,
            border_radius_top_left=self.border_radius_top_left,
            border_radius_top_right=self.border_radius_top_right,
            border_radius_bottom_left=self.border_radius_bottom_left,
            border_radius_bottom_right=self.border_radius_bottom_right,
            foreground_color=self.foreground_color,
            font_family=self.font_family,
            font_size=self.font_size,
            font_weight=self.font_weight,
            font_style=self.font_style,
            text_align=self.text_align,
            line_height=self.line_height,
            letter_spacing=self.letter_spacing,
            opacity=self.opacity,
            padding_left=self.padding_left,
            padding_right=self.padding_right,
            padding_top=self.padding_top,
            padding_bottom=self.padding_bottom,
            margin_left=self.margin_left,
            margin_right=self.margin_right,
            margin_top=self.margin_top,
            margin_bottom=self.margin_bottom,
            shadow_color=self.shadow_color,
            shadow_offset_x=self.shadow_offset_x,
            shadow_offset_y=self.shadow_offset_y,
            shadow_blur=self.shadow_blur,
            shadow_spread=self.shadow_spread,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            rotation=self.rotation,
            translate_x=self.translate_x,
            translate_y=self.translate_y,
            cursor=self.cursor,
            transition_duration=self.transition_duration,
            transition_easing=self.transition_easing,
        )


# ========== State Styles ==========

@dataclass
class StateStyles:
    """
    Collection of styles for different visual states.

    Manages styles for normal, hovered, pressed, focused, disabled, and selected states.
    """

    normal: Style = field(default_factory=Style)
    hovered: Optional[Style] = None
    pressed: Optional[Style] = None
    focused: Optional[Style] = None
    disabled: Optional[Style] = None
    selected: Optional[Style] = None

    def get_style(self, state: VisualState) -> Style:
        """
        Get the computed style for a given state.

        Falls back to normal style if state-specific style is not defined.

        Args:
            state: Visual state

        Returns:
            Computed style for the state
        """
        state_style = getattr(self, state.name.lower(), None)
        if state_style is not None:
            return state_style.merge(self.normal)
        return self.normal

    def set_style(self, state: VisualState, style: Style) -> None:
        """
        Set the style for a given state.

        Args:
            state: Visual state
            style: Style to set
        """
        setattr(self, state.name.lower(), style)

    def get_computed_style(self, states: Set[VisualState]) -> Style:
        """
        Get the computed style for multiple active states.

        State precedence: disabled > pressed > focused > selected > hovered > normal

        Args:
            states: Set of active visual states

        Returns:
            Computed merged style
        """
        result = self.normal.clone()

        # Apply states in precedence order
        precedence = [
            VisualState.HOVERED,
            VisualState.SELECTED,
            VisualState.FOCUSED,
            VisualState.PRESSED,
            VisualState.DISABLED,
        ]

        for state in precedence:
            if state in states:
                state_style = getattr(self, state.name.lower(), None)
                if state_style is not None:
                    result = state_style.merge(result)

        return result

    def clone(self) -> "StateStyles":
        """Create a deep copy of state styles."""
        return StateStyles(
            normal=self.normal.clone(),
            hovered=self.hovered.clone() if self.hovered else None,
            pressed=self.pressed.clone() if self.pressed else None,
            focused=self.focused.clone() if self.focused else None,
            disabled=self.disabled.clone() if self.disabled else None,
            selected=self.selected.clone() if self.selected else None,
        )


# ========== Style Selectors ==========

class SelectorType(Enum):
    """Types of style selectors."""
    TYPE = auto()      # Match by widget type
    NAME = auto()      # Match by widget name
    CLASS = auto()     # Match by style class
    STATE = auto()     # Match by visual state
    ID = auto()        # Match by widget ID
    UNIVERSAL = auto() # Match all widgets


@dataclass(frozen=True)
class StyleSelector:
    """
    A selector for matching widgets to styles.

    Selectors can match by type, name, class, state, or ID.
    """

    selector_type: SelectorType
    value: Union[str, Type, VisualState, None] = None
    pseudo_states: Tuple[VisualState, ...] = ()

    @classmethod
    def by_type(cls, widget_type: Type) -> "StyleSelector":
        """Create a selector that matches by widget type."""
        return cls(SelectorType.TYPE, widget_type)

    @classmethod
    def by_name(cls, name: str) -> "StyleSelector":
        """Create a selector that matches by widget name."""
        return cls(SelectorType.NAME, name)

    @classmethod
    def by_class(cls, class_name: str) -> "StyleSelector":
        """Create a selector that matches by style class."""
        return cls(SelectorType.CLASS, class_name)

    @classmethod
    def by_state(cls, state: VisualState) -> "StyleSelector":
        """Create a selector that matches by visual state."""
        return cls(SelectorType.STATE, state)

    @classmethod
    def by_id(cls, widget_id: str) -> "StyleSelector":
        """Create a selector that matches by widget ID."""
        return cls(SelectorType.ID, widget_id)

    @classmethod
    def universal(cls) -> "StyleSelector":
        """Create a selector that matches all widgets."""
        return cls(SelectorType.UNIVERSAL)

    def with_state(self, *states: VisualState) -> "StyleSelector":
        """Add pseudo-state conditions to the selector."""
        return StyleSelector(
            selector_type=self.selector_type,
            value=self.value,
            pseudo_states=states,
        )

    def matches(
        self,
        widget_type: Optional[Type] = None,
        widget_name: Optional[str] = None,
        widget_id: Optional[str] = None,
        style_classes: Optional[Set[str]] = None,
        active_states: Optional[Set[VisualState]] = None,
    ) -> bool:
        """
        Check if this selector matches the given widget attributes.

        Args:
            widget_type: Widget type/class
            widget_name: Widget name
            widget_id: Widget ID
            style_classes: Widget style classes
            active_states: Widget active visual states

        Returns:
            True if selector matches
        """
        # Check pseudo-states first
        if self.pseudo_states:
            if active_states is None or not all(s in active_states for s in self.pseudo_states):
                return False

        # Check main selector
        if self.selector_type == SelectorType.UNIVERSAL:
            return True
        elif self.selector_type == SelectorType.TYPE:
            return widget_type is not None and (
                widget_type is self.value or
                (isinstance(self.value, type) and issubclass(widget_type, self.value))
            )
        elif self.selector_type == SelectorType.NAME:
            return widget_name == self.value
        elif self.selector_type == SelectorType.CLASS:
            return style_classes is not None and self.value in style_classes
        elif self.selector_type == SelectorType.STATE:
            return active_states is not None and self.value in active_states
        elif self.selector_type == SelectorType.ID:
            return widget_id == self.value

        return False

    @property
    def specificity(self) -> Tuple[int, int, int]:
        """
        Calculate CSS-like specificity.

        Returns:
            Tuple of (id_count, class_count, type_count)
        """
        id_count = 1 if self.selector_type == SelectorType.ID else 0
        class_count = (
            1 if self.selector_type in (SelectorType.CLASS, SelectorType.NAME) else 0
        ) + len(self.pseudo_states)
        type_count = 1 if self.selector_type == SelectorType.TYPE else 0

        return (id_count, class_count, type_count)


# ========== Style Rule ==========

@dataclass
class StyleRule:
    """
    A style rule with selector and associated style.

    Used in stylesheets to apply styles to matching widgets.
    """

    selector: StyleSelector
    style: Style
    priority: int = 0

    @property
    def specificity(self) -> Tuple[int, int, int, int]:
        """
        Calculate total specificity including priority.

        Returns:
            Tuple of (priority, id_count, class_count, type_count)
        """
        s = self.selector.specificity
        return (self.priority, s[0], s[1], s[2])


# ========== Stylesheet ==========

class Stylesheet:
    """
    Collection of style rules.

    Provides style lookup and cascade resolution.
    """

    def __init__(self) -> None:
        """Initialize empty stylesheet."""
        self._rules: List[StyleRule] = []

    def add_rule(self, rule: StyleRule) -> None:
        """Add a style rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.specificity)

    def add_style(
        self,
        selector: StyleSelector,
        style: Style,
        priority: int = 0,
    ) -> None:
        """
        Add a style with selector.

        Args:
            selector: Style selector
            style: Style to apply
            priority: Rule priority (higher = more important)
        """
        self.add_rule(StyleRule(selector, style, priority))

    def remove_rules_for_selector(self, selector: StyleSelector) -> int:
        """
        Remove all rules matching the given selector.

        Args:
            selector: Selector to match

        Returns:
            Number of rules removed
        """
        initial_count = len(self._rules)
        self._rules = [r for r in self._rules if r.selector != selector]
        return initial_count - len(self._rules)

    def get_computed_style(
        self,
        widget_type: Optional[Type] = None,
        widget_name: Optional[str] = None,
        widget_id: Optional[str] = None,
        style_classes: Optional[Set[str]] = None,
        active_states: Optional[Set[VisualState]] = None,
        base_style: Optional[Style] = None,
    ) -> Style:
        """
        Compute the final style for a widget.

        Args:
            widget_type: Widget type/class
            widget_name: Widget name
            widget_id: Widget ID
            style_classes: Widget style classes
            active_states: Widget active visual states
            base_style: Base style to start with

        Returns:
            Computed merged style
        """
        result = base_style.clone() if base_style else Style()

        # Apply matching rules in specificity order
        for rule in self._rules:
            if rule.selector.matches(
                widget_type=widget_type,
                widget_name=widget_name,
                widget_id=widget_id,
                style_classes=style_classes,
                active_states=active_states,
            ):
                result = rule.style.merge(result)

        return result

    def clear(self) -> None:
        """Remove all rules."""
        self._rules.clear()

    def __len__(self) -> int:
        """Return number of rules."""
        return len(self._rules)

    def __iter__(self):
        """Iterate over rules."""
        return iter(self._rules)


# ========== Style Builder ==========

class StyleBuilder:
    """
    Fluent builder for creating Style instances.

    Provides a chainable API for style construction.
    """

    def __init__(self, base_style: Optional[Style] = None) -> None:
        """Initialize builder with optional base style."""
        self._style = base_style.clone() if base_style else Style()

    def background(self, value: Union[Brush, Color, str]) -> "StyleBuilder":
        """Set background brush or color."""
        if isinstance(value, Brush):
            self._style.background = value
        else:
            self._style.background = create_brush(value)
        return self

    def background_color(self, value: Union[Color, str]) -> "StyleBuilder":
        """Set background color."""
        self._style.background_color = Color.parse(value) if isinstance(value, str) else value
        return self

    def border(
        self,
        width: float = 1.0,
        color: Optional[Union[Color, str]] = None,
        radius: float = 0.0,
    ) -> "StyleBuilder":
        """Set border properties."""
        self._style.border_width = width
        if color is not None:
            self._style.border_color = Color.parse(color) if isinstance(color, str) else color
        self._style.border_radius = radius
        return self

    def border_radius(
        self,
        value: Union[float, Tuple[float, float, float, float]],
    ) -> "StyleBuilder":
        """Set border radius (single value or corners tuple)."""
        if isinstance(value, (int, float)):
            self._style.border_radius = float(value)
        else:
            self._style.border_radius_top_left = value[0]
            self._style.border_radius_top_right = value[1]
            self._style.border_radius_bottom_right = value[2]
            self._style.border_radius_bottom_left = value[3]
        return self

    def foreground(self, color: Union[Color, str]) -> "StyleBuilder":
        """Set foreground color."""
        self._style.foreground_color = Color.parse(color) if isinstance(color, str) else color
        return self

    def font(
        self,
        family: Optional[str] = None,
        size: Optional[float] = None,
        weight: Optional[str] = None,
        style: Optional[str] = None,
    ) -> "StyleBuilder":
        """Set font properties."""
        if family is not None:
            self._style.font_family = family
        if size is not None:
            self._style.font_size = size
        if weight is not None:
            self._style.font_weight = weight
        if style is not None:
            self._style.font_style = style
        return self

    def text(
        self,
        align: Optional[str] = None,
        line_height: Optional[float] = None,
        letter_spacing: Optional[float] = None,
    ) -> "StyleBuilder":
        """Set text properties."""
        if align is not None:
            self._style.text_align = align
        if line_height is not None:
            self._style.line_height = line_height
        if letter_spacing is not None:
            self._style.letter_spacing = letter_spacing
        return self

    def opacity(self, value: float) -> "StyleBuilder":
        """Set opacity."""
        self._style.opacity = value
        return self

    def padding(self, *args: float) -> "StyleBuilder":
        """Set padding (1, 2, or 4 values)."""
        if len(args) == 1:
            self._style.padding = args[0]
        elif len(args) == 2:
            self._style.padding = args
        elif len(args) == 4:
            self._style.padding = args
        else:
            raise ValueError("padding accepts 1, 2, or 4 arguments")
        return self

    def margin(self, *args: float) -> "StyleBuilder":
        """Set margin (1, 2, or 4 values)."""
        if len(args) == 1:
            self._style.margin = args[0]
        elif len(args) == 2:
            self._style.margin = args
        elif len(args) == 4:
            self._style.margin = args
        else:
            raise ValueError("margin accepts 1, 2, or 4 arguments")
        return self

    # Default shadow values matching theme.shadows.md (medium shadow)
    _DEFAULT_SHADOW_OFFSET_Y: float = 2.0
    _DEFAULT_SHADOW_BLUR: float = 4.0

    def shadow(
        self,
        color: Union[Color, str],
        offset_x: float = 0.0,
        offset_y: float = _DEFAULT_SHADOW_OFFSET_Y,
        blur: float = _DEFAULT_SHADOW_BLUR,
        spread: float = 0.0,
    ) -> "StyleBuilder":
        """Set shadow properties. Defaults match theme.shadows.md."""
        self._style.shadow_color = Color.parse(color) if isinstance(color, str) else color
        self._style.shadow_offset_x = offset_x
        self._style.shadow_offset_y = offset_y
        self._style.shadow_blur = blur
        self._style.shadow_spread = spread
        return self

    def transform(
        self,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
        rotation: Optional[float] = None,
        translate_x: Optional[float] = None,
        translate_y: Optional[float] = None,
    ) -> "StyleBuilder":
        """Set transform properties."""
        if scale_x is not None:
            self._style.scale_x = scale_x
        if scale_y is not None:
            self._style.scale_y = scale_y
        if rotation is not None:
            self._style.rotation = rotation
        if translate_x is not None:
            self._style.translate_x = translate_x
        if translate_y is not None:
            self._style.translate_y = translate_y
        return self

    def cursor(self, cursor: str) -> "StyleBuilder":
        """Set cursor type."""
        self._style.cursor = cursor
        return self

    # Default transition values matching theme.transitions
    _DEFAULT_TRANSITION_DURATION: float = 0.2
    _DEFAULT_TRANSITION_EASING: str = "ease-out"

    def transition(
        self,
        duration: float = _DEFAULT_TRANSITION_DURATION,
        easing: str = _DEFAULT_TRANSITION_EASING,
    ) -> "StyleBuilder":
        """Set transition properties. Defaults match theme.transitions.duration_normal and ease_out."""
        self._style.transition_duration = duration
        self._style.transition_easing = easing
        return self

    def build(self) -> Style:
        """Build and return the Style instance."""
        return self._style.clone()
