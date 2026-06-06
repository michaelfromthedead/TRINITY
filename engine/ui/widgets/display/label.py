"""
Label Widget Implementation.

A lightweight single-line text display widget with support for:
- Single-line text content
- Auto-sizing based on content
- Optional icon support (leading or trailing)
- Text styling (font, color, size, alignment)
- Text overflow handling (clip, ellipsis, fade)
- Accessibility support

Follows the Standalone pattern with manual property tracking
and dirty flag for efficient rendering updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional


class IconPosition(Enum):
    """Position of icon relative to text."""
    LEADING = auto()   # Before text (left in LTR, right in RTL)
    TRAILING = auto()  # After text (right in LTR, left in RTL)


class TextAlign(Enum):
    """Text alignment within the label."""
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()


class TextOverflow(Enum):
    """How to handle text that exceeds label bounds."""
    CLIP = auto()      # Hard clip at boundary
    ELLIPSIS = auto()  # Show "..." at truncation point
    FADE = auto()      # Fade out at edge


@dataclass(slots=True)
class LabelStyle:
    """Style configuration for label appearance.

    Attributes:
        font_family: Font family name
        font_size: Font size in points
        font_weight: Font weight (normal, bold, light)
        text_color: Text foreground color
        disabled_text_color: Text color when disabled
        icon_size: Icon dimensions if present
        icon_spacing: Gap between icon and text
        icon_color: Icon color (None = inherit text_color)
    """
    font_family: str = "default"
    font_size: float = 14.0
    font_weight: str = "normal"
    text_color: str = "#000000"
    disabled_text_color: str = "#888888"
    icon_size: float = 16.0
    icon_spacing: float = 4.0
    icon_color: Optional[str] = None


class Label:
    """Single-line text display widget.

    A lightweight widget for displaying static or dynamic text.
    Optimized for single-line labels, buttons, and UI elements.

    Attributes:
        text: The text content to display
        icon: Optional icon name or sprite reference
        icon_position: Position of icon relative to text
        text_align: Text alignment (left, center, right)
        text_overflow: How to handle overflow (clip, ellipsis, fade)
        auto_size: If True, automatically size to content
        min_width: Minimum width constraint
        max_width: Maximum width constraint (0 = no limit)
        enabled: Whether the label is enabled
        visible: Whether the label is rendered
        opacity: Opacity from 0.0 (transparent) to 1.0 (opaque)

    Accessibility:
        role: "text" - Indicates this is a text display element
        accessible_text: Returns the text content for screen readers
    """

    __slots__ = (
        '_id', '_text', '_icon', '_icon_position', '_enabled', '_visible',
        '_text_align', '_text_overflow', '_style',
        '_auto_size', '_min_width', '_max_width',
        '_x', '_y', '_width', '_height', '_opacity',
        '_computed_width', '_computed_height',
        '_dirty', '_dirty_layout', '_dirty_fields'
    )

    # Class-level ID counter for unique widget IDs
    _next_id: int = 0

    def __init__(
        self,
        text: str = "",
        icon: Optional[str] = None,
        icon_position: IconPosition = IconPosition.LEADING,
        text_align: TextAlign = TextAlign.LEFT,
        text_overflow: TextOverflow = TextOverflow.ELLIPSIS,
        auto_size: bool = True,
        min_width: float = 0.0,
        max_width: float = 0.0,
        enabled: bool = True,
        visible: bool = True,
        opacity: float = 1.0,
        style: Optional[LabelStyle] = None,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 100.0,
        height: float = 20.0,
    ):
        """Initialize a label widget.

        Args:
            text: Text content to display
            icon: Optional icon identifier
            icon_position: Where to place icon relative to text
            text_align: Text alignment within bounds
            text_overflow: How to handle text overflow
            auto_size: Automatically size to content
            min_width: Minimum width constraint
            max_width: Maximum width constraint (0 = no limit)
            enabled: Initial enabled state
            visible: Initial visibility
            opacity: Initial opacity (0.0-1.0)
            style: Style configuration
            x: X position
            y: Y position
            width: Label width (if not auto-sizing)
            height: Label height
        """
        self._id = Label._next_id
        Label._next_id += 1

        self._text = text
        self._icon = icon
        self._icon_position = icon_position
        self._text_align = text_align
        self._text_overflow = text_overflow
        self._style = style or LabelStyle()

        self._auto_size = auto_size
        self._min_width = max(0.0, min_width)
        self._max_width = max(0.0, max_width)

        self._enabled = enabled
        self._visible = visible
        self._opacity = max(0.0, min(1.0, opacity))

        self._x = x
        self._y = y
        self._width = max(0.0, width)
        self._height = max(0.0, height)

        self._computed_width: float = 0.0
        self._computed_height: float = 0.0

        self._dirty = True
        self._dirty_layout = True
        self._dirty_fields: set[str] = set()

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the ID counter. Used for testing."""
        cls._next_id = 0

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def id(self) -> int:
        """Get the unique widget ID."""
        return self._id

    @property
    def text(self) -> str:
        """Get the text content."""
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """Set the text content."""
        if self._text != value:
            self._text = value
            self._mark_dirty("text")
            self._invalidate_layout()

    @property
    def icon(self) -> Optional[str]:
        """Get the icon reference."""
        return self._icon

    @icon.setter
    def icon(self, value: Optional[str]) -> None:
        """Set the icon reference."""
        if self._icon != value:
            self._icon = value
            self._mark_dirty("icon")
            self._invalidate_layout()

    @property
    def icon_position(self) -> IconPosition:
        """Get the icon position."""
        return self._icon_position

    @icon_position.setter
    def icon_position(self, value: IconPosition) -> None:
        """Set the icon position."""
        if self._icon_position != value:
            self._icon_position = value
            self._mark_dirty("icon_position")
            self._invalidate_layout()

    @property
    def text_align(self) -> TextAlign:
        """Get the text alignment."""
        return self._text_align

    @text_align.setter
    def text_align(self, value: TextAlign) -> None:
        """Set the text alignment."""
        if self._text_align != value:
            self._text_align = value
            self._mark_dirty("text_align")

    @property
    def text_overflow(self) -> TextOverflow:
        """Get the text overflow mode."""
        return self._text_overflow

    @text_overflow.setter
    def text_overflow(self, value: TextOverflow) -> None:
        """Set the text overflow mode."""
        if self._text_overflow != value:
            self._text_overflow = value
            self._mark_dirty("text_overflow")

    @property
    def style(self) -> LabelStyle:
        """Get label style."""
        return self._style

    @style.setter
    def style(self, value: LabelStyle) -> None:
        """Set label style."""
        self._style = value
        self._dirty = True
        self._invalidate_layout()

    @property
    def auto_size(self) -> bool:
        """Get the auto-size setting."""
        return self._auto_size

    @auto_size.setter
    def auto_size(self, value: bool) -> None:
        """Set the auto-size setting."""
        if self._auto_size != value:
            self._auto_size = value
            self._mark_dirty("auto_size")
            self._invalidate_layout()

    @property
    def min_width(self) -> float:
        """Get the minimum width."""
        return self._min_width

    @min_width.setter
    def min_width(self, value: float) -> None:
        """Set the minimum width."""
        value = max(0.0, value)
        if self._min_width != value:
            self._min_width = value
            self._mark_dirty("min_width")
            self._invalidate_layout()

    @property
    def max_width(self) -> float:
        """Get the maximum width."""
        return self._max_width

    @max_width.setter
    def max_width(self, value: float) -> None:
        """Set the maximum width."""
        value = max(0.0, value)
        if self._max_width != value:
            self._max_width = value
            self._mark_dirty("max_width")
            self._invalidate_layout()

    @property
    def enabled(self) -> bool:
        """Check if label is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        if self._enabled != value:
            self._enabled = value
            self._mark_dirty("enabled")

    @property
    def visible(self) -> bool:
        """Check if label is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set visibility."""
        if self._visible != value:
            self._visible = value
            self._mark_dirty("visible")

    @property
    def opacity(self) -> float:
        """Get opacity."""
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set opacity."""
        value = max(0.0, min(1.0, value))
        if self._opacity != value:
            self._opacity = value
            self._mark_dirty("opacity")

    @property
    def font_family(self) -> str:
        """Get the font family from style."""
        return self._style.font_family

    @font_family.setter
    def font_family(self, value: str) -> None:
        """Set the font family."""
        if self._style.font_family != value:
            self._style = LabelStyle(
                font_family=value,
                font_size=self._style.font_size,
                font_weight=self._style.font_weight,
                text_color=self._style.text_color,
                disabled_text_color=self._style.disabled_text_color,
                icon_size=self._style.icon_size,
                icon_spacing=self._style.icon_spacing,
                icon_color=self._style.icon_color,
            )
            self._mark_dirty("font_family")
            self._invalidate_layout()

    @property
    def font_size(self) -> float:
        """Get the font size from style."""
        return self._style.font_size

    @font_size.setter
    def font_size(self, value: float) -> None:
        """Set the font size."""
        value = max(1.0, min(200.0, value))
        if self._style.font_size != value:
            self._style = LabelStyle(
                font_family=self._style.font_family,
                font_size=value,
                font_weight=self._style.font_weight,
                text_color=self._style.text_color,
                disabled_text_color=self._style.disabled_text_color,
                icon_size=self._style.icon_size,
                icon_spacing=self._style.icon_spacing,
                icon_color=self._style.icon_color,
            )
            self._mark_dirty("font_size")
            self._invalidate_layout()

    @property
    def font_weight(self) -> str:
        """Get the font weight from style."""
        return self._style.font_weight

    @font_weight.setter
    def font_weight(self, value: str) -> None:
        """Set the font weight."""
        valid_weights = ("normal", "bold", "light")
        if value not in valid_weights:
            raise ValueError(f"Invalid font weight '{value}'. Must be one of: {valid_weights}")
        if self._style.font_weight != value:
            self._style = LabelStyle(
                font_family=self._style.font_family,
                font_size=self._style.font_size,
                font_weight=value,
                text_color=self._style.text_color,
                disabled_text_color=self._style.disabled_text_color,
                icon_size=self._style.icon_size,
                icon_spacing=self._style.icon_spacing,
                icon_color=self._style.icon_color,
            )
            self._mark_dirty("font_weight")

    @property
    def text_color(self) -> str:
        """Get the text color from style."""
        return self._style.text_color

    @text_color.setter
    def text_color(self, value: str) -> None:
        """Set the text color."""
        if self._style.text_color != value:
            self._style = LabelStyle(
                font_family=self._style.font_family,
                font_size=self._style.font_size,
                font_weight=self._style.font_weight,
                text_color=value,
                disabled_text_color=self._style.disabled_text_color,
                icon_size=self._style.icon_size,
                icon_spacing=self._style.icon_spacing,
                icon_color=self._style.icon_color,
            )
            self._mark_dirty("text_color")

    @property
    def icon_color(self) -> Optional[str]:
        """Get the icon color from style."""
        return self._style.icon_color

    @icon_color.setter
    def icon_color(self, value: Optional[str]) -> None:
        """Set the icon color."""
        if self._style.icon_color != value:
            self._style = LabelStyle(
                font_family=self._style.font_family,
                font_size=self._style.font_size,
                font_weight=self._style.font_weight,
                text_color=self._style.text_color,
                disabled_text_color=self._style.disabled_text_color,
                icon_size=self._style.icon_size,
                icon_spacing=self._style.icon_spacing,
                icon_color=value,
            )
            self._mark_dirty("icon_color")

    @property
    def icon_spacing(self) -> float:
        """Get the icon spacing from style."""
        return self._style.icon_spacing

    @icon_spacing.setter
    def icon_spacing(self, value: float) -> None:
        """Set the icon spacing."""
        value = max(0.0, value)
        if self._style.icon_spacing != value:
            self._style = LabelStyle(
                font_family=self._style.font_family,
                font_size=self._style.font_size,
                font_weight=self._style.font_weight,
                text_color=self._style.text_color,
                disabled_text_color=self._style.disabled_text_color,
                icon_size=self._style.icon_size,
                icon_spacing=value,
                icon_color=self._style.icon_color,
            )
            self._mark_dirty("icon_spacing")
            self._invalidate_layout()

    @property
    def effective_icon_color(self) -> str:
        """Get the effective icon color (icon_color if set, else text_color)."""
        return self._style.icon_color if self._style.icon_color else self._style.text_color

    @property
    def x(self) -> float:
        """Get X position."""
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        """Set X position."""
        if self._x != value:
            self._x = value
            self._mark_dirty("x")

    @property
    def y(self) -> float:
        """Get Y position."""
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        """Set Y position."""
        if self._y != value:
            self._y = value
            self._mark_dirty("y")

    @property
    def width(self) -> float:
        """Get the width (computed if auto_size is True)."""
        if self._auto_size:
            return self._computed_width
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set the width."""
        value = max(0.0, value)
        if self._width != value:
            self._width = value
            self._mark_dirty("width")

    @property
    def height(self) -> float:
        """Get the height (computed if auto_size is True)."""
        if self._auto_size:
            return self._computed_height
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set the height."""
        value = max(0.0, value)
        if self._height != value:
            self._height = value
            self._mark_dirty("height")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get label bounds (x, y, width, height)."""
        return (self._x, self._y, self.width, self.height)

    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================

    @property
    def effective_icon_color(self) -> str:
        """Get the effective icon color (inherits from text_color if not set)."""
        if self._style.icon_color is not None:
            return self._style.icon_color
        return self._style.text_color

    @property
    def has_icon(self) -> bool:
        """Check if label has an icon."""
        return self._icon is not None

    # =========================================================================
    # DIRTY TRACKING
    # =========================================================================

    def _mark_dirty(self, field_name: str) -> None:
        """Mark a field as dirty for change tracking."""
        self._dirty = True
        self._dirty_fields.add(field_name)

    def _invalidate_layout(self) -> None:
        """Mark layout as needing recalculation."""
        self._dirty_layout = True

    def is_dirty(self, field_name: Optional[str] = None) -> bool:
        """Check if label needs re-rendering, or if a specific field is dirty.

        Args:
            field_name: Optional field name to check specifically

        Returns:
            True if dirty (or if field is in dirty fields when specified)
        """
        if field_name is None:
            return self._dirty
        return field_name in self._dirty_fields

    def is_field_dirty(self, field_name: Optional[str] = None) -> bool:
        """Check if a field or any field is dirty.

        Args:
            field_name: Specific field to check, or None for any field.

        Returns:
            True if the specified field (or any field) is dirty.
        """
        if field_name is None:
            return len(self._dirty_fields) > 0
        return field_name in self._dirty_fields

    def mark_clean(self) -> None:
        """Mark the label as rendered."""
        self._dirty = False

    def clear_dirty(self) -> None:
        """Clear all dirty flags."""
        self._dirty_fields.clear()
        self._dirty = False

    # =========================================================================
    # LAYOUT
    # =========================================================================

    def measure(self, available_width: float = float("inf")) -> tuple[float, float]:
        """Measure the label and compute its desired size.

        Args:
            available_width: Maximum available width.

        Returns:
            Tuple of (width, height).
        """
        # Estimate text size (in real implementation, use font metrics)
        char_width = self._style.font_size * 0.6  # Approximate character width
        text_width = len(self._text) * char_width
        text_height = self._style.font_size * 1.2  # Line height

        # Add icon size if present
        icon_width = 0.0
        if self._icon:
            icon_width = self._style.icon_size + self._style.icon_spacing

        total_width = text_width + icon_width
        total_height = max(text_height, self._style.icon_size if self._icon else 0.0)

        # Apply constraints
        if self._min_width > 0:
            total_width = max(total_width, self._min_width)
        if self._max_width > 0:
            total_width = min(total_width, self._max_width)
        total_width = min(total_width, available_width)

        self._computed_width = total_width
        self._computed_height = total_height
        self._dirty_layout = False

        return (total_width, total_height)

    def get_text_bounds(self) -> tuple[float, float, float, float]:
        """Get the bounds of the text area.

        Returns:
            Tuple of (x, y, width, height) for the text area.
        """
        x = self._x
        w = self.width

        if self._icon:
            icon_size = self._style.icon_size
            spacing = self._style.icon_spacing
            if self._icon_position == IconPosition.LEADING:
                x += icon_size + spacing
                w -= icon_size + spacing
            else:
                w -= icon_size + spacing

        return (x, self._y, w, self.height)

    def get_icon_bounds(self) -> Optional[tuple[float, float, float, float]]:
        """Get the bounds of the icon area.

        Returns:
            Tuple of (x, y, width, height) for the icon area, or None if no icon.
        """
        if not self._icon:
            return None

        icon_size = self._style.icon_size
        y = self._y + (self.height - icon_size) / 2  # Vertically center

        if self._icon_position == IconPosition.LEADING:
            x = self._x
        else:
            x = self._x + self.width - icon_size

        return (x, y, icon_size, icon_size)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within the label bounds.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside label bounds
        """
        return (
            self._x <= x <= self._x + self.width and
            self._y <= y <= self._y + self.height
        )

    # =========================================================================
    # STYLE HELPERS
    # =========================================================================

    def get_current_text_color(self) -> str:
        """Get the text color for the current state.

        Returns:
            Text color string for current state
        """
        if not self._enabled:
            return self._style.disabled_text_color
        return self._style.text_color

    # =========================================================================
    # ACCESSIBILITY
    # =========================================================================

    def get_accessible_text(self) -> str:
        """Get text for screen readers.

        Returns:
            The label text content.
        """
        return self._text

    def get_accessible_role(self) -> str:
        """Get the accessibility role.

        Returns:
            The ARIA-style role identifier.
        """
        return "text"

    def get_accessible_properties(self) -> dict[str, Any]:
        """Get accessibility properties for assistive technologies.

        Returns:
            Dictionary of accessibility properties.
        """
        return {
            "role": "text",
            "aria-label": self._text,
            "aria-disabled": not self._enabled,
            "aria-hidden": not self._visible,
        }

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize label to dictionary.

        Returns:
            Dictionary representation of the label.
        """
        return {
            "text": self._text,
            "icon": self._icon,
            "icon_position": self._icon_position.name,
            "text_align": self._text_align.name,
            "text_overflow": self._text_overflow.name,
            "font_family": self._style.font_family,
            "font_size": self._style.font_size,
            "font_weight": self._style.font_weight,
            "text_color": self._style.text_color,
            "disabled_text_color": self._style.disabled_text_color,
            "icon_size": self._style.icon_size,
            "icon_spacing": self._style.icon_spacing,
            "icon_color": self._style.icon_color,
            "auto_size": self._auto_size,
            "min_width": self._min_width,
            "max_width": self._max_width,
            "x": self._x,
            "y": self._y,
            "width": self._width,
            "height": self._height,
            "visible": self._visible,
            "enabled": self._enabled,
            "opacity": self._opacity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Label":
        """Deserialize label from dictionary.

        Args:
            data: Dictionary representation of a label.

        Returns:
            New Label instance.
        """
        style_data = data.get("style", {})
        style = LabelStyle(
            font_family=data.get("font_family", style_data.get("font_family", "default")),
            font_size=data.get("font_size", style_data.get("font_size", 14.0)),
            font_weight=data.get("font_weight", style_data.get("font_weight", "normal")),
            text_color=data.get("text_color", style_data.get("text_color", "#000000")),
            disabled_text_color=data.get("disabled_text_color", style_data.get("disabled_text_color", "#888888")),
            icon_size=data.get("icon_size", style_data.get("icon_size", 16.0)),
            icon_spacing=data.get("icon_spacing", style_data.get("icon_spacing", 4.0)),
            icon_color=data.get("icon_color", style_data.get("icon_color")),
        )

        return cls(
            text=data.get("text", ""),
            icon=data.get("icon"),
            icon_position=IconPosition[data.get("icon_position", "LEADING")],
            text_align=TextAlign[data.get("text_align", "LEFT")],
            text_overflow=TextOverflow[data.get("text_overflow", "ELLIPSIS")],
            style=style,
            auto_size=data.get("auto_size", True),
            min_width=data.get("min_width", 0.0),
            max_width=data.get("max_width", 0.0),
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            width=data.get("width", 100.0),
            height=data.get("height", 20.0),
            visible=data.get("visible", True),
            enabled=data.get("enabled", True),
            opacity=data.get("opacity", 1.0),
        )

    def __repr__(self) -> str:
        """String representation."""
        icon_str = f", icon={self._icon!r}" if self._icon else ""
        return f"Label(text={self._text!r}{icon_str})"
