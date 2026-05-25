"""
Badge widget - notification indicators and count displays.

A Badge displays notification counts or status indicators.
It supports:
    - Count display (number)
    - Dot mode (indicator without number)
    - Maximum count display (99+)
    - Position anchoring (top-right, etc.)
    - Animation on value change
    - Color variants (primary, danger, success, etc.)

Example:
    badge = Badge(count=5)
    badge = Badge(count=100, max_count=99)  # Shows "99+"
    badge = Badge(mode=BadgeMode.DOT, variant=BadgeVariant.DANGER)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, TYPE_CHECKING
import time
import math


class BadgeMode(Enum):
    """Display mode for the badge."""
    COUNT = auto()    # Show numeric count
    DOT = auto()      # Show dot indicator only


class BadgePosition(Enum):
    """Anchor position for the badge relative to parent."""
    TOP_LEFT = auto()
    TOP_CENTER = auto()
    TOP_RIGHT = auto()
    CENTER_LEFT = auto()
    CENTER = auto()
    CENTER_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_CENTER = auto()
    BOTTOM_RIGHT = auto()


class BadgeVariant(Enum):
    """Color variant presets for the badge."""
    DEFAULT = auto()     # Default theme color
    PRIMARY = auto()     # Primary/brand color
    SECONDARY = auto()   # Secondary color
    SUCCESS = auto()     # Success/green
    WARNING = auto()     # Warning/yellow
    DANGER = auto()      # Danger/red
    INFO = auto()        # Info/blue


@dataclass(slots=True)
class BadgeStyle:
    """Style configuration for badge appearance."""
    # Colors (per variant)
    background_colors: dict[BadgeVariant, str] = field(default_factory=lambda: {
        BadgeVariant.DEFAULT: "#6B7280",     # Gray
        BadgeVariant.PRIMARY: "#3B82F6",     # Blue
        BadgeVariant.SECONDARY: "#8B5CF6",   # Purple
        BadgeVariant.SUCCESS: "#22C55E",     # Green
        BadgeVariant.WARNING: "#F59E0B",     # Yellow/Orange
        BadgeVariant.DANGER: "#EF4444",      # Red
        BadgeVariant.INFO: "#06B6D4",        # Cyan
    })
    text_color: str = "#FFFFFF"
    border_color: Optional[str] = None

    # Dimensions
    min_width: float = 18.0       # Minimum width for count mode
    height: float = 18.0          # Badge height
    dot_size: float = 10.0        # Size of dot in dot mode
    padding_x: float = 6.0        # Horizontal padding
    border_width: float = 0.0
    corner_radius: float = 9.0    # Half of height for pill shape

    # Typography
    font_size: float = 11.0
    font_weight: str = "bold"

    # Animation
    animate_changes: bool = True
    scale_duration: float = 0.15   # Duration of scale animation
    bounce_scale: float = 1.3      # Scale during bounce


class Badge:
    """
    Badge widget for notification indicators.

    Features:
    - Numeric count or dot-only mode
    - Maximum count with overflow indicator (99+)
    - Position anchoring relative to parent
    - Animation on value changes
    - Multiple color variants
    """

    __slots__ = (
        '_count', '_max_count', '_overflow_text',
        '_mode', '_position', '_variant',
        '_style',
        '_x', '_y', '_offset_x', '_offset_y',
        '_computed_width', '_computed_height',
        '_visible', '_enabled', '_opacity',
        '_is_animating', '_animation_start_time', '_animation_scale',
        '_pulse_enabled', '_pulse_phase',
        '_on_click',
        '_dirty_fields', '_id',
        '_parent', '_children',
    )

    _next_id: int = 0

    def __init__(
        self,
        count: int = 0,
        max_count: int = 99,
        mode: BadgeMode = BadgeMode.COUNT,
        position: BadgePosition = BadgePosition.TOP_RIGHT,
        variant: BadgeVariant = BadgeVariant.DANGER,
        style: Optional[BadgeStyle] = None,
        x: float = 0.0,
        y: float = 0.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ):
        """
        Initialize the badge widget.

        Args:
            count: Initial count value.
            max_count: Maximum count before showing overflow.
            mode: Display mode (count or dot).
            position: Anchor position relative to parent.
            variant: Color variant.
            style: Custom style configuration.
            x: X position.
            y: Y position.
            offset_x: Additional X offset from anchor.
            offset_y: Additional Y offset from anchor.
        """
        self._id = Badge._next_id
        Badge._next_id += 1

        self._count = max(0, count)
        self._max_count = max(1, max_count)
        self._overflow_text = "+"  # Shown after max_count

        self._mode = mode
        self._position = position
        self._variant = variant

        self._style = style or BadgeStyle()

        self._x = x
        self._y = y
        self._offset_x = offset_x
        self._offset_y = offset_y

        self._computed_width = 0.0
        self._computed_height = 0.0
        self._recalculate_size()

        self._visible = True
        self._enabled = True
        self._opacity = 1.0

        # Animation state
        self._is_animating = False
        self._animation_start_time = 0.0
        self._animation_scale = 1.0

        # Pulse animation for attention
        self._pulse_enabled = False
        self._pulse_phase = 0.0

        self._on_click: Optional[Callable[[], None]] = None

        self._dirty_fields: set[str] = set()

        self._parent = None
        self._children: list = []

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def id(self) -> int:
        """Get the widget ID."""
        return self._id

    @property
    def count(self) -> int:
        """Get the count value."""
        return self._count

    @count.setter
    def count(self, value: int) -> None:
        """Set the count value."""
        value = max(0, value)
        if self._count != value:
            old_count = self._count
            self._count = value
            self._recalculate_size()

            # Trigger animation on change
            if self._style.animate_changes and old_count != value:
                self._start_bounce_animation()

            self._mark_dirty("count")

    @property
    def max_count(self) -> int:
        """Get the maximum count before overflow."""
        return self._max_count

    @max_count.setter
    def max_count(self, value: int) -> None:
        """Set the maximum count."""
        value = max(1, value)
        if self._max_count != value:
            self._max_count = value
            self._recalculate_size()
            self._mark_dirty("max_count")

    @property
    def overflow_text(self) -> str:
        """Get the overflow indicator text."""
        return self._overflow_text

    @overflow_text.setter
    def overflow_text(self, value: str) -> None:
        """Set the overflow indicator text."""
        if self._overflow_text != value:
            self._overflow_text = value
            self._recalculate_size()
            self._mark_dirty("overflow_text")

    @property
    def mode(self) -> BadgeMode:
        """Get the display mode."""
        return self._mode

    @mode.setter
    def mode(self, value: BadgeMode) -> None:
        """Set the display mode."""
        if self._mode != value:
            self._mode = value
            self._recalculate_size()
            self._mark_dirty("mode")

    @property
    def position(self) -> BadgePosition:
        """Get the anchor position."""
        return self._position

    @position.setter
    def position(self, value: BadgePosition) -> None:
        """Set the anchor position."""
        if self._position != value:
            self._position = value
            self._mark_dirty("position")

    @property
    def variant(self) -> BadgeVariant:
        """Get the color variant."""
        return self._variant

    @variant.setter
    def variant(self, value: BadgeVariant) -> None:
        """Set the color variant."""
        if self._variant != value:
            self._variant = value
            self._mark_dirty("variant")

    @property
    def style(self) -> BadgeStyle:
        """Get the style configuration."""
        return self._style

    @style.setter
    def style(self, value: BadgeStyle) -> None:
        """Set the style configuration."""
        self._style = value
        self._recalculate_size()
        self._mark_dirty("style")

    @property
    def x(self) -> float:
        """Get the X position."""
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        """Set the X position."""
        if self._x != value:
            self._x = value
            self._mark_dirty("x")

    @property
    def y(self) -> float:
        """Get the Y position."""
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        """Set the Y position."""
        if self._y != value:
            self._y = value
            self._mark_dirty("y")

    @property
    def offset_x(self) -> float:
        """Get the X offset from anchor."""
        return self._offset_x

    @offset_x.setter
    def offset_x(self, value: float) -> None:
        """Set the X offset from anchor."""
        if self._offset_x != value:
            self._offset_x = value
            self._mark_dirty("offset_x")

    @property
    def offset_y(self) -> float:
        """Get the Y offset from anchor."""
        return self._offset_y

    @offset_y.setter
    def offset_y(self, value: float) -> None:
        """Set the Y offset from anchor."""
        if self._offset_y != value:
            self._offset_y = value
            self._mark_dirty("offset_y")

    @property
    def width(self) -> float:
        """Get the computed width."""
        return self._computed_width

    @property
    def height(self) -> float:
        """Get the computed height."""
        return self._computed_height

    @property
    def visible(self) -> bool:
        """Get visibility."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set visibility."""
        if self._visible != value:
            self._visible = value
            self._mark_dirty("visible")

    @property
    def enabled(self) -> bool:
        """Get enabled state."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        if self._enabled != value:
            self._enabled = value
            self._mark_dirty("enabled")

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
    def pulse_enabled(self) -> bool:
        """Get whether pulse animation is enabled."""
        return self._pulse_enabled

    @pulse_enabled.setter
    def pulse_enabled(self, value: bool) -> None:
        """Set whether pulse animation is enabled."""
        if self._pulse_enabled != value:
            self._pulse_enabled = value
            self._pulse_phase = 0.0
            self._mark_dirty("pulse_enabled")

    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================

    @property
    def display_text(self) -> str:
        """Get the text to display."""
        if self._mode == BadgeMode.DOT:
            return ""

        if self._count > self._max_count:
            return f"{self._max_count}{self._overflow_text}"
        return str(self._count)

    @property
    def is_overflowing(self) -> bool:
        """Check if count exceeds max."""
        return self._count > self._max_count

    @property
    def is_empty(self) -> bool:
        """Check if count is zero."""
        return self._count == 0

    @property
    def background_color(self) -> str:
        """Get the background color for current variant."""
        return self._style.background_colors.get(
            self._variant,
            self._style.background_colors[BadgeVariant.DEFAULT]
        )

    @property
    def current_scale(self) -> float:
        """Get the current animation scale."""
        return self._animation_scale

    @property
    def is_animating(self) -> bool:
        """Check if currently animating."""
        return self._is_animating

    # =========================================================================
    # METHODS
    # =========================================================================

    def increment(self, amount: int = 1) -> None:
        """Increment the count."""
        self.count = self._count + amount

    def decrement(self, amount: int = 1) -> None:
        """Decrement the count (minimum 0)."""
        self.count = max(0, self._count - amount)

    def clear(self) -> None:
        """Clear the count to zero."""
        self.count = 0

    def set_count_silent(self, value: int) -> None:
        """Set count without animation."""
        value = max(0, value)
        if self._count != value:
            self._count = value
            self._recalculate_size()
            self._mark_dirty("count")

    def update(self, delta_time: float) -> None:
        """
        Update animation state.

        Args:
            delta_time: Time since last update in seconds.
        """
        # Update bounce animation
        if self._is_animating:
            elapsed = time.time() - self._animation_start_time
            duration = self._style.scale_duration

            if elapsed >= duration:
                self._is_animating = False
                self._animation_scale = 1.0
            else:
                # Bounce effect: scale up then back down
                t = elapsed / duration
                # Ease out back
                scale_range = self._style.bounce_scale - 1.0
                self._animation_scale = 1.0 + scale_range * math.sin(t * math.pi)

        # Update pulse animation
        if self._pulse_enabled:
            self._pulse_phase += delta_time * 2.0  # 2 Hz
            if self._pulse_phase > 2 * math.pi:
                self._pulse_phase -= 2 * math.pi

    def get_pulse_opacity(self) -> float:
        """Get the current opacity including pulse effect."""
        if not self._pulse_enabled:
            return self._opacity

        # Pulse between 70% and 100% opacity
        pulse_factor = 0.85 + 0.15 * math.sin(self._pulse_phase)
        return self._opacity * pulse_factor

    # =========================================================================
    # POSITIONING
    # =========================================================================

    def calculate_position(
        self,
        parent_x: float,
        parent_y: float,
        parent_width: float,
        parent_height: float,
    ) -> tuple[float, float]:
        """
        Calculate the badge position based on anchor and parent bounds.

        Args:
            parent_x: Parent X position.
            parent_y: Parent Y position.
            parent_width: Parent width.
            parent_height: Parent height.

        Returns:
            Tuple of (x, y) for the badge position.
        """
        # Calculate anchor point
        anchor_x = parent_x
        anchor_y = parent_y

        # Horizontal anchor
        if self._position in (
            BadgePosition.TOP_CENTER,
            BadgePosition.CENTER,
            BadgePosition.BOTTOM_CENTER,
        ):
            anchor_x = parent_x + parent_width / 2 - self._computed_width / 2
        elif self._position in (
            BadgePosition.TOP_RIGHT,
            BadgePosition.CENTER_RIGHT,
            BadgePosition.BOTTOM_RIGHT,
        ):
            anchor_x = parent_x + parent_width - self._computed_width / 2
        else:  # LEFT positions
            anchor_x = parent_x - self._computed_width / 2

        # Vertical anchor
        if self._position in (
            BadgePosition.CENTER_LEFT,
            BadgePosition.CENTER,
            BadgePosition.CENTER_RIGHT,
        ):
            anchor_y = parent_y + parent_height / 2 - self._computed_height / 2
        elif self._position in (
            BadgePosition.BOTTOM_LEFT,
            BadgePosition.BOTTOM_CENTER,
            BadgePosition.BOTTOM_RIGHT,
        ):
            anchor_y = parent_y + parent_height - self._computed_height / 2
        else:  # TOP positions
            anchor_y = parent_y - self._computed_height / 2

        # Apply offset
        final_x = anchor_x + self._offset_x
        final_y = anchor_y + self._offset_y

        return (final_x, final_y)

    def update_position_from_parent(
        self,
        parent_x: float,
        parent_y: float,
        parent_width: float,
        parent_height: float,
    ) -> None:
        """
        Update badge position based on parent bounds.

        Args:
            parent_x: Parent X position.
            parent_y: Parent Y position.
            parent_width: Parent width.
            parent_height: Parent height.
        """
        x, y = self.calculate_position(parent_x, parent_y, parent_width, parent_height)
        self._x = x
        self._y = y
        self._mark_dirty("position")

    # =========================================================================
    # RENDERING HELPERS
    # =========================================================================

    def get_bounds(self) -> tuple[float, float, float, float]:
        """
        Get the badge bounds for rendering.

        Returns:
            Tuple of (x, y, width, height).
        """
        return (self._x, self._y, self._computed_width, self._computed_height)

    def get_scaled_bounds(self) -> tuple[float, float, float, float]:
        """
        Get the badge bounds with current animation scale applied.

        Returns:
            Tuple of (x, y, width, height) with scale applied around center.
        """
        scale = self._animation_scale
        if scale == 1.0:
            return self.get_bounds()

        # Scale around center
        center_x = self._x + self._computed_width / 2
        center_y = self._y + self._computed_height / 2

        scaled_width = self._computed_width * scale
        scaled_height = self._computed_height * scale

        scaled_x = center_x - scaled_width / 2
        scaled_y = center_y - scaled_height / 2

        return (scaled_x, scaled_y, scaled_width, scaled_height)

    def contains_point(self, px: float, py: float) -> bool:
        """
        Check if a point is inside the badge bounds.

        Args:
            px: Point X coordinate.
            py: Point Y coordinate.

        Returns:
            True if point is inside bounds.
        """
        return (
            self._x <= px <= self._x + self._computed_width and
            self._y <= py <= self._y + self._computed_height
        )

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_click(self, callback: Optional[Callable[[], None]]) -> None:
        """Set click callback."""
        self._on_click = callback

    def handle_click(self) -> bool:
        """
        Handle a click event.

        Returns:
            True if the click was handled.
        """
        if not self._visible or not self._enabled:
            return False

        if self._on_click:
            self._on_click()
            return True
        return False

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _recalculate_size(self) -> None:
        """Recalculate the badge dimensions."""
        if self._mode == BadgeMode.DOT:
            self._computed_width = self._style.dot_size
            self._computed_height = self._style.dot_size
        else:
            # Calculate width based on text
            text = self.display_text
            char_count = len(text)

            # Estimate text width
            char_width = self._style.font_size * 0.65
            text_width = char_count * char_width

            # Add padding
            self._computed_width = max(
                self._style.min_width,
                text_width + self._style.padding_x * 2
            )
            self._computed_height = self._style.height

    def _start_bounce_animation(self) -> None:
        """Start the bounce animation."""
        if self._style.animate_changes:
            self._is_animating = True
            self._animation_start_time = time.time()
            self._animation_scale = 1.0

    def _mark_dirty(self, field_name: str) -> None:
        """Mark a field as dirty."""
        self._dirty_fields.add(field_name)

    def is_dirty(self, field_name: Optional[str] = None) -> bool:
        """Check if a field or any field is dirty."""
        if field_name is None:
            return len(self._dirty_fields) > 0
        return field_name in self._dirty_fields

    def clear_dirty(self) -> None:
        """Clear all dirty flags."""
        self._dirty_fields.clear()

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize badge to dictionary."""
        return {
            "count": self._count,
            "max_count": self._max_count,
            "overflow_text": self._overflow_text,
            "mode": self._mode.name,
            "position": self._position.name,
            "variant": self._variant.name,
            "x": self._x,
            "y": self._y,
            "offset_x": self._offset_x,
            "offset_y": self._offset_y,
            "visible": self._visible,
            "enabled": self._enabled,
            "opacity": self._opacity,
            "pulse_enabled": self._pulse_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Badge":
        """Deserialize badge from dictionary."""
        badge = cls(
            count=data.get("count", 0),
            max_count=data.get("max_count", 99),
            mode=BadgeMode[data.get("mode", "COUNT")],
            position=BadgePosition[data.get("position", "TOP_RIGHT")],
            variant=BadgeVariant[data.get("variant", "DANGER")],
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            offset_x=data.get("offset_x", 0.0),
            offset_y=data.get("offset_y", 0.0),
        )

        badge._overflow_text = data.get("overflow_text", "+")
        badge._visible = data.get("visible", True)
        badge._enabled = data.get("enabled", True)
        badge._opacity = data.get("opacity", 1.0)
        badge._pulse_enabled = data.get("pulse_enabled", False)

        return badge

    def __repr__(self) -> str:
        """String representation."""
        if self._mode == BadgeMode.DOT:
            return f"Badge(mode=DOT, variant={self._variant.name})"
        return f"Badge(count={self._count}, variant={self._variant.name})"
