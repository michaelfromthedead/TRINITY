"""
Tooltip system - contextual information display on hover.

The tooltip system provides:
    - TooltipManager singleton for global tooltip handling
    - Delay before showing (configurable hover delay)
    - Position calculation (avoids screen edges)
    - Rich content support (title, description, stats)
    - Custom tooltip widget support
    - Show/hide animations

Example:
    # Simple text tooltip
    manager = TooltipManager.get_instance()
    manager.show("This is a tooltip", target_widget)

    # Rich tooltip
    content = TooltipContent(
        title="Sword of Fire",
        description="A legendary weapon",
        stats=[("Damage", "50-75"), ("Fire Damage", "+10")]
    )
    manager.show_rich(content, target_widget)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING
import time
import math


class TooltipPosition(Enum):
    """Preferred position for tooltip relative to target."""
    AUTO = auto()         # Automatically choose best position
    TOP = auto()          # Above the target
    BOTTOM = auto()       # Below the target
    LEFT = auto()         # Left of the target
    RIGHT = auto()        # Right of the target
    TOP_LEFT = auto()     # Above and to the left
    TOP_RIGHT = auto()    # Above and to the right
    BOTTOM_LEFT = auto()  # Below and to the left
    BOTTOM_RIGHT = auto() # Below and to the right
    CURSOR = auto()       # Follow cursor position


class TooltipAnimation(Enum):
    """Animation style for showing/hiding tooltips."""
    NONE = auto()         # No animation
    FADE = auto()         # Fade in/out
    SCALE = auto()        # Scale from small to full size
    SLIDE = auto()        # Slide in from direction


@dataclass(slots=True)
class TooltipContent:
    """Rich content for tooltips."""
    title: str = ""                          # Bold title text
    description: str = ""                    # Description/body text
    stats: list[tuple[str, str]] = field(default_factory=list)  # Key-value stats
    icon: Optional[str] = None               # Icon name/reference
    rarity_color: Optional[str] = None       # Color for rarity indicator
    footer: str = ""                         # Footer text (e.g., "Right-click to use")
    custom_data: dict[str, Any] = field(default_factory=dict)   # Extension data


@dataclass(slots=True)
class TooltipStyle:
    """Style configuration for tooltip appearance."""
    # Colors
    background_color: str = "#1F2937"        # Dark background
    border_color: str = "#374151"            # Border color
    title_color: str = "#F9FAFB"             # Title text color
    text_color: str = "#D1D5DB"              # Body text color
    stat_label_color: str = "#9CA3AF"        # Stat label color
    stat_value_color: str = "#F9FAFB"        # Stat value color
    footer_color: str = "#6B7280"            # Footer text color

    # Dimensions
    max_width: float = 300.0                 # Maximum tooltip width
    min_width: float = 100.0                 # Minimum tooltip width
    padding: float = 12.0                    # Internal padding
    border_width: float = 1.0                # Border width
    corner_radius: float = 6.0               # Corner radius

    # Typography
    title_font_size: float = 14.0
    text_font_size: float = 12.0
    stat_font_size: float = 11.0
    footer_font_size: float = 10.0
    line_spacing: float = 4.0                # Space between lines
    section_spacing: float = 8.0             # Space between sections

    # Animation
    animation: TooltipAnimation = TooltipAnimation.FADE
    animation_duration: float = 0.15         # Animation duration in seconds

    # Shadow
    shadow_enabled: bool = True
    shadow_color: str = "#00000080"          # Semi-transparent black
    shadow_offset_x: float = 2.0
    shadow_offset_y: float = 4.0
    shadow_blur: float = 8.0


class TooltipTarget(Protocol):
    """Protocol for objects that can be tooltip targets."""

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...


class Tooltip:
    """
    Individual tooltip widget.

    Represents a single tooltip instance with content and positioning.
    """

    __slots__ = (
        '_content', '_text', '_style',
        '_x', '_y', '_width', '_height',
        '_target_x', '_target_y', '_target_width', '_target_height',
        '_preferred_position', '_actual_position',
        '_visible', '_opacity',
        '_animation_progress', '_animation_start_time', '_is_showing',
        '_screen_width', '_screen_height',
        '_margin', '_offset',
        '_dirty_fields', '_id',
    )

    _next_id: int = 0

    def __init__(
        self,
        text: str = "",
        content: Optional[TooltipContent] = None,
        style: Optional[TooltipStyle] = None,
        preferred_position: TooltipPosition = TooltipPosition.AUTO,
        screen_width: float = 1920.0,
        screen_height: float = 1080.0,
    ):
        """
        Initialize the tooltip.

        Args:
            text: Simple text content.
            content: Rich content (overrides text if provided).
            style: Visual style configuration.
            preferred_position: Preferred position relative to target.
            screen_width: Screen width for bounds checking.
            screen_height: Screen height for bounds checking.
        """
        self._id = Tooltip._next_id
        Tooltip._next_id += 1

        self._text = text
        self._content = content
        self._style = style or TooltipStyle()

        self._x = 0.0
        self._y = 0.0
        self._width = 0.0
        self._height = 0.0

        self._target_x = 0.0
        self._target_y = 0.0
        self._target_width = 0.0
        self._target_height = 0.0

        self._preferred_position = preferred_position
        self._actual_position = preferred_position

        self._visible = False
        self._opacity = 0.0

        self._animation_progress = 0.0
        self._animation_start_time = 0.0
        self._is_showing = False

        self._screen_width = screen_width
        self._screen_height = screen_height

        self._margin = 8.0    # Margin from screen edges
        self._offset = 8.0    # Offset from target

        self._dirty_fields: set[str] = set()

        self._calculate_size()

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def id(self) -> int:
        """Get the tooltip ID."""
        return self._id

    @property
    def text(self) -> str:
        """Get the simple text content."""
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """Set the simple text content."""
        if self._text != value:
            self._text = value
            self._content = None  # Clear rich content
            self._calculate_size()
            self._mark_dirty("text")

    @property
    def content(self) -> Optional[TooltipContent]:
        """Get the rich content."""
        return self._content

    @content.setter
    def content(self, value: Optional[TooltipContent]) -> None:
        """Set the rich content."""
        self._content = value
        self._calculate_size()
        self._mark_dirty("content")

    @property
    def style(self) -> TooltipStyle:
        """Get the style configuration."""
        return self._style

    @style.setter
    def style(self, value: TooltipStyle) -> None:
        """Set the style configuration."""
        self._style = value
        self._calculate_size()
        self._mark_dirty("style")

    @property
    def x(self) -> float:
        """Get the X position."""
        return self._x

    @property
    def y(self) -> float:
        """Get the Y position."""
        return self._y

    @property
    def width(self) -> float:
        """Get the width."""
        return self._width

    @property
    def height(self) -> float:
        """Get the height."""
        return self._height

    @property
    def visible(self) -> bool:
        """Get visibility."""
        return self._visible

    @property
    def opacity(self) -> float:
        """Get the current opacity."""
        return self._opacity

    @property
    def is_animating(self) -> bool:
        """Check if currently animating."""
        return 0.0 < self._animation_progress < 1.0

    @property
    def preferred_position(self) -> TooltipPosition:
        """Get the preferred position."""
        return self._preferred_position

    @preferred_position.setter
    def preferred_position(self, value: TooltipPosition) -> None:
        """Set the preferred position."""
        if self._preferred_position != value:
            self._preferred_position = value
            self._mark_dirty("preferred_position")

    @property
    def actual_position(self) -> TooltipPosition:
        """Get the actual position (after adjustment)."""
        return self._actual_position

    @property
    def has_rich_content(self) -> bool:
        """Check if tooltip has rich content."""
        return self._content is not None

    # =========================================================================
    # POSITIONING
    # =========================================================================

    def set_target(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        """
        Set the target widget bounds.

        Args:
            x: Target X position.
            y: Target Y position.
            width: Target width.
            height: Target height.
        """
        self._target_x = x
        self._target_y = y
        self._target_width = width
        self._target_height = height
        self._calculate_position()

    def set_target_widget(self, target: TooltipTarget) -> None:
        """
        Set target from a widget.

        Args:
            target: Widget with x, y, width, height properties.
        """
        self.set_target(target.x, target.y, target.width, target.height)

    def set_cursor_position(self, cursor_x: float, cursor_y: float) -> None:
        """
        Position tooltip near cursor.

        Args:
            cursor_x: Cursor X position.
            cursor_y: Cursor Y position.
        """
        self._target_x = cursor_x
        self._target_y = cursor_y
        self._target_width = 1.0
        self._target_height = 1.0
        self._calculate_position()

    def set_screen_bounds(self, width: float, height: float) -> None:
        """
        Set screen bounds for edge avoidance.

        Args:
            width: Screen width.
            height: Screen height.
        """
        self._screen_width = width
        self._screen_height = height
        self._calculate_position()

    def _calculate_position(self) -> None:
        """Calculate tooltip position based on target and preferred position."""
        # Get candidate positions
        positions = self._get_candidate_positions()

        # Try preferred position first
        if self._preferred_position != TooltipPosition.AUTO:
            x, y = positions.get(self._preferred_position, (0, 0))
            if self._fits_on_screen(x, y):
                self._x = x
                self._y = y
                self._actual_position = self._preferred_position
                return

        # Try each position in order of preference
        preference_order = [
            TooltipPosition.TOP,
            TooltipPosition.BOTTOM,
            TooltipPosition.RIGHT,
            TooltipPosition.LEFT,
            TooltipPosition.TOP_RIGHT,
            TooltipPosition.TOP_LEFT,
            TooltipPosition.BOTTOM_RIGHT,
            TooltipPosition.BOTTOM_LEFT,
        ]

        for pos in preference_order:
            if pos in positions:
                x, y = positions[pos]
                if self._fits_on_screen(x, y):
                    self._x = x
                    self._y = y
                    self._actual_position = pos
                    return

        # Fallback: clamp to screen bounds
        x, y = positions.get(TooltipPosition.BOTTOM, (0, 0))
        self._x = self._clamp_to_screen_x(x)
        self._y = self._clamp_to_screen_y(y)
        self._actual_position = TooltipPosition.BOTTOM

    def _get_candidate_positions(self) -> dict[TooltipPosition, tuple[float, float]]:
        """Get all candidate positions."""
        tx, ty = self._target_x, self._target_y
        tw, th = self._target_width, self._target_height
        offset = self._offset

        # Center of target
        cx = tx + tw / 2
        cy = ty + th / 2

        return {
            TooltipPosition.TOP: (
                cx - self._width / 2,
                ty - self._height - offset
            ),
            TooltipPosition.BOTTOM: (
                cx - self._width / 2,
                ty + th + offset
            ),
            TooltipPosition.LEFT: (
                tx - self._width - offset,
                cy - self._height / 2
            ),
            TooltipPosition.RIGHT: (
                tx + tw + offset,
                cy - self._height / 2
            ),
            TooltipPosition.TOP_LEFT: (
                tx - self._width,
                ty - self._height - offset
            ),
            TooltipPosition.TOP_RIGHT: (
                tx + tw,
                ty - self._height - offset
            ),
            TooltipPosition.BOTTOM_LEFT: (
                tx - self._width,
                ty + th + offset
            ),
            TooltipPosition.BOTTOM_RIGHT: (
                tx + tw,
                ty + th + offset
            ),
            TooltipPosition.CURSOR: (
                tx + 16,  # Offset from cursor
                ty + 20
            ),
        }

    def _fits_on_screen(self, x: float, y: float) -> bool:
        """Check if tooltip fits on screen at given position."""
        return (
            x >= self._margin and
            y >= self._margin and
            x + self._width <= self._screen_width - self._margin and
            y + self._height <= self._screen_height - self._margin
        )

    def _clamp_to_screen_x(self, x: float) -> float:
        """Clamp X position to screen bounds."""
        return max(
            self._margin,
            min(self._screen_width - self._width - self._margin, x)
        )

    def _clamp_to_screen_y(self, y: float) -> float:
        """Clamp Y position to screen bounds."""
        return max(
            self._margin,
            min(self._screen_height - self._height - self._margin, y)
        )

    # =========================================================================
    # SIZE CALCULATION
    # =========================================================================

    def _calculate_size(self) -> None:
        """Calculate tooltip size based on content."""
        padding = self._style.padding

        if self._content:
            self._width, self._height = self._calculate_rich_size()
        elif self._text:
            self._width, self._height = self._calculate_text_size(self._text)
        else:
            self._width = self._style.min_width
            self._height = padding * 2 + self._style.text_font_size

        # Apply constraints
        self._width = max(self._style.min_width, min(self._style.max_width, self._width))

    def _calculate_text_size(self, text: str) -> tuple[float, float]:
        """Calculate size for simple text content."""
        padding = self._style.padding
        font_size = self._style.text_font_size
        char_width = font_size * 0.55

        # Simple width estimation
        width = len(text) * char_width + padding * 2
        width = min(width, self._style.max_width)

        # Wrap text if needed
        max_chars = int((self._style.max_width - padding * 2) / char_width)
        lines = max(1, math.ceil(len(text) / max_chars))

        height = lines * (font_size + self._style.line_spacing) + padding * 2

        return (width, height)

    def _calculate_rich_size(self) -> tuple[float, float]:
        """Calculate size for rich content."""
        if not self._content:
            return (0.0, 0.0)

        padding = self._style.padding
        spacing = self._style.section_spacing
        line_spacing = self._style.line_spacing

        width = self._style.min_width
        height = padding

        # Title
        if self._content.title:
            title_height = self._style.title_font_size + line_spacing
            title_width = len(self._content.title) * self._style.title_font_size * 0.6
            width = max(width, title_width + padding * 2)
            height += title_height + spacing

        # Description
        if self._content.description:
            desc_width, desc_height = self._calculate_text_size(self._content.description)
            width = max(width, desc_width)
            height += desc_height - padding + spacing

        # Stats
        if self._content.stats:
            for label, value in self._content.stats:
                stat_width = (len(label) + len(value) + 2) * self._style.stat_font_size * 0.55
                width = max(width, stat_width + padding * 2)
                height += self._style.stat_font_size + line_spacing
            height += spacing

        # Footer
        if self._content.footer:
            footer_width = len(self._content.footer) * self._style.footer_font_size * 0.55
            width = max(width, footer_width + padding * 2)
            height += self._style.footer_font_size + line_spacing

        height += padding

        return (width, height)

    # =========================================================================
    # SHOW/HIDE
    # =========================================================================

    def show(self) -> None:
        """Show the tooltip with animation."""
        self._visible = True
        self._is_showing = True
        self._animation_start_time = time.time()
        self._animation_progress = 0.0

    def hide(self) -> None:
        """Hide the tooltip with animation."""
        self._is_showing = False
        self._animation_start_time = time.time()
        self._animation_progress = 0.0

    def show_immediate(self) -> None:
        """Show the tooltip immediately without animation."""
        self._visible = True
        self._opacity = 1.0
        self._animation_progress = 1.0
        self._is_showing = True

    def hide_immediate(self) -> None:
        """Hide the tooltip immediately without animation."""
        self._visible = False
        self._opacity = 0.0
        self._animation_progress = 1.0
        self._is_showing = False

    def update(self, delta_time: float) -> None:
        """
        Update animation state.

        Args:
            delta_time: Time since last update in seconds.
        """
        if self._style.animation == TooltipAnimation.NONE:
            self._opacity = 1.0 if self._is_showing else 0.0
            self._visible = self._is_showing
            self._animation_progress = 1.0
            return

        if self._animation_progress < 1.0:
            elapsed = time.time() - self._animation_start_time
            duration = self._style.animation_duration

            if duration > 0:
                self._animation_progress = min(1.0, elapsed / duration)
            else:
                self._animation_progress = 1.0

            # Calculate opacity based on animation state
            if self._is_showing:
                self._opacity = self._ease_out_quad(self._animation_progress)
            else:
                self._opacity = 1.0 - self._ease_out_quad(self._animation_progress)

                # Hide when animation completes
                if self._animation_progress >= 1.0:
                    self._visible = False

    @staticmethod
    def _ease_out_quad(t: float) -> float:
        """Quadratic ease-out."""
        return 1.0 - (1.0 - t) * (1.0 - t)

    # =========================================================================
    # UTILITY
    # =========================================================================

    def _mark_dirty(self, field_name: str) -> None:
        """Mark a field as dirty."""
        self._dirty_fields.add(field_name)

    def is_dirty(self, field_name: Optional[str] = None) -> bool:
        """Check if dirty."""
        if field_name is None:
            return len(self._dirty_fields) > 0
        return field_name in self._dirty_fields

    def clear_dirty(self) -> None:
        """Clear dirty flags."""
        self._dirty_fields.clear()

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Get tooltip bounds."""
        return (self._x, self._y, self._width, self._height)

    def __repr__(self) -> str:
        """String representation."""
        if self._content:
            return f"Tooltip(title={self._content.title!r}, visible={self._visible})"
        return f"Tooltip(text={self._text!r}, visible={self._visible})"


class RichTooltip(Tooltip):
    """
    Specialized tooltip for rich game content.

    Convenience class with preset styling for game items, abilities, etc.
    """

    def __init__(
        self,
        title: str = "",
        description: str = "",
        stats: Optional[list[tuple[str, str]]] = None,
        icon: Optional[str] = None,
        rarity_color: Optional[str] = None,
        footer: str = "",
        **kwargs,
    ):
        """
        Initialize a rich tooltip.

        Args:
            title: Item/ability name.
            description: Flavor text or description.
            stats: List of (label, value) stat pairs.
            icon: Icon name/reference.
            rarity_color: Color for rarity indicator.
            footer: Footer text.
            **kwargs: Additional Tooltip arguments.
        """
        content = TooltipContent(
            title=title,
            description=description,
            stats=stats or [],
            icon=icon,
            rarity_color=rarity_color,
            footer=footer,
        )
        super().__init__(content=content, **kwargs)

    def set_item(
        self,
        name: str,
        description: str = "",
        stats: Optional[list[tuple[str, str]]] = None,
        rarity: str = "common",
    ) -> None:
        """
        Configure tooltip for an item.

        Args:
            name: Item name.
            description: Item description.
            stats: Item stats.
            rarity: Rarity level (common, uncommon, rare, epic, legendary).
        """
        rarity_colors = {
            "common": "#9CA3AF",
            "uncommon": "#22C55E",
            "rare": "#3B82F6",
            "epic": "#A855F7",
            "legendary": "#F59E0B",
        }

        self._content = TooltipContent(
            title=name,
            description=description,
            stats=stats or [],
            rarity_color=rarity_colors.get(rarity.lower(), "#9CA3AF"),
        )
        self._calculate_size()


class TooltipManager:
    """
    Singleton manager for the tooltip system.

    Handles tooltip lifecycle, hover delay, and global configuration.
    """

    _instance: Optional["TooltipManager"] = None

    def __new__(cls) -> "TooltipManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the manager."""
        self._current_tooltip: Optional[Tooltip] = None
        self._pending_target: Optional[Any] = None
        self._pending_content: Optional[Any] = None
        self._pending_start_time: float = 0.0

        self._hover_delay: float = 0.5           # Delay before showing
        self._hide_delay: float = 0.1            # Delay before hiding
        self._default_style: TooltipStyle = TooltipStyle()

        self._screen_width: float = 1920.0
        self._screen_height: float = 1080.0

        self._is_enabled: bool = True
        self._follow_cursor: bool = False
        self._cursor_x: float = 0.0
        self._cursor_y: float = 0.0

        self._hide_start_time: float = 0.0
        self._is_hiding: bool = False

    @classmethod
    def get_instance(cls) -> "TooltipManager":
        """Get the singleton instance."""
        return cls()

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    @property
    def hover_delay(self) -> float:
        """Get the hover delay in seconds."""
        return self._hover_delay

    @hover_delay.setter
    def hover_delay(self, value: float) -> None:
        """Set the hover delay in seconds."""
        self._hover_delay = max(0.0, value)

    @property
    def hide_delay(self) -> float:
        """Get the hide delay in seconds."""
        return self._hide_delay

    @hide_delay.setter
    def hide_delay(self, value: float) -> None:
        """Set the hide delay in seconds."""
        self._hide_delay = max(0.0, value)

    @property
    def default_style(self) -> TooltipStyle:
        """Get the default tooltip style."""
        return self._default_style

    @default_style.setter
    def default_style(self, value: TooltipStyle) -> None:
        """Set the default tooltip style."""
        self._default_style = value

    @property
    def is_enabled(self) -> bool:
        """Get whether tooltips are enabled."""
        return self._is_enabled

    @is_enabled.setter
    def is_enabled(self, value: bool) -> None:
        """Set whether tooltips are enabled."""
        self._is_enabled = value
        if not value:
            self.hide_immediate()

    @property
    def follow_cursor(self) -> bool:
        """Get whether tooltips follow cursor."""
        return self._follow_cursor

    @follow_cursor.setter
    def follow_cursor(self, value: bool) -> None:
        """Set whether tooltips follow cursor."""
        self._follow_cursor = value

    @property
    def current_tooltip(self) -> Optional[Tooltip]:
        """Get the current tooltip."""
        return self._current_tooltip

    @property
    def is_showing(self) -> bool:
        """Check if a tooltip is showing."""
        return self._current_tooltip is not None and self._current_tooltip.visible

    # =========================================================================
    # SCREEN BOUNDS
    # =========================================================================

    def set_screen_bounds(self, width: float, height: float) -> None:
        """
        Set screen bounds for tooltip positioning.

        Args:
            width: Screen width.
            height: Screen height.
        """
        self._screen_width = width
        self._screen_height = height
        if self._current_tooltip:
            self._current_tooltip.set_screen_bounds(width, height)

    # =========================================================================
    # SHOWING TOOLTIPS
    # =========================================================================

    def request_show(
        self,
        text: str,
        target: TooltipTarget,
        position: TooltipPosition = TooltipPosition.AUTO,
        style: Optional[TooltipStyle] = None,
    ) -> None:
        """
        Request to show a simple text tooltip.

        Tooltip will appear after hover delay.

        Args:
            text: Tooltip text.
            target: Target widget.
            position: Preferred position.
            style: Custom style (None = use default).
        """
        if not self._is_enabled:
            return

        self._pending_content = text
        self._pending_target = target
        self._pending_start_time = time.time()
        self._is_hiding = False

    def request_show_rich(
        self,
        content: TooltipContent,
        target: TooltipTarget,
        position: TooltipPosition = TooltipPosition.AUTO,
        style: Optional[TooltipStyle] = None,
    ) -> None:
        """
        Request to show a rich content tooltip.

        Args:
            content: Rich tooltip content.
            target: Target widget.
            position: Preferred position.
            style: Custom style.
        """
        if not self._is_enabled:
            return

        self._pending_content = content
        self._pending_target = target
        self._pending_start_time = time.time()
        self._is_hiding = False

    def show_immediate(
        self,
        text: str,
        target: TooltipTarget,
        position: TooltipPosition = TooltipPosition.AUTO,
        style: Optional[TooltipStyle] = None,
    ) -> None:
        """
        Show a tooltip immediately without delay.

        Args:
            text: Tooltip text.
            target: Target widget.
            position: Preferred position.
            style: Custom style.
        """
        if not self._is_enabled:
            return

        self._create_tooltip(text, target, position, style)
        if self._current_tooltip:
            self._current_tooltip.show_immediate()

    def show_rich_immediate(
        self,
        content: TooltipContent,
        target: TooltipTarget,
        position: TooltipPosition = TooltipPosition.AUTO,
        style: Optional[TooltipStyle] = None,
    ) -> None:
        """
        Show a rich tooltip immediately.

        Args:
            content: Rich content.
            target: Target widget.
            position: Preferred position.
            style: Custom style.
        """
        if not self._is_enabled:
            return

        self._create_tooltip(content, target, position, style)
        if self._current_tooltip:
            self._current_tooltip.show_immediate()

    def _create_tooltip(
        self,
        content: Any,
        target: TooltipTarget,
        position: TooltipPosition,
        style: Optional[TooltipStyle],
    ) -> None:
        """Create and configure a tooltip."""
        tooltip_style = style or self._default_style

        if isinstance(content, TooltipContent):
            self._current_tooltip = Tooltip(
                content=content,
                style=tooltip_style,
                preferred_position=position,
                screen_width=self._screen_width,
                screen_height=self._screen_height,
            )
        else:
            self._current_tooltip = Tooltip(
                text=str(content),
                style=tooltip_style,
                preferred_position=position,
                screen_width=self._screen_width,
                screen_height=self._screen_height,
            )

        self._current_tooltip.set_target_widget(target)

    # =========================================================================
    # HIDING TOOLTIPS
    # =========================================================================

    def request_hide(self) -> None:
        """Request to hide the current tooltip (with delay)."""
        if self._current_tooltip and self._current_tooltip.visible:
            self._is_hiding = True
            self._hide_start_time = time.time()

        # Cancel pending show
        self._pending_target = None
        self._pending_content = None

    def hide(self) -> None:
        """Hide the current tooltip with animation."""
        if self._current_tooltip:
            self._current_tooltip.hide()

        self._pending_target = None
        self._pending_content = None
        self._is_hiding = False

    def hide_immediate(self) -> None:
        """Hide the current tooltip immediately."""
        if self._current_tooltip:
            self._current_tooltip.hide_immediate()

        self._current_tooltip = None
        self._pending_target = None
        self._pending_content = None
        self._is_hiding = False

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(self, delta_time: float) -> None:
        """
        Update the tooltip system.

        Args:
            delta_time: Time since last update in seconds.
        """
        current_time = time.time()

        # Check pending show
        if self._pending_target is not None and self._pending_content is not None:
            elapsed = current_time - self._pending_start_time
            if elapsed >= self._hover_delay:
                self._create_tooltip(
                    self._pending_content,
                    self._pending_target,
                    TooltipPosition.AUTO,
                    None,
                )
                if self._current_tooltip:
                    self._current_tooltip.show()

                self._pending_target = None
                self._pending_content = None

        # Check pending hide
        if self._is_hiding and self._current_tooltip:
            elapsed = current_time - self._hide_start_time
            if elapsed >= self._hide_delay:
                self._current_tooltip.hide()
                self._is_hiding = False

        # Update current tooltip
        if self._current_tooltip:
            self._current_tooltip.update(delta_time)

            # Clean up hidden tooltip
            if not self._current_tooltip.visible and not self._current_tooltip.is_animating:
                self._current_tooltip = None
            elif self._follow_cursor:
                self._current_tooltip.set_cursor_position(self._cursor_x, self._cursor_y)

    def update_cursor(self, x: float, y: float) -> None:
        """
        Update cursor position for cursor-following tooltips.

        Args:
            x: Cursor X position.
            y: Cursor Y position.
        """
        self._cursor_x = x
        self._cursor_y = y

    def cancel_pending(self) -> None:
        """Cancel any pending tooltip show request."""
        self._pending_target = None
        self._pending_content = None

    # =========================================================================
    # UTILITY
    # =========================================================================

    def is_over_tooltip(self, x: float, y: float) -> bool:
        """
        Check if a point is over the current tooltip.

        Args:
            x: Point X coordinate.
            y: Point Y coordinate.

        Returns:
            True if point is over the tooltip.
        """
        if not self._current_tooltip or not self._current_tooltip.visible:
            return False

        tx, ty, tw, th = self._current_tooltip.get_bounds()
        return tx <= x <= tx + tw and ty <= y <= ty + th

    def clear(self) -> None:
        """Clear all tooltips and pending requests."""
        self.hide_immediate()
        self._pending_target = None
        self._pending_content = None
