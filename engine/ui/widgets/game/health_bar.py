"""
Health/Resource bar widget with smooth animations and damage preview.

Provides a flexible health bar widget supporting:
- Current/max value display
- Smooth animated value changes
- Damage preview (pending damage visualization)
- Segmented display option
- Shield/armor overlay
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


class ResourceType(Enum):
    """Types of resources a bar can represent."""
    HEALTH = auto()
    MANA = auto()
    STAMINA = auto()
    ENERGY = auto()
    SHIELD = auto()
    ARMOR = auto()
    EXPERIENCE = auto()
    CUSTOM = auto()


@dataclass(slots=True)
class HealthBarStyle:
    """Style configuration for health bar appearance."""
    # Colors (RGBA as tuples or hex strings)
    fill_color: str = "#22c55e"  # Green
    background_color: str = "#1f2937"  # Dark gray
    border_color: str = "#374151"  # Gray
    damage_preview_color: str = "#ef4444"  # Red
    heal_preview_color: str = "#10b981"  # Teal
    shield_color: str = "#3b82f6"  # Blue
    segment_border_color: str = "#111827"  # Near black

    # Dimensions
    border_width: float = 2.0
    corner_radius: float = 4.0
    segment_gap: float = 2.0

    # Animation
    animation_duration: float = 0.3  # seconds
    damage_preview_duration: float = 1.0  # seconds
    pulse_on_low: bool = True
    low_threshold: float = 0.25  # Pulse when below 25%


@dataclass(slots=True)
class HealthBarSegment:
    """A single segment in a segmented health bar."""
    index: int
    start_percent: float  # 0.0-1.0
    end_percent: float  # 0.0-1.0
    is_filled: bool = True
    is_damaged: bool = False  # Currently taking damage
    custom_color: Optional[str] = None


class HealthBar:
    """Health/resource bar widget with animation support.

    Features:
    - Smooth animated transitions between values
    - Damage preview showing pending damage before animation
    - Segmented display for discrete health chunks
    - Shield/armor overlay on top of base health
    - Configurable styling and animation
    """

    __slots__ = (
        '_x', '_y', '_width', '_height',
        '_current_value', '_max_value', '_display_value',
        '_pending_damage', '_pending_heal',
        '_shield_value', '_shield_max',
        '_armor_value', '_armor_max',
        '_style', '_resource_type',
        '_segments', '_segment_count',
        '_animation_progress', '_animation_start_value', '_animation_target_value',
        '_damage_preview_timer',
        '_is_animating', '_is_visible', '_is_pulsing',
        '_on_value_changed', '_on_depleted', '_on_full',
        '_parent', '_children', '_id',
    )

    _next_id: int = 0

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 200.0,
        height: float = 20.0,
        max_value: float = 100.0,
        current_value: Optional[float] = None,
        style: Optional[HealthBarStyle] = None,
        resource_type: ResourceType = ResourceType.HEALTH,
        segment_count: int = 0,
    ):
        """Initialize the health bar.

        Args:
            x: X position
            y: Y position
            width: Bar width in pixels
            height: Bar height in pixels
            max_value: Maximum value
            current_value: Initial current value (defaults to max)
            style: Visual style configuration
            resource_type: Type of resource this bar represents
            segment_count: Number of segments (0 for smooth bar)
        """
        self._id = HealthBar._next_id
        HealthBar._next_id += 1

        self._x = x
        self._y = y
        self._width = max(1.0, width)
        self._height = max(1.0, height)

        self._max_value = max(0.0, max_value)
        self._current_value = current_value if current_value is not None else max_value
        self._current_value = self._clamp_value(self._current_value)
        self._display_value = self._current_value

        self._pending_damage = 0.0
        self._pending_heal = 0.0

        self._shield_value = 0.0
        self._shield_max = 0.0
        self._armor_value = 0.0
        self._armor_max = 0.0

        self._style = style or HealthBarStyle()
        self._resource_type = resource_type

        self._segment_count = max(0, segment_count)
        self._segments: list[HealthBarSegment] = []
        self._rebuild_segments()

        self._animation_progress = 1.0
        self._animation_start_value = self._current_value
        self._animation_target_value = self._current_value

        self._damage_preview_timer = 0.0

        self._is_animating = False
        self._is_visible = True
        self._is_pulsing = False

        self._on_value_changed: Optional[Callable[[float, float], None]] = None
        self._on_depleted: Optional[Callable[[], None]] = None
        self._on_full: Optional[Callable[[], None]] = None

        self._parent = None
        self._children: list = []

    # Properties
    @property
    def id(self) -> int:
        """Get the widget ID."""
        return self._id

    @property
    def x(self) -> float:
        """Get X position."""
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        """Set X position."""
        self._x = value

    @property
    def y(self) -> float:
        """Get Y position."""
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        """Set Y position."""
        self._y = value

    @property
    def width(self) -> float:
        """Get width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set width."""
        self._width = max(1.0, value)
        self._rebuild_segments()

    @property
    def height(self) -> float:
        """Get height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set height."""
        self._height = max(1.0, value)

    @property
    def current_value(self) -> float:
        """Get current value."""
        return self._current_value

    @property
    def max_value(self) -> float:
        """Get maximum value."""
        return self._max_value

    @max_value.setter
    def max_value(self, value: float) -> None:
        """Set maximum value."""
        old_max = self._max_value
        self._max_value = max(0.0, value)

        # Clamp current value if needed
        if self._current_value > self._max_value:
            self.set_value(self._max_value)

        self._rebuild_segments()

    @property
    def display_value(self) -> float:
        """Get the currently displayed (animated) value."""
        return self._display_value

    @property
    def fill_percent(self) -> float:
        """Get fill percentage (0.0-1.0)."""
        if self._max_value <= 0:
            return 0.0
        return self._display_value / self._max_value

    @property
    def actual_percent(self) -> float:
        """Get actual (non-animated) fill percentage."""
        if self._max_value <= 0:
            return 0.0
        return self._current_value / self._max_value

    @property
    def is_empty(self) -> bool:
        """Check if bar is empty."""
        return self._current_value <= 0

    @property
    def is_full(self) -> bool:
        """Check if bar is full."""
        return self._current_value >= self._max_value

    @property
    def is_low(self) -> bool:
        """Check if bar is below low threshold."""
        return self.actual_percent < self._style.low_threshold

    @property
    def is_animating(self) -> bool:
        """Check if bar is currently animating."""
        return self._is_animating

    @property
    def is_visible(self) -> bool:
        """Check if bar is visible."""
        return self._is_visible

    @is_visible.setter
    def is_visible(self, value: bool) -> None:
        """Set visibility."""
        self._is_visible = value

    @property
    def style(self) -> HealthBarStyle:
        """Get style configuration."""
        return self._style

    @style.setter
    def style(self, value: HealthBarStyle) -> None:
        """Set style configuration."""
        self._style = value

    @property
    def resource_type(self) -> ResourceType:
        """Get resource type."""
        return self._resource_type

    @property
    def segment_count(self) -> int:
        """Get segment count."""
        return self._segment_count

    @segment_count.setter
    def segment_count(self, value: int) -> None:
        """Set segment count."""
        self._segment_count = max(0, value)
        self._rebuild_segments()

    @property
    def segments(self) -> list[HealthBarSegment]:
        """Get segment list."""
        return self._segments.copy()

    @property
    def pending_damage(self) -> float:
        """Get pending damage amount."""
        return self._pending_damage

    @property
    def shield_value(self) -> float:
        """Get current shield value."""
        return self._shield_value

    @property
    def shield_max(self) -> float:
        """Get maximum shield value."""
        return self._shield_max

    @property
    def shield_percent(self) -> float:
        """Get shield fill percentage."""
        if self._shield_max <= 0:
            return 0.0
        return self._shield_value / self._shield_max

    @property
    def armor_value(self) -> float:
        """Get current armor value."""
        return self._armor_value

    @property
    def armor_max(self) -> float:
        """Get maximum armor value."""
        return self._armor_max

    # Value manipulation
    def set_value(self, value: float, animate: bool = True) -> None:
        """Set current value.

        Args:
            value: New value
            animate: Whether to animate the transition
        """
        old_value = self._current_value
        self._current_value = self._clamp_value(value)

        if animate and self._style.animation_duration > 0:
            self._start_animation(old_value, self._current_value)
        else:
            self._display_value = self._current_value
            self._animation_progress = 1.0
            self._is_animating = False

        self._update_segments()
        self._notify_value_changed(old_value, self._current_value)

    def apply_damage(self, amount: float, show_preview: bool = True) -> float:
        """Apply damage to the bar.

        Args:
            amount: Damage amount
            show_preview: Whether to show damage preview

        Returns:
            Actual damage applied (after shields/armor)
        """
        if amount <= 0:
            return 0.0

        actual_damage = amount

        # Apply to shield first
        if self._shield_value > 0:
            shield_damage = min(self._shield_value, actual_damage)
            self._shield_value -= shield_damage
            actual_damage -= shield_damage

        # Apply to armor
        if self._armor_value > 0 and actual_damage > 0:
            armor_damage = min(self._armor_value, actual_damage)
            self._armor_value -= armor_damage
            actual_damage -= armor_damage

        if actual_damage > 0:
            if show_preview:
                self._pending_damage = actual_damage
                self._damage_preview_timer = self._style.damage_preview_duration

            self.set_value(self._current_value - actual_damage)

        return amount - actual_damage + actual_damage  # Total absorbed + applied

    def apply_heal(self, amount: float, show_preview: bool = True) -> float:
        """Apply healing to the bar.

        Args:
            amount: Heal amount
            show_preview: Whether to show heal preview

        Returns:
            Actual healing applied
        """
        if amount <= 0:
            return 0.0

        old_value = self._current_value
        new_value = min(self._max_value, self._current_value + amount)
        actual_heal = new_value - old_value

        if actual_heal > 0:
            if show_preview:
                self._pending_heal = actual_heal

            self.set_value(new_value)

        return actual_heal

    def set_shield(self, current: float, maximum: Optional[float] = None) -> None:
        """Set shield values.

        Args:
            current: Current shield value
            maximum: Maximum shield value (defaults to current if not set)
        """
        if maximum is not None:
            self._shield_max = max(0.0, maximum)
        elif self._shield_max <= 0:
            self._shield_max = max(0.0, current)

        self._shield_value = self._clamp_value(current, 0.0, self._shield_max)

    def set_armor(self, current: float, maximum: Optional[float] = None) -> None:
        """Set armor values.

        Args:
            current: Current armor value
            maximum: Maximum armor value (defaults to current if not set)
        """
        if maximum is not None:
            self._armor_max = max(0.0, maximum)
        elif self._armor_max <= 0:
            self._armor_max = max(0.0, current)

        self._armor_value = self._clamp_value(current, 0.0, self._armor_max)

    def fill(self, animate: bool = True) -> None:
        """Fill the bar to maximum."""
        self.set_value(self._max_value, animate=animate)

    def deplete(self, animate: bool = True) -> None:
        """Deplete the bar to zero."""
        self.set_value(0.0, animate=animate)

    def clear_pending(self) -> None:
        """Clear pending damage/heal previews."""
        self._pending_damage = 0.0
        self._pending_heal = 0.0
        self._damage_preview_timer = 0.0

    # Animation
    def update(self, delta_time: float) -> None:
        """Update animation state.

        Args:
            delta_time: Time since last update in seconds
        """
        # Update animation
        if self._is_animating:
            if self._style.animation_duration > 0:
                self._animation_progress += delta_time / self._style.animation_duration
            else:
                self._animation_progress = 1.0

            if self._animation_progress >= 1.0:
                self._animation_progress = 1.0
                self._is_animating = False
                self._display_value = self._animation_target_value
            else:
                # Smooth easing
                t = self._ease_out_cubic(self._animation_progress)
                self._display_value = self._lerp(
                    self._animation_start_value,
                    self._animation_target_value,
                    t
                )

        # Update damage preview timer
        if self._damage_preview_timer > 0:
            self._damage_preview_timer -= delta_time
            if self._damage_preview_timer <= 0:
                self._pending_damage = 0.0
                self._damage_preview_timer = 0.0

        # Update pulse state
        self._is_pulsing = self._style.pulse_on_low and self.is_low and not self.is_empty

    def skip_animation(self) -> None:
        """Skip current animation to end state."""
        if self._is_animating:
            self._display_value = self._animation_target_value
            self._animation_progress = 1.0
            self._is_animating = False

    # Callbacks
    def on_value_changed(self, callback: Callable[[float, float], None]) -> None:
        """Set callback for value changes.

        Args:
            callback: Function(old_value, new_value)
        """
        self._on_value_changed = callback

    def on_depleted(self, callback: Callable[[], None]) -> None:
        """Set callback for when bar reaches zero.

        Args:
            callback: Function()
        """
        self._on_depleted = callback

    def on_full(self, callback: Callable[[], None]) -> None:
        """Set callback for when bar reaches maximum.

        Args:
            callback: Function()
        """
        self._on_full = callback

    # Rendering helpers
    def get_fill_rect(self) -> tuple[float, float, float, float]:
        """Get the filled portion rectangle.

        Returns:
            (x, y, width, height) of filled area
        """
        fill_width = self._width * self.fill_percent
        return (self._x, self._y, fill_width, self._height)

    def get_damage_preview_rect(self) -> Optional[tuple[float, float, float, float]]:
        """Get the damage preview rectangle.

        Returns:
            (x, y, width, height) of damage preview, or None
        """
        if self._pending_damage <= 0 or self._max_value <= 0:
            return None

        # Preview starts at new value and extends to old value
        new_percent = self.fill_percent
        damage_percent = min(self._pending_damage / self._max_value, 1.0 - new_percent)

        if damage_percent <= 0:
            return None

        x = self._x + (self._width * new_percent)
        width = self._width * damage_percent

        return (x, self._y, width, self._height)

    def get_shield_rect(self) -> Optional[tuple[float, float, float, float]]:
        """Get the shield overlay rectangle.

        Returns:
            (x, y, width, height) of shield area, or None
        """
        if self._shield_value <= 0 or self._shield_max <= 0:
            return None

        # Shield appears on top of health
        fill_width = self._width * self.fill_percent
        shield_width = fill_width * self.shield_percent

        return (self._x, self._y, shield_width, self._height)

    def get_segment_rects(self) -> list[tuple[float, float, float, float, bool]]:
        """Get segment rectangles.

        Returns:
            List of (x, y, width, height, is_filled) tuples
        """
        rects = []
        for segment in self._segments:
            seg_x = self._x + (self._width * segment.start_percent)
            seg_width = self._width * (segment.end_percent - segment.start_percent)
            seg_width -= self._style.segment_gap  # Account for gap
            rects.append((seg_x, self._y, seg_width, self._height, segment.is_filled))
        return rects

    # Private methods
    def _clamp_value(
        self,
        value: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> float:
        """Clamp value to valid range."""
        if min_val is None:
            min_val = 0.0
        if max_val is None:
            max_val = self._max_value
        return max(min_val, min(max_val, value))

    def _start_animation(self, start: float, target: float) -> None:
        """Start value animation."""
        self._animation_start_value = start
        self._animation_target_value = target
        self._animation_progress = 0.0
        self._is_animating = True

    def _rebuild_segments(self) -> None:
        """Rebuild segment list."""
        self._segments.clear()

        if self._segment_count <= 0:
            return

        segment_size = 1.0 / self._segment_count
        for i in range(self._segment_count):
            start = i * segment_size
            end = (i + 1) * segment_size
            self._segments.append(HealthBarSegment(
                index=i,
                start_percent=start,
                end_percent=end,
                is_filled=self.actual_percent >= end,
            ))

        self._update_segments()

    def _update_segments(self) -> None:
        """Update segment fill states."""
        for segment in self._segments:
            segment.is_filled = self.actual_percent >= segment.end_percent
            # Partial fill for last segment
            if segment.start_percent < self.actual_percent < segment.end_percent:
                segment.is_filled = True  # Partially filled

    def _notify_value_changed(self, old_value: float, new_value: float) -> None:
        """Notify callbacks of value change."""
        if self._on_value_changed:
            self._on_value_changed(old_value, new_value)

        if new_value <= 0 and old_value > 0 and self._on_depleted:
            self._on_depleted()
        elif new_value >= self._max_value and old_value < self._max_value and self._on_full:
            self._on_full()

    @staticmethod
    def _ease_out_cubic(t: float) -> float:
        """Cubic ease-out interpolation."""
        return 1.0 - pow(1.0 - t, 3)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + (b - a) * t

    def __repr__(self) -> str:
        return (
            f"HealthBar(id={self._id}, "
            f"value={self._current_value}/{self._max_value}, "
            f"type={self._resource_type.name})"
        )
