"""XR Slider control implementation.

Provides interactive slider controls for XR UI with support for:
- Horizontal and vertical orientations
- Ray/pointer dragging
- Poke/touch interaction
- Value snapping
- Haptic feedback at value changes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any


class XRSliderOrientation(Enum):
    """Slider orientation options."""
    HORIZONTAL = auto()
    VERTICAL = auto()


@dataclass(slots=True)
class XRSliderStyle:
    """Visual style configuration for XR slider."""
    track_color: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)
    fill_color: tuple[float, float, float, float] = (0.2, 0.5, 0.8, 1.0)
    handle_color: tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0)
    handle_hover_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    handle_pressed_color: tuple[float, float, float, float] = (0.8, 0.8, 1.0, 1.0)
    track_height: float = 0.01  # Meters
    handle_radius: float = 0.015  # Meters
    border_radius: float = 0.005  # Meters


@dataclass(slots=True)
class XRSlider:
    """XR interactive slider control.

    Attributes:
        label: Slider label text
        min_value: Minimum slider value
        max_value: Maximum slider value
        value: Current slider value
        step: Value step size (0 for continuous)
        width: Slider track width in meters
        height: Slider track height in meters
        orientation: Horizontal or vertical
        position: Local position relative to parent (x, y)
        is_hovered: Whether slider handle is being pointed at
        is_dragging: Whether slider is being dragged
        show_value: Whether to display current value
        value_format: Format string for value display
        haptic_on_change: Whether to trigger haptic on value changes
    """
    label: str = ""
    min_value: float = 0.0
    max_value: float = 1.0
    value: float = 0.5
    step: float = 0.0  # 0 = continuous
    width: float = 0.2  # Meters
    height: float = 0.03  # Meters
    orientation: XRSliderOrientation = XRSliderOrientation.HORIZONTAL
    position: tuple[float, float] = (0.0, 0.0)
    is_hovered: bool = False
    is_dragging: bool = False
    show_value: bool = True
    value_format: str = "{:.2f}"
    haptic_on_change: bool = True
    style: XRSliderStyle = field(default_factory=XRSliderStyle)
    _on_value_changed: Optional[Callable[[float], None]] = None
    _on_drag_start: Optional[Callable[[], None]] = None
    _on_drag_end: Optional[Callable[[float], None]] = None
    _interactor_id: Optional[int] = None
    _drag_start_value: float = 0.0
    _last_haptic_value: float = 0.0
    _parent: Any = None
    _enabled: bool = True

    def __post_init__(self):
        """Validate and initialize slider."""
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
        self.value = self._clamp_value(self.value)
        self._last_haptic_value = self.value

    @property
    def is_enabled(self) -> bool:
        """Check if slider is enabled."""
        return self._enabled

    @property
    def normalized_value(self) -> float:
        """Get value normalized to 0-1 range."""
        return (self.value - self.min_value) / (self.max_value - self.min_value)

    @property
    def formatted_value(self) -> str:
        """Get formatted value string for display."""
        return self.value_format.format(self.value)

    @property
    def handle_position(self) -> float:
        """Get handle position along track (0-1)."""
        return self.normalized_value

    @property
    def handle_world_offset(self) -> tuple[float, float]:
        """Get handle offset from slider origin in local space."""
        norm = self.normalized_value
        if self.orientation == XRSliderOrientation.HORIZONTAL:
            offset = (norm - 0.5) * self.width
            return (offset, 0.0)
        else:
            offset = (norm - 0.5) * self.height
            return (0.0, offset)

    def _clamp_value(self, value: float) -> float:
        """Clamp value to valid range and apply step."""
        value = max(self.min_value, min(self.max_value, value))

        if self.step > 0:
            # Snap to nearest step
            steps = round((value - self.min_value) / self.step)
            value = self.min_value + steps * self.step
            value = max(self.min_value, min(self.max_value, value))

        return value

    def enable(self) -> None:
        """Enable the slider."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the slider."""
        self._enabled = False
        self.is_hovered = False
        self.is_dragging = False

    def set_value(self, value: float, trigger_callback: bool = True) -> bool:
        """Set slider value.

        Args:
            value: New value
            trigger_callback: Whether to trigger value changed callback

        Returns:
            True if value changed
        """
        old_value = self.value
        self.value = self._clamp_value(value)

        if self.value != old_value:
            if trigger_callback and self._on_value_changed:
                self._on_value_changed(self.value)
            return True
        return False

    def set_range(self, min_value: float, max_value: float) -> None:
        """Update value range.

        Args:
            min_value: New minimum value
            max_value: New maximum value
        """
        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")

        self.min_value = min_value
        self.max_value = max_value
        self.value = self._clamp_value(self.value)

    def on_value_changed(self, callback: Callable[[float], None]) -> None:
        """Set value changed callback."""
        self._on_value_changed = callback

    def on_drag_start(self, callback: Callable[[], None]) -> None:
        """Set drag start callback."""
        self._on_drag_start = callback

    def on_drag_end(self, callback: Callable[[float], None]) -> None:
        """Set drag end callback."""
        self._on_drag_end = callback

    def hover_begin(self, interactor_id: int) -> None:
        """Handle hover begin on slider handle."""
        if not self.is_enabled:
            return
        self.is_hovered = True
        self._interactor_id = interactor_id

    def hover_end(self) -> None:
        """Handle hover end."""
        self.is_hovered = False
        if not self.is_dragging:
            self._interactor_id = None

    def drag_begin(self, interactor_id: int) -> None:
        """Begin dragging slider.

        Args:
            interactor_id: ID of the interacting entity
        """
        if not self.is_enabled:
            return

        self.is_dragging = True
        self._interactor_id = interactor_id
        self._drag_start_value = self.value
        self._last_haptic_value = self.value

        if self._on_drag_start:
            self._on_drag_start()

    def drag_update(
        self,
        normalized_position: float
    ) -> tuple[bool, bool]:
        """Update drag position.

        Args:
            normalized_position: Position along track (0-1)

        Returns:
            Tuple of (value_changed, should_haptic)
        """
        if not self.is_enabled or not self.is_dragging:
            return (False, False)

        # Convert normalized position to value
        new_value = self.min_value + normalized_position * (self.max_value - self.min_value)
        old_value = self.value
        self.value = self._clamp_value(new_value)

        value_changed = self.value != old_value

        # Determine if haptic feedback should trigger
        should_haptic = False
        if self.haptic_on_change and value_changed:
            if self.step > 0:
                # Haptic on each step
                should_haptic = True
            else:
                # Haptic on significant change (10% of range)
                haptic_threshold = (self.max_value - self.min_value) * 0.1
                if abs(self.value - self._last_haptic_value) >= haptic_threshold:
                    self._last_haptic_value = self.value
                    should_haptic = True

        if value_changed and self._on_value_changed:
            self._on_value_changed(self.value)

        return (value_changed, should_haptic)

    def drag_end(self) -> float:
        """End dragging slider.

        Returns:
            Final slider value
        """
        self.is_dragging = False
        self._interactor_id = None

        if self._on_drag_end:
            self._on_drag_end(self.value)

        return self.value

    def hit_test_handle(self, local_x: float, local_y: float) -> bool:
        """Test if point is on slider handle.

        Args:
            local_x: X coordinate in local space
            local_y: Y coordinate in local space

        Returns:
            True if point is on handle
        """
        hx, hy = self.handle_world_offset
        sx, sy = self.position

        # Handle center position
        handle_x = sx + hx
        handle_y = sy + hy

        # Distance to handle center
        dx = local_x - handle_x
        dy = local_y - handle_y
        distance_sq = dx * dx + dy * dy

        return distance_sq <= self.style.handle_radius ** 2

    def hit_test_track(self, local_x: float, local_y: float) -> Optional[float]:
        """Test if point is on slider track and return normalized position.

        Args:
            local_x: X coordinate in local space
            local_y: Y coordinate in local space

        Returns:
            Normalized position (0-1) if on track, None otherwise
        """
        sx, sy = self.position
        half_w = self.width / 2
        half_h = self.height / 2

        # Check if within track bounds
        if not (sx - half_w <= local_x <= sx + half_w and
                sy - half_h <= local_y <= sy + half_h):
            return None

        # Calculate normalized position
        if self.orientation == XRSliderOrientation.HORIZONTAL:
            return (local_x - (sx - half_w)) / self.width
        else:
            return (local_y - (sy - half_h)) / self.height

    def position_to_value(self, local_x: float, local_y: float) -> float:
        """Convert local position to slider value.

        Args:
            local_x: X coordinate in local space
            local_y: Y coordinate in local space

        Returns:
            Value at that position
        """
        sx, sy = self.position
        half_w = self.width / 2
        half_h = self.height / 2

        if self.orientation == XRSliderOrientation.HORIZONTAL:
            norm = (local_x - (sx - half_w)) / self.width
        else:
            norm = (local_y - (sy - half_h)) / self.height

        norm = max(0.0, min(1.0, norm))
        return self._clamp_value(self.min_value + norm * (self.max_value - self.min_value))


def xr_slider(
    min_value: float = 0.0,
    max_value: float = 1.0,
    step: float = 0.0,
    orientation: str = "horizontal",
    haptic: bool = True,
) -> Callable[[type], type]:
    """Decorator to mark a class as an XR slider component.

    Args:
        min_value: Minimum slider value
        max_value: Maximum slider value
        step: Value step size (0 for continuous)
        orientation: "horizontal" or "vertical"
        haptic: Whether to enable haptic feedback

    Returns:
        Decorated class with XR slider metadata

    Example:
        @xr_slider(min_value=0, max_value=100, step=10)
        class VolumeSlider:
            pass
    """
    def decorator(cls: type) -> type:
        # Validate parameters
        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")
        if step < 0:
            raise ValueError("Step must be non-negative")
        if orientation not in ("horizontal", "vertical"):
            raise ValueError("Orientation must be 'horizontal' or 'vertical'")

        # Map orientation string to enum
        orient_enum = (XRSliderOrientation.HORIZONTAL
                       if orientation == "horizontal"
                       else XRSliderOrientation.VERTICAL)

        # Apply metadata
        cls._xr_slider = True
        cls._slider_min = min_value
        cls._slider_max = max_value
        cls._slider_step = step
        cls._slider_orientation = orient_enum
        cls._slider_haptic = haptic

        # Trinity-style tags
        if not hasattr(cls, '_tags'):
            cls._tags = {}
        cls._tags['xr_slider'] = True
        cls._tags['slider_min'] = min_value
        cls._tags['slider_max'] = max_value
        cls._tags['slider_step'] = step

        # Applied decorators tracking
        if not hasattr(cls, '_applied_decorators'):
            cls._applied_decorators = set()
        cls._applied_decorators.add('xr_slider')

        # Registry tracking
        if not hasattr(cls, '_registries'):
            cls._registries = set()
        cls._registries.add('xr')

        return cls

    return decorator


class XRSliderGroup:
    """Group of related sliders.

    Useful for managing multiple sliders that control related values,
    like RGB color components or XYZ coordinates.
    """

    __slots__ = ('_sliders', '_labels', '_on_group_changed')

    def __init__(self):
        """Initialize slider group."""
        self._sliders: dict[str, XRSlider] = {}
        self._labels: list[str] = []
        self._on_group_changed: Optional[Callable[[dict[str, float]], None]] = None

    def add(self, key: str, slider: XRSlider) -> None:
        """Add a slider with a key identifier."""
        self._sliders[key] = slider
        self._labels.append(key)

        # Hook up callback to trigger group changed
        original_callback = slider._on_value_changed

        def combined_callback(value: float) -> None:
            if original_callback:
                original_callback(value)
            if self._on_group_changed:
                self._on_group_changed(self.values)

        slider._on_value_changed = combined_callback

    def remove(self, key: str) -> Optional[XRSlider]:
        """Remove a slider by key."""
        if key in self._sliders:
            slider = self._sliders.pop(key)
            self._labels.remove(key)
            return slider
        return None

    def get(self, key: str) -> Optional[XRSlider]:
        """Get slider by key."""
        return self._sliders.get(key)

    @property
    def values(self) -> dict[str, float]:
        """Get all slider values as dict."""
        return {key: slider.value for key, slider in self._sliders.items()}

    def set_values(self, values: dict[str, float]) -> None:
        """Set multiple slider values at once."""
        for key, value in values.items():
            if key in self._sliders:
                self._sliders[key].set_value(value, trigger_callback=False)

        if self._on_group_changed:
            self._on_group_changed(self.values)

    def on_group_changed(self, callback: Callable[[dict[str, float]], None]) -> None:
        """Set callback for when any slider in group changes."""
        self._on_group_changed = callback

    def enable_all(self) -> None:
        """Enable all sliders in group."""
        for slider in self._sliders.values():
            slider.enable()

    def disable_all(self) -> None:
        """Disable all sliders in group."""
        for slider in self._sliders.values():
            slider.disable()
