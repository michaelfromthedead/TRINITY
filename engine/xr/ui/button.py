"""XR Button implementation with haptic feedback.

Provides interactive buttons for XR UI with support for:
- Ray/pointer interaction
- Poke/touch interaction
- Visual feedback (hover, press states)
- Haptic feedback on press
- Press depth for physical-feel buttons
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any
import time


class XRButtonState(Enum):
    """States for XR button."""
    NORMAL = auto()  # Default state
    HOVERED = auto()  # Being pointed at
    PRESSED = auto()  # Currently pressed
    DISABLED = auto()  # Not interactable


@dataclass(slots=True)
class HapticFeedback:
    """Configuration for haptic feedback."""
    amplitude: float = 0.5  # 0.0 to 1.0
    duration_ms: int = 50  # Milliseconds
    frequency: float = 200.0  # Hz

    def __post_init__(self):
        """Validate feedback parameters."""
        self.amplitude = max(0.0, min(1.0, self.amplitude))
        self.duration_ms = max(0, self.duration_ms)
        self.frequency = max(0.0, self.frequency)


@dataclass(slots=True)
class XRButtonStyle:
    """Visual style configuration for XR button."""
    normal_color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0)
    hover_color: tuple[float, float, float, float] = (0.4, 0.4, 0.5, 1.0)
    pressed_color: tuple[float, float, float, float] = (0.2, 0.4, 0.8, 1.0)
    disabled_color: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.5)
    text_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    border_radius: float = 0.01  # Meters
    border_width: float = 0.002  # Meters
    font_size: float = 0.03  # Meters


@dataclass(slots=True)
class XRButton:
    """XR interactive button with haptic feedback.

    Attributes:
        label: Button text label
        width: Button width in meters
        height: Button height in meters
        position: Local position relative to parent panel (x, y)
        state: Current button state
        is_hovered: Whether button is being pointed at
        is_pressed: Whether button is currently pressed
        press_depth: How far button is pressed (for poke interaction)
        haptic_on_press: Whether to trigger haptic feedback on press
        haptic_config: Haptic feedback configuration
        style: Visual style configuration
        on_click: Callback when button is clicked
        on_hover_enter: Callback when hover begins
        on_hover_exit: Callback when hover ends
    """
    label: str = ""
    width: float = 0.15  # Meters
    height: float = 0.05  # Meters
    position: tuple[float, float] = (0.0, 0.0)
    state: XRButtonState = XRButtonState.NORMAL
    is_hovered: bool = False
    is_pressed: bool = False
    press_depth: float = 0.0
    max_press_depth: float = 0.02  # Meters - how far button can be pushed
    press_threshold: float = 0.015  # Depth required to register press
    haptic_on_press: bool = True
    haptic_config: HapticFeedback = field(default_factory=HapticFeedback)
    style: XRButtonStyle = field(default_factory=XRButtonStyle)
    _on_click: Optional[Callable[[], None]] = None
    _on_hover_enter: Optional[Callable[[], None]] = None
    _on_hover_exit: Optional[Callable[[], None]] = None
    _interactor_id: Optional[int] = None
    _press_start_time: Optional[float] = None
    _last_haptic_time: Optional[float] = None
    _parent: Any = None
    _enabled: bool = True

    @property
    def is_enabled(self) -> bool:
        """Check if button is enabled."""
        return self._enabled and self.state != XRButtonState.DISABLED

    @property
    def current_color(self) -> tuple[float, float, float, float]:
        """Get current display color based on state."""
        if self.state == XRButtonState.DISABLED:
            return self.style.disabled_color
        elif self.is_pressed:
            return self.style.pressed_color
        elif self.is_hovered:
            return self.style.hover_color
        return self.style.normal_color

    @property
    def visual_depth(self) -> float:
        """Get visual depth offset for 3D button effect."""
        if self.is_pressed:
            return -min(self.press_depth, self.max_press_depth)
        return 0.0

    def enable(self) -> None:
        """Enable the button."""
        self._enabled = True
        if self.state == XRButtonState.DISABLED:
            self.state = XRButtonState.NORMAL

    def disable(self) -> None:
        """Disable the button."""
        self._enabled = False
        self.state = XRButtonState.DISABLED
        self.is_hovered = False
        self.is_pressed = False
        self.press_depth = 0.0

    def set_label(self, label: str) -> None:
        """Update button label."""
        self.label = label

    def on_click(self, callback: Callable[[], None]) -> None:
        """Set click callback."""
        self._on_click = callback

    def on_hover_enter(self, callback: Callable[[], None]) -> None:
        """Set hover enter callback."""
        self._on_hover_enter = callback

    def on_hover_exit(self, callback: Callable[[], None]) -> None:
        """Set hover exit callback."""
        self._on_hover_exit = callback

    def hover_begin(self, interactor_id: int) -> Optional[HapticFeedback]:
        """Handle hover begin.

        Args:
            interactor_id: ID of the interacting entity

        Returns:
            Haptic feedback to trigger (if any)
        """
        if not self.is_enabled:
            return None

        was_hovered = self.is_hovered
        self.is_hovered = True
        self._interactor_id = interactor_id
        self.state = XRButtonState.HOVERED

        if not was_hovered and self._on_hover_enter:
            self._on_hover_enter()

        # Light haptic on hover
        if self.haptic_on_press:
            return HapticFeedback(
                amplitude=0.1,
                duration_ms=10,
                frequency=150.0,
            )
        return None

    def hover_end(self) -> None:
        """Handle hover end."""
        was_hovered = self.is_hovered
        self.is_hovered = False
        self.is_pressed = False
        self.press_depth = 0.0
        self._interactor_id = None

        if self.is_enabled:
            self.state = XRButtonState.NORMAL

        if was_hovered and self._on_hover_exit:
            self._on_hover_exit()

    def press_begin(self, interactor_id: int) -> Optional[HapticFeedback]:
        """Handle press begin (trigger pulled or poke started).

        Args:
            interactor_id: ID of the interacting entity

        Returns:
            Haptic feedback to trigger (if any)
        """
        if not self.is_enabled:
            return None

        self.is_pressed = True
        self._interactor_id = interactor_id
        self._press_start_time = time.time()
        self.state = XRButtonState.PRESSED

        if self.haptic_on_press:
            self._last_haptic_time = time.time()
            return self.haptic_config
        return None

    def press_update(self, depth: float) -> Optional[HapticFeedback]:
        """Update press depth (for poke interaction).

        Args:
            depth: Current press depth in meters

        Returns:
            Haptic feedback to trigger (if any)
        """
        if not self.is_enabled:
            return None

        old_depth = self.press_depth
        self.press_depth = max(0.0, min(depth, self.max_press_depth))

        # Check if crossed press threshold
        if old_depth < self.press_threshold <= self.press_depth:
            self.is_pressed = True
            self.state = XRButtonState.PRESSED
            if self.haptic_on_press:
                return self.haptic_config

        return None

    def press_end(self) -> tuple[bool, Optional[HapticFeedback]]:
        """Handle press end.

        Returns:
            Tuple of (was_clicked, haptic_feedback)
        """
        was_pressed = self.is_pressed
        was_deep_enough = self.press_depth >= self.press_threshold

        self.is_pressed = False
        self.press_depth = 0.0
        self._press_start_time = None

        if self.is_enabled:
            self.state = XRButtonState.HOVERED if self.is_hovered else XRButtonState.NORMAL

        # Trigger click if button was properly pressed
        clicked = was_pressed or was_deep_enough
        if clicked and self.is_enabled:
            if self._on_click:
                self._on_click()

            # Click feedback
            if self.haptic_on_press:
                return (True, HapticFeedback(
                    amplitude=0.3,
                    duration_ms=30,
                    frequency=180.0,
                ))

        return (clicked, None)

    def hit_test(self, local_x: float, local_y: float) -> bool:
        """Test if a point is within button bounds.

        Args:
            local_x: X coordinate in panel space
            local_y: Y coordinate in panel space

        Returns:
            True if point is within button
        """
        bx, by = self.position
        half_w = self.width / 2
        half_h = self.height / 2

        return (bx - half_w <= local_x <= bx + half_w and
                by - half_h <= local_y <= by + half_h)


def xr_button(
    label: str = "",
    width: float = 0.15,
    height: float = 0.05,
    haptic: bool = True,
    press_depth: float = 0.02,
) -> Callable[[type], type]:
    """Decorator to mark a class as an XR button component.

    Args:
        label: Default button label
        width: Button width in meters
        height: Button height in meters
        haptic: Whether to enable haptic feedback
        press_depth: Maximum press depth in meters

    Returns:
        Decorated class with XR button metadata

    Example:
        @xr_button(label="Start", haptic=True)
        class StartButton:
            pass
    """
    def decorator(cls: type) -> type:
        # Validate parameters
        if width <= 0:
            raise ValueError("Width must be positive")
        if height <= 0:
            raise ValueError("Height must be positive")
        if press_depth < 0:
            raise ValueError("Press depth must be non-negative")

        # Apply metadata
        cls._xr_button = True
        cls._button_label = label
        cls._button_width = width
        cls._button_height = height
        cls._button_haptic = haptic
        cls._button_press_depth = press_depth

        # Trinity-style tags
        if not hasattr(cls, '_tags'):
            cls._tags = {}
        cls._tags['xr_button'] = True
        cls._tags['button_label'] = label
        cls._tags['button_haptic'] = haptic

        # Applied decorators tracking
        if not hasattr(cls, '_applied_decorators'):
            cls._applied_decorators = set()
        cls._applied_decorators.add('xr_button')

        # Registry tracking
        if not hasattr(cls, '_registries'):
            cls._registries = set()
        cls._registries.add('xr')

        return cls

    return decorator


class XRButtonGroup:
    """Group of related XR buttons.

    Manages a collection of buttons with shared behavior like
    radio selection or toggle groups.
    """

    __slots__ = ('_buttons', '_selection_mode', '_selected_index')

    def __init__(self, selection_mode: str = "none"):
        """Initialize button group.

        Args:
            selection_mode: "none", "single" (radio), or "multiple" (checkboxes)
        """
        self._buttons: list[XRButton] = []
        self._selection_mode = selection_mode
        self._selected_index: Optional[int] = None

    def add(self, button: XRButton) -> None:
        """Add a button to the group."""
        self._buttons.append(button)

    def remove(self, button: XRButton) -> None:
        """Remove a button from the group."""
        if button in self._buttons:
            idx = self._buttons.index(button)
            self._buttons.remove(button)
            if self._selected_index == idx:
                self._selected_index = None
            elif self._selected_index is not None and self._selected_index > idx:
                self._selected_index -= 1

    def select(self, index: int) -> None:
        """Select button at index (for single selection mode)."""
        if self._selection_mode != "single":
            return
        if 0 <= index < len(self._buttons):
            self._selected_index = index

    @property
    def selected(self) -> Optional[XRButton]:
        """Get currently selected button."""
        if self._selected_index is not None:
            return self._buttons[self._selected_index]
        return None

    @property
    def buttons(self) -> list[XRButton]:
        """Get all buttons in group."""
        return self._buttons.copy()

    def enable_all(self) -> None:
        """Enable all buttons in group."""
        for button in self._buttons:
            button.enable()

    def disable_all(self) -> None:
        """Disable all buttons in group."""
        for button in self._buttons:
            button.disable()
