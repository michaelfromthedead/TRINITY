"""
XR Controller input component.

Provides 6-DOF motion controller tracking with buttons, axes, touch sensing,
and haptic feedback integration. Supports both grip and aim poses for
different interaction models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from trinity.descriptors import (
    TrackedDescriptor,
    RangeDescriptor,
    TransientDescriptor,
    ImmutableDescriptor,
    clear_dirty,
)

if TYPE_CHECKING:
    from engine.xr.input.haptics import HapticEffect


# Type aliases
Vec3Tuple = Tuple[float, float, float]
QuatTuple = Tuple[float, float, float, float]
Vec2Tuple = Tuple[float, float]


class XRHand(Enum):
    """Controller hand assignment."""
    LEFT = auto()
    RIGHT = auto()


class XRButton(Enum):
    """XR controller button identifiers."""
    TRIGGER = auto()       # Index finger trigger
    GRIP = auto()          # Side grip/squeeze
    PRIMARY = auto()       # A or X button
    SECONDARY = auto()     # B or Y button
    THUMBSTICK = auto()    # Thumbstick click
    MENU = auto()          # Menu/system button
    THUMBREST = auto()     # Thumbrest touch area


class XRControllerType(Enum):
    """Types of XR controllers."""
    MOTION = auto()        # Standard motion controller
    GAMEPAD = auto()       # Gamepad-style controller
    HAND = auto()          # Hand tracking (virtual controller)
    CUSTOM = auto()        # Custom/third-party controller


@dataclass(slots=True)
class ButtonState:
    """State of a single button."""
    pressed: bool = False
    touched: bool = False
    value: float = 0.0  # Analog value for triggers/grips


@dataclass(slots=True)
class ControllerCapabilities:
    """Capability flags for a controller."""
    has_trigger: bool = True
    has_grip: bool = True
    has_primary_button: bool = True
    has_secondary_button: bool = True
    has_thumbstick: bool = True
    has_menu_button: bool = True
    has_touch_sensing: bool = True
    has_haptics: bool = True
    has_finger_tracking: bool = False


class XRController:
    """
    XR motion controller component.

    Provides tracking, input, and haptic feedback for VR/AR controllers.
    Supports both grip pose (for held objects) and aim pose (for pointing).

    Features:
    - 6-DOF pose tracking with prediction
    - Analog trigger and grip with digital thresholds
    - Thumbstick with deadzone
    - Touch sensing for capacitive buttons
    - Haptic feedback integration
    - Button state tracking (pressed/just_pressed/just_released)

    Attributes:
        hand: Left or right hand
        grip_position: Grip pose position
        grip_orientation: Grip pose orientation
        aim_position: Aim/pointer pose position
        aim_orientation: Aim/pointer pose orientation
        trigger: Trigger analog value (0-1)
        grip: Grip analog value (0-1)
        thumbstick: Thumbstick position (-1 to 1)
    """

    # Pose descriptors (predicted + tracked)
    grip_position = TrackedDescriptor(
        field_type=tuple,
        use_bitmask=True,
        field_offset=0,
    )
    grip_orientation = TrackedDescriptor(
        field_type=tuple,
        use_bitmask=True,
        field_offset=1,
    )
    aim_position = TrackedDescriptor(
        field_type=tuple,
        use_bitmask=True,
        field_offset=2,
    )
    aim_orientation = TrackedDescriptor(
        field_type=tuple,
        use_bitmask=True,
        field_offset=3,
    )

    # Velocity descriptors
    linear_velocity = TrackedDescriptor(
        field_type=tuple,
        use_bitmask=True,
        field_offset=4,
    )
    angular_velocity = TrackedDescriptor(
        field_type=tuple,
        use_bitmask=True,
        field_offset=5,
    )

    # Analog input descriptors (range-clamped)
    trigger = TrackedDescriptor(
        field_type=float,
        use_bitmask=True,
        field_offset=6,
    )
    grip_value = TrackedDescriptor(
        field_type=float,
        use_bitmask=True,
        field_offset=7,
    )
    thumbstick_x = TrackedDescriptor(
        field_type=float,
        use_bitmask=True,
        field_offset=8,
    )
    thumbstick_y = TrackedDescriptor(
        field_type=float,
        use_bitmask=True,
        field_offset=9,
    )

    # Tracking state
    is_tracked = TrackedDescriptor(
        field_type=bool,
        use_bitmask=True,
        field_offset=10,
    )

    __slots__ = (
        "__dict__",
        "__weakref__",
        "_hand",
        "_controller_type",
        "_capabilities",
        "_device_id",
        "_current_buttons",
        "_previous_buttons",
        "_pressed_buttons",
        "_released_buttons",
        "_current_touch",
        "_previous_touch",
        "_just_touched",
        "_just_untouched",
        "_trigger_threshold",
        "_grip_threshold",
        "_thumbstick_deadzone",
        "_on_button_pressed",
        "_on_button_released",
        "_on_trigger_changed",
        "_on_grip_changed",
        "_pending_haptics",
        "_entity_id",
    )

    # Thresholds
    DEFAULT_TRIGGER_THRESHOLD: float = 0.1
    DEFAULT_GRIP_THRESHOLD: float = 0.1
    DEFAULT_THUMBSTICK_DEADZONE: float = 0.15

    def __init__(
        self,
        hand: XRHand = XRHand.LEFT,
        controller_type: XRControllerType = XRControllerType.MOTION,
        capabilities: Optional[ControllerCapabilities] = None,
        device_id: str = "",
        entity_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the XR controller.

        Args:
            hand: Left or right hand
            controller_type: Type of controller
            capabilities: Controller capability flags
            device_id: Unique device identifier
            entity_id: Optional entity ID for ECS
        """
        self._hand = hand
        self._controller_type = controller_type
        self._capabilities = capabilities or ControllerCapabilities()
        self._device_id = device_id
        self._entity_id = entity_id

        # Button state tracking
        self._current_buttons: Set[XRButton] = set()
        self._previous_buttons: Set[XRButton] = set()
        self._pressed_buttons: Set[XRButton] = set()
        self._released_buttons: Set[XRButton] = set()

        # Touch state tracking
        self._current_touch: Set[XRButton] = set()
        self._previous_touch: Set[XRButton] = set()
        self._just_touched: Set[XRButton] = set()
        self._just_untouched: Set[XRButton] = set()

        # Thresholds
        self._trigger_threshold = self.DEFAULT_TRIGGER_THRESHOLD
        self._grip_threshold = self.DEFAULT_GRIP_THRESHOLD
        self._thumbstick_deadzone = self.DEFAULT_THUMBSTICK_DEADZONE

        # Callbacks
        self._on_button_pressed: List[Callable[[XRButton], None]] = []
        self._on_button_released: List[Callable[[XRButton], None]] = []
        self._on_trigger_changed: List[Callable[[float], None]] = []
        self._on_grip_changed: List[Callable[[float], None]] = []

        # Haptics
        self._pending_haptics: List[Any] = []

        # Initialize tracked fields
        self.grip_position = (0.0, 0.0, 0.0)
        self.grip_orientation = (0.0, 0.0, 0.0, 1.0)
        self.aim_position = (0.0, 0.0, 0.0)
        self.aim_orientation = (0.0, 0.0, 0.0, 1.0)
        self.linear_velocity = (0.0, 0.0, 0.0)
        self.angular_velocity = (0.0, 0.0, 0.0)
        self.trigger = 0.0
        self.grip_value = 0.0
        self.thumbstick_x = 0.0
        self.thumbstick_y = 0.0
        self.is_tracked = False

        clear_dirty(self)

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def hand(self) -> XRHand:
        """Get the hand assignment."""
        return self._hand

    @property
    def controller_type(self) -> XRControllerType:
        """Get the controller type."""
        return self._controller_type

    @property
    def capabilities(self) -> ControllerCapabilities:
        """Get controller capabilities."""
        return self._capabilities

    @property
    def device_id(self) -> str:
        """Get the device identifier."""
        return self._device_id

    @property
    def thumbstick(self) -> Vec2Tuple:
        """Get thumbstick position with deadzone applied."""
        x = self._apply_deadzone(self.thumbstick_x)
        y = self._apply_deadzone(self.thumbstick_y)
        return (x, y)

    @property
    def thumbstick_raw(self) -> Vec2Tuple:
        """Get raw thumbstick position without deadzone."""
        return (self.thumbstick_x, self.thumbstick_y)

    @property
    def trigger_pressed(self) -> bool:
        """Check if trigger is pressed past threshold."""
        return self.trigger >= self._trigger_threshold

    @property
    def grip_pressed(self) -> bool:
        """Check if grip is pressed past threshold."""
        return self.grip_value >= self._grip_threshold

    # =========================================================================
    # POSE UPDATE
    # =========================================================================

    def update_pose(
        self,
        grip_position: Vec3Tuple,
        grip_orientation: QuatTuple,
        aim_position: Optional[Vec3Tuple] = None,
        aim_orientation: Optional[QuatTuple] = None,
        linear_velocity: Optional[Vec3Tuple] = None,
        angular_velocity: Optional[Vec3Tuple] = None,
        is_tracked: bool = True,
    ) -> None:
        """
        Update controller pose from tracking system.

        Args:
            grip_position: Grip pose position
            grip_orientation: Grip pose orientation
            aim_position: Aim pose position (defaults to grip)
            aim_orientation: Aim pose orientation (defaults to grip)
            linear_velocity: Linear velocity
            angular_velocity: Angular velocity
            is_tracked: Whether controller is being tracked
        """
        self.grip_position = grip_position
        self.grip_orientation = grip_orientation
        self.aim_position = aim_position if aim_position is not None else grip_position
        self.aim_orientation = aim_orientation if aim_orientation is not None else grip_orientation

        if linear_velocity is not None:
            self.linear_velocity = linear_velocity
        if angular_velocity is not None:
            self.angular_velocity = angular_velocity

        self.is_tracked = is_tracked

    # =========================================================================
    # INPUT UPDATE
    # =========================================================================

    def update_input(
        self,
        trigger: float = 0.0,
        grip: float = 0.0,
        thumbstick: Vec2Tuple = (0.0, 0.0),
        buttons_down: Optional[Set[XRButton]] = None,
        buttons_touched: Optional[Set[XRButton]] = None,
    ) -> None:
        """
        Update controller input state.

        Args:
            trigger: Trigger analog value (0-1)
            grip: Grip analog value (0-1)
            thumbstick: Thumbstick position
            buttons_down: Set of currently pressed buttons
            buttons_touched: Set of currently touched buttons
        """
        # Store previous frame state
        self._previous_buttons = self._current_buttons.copy()
        self._previous_touch = self._current_touch.copy()

        # Update analog inputs
        old_trigger = self.trigger
        old_grip = self.grip_value

        self.trigger = max(0.0, min(1.0, trigger))
        self.grip_value = max(0.0, min(1.0, grip))
        self.thumbstick_x = max(-1.0, min(1.0, thumbstick[0]))
        self.thumbstick_y = max(-1.0, min(1.0, thumbstick[1]))

        # Update button state
        if buttons_down is not None:
            self._current_buttons = buttons_down.copy()

        if buttons_touched is not None:
            self._current_touch = buttons_touched.copy()

        # Calculate frame-specific states
        self._pressed_buttons = self._current_buttons - self._previous_buttons
        self._released_buttons = self._previous_buttons - self._current_buttons
        self._just_touched = self._current_touch - self._previous_touch
        self._just_untouched = self._previous_touch - self._current_touch

        # Fire callbacks
        for button in self._pressed_buttons:
            for callback in self._on_button_pressed:
                callback(button)

        for button in self._released_buttons:
            for callback in self._on_button_released:
                callback(button)

        if abs(self.trigger - old_trigger) > 0.01:
            for callback in self._on_trigger_changed:
                callback(self.trigger)

        if abs(self.grip_value - old_grip) > 0.01:
            for callback in self._on_grip_changed:
                callback(self.grip_value)

    def begin_frame(self) -> None:
        """Clear frame-specific state (call at frame start)."""
        self._pressed_buttons.clear()
        self._released_buttons.clear()
        self._just_touched.clear()
        self._just_untouched.clear()

    # =========================================================================
    # BUTTON QUERIES
    # =========================================================================

    def is_button_down(self, button: XRButton) -> bool:
        """
        Check if a button is currently held down.

        Args:
            button: The button to check

        Returns:
            True if button is down
        """
        return button in self._current_buttons

    def is_button_pressed(self, button: XRButton) -> bool:
        """
        Check if a button was just pressed this frame.

        Args:
            button: The button to check

        Returns:
            True if button was pressed this frame
        """
        return button in self._pressed_buttons

    def is_button_released(self, button: XRButton) -> bool:
        """
        Check if a button was just released this frame.

        Args:
            button: The button to check

        Returns:
            True if button was released this frame
        """
        return button in self._released_buttons

    def is_button_touched(self, button: XRButton) -> bool:
        """
        Check if a button is currently being touched.

        Args:
            button: The button to check

        Returns:
            True if button is touched
        """
        return button in self._current_touch

    def is_button_just_touched(self, button: XRButton) -> bool:
        """
        Check if a button was just touched this frame.

        Args:
            button: The button to check

        Returns:
            True if button was just touched
        """
        return button in self._just_touched

    def is_button_just_untouched(self, button: XRButton) -> bool:
        """
        Check if a button was just untouched this frame.

        Args:
            button: The button to check

        Returns:
            True if button was just untouched
        """
        return button in self._just_untouched

    def get_button_value(self, button: XRButton) -> float:
        """
        Get the analog value of a button.

        Args:
            button: The button to query

        Returns:
            Analog value (0-1)
        """
        if button == XRButton.TRIGGER:
            return self.trigger
        elif button == XRButton.GRIP:
            return self.grip_value
        elif button in self._current_buttons:
            return 1.0
        return 0.0

    # =========================================================================
    # DEADZONE
    # =========================================================================

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to an axis value."""
        if abs(value) < self._thumbstick_deadzone:
            return 0.0

        sign = 1.0 if value > 0 else -1.0
        scaled = (abs(value) - self._thumbstick_deadzone) / (1.0 - self._thumbstick_deadzone)
        return sign * min(1.0, scaled)

    @property
    def thumbstick_deadzone(self) -> float:
        """Get thumbstick deadzone."""
        return self._thumbstick_deadzone

    @thumbstick_deadzone.setter
    def thumbstick_deadzone(self, value: float) -> None:
        """Set thumbstick deadzone (0-1)."""
        self._thumbstick_deadzone = max(0.0, min(0.9, value))

    @property
    def trigger_threshold(self) -> float:
        """Get trigger press threshold."""
        return self._trigger_threshold

    @trigger_threshold.setter
    def trigger_threshold(self, value: float) -> None:
        """Set trigger press threshold (0-1)."""
        self._trigger_threshold = max(0.0, min(1.0, value))

    @property
    def grip_threshold(self) -> float:
        """Get grip press threshold."""
        return self._grip_threshold

    @grip_threshold.setter
    def grip_threshold(self, value: float) -> None:
        """Set grip press threshold (0-1)."""
        self._grip_threshold = max(0.0, min(1.0, value))

    # =========================================================================
    # HAPTICS
    # =========================================================================

    def play_haptic(
        self,
        amplitude: float = 1.0,
        duration_ms: float = 100.0,
        frequency: float = 200.0,
    ) -> bool:
        """
        Queue a haptic effect for playback.

        Args:
            amplitude: Vibration intensity (0-1)
            duration_ms: Duration in milliseconds
            frequency: Vibration frequency in Hz

        Returns:
            True if haptics are supported
        """
        if not self._capabilities.has_haptics:
            return False

        self._pending_haptics.append({
            "amplitude": max(0.0, min(1.0, amplitude)),
            "duration_ms": max(0.0, duration_ms),
            "frequency": max(0.0, frequency),
        })
        return True

    def stop_haptic(self) -> None:
        """Stop all haptic effects."""
        self._pending_haptics.clear()

    def get_pending_haptics(self) -> List[Dict[str, float]]:
        """Get and clear pending haptic effects."""
        effects = self._pending_haptics.copy()
        self._pending_haptics.clear()
        return effects

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_button_pressed(self, callback: Callable[[XRButton], None]) -> None:
        """Register callback for button press."""
        self._on_button_pressed.append(callback)

    def on_button_released(self, callback: Callable[[XRButton], None]) -> None:
        """Register callback for button release."""
        self._on_button_released.append(callback)

    def on_trigger_changed(self, callback: Callable[[float], None]) -> None:
        """Register callback for trigger value changes."""
        self._on_trigger_changed.append(callback)

    def on_grip_changed(self, callback: Callable[[float], None]) -> None:
        """Register callback for grip value changes."""
        self._on_grip_changed.append(callback)

    def remove_button_pressed_callback(self, callback: Callable[[XRButton], None]) -> bool:
        """Remove a button pressed callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if callback was found and removed, False otherwise.
        """
        try:
            self._on_button_pressed.remove(callback)
            return True
        except ValueError:
            return False

    def remove_button_released_callback(self, callback: Callable[[XRButton], None]) -> bool:
        """Remove a button released callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if callback was found and removed, False otherwise.
        """
        try:
            self._on_button_released.remove(callback)
            return True
        except ValueError:
            return False

    def remove_trigger_changed_callback(self, callback: Callable[[float], None]) -> bool:
        """Remove a trigger changed callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if callback was found and removed, False otherwise.
        """
        try:
            self._on_trigger_changed.remove(callback)
            return True
        except ValueError:
            return False

    def remove_grip_changed_callback(self, callback: Callable[[float], None]) -> bool:
        """Remove a grip changed callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if callback was found and removed, False otherwise.
        """
        try:
            self._on_grip_changed.remove(callback)
            return True
        except ValueError:
            return False

    def clear_callbacks(self) -> None:
        """Clear all registered callbacks."""
        self._on_button_pressed.clear()
        self._on_button_released.clear()
        self._on_trigger_changed.clear()
        self._on_grip_changed.clear()

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize controller state to dictionary."""
        return {
            "hand": self._hand.name,
            "controller_type": self._controller_type.name,
            "device_id": self._device_id,
            "grip_position": list(self.grip_position),
            "grip_orientation": list(self.grip_orientation),
            "aim_position": list(self.aim_position),
            "aim_orientation": list(self.aim_orientation),
            "linear_velocity": list(self.linear_velocity),
            "angular_velocity": list(self.angular_velocity),
            "trigger": self.trigger,
            "grip_value": self.grip_value,
            "thumbstick": [self.thumbstick_x, self.thumbstick_y],
            "is_tracked": self.is_tracked,
            "buttons_down": [b.name for b in self._current_buttons],
            "buttons_touched": [b.name for b in self._current_touch],
            "entity_id": self._entity_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> XRController:
        """Deserialize controller state from dictionary."""
        controller = cls(
            hand=XRHand[data.get("hand", "LEFT")],
            controller_type=XRControllerType[data.get("controller_type", "MOTION")],
            device_id=data.get("device_id", ""),
            entity_id=data.get("entity_id"),
        )

        controller.grip_position = tuple(data.get("grip_position", [0.0, 0.0, 0.0]))
        controller.grip_orientation = tuple(data.get("grip_orientation", [0.0, 0.0, 0.0, 1.0]))
        controller.aim_position = tuple(data.get("aim_position", [0.0, 0.0, 0.0]))
        controller.aim_orientation = tuple(data.get("aim_orientation", [0.0, 0.0, 0.0, 1.0]))
        controller.linear_velocity = tuple(data.get("linear_velocity", [0.0, 0.0, 0.0]))
        controller.angular_velocity = tuple(data.get("angular_velocity", [0.0, 0.0, 0.0]))
        controller.trigger = data.get("trigger", 0.0)
        controller.grip_value = data.get("grip_value", 0.0)

        thumbstick = data.get("thumbstick", [0.0, 0.0])
        controller.thumbstick_x = thumbstick[0]
        controller.thumbstick_y = thumbstick[1]
        controller.is_tracked = data.get("is_tracked", False)

        controller._current_buttons = {XRButton[b] for b in data.get("buttons_down", [])}
        controller._current_touch = {XRButton[b] for b in data.get("buttons_touched", [])}

        return controller

    def __repr__(self) -> str:
        return (
            f"XRController(hand={self._hand.name}, "
            f"tracked={self.is_tracked}, "
            f"trigger={self.trigger:.2f}, grip={self.grip_value:.2f})"
        )


# Descriptor setup
XRController.grip_position.__set_name__(XRController, "grip_position")
XRController.grip_orientation.__set_name__(XRController, "grip_orientation")
XRController.aim_position.__set_name__(XRController, "aim_position")
XRController.aim_orientation.__set_name__(XRController, "aim_orientation")
XRController.linear_velocity.__set_name__(XRController, "linear_velocity")
XRController.angular_velocity.__set_name__(XRController, "angular_velocity")
XRController.trigger.__set_name__(XRController, "trigger")
XRController.grip_value.__set_name__(XRController, "grip_value")
XRController.thumbstick_x.__set_name__(XRController, "thumbstick_x")
XRController.thumbstick_y.__set_name__(XRController, "thumbstick_y")
XRController.is_tracked.__set_name__(XRController, "is_tracked")


__all__ = [
    "XRController",
    "XRHand",
    "XRButton",
    "XRControllerType",
    "ControllerCapabilities",
    "ButtonState",
]
