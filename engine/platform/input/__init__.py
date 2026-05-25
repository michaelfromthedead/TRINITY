"""Input subsystem for the game engine platform layer.

This module provides comprehensive input handling for various device types:
- Keyboard
- Mouse
- Gamepad/Controller
- Touch screens
- Pen/Stylus
- XR controllers and hand tracking
- Haptic feedback
"""

from .gamepad import Gamepad, GamepadAxis, GamepadButton, GamepadTrigger
from .haptics import HapticEffect, Haptics, HapticType
from .input_manager import InputDevice, InputDeviceType, InputEvent, InputManager
from .keyboard import KeyCode, KeyState, Keyboard
from .mouse import Mouse, MouseButton
from .pen import PenDevice
from .touch import TouchDevice, TouchPhase, TouchPoint
from .xr_input import (
    HandJoint,
    JointPose,
    Pose,
    XRButton,
    XRController,
    XRHand,
)

__all__ = [
    # Input Manager
    'InputManager',
    'InputDevice',
    'InputDeviceType',
    'InputEvent',

    # Keyboard
    'Keyboard',
    'KeyCode',
    'KeyState',

    # Mouse
    'Mouse',
    'MouseButton',

    # Gamepad
    'Gamepad',
    'GamepadAxis',
    'GamepadTrigger',
    'GamepadButton',

    # Touch
    'TouchDevice',
    'TouchPoint',
    'TouchPhase',

    # Pen
    'PenDevice',

    # Haptics
    'Haptics',
    'HapticType',
    'HapticEffect',

    # XR
    'XRController',
    'XRHand',
    'XRButton',
    'HandJoint',
    'Pose',
    'JointPose',
]
