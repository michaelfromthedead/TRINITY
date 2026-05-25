"""XR input module providing HMD, controller, hand tracking and eye tracking capabilities.

This module provides comprehensive input support for XR (VR/AR/MR) applications,
following the Trinity Pattern with appropriate descriptors for tracking data.

Exports:
    HMD Tracking:
        - HeadMountedDisplay: HMD pose tracking component with prediction
        - HMDTrackingState: Tracking state machine states
        - HMD_STATE_TRANSITIONS: Valid state transitions
        - HMDDisplayInfo: Display specifications
        - PredictionConfig: Prediction configuration

    Controller Input:
        - XRController: Motion controller component
        - XRHand: Left/right hand enum
        - XRButton: Controller button identifiers
        - XRControllerType: Controller type enum
        - ControllerCapabilities: Controller capability flags
        - ButtonState: Single button state

    Input Bindings:
        - xr_action: Decorator for XR input actions
        - xr_axis: Decorator for XR input axes
        - XRActionType: Action type enum
        - XRInputSource: Input source identifiers
        - XRActionBinding: Action-to-input binding
        - XRInputProfile: Controller input profile
        - XRActionRegistry: Action registry
        - get_xr_action_registry: Get global registry
        - bind_action: Bind handler to action
        - unbind_action: Unbind handler from action
        - create_profile: Create input profile
        - get_action_value: Get action value from state

    Haptic Feedback:
        - HapticType: Haptic effect types
        - HapticWaveform: Waveform shapes
        - HapticEffect: Haptic effect definition
        - HapticPattern: Haptic pattern sequence
        - HapticCapabilities: Device haptic capabilities
        - HapticManager: Haptic feedback manager
        - get_haptic_manager: Get global manager
        - play_haptic: Play simple haptic
        - play_click: Play click haptic
        - play_pulse: Play pulse haptic
        - stop_haptics: Stop haptics on device

    Hand Tracking:
        - HandJoint: Enum of 26 hand joints
        - HAND_JOINT_COUNT: Constant for joint count (26)
        - GestureType: Standard gesture types
        - JointData: Single joint position/orientation/radius
        - GestureResult: Gesture detection result
        - HandTrackingData: Component with all hand tracking data
        - GestureRecognizer: Gesture detection system
        - GestureEvent: Event fired on gesture changes
        - HandTracker: High-level hand tracking manager

    Eye Tracking:
        - EyeId: Enum for eye identification
        - CalibrationState: Eye calibration states
        - GazeState: Gaze behavior states
        - EyeData: Single eye data
        - FixationData: Fixation detection data
        - SaccadeData: Saccade detection data
        - BlinkData: Blink detection data
        - EyeTrackingData: Component with all eye tracking data
        - FixationDetector: Fixation/saccade detection
        - BlinkDetector: Blink detection
        - CalibrationPoint: Calibration target point
        - EyeCalibration: Calibration system
        - EyeTracker: High-level eye tracking manager
"""

# HMD tracking
from .hmd import (
    HeadMountedDisplay,
    HMDTrackingState,
    HMD_STATE_TRANSITIONS,
    HMDDisplayInfo,
    PredictionConfig,
)

# Controller input
from .controller import (
    XRController,
    XRHand,
    XRButton,
    XRControllerType,
    ControllerCapabilities,
    ButtonState,
)

# Input bindings
from .bindings import (
    xr_action,
    xr_axis,
    XRActionType,
    XRInputSource,
    XRActionBinding,
    XRInputProfile,
    XRActionRegistry,
    get_xr_action_registry,
    bind_action,
    unbind_action,
    create_profile,
    get_action_value,
)

# Haptic feedback
from .haptics import (
    HapticType,
    HapticWaveform,
    HapticEffect,
    HapticPattern,
    HapticCapabilities,
    HapticManager,
    get_haptic_manager,
    play_haptic,
    play_click,
    play_pulse,
    stop_haptics,
)

# Hand tracking
from .hand_tracking import (
    HandJoint,
    HAND_JOINT_COUNT,
    GestureType,
    JointData,
    GestureResult,
    HandTrackingData,
    GestureRecognizer,
    GestureEvent,
    HandTracker,
)

# Eye tracking
from .eye_tracking import (
    EyeId,
    CalibrationState,
    GazeState,
    EyeData,
    FixationData,
    SaccadeData,
    BlinkData,
    EyeTrackingData,
    FixationDetector,
    BlinkDetector,
    CalibrationPoint,
    EyeCalibration,
    EyeTracker,
)

__all__ = [
    # HMD tracking
    "HeadMountedDisplay",
    "HMDTrackingState",
    "HMD_STATE_TRANSITIONS",
    "HMDDisplayInfo",
    "PredictionConfig",
    # Controller input
    "XRController",
    "XRHand",
    "XRButton",
    "XRControllerType",
    "ControllerCapabilities",
    "ButtonState",
    # Input bindings
    "xr_action",
    "xr_axis",
    "XRActionType",
    "XRInputSource",
    "XRActionBinding",
    "XRInputProfile",
    "XRActionRegistry",
    "get_xr_action_registry",
    "bind_action",
    "unbind_action",
    "create_profile",
    "get_action_value",
    # Haptic feedback
    "HapticType",
    "HapticWaveform",
    "HapticEffect",
    "HapticPattern",
    "HapticCapabilities",
    "HapticManager",
    "get_haptic_manager",
    "play_haptic",
    "play_click",
    "play_pulse",
    "stop_haptics",
    # Hand tracking
    "HandJoint",
    "HAND_JOINT_COUNT",
    "GestureType",
    "JointData",
    "GestureResult",
    "HandTrackingData",
    "GestureRecognizer",
    "GestureEvent",
    "HandTracker",
    # Eye tracking
    "EyeId",
    "CalibrationState",
    "GazeState",
    "EyeData",
    "FixationData",
    "SaccadeData",
    "BlinkData",
    "EyeTrackingData",
    "FixationDetector",
    "BlinkDetector",
    "CalibrationPoint",
    "EyeCalibration",
    "EyeTracker",
]
