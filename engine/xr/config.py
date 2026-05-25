"""
Unified XR Configuration Module

Centralizes all magic numbers and configuration constants used across the XR subsystem.
This module provides a single source of truth for all XR-related parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum, auto


# =============================================================================
# RUNTIME AND INPUT CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class XRRuntimeConfig:
    """Configuration constants for XR runtime and input systems."""

    # IPD (Inter-Pupillary Distance)
    DEFAULT_IPD_MM: float = 63.0
    MIN_IPD_MM: float = 50.0
    MAX_IPD_MM: float = 75.0

    # Field of View
    DEFAULT_FOV_DEGREES: float = 110.0
    MIN_FOV_DEGREES: float = 80.0
    MAX_FOV_DEGREES: float = 140.0

    # Refresh Rate
    DEFAULT_REFRESH_RATE_HZ: int = 90
    MIN_REFRESH_RATE_HZ: int = 60
    MAX_REFRESH_RATE_HZ: int = 144

    # Hand Tracking
    HAND_JOINT_COUNT: int = 26
    FINGER_COUNT: int = 5
    JOINTS_PER_FINGER: int = 4

    # Pinch Detection
    PINCH_THRESHOLD: float = 0.02
    PINCH_RELEASE_THRESHOLD: float = 0.04
    PINCH_STRENGTH_CURVE_POWER: float = 2.0

    # Gesture Recognition
    GESTURE_CONFIDENCE_THRESHOLD: float = 0.8
    GESTURE_HOLD_TIME_SECONDS: float = 0.3
    GESTURE_COOLDOWN_SECONDS: float = 0.5

    # Eye Tracking
    GAZE_SMOOTHING_FACTOR: float = 0.3
    FIXATION_THRESHOLD_DEGREES: float = 1.5
    FIXATION_DURATION_MS: int = 100
    SACCADE_VELOCITY_THRESHOLD: float = 30.0
    PUPIL_DIAMETER_MIN_MM: float = 2.0
    PUPIL_DIAMETER_MAX_MM: float = 8.0
    DEFAULT_PUPIL_DIAMETER_MM: float = 3.0

    # Blink Detection
    BLINK_THRESHOLD: float = 0.3
    AUTO_BLINK_MIN_INTERVAL_SECONDS: float = 3.0
    AUTO_BLINK_RANDOM_RANGE_SECONDS: float = 2.0
    BLINK_DURATION_SECONDS: float = 0.15

    # Eye Tracking Smoothing
    EYE_BLEND_SPEED: float = 10.0


# =============================================================================
# RENDERING CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class XRRenderingConfig:
    """Configuration constants for XR rendering systems."""

    # Foveated Rendering Regions (normalized radii)
    FOVEA_RADIUS: float = 0.1
    INNER_RADIUS: float = 0.3
    OUTER_RADIUS: float = 0.6

    # Resolution Scaling per Region
    FOVEA_RESOLUTION_SCALE: float = 1.0
    INNER_RESOLUTION_SCALE: float = 0.7
    OUTER_RESOLUTION_SCALE: float = 0.4
    PERIPHERAL_RESOLUTION_SCALE: float = 0.25

    # Reprojection
    REPROJECTION_THRESHOLD_MS: float = 11.1  # ~90fps threshold
    MOTION_VECTOR_SCALE: float = 1.0
    MAX_EXTRAPOLATION_MS: float = 22.2

    # Hidden Area Mesh
    HIDDEN_AREA_MESH_SEGMENTS: int = 32
    STENCIL_REFERENCE_VALUE: int = 1

    # Late Latching
    LATE_LATCH_DEADLINE_MS: float = 2.0
    POSE_PREDICTION_MS: float = 20.0

    # Multi-view Rendering
    MAX_VIEW_COUNT: int = 2
    STEREO_CONVERGENCE_DISTANCE: float = 1.0

    # Lens Distortion
    DISTORTION_K1: float = 0.22
    DISTORTION_K2: float = 0.24
    CHROMATIC_ABERRATION_SCALE: float = 0.01


# =============================================================================
# INTERACTION CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class XRInteractionConfig:
    """Configuration constants for XR interaction systems."""

    # Ray Interactor
    RAY_MAX_LENGTH: float = 10.0
    RAY_DEFAULT_LENGTH: float = 5.0
    RAY_WIDTH: float = 0.005
    RAY_HIT_RADIUS: float = 0.01

    # Grab Mechanics
    GRAB_RADIUS: float = 0.05
    GRAB_ACTIVATION_THRESHOLD: float = 0.7
    GRAB_RELEASE_THRESHOLD: float = 0.3

    # Haptic Feedback
    HAPTIC_PULSE_DURATION_MS: float = 50.0
    HAPTIC_DEFAULT_AMPLITUDE: float = 0.5
    HAPTIC_MAX_AMPLITUDE: float = 1.0
    HAPTIC_MIN_AMPLITUDE: float = 0.1

    # Object Manipulation
    THROW_VELOCITY_SCALE: float = 1.5
    THROW_ANGULAR_VELOCITY_SCALE: float = 1.0
    SMOOTHING_FACTOR: float = 0.5

    # Two-Hand Manipulation
    TWO_HAND_SCALE_MIN: float = 0.1
    TWO_HAND_SCALE_MAX: float = 10.0
    TWO_HAND_ROTATION_THRESHOLD_DEGREES: float = 5.0

    # Physics Interaction
    PHYSICS_FOLLOW_FORCE: float = 1000.0
    PHYSICS_FOLLOW_DAMPING: float = 50.0
    MAX_ANGULAR_VELOCITY: float = 20.0

    # UI Interaction
    UI_POKE_DEPTH: float = 0.02
    UI_HOVER_DISTANCE: float = 0.1
    UI_PRESS_THRESHOLD: float = 0.01

    # Velocity Tracking
    VELOCITY_SAMPLE_COUNT: int = 5
    VELOCITY_SAMPLE_WINDOW_SECONDS: float = 0.1


# =============================================================================
# SPATIAL CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class XRSpatialConfig:
    """Configuration constants for XR spatial understanding systems."""

    # Anchor System
    ANCHOR_PERSISTENCE_KEY_PREFIX: str = "xr_anchor_"
    ANCHOR_UPDATE_INTERVAL_MS: int = 100
    ANCHOR_CONFIDENCE_THRESHOLD: float = 0.7
    MAX_ANCHORS_PER_SESSION: int = 100

    # Cloud Anchors
    DEFAULT_CLOUD_ANCHOR_EXPIRY_DAYS: int = 365
    CLOUD_ANCHOR_TIMEOUT_SECONDS: float = 30.0
    CONFIDENCE_DECAY_RATE: float = 0.1

    # Plane Detection
    PLANE_MIN_AREA_M2: float = 0.25
    PLANE_MERGE_DISTANCE_M: float = 0.05
    PLANE_MERGE_ANGLE_DEGREES: float = 5.0
    PLANE_UPDATE_INTERVAL_MS: int = 200
    HORIZONTAL_PLANE_THRESHOLD: float = 0.9
    VERTICAL_PLANE_THRESHOLD: float = 0.1
    DEFAULT_PLANE_TOLERANCE: float = 0.01
    MIN_PLACEMENT_AREA_M2: float = 0.1

    # Raycasting
    RAY_EPSILON: float = 1e-6
    MAX_RAYCAST_DISTANCE: float = 100.0

    # Mesh Generation
    MESH_VOXEL_SIZE_M: float = 0.02
    MESH_MAX_TRIANGLES: int = 100000
    MESH_UPDATE_INTERVAL_MS: int = 500

    # Scene Understanding
    SCENE_UPDATE_INTERVAL_MS: int = 1000
    SEMANTIC_CONFIDENCE_THRESHOLD: float = 0.6
    MAX_SCENE_OBJECTS: int = 200

    # Occlusion
    OCCLUSION_FADE_DISTANCE_M: float = 0.1
    DEPTH_BUFFER_PRECISION_BITS: int = 24


# =============================================================================
# AVATAR CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class XRAvatarConfig:
    """Configuration constants for XR avatar systems."""

    # IK Solver
    IK_MAX_ITERATIONS: int = 10
    IK_TOLERANCE: float = 0.001
    IK_POSITION_WEIGHT: float = 1.0
    IK_ROTATION_WEIGHT: float = 0.5

    # Body Proportions (normalized to height)
    ARM_LENGTH_RATIO: float = 0.44
    LEG_LENGTH_RATIO: float = 0.53
    SHOULDER_WIDTH_RATIO: float = 0.26
    HIP_WIDTH_RATIO: float = 0.20
    HEAD_HEIGHT_RATIO: float = 0.13
    NECK_LENGTH_RATIO: float = 0.05
    TORSO_LENGTH_RATIO: float = 0.30

    # Calibration
    CALIBRATION_SAMPLE_COUNT: int = 60
    CALIBRATION_DURATION_SECONDS: float = 3.0
    HEIGHT_ESTIMATION_OFFSET_M: float = 0.1
    DEFAULT_AVATAR_HEIGHT_M: float = 1.75
    MIN_AVATAR_HEIGHT_M: float = 1.2
    MAX_AVATAR_HEIGHT_M: float = 2.2

    # Lip Sync
    VISEME_BLEND_SPEED: float = 15.0
    AUDIO_SAMPLE_WINDOW_MS: int = 50
    VISEME_THRESHOLD: float = 0.1
    VISEME_COUNT: int = 15

    # Eye Animation
    EYE_LOOK_AT_SPEED: float = 10.0
    EYE_RANDOM_LOOK_INTERVAL_SECONDS: float = 3.0
    EYE_RANDOM_LOOK_RANGE_DEGREES: float = 15.0

    # Face Tracking
    FACE_BLEND_SHAPE_COUNT: int = 52
    FACE_TRACKING_FPS: int = 60
    EXPRESSION_SMOOTHING_FACTOR: float = 0.3

    # Network Sync
    AVATAR_SYNC_RATE_HZ: int = 30
    POSITION_INTERPOLATION_SPEED: float = 10.0
    ROTATION_INTERPOLATION_SPEED: float = 15.0


# =============================================================================
# LOCOMOTION CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class XRLocomotionConfig:
    """Configuration constants for XR locomotion systems."""

    # Teleport
    TELEPORT_MAX_DISTANCE_M: float = 10.0
    TELEPORT_ARC_VELOCITY: float = 10.0
    TELEPORT_ARC_GRAVITY: float = 9.81
    TELEPORT_FADE_DURATION_SECONDS: float = 0.2
    TELEPORT_COOLDOWN_SECONDS: float = 0.5
    TELEPORT_VALID_SLOPE_DEGREES: float = 45.0

    # Continuous Movement
    MOVE_SPEED_DEFAULT_MPS: float = 3.0
    MOVE_SPEED_MIN_MPS: float = 1.0
    MOVE_SPEED_MAX_MPS: float = 6.0
    ACCELERATION_MPS2: float = 10.0
    DECELERATION_MPS2: float = 15.0

    # Snap Turn
    SNAP_TURN_ANGLE_DEGREES: float = 45.0
    SNAP_TURN_COOLDOWN_SECONDS: float = 0.3
    SNAP_TURN_ANGLE_MIN_DEGREES: float = 15.0
    SNAP_TURN_ANGLE_MAX_DEGREES: float = 90.0

    # Smooth Turn
    SMOOTH_TURN_SPEED_DPS: float = 90.0
    SMOOTH_TURN_SPEED_MIN_DPS: float = 30.0
    SMOOTH_TURN_SPEED_MAX_DPS: float = 180.0

    # Comfort Vignette
    VIGNETTE_INTENSITY_DEFAULT: float = 0.5
    VIGNETTE_INTENSITY_MIN: float = 0.0
    VIGNETTE_INTENSITY_MAX: float = 1.0
    VIGNETTE_FADE_SPEED: float = 5.0
    VIGNETTE_RADIUS_INNER: float = 0.4
    VIGNETTE_RADIUS_OUTER: float = 0.7

    # Comfort Presets
    COMFORT_LOW_VIGNETTE_INTENSITY: float = 0.8
    COMFORT_LOW_FADE_DURATION: float = 0.25
    COMFORT_LOW_SPEED_SCALE: float = 0.5
    COMFORT_LOW_SNAP_ANGLE: float = 30.0

    COMFORT_MEDIUM_VIGNETTE_INTENSITY: float = 0.5
    COMFORT_MEDIUM_FADE_DURATION: float = 0.15
    COMFORT_MEDIUM_SPEED_SCALE: float = 0.75
    COMFORT_MEDIUM_SNAP_ANGLE: float = 45.0

    COMFORT_HIGH_VIGNETTE_INTENSITY: float = 0.3
    COMFORT_HIGH_FADE_DURATION: float = 0.05
    COMFORT_HIGH_SPEED_SCALE: float = 1.0
    COMFORT_HIGH_SNAP_ANGLE: float = 45.0

    # Angular Velocity Thresholds
    ANGULAR_VELOCITY_LOW_THRESHOLD_DPS: float = 30.0
    ANGULAR_VELOCITY_HIGH_THRESHOLD_DPS: float = 60.0

    # Physical Movement
    PHYSICAL_CROUCH_THRESHOLD_M: float = 0.4
    PHYSICAL_JUMP_VELOCITY_MPS: float = 3.0
    PHYSICAL_STEP_HEIGHT_M: float = 0.3


# =============================================================================
# UI CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class XRUIConfig:
    """Configuration constants for XR UI systems."""

    # World Space UI
    UI_PIXELS_PER_METER: float = 1000.0
    UI_DEFAULT_DISTANCE_M: float = 2.0
    UI_MIN_DISTANCE_M: float = 0.5
    UI_MAX_DISTANCE_M: float = 10.0

    # Panel Sizes
    UI_PANEL_WIDTH_M: float = 0.5
    UI_PANEL_HEIGHT_M: float = 0.3
    UI_BUTTON_SIZE_M: float = 0.05
    UI_BUTTON_SPACING_M: float = 0.01

    # Wrist UI
    WRIST_OFFSET_X: float = 0.03
    WRIST_OFFSET_Y: float = 0.02
    WRIST_OFFSET_Z: float = 0.0
    WRIST_UI_SIZE: float = 0.08
    WRIST_UI_SCALE: float = 0.001
    WRIST_LENGTH_EPSILON: float = 0.001

    # Wrist UI Activation
    WRIST_LOOK_THRESHOLD_DEGREES: float = 30.0
    WRIST_PALM_UP_THRESHOLD_DEGREES: float = 45.0
    WRIST_UI_FADE_DURATION: float = 0.3

    # Cursor
    CURSOR_SIZE_DEFAULT: float = 0.01
    CURSOR_SIZE_HOVER: float = 0.015
    CURSOR_DEPTH_OFFSET: float = 0.001

    # Haptic Feedback
    UI_HAPTIC_CLICK_DURATION_MS: float = 10.0
    UI_HAPTIC_CLICK_AMPLITUDE: float = 0.3
    UI_HAPTIC_HOVER_DURATION_MS: float = 5.0
    UI_HAPTIC_HOVER_AMPLITUDE: float = 0.1

    # Keyboard
    KEYBOARD_KEY_SIZE_M: float = 0.04
    KEYBOARD_KEY_SPACING_M: float = 0.005
    KEYBOARD_DISTANCE_M: float = 0.4
    KEYBOARD_ANGLE_DEGREES: float = 30.0

    # Tooltip
    TOOLTIP_DELAY_MS: int = 500
    TOOLTIP_FADE_DURATION_MS: int = 200
    TOOLTIP_OFFSET_M: float = 0.02

    # Scrolling
    SCROLL_SENSITIVITY: float = 0.1
    SCROLL_INERTIA_DECAY: float = 0.95
    SCROLL_MAX_VELOCITY: float = 2.0


# =============================================================================
# PLATFORM CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class XRPlatformConfig:
    """Configuration constants for platform-specific features."""

    # Guardian/Boundary System
    GUARDIAN_WARNING_DISTANCE_M: float = 0.5
    GUARDIAN_FADE_START_DISTANCE_M: float = 1.0
    GUARDIAN_FADE_END_DISTANCE_M: float = 0.3
    GUARDIAN_GRID_SIZE_M: float = 0.1
    GUARDIAN_HEIGHT_M: float = 2.5
    GUARDIAN_INNER_ALPHA: float = 0.5
    GUARDIAN_OUTER_ALPHA: float = 0.8
    GUARDIAN_UPDATE_INTERVAL_SECONDS: float = 0.1

    # Passthrough
    PASSTHROUGH_OPACITY_DEFAULT: float = 1.0
    PASSTHROUGH_BRIGHTNESS_DEFAULT: float = 1.0
    PASSTHROUGH_CONTRAST_DEFAULT: float = 1.0
    PASSTHROUGH_EDGE_ENHANCEMENT: float = 0.0

    # Mixed Reality
    MR_DEPTH_OCCLUSION_THRESHOLD: float = 0.02
    MR_SHADOW_OPACITY: float = 0.5
    MR_LIGHTING_ESTIMATION_INTERVAL_MS: int = 100

    # Performance
    TARGET_FRAME_TIME_MS_90HZ: float = 11.1
    TARGET_FRAME_TIME_MS_72HZ: float = 13.9
    TARGET_FRAME_TIME_MS_60HZ: float = 16.7
    GPU_HEADROOM_PERCENTAGE: float = 0.1

    # Power Management
    THERMAL_THROTTLE_TEMP_C: float = 40.0
    BATTERY_LOW_THRESHOLD_PERCENT: int = 20
    BATTERY_CRITICAL_THRESHOLD_PERCENT: int = 5

    # Social/Multiplayer
    VOICE_CHAT_SAMPLE_RATE_HZ: int = 16000
    VOICE_CHAT_BUFFER_MS: int = 100
    AVATAR_VISIBLE_DISTANCE_M: float = 50.0
    NAME_TAG_VISIBLE_DISTANCE_M: float = 20.0
    NAME_TAG_FADE_DISTANCE_M: float = 15.0

    # Network
    POSE_SYNC_INTERVAL_MS: int = 33  # ~30Hz
    INTERPOLATION_DELAY_MS: int = 100
    MAX_NETWORK_LATENCY_MS: int = 200


# =============================================================================
# GLOBAL CONFIG INSTANCE
# =============================================================================

class XRConfig:
    """
    Global XR configuration container.

    Provides access to all XR subsystem configurations through a single interface.
    All configurations are immutable (frozen dataclasses) to prevent runtime modification.

    Usage:
        from engine.xr.config import XR_CONFIG

        ipd = XR_CONFIG.runtime.DEFAULT_IPD_MM
        ray_length = XR_CONFIG.interaction.RAY_MAX_LENGTH
    """

    runtime: XRRuntimeConfig = XRRuntimeConfig()
    rendering: XRRenderingConfig = XRRenderingConfig()
    interaction: XRInteractionConfig = XRInteractionConfig()
    spatial: XRSpatialConfig = XRSpatialConfig()
    avatar: XRAvatarConfig = XRAvatarConfig()
    locomotion: XRLocomotionConfig = XRLocomotionConfig()
    ui: XRUIConfig = XRUIConfig()
    platform: XRPlatformConfig = XRPlatformConfig()


# Global singleton instance
XR_CONFIG = XRConfig()


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

# For direct imports of specific config classes
__all__ = [
    'XRConfig',
    'XR_CONFIG',
    'XRRuntimeConfig',
    'XRRenderingConfig',
    'XRInteractionConfig',
    'XRSpatialConfig',
    'XRAvatarConfig',
    'XRLocomotionConfig',
    'XRUIConfig',
    'XRPlatformConfig',
]
