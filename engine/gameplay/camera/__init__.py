"""Camera System - Complete camera control for the AI Game Engine.

This module provides a comprehensive camera system including:

- Multiple camera controllers (first-person, third-person, orbit, etc.)
- Camera collision detection and response
- Visual effects (shake, DOF, motion blur, vignette)
- Rails and paths for cinematic cameras
- Smooth blending and transitions
- Priority-based camera selection
- Split-screen support

Example usage:

    from engine.gameplay.camera import (
        ThirdPersonController,
        CameraCollision,
        CameraShake,
        CameraPriority,
    )

    # Create a third-person camera
    camera = ThirdPersonController(
        boom_length=300.0,
        socket_offset=Vec3(0, 60, 0),
    )
    camera.target = player.transform

    # Add collision
    collision = CameraCollision()

    # Add shake effect
    shake = CameraShake()
    shake.add_trauma(0.5)

    # Use priority system for multiple cameras
    priority = CameraPriority()
    priority.register(camera, priority=0, tag="gameplay")
    priority.register(cinematic_cam, priority=100, tag="cinematic")
"""

from engine.gameplay.camera.constants import (
    # FOV constants
    DEFAULT_FOV,
    MIN_FOV,
    MAX_FOV,
    ZOOMED_FOV,
    SPRINT_FOV_INCREASE,
    FOV_TRANSITION_SPEED,
    # Distance constants
    DEFAULT_CAMERA_DISTANCE,
    MIN_CAMERA_DISTANCE,
    MAX_CAMERA_DISTANCE,
    DEFAULT_ORBIT_DISTANCE,
    MIN_ORBIT_DISTANCE,
    MAX_ORBIT_DISTANCE,
    DEFAULT_ZOOM_SPEED,
    SCROLL_ZOOM_SENSITIVITY,
    # Third-person constants
    DEFAULT_BOOM_ARM_LENGTH,
    DEFAULT_SOCKET_OFFSET_X,
    DEFAULT_SOCKET_OFFSET_Y,
    DEFAULT_SOCKET_OFFSET_Z,
    DEFAULT_TARGET_OFFSET_X,
    DEFAULT_TARGET_OFFSET_Y,
    DEFAULT_TARGET_OFFSET_Z,
    DEFAULT_CAMERA_LAG_SPEED,
    DEFAULT_ROTATION_LAG_SPEED,
    MAX_LAG_DISTANCE,
    LAG_RECOVERY_SPEED,
    # First-person constants
    DEFAULT_EYE_HEIGHT,
    CROUCH_EYE_HEIGHT,
    DEFAULT_HEAD_BOB_AMPLITUDE,
    DEFAULT_HEAD_BOB_FREQUENCY,
    DEFAULT_HEAD_BOB_SWAY,
    WEAPON_SWAY_AMPLITUDE,
    LANDING_IMPACT_SCALE,
    # Collision constants
    COLLISION_PROBE_RADIUS,
    MIN_COLLISION_DISTANCE,
    COLLISION_PULL_IN_SPEED,
    COLLISION_PUSH_OUT_SPEED,
    DEFAULT_COLLISION_MASK,
    MAX_COLLISION_PROBES,
    COLLISION_INTERP_SPEED,
    # Shake constants
    SHAKE_DECAY_RATE,
    SHAKE_TRAUMA_EXPONENT,
    MAX_TRAUMA,
    DEFAULT_SHAKE_FREQUENCY,
    DEFAULT_SHAKE_AMPLITUDE_TRANSLATION,
    DEFAULT_SHAKE_AMPLITUDE_ROTATION,
    SHAKE_NOISE_OCTAVES,
    SHAKE_NOISE_PERSISTENCE,
    EXPLOSION_SHAKE_TRAUMA,
    DAMAGE_SHAKE_TRAUMA,
    FOOTSTEP_SHAKE_TRAUMA,
    # Blend constants
    BLEND_DURATION_CUT,
    BLEND_DURATION_FAST,
    BLEND_DURATION_EASE,
    BLEND_DURATION_SMOOTH,
    BLEND_DURATION_LONG,
    DEATH_CAMERA_BLEND,
    VEHICLE_CAMERA_BLEND,
    ADS_BLEND_DURATION,
    # Rail constants
    RAIL_SPLINE_RESOLUTION,
    MIN_RAIL_SEGMENT_LENGTH,
    MAX_RAIL_POINTS,
    DEFAULT_SPLINE_TENSION,
    DEFAULT_RAIL_SPEED,
    DEFAULT_DOLLY_SPEED,
    DEFAULT_CRANE_ARC_ANGLE,
    DEFAULT_CRANE_ARM_LENGTH,
    # Angle limits
    MIN_PITCH_ANGLE,
    MAX_PITCH_ANGLE,
    MIN_ORBIT_PITCH,
    MAX_ORBIT_PITCH,
    TOP_DOWN_PITCH,
    ISOMETRIC_PITCH,
    ISOMETRIC_ROTATION_SNAP,
    # Sensitivity
    DEFAULT_MOUSE_SENSITIVITY,
    MIN_MOUSE_SENSITIVITY,
    MAX_MOUSE_SENSITIVITY,
    ADS_SENSITIVITY_MULTIPLIER,
    DEFAULT_CONTROLLER_SENSITIVITY,
    DEFAULT_ORBIT_ROTATION_SPEED,
    DEFAULT_FREE_CAM_SPEED,
    FREE_CAM_FAST_MULTIPLIER,
    FREE_CAM_SLOW_MULTIPLIER,
    # DOF constants
    DEFAULT_FOCUS_DISTANCE,
    DEFAULT_APERTURE,
    DEFAULT_FOCAL_LENGTH,
    MIN_FOCUS_DISTANCE,
    MAX_FOCUS_DISTANCE,
    AUTO_FOCUS_SPEED,
    # Motion blur constants
    DEFAULT_MOTION_BLUR_INTENSITY,
    MAX_MOTION_BLUR,
    MOTION_BLUR_VELOCITY_THRESHOLD,
    MOTION_BLUR_SAMPLES,
    # Vignette constants
    DEFAULT_VIGNETTE_INTENSITY,
    DAMAGE_VIGNETTE_INTENSITY,
    LOW_HEALTH_VIGNETTE_INTENSITY,
    DEFAULT_VIGNETTE_FEATHER,
    VIGNETTE_TRANSITION_SPEED,
    # Priority constants
    DEFAULT_CAMERA_PRIORITY,
    CINEMATIC_CAMERA_PRIORITY,
    CUTSCENE_CAMERA_PRIORITY,
    DEBUG_CAMERA_PRIORITY,
    # Plane constants
    DEFAULT_NEAR_PLANE,
    DEFAULT_FAR_PLANE,
    MIN_NEAR_PLANE,
    MAX_FAR_PLANE,
    # Timing constants
    MIN_DELTA_TIME,
    MAX_DELTA_TIME,
    CAMERA_UPDATE_RATE,
    INTERPOLATION_BUFFER_SIZE,
    # Math constants
    DEG_TO_RAD,
    RAD_TO_DEG,
    CAMERA_EPSILON,
    SMALL_ANGLE_THRESHOLD,
    # Trigger constants
    TRIGGER_CHECK_INTERVAL,
    TRIGGER_HYSTERESIS,
    MAX_ACTIVE_TRIGGERS,
)

from engine.gameplay.camera.controller import (
    CameraMode,
    CameraState,
    BaseCameraController,
    FirstPersonController,
    ThirdPersonController,
    OrbitController,
    FollowController,
    FreeController,
    CinematicController,
    CinematicKeyframe,
    TopDownController,
    IsometricController,
)

from engine.gameplay.camera.collision import (
    CollisionResponse,
    CollisionHit,
    CollisionSettings,
    CameraCollision,
    OcclusionDetector,
    TransparencyManager,
)

from engine.gameplay.camera.effects import (
    ShakeType,
    ShakeSettings,
    CameraShake,
    ShakeInstance,
    ScreenShake,
    FOVEffect,
    TiltEffect,
    DOFSettings,
    DOFEffect,
    MotionBlurSettings,
    MotionBlur,
    VignetteSettings,
    VignetteEffect,
    CameraEffectsManager,
)

from engine.gameplay.camera.rails import (
    LoopMode,
    SplineType,
    RailPoint,
    CameraRail,
    RailFollower,
    TriggerBounds,
    TriggerVolume,
    BlendRegion,
    Dolly,
    Crane,
    TriggerVolumeManager,
)

from engine.gameplay.camera.blending import (
    BlendType,
    BlendCurve,
    CameraBlendState,
    CameraBlend,
    BlendStack,
    ViewportRect,
    SplitScreenLayout,
    ViewportSplit,
    PrioritizedCamera,
    CameraPriority,
    CameraDirector,
)


__all__ = [
    # === Constants (key ones - full list in constants.py) ===
    "DEFAULT_FOV",
    "MIN_FOV",
    "MAX_FOV",
    "DEFAULT_CAMERA_DISTANCE",
    "DEFAULT_BOOM_ARM_LENGTH",
    "DEFAULT_EYE_HEIGHT",
    "COLLISION_PROBE_RADIUS",
    "SHAKE_DECAY_RATE",
    "BLEND_DURATION_EASE",
    "RAIL_SPLINE_RESOLUTION",
    "DEFAULT_MOUSE_SENSITIVITY",
    "DEFAULT_CAMERA_PRIORITY",
    "DEG_TO_RAD",
    "RAD_TO_DEG",

    # === Controller ===
    "CameraMode",
    "CameraState",
    "BaseCameraController",
    "FirstPersonController",
    "ThirdPersonController",
    "OrbitController",
    "FollowController",
    "FreeController",
    "CinematicController",
    "CinematicKeyframe",
    "TopDownController",
    "IsometricController",

    # === Collision ===
    "CollisionResponse",
    "CollisionHit",
    "CollisionSettings",
    "CameraCollision",
    "OcclusionDetector",
    "TransparencyManager",

    # === Effects ===
    "ShakeType",
    "ShakeSettings",
    "CameraShake",
    "ShakeInstance",
    "ScreenShake",
    "FOVEffect",
    "TiltEffect",
    "DOFSettings",
    "DOFEffect",
    "MotionBlurSettings",
    "MotionBlur",
    "VignetteSettings",
    "VignetteEffect",
    "CameraEffectsManager",

    # === Rails ===
    "LoopMode",
    "SplineType",
    "RailPoint",
    "CameraRail",
    "RailFollower",
    "TriggerBounds",
    "TriggerVolume",
    "BlendRegion",
    "Dolly",
    "Crane",
    "TriggerVolumeManager",

    # === Blending ===
    "BlendType",
    "BlendCurve",
    "CameraBlendState",
    "CameraBlend",
    "BlendStack",
    "ViewportRect",
    "SplitScreenLayout",
    "ViewportSplit",
    "PrioritizedCamera",
    "CameraPriority",
    "CameraDirector",
]
