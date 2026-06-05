"""
Cinematic animation system.

Provides:
- Cutscene playback with timeline-based events
- Camera track animation with spline interpolation
- Gameplay blending and state management
"""

from engine.animation.cinematics.camera_track import (
    BezierControlPoint,
    BlendState,
    CameraKeyframe,
    CameraState,
    CameraTrack,
    CameraTrackManager,
    InterpolationMode,
    LookAtTarget,
    camera_track,
    catmull_rom_interpolate,
    catmull_rom_tangent,
    create_camera_track,
    create_track_from_class,
    cubic_bezier_interpolate,
    cubic_bezier_tangent,
    get_camera_track_registry,
    quat_look_at,
    quat_slerp,
    vec3_lerp,
)

from engine.animation.cinematics.cutscene import (
    Cutscene,
    CutsceneConfig,
    CutsceneEndEvent,
    CutsceneEvent,
    CutsceneEventExecuted,
    CutsceneEventType,
    CutsceneManager,
    CutscenePauseEvent,
    CutsceneResumeEvent,
    CutsceneSkipEvent,
    CutsceneStartEvent,
    CutsceneState,
    CutsceneTimeline,
    SkipPolicy,
    build_cutscene_from_class,
    create_cutscene,
    cutscene,
    get_cutscene_registry,
    get_registered_cutscene,
    register_cutscene,
)

__all__ = [
    # Camera Track Enums
    "InterpolationMode",
    "BlendState",
    # Camera Track Data classes
    "BezierControlPoint",
    "CameraKeyframe",
    "CameraState",
    "LookAtTarget",
    # Camera Track Main classes
    "CameraTrack",
    "CameraTrackManager",
    # Camera Track Decorator
    "camera_track",
    # Camera Track Factory functions
    "create_camera_track",
    "create_track_from_class",
    "get_camera_track_registry",
    # Camera Track Interpolation functions
    "catmull_rom_interpolate",
    "catmull_rom_tangent",
    "cubic_bezier_interpolate",
    "cubic_bezier_tangent",
    # Camera Track Vector/quat math
    "vec3_lerp",
    "quat_slerp",
    "quat_look_at",
    # Cutscene core classes
    "Cutscene",
    "CutsceneConfig",
    "CutsceneEvent",
    "CutsceneManager",
    "CutsceneTimeline",
    # Cutscene Enums
    "CutsceneEventType",
    "CutsceneState",
    "SkipPolicy",
    # Cutscene Events
    "CutsceneEndEvent",
    "CutsceneEventExecuted",
    "CutscenePauseEvent",
    "CutsceneResumeEvent",
    "CutsceneSkipEvent",
    "CutsceneStartEvent",
    # Cutscene Decorator
    "cutscene",
    # Cutscene Registry
    "get_cutscene_registry",
    "get_registered_cutscene",
    "register_cutscene",
    # Cutscene Helpers
    "build_cutscene_from_class",
    "create_cutscene",
]
