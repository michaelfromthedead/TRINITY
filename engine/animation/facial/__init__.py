"""
Facial Animation Module.

Provides comprehensive facial animation including:
- Blend shapes / morph targets
- FACS (Facial Action Coding System)
- Lip synchronization
- Eye animation and tracking
- Complete face rig integration
- Motion capture playback and retargeting
"""

from .blend_shapes import (
    ARKIT_BLEND_SHAPES,
    BlendShape,
    BlendShapeController,
    BlendShapeSet,
    CorrectiveBlendShape,
    CorrectiveShape,
    apply_arkit_data,
    apply_blend_shape,
    apply_blend_shapes,
    apply_blend_shapes_with_correctives,
    apply_corrective_shape,
    create_arkit_compatible_set,
    remap_blend_shape_weights,
    validate_arkit_data,
)
from .eye_animation import (
    BlinkController,
    BlinkSettings,
    EyeController,
    EyeLimits,
    EyeState,
    EyeTransform,
    PupilSettings,
    SaccadeSettings,
)
from .face_capture import (
    AnimationCurve,
    FaceCaptureClip,
    FaceCapturePlayer,
    FaceCaptureRetargeter,
    InterpolationMode,
    Keyframe,
    PlaybackState,
    RetargetMapping,
    create_clip_from_samples,
    merge_clips,
)
from .face_rig import (
    AnimationLayer,
    AnimationPriority,
    EmotionState,
    FaceRig,
    LayerPriority,
    RigLayer,
    create_face_rig,
)
from .facs import (
    AU,
    ActionUnit,
    ActionUnitData,
    Expression,
    ExpressionData,
    FACSController,
    get_default_au_mappings,
    get_default_expressions,
)
from .lip_sync import (
    CoarticulationSettings,
    LipSyncController,
    PhonemeEvent,
    Viseme,
    VisemeEvent,
    VisemeMapping,
    apply_coarticulation,
    create_phoneme_events_from_text,
    get_default_viseme_mappings,
    phoneme_to_viseme,
    PHONEME_TO_VISEME,
)


__all__ = [
    # Blend Shapes
    "BlendShape",
    "BlendShapeSet",
    "BlendShapeController",
    "CorrectiveBlendShape",
    "CorrectiveShape",
    "apply_arkit_data",
    "apply_blend_shape",
    "apply_blend_shapes",
    "apply_blend_shapes_with_correctives",
    "apply_corrective_shape",
    "ARKIT_BLEND_SHAPES",
    "create_arkit_compatible_set",
    "remap_blend_shape_weights",
    "validate_arkit_data",

    # FACS
    "ActionUnit",
    "AU",
    "ActionUnitData",
    "Expression",
    "ExpressionData",
    "FACSController",
    "get_default_au_mappings",
    "get_default_expressions",

    # Lip Sync
    "Viseme",
    "VisemeMapping",
    "VisemeEvent",
    "PhonemeEvent",
    "CoarticulationSettings",
    "LipSyncController",
    "phoneme_to_viseme",
    "PHONEME_TO_VISEME",
    "get_default_viseme_mappings",
    "apply_coarticulation",
    "create_phoneme_events_from_text",

    # Eye Animation
    "EyeLimits",
    "BlinkSettings",
    "SaccadeSettings",
    "PupilSettings",
    "EyeState",
    "EyeTransform",
    "BlinkController",
    "EyeController",

    # Face Rig
    "AnimationPriority",
    "AnimationLayer",
    "EmotionState",
    "FaceRig",
    "LayerPriority",
    "RigLayer",
    "create_face_rig",

    # Face Capture
    "InterpolationMode",
    "Keyframe",
    "AnimationCurve",
    "FaceCaptureClip",
    "PlaybackState",
    "FaceCapturePlayer",
    "RetargetMapping",
    "FaceCaptureRetargeter",
    "create_clip_from_samples",
    "merge_clips",
]
