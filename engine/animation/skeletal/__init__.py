"""Skeletal animation subsystem.

This package provides:
- skeleton: Bone hierarchy and skeleton representation
- pose: Animation pose representation
- clip: Animation clip data with keyframes
- clip_player: Animation playback control
- blending: Pose blending utilities and masks
- skinning: Mesh skinning (LBS, DQS, GPU preparation)
- root_motion: Root motion extraction and application
- retargeting: Animation retargeting between skeletons
- compression: Animation data compression
"""

# Core skeletal animation
from engine.animation.skeletal.skeleton import (
    Bone,
    Skeleton,
    AnimationMeta,
    animation_data,
    create_humanoid_skeleton,
)

from engine.animation.skeletal.pose import (
    BoneTransform,
    Pose,
    PoseSpace,
    PoseBuffer,
    lerp_poses,
    additive_blend,
    compute_additive_pose,
    blend_multiple_poses,
)

from engine.animation.skeletal.clip import (
    Keyframe as ClipKeyframe,
    AnimationCurve,
    AnimationEvent,
    BoneTrack,
    AnimationClip,
    InterpolationType,
    create_simple_clip,
)

from engine.animation.skeletal.clip_player import (
    ClipPlayer,
    ClipQueue,
    CrossfadePlayer,
    PlaybackMode,
    PlaybackState,
    PlaybackEvent,
    EventCallback,
)

from engine.animation.skeletal.blending import (
    BlendMode,
    BoneMask,
    LayeredBlender,
    PoseCache,
    blend_poses,
    blend_multiple_poses as blend_poses_weighted,
    compute_additive_pose as compute_delta_pose,
    apply_additive_pose,
)

# Skinning and other modules
from engine.animation.skeletal.skinning import (
    SkinningMethod,
    VertexWeight,
    SkinningData,
    DualQuaternion,
    LinearBlendSkinning,
    DualQuaternionSkinning,
    GPUSkinningData,
    SkinningCache,
    prepare_gpu_skinning_data,
    skin_mesh,
)

from engine.animation.skeletal.root_motion import (
    RootMotionMode,
    RootMotionData,
    RootBoneTransform,
    RootMotionAccumulator,
    RootMotionConfig,
    RootMotionBlender,
    extract_root_motion,
    apply_root_motion,
    blend_root_motion,
)

from engine.animation.skeletal.retargeting import (
    BoneMappingStrategy,
    BoneMapping,
    RetargetMap,
    RetargetConfig,
    SkeletonInfo,
    PoseData,
    RetargetPipeline,
    create_retarget_map,
    retarget_pose,
    compute_scale_factor,
    preserve_foot_contact,
    validate_retarget_map,
)

from engine.animation.skeletal.compression import (
    CompressionMethod,
    TrackType,
    CompressionSettings,
    Keyframe,
    AnimationTrack,
    QuantizedValue,
    QuantizedCurve,
    CompressedTrack,
    CompressedClip,
    CompressionErrorMetrics,
    AnimationClipData,
    compress_clip,
    decompress_clip,
    decompress_track,
    compute_compression_error,
    estimate_compressed_size,
)

from engine.animation.skeletal.constants import (
    WEIGHT_NORMALIZATION_EPSILON,
    DEFAULT_BONE_WEIGHT,
    MAX_BONE_INFLUENCES,
    COMPRESSION_BITS_LOW,
    COMPRESSION_BITS_MEDIUM,
    COMPRESSION_BITS_HIGH,
    DEFAULT_TRANSLATION_BITS,
    DEFAULT_ROTATION_BITS,
    DEFAULT_SCALE_BITS,
    DEFAULT_TRANSLATION_ERROR_THRESHOLD,
    DEFAULT_ROTATION_ERROR_THRESHOLD,
    DEFAULT_SCALE_ERROR_THRESHOLD,
    DEFAULT_CURVE_FITTING_TOLERANCE,
    RETARGET_POSITION_MATCH_THRESHOLD,
    DEFAULT_UNMAPPED_BLEND_FACTOR,
    DEFAULT_FRAME_RATE,
)

__all__ = [
    # Skeleton
    "Bone",
    "Skeleton",
    "AnimationMeta",
    "animation_data",
    "create_humanoid_skeleton",
    # Pose
    "BoneTransform",
    "Pose",
    "PoseSpace",
    "PoseBuffer",
    "lerp_poses",
    "additive_blend",
    "compute_additive_pose",
    "blend_multiple_poses",
    # Clip
    "ClipKeyframe",
    "AnimationCurve",
    "AnimationEvent",
    "BoneTrack",
    "AnimationClip",
    "InterpolationType",
    "create_simple_clip",
    # Player
    "ClipPlayer",
    "ClipQueue",
    "CrossfadePlayer",
    "PlaybackMode",
    "PlaybackState",
    "PlaybackEvent",
    "EventCallback",
    # Blending
    "BlendMode",
    "BoneMask",
    "LayeredBlender",
    "PoseCache",
    "blend_poses",
    "blend_poses_weighted",
    "compute_delta_pose",
    "apply_additive_pose",
    # Skinning
    "SkinningMethod",
    "VertexWeight",
    "SkinningData",
    "DualQuaternion",
    "LinearBlendSkinning",
    "DualQuaternionSkinning",
    "GPUSkinningData",
    "SkinningCache",
    "prepare_gpu_skinning_data",
    "skin_mesh",
    # Root Motion
    "RootMotionMode",
    "RootMotionData",
    "RootBoneTransform",
    "RootMotionAccumulator",
    "RootMotionConfig",
    "RootMotionBlender",
    "extract_root_motion",
    "apply_root_motion",
    "blend_root_motion",
    # Retargeting
    "BoneMappingStrategy",
    "BoneMapping",
    "RetargetMap",
    "RetargetConfig",
    "SkeletonInfo",
    "PoseData",
    "RetargetPipeline",
    "create_retarget_map",
    "retarget_pose",
    "compute_scale_factor",
    "preserve_foot_contact",
    "validate_retarget_map",
    # Compression
    "CompressionMethod",
    "TrackType",
    "CompressionSettings",
    "Keyframe",
    "AnimationTrack",
    "QuantizedValue",
    "QuantizedCurve",
    "CompressedTrack",
    "CompressedClip",
    "CompressionErrorMetrics",
    "AnimationClipData",
    "compress_clip",
    "decompress_clip",
    "decompress_track",
    "compute_compression_error",
    "estimate_compressed_size",
    # Constants
    "WEIGHT_NORMALIZATION_EPSILON",
    "DEFAULT_BONE_WEIGHT",
    "MAX_BONE_INFLUENCES",
    "COMPRESSION_BITS_LOW",
    "COMPRESSION_BITS_MEDIUM",
    "COMPRESSION_BITS_HIGH",
    "DEFAULT_TRANSLATION_BITS",
    "DEFAULT_ROTATION_BITS",
    "DEFAULT_SCALE_BITS",
    "DEFAULT_TRANSLATION_ERROR_THRESHOLD",
    "DEFAULT_ROTATION_ERROR_THRESHOLD",
    "DEFAULT_SCALE_ERROR_THRESHOLD",
    "DEFAULT_CURVE_FITTING_TOLERANCE",
    "RETARGET_POSITION_MATCH_THRESHOLD",
    "DEFAULT_UNMAPPED_BLEND_FACTOR",
    "DEFAULT_FRAME_RATE",
]
