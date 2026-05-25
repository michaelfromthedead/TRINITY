"""Animation Tools subsystem for the AI Game Engine tooling layer.

This module provides comprehensive animation editing tools including:
- Sequencer: Timeline-based animation editing with multiple track types
- Curve Editor: Animation curves with tangent editing and easing functions
- Skeleton Editor: Skeleton hierarchy editing and retargeting setup
- Montage Editor: Animation montage creation with sections and notifies
- Anim Graph Editor: Visual animation graph editing (state machines, blend trees)
- Pose Editor: Pose library, pose blending, and additive poses
- IK Setup: Inverse kinematics chain setup and solver configuration
- Notifies Editor: Animation notify events with timing
- Preview Scene: Animation preview with ground, lighting, and props

All tools integrate with the Foundation layer for undo/redo, dirty tracking,
and live editing support.
"""

from .sequencer import (
    # Track types
    TrackType,
    AnimationTrack,
    TransformTrack,
    SkeletalTrack,
    CameraTrack,
    EventTrack,
    AudioTrack,
    PropertyTrack,
    # Keyframes
    Keyframe,
    TransformKeyframe,
    EventKeyframe,
    PropertyKeyframe,
    # Timeline
    Timeline,
    TimelineRange,
    # Sequencer
    AnimationSequencer,
    SequencerPlayback,
    PlaybackMode,
)

from .curve_editor import (
    # Curve types
    CurveType,
    TangentMode,
    TangentHandle,
    # Curve keys
    CurveKey,
    # Curves
    AnimationCurve,
    BezierCurve,
    HermiteCurve,
    LinearCurve,
    SteppedCurve,
    # Easing
    EasingFunction,
    EasingType,
    # Curve editor
    CurveEditor,
    CurveSelection,
)

from .skeleton_editor import (
    # Bone editing
    BoneEditMode,
    BoneSelection,
    # Sockets
    Socket,
    SocketAttachment,
    # Virtual bones
    VirtualBone,
    VirtualBoneType,
    # Retargeting
    RetargetSource,
    RetargetMapping,
    BoneMirrorPair,
    # Editor
    SkeletonEditor,
    SkeletonPreview,
)

from .montage_editor import (
    # Sections
    MontageSection,
    SectionLink,
    SectionLoopConfig,
    # Slots
    AnimSlot,
    SlotGroup,
    # Montage
    AnimMontage,
    MontageBlendSettings,
    # Editor
    MontageEditor,
    MontagePreview,
)

from .anim_graph_editor import (
    # Graph elements
    GraphNode,
    GraphConnection,
    GraphSocket,
    SocketType,
    # State machine nodes
    StateNode,
    TransitionNode,
    ConduitNode,
    EntryNode,
    # Blend tree nodes
    BlendNode,
    BlendType,
    AdditiveNode,
    LayeredBlendNode,
    BoneMaskNode,
    BlendSpaceNode,
    # Blend spaces
    BlendSpace1D,
    BlendSpace2D,
    BlendSample,
    # Editor
    AnimGraphEditor,
    AnimGraphPreview,
    NodePalette,
)

from .pose_editor import (
    # Poses
    AnimPose,
    PoseType,
    PoseBlendMode,
    # Pose library
    PoseLibrary,
    PoseCategory,
    PoseAsset,
    # Additive poses
    AdditivePose,
    AdditiveType,
    # Editor
    PoseEditor,
    PosePreview,
    PoseMirrorSettings,
)

from .ik_setup import (
    # IK types
    IKChainType,
    IKSolverType,
    # Chain
    IKChain,
    IKBone,
    IKEffector,
    # Solvers
    IKSolverConfig,
    TwoBoneSolverConfig,
    FABRIKSolverConfig,
    CCDSolverConfig,
    FullBodySolverConfig,
    # Constraints
    IKConstraint,
    IKConstraintType,
    IKPoleVector,
    # Editor
    IKSetupEditor,
    IKPreview,
)

from .notifies_editor import (
    # Notify types
    NotifyType,
    # Notifies
    AnimNotify,
    AnimNotifyState,
    NotifyPayload,
    # Built-in notifies
    SoundNotify,
    ParticleNotify,
    CustomEventNotify,
    FootstepNotify,
    # Editor
    NotifiesEditor,
    NotifyTrack,
    NotifyTiming,
)

from .preview_scene import (
    # Preview settings
    PreviewSettings,
    GroundSettings,
    LightingSettings,
    CameraSettings,
    # Props
    PreviewProp,
    PropType,
    # Scene
    PreviewScene,
    PreviewViewport,
    PreviewPlayback,
)

__all__ = [
    # Sequencer
    "TrackType",
    "AnimationTrack",
    "TransformTrack",
    "SkeletalTrack",
    "CameraTrack",
    "EventTrack",
    "AudioTrack",
    "PropertyTrack",
    "Keyframe",
    "TransformKeyframe",
    "EventKeyframe",
    "PropertyKeyframe",
    "Timeline",
    "TimelineRange",
    "AnimationSequencer",
    "SequencerPlayback",
    "PlaybackMode",
    # Curve editor
    "CurveType",
    "TangentMode",
    "TangentHandle",
    "CurveKey",
    "AnimationCurve",
    "BezierCurve",
    "HermiteCurve",
    "LinearCurve",
    "SteppedCurve",
    "EasingFunction",
    "EasingType",
    "CurveEditor",
    "CurveSelection",
    # Skeleton editor
    "BoneEditMode",
    "BoneSelection",
    "Socket",
    "SocketAttachment",
    "VirtualBone",
    "VirtualBoneType",
    "RetargetSource",
    "RetargetMapping",
    "BoneMirrorPair",
    "SkeletonEditor",
    "SkeletonPreview",
    # Montage editor
    "MontageSection",
    "SectionLink",
    "SectionLoopConfig",
    "AnimSlot",
    "SlotGroup",
    "AnimMontage",
    "MontageBlendSettings",
    "MontageEditor",
    "MontagePreview",
    # Anim graph editor
    "GraphNode",
    "GraphConnection",
    "GraphSocket",
    "SocketType",
    "StateNode",
    "TransitionNode",
    "ConduitNode",
    "EntryNode",
    "BlendNode",
    "BlendType",
    "AdditiveNode",
    "LayeredBlendNode",
    "BoneMaskNode",
    "BlendSpaceNode",
    "BlendSpace1D",
    "BlendSpace2D",
    "BlendSample",
    "AnimGraphEditor",
    "AnimGraphPreview",
    "NodePalette",
    # Pose editor
    "AnimPose",
    "PoseType",
    "PoseBlendMode",
    "PoseLibrary",
    "PoseCategory",
    "PoseAsset",
    "AdditivePose",
    "AdditiveType",
    "PoseEditor",
    "PosePreview",
    "PoseMirrorSettings",
    # IK setup
    "IKChainType",
    "IKSolverType",
    "IKChain",
    "IKBone",
    "IKEffector",
    "IKSolverConfig",
    "TwoBoneSolverConfig",
    "FABRIKSolverConfig",
    "CCDSolverConfig",
    "FullBodySolverConfig",
    "IKConstraint",
    "IKConstraintType",
    "IKPoleVector",
    "IKSetupEditor",
    "IKPreview",
    # Notifies editor
    "NotifyType",
    "AnimNotify",
    "AnimNotifyState",
    "NotifyPayload",
    "SoundNotify",
    "ParticleNotify",
    "CustomEventNotify",
    "FootstepNotify",
    "NotifiesEditor",
    "NotifyTrack",
    "NotifyTiming",
    # Preview scene
    "PreviewSettings",
    "GroundSettings",
    "LightingSettings",
    "CameraSettings",
    "PreviewProp",
    "PropType",
    "PreviewScene",
    "PreviewViewport",
    "PreviewPlayback",
]
