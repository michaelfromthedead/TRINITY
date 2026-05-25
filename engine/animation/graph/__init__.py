"""
Animation Graph subsystem.

This module provides the core animation graph functionality:
- Animation graphs with nodes, connections, and parameters
- Finite state machines for animation control flow
- Blend trees for parametric animation blending
- Individual blend nodes for specific animation operations
- Animation layering with bone masks
- Animation synchronization for smooth blending

Example usage:
    from engine.animation.graph import (
        AnimationGraph,
        StateMachine,
        BlendTree1D,
        ClipNode,
        LayerStack,
        SyncGroup,
    )

    # Create a simple animation graph
    graph = AnimationGraph("locomotion")

    # Create a 1D blend tree for speed-based blending
    blend_tree = BlendTree1D("speed_blend", "speed")
    blend_tree.add_entry(0.0, idle_clip)
    blend_tree.add_entry(2.0, walk_clip)
    blend_tree.add_entry(5.0, run_clip)

    # Add to graph and evaluate
    graph.add_node(blend_tree)
    graph.set_output_node("speed_blend")
    pose = graph.evaluate(context)
"""

# Core graph components
from .animation_graph import (
    # Metaclass
    GraphNodeMeta,
    # Slot types
    SlotType,
    InputSlot,
    OutputSlot,
    # Transform and Pose
    Transform,
    Pose,
    # Skeleton
    Bone,
    Skeleton,
    # Parameters
    ParameterType,
    GraphParameter,
    # Context
    GraphContext,
    ContextPool,
    # Nodes
    AnimationNode,
    SubgraphNode,
    # Graph
    Connection,
    AnimationGraph,
    # Cycle detection
    detect_cycles,
)

# Bone and Skeleton from skeleton.py (first-class contract surface;
# qualified names disambiguate from animation_graph.py's Bone/Skeleton)
from .skeleton import (
    Bone as SkeletonBone,
    Skeleton as SkeletonHierarchy,
)

# Bone mask system (name-based, integrates with skeleton.py's Skeleton)
from .bone_mask import (
    MissingBoneMode,
    BoneMask,
)

# State machine
from .state_machine import (
    # Blend curves
    BlendCurve,
    evaluate_blend_curve,
    # Conditions
    ComparisonOp,
    TransitionCondition,
    # State
    AnimationState,
    # Transition
    TransitionSyncMode,
    StateTransition,
    ActiveTransition,
    # State machine
    StateMachine,
    StateMachineBuilder,
    # Decorator
    state_machine,
)

# Blend trees
from .blend_tree import (
    # Base
    BlendTree,
    # 1D
    BlendTree1DEntry,
    BlendTree1D,
    # 2D
    BlendTree2DMode,
    BlendTree2DSample,
    Triangle,
    BlendTree2D,
    # Direct
    BlendTreeDirectEntry,
    BlendTreeDirect,
    # Decorator
    blend_tree,
)

# Blend nodes
from .blend_node import (
    # Clip
    AnimationKeyframe,
    AnimationTrack,
    LoopMode,
    AnimationClip,
    # Nodes
    ClipNode,
    BlendNode,
    AdditiveNode,
    LayerBlendMode,
    AnimationLayerInput,
    LayerNode,
    BoneMirrorPair,
    MirrorNode,
    TimeScaleNode,
    PoseCacheNode,
    SelectNode,
)

# Layer system
from .layer import (
    # Blend mode (re-export with different name to avoid conflict)
    LayerBlendMode as LayerMode,
    # Layer
    AnimationLayer,
    # Stack
    LayerStack,
    LayerStackBuilder,
)

# Bone mask presets (name-based, integrates with skeleton.py's Skeleton)
from .bone_mask import (
    BoneMaskPresets,
)

# Synchronization
from .sync import (
    # Markers
    SyncMarker,
    SyncMarkerTrack,
    # Sync
    SyncMode,
    SyncEntry,
    SyncGroup,
    SyncGroupNode,
    # Utilities
    sync_animations,
    create_locomotion_markers,
    calculate_phase_offset,
    # Events
    SyncEvent,
    EventSynchronizer,
)

# Configuration
from .config import (
    AnimationGraphConfig,
    TransitionConfig,
    BlendTreeConfig,
    SyncConfig,
    LayerConfig,
    QuaternionConfig,
    BlendConfig,
    get_config,
)


__all__ = [
    # animation_graph.py
    "GraphNodeMeta",
    "SlotType",
    "InputSlot",
    "OutputSlot",
    "Transform",
    "Pose",
    "Bone",
    "Skeleton",
    "ParameterType",
    "GraphParameter",
    "GraphContext",
    "ContextPool",
    "AnimationNode",
    "SubgraphNode",
    "Connection",
    "AnimationGraph",
    "detect_cycles",
    # skeleton.py
    "SkeletonBone",
    "SkeletonHierarchy",
    # bone_mask.py
    "MissingBoneMode",
    "BoneMask",
    "BoneMaskPresets",
    # state_machine.py
    "BlendCurve",
    "evaluate_blend_curve",
    "ComparisonOp",
    "TransitionCondition",
    "AnimationState",
    "TransitionSyncMode",
    "StateTransition",
    "ActiveTransition",
    "StateMachine",
    "StateMachineBuilder",
    "state_machine",
    # blend_tree.py
    "BlendTree",
    "BlendTree1DEntry",
    "BlendTree1D",
    "BlendTree2DMode",
    "BlendTree2DSample",
    "Triangle",
    "BlendTree2D",
    "BlendTreeDirectEntry",
    "BlendTreeDirect",
    "blend_tree",
    # blend_node.py
    "AnimationKeyframe",
    "AnimationTrack",
    "LoopMode",
    "AnimationClip",
    "ClipNode",
    "BlendNode",
    "AdditiveNode",
    "LayerBlendMode",
    "AnimationLayerInput",
    "LayerNode",
    "BoneMirrorPair",
    "MirrorNode",
    "TimeScaleNode",
    "PoseCacheNode",
    "SelectNode",
    # layer.py
    "LayerMode",
    "AnimationLayer",
    "LayerStack",
    "LayerStackBuilder",
    "BoneMaskPresets",
    # sync.py
    "SyncMarker",
    "SyncMarkerTrack",
    "SyncMode",
    "SyncEntry",
    "SyncGroup",
    "SyncGroupNode",
    "sync_animations",
    "create_locomotion_markers",
    "calculate_phase_offset",
    "SyncEvent",
    "EventSynchronizer",
    # config.py
    "AnimationGraphConfig",
    "TransitionConfig",
    "BlendTreeConfig",
    "SyncConfig",
    "LayerConfig",
    "QuaternionConfig",
    "BlendConfig",
    "get_config",
]
