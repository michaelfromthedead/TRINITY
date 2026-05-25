"""Crowd animation subsystem: GPU crowd rendering, LOD, and behavior."""

from .animation_texture import (
    AnimationTexture,
    AnimationTextureAtlas,
    bake_clip_to_texture,
    encode_transform_to_pixels,
    decode_pixels_to_transform,
    TextureFormat,
)
from .crowd_renderer import (
    CrowdInstance,
    CrowdRenderer,
    CrowdRenderBatch,
    InstanceBuffer,
)
from .crowd_lod import (
    LODLevel,
    CrowdLOD,
    LODTransition,
    create_reduced_skeleton,
)
from .crowd_behavior import (
    CrowdAgent,
    CrowdBehavior,
    IdleBehavior,
    WalkingBehavior,
    WaitingBehavior,
    FleeingBehavior,
    FormationBehavior,
    CrowdSimulator,
    AgentState,
)

__all__ = [
    # Animation Texture
    "AnimationTexture",
    "AnimationTextureAtlas",
    "bake_clip_to_texture",
    "encode_transform_to_pixels",
    "decode_pixels_to_transform",
    "TextureFormat",
    # Crowd Renderer
    "CrowdInstance",
    "CrowdRenderer",
    "CrowdRenderBatch",
    "InstanceBuffer",
    # Crowd LOD
    "LODLevel",
    "CrowdLOD",
    "LODTransition",
    "create_reduced_skeleton",
    # Crowd Behavior
    "CrowdAgent",
    "CrowdBehavior",
    "IdleBehavior",
    "WalkingBehavior",
    "WaitingBehavior",
    "FleeingBehavior",
    "FormationBehavior",
    "CrowdSimulator",
    "AgentState",
]
