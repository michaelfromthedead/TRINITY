"""
Physics-Animation Blending System.

Provides blending between physics simulation and animation data,
supporting various blend modes for hit reactions, procedural effects,
and dynamic character motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .character_controller import Quaternion, Transform, Vector3
from .config import (
    BLEND_ADDITIVE,
    BLEND_CHAIN,
    BLEND_POSE,
    DEFAULT_BLEND_WEIGHT,
    HIT_REACTION_BLEND_IN_MS,
    HIT_REACTION_BLEND_OUT_MS,
    BlendMode,
)


# =============================================================================
# Blend Data Structures
# =============================================================================

@dataclass
class BonePose:
    """
    Pose data for a single bone.

    Attributes:
        position: Local position
        rotation: Local rotation
        scale: Local scale
    """
    position: Vector3 = field(default_factory=Vector3.zero)
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    scale: Vector3 = field(default_factory=Vector3.one)

    def lerp(self, other: BonePose, t: float) -> BonePose:
        """Linearly interpolate between poses."""
        return BonePose(
            position=Vector3.lerp(self.position, other.position, t),
            rotation=self._slerp(self.rotation, other.rotation, t),
            scale=Vector3.lerp(self.scale, other.scale, t),
        )

    def _slerp(self, a: Quaternion, b: Quaternion, t: float) -> Quaternion:
        """Spherical linear interpolation between quaternions."""
        # Compute cosine of angle
        dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w

        # If negative, negate one quaternion
        if dot < 0:
            b = Quaternion(-b.x, -b.y, -b.z, -b.w)
            dot = -dot

        # Clamp dot for acos
        dot = min(1.0, max(-1.0, dot))

        # If very close, use linear interpolation
        if dot > 0.9995:
            result = Quaternion(
                a.x + t * (b.x - a.x),
                a.y + t * (b.y - a.y),
                a.z + t * (b.z - a.z),
                a.w + t * (b.w - a.w),
            )
            # Normalize
            mag = math.sqrt(
                result.x ** 2 + result.y ** 2 + result.z ** 2 + result.w ** 2
            )
            return Quaternion(
                result.x / mag, result.y / mag, result.z / mag, result.w / mag
            )

        # Slerp
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        wa = math.sin((1 - t) * theta) / sin_theta
        wb = math.sin(t * theta) / sin_theta

        return Quaternion(
            wa * a.x + wb * b.x,
            wa * a.y + wb * b.y,
            wa * a.z + wb * b.z,
            wa * a.w + wb * b.w,
        )


@dataclass
class SkeletonPose:
    """
    Complete skeleton pose.

    Attributes:
        bones: Dictionary mapping bone names to poses
        root_motion: Root motion delta
    """
    bones: dict[str, BonePose] = field(default_factory=dict)
    root_motion: Vector3 = field(default_factory=Vector3.zero)
    root_rotation_delta: Quaternion = field(default_factory=Quaternion.identity)


@dataclass
class BlendLayer:
    """
    A layer in the blend stack.

    Attributes:
        name: Layer identifier
        mode: Blend mode
        weight: Blend weight (0-1)
        mask: Bone mask for partial blending
        source: Source pose or callback
    """
    name: str = ""
    mode: BlendMode = BlendMode.POSE
    weight: float = 1.0
    mask: Optional[dict[str, float]] = None
    source: Optional[SkeletonPose] = None
    enabled: bool = True


@dataclass
class HitReaction:
    """
    Data for a hit reaction.

    Attributes:
        hit_point: World position of hit
        hit_direction: Direction of impact
        hit_force: Force magnitude
        affected_bones: Bones affected by this hit
        start_time: When hit occurred
        blend_weight: Current blend weight
    """
    hit_point: Vector3 = field(default_factory=Vector3.zero)
    hit_direction: Vector3 = field(default_factory=Vector3.zero)
    hit_force: float = 0.0
    affected_bones: list[str] = field(default_factory=list)
    start_time: float = 0.0
    blend_weight: float = 0.0
    pose_delta: Optional[SkeletonPose] = None


# =============================================================================
# Physics Animation Blender
# =============================================================================

class PhysicsAnimationBlender:
    """
    Blends physics simulation results with animation data.

    Features:
    - Multiple blend modes (pose, additive, chain)
    - Per-limb blend weights
    - Hit reaction integration
    - Smooth transitions
    """

    def __init__(self):
        # Layer stack
        self._layers: list[BlendLayer] = []

        # Per-bone blend weights
        self._limb_weights: dict[str, float] = {}
        self._default_weight = DEFAULT_BLEND_WEIGHT

        # Hit reactions
        self._hit_reactions: list[HitReaction] = []
        self._max_hit_reactions = 4

        # Bone hierarchy (for chain blending)
        self._bone_hierarchy: dict[str, list[str]] = {}
        self._bone_parent: dict[str, str] = {}

        # Output
        self._output_pose: Optional[SkeletonPose] = None

        # Callbacks
        self._on_blend_complete: Optional[Callable[[str], None]] = None

    # -------------------------------------------------------------------------
    # Layer Management
    # -------------------------------------------------------------------------

    def add_layer(
        self,
        name: str,
        mode: BlendMode = BlendMode.POSE,
        weight: float = 1.0,
        mask: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Add a blend layer.

        Args:
            name: Layer identifier
            mode: Blend mode for this layer
            weight: Layer weight
            mask: Optional bone mask
        """
        layer = BlendLayer(name=name, mode=mode, weight=weight, mask=mask)
        self._layers.append(layer)

    def remove_layer(self, name: str) -> bool:
        """Remove a layer by name."""
        for i, layer in enumerate(self._layers):
            if layer.name == name:
                self._layers.pop(i)
                return True
        return False

    def set_layer_weight(self, name: str, weight: float) -> None:
        """Set the weight of a layer."""
        # Clamp weight to valid range [0, 1]
        clamped_weight = max(0.0, min(1.0, weight))
        for layer in self._layers:
            if layer.name == name:
                layer.weight = clamped_weight
                break

    def set_layer_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a layer."""
        for layer in self._layers:
            if layer.name == name:
                layer.enabled = enabled
                break

    def set_layer_pose(self, name: str, pose: SkeletonPose) -> None:
        """Set the source pose for a layer."""
        for layer in self._layers:
            if layer.name == name:
                layer.source = pose
                break

    # -------------------------------------------------------------------------
    # Per-Limb Weights
    # -------------------------------------------------------------------------

    @property
    def per_limb_blend_weights(self) -> dict[str, float]:
        """Per-bone blend weights."""
        return self._limb_weights.copy()

    def set_bone_weight(self, bone_name: str, weight: float) -> None:
        """Set blend weight for a specific bone."""
        self._limb_weights[bone_name] = max(0.0, min(1.0, weight))

    def set_limb_weight(self, limb_bones: list[str], weight: float) -> None:
        """Set blend weight for a group of bones."""
        for bone in limb_bones:
            self._limb_weights[bone] = max(0.0, min(1.0, weight))

    def clear_bone_weights(self) -> None:
        """Clear all per-bone weights."""
        self._limb_weights.clear()

    def get_bone_weight(self, bone_name: str) -> float:
        """Get blend weight for a bone."""
        return self._limb_weights.get(bone_name, self._default_weight)

    # -------------------------------------------------------------------------
    # Bone Hierarchy
    # -------------------------------------------------------------------------

    def set_bone_hierarchy(
        self,
        hierarchy: dict[str, list[str]],
        parents: dict[str, str],
    ) -> None:
        """
        Set the bone hierarchy for chain blending.

        Args:
            hierarchy: Mapping of bones to their children
            parents: Mapping of bones to their parent
        """
        self._bone_hierarchy = hierarchy.copy()
        self._bone_parent = parents.copy()

    def get_bone_chain(self, start_bone: str, end_bone: str) -> list[str]:
        """Get chain of bones from start to end."""
        chain = []
        current = end_bone

        while current and current != start_bone:
            chain.append(current)
            current = self._bone_parent.get(current)

        if current == start_bone:
            chain.append(start_bone)

        return list(reversed(chain))

    # -------------------------------------------------------------------------
    # Pose Blending
    # -------------------------------------------------------------------------

    def blend_poses(
        self,
        anim_pose: SkeletonPose,
        physics_pose: SkeletonPose,
        weight: float,
    ) -> SkeletonPose:
        """
        Blend between animation and physics poses.

        Args:
            anim_pose: Pose from animation system
            physics_pose: Pose from physics simulation
            weight: Blend weight (0 = anim, 1 = physics)

        Returns:
            Blended pose
        """
        result = SkeletonPose()
        weight = max(0.0, min(1.0, weight))

        # Get all bones from both poses
        all_bones = set(anim_pose.bones.keys()) | set(physics_pose.bones.keys())

        for bone_name in all_bones:
            anim_bone = anim_pose.bones.get(bone_name, BonePose())
            physics_bone = physics_pose.bones.get(bone_name, BonePose())

            # Apply per-bone weight
            bone_weight = weight * self.get_bone_weight(bone_name)

            # Interpolate
            result.bones[bone_name] = anim_bone.lerp(physics_bone, bone_weight)

        # Blend root motion
        result.root_motion = Vector3.lerp(
            anim_pose.root_motion, physics_pose.root_motion, weight
        )

        return result

    def additive_physics(
        self,
        base_pose: SkeletonPose,
        physics_delta: SkeletonPose,
        weight: float = 1.0,
    ) -> SkeletonPose:
        """
        Add physics delta on top of animation.

        Args:
            base_pose: Base animation pose
            physics_delta: Physics-based delta pose
            weight: Weight for additive blend

        Returns:
            Pose with physics added
        """
        result = SkeletonPose()
        weight = max(0.0, min(1.0, weight))

        for bone_name, base_bone in base_pose.bones.items():
            delta_bone = physics_delta.bones.get(bone_name)

            if delta_bone:
                bone_weight = weight * self.get_bone_weight(bone_name)

                # Add weighted delta
                result.bones[bone_name] = BonePose(
                    position=base_bone.position + delta_bone.position * bone_weight,
                    rotation=self._multiply_quaternion_blend(
                        base_bone.rotation, delta_bone.rotation, bone_weight
                    ),
                    scale=base_bone.scale,
                )
            else:
                result.bones[bone_name] = base_bone

        # Add root motion
        result.root_motion = base_pose.root_motion + physics_delta.root_motion * weight

        return result

    def _multiply_quaternion_blend(
        self,
        base: Quaternion,
        delta: Quaternion,
        weight: float,
    ) -> Quaternion:
        """Multiply quaternion with weighted delta."""
        # Interpolate delta with identity
        identity = Quaternion.identity()
        weighted_delta = self._slerp(identity, delta, weight)

        # Multiply
        return Quaternion(
            base.w * weighted_delta.x + base.x * weighted_delta.w +
            base.y * weighted_delta.z - base.z * weighted_delta.y,
            base.w * weighted_delta.y - base.x * weighted_delta.z +
            base.y * weighted_delta.w + base.z * weighted_delta.x,
            base.w * weighted_delta.z + base.x * weighted_delta.y -
            base.y * weighted_delta.x + base.z * weighted_delta.w,
            base.w * weighted_delta.w - base.x * weighted_delta.x -
            base.y * weighted_delta.y - base.z * weighted_delta.z,
        )

    def _slerp(self, a: Quaternion, b: Quaternion, t: float) -> Quaternion:
        """Spherical linear interpolation."""
        dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w

        if dot < 0:
            b = Quaternion(-b.x, -b.y, -b.z, -b.w)
            dot = -dot

        if dot > 0.9995:
            result = Quaternion(
                a.x + t * (b.x - a.x),
                a.y + t * (b.y - a.y),
                a.z + t * (b.z - a.z),
                a.w + t * (b.w - a.w),
            )
            mag = math.sqrt(
                result.x ** 2 + result.y ** 2 + result.z ** 2 + result.w ** 2
            )
            return Quaternion(
                result.x / mag, result.y / mag, result.z / mag, result.w / mag
            )

        theta = math.acos(min(1.0, max(-1.0, dot)))
        sin_theta = math.sin(theta)
        wa = math.sin((1 - t) * theta) / sin_theta
        wb = math.sin(t * theta) / sin_theta

        return Quaternion(
            wa * a.x + wb * b.x,
            wa * a.y + wb * b.y,
            wa * a.z + wb * b.z,
            wa * a.w + wb * b.w,
        )

    # -------------------------------------------------------------------------
    # Chain Blending
    # -------------------------------------------------------------------------

    def blend_chain(
        self,
        base_pose: SkeletonPose,
        physics_pose: SkeletonPose,
        chain_root: str,
        weight: float = 1.0,
    ) -> SkeletonPose:
        """
        Blend physics for an entire bone chain.

        Args:
            base_pose: Base animation pose
            physics_pose: Physics pose
            chain_root: Root bone of chain to blend
            weight: Blend weight

        Returns:
            Blended pose
        """
        result = SkeletonPose()

        # Copy all bones from base
        for name, bone in base_pose.bones.items():
            result.bones[name] = BonePose(
                position=Vector3(bone.position.x, bone.position.y, bone.position.z),
                rotation=Quaternion(
                    bone.rotation.x, bone.rotation.y,
                    bone.rotation.z, bone.rotation.w
                ),
                scale=Vector3(bone.scale.x, bone.scale.y, bone.scale.z),
            )

        # Get all bones in chain
        chain_bones = self._get_chain_bones(chain_root)

        # Blend chain bones
        for bone_name in chain_bones:
            base_bone = base_pose.bones.get(bone_name, BonePose())
            physics_bone = physics_pose.bones.get(bone_name, BonePose())

            bone_weight = weight * self.get_bone_weight(bone_name)
            result.bones[bone_name] = base_bone.lerp(physics_bone, bone_weight)

        return result

    def _get_chain_bones(self, root: str) -> list[str]:
        """Get all bones in a chain starting from root."""
        bones = [root]
        children = self._bone_hierarchy.get(root, [])

        for child in children:
            bones.extend(self._get_chain_bones(child))

        return bones

    # -------------------------------------------------------------------------
    # Hit Reactions
    # -------------------------------------------------------------------------

    def add_hit_reaction(
        self,
        hit_point: Vector3,
        hit_direction: Vector3,
        hit_force: float,
        affected_bones: list[str],
        current_time: float,
    ) -> None:
        """
        Add a hit reaction.

        Args:
            hit_point: World position of hit
            hit_direction: Direction of impact
            hit_force: Force magnitude
            affected_bones: Bones to affect
            current_time: Current game time
        """
        # Remove oldest if at limit
        if len(self._hit_reactions) >= self._max_hit_reactions:
            self._hit_reactions.pop(0)

        reaction = HitReaction(
            hit_point=hit_point,
            hit_direction=hit_direction,
            hit_force=hit_force,
            affected_bones=affected_bones,
            start_time=current_time,
            blend_weight=0.0,
        )

        self._hit_reactions.append(reaction)

    def update_hit_reactions(
        self,
        base_pose: SkeletonPose,
        current_time: float,
        dt: float,
    ) -> SkeletonPose:
        """
        Update and apply hit reactions.

        Args:
            base_pose: Base animation pose
            current_time: Current game time
            dt: Delta time

        Returns:
            Pose with hit reactions applied
        """
        result = SkeletonPose()

        # Copy base pose
        for name, bone in base_pose.bones.items():
            result.bones[name] = BonePose(
                position=Vector3(bone.position.x, bone.position.y, bone.position.z),
                rotation=Quaternion(
                    bone.rotation.x, bone.rotation.y,
                    bone.rotation.z, bone.rotation.w
                ),
                scale=Vector3(bone.scale.x, bone.scale.y, bone.scale.z),
            )

        # Process each reaction
        reactions_to_remove = []

        for i, reaction in enumerate(self._hit_reactions):
            elapsed = (current_time - reaction.start_time) * 1000.0

            # Update blend weight
            if elapsed < HIT_REACTION_BLEND_IN_MS:
                reaction.blend_weight = elapsed / HIT_REACTION_BLEND_IN_MS
            elif elapsed < HIT_REACTION_BLEND_IN_MS + HIT_REACTION_BLEND_OUT_MS:
                out_elapsed = elapsed - HIT_REACTION_BLEND_IN_MS
                reaction.blend_weight = 1.0 - (out_elapsed / HIT_REACTION_BLEND_OUT_MS)
            else:
                reactions_to_remove.append(i)
                continue

            # Apply hit effect to affected bones
            for bone_name in reaction.affected_bones:
                if bone_name not in result.bones:
                    continue

                bone = result.bones[bone_name]

                # Calculate push effect
                push_strength = reaction.hit_force * reaction.blend_weight * 0.01
                push_delta = reaction.hit_direction * push_strength

                # Apply position offset
                bone.position = bone.position + push_delta

                # Apply rotation effect (simplified - rotate away from hit)
                rotation_strength = reaction.hit_force * reaction.blend_weight * 0.05
                # This would need proper quaternion construction in real implementation

        # Remove completed reactions
        for i in reversed(reactions_to_remove):
            self._hit_reactions.pop(i)

        return result

    def clear_hit_reactions(self) -> None:
        """Clear all active hit reactions."""
        self._hit_reactions.clear()

    # -------------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------------

    def process(
        self,
        anim_pose: SkeletonPose,
        physics_pose: Optional[SkeletonPose],
        current_time: float,
        dt: float,
    ) -> SkeletonPose:
        """
        Process all blend layers and produce output pose.

        Args:
            anim_pose: Input animation pose
            physics_pose: Input physics pose (optional)
            current_time: Current game time
            dt: Delta time

        Returns:
            Final blended pose
        """
        result = anim_pose

        # Apply layers in order
        for layer in self._layers:
            if not layer.enabled or layer.weight <= 0:
                continue

            source = layer.source or physics_pose
            if source is None:
                continue

            if layer.mode == BlendMode.POSE:
                result = self.blend_poses(result, source, layer.weight)
            elif layer.mode == BlendMode.ADDITIVE:
                result = self.additive_physics(result, source, layer.weight)
            elif layer.mode == BlendMode.CHAIN:
                # Chain blend needs a root bone specified in mask
                if layer.mask:
                    for bone_name in layer.mask.keys():
                        result = self.blend_chain(result, source, bone_name, layer.weight)

        # Apply hit reactions
        result = self.update_hit_reactions(result, current_time, dt)

        self._output_pose = result
        return result

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def get_debug_info(self) -> dict[str, Any]:
        """Get debug information."""
        return {
            "layer_count": len(self._layers),
            "layers": [
                {
                    "name": layer.name,
                    "mode": layer.mode.value,
                    "weight": layer.weight,
                    "enabled": layer.enabled,
                }
                for layer in self._layers
            ],
            "limb_weights": self._limb_weights.copy(),
            "hit_reaction_count": len(self._hit_reactions),
            "active_reactions": [
                {
                    "force": r.hit_force,
                    "blend_weight": r.blend_weight,
                    "affected_bones": len(r.affected_bones),
                }
                for r in self._hit_reactions
            ],
        }
