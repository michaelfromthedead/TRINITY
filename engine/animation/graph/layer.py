"""
Animation layering system.

Provides functionality for layered animation blending:
- AnimationLayer: Individual layer with weight, mask, and blend mode
- LayerStack: Ordered collection of layers for evaluation
- BoneMask presets: Common masks for partial body animation

Layers allow for complex animation combinations such as:
- Upper body attack with lower body locomotion
- Additive breathing on top of any animation
- Partial body overrides for aiming/looking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from .animation_graph import (
    AnimationNode,
    BoneMask,
    GraphContext,
    Pose,
    Skeleton,
    Transform,
)


# =============================================================================
# LAYER BLEND MODE
# =============================================================================


class LayerBlendMode(Enum):
    """How a layer blends with layers below it."""

    OVERRIDE = auto()      # Replace bones with layer values
    ADDITIVE = auto()      # Add layer transforms to base
    MULTIPLY = auto()      # Multiply base by layer (component-wise)
    OVERRIDE_ADDITIVE = auto()  # Override with additive result


# =============================================================================
# ANIMATION LAYER
# =============================================================================


@dataclass
class AnimationLayer:
    """
    A single animation layer in a layer stack.

    Each layer has:
    - A source node that produces the pose
    - A weight (0-1) controlling the layer's influence
    - An optional bone mask for partial body animation
    - A blend mode determining how it combines with lower layers
    """

    name: str
    source: Optional[AnimationNode] = None
    weight: float = 1.0
    mask: Optional[BoneMask] = None
    blend_mode: LayerBlendMode = LayerBlendMode.OVERRIDE

    # Optional parameter bindings
    weight_parameter: Optional[str] = None

    # Layer state
    is_active: bool = True
    is_synced: bool = False  # Sync time with other layers
    sync_group: Optional[str] = None

    def get_effective_weight(self, context: GraphContext) -> float:
        """Get the effective weight considering parameters and active state."""
        if not self.is_active:
            return 0.0

        weight = self.weight
        if self.weight_parameter:
            weight = context.get_parameter_float(self.weight_parameter, self.weight)

        return max(0.0, min(1.0, weight))

    def evaluate(self, context: GraphContext) -> Optional[Pose]:
        """Evaluate this layer's source node."""
        if not self.is_active or not self.source:
            return None

        # Create layer context
        layer_context = GraphContext(
            parameters=context.parameters,
            dt=context.dt,
            skeleton=context.skeleton,
            bone_masks=context.bone_masks,
            normalized_time=context.normalized_time,
            sync_group=self.sync_group or context.sync_group,
            layer_weight=self.get_effective_weight(context),
        )

        return self.source.evaluate(layer_context)

    def apply_to_pose(self, base_pose: Pose, context: GraphContext) -> Pose:
        """Apply this layer to a base pose and return the result."""
        weight = self.get_effective_weight(context)
        if weight <= 0:
            return base_pose

        layer_pose = self.evaluate(context)
        if not layer_pose:
            return base_pose

        # Apply based on blend mode and mask
        if self.mask:
            return self._apply_masked(base_pose, layer_pose, weight)
        else:
            return self._apply_full(base_pose, layer_pose, weight)

    def _apply_masked(self, base_pose: Pose, layer_pose: Pose,
                      weight: float) -> Pose:
        """Apply layer with bone mask."""
        result = base_pose.copy()

        for i in range(min(base_pose.bone_count(), layer_pose.bone_count())):
            mask_weight = self.mask.get_weight(i) * weight
            if mask_weight <= 0:
                continue

            if self.blend_mode == LayerBlendMode.ADDITIVE:
                # Additive blend
                additive = Transform.identity().lerp(layer_pose.transforms[i], mask_weight)
                result.transforms[i] = result.transforms[i] + additive
            elif self.blend_mode == LayerBlendMode.MULTIPLY:
                # Multiply blend: result = base * layer (component-wise)
                multiplied = self._multiply_transform(
                    base_pose.transforms[i], layer_pose.transforms[i]
                )
                result.transforms[i] = base_pose.transforms[i].lerp(multiplied, mask_weight)
            elif self.blend_mode == LayerBlendMode.OVERRIDE_ADDITIVE:
                # Override with additive transform
                additive_result = base_pose.transforms[i] + layer_pose.transforms[i]
                result.transforms[i] = base_pose.transforms[i].lerp(additive_result, mask_weight)
            else:  # OVERRIDE
                result.transforms[i] = base_pose.transforms[i].lerp(
                    layer_pose.transforms[i], mask_weight
                )

        return result

    def _apply_full(self, base_pose: Pose, layer_pose: Pose,
                    weight: float) -> Pose:
        """Apply layer to all bones."""
        if self.blend_mode == LayerBlendMode.ADDITIVE:
            return base_pose.additive_blend(layer_pose, weight)
        elif self.blend_mode == LayerBlendMode.MULTIPLY:
            return self._multiply_blend(base_pose, layer_pose, weight)
        elif self.blend_mode == LayerBlendMode.OVERRIDE_ADDITIVE:
            additive_result = base_pose.additive_blend(layer_pose, 1.0)
            return base_pose.lerp(additive_result, weight)
        else:  # OVERRIDE
            return base_pose.lerp(layer_pose, weight)

    def _multiply_transform(self, base: Transform, factor: Transform) -> Transform:
        """Multiply two transforms component-wise."""
        # Position: component-wise multiplication
        new_position = tuple(
            b * f for b, f in zip(base.position, factor.position)
        )
        # Rotation: quaternion multiplication (compose rotations)
        new_rotation = Transform._multiply_quaternion(base.rotation, factor.rotation)
        # Scale: component-wise multiplication
        new_scale = tuple(
            b * f for b, f in zip(base.scale, factor.scale)
        )
        return Transform(position=new_position, rotation=new_rotation, scale=new_scale)

    def _multiply_blend(self, base_pose: Pose, layer_pose: Pose,
                        weight: float) -> Pose:
        """Apply multiply blend to all bones."""
        max_bones = max(base_pose.bone_count(), layer_pose.bone_count())
        result_transforms = []

        for i in range(max_bones):
            base = base_pose.get_transform(i)
            factor = layer_pose.get_transform(i)

            # Compute multiplied result
            multiplied = self._multiply_transform(base, factor)

            # Lerp from base to multiplied based on weight
            result_transforms.append(base.lerp(multiplied, weight))

        return Pose(
            transforms=result_transforms,
            root_motion=base_pose.root_motion,
            skeleton=base_pose.skeleton,
        )


# =============================================================================
# LAYER STACK
# =============================================================================


class LayerStack(AnimationNode):
    """
    An ordered collection of animation layers.

    Layers are evaluated from bottom to top (index 0 is the base).
    Each layer modifies the result of the layers below it.

    The layer stack itself is an AnimationNode, so it can be used
    as a node in animation graphs.
    """

    _abstract = False

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.layers: List[AnimationLayer] = []
        self._layer_by_name: Dict[str, AnimationLayer] = {}

    def add_layer(self, layer: AnimationLayer, index: Optional[int] = None) -> int:
        """Add a layer to the stack."""
        if layer.name in self._layer_by_name:
            raise ValueError(f"Layer '{layer.name}' already exists")

        if index is None:
            self.layers.append(layer)
            actual_index = len(self.layers) - 1
        else:
            index = max(0, min(len(self.layers), index))
            self.layers.insert(index, layer)
            actual_index = index

        self._layer_by_name[layer.name] = layer
        return actual_index

    def remove_layer(self, name_or_index: Any) -> bool:
        """Remove a layer by name or index."""
        if isinstance(name_or_index, str):
            layer = self._layer_by_name.get(name_or_index)
            if layer:
                self.layers.remove(layer)
                del self._layer_by_name[name_or_index]
                return True
        elif isinstance(name_or_index, int):
            if 0 <= name_or_index < len(self.layers):
                layer = self.layers.pop(name_or_index)
                del self._layer_by_name[layer.name]
                return True
        return False

    def get_layer(self, name: str) -> Optional[AnimationLayer]:
        """Get a layer by name."""
        return self._layer_by_name.get(name)

    def get_layer_by_index(self, index: int) -> Optional[AnimationLayer]:
        """Get a layer by index."""
        if 0 <= index < len(self.layers):
            return self.layers[index]
        return None

    def move_layer(self, name: str, new_index: int) -> bool:
        """Move a layer to a new position in the stack."""
        layer = self._layer_by_name.get(name)
        if not layer:
            return False

        self.layers.remove(layer)
        new_index = max(0, min(len(self.layers), new_index))
        self.layers.insert(new_index, layer)
        return True

    def set_layer_weight(self, name: str, weight: float) -> bool:
        """Set a layer's weight."""
        layer = self._layer_by_name.get(name)
        if layer:
            layer.weight = max(0.0, min(1.0, weight))
            return True
        return False

    def set_layer_active(self, name: str, active: bool) -> bool:
        """Set a layer's active state."""
        layer = self._layer_by_name.get(name)
        if layer:
            layer.is_active = active
            return True
        return False

    def layer_count(self) -> int:
        """Get the number of layers."""
        return len(self.layers)

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate all layers and combine them."""
        if not self.layers:
            return Pose()

        # Start with the first active layer as base
        result_pose = None

        for layer in self.layers:
            if not layer.is_active:
                continue

            weight = layer.get_effective_weight(context)
            if weight <= 0:
                continue

            if result_pose is None:
                # First layer becomes the base
                layer_pose = layer.evaluate(context)
                if layer_pose:
                    result_pose = layer_pose.copy()
            else:
                # Apply subsequent layers
                result_pose = layer.apply_to_pose(result_pose, context)

        return result_pose or Pose()

    def get_active_layers(self) -> List[AnimationLayer]:
        """Get all active layers."""
        return [layer for layer in self.layers if layer.is_active]

    def get_layers_with_weight(self) -> List[Tuple[AnimationLayer, float]]:
        """Get all layers with their effective weights."""
        result = []
        for layer in self.layers:
            if layer.is_active:
                # Note: We can't get parameter-based weight without context
                result.append((layer, layer.weight))
        return result


# =============================================================================
# BONE MASK PRESETS
# =============================================================================


class BoneMaskPresets:
    """Factory for common bone mask configurations."""

    # Standard humanoid bone name conventions
    UPPER_BODY_BONES = [
        "Spine", "Spine1", "Spine2", "Chest",
        "Neck", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        # Include fingers
        "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
        "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
        "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
        "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
        "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
        "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
        "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
        "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
        "RightHandRing1", "RightHandRing2", "RightHandRing3",
        "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
    ]

    LOWER_BODY_BONES = [
        "Hips", "Pelvis",
        "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
        "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    ]

    LEFT_ARM_BONES = [
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
        "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
        "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
        "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
        "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
    ]

    RIGHT_ARM_BONES = [
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
        "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
        "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
        "RightHandRing1", "RightHandRing2", "RightHandRing3",
        "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
    ]

    SPINE_BONES = ["Spine", "Spine1", "Spine2", "Chest"]

    HEAD_BONES = ["Neck", "Head"]

    LEFT_LEG_BONES = ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"]

    RIGHT_LEG_BONES = ["RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]

    @classmethod
    def upper_body(cls, skeleton: Skeleton) -> BoneMask:
        """Create an upper body mask."""
        return BoneMask.from_bone_names(
            skeleton, "UpperBody", cls.UPPER_BODY_BONES,
            weight=1.0, include_children=True
        )

    @classmethod
    def lower_body(cls, skeleton: Skeleton) -> BoneMask:
        """Create a lower body mask."""
        return BoneMask.from_bone_names(
            skeleton, "LowerBody", cls.LOWER_BODY_BONES,
            weight=1.0, include_children=True
        )

    @classmethod
    def left_arm(cls, skeleton: Skeleton) -> BoneMask:
        """Create a left arm mask."""
        return BoneMask.from_bone_names(
            skeleton, "LeftArm", cls.LEFT_ARM_BONES,
            weight=1.0, include_children=True
        )

    @classmethod
    def right_arm(cls, skeleton: Skeleton) -> BoneMask:
        """Create a right arm mask."""
        return BoneMask.from_bone_names(
            skeleton, "RightArm", cls.RIGHT_ARM_BONES,
            weight=1.0, include_children=True
        )

    @classmethod
    def spine(cls, skeleton: Skeleton) -> BoneMask:
        """Create a spine mask."""
        return BoneMask.from_bone_names(
            skeleton, "Spine", cls.SPINE_BONES,
            weight=1.0, include_children=False
        )

    @classmethod
    def head(cls, skeleton: Skeleton) -> BoneMask:
        """Create a head mask."""
        return BoneMask.from_bone_names(
            skeleton, "Head", cls.HEAD_BONES,
            weight=1.0, include_children=True
        )

    @classmethod
    def left_leg(cls, skeleton: Skeleton) -> BoneMask:
        """Create a left leg mask."""
        return BoneMask.from_bone_names(
            skeleton, "LeftLeg", cls.LEFT_LEG_BONES,
            weight=1.0, include_children=True
        )

    @classmethod
    def right_leg(cls, skeleton: Skeleton) -> BoneMask:
        """Create a right leg mask."""
        return BoneMask.from_bone_names(
            skeleton, "RightLeg", cls.RIGHT_LEG_BONES,
            weight=1.0, include_children=True
        )

    @classmethod
    def arms(cls, skeleton: Skeleton) -> BoneMask:
        """Create a mask for both arms."""
        mask = BoneMask(name="Arms")
        for bone_name in cls.LEFT_ARM_BONES + cls.RIGHT_ARM_BONES:
            bone = skeleton.get_bone_by_name(bone_name)
            if bone:
                mask.set_weight(bone.index, 1.0)
        return mask

    @classmethod
    def legs(cls, skeleton: Skeleton) -> BoneMask:
        """Create a mask for both legs."""
        mask = BoneMask(name="Legs")
        for bone_name in cls.LEFT_LEG_BONES + cls.RIGHT_LEG_BONES:
            bone = skeleton.get_bone_by_name(bone_name)
            if bone:
                mask.set_weight(bone.index, 1.0)
        return mask

    @classmethod
    def full_body(cls, skeleton: Skeleton) -> BoneMask:
        """Create a full body mask (all bones at weight 1.0)."""
        return BoneMask.full(skeleton, "FullBody")

    @classmethod
    def gradient_upper_lower(cls, skeleton: Skeleton,
                              spine_weights: Optional[List[float]] = None) -> BoneMask:
        """
        Create a gradient mask for smooth upper/lower body blending.

        The spine bones get intermediate weights for smooth transition.
        """
        mask = BoneMask(name="GradientUpperLower")

        # Default gradient: Hips=0.0, Spine=0.25, Spine1=0.5, Spine2=0.75, Chest=1.0
        if spine_weights is None:
            spine_weights = [0.25, 0.5, 0.75, 1.0]

        # Lower body = 0
        for bone_name in cls.LOWER_BODY_BONES:
            bone = skeleton.get_bone_by_name(bone_name)
            if bone:
                mask.set_weight(bone.index, 0.0)

        # Spine gradient
        for i, bone_name in enumerate(cls.SPINE_BONES):
            bone = skeleton.get_bone_by_name(bone_name)
            if bone and i < len(spine_weights):
                mask.set_weight(bone.index, spine_weights[i])

        # Upper body = 1
        for bone_name in cls.UPPER_BODY_BONES:
            if bone_name not in cls.SPINE_BONES:
                bone = skeleton.get_bone_by_name(bone_name)
                if bone:
                    mask.set_weight(bone.index, 1.0)

        return mask


# =============================================================================
# LAYER STACK BUILDER
# =============================================================================


class LayerStackBuilder:
    """Fluent builder for creating layer stacks."""

    def __init__(self, node_id: str) -> None:
        self._node_id = node_id
        self._layers: List[AnimationLayer] = []

    def add_layer(
        self,
        name: str,
        source: Optional[AnimationNode] = None,
        weight: float = 1.0,
        mask: Optional[BoneMask] = None,
        blend_mode: LayerBlendMode = LayerBlendMode.OVERRIDE,
    ) -> "LayerStackBuilder":
        """Add a layer to the builder."""
        layer = AnimationLayer(
            name=name,
            source=source,
            weight=weight,
            mask=mask,
            blend_mode=blend_mode,
        )
        self._layers.append(layer)
        return self

    def add_override_layer(
        self,
        name: str,
        source: Optional[AnimationNode] = None,
        weight: float = 1.0,
        mask: Optional[BoneMask] = None,
    ) -> "LayerStackBuilder":
        """Add an override layer."""
        return self.add_layer(name, source, weight, mask, LayerBlendMode.OVERRIDE)

    def add_additive_layer(
        self,
        name: str,
        source: Optional[AnimationNode] = None,
        weight: float = 1.0,
        mask: Optional[BoneMask] = None,
    ) -> "LayerStackBuilder":
        """Add an additive layer."""
        return self.add_layer(name, source, weight, mask, LayerBlendMode.ADDITIVE)

    def set_base(self, source: AnimationNode) -> "LayerStackBuilder":
        """Set the base layer source.

        Inserts a base layer at position 0 with full weight, no mask,
        and OVERRIDE blend mode.
        """
        base_layer = AnimationLayer(
            name="_base",
            source=source,
            weight=1.0,
            blend_mode=LayerBlendMode.OVERRIDE,
        )
        self._layers.insert(0, base_layer)
        return self

    def build(self) -> LayerStack:
        """Build the layer stack."""
        stack = LayerStack(self._node_id)
        for layer in self._layers:
            stack.add_layer(layer)
        return stack


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Blend mode
    "LayerBlendMode",
    # Layer
    "AnimationLayer",
    # Stack
    "LayerStack",
    "LayerStackBuilder",
    # Presets
    "BoneMaskPresets",
]
