"""Animation blending utilities.

Provides functions and classes for blending poses together using
different modes: override (replace), additive (add delta), and
multiply (scale). Also supports bone masks for selective blending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from engine.core.math import Quat, Vec3


# =============================================================================
# Configuration Constants
# =============================================================================

# Tolerance for weight comparisons (weights below this are treated as zero)
WEIGHT_EPSILON = 1e-9

# Tolerance for scale division (prevents division by zero in scale delta computation)
SCALE_EPSILON = 1e-9

if TYPE_CHECKING:
    from engine.animation.skeletal.skeleton import Skeleton
    from engine.animation.skeletal.pose import Pose, BoneTransform


class BlendMode(Enum):
    """Mode for blending poses."""

    OVERRIDE = auto()  # Replace base with blended value
    ADDITIVE = auto()  # Add delta on top of base
    MULTIPLY = auto()  # Multiply base by factor


def animation_data(cls):
    """Decorator for animation data classes."""
    cls._animation_data = True
    cls._animation_type = cls.__name__
    return cls


@animation_data
@dataclass
class BoneMask:
    """Mask defining which bones are affected by blending.

    Allows per-bone control over blend weights. Bones not in the mask
    use the default weight (usually 0, meaning not affected).

    Attributes:
        bone_weights: Dictionary of bone index -> weight [0, 1].
        default_weight: Weight for bones not in the dictionary.
    """

    bone_weights: Dict[int, float] = field(default_factory=dict)
    default_weight: float = 0.0

    def get_weight(self, bone_index: int) -> float:
        """Get weight for a bone.

        Args:
            bone_index: Bone index.

        Returns:
            Weight for the bone [0, 1].
        """
        return self.bone_weights.get(bone_index, self.default_weight)

    def set_weight(self, bone_index: int, weight: float) -> None:
        """Set weight for a bone.

        Args:
            bone_index: Bone index.
            weight: Weight value [0, 1].
        """
        self.bone_weights[bone_index] = max(0.0, min(1.0, weight))

    def set_weights(self, bone_indices: List[int], weight: float) -> None:
        """Set same weight for multiple bones.

        Args:
            bone_indices: List of bone indices.
            weight: Weight value [0, 1].
        """
        clamped = max(0.0, min(1.0, weight))
        for idx in bone_indices:
            self.bone_weights[idx] = clamped

    def include_bone(self, bone_index: int) -> None:
        """Include a bone with full weight."""
        self.bone_weights[bone_index] = 1.0

    def exclude_bone(self, bone_index: int) -> None:
        """Exclude a bone (set weight to 0)."""
        self.bone_weights[bone_index] = 0.0

    def include_all(self, skeleton: Skeleton) -> None:
        """Include all bones with full weight.

        Args:
            skeleton: Skeleton to get bone count from.
        """
        for i in range(skeleton.bone_count):
            self.bone_weights[i] = 1.0

    def exclude_all(self) -> None:
        """Exclude all bones (clear weights, use default)."""
        self.bone_weights.clear()

    def invert(self) -> BoneMask:
        """Create inverted mask.

        Returns:
            New mask with inverted weights.
        """
        inverted_weights = {
            idx: 1.0 - weight for idx, weight in self.bone_weights.items()
        }
        return BoneMask(
            bone_weights=inverted_weights,
            default_weight=1.0 - self.default_weight,
        )

    def combine(self, other: BoneMask) -> BoneMask:
        """Combine with another mask (multiply weights).

        Args:
            other: Other mask.

        Returns:
            New combined mask.
        """
        all_indices = set(self.bone_weights.keys()) | set(other.bone_weights.keys())
        combined_weights = {}

        for idx in all_indices:
            w1 = self.get_weight(idx)
            w2 = other.get_weight(idx)
            combined_weights[idx] = w1 * w2

        return BoneMask(
            bone_weights=combined_weights,
            default_weight=self.default_weight * other.default_weight,
        )

    @staticmethod
    def full_body(skeleton: Skeleton) -> BoneMask:
        """Create mask that includes all bones.

        Args:
            skeleton: Skeleton to create mask for.

        Returns:
            Mask with all bones at full weight.
        """
        mask = BoneMask(default_weight=1.0)
        for i in range(skeleton.bone_count):
            mask.bone_weights[i] = 1.0
        return mask

    @staticmethod
    def from_bone_chain(
        skeleton: Skeleton,
        start_bone: str,
        include_descendants: bool = True,
    ) -> BoneMask:
        """Create mask from a bone and optionally its descendants.

        Args:
            skeleton: Skeleton.
            start_bone: Name of starting bone.
            include_descendants: Whether to include descendant bones.

        Returns:
            Mask for the bone chain.
        """
        mask = BoneMask(default_weight=0.0)

        bone = skeleton.get_bone_by_name(start_bone)
        if bone is None:
            return mask

        mask.include_bone(bone.index)

        if include_descendants:
            descendants = skeleton.get_bone_descendants(bone.index)
            for idx in descendants:
                mask.include_bone(idx)

        return mask

    @staticmethod
    def upper_body(skeleton: Skeleton) -> BoneMask:
        """Create mask for upper body bones (spine and above).

        Args:
            skeleton: Skeleton (assumes standard naming).

        Returns:
            Upper body mask.
        """
        mask = BoneMask(default_weight=0.0)

        # Common upper body bone name patterns
        upper_patterns = [
            "spine", "chest", "neck", "head",
            "shoulder", "clavicle", "arm", "hand", "finger",
        ]

        for bone in skeleton:
            name_lower = bone.name.lower()
            for pattern in upper_patterns:
                if pattern in name_lower:
                    mask.include_bone(bone.index)
                    break

        return mask

    @staticmethod
    def lower_body(skeleton: Skeleton) -> BoneMask:
        """Create mask for lower body bones.

        Args:
            skeleton: Skeleton (assumes standard naming).

        Returns:
            Lower body mask.
        """
        mask = BoneMask(default_weight=0.0)

        # Common lower body bone name patterns
        lower_patterns = [
            "pelvis", "hip", "thigh", "leg", "calf", "foot", "toe",
        ]

        for bone in skeleton:
            name_lower = bone.name.lower()
            for pattern in lower_patterns:
                if pattern in name_lower:
                    mask.include_bone(bone.index)
                    break

        return mask

    def copy(self) -> BoneMask:
        """Create a deep copy of this mask."""
        return BoneMask(
            bone_weights=dict(self.bone_weights),
            default_weight=self.default_weight,
        )

    def __repr__(self) -> str:
        active = sum(1 for w in self.bone_weights.values() if w > 0)
        return f"BoneMask(active_bones={active}, default={self.default_weight})"


def blend_poses(
    pose_a: Pose,
    pose_b: Pose,
    alpha: float,
    mode: BlendMode = BlendMode.OVERRIDE,
    mask: Optional[BoneMask] = None,
) -> Pose:
    """Blend two poses together.

    Args:
        pose_a: First pose (base).
        pose_b: Second pose (to blend in).
        alpha: Blend factor [0, 1]. 0 = pose_a, 1 = pose_b.
        mode: Blend mode.
        mask: Optional bone mask for selective blending.

    Returns:
        Blended pose.

    Raises:
        ValueError: If poses have different skeletons.
    """
    from engine.animation.skeletal.pose import Pose, BoneTransform, PoseSpace

    if pose_a.skeleton is not pose_b.skeleton:
        raise ValueError("Cannot blend poses with different skeletons")

    # Clamp alpha
    alpha = max(0.0, min(1.0, alpha))

    if alpha == 0.0:
        return pose_a.copy()
    # Only skip to pose_b if no mask is used (mask affects per-bone blending)
    if alpha == 1.0 and mode == BlendMode.OVERRIDE and mask is None:
        return pose_b.copy()

    result_transforms = []

    for bone_idx in range(pose_a.bone_count):
        transform_a = pose_a._transforms[bone_idx]
        transform_b = pose_b._transforms[bone_idx]

        # Get effective alpha for this bone
        effective_alpha = alpha
        if mask is not None:
            mask_weight = mask.get_weight(bone_idx)
            effective_alpha = alpha * mask_weight

        if effective_alpha <= 0.0:
            result_transforms.append(transform_a.copy())
            continue

        if mode == BlendMode.OVERRIDE:
            result_transforms.append(
                _blend_override(transform_a, transform_b, effective_alpha)
            )
        elif mode == BlendMode.ADDITIVE:
            result_transforms.append(
                _blend_additive(transform_a, transform_b, effective_alpha)
            )
        elif mode == BlendMode.MULTIPLY:
            result_transforms.append(
                _blend_multiply(transform_a, transform_b, effective_alpha)
            )
        else:
            result_transforms.append(
                _blend_override(transform_a, transform_b, effective_alpha)
            )

    return Pose(
        skeleton=pose_a.skeleton,
        space=pose_a.space,
        bone_transforms=result_transforms,
    )


def _blend_override(
    a: BoneTransform, b: BoneTransform, alpha: float
) -> BoneTransform:
    """Override blend (standard lerp/slerp)."""
    from engine.animation.skeletal.pose import BoneTransform

    return BoneTransform(
        translation=a.translation.lerp(b.translation, alpha),
        rotation=a.rotation.slerp(b.rotation, alpha),
        scale=a.scale.lerp(b.scale, alpha),
    )


def _blend_additive(
    base: BoneTransform, additive: BoneTransform, weight: float
) -> BoneTransform:
    """Additive blend (base + weighted delta)."""
    from engine.animation.skeletal.pose import BoneTransform

    # Translation: base + additive * weight
    new_translation = Vec3(
        base.translation.x + additive.translation.x * weight,
        base.translation.y + additive.translation.y * weight,
        base.translation.z + additive.translation.z * weight,
    )

    # Rotation: base * slerp(identity, additive, weight)
    identity = Quat.identity()
    weighted_rotation = identity.slerp(additive.rotation, weight)
    new_rotation = base.rotation * weighted_rotation

    # Scale: base * (1 + (additive - 1) * weight)
    new_scale = Vec3(
        base.scale.x * (1.0 + (additive.scale.x - 1.0) * weight),
        base.scale.y * (1.0 + (additive.scale.y - 1.0) * weight),
        base.scale.z * (1.0 + (additive.scale.z - 1.0) * weight),
    )

    return BoneTransform(
        translation=new_translation,
        rotation=new_rotation,
        scale=new_scale,
    )


# Configuration constants for blending
BLEND_TRANSLATION_EPSILON = 0.001  # Minimum translation magnitude for multiply blend
BLEND_WEIGHT_EPSILON = 1e-9  # Minimum weight threshold


def _blend_multiply(
    base: BoneTransform, factor: BoneTransform, weight: float
) -> BoneTransform:
    """Multiply blend (base scaled by factor)."""
    from engine.animation.skeletal.pose import BoneTransform

    # For translation with near-zero base, use additive instead of multiplicative
    base_mag_sq = (
        base.translation.x * base.translation.x +
        base.translation.y * base.translation.y +
        base.translation.z * base.translation.z
    )

    if base_mag_sq > BLEND_TRANSLATION_EPSILON * BLEND_TRANSLATION_EPSILON:
        # Safe to use multiplicative blend - avoid division by zero with max()
        new_translation = Vec3(
            base.translation.x * (1.0 + (factor.translation.x - base.translation.x) * weight / max(abs(base.translation.x), BLEND_TRANSLATION_EPSILON)),
            base.translation.y * (1.0 + (factor.translation.y - base.translation.y) * weight / max(abs(base.translation.y), BLEND_TRANSLATION_EPSILON)),
            base.translation.z * (1.0 + (factor.translation.z - base.translation.z) * weight / max(abs(base.translation.z), BLEND_TRANSLATION_EPSILON)),
        )
    else:
        # Fall back to additive for near-zero base
        new_translation = Vec3(
            base.translation.x + factor.translation.x * weight,
            base.translation.y + factor.translation.y * weight,
            base.translation.z + factor.translation.z * weight,
        )

    # Rotation: lerp between base and base*factor
    combined_rotation = base.rotation * factor.rotation
    new_rotation = base.rotation.slerp(combined_rotation, weight)

    # Scale: multiply
    new_scale = Vec3(
        base.scale.x * (1.0 + (factor.scale.x - 1.0) * weight),
        base.scale.y * (1.0 + (factor.scale.y - 1.0) * weight),
        base.scale.z * (1.0 + (factor.scale.z - 1.0) * weight),
    )

    return BoneTransform(
        translation=new_translation,
        rotation=new_rotation,
        scale=new_scale,
    )


def blend_multiple_poses(
    poses: List[Pose],
    weights: List[float],
    mode: BlendMode = BlendMode.OVERRIDE,
    normalize: bool = True,
) -> Pose:
    """Blend multiple poses with weights.

    Args:
        poses: List of poses to blend.
        weights: List of blend weights.
        mode: Blend mode (OVERRIDE recommended for multi-pose).
        normalize: Whether to normalize weights to sum to 1.

    Returns:
        Blended pose.

    Raises:
        ValueError: If poses/weights mismatch or empty.
    """
    from engine.animation.skeletal.pose import Pose, BoneTransform, PoseSpace

    if not poses:
        raise ValueError("Cannot blend empty pose list")
    if len(poses) != len(weights):
        raise ValueError(
            f"Pose count ({len(poses)}) must match weight count ({len(weights)})"
        )

    # Check all same skeleton
    skeleton = poses[0].skeleton
    space = poses[0].space
    for pose in poses[1:]:
        if pose.skeleton is not skeleton:
            raise ValueError("All poses must have same skeleton")
        if pose.space != space:
            raise ValueError("All poses must be in same space")

    # Normalize weights
    if normalize:
        total = sum(max(0.0, w) for w in weights)
        if total < WEIGHT_EPSILON:
            return poses[0].copy()
        weights = [max(0.0, w) / total for w in weights]

    # Blend all poses
    result_transforms = []

    for bone_idx in range(skeleton.bone_count):
        if mode == BlendMode.OVERRIDE:
            blended = _blend_multiple_override(
                [p._transforms[bone_idx] for p in poses],
                weights,
            )
        elif mode == BlendMode.ADDITIVE:
            # For additive, use first as base, add rest
            blended = poses[0]._transforms[bone_idx].copy()
            for i in range(1, len(poses)):
                if weights[i] > WEIGHT_EPSILON:
                    blended = _blend_additive(
                        blended,
                        poses[i]._transforms[bone_idx],
                        weights[i],
                    )
        else:
            blended = _blend_multiple_override(
                [p._transforms[bone_idx] for p in poses],
                weights,
            )

        result_transforms.append(blended)

    return Pose(skeleton, space, result_transforms)


def _blend_multiple_override(
    transforms: List[BoneTransform],
    weights: List[float],
) -> BoneTransform:
    """Blend multiple transforms with weighted average."""
    from engine.animation.skeletal.pose import BoneTransform

    # Weighted average of translations and scales
    total_translation = Vec3.zero()
    total_scale = Vec3.zero()

    for t, w in zip(transforms, weights):
        if w > WEIGHT_EPSILON:
            total_translation = Vec3(
                total_translation.x + t.translation.x * w,
                total_translation.y + t.translation.y * w,
                total_translation.z + t.translation.z * w,
            )
            total_scale = Vec3(
                total_scale.x + t.scale.x * w,
                total_scale.y + t.scale.y * w,
                total_scale.z + t.scale.z * w,
            )

    # For rotations, use iterative slerp
    blended_rotation = transforms[0].rotation
    accumulated_weight = weights[0]

    for i in range(1, len(transforms)):
        if weights[i] > WEIGHT_EPSILON:
            accumulated_weight += weights[i]
            if accumulated_weight > WEIGHT_EPSILON:
                slerp_factor = weights[i] / accumulated_weight
                blended_rotation = blended_rotation.slerp(
                    transforms[i].rotation, slerp_factor
                )

    return BoneTransform(
        translation=total_translation,
        rotation=blended_rotation,
        scale=total_scale,
    )


def compute_additive_pose(
    reference_pose: Pose,
    target_pose: Pose,
) -> Pose:
    """Compute additive pose (delta) from reference to target.

    Args:
        reference_pose: Base reference pose.
        target_pose: Target pose.

    Returns:
        Additive pose (delta).

    Raises:
        ValueError: If poses have different skeletons.
    """
    from engine.animation.skeletal.pose import Pose, BoneTransform, PoseSpace

    if reference_pose.skeleton is not target_pose.skeleton:
        raise ValueError("Poses must have same skeleton")

    additive_transforms = []

    for bone_idx in range(reference_pose.bone_count):
        ref = reference_pose._transforms[bone_idx]
        tgt = target_pose._transforms[bone_idx]

        # Compute deltas
        delta_translation = Vec3(
            tgt.translation.x - ref.translation.x,
            tgt.translation.y - ref.translation.y,
            tgt.translation.z - ref.translation.z,
        )

        delta_rotation = ref.rotation.inverse() * tgt.rotation

        # Scale as ratio - use epsilon to prevent division by near-zero
        delta_scale = Vec3(
            tgt.scale.x / ref.scale.x if abs(ref.scale.x) > SCALE_EPSILON else 1.0,
            tgt.scale.y / ref.scale.y if abs(ref.scale.y) > SCALE_EPSILON else 1.0,
            tgt.scale.z / ref.scale.z if abs(ref.scale.z) > SCALE_EPSILON else 1.0,
        )

        additive_transforms.append(
            BoneTransform(
                translation=delta_translation,
                rotation=delta_rotation,
                scale=delta_scale,
            )
        )

    return Pose(
        skeleton=reference_pose.skeleton,
        space=reference_pose.space,
        bone_transforms=additive_transforms,
    )


def apply_additive_pose(
    base_pose: Pose,
    additive_pose: Pose,
    weight: float = 1.0,
    mask: Optional[BoneMask] = None,
) -> Pose:
    """Apply an additive pose on top of a base pose.

    Args:
        base_pose: Base pose.
        additive_pose: Additive delta pose.
        weight: How much additive to apply [0, 1].
        mask: Optional bone mask.

    Returns:
        Resulting pose.

    Raises:
        ValueError: If poses have different skeletons.
    """
    from engine.animation.skeletal.pose import Pose, BoneTransform, PoseSpace

    if base_pose.skeleton is not additive_pose.skeleton:
        raise ValueError("Poses must have same skeleton")

    weight = max(0.0, min(1.0, weight))

    if weight <= 0.0:
        return base_pose.copy()

    result_transforms = []

    for bone_idx in range(base_pose.bone_count):
        base = base_pose._transforms[bone_idx]
        add = additive_pose._transforms[bone_idx]

        # Get effective weight
        effective_weight = weight
        if mask is not None:
            effective_weight *= mask.get_weight(bone_idx)

        if effective_weight <= 0.0:
            result_transforms.append(base.copy())
            continue

        result_transforms.append(_blend_additive(base, add, effective_weight))

    return Pose(
        skeleton=base_pose.skeleton,
        space=base_pose.space,
        bone_transforms=result_transforms,
    )


@animation_data
class LayeredBlender:
    """Multi-layer pose blending system.

    Allows stacking multiple layers with different blend modes and masks.
    """

    @dataclass
    class Layer:
        """A single blend layer."""

        name: str
        pose: Optional[Pose] = None
        weight: float = 1.0
        mode: BlendMode = BlendMode.OVERRIDE
        mask: Optional[BoneMask] = None
        enabled: bool = True

    def __init__(self, skeleton: Skeleton) -> None:
        """Initialize blender.

        Args:
            skeleton: Skeleton for poses.
        """
        self._skeleton = skeleton
        self._layers: List[LayeredBlender.Layer] = []

    @property
    def skeleton(self) -> Skeleton:
        """Get skeleton."""
        return self._skeleton

    @property
    def layer_count(self) -> int:
        """Get number of layers."""
        return len(self._layers)

    def add_layer(
        self,
        name: str,
        mode: BlendMode = BlendMode.OVERRIDE,
        mask: Optional[BoneMask] = None,
        weight: float = 1.0,
    ) -> int:
        """Add a new layer.

        Args:
            name: Layer name.
            mode: Blend mode.
            mask: Optional bone mask.
            weight: Layer weight.

        Returns:
            Index of new layer.
        """
        layer = LayeredBlender.Layer(
            name=name,
            weight=weight,
            mode=mode,
            mask=mask,
        )
        self._layers.append(layer)
        return len(self._layers) - 1

    def remove_layer(self, index: int) -> None:
        """Remove a layer.

        Args:
            index: Layer index.
        """
        if 0 <= index < len(self._layers):
            self._layers.pop(index)

    def get_layer(self, index: int) -> Optional[Layer]:
        """Get layer by index."""
        if 0 <= index < len(self._layers):
            return self._layers[index]
        return None

    def get_layer_by_name(self, name: str) -> Optional[Layer]:
        """Get layer by name."""
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None

    def set_layer_pose(self, index: int, pose: Pose) -> None:
        """Set pose for a layer.

        Args:
            index: Layer index.
            pose: Pose to set.
        """
        if 0 <= index < len(self._layers):
            if pose.skeleton is not self._skeleton:
                raise ValueError("Pose skeleton must match blender skeleton")
            self._layers[index].pose = pose

    def set_layer_weight(self, index: int, weight: float) -> None:
        """Set weight for a layer.

        Args:
            index: Layer index.
            weight: New weight [0, 1].
        """
        if 0 <= index < len(self._layers):
            self._layers[index].weight = max(0.0, min(1.0, weight))

    def set_layer_enabled(self, index: int, enabled: bool) -> None:
        """Enable/disable a layer.

        Args:
            index: Layer index.
            enabled: Whether layer is enabled.
        """
        if 0 <= index < len(self._layers):
            self._layers[index].enabled = enabled

    def blend(self, base_pose: Optional[Pose] = None) -> Optional[Pose]:
        """Blend all layers together.

        Args:
            base_pose: Optional base pose (if None, uses first layer).

        Returns:
            Blended pose, or None if no valid layers.
        """
        from engine.animation.skeletal.pose import Pose

        # Start with base
        if base_pose is not None:
            result = base_pose.copy()
        else:
            # Find first enabled layer with pose
            result = None
            for layer in self._layers:
                if layer.enabled and layer.pose is not None:
                    result = layer.pose.copy()
                    break

        if result is None:
            return None

        # Apply each layer
        for layer in self._layers:
            if not layer.enabled or layer.pose is None or layer.weight <= 0:
                continue

            if layer.pose is result:
                continue

            result = blend_poses(
                result,
                layer.pose,
                layer.weight,
                layer.mode,
                layer.mask,
            )

        return result

    def clear(self) -> None:
        """Remove all layers."""
        self._layers.clear()

    def __repr__(self) -> str:
        enabled = sum(1 for l in self._layers if l.enabled)
        return f"LayeredBlender(layers={len(self._layers)}, enabled={enabled})"


@animation_data
class PoseCache:
    """Cache for storing and reusing poses.

    Useful for avoiding repeated pose sampling when the same
    pose is needed multiple times per frame.
    """

    def __init__(self, skeleton: Skeleton, capacity: int = 8) -> None:
        """Initialize cache.

        Args:
            skeleton: Skeleton for poses.
            capacity: Maximum cached poses.
        """
        self._skeleton = skeleton
        self._capacity = max(1, capacity)
        self._cache: Dict[str, Pose] = {}
        self._access_order: List[str] = []

    @property
    def skeleton(self) -> Skeleton:
        """Get skeleton."""
        return self._skeleton

    @property
    def size(self) -> int:
        """Get number of cached poses."""
        return len(self._cache)

    def get(self, key: str) -> Optional[Pose]:
        """Get cached pose.

        Args:
            key: Cache key.

        Returns:
            Cached pose copy, or None if not found.
        """
        if key in self._cache:
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key].copy()
        return None

    def put(self, key: str, pose: Pose) -> None:
        """Cache a pose.

        Args:
            key: Cache key.
            pose: Pose to cache.
        """
        if pose.skeleton is not self._skeleton:
            raise ValueError("Pose skeleton must match cache skeleton")

        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._capacity:
            # Evict oldest
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = pose.copy()
        self._access_order.append(key)

    def contains(self, key: str) -> bool:
        """Check if key is cached."""
        return key in self._cache

    def remove(self, key: str) -> bool:
        """Remove cached pose.

        Returns:
            True if removed, False if not found.
        """
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)
            return True
        return False

    def clear(self) -> None:
        """Clear all cached poses."""
        self._cache.clear()
        self._access_order.clear()

    def __repr__(self) -> str:
        return f"PoseCache(size={len(self._cache)}/{self._capacity})"
