"""Animation pose representation.

A Pose represents the transforms of all bones in a skeleton at a specific
point in time. Poses can be in local-space (relative to parent) or
model-space (relative to root).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional

from engine.core.math import Mat4, Quat, Transform, Vec3

if TYPE_CHECKING:
    from engine.animation.skeletal.skeleton import Skeleton


# =============================================================================
# Configuration Constants
# =============================================================================

# Tolerance for weight comparisons in pose blending
WEIGHT_EPSILON = 1e-9

# Tolerance for scale division to prevent division by zero/near-zero
SCALE_EPSILON = 1e-9


class PoseSpace(Enum):
    """Coordinate space for pose transforms."""

    LOCAL = auto()  # Relative to parent bone
    MODEL = auto()  # Relative to skeleton root


def animation_data(cls):
    """Decorator for animation data classes."""
    cls._animation_data = True
    cls._animation_type = cls.__name__
    return cls


@animation_data
@dataclass
class BoneTransform:
    """Transform data for a single bone.

    Attributes:
        translation: Position offset (Vec3).
        rotation: Orientation (Quaternion).
        scale: Scale factors (Vec3).
    """

    translation: Vec3 = field(default_factory=Vec3.zero)
    rotation: Quat = field(default_factory=Quat.identity)
    scale: Vec3 = field(default_factory=Vec3.one)

    def to_transform(self) -> Transform:
        """Convert to a Transform object."""
        return Transform(
            translation=Vec3(self.translation.x, self.translation.y, self.translation.z),
            rotation=Quat(
                self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w
            ),
            scale=Vec3(self.scale.x, self.scale.y, self.scale.z),
        )

    @staticmethod
    def from_transform(transform: Transform) -> BoneTransform:
        """Create from a Transform object."""
        return BoneTransform(
            translation=Vec3(
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ),
            rotation=Quat(
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ),
            scale=Vec3(
                transform.scale.x,
                transform.scale.y,
                transform.scale.z,
            ),
        )

    @staticmethod
    def identity() -> BoneTransform:
        """Create an identity transform."""
        return BoneTransform(
            translation=Vec3.zero(),
            rotation=Quat.identity(),
            scale=Vec3.one(),
        )

    def copy(self) -> BoneTransform:
        """Create a deep copy."""
        return BoneTransform(
            translation=Vec3(self.translation.x, self.translation.y, self.translation.z),
            rotation=Quat(
                self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w
            ),
            scale=Vec3(self.scale.x, self.scale.y, self.scale.z),
        )

    def lerp(self, other: BoneTransform, t: float) -> BoneTransform:
        """Linear interpolation to another transform.

        Uses linear interpolation for translation and scale,
        spherical linear interpolation for rotation.

        Args:
            other: Target transform.
            t: Interpolation factor [0, 1].

        Returns:
            Interpolated transform.
        """
        return BoneTransform(
            translation=self.translation.lerp(other.translation, t),
            rotation=self.rotation.slerp(other.rotation, t),
            scale=self.scale.lerp(other.scale, t),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoneTransform):
            return NotImplemented
        return (
            self.translation == other.translation
            and self.rotation == other.rotation
            and self.scale == other.scale
        )

    def __repr__(self) -> str:
        return (
            f"BoneTransform(t={self.translation}, "
            f"r={self.rotation}, s={self.scale})"
        )


@animation_data
class Pose:
    """A complete pose for a skeleton.

    Represents the transforms of all bones at a single point in time.
    Can be in local-space or model-space.

    Attributes:
        skeleton: Reference to the skeleton this pose belongs to.
        bone_transforms: List of transforms, one per bone.
        space: Coordinate space of the transforms.
    """

    def __init__(
        self,
        skeleton: Skeleton,
        space: PoseSpace = PoseSpace.LOCAL,
        bone_transforms: Optional[List[BoneTransform]] = None,
    ) -> None:
        """Initialize a pose.

        Args:
            skeleton: The skeleton this pose belongs to.
            space: Coordinate space for transforms.
            bone_transforms: Initial transforms. If None, uses identity.
        """
        self._skeleton = skeleton
        self._space = space

        if bone_transforms is not None:
            if len(bone_transforms) != skeleton.bone_count:
                raise ValueError(
                    f"Transform count ({len(bone_transforms)}) must match "
                    f"bone count ({skeleton.bone_count})"
                )
            self._transforms = [t.copy() for t in bone_transforms]
        else:
            self._transforms = [
                BoneTransform.identity() for _ in range(skeleton.bone_count)
            ]

    @property
    def skeleton(self) -> Skeleton:
        """Get the associated skeleton."""
        return self._skeleton

    @property
    def space(self) -> PoseSpace:
        """Get the coordinate space."""
        return self._space

    @property
    def bone_transforms(self) -> List[BoneTransform]:
        """Get the list of bone transforms (read-only view)."""
        return list(self._transforms)

    @property
    def bone_count(self) -> int:
        """Get the number of bones in the pose."""
        return len(self._transforms)

    def get_bone_transform(self, bone_index: int) -> BoneTransform:
        """Get the transform for a specific bone.

        Args:
            bone_index: Index of the bone.

        Returns:
            Copy of the bone's transform.

        Raises:
            IndexError: If bone index is out of range.
        """
        if bone_index < 0 or bone_index >= len(self._transforms):
            raise IndexError(
                f"Bone index {bone_index} out of range [0, {len(self._transforms)})"
            )
        return self._transforms[bone_index].copy()

    def set_bone_transform(
        self,
        bone_index: int,
        transform: BoneTransform,
    ) -> None:
        """Set the transform for a specific bone.

        Args:
            bone_index: Index of the bone.
            transform: New transform value.

        Raises:
            IndexError: If bone index is out of range.
        """
        if bone_index < 0 or bone_index >= len(self._transforms):
            raise IndexError(
                f"Bone index {bone_index} out of range [0, {len(self._transforms)})"
            )
        self._transforms[bone_index] = transform.copy()

    def get_bone_transform_by_name(self, name: str) -> Optional[BoneTransform]:
        """Get transform by bone name.

        Args:
            name: Bone name.

        Returns:
            Copy of the transform, or None if bone not found.
        """
        bone = self._skeleton.get_bone_by_name(name)
        if bone is None:
            return None
        return self._transforms[bone.index].copy()

    def set_bone_transform_by_name(
        self,
        name: str,
        transform: BoneTransform,
    ) -> bool:
        """Set transform by bone name.

        Args:
            name: Bone name.
            transform: New transform value.

        Returns:
            True if bone was found and set, False otherwise.
        """
        bone = self._skeleton.get_bone_by_name(name)
        if bone is None:
            return False
        self._transforms[bone.index] = transform.copy()
        return True

    def reset_to_bind_pose(self) -> None:
        """Reset all transforms to the skeleton's bind pose."""
        for i, bone in enumerate(self._skeleton):
            self._transforms[i] = BoneTransform.from_transform(bone.local_bind_pose)
        self._space = PoseSpace.LOCAL

    def reset_to_identity(self) -> None:
        """Reset all transforms to identity."""
        for i in range(len(self._transforms)):
            self._transforms[i] = BoneTransform.identity()

    def to_local_space(self) -> Pose:
        """Convert this pose to local space.

        Returns:
            New pose in local space, or self if already local.
        """
        if self._space == PoseSpace.LOCAL:
            return self.copy()

        local_transforms = []
        model_matrices = self._compute_model_matrices()

        for i, bone in enumerate(self._skeleton):
            if bone.is_root():
                # Root bone: local = model
                local_transforms.append(self._transforms[i].copy())
            else:
                # Child bone: local = inverse(parent_model) * model
                parent_matrix = model_matrices[bone.parent_index]
                model_matrix = model_matrices[i]
                local_matrix = parent_matrix.inverse() @ model_matrix
                local_transform = Transform.from_matrix(local_matrix)
                local_transforms.append(BoneTransform.from_transform(local_transform))

        return Pose(self._skeleton, PoseSpace.LOCAL, local_transforms)

    def to_model_space(self) -> Pose:
        """Convert this pose to model space.

        Returns:
            New pose in model space, or self if already model.
        """
        if self._space == PoseSpace.MODEL:
            return self.copy()

        model_matrices = self._compute_model_matrices()
        model_transforms = []

        for matrix in model_matrices:
            transform = Transform.from_matrix(matrix)
            model_transforms.append(BoneTransform.from_transform(transform))

        return Pose(self._skeleton, PoseSpace.MODEL, model_transforms)

    def _compute_model_matrices(self) -> List[Mat4]:
        """Compute model-space matrices from current transforms."""
        if self._space == PoseSpace.MODEL:
            return [t.to_transform().to_matrix() for t in self._transforms]

        model_matrices: List[Mat4] = []

        for bone in self._skeleton:
            local_matrix = self._transforms[bone.index].to_transform().to_matrix()

            if bone.is_root():
                model_matrices.append(local_matrix)
            else:
                parent_matrix = model_matrices[bone.parent_index]
                model_matrices.append(parent_matrix @ local_matrix)

        return model_matrices

    def get_world_matrices(self) -> List[Mat4]:
        """Get world-space matrices for all bones.

        Returns:
            List of 4x4 matrices in world space.
        """
        return self._compute_model_matrices()

    def get_skinning_matrices(self) -> List[Mat4]:
        """Get skinning matrices for vertex transformation.

        Returns:
            List of skinning matrices (world * inverse_bind).
        """
        world_matrices = self._compute_model_matrices()
        return self._skeleton.compute_skinning_matrices(world_matrices)

    def copy(self) -> Pose:
        """Create a deep copy of this pose.

        Returns:
            New pose with copied transforms.
        """
        return Pose(
            skeleton=self._skeleton,
            space=self._space,
            bone_transforms=[t.copy() for t in self._transforms],
        )

    def __repr__(self) -> str:
        return (
            f"Pose(skeleton='{self._skeleton.name}', "
            f"bones={len(self._transforms)}, space={self._space.name})"
        )


def lerp_poses(pose_a: Pose, pose_b: Pose, alpha: float) -> Pose:
    """Linearly interpolate between two poses.

    Both poses must reference the same skeleton and be in the same space.

    Args:
        pose_a: First pose (alpha=0).
        pose_b: Second pose (alpha=1).
        alpha: Interpolation factor [0, 1].

    Returns:
        Interpolated pose.

    Raises:
        ValueError: If poses have different skeletons or spaces.
    """
    if pose_a.skeleton is not pose_b.skeleton:
        raise ValueError("Cannot lerp poses with different skeletons")
    if pose_a.space != pose_b.space:
        raise ValueError(
            f"Cannot lerp poses with different spaces: "
            f"{pose_a.space.name} vs {pose_b.space.name}"
        )

    # Clamp alpha
    alpha = max(0.0, min(1.0, alpha))

    if alpha == 0.0:
        return pose_a.copy()
    if alpha == 1.0:
        return pose_b.copy()

    blended_transforms = []
    for i in range(pose_a.bone_count):
        transform_a = pose_a._transforms[i]
        transform_b = pose_b._transforms[i]
        blended = transform_a.lerp(transform_b, alpha)
        blended_transforms.append(blended)

    return Pose(
        skeleton=pose_a.skeleton,
        space=pose_a.space,
        bone_transforms=blended_transforms,
    )


def additive_blend(
    base_pose: Pose,
    additive_pose: Pose,
    weight: float = 1.0,
) -> Pose:
    """Apply an additive pose on top of a base pose.

    The additive pose represents a delta from identity/reference.

    Args:
        base_pose: The base pose to add to.
        additive_pose: The additive delta pose.
        weight: How much of the additive to apply [0, 1].

    Returns:
        New pose with additive applied.

    Raises:
        ValueError: If poses have different skeletons.
    """
    if base_pose.skeleton is not additive_pose.skeleton:
        raise ValueError("Cannot blend poses with different skeletons")

    weight = max(0.0, min(1.0, weight))

    if weight == 0.0:
        return base_pose.copy()

    result_transforms = []

    for i in range(base_pose.bone_count):
        base_t = base_pose._transforms[i]
        add_t = additive_pose._transforms[i]

        # Additive blending:
        # translation: base + (additive * weight)
        # rotation: base * (identity.slerp(additive, weight))
        # scale: base * (1 + (additive - 1) * weight)

        # For translation, additive pose represents delta from zero
        new_translation = Vec3(
            base_t.translation.x + add_t.translation.x * weight,
            base_t.translation.y + add_t.translation.y * weight,
            base_t.translation.z + add_t.translation.z * weight,
        )

        # For rotation, additive represents delta from identity
        identity = Quat.identity()
        weighted_rotation = identity.slerp(add_t.rotation, weight)
        new_rotation = base_t.rotation * weighted_rotation

        # For scale, additive represents multiplier (1 = no change)
        new_scale = Vec3(
            base_t.scale.x * (1.0 + (add_t.scale.x - 1.0) * weight),
            base_t.scale.y * (1.0 + (add_t.scale.y - 1.0) * weight),
            base_t.scale.z * (1.0 + (add_t.scale.z - 1.0) * weight),
        )

        result_transforms.append(
            BoneTransform(
                translation=new_translation,
                rotation=new_rotation,
                scale=new_scale,
            )
        )

    return Pose(
        skeleton=base_pose.skeleton,
        space=base_pose.space,
        bone_transforms=result_transforms,
    )


def compute_additive_pose(
    reference_pose: Pose,
    target_pose: Pose,
) -> Pose:
    """Compute an additive pose (delta) between two poses.

    The result represents what needs to be added to reference to get target.

    Args:
        reference_pose: The reference pose (usually bind pose).
        target_pose: The target pose.

    Returns:
        Additive pose representing the delta.

    Raises:
        ValueError: If poses have different skeletons.
    """
    if reference_pose.skeleton is not target_pose.skeleton:
        raise ValueError("Cannot compute additive from poses with different skeletons")

    additive_transforms = []

    for i in range(reference_pose.bone_count):
        ref_t = reference_pose._transforms[i]
        tgt_t = target_pose._transforms[i]

        # Compute delta:
        # translation delta: target - reference
        # rotation delta: inverse(reference) * target
        # scale delta: target / reference (as multiplier relative to 1)

        delta_translation = Vec3(
            tgt_t.translation.x - ref_t.translation.x,
            tgt_t.translation.y - ref_t.translation.y,
            tgt_t.translation.z - ref_t.translation.z,
        )

        delta_rotation = ref_t.rotation.inverse() * tgt_t.rotation

        # Scale delta as ratio (1 = no change) - use epsilon to prevent division by near-zero
        delta_scale = Vec3(
            tgt_t.scale.x / ref_t.scale.x if abs(ref_t.scale.x) > SCALE_EPSILON else 1.0,
            tgt_t.scale.y / ref_t.scale.y if abs(ref_t.scale.y) > SCALE_EPSILON else 1.0,
            tgt_t.scale.z / ref_t.scale.z if abs(ref_t.scale.z) > SCALE_EPSILON else 1.0,
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


def blend_multiple_poses(
    poses: List[Pose],
    weights: List[float],
) -> Pose:
    """Blend multiple poses with normalized weights.

    Args:
        poses: List of poses to blend.
        weights: List of weights (will be normalized).

    Returns:
        Blended pose.

    Raises:
        ValueError: If poses list is empty, weights don't match, or different skeletons.
    """
    if not poses:
        raise ValueError("Cannot blend empty pose list")
    if len(poses) != len(weights):
        raise ValueError(
            f"Pose count ({len(poses)}) must match weight count ({len(weights)})"
        )

    # Check all poses have same skeleton
    skeleton = poses[0].skeleton
    space = poses[0].space
    for pose in poses[1:]:
        if pose.skeleton is not skeleton:
            raise ValueError("All poses must have the same skeleton")
        if pose.space != space:
            raise ValueError("All poses must be in the same space")

    # Normalize weights
    total_weight = sum(max(0.0, w) for w in weights)
    if total_weight < WEIGHT_EPSILON:
        # All weights are zero, return copy of first pose
        return poses[0].copy()

    normalized_weights = [max(0.0, w) / total_weight for w in weights]

    # Blend transforms
    result_transforms = []

    for bone_idx in range(skeleton.bone_count):
        # Start with first pose weighted
        first_weight = normalized_weights[0]
        first_transform = poses[0]._transforms[bone_idx]

        blended_translation = Vec3(
            first_transform.translation.x * first_weight,
            first_transform.translation.y * first_weight,
            first_transform.translation.z * first_weight,
        )
        blended_scale = Vec3(
            first_transform.scale.x * first_weight,
            first_transform.scale.y * first_weight,
            first_transform.scale.z * first_weight,
        )

        # For quaternion, we'll accumulate using slerp
        blended_rotation = first_transform.rotation

        accumulated_weight = first_weight

        # Add remaining poses
        for pose_idx in range(1, len(poses)):
            weight = normalized_weights[pose_idx]
            if weight < WEIGHT_EPSILON:
                continue

            transform = poses[pose_idx]._transforms[bone_idx]

            # Accumulate translation and scale
            blended_translation = Vec3(
                blended_translation.x + transform.translation.x * weight,
                blended_translation.y + transform.translation.y * weight,
                blended_translation.z + transform.translation.z * weight,
            )
            blended_scale = Vec3(
                blended_scale.x + transform.scale.x * weight,
                blended_scale.y + transform.scale.y * weight,
                blended_scale.z + transform.scale.z * weight,
            )

            # Slerp rotation with accumulated weight
            accumulated_weight += weight
            slerp_factor = weight / accumulated_weight
            blended_rotation = blended_rotation.slerp(
                transform.rotation, slerp_factor
            )

        result_transforms.append(
            BoneTransform(
                translation=blended_translation,
                rotation=blended_rotation,
                scale=blended_scale,
            )
        )

    return Pose(
        skeleton=skeleton,
        space=space,
        bone_transforms=result_transforms,
    )


class PoseBuffer:
    """Buffer for storing and managing multiple poses.

    Useful for animation playback where you need to store
    intermediate poses for blending.
    """

    def __init__(self, skeleton: Skeleton, capacity: int = 4) -> None:
        """Initialize pose buffer.

        Args:
            skeleton: The skeleton for all poses.
            capacity: Maximum number of poses to store.
        """
        self._skeleton = skeleton
        self._capacity = max(1, capacity)
        self._poses: List[Pose] = []

    @property
    def skeleton(self) -> Skeleton:
        """Get the skeleton."""
        return self._skeleton

    @property
    def capacity(self) -> int:
        """Get buffer capacity."""
        return self._capacity

    @property
    def count(self) -> int:
        """Get number of stored poses."""
        return len(self._poses)

    def push(self, pose: Pose) -> None:
        """Add a pose to the buffer.

        If at capacity, removes oldest pose.

        Args:
            pose: Pose to add.

        Raises:
            ValueError: If pose has different skeleton.
        """
        if pose.skeleton is not self._skeleton:
            raise ValueError("Pose must use the same skeleton as buffer")

        self._poses.append(pose.copy())

        while len(self._poses) > self._capacity:
            self._poses.pop(0)

    def get(self, index: int) -> Pose:
        """Get a pose by index.

        Args:
            index: Pose index (0 = oldest).

        Returns:
            Copy of the pose.

        Raises:
            IndexError: If index out of range.
        """
        if index < 0 or index >= len(self._poses):
            raise IndexError(f"Pose index {index} out of range [0, {len(self._poses)})")
        return self._poses[index].copy()

    def get_latest(self) -> Optional[Pose]:
        """Get the most recent pose.

        Returns:
            Copy of latest pose, or None if empty.
        """
        if not self._poses:
            return None
        return self._poses[-1].copy()

    def clear(self) -> None:
        """Clear all stored poses."""
        self._poses.clear()

    def __len__(self) -> int:
        return len(self._poses)

    def __repr__(self) -> str:
        return f"PoseBuffer(skeleton='{self._skeleton.name}', count={len(self._poses)}/{self._capacity})"
