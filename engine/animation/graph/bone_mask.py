"""Bone mask system for partial animation blending.

Provides BoneMask for per-bone weight control over animation application,
enabling partial-body effects such as upper-body attack while lower-body
walks.  BoneMaskPresets offers factory methods for common humanoid masks
and a gradient generator for smooth weight falloff.

Integration
-----------
Bone names reference the hierarchical Skeleton from ``skeleton.py``
(T-AG-1.3).  Mask presets traverse the bone hierarchy to build region
masks (upper body, left arm, etc.) without hard-coding bone lists when
a naming convention is already embedded in the skeleton.

Usage
-----
    from engine.animation.graph.bone_mask import BoneMask, BoneMaskPresets

    mask = BoneMaskPresets.upper_body(skeleton)
    masked = mask.apply(pose_transforms)       # Dict[str, Transform]

    # Or with Pose object:
    from engine.animation.graph.pose import Pose
    masked_pose = mask.apply(pose)             # Pose -> Pose

    # Combine masks with different modes:
    combined = mask1.combine(mask2, mode='multiply')  # Default
    combined = mask1.combine(mask2, mode='add')
    combined = mask1.combine(mask2, mode='max')
    combined = mask1.combine(mask2, mode='min')
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from engine.core.math.transform import Transform

from .skeleton import Skeleton

if TYPE_CHECKING:
    from .pose import Pose as DictPose


class MissingBoneMode(Enum):
    """How bones not explicitly listed in the mask are handled.

    ``ZERO``
        Missing bones default to weight 0.0 (unaffected by the mask).
    ``ONE``
        Missing bones default to weight 1.0 (fully affected by the mask).
    """
    ZERO = 0
    ONE = 1


class CombineMode(Enum):
    """Mode for combining two bone masks.

    ``MULTIPLY``
        Weights are multiplied (default). Good for layered masking.
    ``ADD``
        Weights are added, clamped to [0, 1]. Good for union of masks.
    ``MAX``
        Maximum weight is taken. Good for logical OR of masks.
    ``MIN``
        Minimum weight is taken. Good for logical AND of masks.
    """
    MULTIPLY = "multiply"
    ADD = "add"
    MAX = "max"
    MIN = "min"


class BoneMask:
    """Per-bone weight mask controlling partial animation application.

    Each bone is assigned a weight in ``[0, 1]``:

    * ``1.0`` -- fully affected (the animation value is used as-is).
    * ``0.0`` -- not affected (the bone stays at rest / identity).
    * Intermediate values blend between identity and the animation value.

    Bone names must match those registered on the ``Skeleton`` instance
    passed to factory methods.
    """

    def __init__(
        self,
        name: str = "mask",
        mode: MissingBoneMode = MissingBoneMode.ZERO,
        default_weight: Optional[float] = None,
    ) -> None:
        """Initialize a bone mask.

        Parameters
        ----------
        name : str
            Identifier for this mask.
        mode : MissingBoneMode
            How to handle bones not in the weights dict. Ignored if
            ``default_weight`` is explicitly provided.
        default_weight : float or None
            Explicit default weight for bones not in the weights dict.
            If provided, overrides the ``mode`` setting. Must be in [0, 1].
        """
        self.name = name
        self.mode = mode
        self._weights: Dict[str, float] = {}
        # Store explicit default if provided, else derive from mode
        if default_weight is not None:
            self._default_weight: float = max(0.0, min(1.0, default_weight))
        else:
            self._default_weight = 1.0 if mode is MissingBoneMode.ONE else 0.0

    # -- weight access ----------------------------------------------------------

    def set_weight(self, bone_name: str, weight: float) -> None:
        """Set the weight for *bone_name*, clamped to ``[0, 1]``."""
        self._weights[bone_name] = max(0.0, min(1.0, weight))

    @property
    def default_weight(self) -> float:
        """The weight applied to bones not explicitly in the mask."""
        return self._default_weight

    @default_weight.setter
    def default_weight(self, value: float) -> None:
        """Set the default weight for bones not in the mask."""
        self._default_weight = max(0.0, min(1.0, value))

    def get_weight(self, bone_name: str) -> float:
        """Return the effective weight for *bone_name*.

        Returns the explicit weight if previously set, otherwise the
        default weight (derived from ``mode`` or explicit ``default_weight``).
        """
        if bone_name in self._weights:
            return self._weights[bone_name]
        return self._default_weight

    @property
    def weights(self) -> Dict[str, float]:
        """Read-only snapshot of the explicit weight entries."""
        return dict(self._weights)

    def has_bone(self, bone_name: str) -> bool:
        """Return ``True`` when *bone_name* has an explicit entry."""
        return bone_name in self._weights

    @property
    def bone_count(self) -> int:
        """Number of bones with explicit weight entries."""
        return len(self._weights)

    # -- core mask operations ---------------------------------------------------

    def apply(
        self, pose: Union[Dict[str, Transform], "DictPose"]
    ) -> Union[Dict[str, Transform], "DictPose"]:
        """Apply this mask to a pose or bone-name-to-transform mapping.

        Each transform is blended toward identity proportionally to
        ``1 - weight``:

        * weight ``1.0`` -- transform is returned unchanged.
        * weight ``0.0`` -- transform is replaced with identity.
        * weight ``0.5`` -- halfway between identity and the original.

        Parameters
        ----------
        pose
            Either a ``Dict[str, Transform]`` mapping bone names to transforms,
            or a ``Pose`` object from ``pose.py``.

        Returns
        -------
        Dict[str, Transform] or Pose
            New mapping/Pose with weighted transforms. Returns the same type
            as the input.
        """
        # Import Pose here to avoid circular imports
        from .pose import Pose as DictPose, Transform as PoseTransform

        # Check if input is a Pose object
        is_pose_object = isinstance(pose, DictPose)

        if is_pose_object:
            bone_transforms = pose.bone_transforms
        else:
            bone_transforms = pose

        result: Dict[str, Transform] = {}

        # Use the appropriate identity transform
        if is_pose_object:
            identity = PoseTransform.identity()
            for bone_name, xform in bone_transforms.items():
                w = self.get_weight(bone_name)
                result[bone_name] = identity.blend(xform, w)
            return DictPose(bone_transforms=result)
        else:
            identity = Transform()
            for bone_name, xform in bone_transforms.items():
                w = self.get_weight(bone_name)
                result[bone_name] = identity.lerp(xform, w)
            return result

    def combine(
        self,
        other: BoneMask,
        mode: str = "multiply",
        name: Optional[str] = None,
    ) -> BoneMask:
        """Combine this mask with *other* using the specified mode.

        The result contains the union of both masks' bone entries.
        Each mask's own ``MissingBoneMode`` or ``default_weight`` applies
        for missing-bone defaults during combination.

        Parameters
        ----------
        other
            The mask to combine with.
        mode
            Combine mode: ``'multiply'`` (default), ``'add'``, ``'max'``,
            or ``'min'``. Can also be a ``CombineMode`` enum value.
        name
            Optional name for the combined mask.

        Returns
        -------
        BoneMask
            A new mask with combined weights.

        Examples
        --------
        >>> combined = mask1.combine(mask2, mode='multiply')  # Layer masking
        >>> combined = mask1.combine(mask2, mode='add')       # Union
        >>> combined = mask1.combine(mask2, mode='max')       # Logical OR
        >>> combined = mask1.combine(mask2, mode='min')       # Logical AND
        """
        # Normalize mode to string
        if isinstance(mode, CombineMode):
            mode_str = mode.value
        else:
            mode_str = mode.lower()

        # Build name suffix based on mode
        mode_suffix = {
            "multiply": "x",
            "add": "+",
            "max": "max",
            "min": "min",
        }.get(mode_str, "x")

        combined = BoneMask(
            name=name or f"{self.name}_{mode_suffix}_{other.name}",
            mode=self.mode,
            default_weight=self._default_weight,
        )

        all_names: set[str] = set(self._weights) | set(other._weights)

        for bn in all_names:
            w1 = self.get_weight(bn)
            w2 = other.get_weight(bn)

            if mode_str == "multiply":
                w = w1 * w2
            elif mode_str == "add":
                w = w1 + w2
            elif mode_str == "max":
                w = max(w1, w2)
            elif mode_str == "min":
                w = min(w1, w2)
            else:
                # Fallback to multiply for unknown modes
                w = w1 * w2

            combined._weights[bn] = max(0.0, min(1.0, w))

        return combined

    def invert(self, name: Optional[str] = None) -> BoneMask:
        """Return a mask where every explicit weight is ``1 - weight``.

        Parameters
        ----------
        name
            Optional name for the inverted mask.

        Returns
        -------
        BoneMask
            New mask with inverted weights.
        """
        inv = BoneMask(name=name or f"not_{self.name}", mode=self.mode)
        for bn, w in self._weights.items():
            inv._weights[bn] = 1.0 - w
        return inv

    # -- factory helpers --------------------------------------------------------

    @classmethod
    def full(cls, skeleton: Skeleton, name: str = "full") -> BoneMask:
        """Create a mask with every skeleton bone at weight ``1.0``.

        Parameters
        ----------
        skeleton
            The skeleton whose bones will be added.
        name
            Name for the mask.

        Returns
        -------
        BoneMask
            A mask where every bone has weight ``1.0``.
        """
        mask = cls(name=name, mode=MissingBoneMode.ONE)
        for bone in skeleton:
            mask._weights[bone.name] = 1.0
        return mask

    @classmethod
    def from_bone_names(
        cls,
        skeleton: Skeleton,
        name: str,
        bone_names: List[str],
        weight: float = 1.0,
        include_children: bool = False,
    ) -> BoneMask:
        """Create a mask from an explicit list of bone names.

        Bones that do not exist in *skeleton* are silently skipped.

        Parameters
        ----------
        skeleton
            The skeleton to validate bone names against.
        name
            Name for the mask.
        bone_names
            Names of bones to include.
        weight
            Weight value for all named bones (default ``1.0``).
        include_children
            When ``True``, every descendant of each named bone is also
            included at the same weight.

        Returns
        -------
        BoneMask
            Mask with the named (and optionally descendant) bones set.
        """
        mask = cls(name=name, mode=MissingBoneMode.ZERO)
        w = max(0.0, min(1.0, weight))
        for bn in bone_names:
            b = skeleton.get_bone(bn)
            if b is not None:
                mask._weights[bn] = w
                if include_children:
                    for child in b.get_descendants():
                        mask._weights[child.name] = w
        return mask

    # -- copy / representation --------------------------------------------------

    def copy(self, name: Optional[str] = None) -> BoneMask:
        """Create an independent copy of this mask.

        Parameters
        ----------
        name
            Optional new name for the copy.

        Returns
        -------
        BoneMask
            A deep-ish copy (weight dict is duplicated; strings are
            immutable so sharing is safe).
        """
        m = BoneMask(
            name=name or self.name,
            mode=self.mode,
            default_weight=self._default_weight,
        )
        m._weights = dict(self._weights)
        return m

    def __repr__(self) -> str:
        return (
            f"BoneMask('{self.name}', "
            f"bones={self.bone_count}, "
            f"mode={self.mode.name})"
        )


# =============================================================================
# BONE MASK PRESETS
# =============================================================================

# Standard humanoid bone name conventions (UE4 / Mixamo compatible).
_UPPER_BODY_BONES = [
    "Spine", "Spine1", "Spine2", "Chest",
    "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
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

_LOWER_BODY_BONES = [
    "Hips", "Pelvis",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
]

_LEFT_ARM_BONES = [
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
]

_RIGHT_ARM_BONES = [
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
]

_LEFT_LEG_BONES = [
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
]

_RIGHT_LEG_BONES = [
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
]


class BoneMaskPresets:
    """Factory for common BoneMask configurations.

    Each static method returns a pre-built ``BoneMask`` for the specified
    skeleton.  Bone naming follows the UE4 / Mixayo convention, which is
    also compatible with the ``skeleton.py`` hierarchical Skeleton.

    Presets
    -------
    ``upper_body``
        Spine, chest, neck, head, both arms (including fingers).
    ``lower_body``
        Hips, pelvis, both legs (including feet and toes).
    ``left_arm``
        Left shoulder through hand and fingers.
    ``right_arm``
        Right shoulder through hand and fingers.
    ``left_leg``
        Left upper leg through foot and toe base.
    ``right_leg``
        Right upper leg through foot and toe base.
    ``gradient``
        Smooth weight falloff from a start bone through its descendants.
    """

    @staticmethod
    def upper_body(skeleton: Skeleton) -> BoneMask:
        """Create an upper-body mask (spine, chest, neck, head, arms)."""
        return BoneMask.from_bone_names(
            skeleton, "UpperBody", _UPPER_BODY_BONES,
            weight=1.0, include_children=True,
        )

    @staticmethod
    def lower_body(skeleton: Skeleton) -> BoneMask:
        """Create a lower-body mask (hips, pelvis, legs)."""
        return BoneMask.from_bone_names(
            skeleton, "LowerBody", _LOWER_BODY_BONES,
            weight=1.0, include_children=True,
        )

    @staticmethod
    def left_arm(skeleton: Skeleton) -> BoneMask:
        """Create a left-arm mask."""
        return BoneMask.from_bone_names(
            skeleton, "LeftArm", _LEFT_ARM_BONES,
            weight=1.0, include_children=True,
        )

    @staticmethod
    def right_arm(skeleton: Skeleton) -> BoneMask:
        """Create a right-arm mask."""
        return BoneMask.from_bone_names(
            skeleton, "RightArm", _RIGHT_ARM_BONES,
            weight=1.0, include_children=True,
        )

    @staticmethod
    def left_leg(skeleton: Skeleton) -> BoneMask:
        """Create a left-leg mask."""
        return BoneMask.from_bone_names(
            skeleton, "LeftLeg", _LEFT_LEG_BONES,
            weight=1.0, include_children=True,
        )

    @staticmethod
    def right_leg(skeleton: Skeleton) -> BoneMask:
        """Create a right-leg mask."""
        return BoneMask.from_bone_names(
            skeleton, "RightLeg", _RIGHT_LEG_BONES,
            weight=1.0, include_children=True,
        )

    @staticmethod
    def gradient(
        skeleton: Skeleton,
        start_bone: str,
        falloff: str = "linear",
        rate: float = 1.0,
        root_weight: float = 0.0,
        tip_weight: float = 1.0,
    ) -> BoneMask:
        """Create a mask with smooth weight falloff through the hierarchy.

        Weights progress from *root_weight* at *start_bone* to
        *tip_weight* at the farthest descendant, producing a smooth
        gradient suited for transitional blending (e.g. spine gradient
        for upper/lower body cross-fade).

        Parameters
        ----------
        skeleton
            The bone hierarchy.
        start_bone
            Name of the bone where the gradient begins.  All descendants
            are included.
        falloff
            ``"linear"`` or ``"exponential"``.
        rate
            Steepness factor for exponential falloff (ignored for
            ``"linear"``).
        root_weight
            Weight at *start_bone* (default ``0.0``).
        tip_weight
            Weight at the farthest descendant (default ``1.0``).

        Returns
        -------
        BoneMask
            A mask whose weights transition smoothly from *root_weight*
            to *tip_weight* across the bone hierarchy.
        """
        mask = BoneMask(name=f"gradient_{start_bone}")
        bone = skeleton.get_bone(start_bone)
        if bone is None:
            return mask

        chain = [bone] + bone.get_descendants()
        start_depth = bone.depth
        max_rel_depth = max(b.depth - start_depth for b in chain)

        for b in chain:
            rel = b.depth - start_depth
            t = rel / max_rel_depth if max_rel_depth > 0 else 0.0
            if falloff == "exponential":
                t = 1.0 - math.exp(-rate * t)
            w = root_weight + (tip_weight - root_weight) * t
            mask._weights[b.name] = max(0.0, min(1.0, w))

        return mask

    @staticmethod
    def create_gradient(
        skeleton: Skeleton,
        root: str,
        leaves: List[str],
        falloff: str = "linear",
        rate: float = 1.0,
    ) -> BoneMask:
        """Create a mask with smooth weight falloff from root to leaf bones.

        This factory method creates a gradient mask that assigns weights
        based on distance from *root* to each of the *leaves*. Bones are
        assigned weights proportional to their position in the chain from
        root to leaf.

        Parameters
        ----------
        skeleton
            The bone hierarchy.
        root
            Name of the root bone where the gradient begins (weight 0.0).
        leaves
            List of leaf bone names where the gradient ends (weight 1.0).
            The gradient is computed along the chain from root to each leaf.
        falloff
            ``"linear"`` (default) or ``"exponential"``.
        rate
            Steepness factor for exponential falloff (ignored for linear).

        Returns
        -------
        BoneMask
            A mask whose weights transition smoothly from 0.0 at *root*
            to 1.0 at each leaf, with intermediate bones weighted by their
            relative position in the chain.

        Examples
        --------
        >>> # Gradient from spine to hands
        >>> mask = BoneMaskPresets.create_gradient(
        ...     skeleton,
        ...     root="Spine",
        ...     leaves=["LeftHand", "RightHand"],
        ...     falloff="linear"
        ... )
        """
        mask = BoneMask(name=f"gradient_{root}_to_leaves")
        root_bone = skeleton.get_bone(root)
        if root_bone is None:
            return mask

        # Track maximum weight per bone (in case multiple paths)
        bone_weights: Dict[str, float] = {}

        for leaf_name in leaves:
            # Get the chain from root to leaf
            chain = skeleton.get_chain(root, leaf_name)
            if not chain:
                continue

            chain_length = len(chain) - 1  # Number of steps
            if chain_length <= 0:
                # Root == leaf
                bone_weights[root] = 1.0
                continue

            for i, bone in enumerate(chain):
                # Compute normalized position in chain [0, 1]
                t = i / chain_length

                # Apply falloff function
                if falloff == "exponential":
                    t = 1.0 - math.exp(-rate * t * 3.0)  # Scale for visibility

                # Keep maximum weight if bone appears in multiple chains
                existing = bone_weights.get(bone.name, 0.0)
                bone_weights[bone.name] = max(existing, t)

        # Clamp and store weights
        for bone_name, w in bone_weights.items():
            mask._weights[bone_name] = max(0.0, min(1.0, w))

        return mask


__all__ = [
    "MissingBoneMode",
    "CombineMode",
    "BoneMask",
    "BoneMaskPresets",
]
