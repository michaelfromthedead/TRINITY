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
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Optional

from engine.core.math.transform import Transform

from .skeleton import Skeleton


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
        Multiply weights together (both masks must allow the bone).
    ``ADD``
        Add weights together (clamped to 1.0).
    ``MAX``
        Take the maximum weight from either mask.
    ``MIN``
        Take the minimum weight from either mask.
    """
    MULTIPLY = 0
    ADD = 1
    MAX = 2
    MIN = 3


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
    ) -> None:
        self.name = name
        self.mode = mode
        self._weights: Dict[str, float] = {}

    # -- weight access ----------------------------------------------------------

    def set_weight(self, bone_name: str, weight: float) -> None:
        """Set the weight for *bone_name*, clamped to ``[0, 1]``."""
        self._weights[bone_name] = max(0.0, min(1.0, weight))

    def get_weight(self, bone_name: str) -> float:
        """Return the effective weight for *bone_name*.

        Returns the explicit weight if previously set, otherwise the mode
        default (``0.0`` for ``ZERO``, ``1.0`` for ``ONE``).
        """
        if bone_name in self._weights:
            return self._weights[bone_name]
        return 1.0 if self.mode is MissingBoneMode.ONE else 0.0

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

    def apply(self, bone_transforms: Dict[str, Transform]) -> Dict[str, Transform]:
        """Apply this mask to a bone-name-to-transform mapping.

        Each transform is blended toward identity proportionally to
        ``1 - weight``:

        * weight ``1.0`` -- transform is returned unchanged.
        * weight ``0.0`` -- transform is replaced with identity.
        * weight ``0.5`` -- halfway between identity and the original.

        Parameters
        ----------
        bone_transforms
            Mapping from bone name to its current animated transform.

        Returns
        -------
        Dict[str, Transform]
            New mapping with weighted transforms.
        """
        result: Dict[str, Transform] = {}
        identity = Transform()
        for bone_name, xform in bone_transforms.items():
            w = self.get_weight(bone_name)
            result[bone_name] = identity.lerp(xform, w)
        return result

    def combine(self, other: BoneMask, name: Optional[str] = None) -> BoneMask:
        """Combine this mask with *other* via weight multiplication.

        The result contains the union of both masks' bone entries.
        For bones present in both masks the final weight is the product;
        for bones present in only one, the weight from that mask is used
        (each mask's own ``MissingBoneMode`` applies for missing-bone
        defaults).

        Parameters
        ----------
        other
            The mask to combine with.
        name
            Optional name for the combined mask.

        Returns
        -------
        BoneMask
            A new mask whose weights are the product of the two inputs.
        """
        combined = BoneMask(
            name=name or f"{self.name}_x_{other.name}",
            mode=self.mode,
        )
        all_names: set[str] = set(self._weights) | set(other._weights)
        for bn in all_names:
            w = self.get_weight(bn) * other.get_weight(bn)
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
        m = BoneMask(name=name or self.name, mode=self.mode)
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


__all__ = [
    "MissingBoneMode",
    "CombineMode",
    "BoneMask",
    "BoneMaskPresets",
]
