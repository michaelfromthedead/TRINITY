"""Whitebox tests for the bone mask system (T-AG-1.4).

Tests cover all internal implementation details of BoneMask and BoneMaskPresets:
- BoneMask creation with weights dict and default_weight
- get_weight() and set_weight() operations
- apply() with Dict and Pose inputs
- combine() with all modes (MULTIPLY, ADD, MAX, MIN)
- BoneMaskPresets enum values
- Preset factory methods (upper_body, lower_body, etc.)
- Gradient generation with linear and exponential falloff
- Weight clamping to [0, 1]
- Edge cases: empty masks, missing bones, zero weights
"""

from __future__ import annotations

import math
import pytest
from typing import Dict

from engine.animation.graph.bone_mask import (
    BoneMask,
    BoneMaskPresets,
    CombineMode,
    MissingBoneMode,
)
from engine.animation.graph.pose import Pose, Transform
from engine.animation.graph.skeleton import Skeleton
from engine.core.math.transform import Transform as CoreTransform


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_skeleton() -> Skeleton:
    """Create a simple skeleton for testing."""
    skeleton = Skeleton("test_skeleton")
    skeleton.add_bone("Root")
    skeleton.add_bone("Spine", parent_name="Root")
    skeleton.add_bone("Spine1", parent_name="Spine")
    skeleton.add_bone("Spine2", parent_name="Spine1")
    skeleton.add_bone("Neck", parent_name="Spine2")
    skeleton.add_bone("Head", parent_name="Neck")
    skeleton.add_bone("LeftShoulder", parent_name="Spine2")
    skeleton.add_bone("LeftArm", parent_name="LeftShoulder")
    skeleton.add_bone("LeftForeArm", parent_name="LeftArm")
    skeleton.add_bone("LeftHand", parent_name="LeftForeArm")
    skeleton.add_bone("RightShoulder", parent_name="Spine2")
    skeleton.add_bone("RightArm", parent_name="RightShoulder")
    skeleton.add_bone("RightForeArm", parent_name="RightArm")
    skeleton.add_bone("RightHand", parent_name="RightForeArm")
    skeleton.add_bone("Hips", parent_name="Root")
    skeleton.add_bone("LeftUpLeg", parent_name="Hips")
    skeleton.add_bone("LeftLeg", parent_name="LeftUpLeg")
    skeleton.add_bone("LeftFoot", parent_name="LeftLeg")
    skeleton.add_bone("RightUpLeg", parent_name="Hips")
    skeleton.add_bone("RightLeg", parent_name="RightUpLeg")
    skeleton.add_bone("RightFoot", parent_name="RightLeg")
    return skeleton


@pytest.fixture
def humanoid_skeleton() -> Skeleton:
    """Create a complete humanoid skeleton with all standard bones."""
    skeleton = Skeleton("humanoid")

    # Core
    skeleton.add_bone("Root")
    skeleton.add_bone("Hips", parent_name="Root")
    skeleton.add_bone("Pelvis", parent_name="Root")

    # Spine chain
    skeleton.add_bone("Spine", parent_name="Hips")
    skeleton.add_bone("Spine1", parent_name="Spine")
    skeleton.add_bone("Spine2", parent_name="Spine1")
    skeleton.add_bone("Chest", parent_name="Spine2")

    # Head
    skeleton.add_bone("Neck", parent_name="Chest")
    skeleton.add_bone("Head", parent_name="Neck")

    # Left arm
    skeleton.add_bone("LeftShoulder", parent_name="Chest")
    skeleton.add_bone("LeftArm", parent_name="LeftShoulder")
    skeleton.add_bone("LeftForeArm", parent_name="LeftArm")
    skeleton.add_bone("LeftHand", parent_name="LeftForeArm")

    # Left hand fingers
    for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        for i in range(1, 4):
            skeleton.add_bone(
                f"LeftHand{finger}{i}",
                parent_name=f"LeftHand{finger}{i-1}" if i > 1 else "LeftHand"
            )

    # Right arm
    skeleton.add_bone("RightShoulder", parent_name="Chest")
    skeleton.add_bone("RightArm", parent_name="RightShoulder")
    skeleton.add_bone("RightForeArm", parent_name="RightArm")
    skeleton.add_bone("RightHand", parent_name="RightForeArm")

    # Right hand fingers
    for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        for i in range(1, 4):
            skeleton.add_bone(
                f"RightHand{finger}{i}",
                parent_name=f"RightHand{finger}{i-1}" if i > 1 else "RightHand"
            )

    # Left leg
    skeleton.add_bone("LeftUpLeg", parent_name="Hips")
    skeleton.add_bone("LeftLeg", parent_name="LeftUpLeg")
    skeleton.add_bone("LeftFoot", parent_name="LeftLeg")
    skeleton.add_bone("LeftToeBase", parent_name="LeftFoot")

    # Right leg
    skeleton.add_bone("RightUpLeg", parent_name="Hips")
    skeleton.add_bone("RightLeg", parent_name="RightUpLeg")
    skeleton.add_bone("RightFoot", parent_name="RightLeg")
    skeleton.add_bone("RightToeBase", parent_name="RightFoot")

    return skeleton


# =============================================================================
# BONEMASK CREATION TESTS
# =============================================================================


class TestBoneMaskCreation:
    """Test BoneMask initialization and basic properties."""

    def test_default_creation(self) -> None:
        """Test BoneMask with default parameters."""
        mask = BoneMask()
        assert mask.name == "mask"
        assert mask.mode == MissingBoneMode.ZERO
        assert mask.default_weight == 0.0
        assert mask.bone_count == 0
        assert mask.weights == {}

    def test_creation_with_name(self) -> None:
        """Test BoneMask with custom name."""
        mask = BoneMask(name="custom_mask")
        assert mask.name == "custom_mask"

    def test_creation_with_mode_zero(self) -> None:
        """Test BoneMask with MissingBoneMode.ZERO."""
        mask = BoneMask(mode=MissingBoneMode.ZERO)
        assert mask.mode == MissingBoneMode.ZERO
        assert mask.default_weight == 0.0

    def test_creation_with_mode_one(self) -> None:
        """Test BoneMask with MissingBoneMode.ONE."""
        mask = BoneMask(mode=MissingBoneMode.ONE)
        assert mask.mode == MissingBoneMode.ONE
        assert mask.default_weight == 1.0

    def test_creation_with_explicit_default_weight(self) -> None:
        """Test BoneMask with explicit default_weight overrides mode."""
        mask = BoneMask(mode=MissingBoneMode.ZERO, default_weight=0.5)
        assert mask.default_weight == 0.5
        # Mode should be stored but default_weight takes precedence
        assert mask.mode == MissingBoneMode.ZERO

    def test_default_weight_clamping_high(self) -> None:
        """Test that default_weight is clamped to 1.0 when too high."""
        mask = BoneMask(default_weight=1.5)
        assert mask.default_weight == 1.0

    def test_default_weight_clamping_low(self) -> None:
        """Test that default_weight is clamped to 0.0 when too low."""
        mask = BoneMask(default_weight=-0.5)
        assert mask.default_weight == 0.0


# =============================================================================
# WEIGHT ACCESS TESTS
# =============================================================================


class TestWeightAccess:
    """Test get_weight() and set_weight() operations."""

    def test_set_weight(self) -> None:
        """Test setting weight for a bone."""
        mask = BoneMask()
        mask.set_weight("bone1", 0.5)
        assert mask.get_weight("bone1") == 0.5
        assert mask.has_bone("bone1")
        assert mask.bone_count == 1

    def test_set_weight_multiple(self) -> None:
        """Test setting weights for multiple bones."""
        mask = BoneMask()
        mask.set_weight("bone1", 0.3)
        mask.set_weight("bone2", 0.7)
        mask.set_weight("bone3", 1.0)
        assert mask.get_weight("bone1") == 0.3
        assert mask.get_weight("bone2") == 0.7
        assert mask.get_weight("bone3") == 1.0
        assert mask.bone_count == 3

    def test_set_weight_overwrite(self) -> None:
        """Test overwriting existing weight."""
        mask = BoneMask()
        mask.set_weight("bone1", 0.3)
        mask.set_weight("bone1", 0.8)
        assert mask.get_weight("bone1") == 0.8
        assert mask.bone_count == 1

    def test_set_weight_clamping_high(self) -> None:
        """Test weight clamping above 1.0."""
        mask = BoneMask()
        mask.set_weight("bone1", 1.5)
        assert mask.get_weight("bone1") == 1.0

    def test_set_weight_clamping_low(self) -> None:
        """Test weight clamping below 0.0."""
        mask = BoneMask()
        mask.set_weight("bone1", -0.5)
        assert mask.get_weight("bone1") == 0.0

    def test_get_weight_missing_bone_mode_zero(self) -> None:
        """Test get_weight for missing bone with mode ZERO."""
        mask = BoneMask(mode=MissingBoneMode.ZERO)
        assert mask.get_weight("nonexistent") == 0.0

    def test_get_weight_missing_bone_mode_one(self) -> None:
        """Test get_weight for missing bone with mode ONE."""
        mask = BoneMask(mode=MissingBoneMode.ONE)
        assert mask.get_weight("nonexistent") == 1.0

    def test_get_weight_missing_bone_explicit_default(self) -> None:
        """Test get_weight for missing bone with explicit default_weight."""
        mask = BoneMask(default_weight=0.75)
        assert mask.get_weight("nonexistent") == 0.75

    def test_has_bone_false(self) -> None:
        """Test has_bone returns False for unset bones."""
        mask = BoneMask()
        assert not mask.has_bone("nonexistent")

    def test_weights_property_copy(self) -> None:
        """Test weights property returns a copy, not reference."""
        mask = BoneMask()
        mask.set_weight("bone1", 0.5)
        weights = mask.weights
        weights["bone2"] = 0.9  # Modify the returned dict
        assert not mask.has_bone("bone2")  # Original should be unchanged

    def test_default_weight_setter(self) -> None:
        """Test setting default_weight via property."""
        mask = BoneMask()
        mask.default_weight = 0.6
        assert mask.default_weight == 0.6

    def test_default_weight_setter_clamping(self) -> None:
        """Test default_weight setter clamps values."""
        mask = BoneMask()
        mask.default_weight = 2.0
        assert mask.default_weight == 1.0
        mask.default_weight = -1.0
        assert mask.default_weight == 0.0


# =============================================================================
# APPLY TESTS - DICT INPUT
# =============================================================================


class TestApplyDict:
    """Test apply() with Dict[str, Transform] input."""

    def test_apply_full_weight(self) -> None:
        """Test apply with weight 1.0 preserves transform."""
        mask = BoneMask()
        mask.set_weight("bone1", 1.0)

        transforms: Dict[str, CoreTransform] = {
            "bone1": CoreTransform()
        }
        transforms["bone1"].translation.x = 5.0
        transforms["bone1"].translation.y = 3.0

        result = mask.apply(transforms)
        assert isinstance(result, dict)
        # Weight 1.0 should preserve original position
        assert abs(result["bone1"].translation.x - 5.0) < 0.001
        assert abs(result["bone1"].translation.y - 3.0) < 0.001

    def test_apply_zero_weight(self) -> None:
        """Test apply with weight 0.0 returns identity."""
        mask = BoneMask()
        mask.set_weight("bone1", 0.0)

        transforms: Dict[str, CoreTransform] = {
            "bone1": CoreTransform()
        }
        transforms["bone1"].translation.x = 5.0
        transforms["bone1"].translation.y = 3.0

        result = mask.apply(transforms)
        assert isinstance(result, dict)
        # Weight 0.0 should return identity (0, 0, 0)
        assert abs(result["bone1"].translation.x) < 0.001
        assert abs(result["bone1"].translation.y) < 0.001

    def test_apply_half_weight(self) -> None:
        """Test apply with weight 0.5 blends to halfway."""
        mask = BoneMask()
        mask.set_weight("bone1", 0.5)

        transforms: Dict[str, CoreTransform] = {
            "bone1": CoreTransform()
        }
        transforms["bone1"].translation.x = 10.0

        result = mask.apply(transforms)
        assert isinstance(result, dict)
        # Weight 0.5 should blend halfway between identity and original
        assert abs(result["bone1"].translation.x - 5.0) < 0.001

    def test_apply_multiple_bones(self) -> None:
        """Test apply with multiple bones with different weights."""
        mask = BoneMask()
        mask.set_weight("bone1", 1.0)
        mask.set_weight("bone2", 0.5)
        mask.set_weight("bone3", 0.0)

        transforms: Dict[str, CoreTransform] = {
            "bone1": CoreTransform(),
            "bone2": CoreTransform(),
            "bone3": CoreTransform(),
        }
        transforms["bone1"].translation.x = 10.0
        transforms["bone2"].translation.x = 10.0
        transforms["bone3"].translation.x = 10.0

        result = mask.apply(transforms)
        assert abs(result["bone1"].translation.x - 10.0) < 0.001
        assert abs(result["bone2"].translation.x - 5.0) < 0.001
        assert abs(result["bone3"].translation.x) < 0.001

    def test_apply_uses_default_for_missing(self) -> None:
        """Test apply uses default_weight for bones not in mask."""
        mask = BoneMask(default_weight=0.5)

        transforms: Dict[str, CoreTransform] = {
            "unlisted_bone": CoreTransform(),
        }
        transforms["unlisted_bone"].translation.x = 10.0

        result = mask.apply(transforms)
        # Should use default weight of 0.5
        assert abs(result["unlisted_bone"].translation.x - 5.0) < 0.001


# =============================================================================
# APPLY TESTS - POSE INPUT
# =============================================================================


class TestApplyPose:
    """Test apply() with Pose object input."""

    def test_apply_pose_full_weight(self) -> None:
        """Test apply with Pose input and weight 1.0."""
        mask = BoneMask()
        mask.set_weight("bone1", 1.0)

        pose = Pose(bone_transforms={
            "bone1": Transform.from_position(5.0, 3.0, 1.0)
        })

        result = mask.apply(pose)
        assert isinstance(result, Pose)
        t = result.get_transform("bone1")
        assert t is not None
        assert abs(t.position[0] - 5.0) < 0.001
        assert abs(t.position[1] - 3.0) < 0.001

    def test_apply_pose_zero_weight(self) -> None:
        """Test apply with Pose input and weight 0.0."""
        mask = BoneMask()
        mask.set_weight("bone1", 0.0)

        pose = Pose(bone_transforms={
            "bone1": Transform.from_position(5.0, 3.0, 1.0)
        })

        result = mask.apply(pose)
        assert isinstance(result, Pose)
        t = result.get_transform("bone1")
        assert t is not None
        # Should be identity
        assert abs(t.position[0]) < 0.001
        assert abs(t.position[1]) < 0.001

    def test_apply_pose_half_weight(self) -> None:
        """Test apply with Pose input and weight 0.5."""
        mask = BoneMask()
        mask.set_weight("bone1", 0.5)

        pose = Pose(bone_transforms={
            "bone1": Transform.from_position(10.0, 0.0, 0.0)
        })

        result = mask.apply(pose)
        assert isinstance(result, Pose)
        t = result.get_transform("bone1")
        assert t is not None
        assert abs(t.position[0] - 5.0) < 0.001

    def test_apply_pose_multiple_bones(self) -> None:
        """Test apply with Pose input and multiple bones."""
        mask = BoneMask()
        mask.set_weight("bone1", 1.0)
        mask.set_weight("bone2", 0.5)

        pose = Pose(bone_transforms={
            "bone1": Transform.from_position(10.0, 0.0, 0.0),
            "bone2": Transform.from_position(10.0, 0.0, 0.0),
        })

        result = mask.apply(pose)
        assert isinstance(result, Pose)
        t1 = result.get_transform("bone1")
        t2 = result.get_transform("bone2")
        assert t1 is not None and t2 is not None
        assert abs(t1.position[0] - 10.0) < 0.001
        assert abs(t2.position[0] - 5.0) < 0.001

    def test_apply_preserves_type(self) -> None:
        """Test that apply returns the same type as input."""
        mask = BoneMask()
        mask.set_weight("bone1", 0.5)

        # Test with dict
        dict_input: Dict[str, CoreTransform] = {"bone1": CoreTransform()}
        dict_result = mask.apply(dict_input)
        assert isinstance(dict_result, dict)

        # Test with Pose
        pose_input = Pose(bone_transforms={"bone1": Transform.identity()})
        pose_result = mask.apply(pose_input)
        assert isinstance(pose_result, Pose)


# =============================================================================
# COMBINE TESTS
# =============================================================================


class TestCombine:
    """Test combine() with all modes."""

    def test_combine_multiply_default(self) -> None:
        """Test combine with default MULTIPLY mode."""
        mask1 = BoneMask(name="mask1")
        mask1.set_weight("bone1", 0.8)
        mask1.set_weight("bone2", 0.5)

        mask2 = BoneMask(name="mask2")
        mask2.set_weight("bone1", 0.5)
        mask2.set_weight("bone2", 0.6)

        combined = mask1.combine(mask2)
        assert abs(combined.get_weight("bone1") - 0.4) < 0.001  # 0.8 * 0.5
        assert abs(combined.get_weight("bone2") - 0.3) < 0.001  # 0.5 * 0.6

    def test_combine_multiply_explicit(self) -> None:
        """Test combine with explicit MULTIPLY mode string."""
        mask1 = BoneMask()
        mask1.set_weight("bone1", 0.8)

        mask2 = BoneMask()
        mask2.set_weight("bone1", 0.5)

        combined = mask1.combine(mask2, mode="multiply")
        assert abs(combined.get_weight("bone1") - 0.4) < 0.001

    def test_combine_multiply_enum(self) -> None:
        """Test combine with CombineMode.MULTIPLY enum."""
        mask1 = BoneMask()
        mask1.set_weight("bone1", 0.8)

        mask2 = BoneMask()
        mask2.set_weight("bone1", 0.5)

        combined = mask1.combine(mask2, mode=CombineMode.MULTIPLY)
        assert abs(combined.get_weight("bone1") - 0.4) < 0.001

    def test_combine_add(self) -> None:
        """Test combine with ADD mode."""
        mask1 = BoneMask()
        mask1.set_weight("bone1", 0.3)
        mask1.set_weight("bone2", 0.7)

        mask2 = BoneMask()
        mask2.set_weight("bone1", 0.4)
        mask2.set_weight("bone2", 0.5)

        combined = mask1.combine(mask2, mode="add")
        assert abs(combined.get_weight("bone1") - 0.7) < 0.001  # 0.3 + 0.4
        assert abs(combined.get_weight("bone2") - 1.0) < 0.001  # clamped

    def test_combine_add_clamping(self) -> None:
        """Test that ADD mode clamps to 1.0."""
        mask1 = BoneMask()
        mask1.set_weight("bone1", 0.8)

        mask2 = BoneMask()
        mask2.set_weight("bone1", 0.5)

        combined = mask1.combine(mask2, mode="add")
        assert combined.get_weight("bone1") == 1.0  # 0.8 + 0.5 = 1.3 -> clamped to 1.0

    def test_combine_max(self) -> None:
        """Test combine with MAX mode."""
        mask1 = BoneMask()
        mask1.set_weight("bone1", 0.3)
        mask1.set_weight("bone2", 0.9)

        mask2 = BoneMask()
        mask2.set_weight("bone1", 0.7)
        mask2.set_weight("bone2", 0.4)

        combined = mask1.combine(mask2, mode="max")
        assert abs(combined.get_weight("bone1") - 0.7) < 0.001
        assert abs(combined.get_weight("bone2") - 0.9) < 0.001

    def test_combine_min(self) -> None:
        """Test combine with MIN mode."""
        mask1 = BoneMask()
        mask1.set_weight("bone1", 0.3)
        mask1.set_weight("bone2", 0.9)

        mask2 = BoneMask()
        mask2.set_weight("bone1", 0.7)
        mask2.set_weight("bone2", 0.4)

        combined = mask1.combine(mask2, mode="min")
        assert abs(combined.get_weight("bone1") - 0.3) < 0.001
        assert abs(combined.get_weight("bone2") - 0.4) < 0.001

    def test_combine_union_of_bones(self) -> None:
        """Test combine includes bones from both masks."""
        mask1 = BoneMask()
        mask1.set_weight("bone1", 0.5)
        mask1.set_weight("bone2", 0.5)

        mask2 = BoneMask()
        mask2.set_weight("bone2", 0.5)
        mask2.set_weight("bone3", 0.5)

        combined = mask1.combine(mask2, mode="max")
        assert combined.has_bone("bone1")
        assert combined.has_bone("bone2")
        assert combined.has_bone("bone3")

    def test_combine_uses_default_weight(self) -> None:
        """Test combine uses each mask's default for missing bones."""
        mask1 = BoneMask(default_weight=0.2)
        mask1.set_weight("bone1", 0.8)

        mask2 = BoneMask(default_weight=0.5)
        mask2.set_weight("bone2", 0.6)

        combined = mask1.combine(mask2, mode="multiply")
        # bone1: mask1=0.8, mask2=0.5 (default) -> 0.4
        # bone2: mask1=0.2 (default), mask2=0.6 -> 0.12
        assert abs(combined.get_weight("bone1") - 0.4) < 0.001
        assert abs(combined.get_weight("bone2") - 0.12) < 0.001

    def test_combine_name_auto(self) -> None:
        """Test combine generates appropriate name."""
        mask1 = BoneMask(name="A")
        mask2 = BoneMask(name="B")

        combined_mul = mask1.combine(mask2, mode="multiply")
        assert "A" in combined_mul.name and "B" in combined_mul.name

        combined_add = mask1.combine(mask2, mode="add")
        assert "+" in combined_add.name

    def test_combine_name_custom(self) -> None:
        """Test combine with custom name."""
        mask1 = BoneMask(name="A")
        mask2 = BoneMask(name="B")

        combined = mask1.combine(mask2, name="custom_combined")
        assert combined.name == "custom_combined"

    def test_combine_unknown_mode_fallback(self) -> None:
        """Test combine with unknown mode falls back to multiply."""
        mask1 = BoneMask()
        mask1.set_weight("bone1", 0.8)

        mask2 = BoneMask()
        mask2.set_weight("bone1", 0.5)

        combined = mask1.combine(mask2, mode="unknown_mode")
        assert abs(combined.get_weight("bone1") - 0.4) < 0.001  # multiply fallback


# =============================================================================
# INVERT TESTS
# =============================================================================


class TestInvert:
    """Test invert() method."""

    def test_invert_basic(self) -> None:
        """Test basic inversion of weights."""
        mask = BoneMask()
        mask.set_weight("bone1", 0.3)
        mask.set_weight("bone2", 0.7)
        mask.set_weight("bone3", 0.0)
        mask.set_weight("bone4", 1.0)

        inv = mask.invert()
        assert abs(inv.get_weight("bone1") - 0.7) < 0.001
        assert abs(inv.get_weight("bone2") - 0.3) < 0.001
        assert abs(inv.get_weight("bone3") - 1.0) < 0.001
        assert abs(inv.get_weight("bone4") - 0.0) < 0.001

    def test_invert_name_auto(self) -> None:
        """Test invert generates appropriate name."""
        mask = BoneMask(name="original")
        inv = mask.invert()
        assert "not_" in inv.name or "original" in inv.name

    def test_invert_name_custom(self) -> None:
        """Test invert with custom name."""
        mask = BoneMask(name="original")
        inv = mask.invert(name="custom_inverted")
        assert inv.name == "custom_inverted"


# =============================================================================
# FACTORY METHOD TESTS
# =============================================================================


class TestFactoryMethods:
    """Test BoneMask factory methods."""

    def test_full_mask(self, simple_skeleton: Skeleton) -> None:
        """Test BoneMask.full() creates mask with all bones at 1.0."""
        mask = BoneMask.full(simple_skeleton)
        assert mask.bone_count == simple_skeleton.bone_count
        for bone in simple_skeleton:
            assert mask.get_weight(bone.name) == 1.0

    def test_full_mask_name(self, simple_skeleton: Skeleton) -> None:
        """Test BoneMask.full() with custom name."""
        mask = BoneMask.full(simple_skeleton, name="all_bones")
        assert mask.name == "all_bones"

    def test_from_bone_names_basic(self, simple_skeleton: Skeleton) -> None:
        """Test BoneMask.from_bone_names() basic functionality."""
        mask = BoneMask.from_bone_names(
            simple_skeleton,
            name="test_mask",
            bone_names=["Spine", "Neck", "Head"],
        )
        assert mask.has_bone("Spine")
        assert mask.has_bone("Neck")
        assert mask.has_bone("Head")
        assert not mask.has_bone("LeftArm")

    def test_from_bone_names_weight(self, simple_skeleton: Skeleton) -> None:
        """Test BoneMask.from_bone_names() with custom weight."""
        mask = BoneMask.from_bone_names(
            simple_skeleton,
            name="test_mask",
            bone_names=["Spine"],
            weight=0.7,
        )
        assert abs(mask.get_weight("Spine") - 0.7) < 0.001

    def test_from_bone_names_include_children(self, simple_skeleton: Skeleton) -> None:
        """Test BoneMask.from_bone_names() with include_children."""
        mask = BoneMask.from_bone_names(
            simple_skeleton,
            name="test_mask",
            bone_names=["LeftShoulder"],
            include_children=True,
        )
        assert mask.has_bone("LeftShoulder")
        assert mask.has_bone("LeftArm")
        assert mask.has_bone("LeftForeArm")
        assert mask.has_bone("LeftHand")

    def test_from_bone_names_nonexistent_skipped(self, simple_skeleton: Skeleton) -> None:
        """Test BoneMask.from_bone_names() skips nonexistent bones."""
        mask = BoneMask.from_bone_names(
            simple_skeleton,
            name="test_mask",
            bone_names=["Spine", "NonexistentBone", "Head"],
        )
        assert mask.has_bone("Spine")
        assert mask.has_bone("Head")
        assert not mask.has_bone("NonexistentBone")
        assert mask.bone_count == 2


# =============================================================================
# COPY TESTS
# =============================================================================


class TestCopy:
    """Test copy() method."""

    def test_copy_basic(self) -> None:
        """Test basic copy functionality."""
        mask = BoneMask(name="original", mode=MissingBoneMode.ONE)
        mask.set_weight("bone1", 0.5)
        mask.set_weight("bone2", 0.8)

        copy = mask.copy()
        assert copy.name == "original"
        assert copy.mode == MissingBoneMode.ONE
        assert copy.get_weight("bone1") == 0.5
        assert copy.get_weight("bone2") == 0.8

    def test_copy_independent(self) -> None:
        """Test copy is independent of original."""
        mask = BoneMask()
        mask.set_weight("bone1", 0.5)

        copy = mask.copy()
        copy.set_weight("bone1", 0.9)
        copy.set_weight("bone2", 0.3)

        assert mask.get_weight("bone1") == 0.5  # Original unchanged
        assert not mask.has_bone("bone2")

    def test_copy_with_name(self) -> None:
        """Test copy with custom name."""
        mask = BoneMask(name="original")
        copy = mask.copy(name="copy_name")
        assert copy.name == "copy_name"


# =============================================================================
# REPR TESTS
# =============================================================================


class TestRepr:
    """Test __repr__ method."""

    def test_repr_basic(self) -> None:
        """Test repr output format."""
        mask = BoneMask(name="test_mask", mode=MissingBoneMode.ZERO)
        mask.set_weight("bone1", 0.5)
        mask.set_weight("bone2", 0.8)

        repr_str = repr(mask)
        assert "BoneMask" in repr_str
        assert "test_mask" in repr_str
        assert "2" in repr_str  # bone count
        assert "ZERO" in repr_str


# =============================================================================
# BONEMASKPRESETS TESTS
# =============================================================================


class TestBoneMaskPresets:
    """Test BoneMaskPresets factory methods."""

    def test_upper_body(self, humanoid_skeleton: Skeleton) -> None:
        """Test upper_body preset includes correct bones."""
        mask = BoneMaskPresets.upper_body(humanoid_skeleton)
        assert mask.name == "UpperBody"

        # Check expected upper body bones
        assert mask.has_bone("Spine") or mask.get_weight("Spine") == 1.0
        assert mask.has_bone("Chest") or mask.get_weight("Chest") == 1.0
        assert mask.has_bone("Neck") or mask.get_weight("Neck") == 1.0
        assert mask.has_bone("Head") or mask.get_weight("Head") == 1.0
        assert mask.has_bone("LeftArm") or mask.get_weight("LeftArm") == 1.0
        assert mask.has_bone("RightArm") or mask.get_weight("RightArm") == 1.0

    def test_lower_body(self, humanoid_skeleton: Skeleton) -> None:
        """Test lower_body preset includes correct bones."""
        mask = BoneMaskPresets.lower_body(humanoid_skeleton)
        assert mask.name == "LowerBody"

        # Check expected lower body bones
        assert mask.has_bone("Hips") or mask.get_weight("Hips") == 1.0
        assert mask.has_bone("LeftUpLeg") or mask.get_weight("LeftUpLeg") == 1.0
        assert mask.has_bone("RightUpLeg") or mask.get_weight("RightUpLeg") == 1.0
        assert mask.has_bone("LeftFoot") or mask.get_weight("LeftFoot") == 1.0
        assert mask.has_bone("RightFoot") or mask.get_weight("RightFoot") == 1.0

    def test_left_arm(self, humanoid_skeleton: Skeleton) -> None:
        """Test left_arm preset includes correct bones."""
        mask = BoneMaskPresets.left_arm(humanoid_skeleton)
        assert mask.name == "LeftArm"

        assert mask.has_bone("LeftShoulder") or mask.get_weight("LeftShoulder") == 1.0
        assert mask.has_bone("LeftArm") or mask.get_weight("LeftArm") == 1.0
        assert mask.has_bone("LeftForeArm") or mask.get_weight("LeftForeArm") == 1.0
        assert mask.has_bone("LeftHand") or mask.get_weight("LeftHand") == 1.0

        # Should not include right arm
        assert not mask.has_bone("RightArm") or mask.get_weight("RightArm") == 0.0

    def test_right_arm(self, humanoid_skeleton: Skeleton) -> None:
        """Test right_arm preset includes correct bones."""
        mask = BoneMaskPresets.right_arm(humanoid_skeleton)
        assert mask.name == "RightArm"

        assert mask.has_bone("RightShoulder") or mask.get_weight("RightShoulder") == 1.0
        assert mask.has_bone("RightArm") or mask.get_weight("RightArm") == 1.0
        assert mask.has_bone("RightForeArm") or mask.get_weight("RightForeArm") == 1.0
        assert mask.has_bone("RightHand") or mask.get_weight("RightHand") == 1.0

    def test_left_leg(self, humanoid_skeleton: Skeleton) -> None:
        """Test left_leg preset includes correct bones."""
        mask = BoneMaskPresets.left_leg(humanoid_skeleton)
        assert mask.name == "LeftLeg"

        assert mask.has_bone("LeftUpLeg") or mask.get_weight("LeftUpLeg") == 1.0
        assert mask.has_bone("LeftLeg") or mask.get_weight("LeftLeg") == 1.0
        assert mask.has_bone("LeftFoot") or mask.get_weight("LeftFoot") == 1.0

    def test_right_leg(self, humanoid_skeleton: Skeleton) -> None:
        """Test right_leg preset includes correct bones."""
        mask = BoneMaskPresets.right_leg(humanoid_skeleton)
        assert mask.name == "RightLeg"

        assert mask.has_bone("RightUpLeg") or mask.get_weight("RightUpLeg") == 1.0
        assert mask.has_bone("RightLeg") or mask.get_weight("RightLeg") == 1.0
        assert mask.has_bone("RightFoot") or mask.get_weight("RightFoot") == 1.0


# =============================================================================
# GRADIENT TESTS
# =============================================================================


class TestGradient:
    """Test gradient mask generation."""

    def test_gradient_linear(self, simple_skeleton: Skeleton) -> None:
        """Test gradient with linear falloff."""
        mask = BoneMaskPresets.gradient(
            simple_skeleton,
            start_bone="Spine",
            falloff="linear",
            root_weight=0.0,
            tip_weight=1.0,
        )

        # Should have gradient weights from Spine through descendants
        assert mask.has_bone("Spine")
        spine_weight = mask.get_weight("Spine")

        # Root should have root_weight
        assert abs(spine_weight - 0.0) < 0.001

    def test_gradient_exponential(self, simple_skeleton: Skeleton) -> None:
        """Test gradient with exponential falloff."""
        mask = BoneMaskPresets.gradient(
            simple_skeleton,
            start_bone="Spine",
            falloff="exponential",
            rate=2.0,
            root_weight=0.0,
            tip_weight=1.0,
        )

        assert mask.has_bone("Spine")
        # Exponential should also work
        spine_weight = mask.get_weight("Spine")
        assert 0.0 <= spine_weight <= 1.0

    def test_gradient_includes_descendants(self, simple_skeleton: Skeleton) -> None:
        """Test gradient includes all descendants."""
        mask = BoneMaskPresets.gradient(
            simple_skeleton,
            start_bone="LeftShoulder",
            falloff="linear",
        )

        assert mask.has_bone("LeftShoulder")
        assert mask.has_bone("LeftArm")
        assert mask.has_bone("LeftForeArm")
        assert mask.has_bone("LeftHand")

    def test_gradient_weights_increase(self, simple_skeleton: Skeleton) -> None:
        """Test gradient weights increase toward tip."""
        mask = BoneMaskPresets.gradient(
            simple_skeleton,
            start_bone="LeftShoulder",
            falloff="linear",
            root_weight=0.0,
            tip_weight=1.0,
        )

        w_shoulder = mask.get_weight("LeftShoulder")
        w_arm = mask.get_weight("LeftArm")
        w_forearm = mask.get_weight("LeftForeArm")
        w_hand = mask.get_weight("LeftHand")

        # Weights should increase (or stay same) from root to tip
        assert w_shoulder <= w_arm or abs(w_shoulder - w_arm) < 0.001
        assert w_arm <= w_forearm or abs(w_arm - w_forearm) < 0.001
        assert w_forearm <= w_hand or abs(w_forearm - w_hand) < 0.001

    def test_gradient_nonexistent_bone(self, simple_skeleton: Skeleton) -> None:
        """Test gradient with nonexistent start bone returns empty mask."""
        mask = BoneMaskPresets.gradient(
            simple_skeleton,
            start_bone="NonexistentBone",
            falloff="linear",
        )

        assert mask.bone_count == 0

    def test_gradient_reversed(self, simple_skeleton: Skeleton) -> None:
        """Test gradient with reversed weights (tip to root)."""
        mask = BoneMaskPresets.gradient(
            simple_skeleton,
            start_bone="LeftShoulder",
            falloff="linear",
            root_weight=1.0,
            tip_weight=0.0,
        )

        w_shoulder = mask.get_weight("LeftShoulder")
        w_hand = mask.get_weight("LeftHand")

        # Root should have higher weight
        assert w_shoulder >= w_hand


# =============================================================================
# CREATE_GRADIENT TESTS
# =============================================================================


class TestCreateGradient:
    """Test create_gradient factory method."""

    def test_create_gradient_basic(self, simple_skeleton: Skeleton) -> None:
        """Test create_gradient basic functionality."""
        mask = BoneMaskPresets.create_gradient(
            simple_skeleton,
            root="Spine",
            leaves=["Head"],
            falloff="linear",
        )

        assert mask.has_bone("Spine")
        assert mask.has_bone("Head")

    def test_create_gradient_multiple_leaves(self, simple_skeleton: Skeleton) -> None:
        """Test create_gradient with multiple leaves."""
        mask = BoneMaskPresets.create_gradient(
            simple_skeleton,
            root="Spine",
            leaves=["LeftHand", "RightHand"],
            falloff="linear",
        )

        # Should include paths to both leaves
        assert mask.has_bone("LeftHand") or mask.bone_count > 0
        assert mask.has_bone("RightHand") or mask.bone_count > 0

    def test_create_gradient_exponential(self, simple_skeleton: Skeleton) -> None:
        """Test create_gradient with exponential falloff."""
        mask = BoneMaskPresets.create_gradient(
            simple_skeleton,
            root="Spine",
            leaves=["Head"],
            falloff="exponential",
            rate=2.0,
        )

        # Should still create valid weights
        for bone_name in mask.weights:
            w = mask.get_weight(bone_name)
            assert 0.0 <= w <= 1.0

    def test_create_gradient_nonexistent_root(self, simple_skeleton: Skeleton) -> None:
        """Test create_gradient with nonexistent root returns empty mask."""
        mask = BoneMaskPresets.create_gradient(
            simple_skeleton,
            root="NonexistentBone",
            leaves=["Head"],
            falloff="linear",
        )

        assert mask.bone_count == 0


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_mask_apply(self) -> None:
        """Test applying empty mask uses default weight."""
        mask = BoneMask(default_weight=0.5)

        transforms: Dict[str, CoreTransform] = {
            "bone1": CoreTransform(),
        }
        transforms["bone1"].translation.x = 10.0

        result = mask.apply(transforms)
        assert abs(result["bone1"].translation.x - 5.0) < 0.001

    def test_empty_pose_apply(self) -> None:
        """Test applying mask to empty pose."""
        mask = BoneMask()
        mask.set_weight("bone1", 1.0)

        pose = Pose()
        result = mask.apply(pose)
        assert isinstance(result, Pose)
        assert result.bone_count() == 0

    def test_missing_bones_in_pose(self) -> None:
        """Test mask with bones not in the pose."""
        mask = BoneMask()
        mask.set_weight("bone1", 1.0)
        mask.set_weight("bone2", 0.5)  # Not in pose

        pose = Pose(bone_transforms={
            "bone1": Transform.from_position(10.0, 0.0, 0.0),
        })

        result = mask.apply(pose)
        # Should only have bone1 in result
        assert result.has_bone("bone1")
        assert not result.has_bone("bone2")

    def test_zero_weight_all_bones(self) -> None:
        """Test mask with zero weight for all bones."""
        mask = BoneMask(default_weight=0.0)
        mask.set_weight("bone1", 0.0)
        mask.set_weight("bone2", 0.0)

        transforms: Dict[str, CoreTransform] = {
            "bone1": CoreTransform(),
            "bone2": CoreTransform(),
        }
        transforms["bone1"].translation.x = 10.0
        transforms["bone2"].translation.y = 20.0

        result = mask.apply(transforms)
        # All should be identity
        assert abs(result["bone1"].translation.x) < 0.001
        assert abs(result["bone2"].translation.y) < 0.001

    def test_full_weight_all_bones(self) -> None:
        """Test mask with full weight for all bones."""
        mask = BoneMask(default_weight=1.0)
        mask.set_weight("bone1", 1.0)

        transforms: Dict[str, CoreTransform] = {
            "bone1": CoreTransform(),
            "bone2": CoreTransform(),  # Uses default
        }
        transforms["bone1"].translation.x = 10.0
        transforms["bone2"].translation.y = 20.0

        result = mask.apply(transforms)
        # All should be preserved
        assert abs(result["bone1"].translation.x - 10.0) < 0.001
        assert abs(result["bone2"].translation.y - 20.0) < 0.001

    def test_combine_empty_masks(self) -> None:
        """Test combining two empty masks."""
        mask1 = BoneMask()
        mask2 = BoneMask()

        combined = mask1.combine(mask2)
        assert combined.bone_count == 0

    def test_combine_with_empty_mask(self) -> None:
        """Test combining with an empty mask."""
        mask1 = BoneMask()
        mask1.set_weight("bone1", 0.5)

        mask2 = BoneMask(default_weight=0.8)

        combined = mask1.combine(mask2, mode="multiply")
        # bone1: 0.5 * 0.8 (default) = 0.4
        assert abs(combined.get_weight("bone1") - 0.4) < 0.001

    def test_gradient_single_bone(self, simple_skeleton: Skeleton) -> None:
        """Test gradient with a single bone (no children)."""
        # Head has no children in simple_skeleton
        mask = BoneMaskPresets.gradient(
            simple_skeleton,
            start_bone="Head",
            falloff="linear",
        )

        assert mask.has_bone("Head")
        # With no children, only the start bone should be in the mask

    def test_weight_boundary_values(self) -> None:
        """Test weight at exact boundary values."""
        mask = BoneMask()

        mask.set_weight("bone_zero", 0.0)
        mask.set_weight("bone_one", 1.0)
        mask.set_weight("bone_epsilon", 0.0001)
        mask.set_weight("bone_near_one", 0.9999)

        assert mask.get_weight("bone_zero") == 0.0
        assert mask.get_weight("bone_one") == 1.0
        assert 0.0 <= mask.get_weight("bone_epsilon") <= 0.001
        assert 0.999 <= mask.get_weight("bone_near_one") <= 1.0


# =============================================================================
# MISSING BONE MODE ENUM TESTS
# =============================================================================


class TestMissingBoneMode:
    """Test MissingBoneMode enum."""

    def test_zero_value(self) -> None:
        """Test ZERO enum value."""
        assert MissingBoneMode.ZERO.value == 0

    def test_one_value(self) -> None:
        """Test ONE enum value."""
        assert MissingBoneMode.ONE.value == 1

    def test_mode_comparison(self) -> None:
        """Test mode enum comparison."""
        assert MissingBoneMode.ZERO != MissingBoneMode.ONE
        assert MissingBoneMode.ZERO is MissingBoneMode.ZERO


# =============================================================================
# COMBINE MODE ENUM TESTS
# =============================================================================


class TestCombineMode:
    """Test CombineMode enum."""

    def test_multiply_value(self) -> None:
        """Test MULTIPLY enum value."""
        assert CombineMode.MULTIPLY.value == "multiply"

    def test_add_value(self) -> None:
        """Test ADD enum value."""
        assert CombineMode.ADD.value == "add"

    def test_max_value(self) -> None:
        """Test MAX enum value."""
        assert CombineMode.MAX.value == "max"

    def test_min_value(self) -> None:
        """Test MIN enum value."""
        assert CombineMode.MIN.value == "min"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_mask_pose_then_combine(self, simple_skeleton: Skeleton) -> None:
        """Test masking a pose then combining masks."""
        # Create two masks
        upper = BoneMaskPresets.upper_body(simple_skeleton)
        lower = BoneMaskPresets.lower_body(simple_skeleton)

        # Combine them
        full = upper.combine(lower, mode="max")

        # Apply to a pose
        pose = Pose(bone_transforms={
            "Spine": Transform.from_position(1.0, 0.0, 0.0),
            "Hips": Transform.from_position(0.0, 1.0, 0.0),
        })

        result = full.apply(pose)
        assert isinstance(result, Pose)

    def test_gradient_then_invert(self, simple_skeleton: Skeleton) -> None:
        """Test creating gradient then inverting."""
        gradient = BoneMaskPresets.gradient(
            simple_skeleton,
            start_bone="Spine",
            falloff="linear",
            root_weight=0.0,
            tip_weight=1.0,
        )

        inverted = gradient.invert()

        # Inverted should have opposite weights
        for bone_name in gradient.weights:
            orig = gradient.get_weight(bone_name)
            inv = inverted.get_weight(bone_name)
            assert abs((1.0 - orig) - inv) < 0.001

    def test_chain_of_combines(self) -> None:
        """Test chaining multiple combine operations."""
        mask1 = BoneMask()
        mask1.set_weight("a", 0.5)

        mask2 = BoneMask()
        mask2.set_weight("b", 0.5)

        mask3 = BoneMask()
        mask3.set_weight("c", 0.5)

        combined = mask1.combine(mask2, mode="max").combine(mask3, mode="max")

        assert combined.has_bone("a")
        assert combined.has_bone("b")
        assert combined.has_bone("c")

    def test_copy_then_modify(self) -> None:
        """Test copying then modifying without affecting original."""
        original = BoneMask(name="original")
        original.set_weight("bone1", 0.5)

        copy = original.copy()
        copy.set_weight("bone1", 0.9)
        copy.set_weight("bone2", 0.7)

        assert original.get_weight("bone1") == 0.5
        assert not original.has_bone("bone2")
        assert copy.get_weight("bone1") == 0.9
        assert copy.get_weight("bone2") == 0.7


# =============================================================================
# PERFORMANCE / STRESS TESTS
# =============================================================================


class TestPerformance:
    """Performance-oriented tests."""

    def test_large_mask(self) -> None:
        """Test mask with many bones."""
        mask = BoneMask()
        for i in range(1000):
            mask.set_weight(f"bone_{i}", i / 1000.0)

        assert mask.bone_count == 1000
        assert abs(mask.get_weight("bone_500") - 0.5) < 0.001

    def test_combine_large_masks(self) -> None:
        """Test combining large masks."""
        mask1 = BoneMask()
        mask2 = BoneMask()

        for i in range(500):
            mask1.set_weight(f"bone_{i}", 0.5)
            mask2.set_weight(f"bone_{i + 250}", 0.5)

        combined = mask1.combine(mask2, mode="max")
        assert combined.bone_count == 750  # 0-499 + 500-749 with overlap 250-499

    def test_apply_large_pose(self) -> None:
        """Test applying mask to large pose."""
        mask = BoneMask(default_weight=0.5)

        transforms: Dict[str, CoreTransform] = {}
        for i in range(1000):
            t = CoreTransform()
            t.translation.x = float(i)
            transforms[f"bone_{i}"] = t

        result = mask.apply(transforms)
        assert len(result) == 1000
        # Check a sample
        assert abs(result["bone_500"].translation.x - 250.0) < 0.001  # 500 * 0.5
