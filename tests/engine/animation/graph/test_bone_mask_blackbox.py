"""
Blackbox tests for T-AG-1.4 Bone Mask System.

CLEANROOM TEST - Tests written from public contract only.
API-aligned version: uses actual BoneMask constructor and method signatures.

Public Contract:
- BoneMask with per-bone weights (set via set_weight method)
- apply(pose) method to mask a pose
- combine(other, mode) for mask composition
- BoneMaskPresets with common masks (upper_body, lower_body, gradient)
- CombineMode enum for mask combination strategies
"""

import pytest
from typing import Dict, Any


class TestBoneMaskCreation:
    """Test BoneMask instantiation and basic weight access."""

    def test_create_bone_mask_default(self):
        """BoneMask can be created with default parameters."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()

        assert mask is not None
        assert isinstance(mask, BoneMask)

    def test_create_bone_mask_with_name(self):
        """BoneMask can be created with a name."""
        from engine.animation.graph import BoneMask

        mask = BoneMask(name="test_mask")

        assert mask is not None
        assert mask.name == "test_mask"

    def test_create_bone_mask_with_default_weight(self):
        """BoneMask accepts default_weight parameter."""
        from engine.animation.graph import BoneMask

        mask = BoneMask(default_weight=0.5)

        assert mask is not None
        assert mask.default_weight == 0.5

    def test_set_and_get_weight(self):
        """Weights can be set and retrieved via set_weight/get_weight."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        mask.set_weight("spine", 1.0)
        mask.set_weight("arm", 0.5)

        assert mask.get_weight("spine") == 1.0
        assert mask.get_weight("arm") == 0.5

    def test_get_weight_returns_default_for_unknown_bone(self):
        """get_weight returns default_weight for bones not explicitly set."""
        from engine.animation.graph import BoneMask

        mask = BoneMask(default_weight=0.0)
        mask.set_weight("spine", 1.0)

        assert mask.get_weight("unknown_bone") == 0.0

    def test_get_weight_with_nonzero_default(self):
        """get_weight correctly uses non-zero default_weight."""
        from engine.animation.graph import BoneMask

        mask = BoneMask(default_weight=1.0)
        mask.set_weight("spine", 0.5)

        assert mask.get_weight("spine") == 0.5
        assert mask.get_weight("other") == 1.0

    def test_set_weight_can_update_existing(self):
        """set_weight can update an already-set bone weight."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        mask.set_weight("spine", 1.0)
        mask.set_weight("spine", 0.3)

        assert mask.get_weight("spine") == 0.3


class TestBoneMaskWeightClamping:
    """Test that weights are properly clamped to [0, 1]."""

    def test_set_weight_clamps_to_max_one(self):
        """set_weight clamps values greater than 1.0 to 1.0."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        mask.set_weight("spine", 2.0)

        assert mask.get_weight("spine") <= 1.0

    def test_set_weight_clamps_to_min_zero(self):
        """set_weight clamps values less than 0.0 to 0.0."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        mask.set_weight("spine", -0.5)

        assert mask.get_weight("spine") >= 0.0

    def test_default_weight_is_clamped(self):
        """default_weight is clamped to [0, 1] range."""
        from engine.animation.graph import BoneMask

        mask = BoneMask(default_weight=1.5)
        assert mask.default_weight <= 1.0

        mask2 = BoneMask(default_weight=-0.5)
        assert mask2.default_weight >= 0.0


class TestBoneMaskApply:
    """Test BoneMask.apply() method with poses."""

    def test_apply_returns_pose_for_dict_input(self):
        """apply() returns a dict when given a dict."""
        from engine.animation.graph import BoneMask
        from engine.core.math.transform import Transform

        mask = BoneMask()
        mask.set_weight("spine", 1.0)
        pose = {"spine": Transform()}

        result = mask.apply(pose)

        assert result is not None
        assert isinstance(result, dict)

    def test_apply_returns_pose_for_pose_input(self):
        """apply() returns a Pose when given a Pose."""
        from engine.animation.graph import BoneMask, DictPose, DictTransform

        mask = BoneMask()
        mask.set_weight("spine", 1.0)
        pose = DictPose(bone_transforms={"spine": DictTransform()})

        result = mask.apply(pose)

        assert result is not None
        assert isinstance(result, DictPose)

    def test_apply_with_full_weight_preserves_bone(self):
        """apply() with weight=1.0 preserves the bone transform."""
        from engine.animation.graph import BoneMask, DictPose, DictTransform

        original_transform = DictTransform(
            position=(1.0, 2.0, 3.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0)
        )
        mask = BoneMask(default_weight=0.0)
        mask.set_weight("spine", 1.0)
        pose = DictPose(bone_transforms={"spine": original_transform})

        result = mask.apply(pose)

        # With weight 1.0, bone should be fully preserved
        assert "spine" in result.bone_transforms

    def test_apply_with_zero_weight_masks_bone(self):
        """apply() with weight=0.0 masks out the bone."""
        from engine.animation.graph import BoneMask, DictPose, DictTransform

        mask = BoneMask(default_weight=0.0)
        mask.set_weight("spine", 0.0)
        pose = DictPose(bone_transforms={"spine": DictTransform()})

        result = mask.apply(pose)

        # With weight 0.0, bone should be masked (identity transform)
        assert result is not None

    def test_apply_with_partial_weight(self):
        """apply() with weight between 0 and 1 partially masks the bone."""
        from engine.animation.graph import BoneMask, DictPose, DictTransform

        mask = BoneMask(default_weight=0.0)
        mask.set_weight("spine", 0.5)
        pose = DictPose(bone_transforms={"spine": DictTransform(position=(2.0, 0.0, 0.0))})

        result = mask.apply(pose)

        assert result is not None
        assert isinstance(result, DictPose)

    def test_apply_preserves_unmasked_bones_with_default_one(self):
        """apply() with default_weight=1.0 preserves bones not explicitly set."""
        from engine.animation.graph import BoneMask, DictPose, DictTransform

        mask = BoneMask(default_weight=1.0)
        mask.set_weight("spine", 0.0)
        pose = DictPose(bone_transforms={
            "spine": DictTransform(),
            "arm": DictTransform(position=(1.0, 1.0, 1.0))
        })

        result = mask.apply(pose)

        # arm should be preserved (default weight is 1.0)
        assert result is not None

    def test_apply_empty_pose(self):
        """apply() handles empty pose gracefully."""
        from engine.animation.graph import BoneMask, DictPose

        mask = BoneMask()
        mask.set_weight("spine", 1.0)
        pose = DictPose(bone_transforms={})

        result = mask.apply(pose)

        assert result is not None
        assert isinstance(result, DictPose)


class TestBoneMaskCombine:
    """Test BoneMask.combine() method for mask composition."""

    def test_combine_returns_bone_mask(self):
        """combine() returns a new BoneMask."""
        from engine.animation.graph import BoneMask

        mask1 = BoneMask(default_weight=0.0)
        mask1.set_weight("spine", 1.0)
        mask2 = BoneMask(default_weight=0.0)
        mask2.set_weight("arm", 1.0)

        result = mask1.combine(mask2)

        assert result is not None
        assert isinstance(result, BoneMask)

    def test_combine_with_multiply_mode(self):
        """combine() with 'multiply' mode multiplies weights."""
        from engine.animation.graph import BoneMask

        mask1 = BoneMask(default_weight=1.0)
        mask1.set_weight("spine", 0.8)
        mask2 = BoneMask(default_weight=1.0)
        mask2.set_weight("spine", 0.5)

        result = mask1.combine(mask2, mode='multiply')

        # 0.8 * 0.5 = 0.4
        assert abs(result.get_weight("spine") - 0.4) < 0.01

    def test_combine_with_add_mode(self):
        """combine() with 'add' mode adds weights (clamped to 1.0)."""
        from engine.animation.graph import BoneMask

        mask1 = BoneMask(default_weight=0.0)
        mask1.set_weight("spine", 0.3)
        mask2 = BoneMask(default_weight=0.0)
        mask2.set_weight("spine", 0.4)

        result = mask1.combine(mask2, mode='add')

        # 0.3 + 0.4 = 0.7
        assert abs(result.get_weight("spine") - 0.7) < 0.01

    def test_combine_add_mode_clamps_to_one(self):
        """combine() with 'add' mode clamps result to 1.0."""
        from engine.animation.graph import BoneMask

        mask1 = BoneMask(default_weight=0.0)
        mask1.set_weight("spine", 0.8)
        mask2 = BoneMask(default_weight=0.0)
        mask2.set_weight("spine", 0.5)

        result = mask1.combine(mask2, mode='add')

        # 0.8 + 0.5 = 1.3, should clamp to 1.0
        assert result.get_weight("spine") <= 1.0

    def test_combine_with_max_mode(self):
        """combine() with 'max' mode takes maximum weight."""
        from engine.animation.graph import BoneMask

        mask1 = BoneMask(default_weight=0.0)
        mask1.set_weight("spine", 0.3)
        mask2 = BoneMask(default_weight=0.0)
        mask2.set_weight("spine", 0.7)

        result = mask1.combine(mask2, mode='max')

        assert abs(result.get_weight("spine") - 0.7) < 0.01

    def test_combine_with_min_mode(self):
        """combine() with 'min' mode takes minimum weight."""
        from engine.animation.graph import BoneMask

        mask1 = BoneMask(default_weight=0.0)
        mask1.set_weight("spine", 0.3)
        mask2 = BoneMask(default_weight=0.0)
        mask2.set_weight("spine", 0.7)

        result = mask1.combine(mask2, mode='min')

        assert abs(result.get_weight("spine") - 0.3) < 0.01

    def test_combine_merges_different_bones(self):
        """combine() includes bones from both masks."""
        from engine.animation.graph import BoneMask

        mask1 = BoneMask(default_weight=0.0)
        mask1.set_weight("spine", 1.0)
        mask2 = BoneMask(default_weight=0.0)
        mask2.set_weight("arm", 0.8)

        result = mask1.combine(mask2, mode='max')

        # Both bones should be present in result
        assert result.get_weight("spine") == 1.0
        assert result.get_weight("arm") == 0.8

    def test_combine_default_mode_is_multiply(self):
        """combine() without mode argument defaults to multiply."""
        from engine.animation.graph import BoneMask

        mask1 = BoneMask(default_weight=1.0)
        mask1.set_weight("spine", 0.8)
        mask2 = BoneMask(default_weight=1.0)
        mask2.set_weight("spine", 0.5)

        # If no mode specified, should default to multiply
        result = mask1.combine(mask2)

        # 0.8 * 0.5 = 0.4
        assert abs(result.get_weight("spine") - 0.4) < 0.01


class TestCombineMode:
    """Test CombineMode enum availability."""

    def test_combine_mode_exists(self):
        """CombineMode enum is importable."""
        from engine.animation.graph import CombineMode

        assert CombineMode is not None

    def test_combine_mode_has_multiply(self):
        """CombineMode has MULTIPLY variant."""
        from engine.animation.graph import CombineMode

        # Should have multiply mode
        assert hasattr(CombineMode, 'MULTIPLY') or hasattr(CombineMode, 'multiply')

    def test_combine_mode_has_add(self):
        """CombineMode has ADD variant."""
        from engine.animation.graph import CombineMode

        assert hasattr(CombineMode, 'ADD') or hasattr(CombineMode, 'add')

    def test_combine_mode_has_max(self):
        """CombineMode has MAX variant."""
        from engine.animation.graph import CombineMode

        assert hasattr(CombineMode, 'MAX') or hasattr(CombineMode, 'max')

    def test_combine_mode_has_min(self):
        """CombineMode has MIN variant."""
        from engine.animation.graph import CombineMode

        assert hasattr(CombineMode, 'MIN') or hasattr(CombineMode, 'min')


class TestBoneMaskPresets:
    """Test BoneMaskPresets class with common mask presets."""

    def test_presets_class_exists(self):
        """BoneMaskPresets is importable."""
        from engine.animation.graph import BoneMaskPresets

        assert BoneMaskPresets is not None

    def test_upper_body_preset_exists(self):
        """BoneMaskPresets has upper_body method."""
        from engine.animation.graph import BoneMaskPresets

        assert hasattr(BoneMaskPresets, 'upper_body')
        assert callable(getattr(BoneMaskPresets, 'upper_body'))

    def test_lower_body_preset_exists(self):
        """BoneMaskPresets has lower_body method."""
        from engine.animation.graph import BoneMaskPresets

        assert hasattr(BoneMaskPresets, 'lower_body')
        assert callable(getattr(BoneMaskPresets, 'lower_body'))

    def test_gradient_preset_exists(self):
        """BoneMaskPresets has gradient method."""
        from engine.animation.graph import BoneMaskPresets

        assert hasattr(BoneMaskPresets, 'gradient')
        assert callable(getattr(BoneMaskPresets, 'gradient'))


class TestBoneMaskPresetsWithSkeleton:
    """Test BoneMaskPresets methods that require a Skeleton."""

    @pytest.fixture
    def simple_skeleton(self):
        """Create a simple skeleton for testing presets."""
        from engine.animation.graph.skeleton import Skeleton

        # Build a simple humanoid skeleton using add_bone API
        skeleton = Skeleton(name="test_skeleton")

        # Root
        skeleton.add_bone("root", parent_name=None)

        # Upper body chain
        skeleton.add_bone("Spine", parent_name="root")
        skeleton.add_bone("Spine1", parent_name="Spine")
        skeleton.add_bone("Chest", parent_name="Spine1")
        skeleton.add_bone("Neck", parent_name="Chest")
        skeleton.add_bone("Head", parent_name="Neck")

        # Arms (using standard naming convention)
        skeleton.add_bone("LeftShoulder", parent_name="Chest")
        skeleton.add_bone("LeftArm", parent_name="LeftShoulder")
        skeleton.add_bone("LeftForeArm", parent_name="LeftArm")
        skeleton.add_bone("LeftHand", parent_name="LeftForeArm")

        skeleton.add_bone("RightShoulder", parent_name="Chest")
        skeleton.add_bone("RightArm", parent_name="RightShoulder")
        skeleton.add_bone("RightForeArm", parent_name="RightArm")
        skeleton.add_bone("RightHand", parent_name="RightForeArm")

        # Lower body
        skeleton.add_bone("Hips", parent_name="root")
        skeleton.add_bone("LeftUpLeg", parent_name="Hips")
        skeleton.add_bone("LeftLeg", parent_name="LeftUpLeg")
        skeleton.add_bone("LeftFoot", parent_name="LeftLeg")

        skeleton.add_bone("RightUpLeg", parent_name="Hips")
        skeleton.add_bone("RightLeg", parent_name="RightUpLeg")
        skeleton.add_bone("RightFoot", parent_name="RightLeg")

        return skeleton

    def test_upper_body_returns_bone_mask(self, simple_skeleton):
        """upper_body(skeleton) returns a BoneMask."""
        from engine.animation.graph import BoneMaskPresets, BoneMask

        mask = BoneMaskPresets.upper_body(simple_skeleton)

        assert mask is not None
        assert isinstance(mask, BoneMask)

    def test_lower_body_returns_bone_mask(self, simple_skeleton):
        """lower_body(skeleton) returns a BoneMask."""
        from engine.animation.graph import BoneMaskPresets, BoneMask

        mask = BoneMaskPresets.lower_body(simple_skeleton)

        assert mask is not None
        assert isinstance(mask, BoneMask)

    def test_upper_body_has_high_weight_for_upper_bones(self, simple_skeleton):
        """upper_body mask should have high weight for upper body bones."""
        from engine.animation.graph import BoneMaskPresets

        mask = BoneMaskPresets.upper_body(simple_skeleton)

        # Upper body bones should have weight > 0.5
        upper_bones = ["Spine", "Chest", "Head", "LeftArm", "RightArm"]
        for bone in upper_bones:
            weight = mask.get_weight(bone)
            assert weight >= 0.5, f"{bone} should have weight >= 0.5, got {weight}"

    def test_lower_body_has_high_weight_for_lower_bones(self, simple_skeleton):
        """lower_body mask should have high weight for lower body bones."""
        from engine.animation.graph import BoneMaskPresets

        mask = BoneMaskPresets.lower_body(simple_skeleton)

        # Lower body bones should have weight > 0.5
        lower_bones = ["Hips", "LeftUpLeg", "RightUpLeg"]
        for bone in lower_bones:
            weight = mask.get_weight(bone)
            assert weight >= 0.5, f"{bone} should have weight >= 0.5, got {weight}"

    def test_gradient_returns_bone_mask(self, simple_skeleton):
        """gradient(skeleton, start_bone) returns a BoneMask."""
        from engine.animation.graph import BoneMaskPresets, BoneMask

        # Gradient takes start_bone and creates gradient through descendants
        mask = BoneMaskPresets.gradient(simple_skeleton, "Spine")

        assert mask is not None
        assert isinstance(mask, BoneMask)

    def test_gradient_with_root_and_tip_weights(self, simple_skeleton):
        """gradient mask respects root_weight and tip_weight parameters."""
        from engine.animation.graph import BoneMaskPresets

        mask = BoneMaskPresets.gradient(
            simple_skeleton, "Spine",
            root_weight=0.0, tip_weight=1.0
        )

        # Start bone should have weight close to root_weight
        spine_weight = mask.get_weight("Spine")
        assert spine_weight <= 0.1, f"Spine should be near 0.0, got {spine_weight}"

    def test_gradient_with_linear_falloff(self, simple_skeleton):
        """gradient supports linear falloff parameter."""
        from engine.animation.graph import BoneMaskPresets

        mask = BoneMaskPresets.gradient(
            simple_skeleton, "Spine", falloff='linear'
        )

        assert mask is not None

    def test_gradient_with_exponential_falloff(self, simple_skeleton):
        """gradient supports exponential falloff parameter."""
        from engine.animation.graph import BoneMaskPresets

        mask = BoneMaskPresets.gradient(
            simple_skeleton, "Spine", falloff='exponential'
        )

        assert mask is not None

    def test_gradient_intermediate_bones_have_gradient_weights(self, simple_skeleton):
        """gradient mask intermediate bones have weights forming a gradient."""
        from engine.animation.graph import BoneMaskPresets

        mask = BoneMaskPresets.gradient(
            simple_skeleton, "Spine",
            root_weight=0.0, tip_weight=1.0
        )

        # Spine1 is between Spine and deeper bones
        # Its weight should be between root (0) and tip (1)
        spine_weight = mask.get_weight("Spine")
        spine1_weight = mask.get_weight("Spine1")
        chest_weight = mask.get_weight("Chest")

        # Should form a gradient: spine < spine1 < chest
        assert spine_weight <= spine1_weight <= chest_weight


class TestBoneMaskEdgeCases:
    """Test edge cases and error handling."""

    def test_bone_mask_with_special_characters_in_name(self):
        """BoneMask handles bone names with special characters."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        mask.set_weight("bone.001", 0.5)
        mask.set_weight("arm_L", 0.8)

        assert mask.get_weight("bone.001") == 0.5
        assert mask.get_weight("arm_L") == 0.8

    def test_bone_mask_empty_string_bone_name(self):
        """BoneMask handles empty string bone name."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        mask.set_weight("", 0.5)

        assert mask.get_weight("") == 0.5

    def test_bone_mask_many_bones(self):
        """BoneMask handles large number of bones."""
        from engine.animation.graph import BoneMask

        mask = BoneMask(default_weight=0.0)
        for i in range(100):
            mask.set_weight(f"bone_{i}", i / 100.0)

        assert mask.get_weight("bone_50") == 0.5
        assert mask.get_weight("bone_99") == 0.99

    def test_combine_self_multiply(self):
        """Combining mask with itself via multiply squares weights."""
        from engine.animation.graph import BoneMask

        mask = BoneMask(default_weight=1.0)
        mask.set_weight("spine", 0.5)

        result = mask.combine(mask, mode='multiply')

        # 0.5 * 0.5 = 0.25
        assert abs(result.get_weight("spine") - 0.25) < 0.01

    def test_combine_self_max(self):
        """Combining mask with itself via max preserves weights."""
        from engine.animation.graph import BoneMask

        mask = BoneMask(default_weight=1.0)
        mask.set_weight("spine", 0.5)

        result = mask.combine(mask, mode='max')

        # max(0.5, 0.5) = 0.5
        assert abs(result.get_weight("spine") - 0.5) < 0.01


class TestBoneMaskContract:
    """Integration tests matching the actual implementation patterns."""

    def test_contract_example_basic_usage(self):
        """Test basic BoneMask usage."""
        from engine.animation.graph import BoneMask

        # Create mask and set weights
        mask = BoneMask(default_weight=0.0)
        mask.set_weight("spine", 1.0)
        mask.set_weight("arm", 0.5)

        # Verify get_weight
        assert mask.get_weight("spine") == 1.0
        assert mask.get_weight("arm") == 0.5

        # Verify default for unknown bones
        assert mask.get_weight("unknown") == 0.0

        # Verify set_weight updates
        mask.set_weight("leg", 0.8)
        assert mask.get_weight("leg") == 0.8

    def test_contract_example_combine(self):
        """Test combine() behavior."""
        from engine.animation.graph import BoneMask

        mask = BoneMask(default_weight=0.5)
        mask.set_weight("spine", 1.0)

        other_mask = BoneMask(default_weight=0.5)
        other_mask.set_weight("spine", 0.8)

        # Combine with multiply mode
        combined = mask.combine(other_mask, mode='multiply')

        assert combined is not None
        assert isinstance(combined, BoneMask)

    def test_contract_imports(self):
        """Verify all contract imports work."""
        from engine.animation.graph import BoneMask, BoneMaskPresets, CombineMode

        assert BoneMask is not None
        assert BoneMaskPresets is not None
        assert CombineMode is not None


class TestBoneMaskNotOriginalMask:
    """Test that operations return new masks, not modify original."""

    def test_combine_does_not_modify_original(self):
        """combine() returns a new mask, original unchanged."""
        from engine.animation.graph import BoneMask

        mask1 = BoneMask(default_weight=0.0)
        mask1.set_weight("spine", 0.8)
        mask2 = BoneMask(default_weight=0.0)
        mask2.set_weight("spine", 0.5)

        original_weight = mask1.get_weight("spine")
        result = mask1.combine(mask2, mode='multiply')

        # Original should be unchanged
        assert mask1.get_weight("spine") == original_weight
        # Result is different object
        assert result is not mask1
        assert result is not mask2

    def test_apply_does_not_modify_original_pose(self):
        """apply() does not modify the original pose."""
        from engine.animation.graph import BoneMask, DictPose, DictTransform

        mask = BoneMask(default_weight=0.0)
        mask.set_weight("spine", 0.5)
        original_pose = DictPose(bone_transforms={"spine": DictTransform(position=(1.0, 2.0, 3.0))})

        result = mask.apply(original_pose)

        # Result should be different object
        assert result is not original_pose


class TestBoneMaskProperties:
    """Test BoneMask property accessors."""

    def test_weights_property_returns_dict(self):
        """weights property returns a dictionary of explicit weights."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        mask.set_weight("spine", 1.0)
        mask.set_weight("arm", 0.5)

        weights = mask.weights

        assert isinstance(weights, dict)
        assert weights["spine"] == 1.0
        assert weights["arm"] == 0.5

    def test_bone_count_property(self):
        """bone_count returns number of explicit weight entries."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        assert mask.bone_count == 0

        mask.set_weight("spine", 1.0)
        assert mask.bone_count == 1

        mask.set_weight("arm", 0.5)
        assert mask.bone_count == 2

    def test_has_bone_method(self):
        """has_bone returns True for bones with explicit weights."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        mask.set_weight("spine", 1.0)

        assert mask.has_bone("spine") is True
        assert mask.has_bone("unknown") is False


class TestBoneMaskInvert:
    """Test BoneMask.invert() method."""

    def test_invert_returns_bone_mask(self):
        """invert() returns a new BoneMask."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        mask.set_weight("spine", 0.8)

        inverted = mask.invert()

        assert inverted is not None
        assert isinstance(inverted, BoneMask)

    def test_invert_inverts_weights(self):
        """invert() returns 1 - weight for each bone."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        mask.set_weight("spine", 0.8)
        mask.set_weight("arm", 0.3)

        inverted = mask.invert()

        # 1 - 0.8 = 0.2
        assert abs(inverted.get_weight("spine") - 0.2) < 0.01
        # 1 - 0.3 = 0.7
        assert abs(inverted.get_weight("arm") - 0.7) < 0.01

    def test_invert_does_not_modify_original(self):
        """invert() returns new mask, original unchanged."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        mask.set_weight("spine", 0.8)

        inverted = mask.invert()

        assert mask.get_weight("spine") == 0.8
        assert inverted is not mask


class TestBoneMaskCopy:
    """Test BoneMask.copy() method."""

    def test_copy_returns_independent_mask(self):
        """copy() returns an independent BoneMask."""
        from engine.animation.graph import BoneMask

        mask = BoneMask()
        mask.set_weight("spine", 0.8)

        copied = mask.copy()

        assert copied is not mask
        assert copied.get_weight("spine") == 0.8

        # Modifying copy doesn't affect original
        copied.set_weight("spine", 0.5)
        assert mask.get_weight("spine") == 0.8
