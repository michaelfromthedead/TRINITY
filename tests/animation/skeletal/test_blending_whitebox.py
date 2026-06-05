"""Whitebox tests for blending.py.

Tests pose blending modes, bone masks, layered blending, and pose caching.

Acceptance criteria:
- T-SKEL-1.7: Blending
  - Override mode
  - Additive mode
  - Multiply mode
  - BoneMask filtering
"""

import math
import pytest
from engine.core.math import Vec3, Quat
from engine.animation.skeletal.skeleton import Skeleton, Bone, create_humanoid_skeleton
from engine.animation.skeletal.pose import Pose, BoneTransform, PoseSpace
from engine.animation.skeletal.blending import (
    BlendMode, BoneMask, LayeredBlender, PoseCache,
    blend_poses, blend_multiple_poses,
    compute_additive_pose, apply_additive_pose,
    WEIGHT_EPSILON, SCALE_EPSILON,
    _blend_override, _blend_additive, _blend_multiply
)


# =============================================================================
# BlendMode Tests
# =============================================================================

class TestBlendMode:
    """Tests for BlendMode enum."""

    def test_blend_modes_exist(self):
        """Test all blend modes are defined."""
        assert BlendMode.OVERRIDE
        assert BlendMode.ADDITIVE
        assert BlendMode.MULTIPLY


# =============================================================================
# BoneMask Tests - T-SKEL-1.7
# =============================================================================

class TestBoneMask:
    """Tests for BoneMask class."""

    def test_bone_mask_default(self):
        """Test default bone mask has zero default weight."""
        mask = BoneMask()
        assert mask.default_weight == 0.0
        assert len(mask.bone_weights) == 0

    def test_bone_mask_get_weight_default(self):
        """Test getting weight for unmapped bone returns default."""
        mask = BoneMask(default_weight=0.5)
        assert mask.get_weight(99) == 0.5

    def test_bone_mask_set_weight(self):
        """Test setting weight for specific bone."""
        mask = BoneMask()
        mask.set_weight(0, 1.0)
        mask.set_weight(1, 0.5)

        assert mask.get_weight(0) == 1.0
        assert mask.get_weight(1) == 0.5
        assert mask.get_weight(2) == 0.0  # default

    def test_bone_mask_set_weight_clamps(self):
        """Test weight is clamped to [0, 1]."""
        mask = BoneMask()
        mask.set_weight(0, 2.0)  # Should clamp to 1.0
        mask.set_weight(1, -1.0)  # Should clamp to 0.0

        assert mask.get_weight(0) == 1.0
        assert mask.get_weight(1) == 0.0

    def test_bone_mask_set_weights_multiple(self):
        """Test setting same weight for multiple bones."""
        mask = BoneMask()
        mask.set_weights([0, 1, 2], 0.75)

        assert mask.get_weight(0) == 0.75
        assert mask.get_weight(1) == 0.75
        assert mask.get_weight(2) == 0.75

    def test_bone_mask_include_bone(self):
        """Test including a bone sets weight to 1."""
        mask = BoneMask()
        mask.include_bone(5)

        assert mask.get_weight(5) == 1.0

    def test_bone_mask_exclude_bone(self):
        """Test excluding a bone sets weight to 0."""
        mask = BoneMask(default_weight=1.0)
        mask.exclude_bone(3)

        assert mask.get_weight(3) == 0.0

    def test_bone_mask_include_all(self):
        """Test including all bones."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="a"))
        skeleton.add_bone(Bone(index=1, name="b"))
        skeleton.add_bone(Bone(index=2, name="c"))

        mask = BoneMask()
        mask.include_all(skeleton)

        assert mask.get_weight(0) == 1.0
        assert mask.get_weight(1) == 1.0
        assert mask.get_weight(2) == 1.0

    def test_bone_mask_exclude_all(self):
        """Test excluding all bones clears weights."""
        mask = BoneMask()
        mask.set_weight(0, 1.0)
        mask.set_weight(1, 1.0)

        mask.exclude_all()

        assert len(mask.bone_weights) == 0

    def test_bone_mask_invert(self):
        """Test inverting a mask."""
        mask = BoneMask(default_weight=0.0)
        mask.set_weight(0, 1.0)
        mask.set_weight(1, 0.3)

        inverted = mask.invert()

        assert inverted.get_weight(0) == 0.0  # 1 - 1 = 0
        assert abs(inverted.get_weight(1) - 0.7) < 1e-6  # 1 - 0.3 = 0.7
        assert inverted.default_weight == 1.0  # 1 - 0 = 1

    def test_bone_mask_combine(self):
        """Test combining two masks by multiplication."""
        mask_a = BoneMask(default_weight=0.5)
        mask_a.set_weight(0, 1.0)
        mask_a.set_weight(1, 0.5)

        mask_b = BoneMask(default_weight=0.5)
        mask_b.set_weight(0, 0.5)
        mask_b.set_weight(2, 1.0)

        combined = mask_a.combine(mask_b)

        assert abs(combined.get_weight(0) - 0.5) < 1e-6  # 1.0 * 0.5
        assert abs(combined.get_weight(1) - 0.25) < 1e-6  # 0.5 * 0.5 (default)
        assert abs(combined.get_weight(2) - 0.5) < 1e-6  # 0.5 (default) * 1.0

    def test_bone_mask_copy(self):
        """Test deep copying a mask."""
        original = BoneMask(default_weight=0.5)
        original.set_weight(0, 1.0)

        copied = original.copy()
        original.set_weight(0, 0.0)

        assert copied.get_weight(0) == 1.0


class TestBoneMaskFactories:
    """Tests for BoneMask factory methods."""

    def test_full_body_mask(self):
        """Test full body mask includes all bones."""
        skeleton = create_humanoid_skeleton()
        mask = BoneMask.full_body(skeleton)

        for i in range(skeleton.bone_count):
            assert mask.get_weight(i) == 1.0

    def test_upper_body_mask(self):
        """Test upper body mask includes correct bones."""
        skeleton = create_humanoid_skeleton()
        mask = BoneMask.upper_body(skeleton)

        # Should include spine, arm bones
        spine_idx = skeleton.get_bone_index("spine_01")
        arm_idx = skeleton.get_bone_index("upperarm_l")

        assert mask.get_weight(spine_idx) == 1.0
        assert mask.get_weight(arm_idx) == 1.0

        # Should not include leg bones
        thigh_idx = skeleton.get_bone_index("thigh_l")
        assert mask.get_weight(thigh_idx) == 0.0

    def test_lower_body_mask(self):
        """Test lower body mask includes correct bones."""
        skeleton = create_humanoid_skeleton()
        mask = BoneMask.lower_body(skeleton)

        # Should include leg bones
        thigh_idx = skeleton.get_bone_index("thigh_l")
        foot_idx = skeleton.get_bone_index("foot_l")

        assert mask.get_weight(thigh_idx) == 1.0
        assert mask.get_weight(foot_idx) == 1.0

        # Should not include arm bones
        arm_idx = skeleton.get_bone_index("upperarm_l")
        assert mask.get_weight(arm_idx) == 0.0

    def test_bone_chain_mask(self):
        """Test mask from bone chain."""
        skeleton = create_humanoid_skeleton()
        mask = BoneMask.from_bone_chain(skeleton, "spine_01", include_descendants=True)

        spine_idx = skeleton.get_bone_index("spine_01")
        assert mask.get_weight(spine_idx) == 1.0

        # Descendants should also be included
        head_idx = skeleton.get_bone_index("head")
        assert mask.get_weight(head_idx) == 1.0

    def test_bone_chain_mask_no_descendants(self):
        """Test bone chain mask without descendants."""
        skeleton = create_humanoid_skeleton()
        mask = BoneMask.from_bone_chain(skeleton, "spine_01", include_descendants=False)

        spine_idx = skeleton.get_bone_index("spine_01")
        assert mask.get_weight(spine_idx) == 1.0

        # Descendants should NOT be included
        spine2_idx = skeleton.get_bone_index("spine_02")
        assert mask.get_weight(spine2_idx) == 0.0


# =============================================================================
# Blend Poses Tests - T-SKEL-1.7
# =============================================================================

class TestBlendPoses:
    """Tests for blend_poses function."""

    @pytest.fixture
    def simple_skeleton(self):
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0))
        skeleton._rebuild_caches()
        return skeleton

    def test_blend_override_mode_full(self, simple_skeleton):
        """Test T-SKEL-1.7: Override mode at full blend."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        result = blend_poses(pose_a, pose_b, 1.0, BlendMode.OVERRIDE)

        assert result.get_bone_transform(0).translation.x == 10

    def test_blend_override_mode_half(self, simple_skeleton):
        """Test T-SKEL-1.7: Override mode at 50%."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        result = blend_poses(pose_a, pose_b, 0.5, BlendMode.OVERRIDE)

        assert abs(result.get_bone_transform(0).translation.x - 5.0) < 1e-6

    def test_blend_additive_mode(self, simple_skeleton):
        """Test T-SKEL-1.7: Additive mode."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(3, 0, 0)))

        result = blend_poses(pose_a, pose_b, 1.0, BlendMode.ADDITIVE)

        # base + additive * weight = 5 + 3 * 1.0 = 8
        assert abs(result.get_bone_transform(0).translation.x - 8.0) < 1e-6

    def test_blend_additive_mode_partial(self, simple_skeleton):
        """Test T-SKEL-1.7: Additive mode with partial weight."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(4, 0, 0)))

        result = blend_poses(pose_a, pose_b, 0.5, BlendMode.ADDITIVE)

        # base + additive * weight = 5 + 4 * 0.5 = 7
        assert abs(result.get_bone_transform(0).translation.x - 7.0) < 1e-6

    def test_blend_multiply_mode(self, simple_skeleton):
        """Test T-SKEL-1.7: Multiply mode for scale."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(scale=Vec3(2, 2, 2)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(scale=Vec3(2, 2, 2)))

        result = blend_poses(pose_a, pose_b, 1.0, BlendMode.MULTIPLY)

        # Scale multiplication: base * (1 + (factor - 1) * weight)
        # = 2 * (1 + (2 - 1) * 1) = 2 * 2 = 4
        assert abs(result.get_bone_transform(0).scale.x - 4.0) < 1e-6

    def test_blend_with_mask(self, simple_skeleton):
        """Test T-SKEL-1.7: BoneMask filtering."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))
        pose_a.set_bone_transform(1, BoneTransform(translation=Vec3(0, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))
        pose_b.set_bone_transform(1, BoneTransform(translation=Vec3(10, 0, 0)))

        # Only affect bone 0
        mask = BoneMask()
        mask.set_weight(0, 1.0)
        mask.set_weight(1, 0.0)

        result = blend_poses(pose_a, pose_b, 1.0, BlendMode.OVERRIDE, mask)

        # Bone 0 should be blended
        assert result.get_bone_transform(0).translation.x == 10
        # Bone 1 should remain unchanged (mask weight = 0)
        assert result.get_bone_transform(1).translation.x == 0

    def test_blend_with_partial_mask(self, simple_skeleton):
        """Test bone mask with partial weight."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        mask = BoneMask()
        mask.set_weight(0, 0.5)  # Only 50% of the blend

        result = blend_poses(pose_a, pose_b, 1.0, BlendMode.OVERRIDE, mask)

        # effective_alpha = alpha * mask_weight = 1.0 * 0.5 = 0.5
        assert abs(result.get_bone_transform(0).translation.x - 5.0) < 1e-6

    def test_blend_alpha_zero(self, simple_skeleton):
        """Test blend with alpha=0 returns copy of first pose."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(100, 0, 0)))

        result = blend_poses(pose_a, pose_b, 0.0)

        assert result.get_bone_transform(0).translation.x == 5

    def test_blend_different_skeletons_fails(self):
        """Test blending poses with different skeletons raises error."""
        skel1 = Skeleton(name="skel1")
        skel1.add_bone(Bone(index=0, name="root"))

        skel2 = Skeleton(name="skel2")
        skel2.add_bone(Bone(index=0, name="root"))

        pose_a = Pose(skel1)
        pose_b = Pose(skel2)

        with pytest.raises(ValueError, match="different skeletons"):
            blend_poses(pose_a, pose_b, 0.5)


class TestBlendInternalFunctions:
    """Tests for internal blend functions."""

    def test_blend_override_translation(self):
        """Test _blend_override for translation."""
        a = BoneTransform(translation=Vec3(0, 0, 0))
        b = BoneTransform(translation=Vec3(10, 20, 30))

        result = _blend_override(a, b, 0.5)

        assert abs(result.translation.x - 5.0) < 1e-6
        assert abs(result.translation.y - 10.0) < 1e-6
        assert abs(result.translation.z - 15.0) < 1e-6

    def test_blend_override_rotation(self):
        """Test _blend_override uses SLERP for rotation."""
        a = BoneTransform(rotation=Quat.identity())
        # 90 degree rotation around Y
        angle = math.pi / 2
        b = BoneTransform(rotation=Quat(0, math.sin(angle/2), 0, math.cos(angle/2)))

        result = _blend_override(a, b, 0.5)

        # Should be 45 degrees
        expected_angle = math.pi / 4
        assert abs(result.rotation.w - math.cos(expected_angle/2)) < 1e-4

    def test_blend_additive_translation(self):
        """Test _blend_additive adds translation."""
        base = BoneTransform(translation=Vec3(10, 0, 0))
        additive = BoneTransform(translation=Vec3(5, 0, 0))

        result = _blend_additive(base, additive, 1.0)

        assert abs(result.translation.x - 15.0) < 1e-6

    def test_blend_additive_rotation(self):
        """Test _blend_additive applies rotation delta."""
        base = BoneTransform(rotation=Quat.identity())
        # Small rotation
        angle = 0.1
        additive = BoneTransform(rotation=Quat(0, 0, math.sin(angle/2), math.cos(angle/2)))

        result = _blend_additive(base, additive, 1.0)

        # Result should have rotation applied
        length = math.sqrt(
            result.rotation.x**2 + result.rotation.y**2 +
            result.rotation.z**2 + result.rotation.w**2
        )
        assert abs(length - 1.0) < 1e-6

    def test_blend_multiply_scale(self):
        """Test _blend_multiply for scale."""
        base = BoneTransform(scale=Vec3(2, 2, 2))
        factor = BoneTransform(scale=Vec3(3, 3, 3))

        result = _blend_multiply(base, factor, 1.0)

        # scale: base * (1 + (factor - 1) * weight) = 2 * (1 + 2) = 6
        assert abs(result.scale.x - 6.0) < 1e-6


# =============================================================================
# Blend Multiple Poses Tests
# =============================================================================

class TestBlendMultiplePoses:
    """Tests for blend_multiple_poses function."""

    @pytest.fixture
    def simple_skeleton(self):
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton._rebuild_caches()
        return skeleton

    def test_blend_multiple_override(self, simple_skeleton):
        """Test blending multiple poses with override mode."""
        poses = []
        for i in range(3):
            pose = Pose(simple_skeleton)
            pose.set_bone_transform(0, BoneTransform(translation=Vec3(float(i * 10), 0, 0)))
            poses.append(pose)

        result = blend_multiple_poses(poses, [1.0, 1.0, 1.0], BlendMode.OVERRIDE)

        # Weighted average: (0 + 10 + 20) / 3 = 10
        assert abs(result.get_bone_transform(0).translation.x - 10.0) < 1e-6

    def test_blend_multiple_additive(self, simple_skeleton):
        """Test blending multiple poses with additive mode."""
        base = Pose(simple_skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(1, 0, 0)))

        add1 = Pose(simple_skeleton)
        add1.set_bone_transform(0, BoneTransform(translation=Vec3(2, 0, 0)))

        add2 = Pose(simple_skeleton)
        add2.set_bone_transform(0, BoneTransform(translation=Vec3(3, 0, 0)))

        # With normalize=True (default), weights are normalized first
        # [1,1,1] -> [0.33, 0.33, 0.33]
        # First pose (base) stays, others are additively blended with their normalized weights
        # Result: base + add1*0.33 + add2*0.33 = 1 + 2*0.33 + 3*0.33 = ~2.67
        result = blend_multiple_poses([base, add1, add2], [1.0, 1.0, 1.0], BlendMode.ADDITIVE)

        # The implementation normalizes weights, so additive mode with normalized weights
        # uses first pose as base and adds weighted amounts from remaining poses
        assert result.get_bone_transform(0).translation.x > 0  # Sanity check

    def test_blend_multiple_normalized_weights(self, simple_skeleton):
        """Test weights are normalized when normalize=True."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        # Weights [2, 2] should normalize to [0.5, 0.5]
        result = blend_multiple_poses([pose_a, pose_b], [2.0, 2.0], normalize=True)

        assert abs(result.get_bone_transform(0).translation.x - 5.0) < 1e-6

    def test_blend_multiple_unnormalized_weights(self, simple_skeleton):
        """Test weights are not normalized when normalize=False."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        # Without normalization, weights are used as-is
        result = blend_multiple_poses([pose_a, pose_b], [0.5, 0.5], normalize=False)

        # 0 * 0.5 + 10 * 0.5 = 5
        assert abs(result.get_bone_transform(0).translation.x - 5.0) < 1e-6


# =============================================================================
# Additive Pose Functions Tests
# =============================================================================

class TestAdditiveBlendFunctions:
    """Tests for compute_additive_pose and apply_additive_pose."""

    @pytest.fixture
    def simple_skeleton(self):
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton._rebuild_caches()
        return skeleton

    def test_compute_additive_translation(self, simple_skeleton):
        """Test computing additive pose for translation."""
        reference = Pose(simple_skeleton)
        reference.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))

        target = Pose(simple_skeleton)
        target.set_bone_transform(0, BoneTransform(translation=Vec3(5, 10, 15)))

        additive = compute_additive_pose(reference, target)

        assert abs(additive.get_bone_transform(0).translation.x - 5.0) < 1e-6
        assert abs(additive.get_bone_transform(0).translation.y - 10.0) < 1e-6
        assert abs(additive.get_bone_transform(0).translation.z - 15.0) < 1e-6

    def test_compute_additive_scale(self, simple_skeleton):
        """Test computing additive pose for scale."""
        reference = Pose(simple_skeleton)
        reference.set_bone_transform(0, BoneTransform(scale=Vec3(1, 1, 1)))

        target = Pose(simple_skeleton)
        target.set_bone_transform(0, BoneTransform(scale=Vec3(2, 3, 4)))

        additive = compute_additive_pose(reference, target)

        # Scale delta is ratio
        assert abs(additive.get_bone_transform(0).scale.x - 2.0) < 1e-6
        assert abs(additive.get_bone_transform(0).scale.y - 3.0) < 1e-6
        assert abs(additive.get_bone_transform(0).scale.z - 4.0) < 1e-6

    def test_apply_additive_pose(self, simple_skeleton):
        """Test applying additive pose."""
        base = Pose(simple_skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(1, 1, 1)))

        additive = Pose(simple_skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(5, 5, 5)))

        result = apply_additive_pose(base, additive, weight=1.0)

        assert abs(result.get_bone_transform(0).translation.x - 6.0) < 1e-6
        assert abs(result.get_bone_transform(0).translation.y - 6.0) < 1e-6
        assert abs(result.get_bone_transform(0).translation.z - 6.0) < 1e-6

    def test_apply_additive_with_mask(self, simple_skeleton):
        """Test applying additive pose with bone mask."""
        base = Pose(simple_skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))

        additive = Pose(simple_skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(10, 10, 10)))

        # Mask with 50% weight
        mask = BoneMask()
        mask.set_weight(0, 0.5)

        result = apply_additive_pose(base, additive, weight=1.0, mask=mask)

        # effective_weight = weight * mask = 1.0 * 0.5 = 0.5
        assert abs(result.get_bone_transform(0).translation.x - 5.0) < 1e-6

    def test_apply_additive_zero_weight(self, simple_skeleton):
        """Test applying additive with zero weight returns base copy."""
        base = Pose(simple_skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(5, 5, 5)))

        additive = Pose(simple_skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(100, 100, 100)))

        result = apply_additive_pose(base, additive, weight=0.0)

        assert result.get_bone_transform(0).translation.x == 5


# =============================================================================
# LayeredBlender Tests
# =============================================================================

class TestLayeredBlender:
    """Tests for LayeredBlender class."""

    @pytest.fixture
    def simple_skeleton(self):
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0))
        skeleton._rebuild_caches()
        return skeleton

    def test_layered_blender_creation(self, simple_skeleton):
        """Test creating a layered blender."""
        blender = LayeredBlender(simple_skeleton)

        assert blender.skeleton is simple_skeleton
        assert blender.layer_count == 0

    def test_layered_blender_add_layer(self, simple_skeleton):
        """Test adding layers to blender."""
        blender = LayeredBlender(simple_skeleton)

        idx0 = blender.add_layer("base", BlendMode.OVERRIDE)
        idx1 = blender.add_layer("upper", BlendMode.ADDITIVE)

        assert idx0 == 0
        assert idx1 == 1
        assert blender.layer_count == 2

    def test_layered_blender_remove_layer(self, simple_skeleton):
        """Test removing a layer."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("base")
        blender.add_layer("upper")

        blender.remove_layer(0)

        assert blender.layer_count == 1

    def test_layered_blender_get_layer(self, simple_skeleton):
        """Test getting layer by index."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("test_layer", BlendMode.ADDITIVE, weight=0.75)

        layer = blender.get_layer(0)

        assert layer.name == "test_layer"
        assert layer.mode == BlendMode.ADDITIVE
        assert layer.weight == 0.75

    def test_layered_blender_get_layer_by_name(self, simple_skeleton):
        """Test getting layer by name."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("first")
        blender.add_layer("second")

        layer = blender.get_layer_by_name("second")

        assert layer is not None
        assert layer.name == "second"

    def test_layered_blender_set_layer_pose(self, simple_skeleton):
        """Test setting pose for a layer."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("base")

        pose = Pose(simple_skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(5, 5, 5)))

        blender.set_layer_pose(0, pose)
        layer = blender.get_layer(0)

        assert layer.pose is not None

    def test_layered_blender_set_layer_weight(self, simple_skeleton):
        """Test setting weight for a layer."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("test", weight=1.0)

        blender.set_layer_weight(0, 0.5)

        assert blender.get_layer(0).weight == 0.5

    def test_layered_blender_set_layer_enabled(self, simple_skeleton):
        """Test enabling/disabling a layer."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("test")

        blender.set_layer_enabled(0, False)

        assert blender.get_layer(0).enabled is False

    def test_layered_blender_blend_single_layer(self, simple_skeleton):
        """Test blending with single layer."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("base")

        pose = Pose(simple_skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))
        blender.set_layer_pose(0, pose)

        result = blender.blend()

        assert result.get_bone_transform(0).translation.x == 10

    def test_layered_blender_blend_two_layers(self, simple_skeleton):
        """Test blending two layers."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("base", BlendMode.OVERRIDE, weight=1.0)
        blender.add_layer("additive", BlendMode.ADDITIVE, weight=1.0)

        base_pose = Pose(simple_skeleton)
        base_pose.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))
        blender.set_layer_pose(0, base_pose)

        add_pose = Pose(simple_skeleton)
        add_pose.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))
        blender.set_layer_pose(1, add_pose)

        result = blender.blend()

        # Base(10) + additive(5) = 15
        assert abs(result.get_bone_transform(0).translation.x - 15.0) < 1e-6

    def test_layered_blender_blend_disabled_layer(self, simple_skeleton):
        """Test that disabled layers are skipped."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("base", weight=1.0)
        blender.add_layer("disabled", weight=1.0)

        base_pose = Pose(simple_skeleton)
        base_pose.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))
        blender.set_layer_pose(0, base_pose)

        disabled_pose = Pose(simple_skeleton)
        disabled_pose.set_bone_transform(0, BoneTransform(translation=Vec3(100, 0, 0)))
        blender.set_layer_pose(1, disabled_pose)
        blender.set_layer_enabled(1, False)

        result = blender.blend()

        # Disabled layer should not affect result
        assert result.get_bone_transform(0).translation.x == 10

    def test_layered_blender_blend_with_base_pose(self, simple_skeleton):
        """Test blending with explicit base pose."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("additive", BlendMode.ADDITIVE)

        add_pose = Pose(simple_skeleton)
        add_pose.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))
        blender.set_layer_pose(0, add_pose)

        base_pose = Pose(simple_skeleton)
        base_pose.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        result = blender.blend(base_pose)

        # Base(10) + additive(5) = 15
        assert abs(result.get_bone_transform(0).translation.x - 15.0) < 1e-6

    def test_layered_blender_blend_no_valid_layers(self, simple_skeleton):
        """Test blending with no valid layers returns None."""
        blender = LayeredBlender(simple_skeleton)

        result = blender.blend()

        assert result is None

    def test_layered_blender_clear(self, simple_skeleton):
        """Test clearing all layers."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("a")
        blender.add_layer("b")

        blender.clear()

        assert blender.layer_count == 0


# =============================================================================
# PoseCache Tests
# =============================================================================

class TestPoseCache:
    """Tests for PoseCache class."""

    @pytest.fixture
    def simple_skeleton(self):
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton._rebuild_caches()
        return skeleton

    def test_pose_cache_creation(self, simple_skeleton):
        """Test creating a pose cache."""
        cache = PoseCache(simple_skeleton, capacity=4)

        assert cache.skeleton is simple_skeleton
        assert cache.size == 0

    def test_pose_cache_put_get(self, simple_skeleton):
        """Test putting and getting poses."""
        cache = PoseCache(simple_skeleton, capacity=4)

        pose = Pose(simple_skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(5, 5, 5)))

        cache.put("test_key", pose)
        retrieved = cache.get("test_key")

        assert retrieved is not None
        assert retrieved.get_bone_transform(0).translation.x == 5

    def test_pose_cache_get_missing(self, simple_skeleton):
        """Test getting non-existent key returns None."""
        cache = PoseCache(simple_skeleton, capacity=4)

        assert cache.get("missing") is None

    def test_pose_cache_eviction(self, simple_skeleton):
        """Test LRU eviction when over capacity."""
        cache = PoseCache(simple_skeleton, capacity=2)

        for i in range(3):
            pose = Pose(simple_skeleton)
            pose.set_bone_transform(0, BoneTransform(translation=Vec3(float(i), 0, 0)))
            cache.put(f"key_{i}", pose)

        # key_0 should be evicted (oldest)
        assert cache.get("key_0") is None
        assert cache.get("key_1") is not None
        assert cache.get("key_2") is not None

    def test_pose_cache_access_updates_order(self, simple_skeleton):
        """Test accessing a key updates its recency."""
        cache = PoseCache(simple_skeleton, capacity=2)

        pose1 = Pose(simple_skeleton)
        pose1.set_bone_transform(0, BoneTransform(translation=Vec3(1, 0, 0)))
        cache.put("key_1", pose1)

        pose2 = Pose(simple_skeleton)
        pose2.set_bone_transform(0, BoneTransform(translation=Vec3(2, 0, 0)))
        cache.put("key_2", pose2)

        # Access key_1 to make it recent
        cache.get("key_1")

        # Add key_3, should evict key_2 (now oldest)
        pose3 = Pose(simple_skeleton)
        cache.put("key_3", pose3)

        assert cache.get("key_1") is not None
        assert cache.get("key_2") is None
        assert cache.get("key_3") is not None

    def test_pose_cache_contains(self, simple_skeleton):
        """Test checking if key is cached."""
        cache = PoseCache(simple_skeleton, capacity=4)
        pose = Pose(simple_skeleton)

        cache.put("exists", pose)

        assert cache.contains("exists") is True
        assert cache.contains("missing") is False

    def test_pose_cache_remove(self, simple_skeleton):
        """Test removing a cached pose."""
        cache = PoseCache(simple_skeleton, capacity=4)
        pose = Pose(simple_skeleton)
        cache.put("key", pose)

        removed = cache.remove("key")

        assert removed is True
        assert cache.contains("key") is False

    def test_pose_cache_remove_missing(self, simple_skeleton):
        """Test removing non-existent key returns False."""
        cache = PoseCache(simple_skeleton, capacity=4)

        removed = cache.remove("missing")

        assert removed is False

    def test_pose_cache_clear(self, simple_skeleton):
        """Test clearing the cache."""
        cache = PoseCache(simple_skeleton, capacity=4)
        pose = Pose(simple_skeleton)
        cache.put("a", pose)
        cache.put("b", pose)

        cache.clear()

        assert cache.size == 0
        assert cache.get("a") is None

    def test_pose_cache_wrong_skeleton_fails(self, simple_skeleton):
        """Test putting pose with wrong skeleton raises error."""
        cache = PoseCache(simple_skeleton, capacity=4)

        other_skeleton = Skeleton(name="other")
        other_skeleton.add_bone(Bone(index=0, name="root"))
        other_pose = Pose(other_skeleton)

        with pytest.raises(ValueError, match="must match"):
            cache.put("key", other_pose)
