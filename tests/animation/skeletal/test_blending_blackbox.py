"""
Blackbox tests for pose blending operations.

Tests cover blending two poses, blend modes, bone masks, and the
LayeredBlender system without knowledge of implementation details.
"""

import math
import pytest


class TestBasicBlending:
    """Tests for basic pose blending operations."""

    def test_blend_two_poses_equal_weights(self):
        """Blending two poses with equal weights should average them."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose1 = Pose(skeleton)
        pose1.set_bone_transform(0, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))

        pose2 = Pose(skeleton)
        pose2.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        result = blend_poses(pose1, pose2, 0.5)

        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(5.0)

    def test_blend_at_zero(self):
        """Blending at alpha=0 should return first pose."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose1 = Pose(skeleton)
        pose1.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 0.0, 0.0)))

        pose2 = Pose(skeleton)
        pose2.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        result = blend_poses(pose1, pose2, 0.0)

        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(5.0)

    def test_blend_at_one(self):
        """Blending at alpha=1 should return second pose."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose1 = Pose(skeleton)
        pose1.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 0.0, 0.0)))

        pose2 = Pose(skeleton)
        pose2.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        result = blend_poses(pose1, pose2, 1.0)

        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(10.0)

    def test_blend_unequal_alpha(self):
        """Blending with unequal alpha should favor higher weight."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose1 = Pose(skeleton)
        pose1.set_bone_transform(0, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))

        pose2 = Pose(skeleton)
        pose2.set_bone_transform(0, BoneTransform(translation=Vec3(100.0, 0.0, 0.0)))

        # 75% weight on pose2
        result = blend_poses(pose1, pose2, 0.75)

        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(75.0)


class TestBlendModes:
    """Tests for different blend modes."""

    def test_blend_mode_override(self):
        """Override mode should interpolate between poses."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses, BlendMode
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        base = Pose(skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        override = Pose(skeleton)
        override.set_bone_transform(0, BoneTransform(translation=Vec3(50.0, 0.0, 0.0)))

        result = blend_poses(base, override, 1.0, mode=BlendMode.OVERRIDE)

        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(50.0)

    def test_blend_mode_additive(self):
        """Additive mode should add transforms together."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses, BlendMode
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        base = Pose(skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        additive = Pose(skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 0.0, 0.0)))

        result = blend_poses(base, additive, 1.0, mode=BlendMode.ADDITIVE)

        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(15.0)

    def test_blend_mode_multiply_scale(self):
        """Multiply mode should multiply scale transforms."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses, BlendMode
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        base = Pose(skeleton)
        base.set_bone_transform(0, BoneTransform(scale=Vec3(2.0, 2.0, 2.0)))

        mult = Pose(skeleton)
        mult.set_bone_transform(0, BoneTransform(scale=Vec3(3.0, 3.0, 3.0)))

        result = blend_poses(base, mult, 1.0, mode=BlendMode.MULTIPLY)

        transform = result.get_bone_transform(0)
        assert transform.scale.x == pytest.approx(6.0)

    def test_partial_additive_blend(self):
        """Additive blend with alpha < 1 should scale the addition."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses, BlendMode
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        base = Pose(skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        additive = Pose(skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(20.0, 0.0, 0.0)))

        result = blend_poses(base, additive, 0.5, mode=BlendMode.ADDITIVE)

        transform = result.get_bone_transform(0)
        # 10 + 20 * 0.5 = 20
        assert transform.translation.x == pytest.approx(20.0)


class TestBoneMask:
    """Tests for bone mask filtering."""

    def test_bone_mask_get_weight(self):
        """Should get weight for specific bones."""
        from engine.animation.skeletal.blending import BoneMask

        mask = BoneMask(bone_weights={0: 1.0, 1: 0.5})

        assert mask.get_weight(0) == pytest.approx(1.0)
        assert mask.get_weight(1) == pytest.approx(0.5)
        assert mask.get_weight(2) == pytest.approx(0.0)  # default

    def test_bone_mask_with_default_weight(self):
        """Should use default weight for unspecified bones."""
        from engine.animation.skeletal.blending import BoneMask

        mask = BoneMask(default_weight=1.0)

        assert mask.get_weight(0) == pytest.approx(1.0)
        assert mask.get_weight(100) == pytest.approx(1.0)

    def test_bone_mask_set_weight(self):
        """Should set weight for bone."""
        from engine.animation.skeletal.blending import BoneMask

        mask = BoneMask()
        mask.set_weight(0, 0.75)

        assert mask.get_weight(0) == pytest.approx(0.75)

    def test_bone_mask_include_bone(self):
        """Include should set weight to 1.0."""
        from engine.animation.skeletal.blending import BoneMask

        mask = BoneMask()
        mask.include_bone(0)

        assert mask.get_weight(0) == pytest.approx(1.0)

    def test_bone_mask_exclude_bone(self):
        """Exclude should set weight to 0.0."""
        from engine.animation.skeletal.blending import BoneMask

        mask = BoneMask(default_weight=1.0)
        mask.exclude_bone(0)

        assert mask.get_weight(0) == pytest.approx(0.0)

    def test_bone_mask_invert(self):
        """Invert should flip weights."""
        from engine.animation.skeletal.blending import BoneMask

        mask = BoneMask(bone_weights={0: 1.0, 1: 0.0})
        inverted = mask.invert()

        assert inverted.get_weight(0) == pytest.approx(0.0)
        assert inverted.get_weight(1) == pytest.approx(1.0)

    def test_bone_mask_combine(self):
        """Combine should multiply weights."""
        from engine.animation.skeletal.blending import BoneMask

        mask1 = BoneMask(bone_weights={0: 0.5})
        mask2 = BoneMask(bone_weights={0: 0.5})

        combined = mask1.combine(mask2)

        assert combined.get_weight(0) == pytest.approx(0.25)


class TestBlendWithMask:
    """Tests for blending with bone masks."""

    def test_blend_with_bone_mask_single_bone(self):
        """Blend should only affect masked bones."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses, BoneMask
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="arm", parent_index=1))

        pose1 = Pose(skeleton)
        for i in range(3):
            pose1.set_bone_transform(i, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))

        pose2 = Pose(skeleton)
        for i in range(3):
            pose2.set_bone_transform(i, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        # Only blend bone 1 (spine)
        mask = BoneMask(bone_weights={1: 1.0})
        result = blend_poses(pose1, pose2, 0.5, mask=mask)

        # Root should be unchanged (from pose1)
        assert result.get_bone_transform(0).translation.x == pytest.approx(0.0)
        # Spine should be blended
        assert result.get_bone_transform(1).translation.x == pytest.approx(5.0)
        # Arm should be unchanged
        assert result.get_bone_transform(2).translation.x == pytest.approx(0.0)

    def test_blend_with_bone_mask_multiple_bones(self):
        """Blend should affect all bones in mask."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses, BoneMask
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="left_arm", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="right_arm", parent_index=0))
        skeleton.add_bone(Bone(index=3, name="head", parent_index=0))

        pose1 = Pose(skeleton)
        for i in range(4):
            pose1.set_bone_transform(i, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))

        pose2 = Pose(skeleton)
        for i in range(4):
            pose2.set_bone_transform(i, BoneTransform(translation=Vec3(20.0, 0.0, 0.0)))

        # Only blend arms (indices 1 and 2)
        mask = BoneMask(bone_weights={1: 1.0, 2: 1.0})
        result = blend_poses(pose1, pose2, 0.5, mask=mask)

        assert result.get_bone_transform(0).translation.x == pytest.approx(0.0)
        assert result.get_bone_transform(1).translation.x == pytest.approx(10.0)
        assert result.get_bone_transform(2).translation.x == pytest.approx(10.0)
        assert result.get_bone_transform(3).translation.x == pytest.approx(0.0)


class TestBlendingRotations:
    """Tests for rotation blending behavior."""

    def test_blend_identity_rotations(self):
        """Blending identity rotations should return identity."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses
        from engine.core.math import Quat

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose1 = Pose(skeleton)
        pose1.set_bone_transform(0, BoneTransform(rotation=Quat.identity()))

        pose2 = Pose(skeleton)
        pose2.set_bone_transform(0, BoneTransform(rotation=Quat.identity()))

        result = blend_poses(pose1, pose2, 0.5)

        rot = result.get_bone_transform(0).rotation
        assert rot.w == pytest.approx(1.0, abs=1e-5)

    def test_blend_opposite_rotations(self):
        """Blending opposite rotations should give midpoint."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses
        from engine.core.math import Quat

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        # Identity
        pose1 = Pose(skeleton)
        pose1.set_bone_transform(0, BoneTransform(rotation=Quat.identity()))

        # 90 degrees around Y
        angle = math.pi / 2
        qw = math.cos(angle / 2)
        qy = math.sin(angle / 2)

        pose2 = Pose(skeleton)
        pose2.set_bone_transform(0, BoneTransform(rotation=Quat(0.0, qy, 0.0, qw)))

        result = blend_poses(pose1, pose2, 0.5)

        # Should be 45 degrees around Y
        rot = result.get_bone_transform(0).rotation
        # Verify it's a valid quaternion
        magnitude = math.sqrt(rot.x**2 + rot.y**2 + rot.z**2 + rot.w**2)
        assert magnitude == pytest.approx(1.0, abs=1e-5)


class TestBlendingScales:
    """Tests for scale blending behavior."""

    def test_blend_uniform_scales(self):
        """Blending uniform scales should average them."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose1 = Pose(skeleton)
        pose1.set_bone_transform(0, BoneTransform(scale=Vec3(1.0, 1.0, 1.0)))

        pose2 = Pose(skeleton)
        pose2.set_bone_transform(0, BoneTransform(scale=Vec3(3.0, 3.0, 3.0)))

        result = blend_poses(pose1, pose2, 0.5)

        scale = result.get_bone_transform(0).scale
        assert scale.x == pytest.approx(2.0)
        assert scale.y == pytest.approx(2.0)
        assert scale.z == pytest.approx(2.0)

    def test_blend_non_uniform_scales(self):
        """Blending non-uniform scales should blend each axis."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_poses
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose1 = Pose(skeleton)
        pose1.set_bone_transform(0, BoneTransform(scale=Vec3(1.0, 2.0, 3.0)))

        pose2 = Pose(skeleton)
        pose2.set_bone_transform(0, BoneTransform(scale=Vec3(3.0, 4.0, 5.0)))

        result = blend_poses(pose1, pose2, 0.5)

        scale = result.get_bone_transform(0).scale
        assert scale.x == pytest.approx(2.0)
        assert scale.y == pytest.approx(3.0)
        assert scale.z == pytest.approx(4.0)


class TestBlendMultiplePoses:
    """Tests for blending multiple poses."""

    def test_blend_multiple_poses_equal_weights(self):
        """Should blend multiple poses with equal weights."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_multiple_poses
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose1 = Pose(skeleton)
        pose1.set_bone_transform(0, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))

        pose2 = Pose(skeleton)
        pose2.set_bone_transform(0, BoneTransform(translation=Vec3(30.0, 0.0, 0.0)))

        pose3 = Pose(skeleton)
        pose3.set_bone_transform(0, BoneTransform(translation=Vec3(60.0, 0.0, 0.0)))

        result = blend_multiple_poses([pose1, pose2, pose3], [1/3, 1/3, 1/3])

        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(30.0, abs=0.1)

    def test_blend_multiple_poses_different_weights(self):
        """Blending with different weights."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import blend_multiple_poses
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        poses = []
        for i in range(4):
            pose = Pose(skeleton)
            pose.set_bone_transform(0, BoneTransform(translation=Vec3(i * 10.0, 0.0, 0.0)))
            poses.append(pose)

        # Weights: 0.1, 0.2, 0.3, 0.4
        weights = [0.1, 0.2, 0.3, 0.4]
        result = blend_multiple_poses(poses, weights)

        # Expected: 0*0.1 + 10*0.2 + 20*0.3 + 30*0.4 = 0 + 2 + 6 + 12 = 20
        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(20.0)

    def test_blend_empty_list_raises(self):
        """Blending empty list should raise error."""
        from engine.animation.skeletal.blending import blend_multiple_poses

        with pytest.raises(ValueError):
            blend_multiple_poses([], [])

    def test_blend_mismatched_lengths_raises(self):
        """Different number of poses and weights should raise error."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose
        from engine.animation.skeletal.blending import blend_multiple_poses

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose = Pose(skeleton)

        with pytest.raises(ValueError):
            blend_multiple_poses([pose, pose], [0.5])


class TestComputeAdditivePose:
    """Tests for computing additive poses."""

    def test_compute_additive_pose(self):
        """Should compute delta between poses."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import compute_additive_pose
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        reference = Pose(skeleton)
        reference.set_bone_transform(0, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))

        target = Pose(skeleton)
        target.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 5.0, 0.0)))

        additive = compute_additive_pose(reference, target)

        delta = additive.get_bone_transform(0)
        assert delta.translation.x == pytest.approx(10.0)
        assert delta.translation.y == pytest.approx(5.0)


class TestApplyAdditivePose:
    """Tests for applying additive poses."""

    def test_apply_additive_pose(self):
        """Should apply additive pose on top of base."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import apply_additive_pose
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        base = Pose(skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        additive = Pose(skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 0.0, 0.0)))

        result = apply_additive_pose(base, additive, 1.0)

        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(15.0)

    def test_apply_additive_pose_with_mask(self):
        """Should only apply to masked bones."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import apply_additive_pose, BoneMask
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))

        base = Pose(skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))
        base.set_bone_transform(1, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))

        additive = Pose(skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))
        additive.set_bone_transform(1, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        mask = BoneMask(bone_weights={1: 1.0})  # Only apply to spine
        result = apply_additive_pose(base, additive, 1.0, mask=mask)

        # Root should be unchanged
        assert result.get_bone_transform(0).translation.x == pytest.approx(0.0)
        # Spine should have additive applied
        assert result.get_bone_transform(1).translation.x == pytest.approx(10.0)


class TestLayeredBlender:
    """Tests for LayeredBlender system."""

    def test_layered_blender_add_layer(self):
        """Should add layers."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.blending import LayeredBlender

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        blender = LayeredBlender(skeleton)
        idx = blender.add_layer("base")

        assert idx == 0
        assert blender.layer_count == 1

    def test_layered_blender_set_layer_pose(self):
        """Should set pose for layer."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import LayeredBlender
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        blender = LayeredBlender(skeleton)
        blender.add_layer("base")

        pose = Pose(skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 0.0, 0.0)))
        blender.set_layer_pose(0, pose)

        result = blender.blend()
        assert result is not None
        assert result.get_bone_transform(0).translation.x == pytest.approx(5.0)

    def test_layered_blender_multiple_layers(self):
        """Should blend multiple layers."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import LayeredBlender, BlendMode
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        blender = LayeredBlender(skeleton)
        blender.add_layer("base", mode=BlendMode.OVERRIDE)
        blender.add_layer("additive", mode=BlendMode.ADDITIVE)

        base_pose = Pose(skeleton)
        base_pose.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))
        blender.set_layer_pose(0, base_pose)

        additive_pose = Pose(skeleton)
        additive_pose.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 0.0, 0.0)))
        blender.set_layer_pose(1, additive_pose)

        result = blender.blend()
        # Base 10 + additive 5 = 15
        assert result.get_bone_transform(0).translation.x == pytest.approx(15.0)

    def test_layered_blender_enable_disable(self):
        """Should enable/disable layers."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import LayeredBlender, BlendMode
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        blender = LayeredBlender(skeleton)
        blender.add_layer("base")
        blender.add_layer("additive", mode=BlendMode.ADDITIVE)

        base_pose = Pose(skeleton)
        base_pose.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))
        blender.set_layer_pose(0, base_pose)

        additive_pose = Pose(skeleton)
        additive_pose.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 0.0, 0.0)))
        blender.set_layer_pose(1, additive_pose)

        # Disable additive layer
        blender.set_layer_enabled(1, False)

        result = blender.blend()
        # Only base should apply
        assert result.get_bone_transform(0).translation.x == pytest.approx(10.0)


class TestPoseCache:
    """Tests for PoseCache."""

    def test_pose_cache_put_get(self):
        """Should put and get poses."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import PoseCache
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        cache = PoseCache(skeleton, capacity=4)

        pose = Pose(skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 0.0, 0.0)))
        cache.put("test", pose)

        retrieved = cache.get("test")
        assert retrieved is not None
        assert retrieved.get_bone_transform(0).translation.x == pytest.approx(5.0)

    def test_pose_cache_eviction(self):
        """Should evict oldest when at capacity."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.animation.skeletal.blending import PoseCache
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        cache = PoseCache(skeleton, capacity=2)

        for i in range(3):
            pose = Pose(skeleton)
            pose.set_bone_transform(0, BoneTransform(translation=Vec3(float(i), 0.0, 0.0)))
            cache.put(f"pose_{i}", pose)

        # First pose should be evicted
        assert not cache.contains("pose_0")
        assert cache.contains("pose_1")
        assert cache.contains("pose_2")
