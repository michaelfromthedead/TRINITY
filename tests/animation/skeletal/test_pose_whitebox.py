"""Whitebox tests for pose.py.

Tests pose data, quaternion SLERP, bone transforms, pose interpolation,
and additive blending.

Acceptance criteria:
- T-SKEL-1.2: Quaternion SLERP
  - SLERP normalization
  - Near-identity fallback
  - Antipodality handling
"""

import math
import pytest
from engine.core.math import Transform, Vec3, Quat, Mat4
from engine.animation.skeletal.skeleton import Skeleton, Bone, create_humanoid_skeleton
from engine.animation.skeletal.pose import (
    BoneTransform, Pose, PoseSpace, PoseBuffer,
    lerp_poses, additive_blend, compute_additive_pose, blend_multiple_poses,
    WEIGHT_EPSILON, SCALE_EPSILON
)


# =============================================================================
# BoneTransform Tests
# =============================================================================

class TestBoneTransform:
    """Tests for BoneTransform dataclass."""

    def test_bone_transform_default(self):
        """Test default bone transform is identity-like."""
        bt = BoneTransform()
        assert bt.translation.x == 0
        assert bt.translation.y == 0
        assert bt.translation.z == 0
        assert bt.rotation.w == 1  # Identity quaternion
        assert bt.scale.x == 1
        assert bt.scale.y == 1
        assert bt.scale.z == 1

    def test_bone_transform_identity_factory(self):
        """Test identity factory method."""
        bt = BoneTransform.identity()
        assert bt.translation == Vec3.zero()
        assert bt.scale == Vec3.one()

    def test_bone_transform_to_transform(self):
        """Test conversion to Transform object."""
        bt = BoneTransform(
            translation=Vec3(1, 2, 3),
            rotation=Quat(0, 0, 0, 1),
            scale=Vec3(2, 2, 2)
        )
        t = bt.to_transform()

        assert t.translation.x == 1
        assert t.translation.y == 2
        assert t.translation.z == 3
        assert t.scale.x == 2

    def test_bone_transform_from_transform(self):
        """Test creation from Transform object."""
        t = Transform(
            translation=Vec3(5, 6, 7),
            rotation=Quat(0, 0, 0, 1),
            scale=Vec3(3, 3, 3)
        )
        bt = BoneTransform.from_transform(t)

        assert bt.translation.x == 5
        assert bt.translation.y == 6
        assert bt.translation.z == 7
        assert bt.scale.x == 3

    def test_bone_transform_copy(self):
        """Test deep copy of bone transform."""
        original = BoneTransform(
            translation=Vec3(1, 2, 3),
            rotation=Quat(0.5, 0.5, 0.5, 0.5),
            scale=Vec3(2, 2, 2)
        )
        copied = original.copy()

        assert copied.translation.x == original.translation.x
        # Modify original
        original.translation.x = 999
        assert copied.translation.x == 1

    def test_bone_transform_equality(self):
        """Test bone transform equality comparison."""
        bt1 = BoneTransform(
            translation=Vec3(1, 2, 3),
            rotation=Quat(0, 0, 0, 1),
            scale=Vec3(1, 1, 1)
        )
        bt2 = BoneTransform(
            translation=Vec3(1, 2, 3),
            rotation=Quat(0, 0, 0, 1),
            scale=Vec3(1, 1, 1)
        )

        assert bt1 == bt2


class TestBoneTransformLerp:
    """Tests for T-SKEL-1.2: SLERP and LERP interpolation."""

    def test_lerp_translation(self):
        """Test translation linear interpolation."""
        a = BoneTransform(translation=Vec3(0, 0, 0))
        b = BoneTransform(translation=Vec3(10, 0, 0))

        result = a.lerp(b, 0.5)
        assert abs(result.translation.x - 5.0) < 1e-6

    def test_lerp_scale(self):
        """Test scale linear interpolation."""
        a = BoneTransform(scale=Vec3(1, 1, 1))
        b = BoneTransform(scale=Vec3(3, 3, 3))

        result = a.lerp(b, 0.5)
        assert abs(result.scale.x - 2.0) < 1e-6

    def test_slerp_rotation(self):
        """Test rotation spherical linear interpolation."""
        # Identity to 90 degree rotation around Y
        a = BoneTransform(rotation=Quat.identity())
        angle = math.pi / 2
        b = BoneTransform(rotation=Quat(0, math.sin(angle/2), 0, math.cos(angle/2)))

        result = a.lerp(b, 0.5)

        # At t=0.5, should be 45 degrees
        expected_angle = math.pi / 4
        # Verify rotation is approximately 45 degrees
        assert abs(result.rotation.w - math.cos(expected_angle/2)) < 1e-4

    def test_slerp_normalization(self):
        """Test SLERP output is normalized quaternion."""
        a = BoneTransform(rotation=Quat(0.1, 0.2, 0.3, 0.9).normalized())
        b = BoneTransform(rotation=Quat(0.5, 0.5, 0.5, 0.5).normalized())

        result = a.lerp(b, 0.5)
        length = math.sqrt(
            result.rotation.x**2 + result.rotation.y**2 +
            result.rotation.z**2 + result.rotation.w**2
        )
        assert abs(length - 1.0) < 1e-6

    def test_lerp_at_zero(self):
        """Test lerp at t=0 returns first transform."""
        a = BoneTransform(translation=Vec3(1, 2, 3))
        b = BoneTransform(translation=Vec3(10, 20, 30))

        result = a.lerp(b, 0.0)
        assert result.translation.x == 1
        assert result.translation.y == 2
        assert result.translation.z == 3

    def test_lerp_at_one(self):
        """Test lerp at t=1 returns second transform."""
        a = BoneTransform(translation=Vec3(1, 2, 3))
        b = BoneTransform(translation=Vec3(10, 20, 30))

        result = a.lerp(b, 1.0)
        assert result.translation.x == 10
        assert result.translation.y == 20
        assert result.translation.z == 30


# =============================================================================
# Pose Tests
# =============================================================================

class TestPose:
    """Tests for Pose class."""

    @pytest.fixture
    def simple_skeleton(self):
        """Create a simple 3-bone skeleton for testing."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="mid", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="tip", parent_index=1))
        skeleton._rebuild_caches()
        return skeleton

    def test_pose_creation_default(self, simple_skeleton):
        """Test pose creation with default identity transforms."""
        pose = Pose(simple_skeleton)

        assert pose.bone_count == 3
        assert pose.space == PoseSpace.LOCAL

    def test_pose_creation_with_transforms(self, simple_skeleton):
        """Test pose creation with explicit transforms."""
        transforms = [
            BoneTransform(translation=Vec3(1, 0, 0)),
            BoneTransform(translation=Vec3(0, 1, 0)),
            BoneTransform(translation=Vec3(0, 0, 1)),
        ]
        pose = Pose(simple_skeleton, bone_transforms=transforms)

        assert pose.get_bone_transform(0).translation.x == 1
        assert pose.get_bone_transform(1).translation.y == 1
        assert pose.get_bone_transform(2).translation.z == 1

    def test_pose_creation_wrong_transform_count(self, simple_skeleton):
        """Test that wrong transform count raises error."""
        transforms = [BoneTransform()]  # Only 1, need 3

        with pytest.raises(ValueError, match="must match"):
            Pose(simple_skeleton, bone_transforms=transforms)

    def test_pose_get_set_bone_transform(self, simple_skeleton):
        """Test getting and setting bone transforms."""
        pose = Pose(simple_skeleton)

        new_transform = BoneTransform(translation=Vec3(5, 5, 5))
        pose.set_bone_transform(1, new_transform)

        result = pose.get_bone_transform(1)
        assert result.translation.x == 5
        assert result.translation.y == 5
        assert result.translation.z == 5

    def test_pose_get_bone_transform_invalid_index(self, simple_skeleton):
        """Test getting bone transform with invalid index."""
        pose = Pose(simple_skeleton)

        with pytest.raises(IndexError):
            pose.get_bone_transform(99)

    def test_pose_get_transform_by_name(self, simple_skeleton):
        """Test getting transform by bone name."""
        pose = Pose(simple_skeleton)
        pose.set_bone_transform(1, BoneTransform(translation=Vec3(7, 8, 9)))

        result = pose.get_bone_transform_by_name("mid")
        assert result is not None
        assert result.translation.x == 7

    def test_pose_get_transform_by_name_not_found(self, simple_skeleton):
        """Test getting transform by non-existent name."""
        pose = Pose(simple_skeleton)

        assert pose.get_bone_transform_by_name("missing") is None

    def test_pose_set_transform_by_name(self, simple_skeleton):
        """Test setting transform by bone name."""
        pose = Pose(simple_skeleton)

        success = pose.set_bone_transform_by_name("tip", BoneTransform(
            translation=Vec3(10, 11, 12)
        ))

        assert success is True
        assert pose.get_bone_transform(2).translation.x == 10

    def test_pose_reset_to_identity(self, simple_skeleton):
        """Test resetting pose to identity."""
        pose = Pose(simple_skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(100, 100, 100)))

        pose.reset_to_identity()

        for i in range(3):
            t = pose.get_bone_transform(i)
            assert t.translation.x == 0
            assert t.translation.y == 0
            assert t.translation.z == 0

    def test_pose_copy(self, simple_skeleton):
        """Test deep copying a pose."""
        pose = Pose(simple_skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(1, 2, 3)))

        copied = pose.copy()

        # Modify original
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(999, 999, 999)))

        # Copy should be unchanged
        assert copied.get_bone_transform(0).translation.x == 1


class TestPoseSpaceConversion:
    """Tests for pose space conversions."""

    @pytest.fixture
    def simple_skeleton(self):
        """Create skeleton with translations for space conversion tests."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(
            index=0, name="root",
            local_bind_pose=Transform(translation=Vec3(0, 0, 0))
        ))
        skeleton.add_bone(Bone(
            index=1, name="child", parent_index=0,
            local_bind_pose=Transform(translation=Vec3(1, 0, 0))
        ))
        skeleton._rebuild_caches()
        return skeleton

    def test_local_to_local_returns_copy(self, simple_skeleton):
        """Test converting local to local returns copy."""
        pose = Pose(simple_skeleton, space=PoseSpace.LOCAL)
        result = pose.to_local_space()

        assert result.space == PoseSpace.LOCAL
        assert result is not pose

    def test_model_to_model_returns_copy(self, simple_skeleton):
        """Test converting model to model returns copy."""
        pose = Pose(simple_skeleton, space=PoseSpace.MODEL)
        result = pose.to_model_space()

        assert result.space == PoseSpace.MODEL
        assert result is not pose

    def test_local_to_model_space(self, simple_skeleton):
        """Test converting local space to model space."""
        transforms = [
            BoneTransform(translation=Vec3(0, 0, 0)),  # root
            BoneTransform(translation=Vec3(2, 0, 0)),  # child at local (2,0,0)
        ]
        pose = Pose(simple_skeleton, PoseSpace.LOCAL, transforms)

        model_pose = pose.to_model_space()

        # Child world pos should be root(0,0,0) + local(2,0,0) = (2,0,0)
        child_t = model_pose.get_bone_transform(1)
        assert abs(child_t.translation.x - 2.0) < 1e-6


class TestLerpPoses:
    """Tests for pose linear interpolation."""

    @pytest.fixture
    def simple_skeleton(self):
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0))
        skeleton._rebuild_caches()
        return skeleton

    def test_lerp_poses_at_zero(self, simple_skeleton):
        """Test lerp at alpha=0 returns first pose."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(1, 1, 1)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 10, 10)))

        result = lerp_poses(pose_a, pose_b, 0.0)
        assert result.get_bone_transform(0).translation.x == 1

    def test_lerp_poses_at_one(self, simple_skeleton):
        """Test lerp at alpha=1 returns second pose."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(1, 1, 1)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 10, 10)))

        result = lerp_poses(pose_a, pose_b, 1.0)
        assert result.get_bone_transform(0).translation.x == 10

    def test_lerp_poses_midpoint(self, simple_skeleton):
        """Test lerp at alpha=0.5."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 10, 10)))

        result = lerp_poses(pose_a, pose_b, 0.5)
        assert abs(result.get_bone_transform(0).translation.x - 5.0) < 1e-6

    def test_lerp_poses_different_skeletons_fails(self):
        """Test lerping poses with different skeletons raises error."""
        skel1 = Skeleton(name="skel1")
        skel1.add_bone(Bone(index=0, name="root"))

        skel2 = Skeleton(name="skel2")
        skel2.add_bone(Bone(index=0, name="root"))

        pose_a = Pose(skel1)
        pose_b = Pose(skel2)

        with pytest.raises(ValueError, match="different skeletons"):
            lerp_poses(pose_a, pose_b, 0.5)

    def test_lerp_poses_different_spaces_fails(self, simple_skeleton):
        """Test lerping poses with different spaces raises error."""
        pose_a = Pose(simple_skeleton, space=PoseSpace.LOCAL)
        pose_b = Pose(simple_skeleton, space=PoseSpace.MODEL)

        with pytest.raises(ValueError, match="different spaces"):
            lerp_poses(pose_a, pose_b, 0.5)

    def test_lerp_poses_clamps_alpha(self, simple_skeleton):
        """Test that alpha is clamped to [0, 1]."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 10, 10)))

        # Alpha > 1 should clamp to 1
        result = lerp_poses(pose_a, pose_b, 2.0)
        assert result.get_bone_transform(0).translation.x == 10

        # Alpha < 0 should clamp to 0
        result = lerp_poses(pose_a, pose_b, -1.0)
        assert result.get_bone_transform(0).translation.x == 0


class TestAdditiveBlend:
    """Tests for additive pose blending."""

    @pytest.fixture
    def simple_skeleton(self):
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton._rebuild_caches()
        return skeleton

    def test_additive_blend_translation(self, simple_skeleton):
        """Test additive blending adds translation."""
        base = Pose(simple_skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(1, 0, 0)))

        additive = Pose(simple_skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))

        result = additive_blend(base, additive, weight=1.0)

        # Base + additive * 1.0 = 1 + 5 = 6
        assert abs(result.get_bone_transform(0).translation.x - 6.0) < 1e-6

    def test_additive_blend_weight_zero(self, simple_skeleton):
        """Test additive blend with zero weight returns base."""
        base = Pose(simple_skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(1, 0, 0)))

        additive = Pose(simple_skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(100, 0, 0)))

        result = additive_blend(base, additive, weight=0.0)

        assert result.get_bone_transform(0).translation.x == 1

    def test_additive_blend_scale(self, simple_skeleton):
        """Test additive blending handles scale correctly."""
        base = Pose(simple_skeleton)
        base.set_bone_transform(0, BoneTransform(scale=Vec3(1, 1, 1)))

        additive = Pose(simple_skeleton)
        # Scale of 2 in additive means multiply base by 2
        additive.set_bone_transform(0, BoneTransform(scale=Vec3(2, 2, 2)))

        result = additive_blend(base, additive, weight=1.0)

        # Scale: base * (1 + (additive - 1) * weight) = 1 * (1 + 1) = 2
        assert abs(result.get_bone_transform(0).scale.x - 2.0) < 1e-6


class TestComputeAdditivePose:
    """Tests for computing additive (delta) poses."""

    @pytest.fixture
    def simple_skeleton(self):
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton._rebuild_caches()
        return skeleton

    def test_compute_additive_translation(self, simple_skeleton):
        """Test computing translation delta."""
        reference = Pose(simple_skeleton)
        reference.set_bone_transform(0, BoneTransform(translation=Vec3(1, 0, 0)))

        target = Pose(simple_skeleton)
        target.set_bone_transform(0, BoneTransform(translation=Vec3(6, 0, 0)))

        additive = compute_additive_pose(reference, target)

        # Delta = target - reference = 6 - 1 = 5
        assert abs(additive.get_bone_transform(0).translation.x - 5.0) < 1e-6

    def test_compute_additive_scale_with_epsilon(self, simple_skeleton):
        """Test scale delta handles near-zero reference gracefully."""
        reference = Pose(simple_skeleton)
        reference.set_bone_transform(0, BoneTransform(scale=Vec3(SCALE_EPSILON / 2, 1, 1)))

        target = Pose(simple_skeleton)
        target.set_bone_transform(0, BoneTransform(scale=Vec3(2, 2, 2)))

        additive = compute_additive_pose(reference, target)

        # Near-zero scale in reference should fallback to 1.0
        assert additive.get_bone_transform(0).scale.x == 1.0


class TestBlendMultiplePoses:
    """Tests for blending multiple poses."""

    @pytest.fixture
    def simple_skeleton(self):
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton._rebuild_caches()
        return skeleton

    def test_blend_single_pose(self, simple_skeleton):
        """Test blending single pose returns copy."""
        pose = Pose(simple_skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(5, 5, 5)))

        result = blend_multiple_poses([pose], [1.0])

        assert result.get_bone_transform(0).translation.x == 5

    def test_blend_two_equal_poses(self, simple_skeleton):
        """Test blending two poses with equal weights."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        result = blend_multiple_poses([pose_a, pose_b], [0.5, 0.5])

        assert abs(result.get_bone_transform(0).translation.x - 5.0) < 1e-6

    def test_blend_three_poses(self, simple_skeleton):
        """Test blending three poses."""
        poses = [
            Pose(simple_skeleton),
            Pose(simple_skeleton),
            Pose(simple_skeleton),
        ]
        poses[0].set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))
        poses[1].set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))
        poses[2].set_bone_transform(0, BoneTransform(translation=Vec3(20, 0, 0)))

        result = blend_multiple_poses(poses, [1.0, 1.0, 1.0])

        # Average: (0 + 10 + 20) / 3 = 10
        assert abs(result.get_bone_transform(0).translation.x - 10.0) < 1e-6

    def test_blend_empty_list_fails(self, simple_skeleton):
        """Test blending empty list raises error."""
        with pytest.raises(ValueError, match="empty"):
            blend_multiple_poses([], [])

    def test_blend_mismatched_counts_fails(self, simple_skeleton):
        """Test mismatched pose/weight counts raises error."""
        pose = Pose(simple_skeleton)

        with pytest.raises(ValueError, match="must match"):
            blend_multiple_poses([pose], [0.5, 0.5])

    def test_blend_all_zero_weights(self, simple_skeleton):
        """Test all zero weights returns first pose."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(1, 0, 0)))

        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        result = blend_multiple_poses([pose_a, pose_b], [0.0, 0.0])

        # Should return copy of first pose
        assert result.get_bone_transform(0).translation.x == 1


# =============================================================================
# PoseBuffer Tests
# =============================================================================

class TestPoseBuffer:
    """Tests for PoseBuffer class."""

    @pytest.fixture
    def simple_skeleton(self):
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton._rebuild_caches()
        return skeleton

    def test_buffer_creation(self, simple_skeleton):
        """Test buffer creation."""
        buffer = PoseBuffer(simple_skeleton, capacity=4)

        assert buffer.skeleton is simple_skeleton
        assert buffer.capacity == 4
        assert buffer.count == 0

    def test_buffer_push(self, simple_skeleton):
        """Test pushing poses to buffer."""
        buffer = PoseBuffer(simple_skeleton, capacity=4)
        pose = Pose(simple_skeleton)

        buffer.push(pose)

        assert buffer.count == 1

    def test_buffer_push_overflow(self, simple_skeleton):
        """Test buffer evicts oldest when over capacity."""
        buffer = PoseBuffer(simple_skeleton, capacity=2)

        for i in range(3):
            pose = Pose(simple_skeleton)
            pose.set_bone_transform(0, BoneTransform(translation=Vec3(float(i), 0, 0)))
            buffer.push(pose)

        assert buffer.count == 2
        # Oldest (0) should be evicted, 1 and 2 remain
        assert buffer.get(0).get_bone_transform(0).translation.x == 1
        assert buffer.get(1).get_bone_transform(0).translation.x == 2

    def test_buffer_get_latest(self, simple_skeleton):
        """Test getting latest pose."""
        buffer = PoseBuffer(simple_skeleton, capacity=4)

        pose1 = Pose(simple_skeleton)
        pose1.set_bone_transform(0, BoneTransform(translation=Vec3(1, 0, 0)))
        buffer.push(pose1)

        pose2 = Pose(simple_skeleton)
        pose2.set_bone_transform(0, BoneTransform(translation=Vec3(2, 0, 0)))
        buffer.push(pose2)

        latest = buffer.get_latest()
        assert latest.get_bone_transform(0).translation.x == 2

    def test_buffer_get_latest_empty(self, simple_skeleton):
        """Test getting latest from empty buffer returns None."""
        buffer = PoseBuffer(simple_skeleton)

        assert buffer.get_latest() is None

    def test_buffer_clear(self, simple_skeleton):
        """Test clearing buffer."""
        buffer = PoseBuffer(simple_skeleton, capacity=4)
        pose = Pose(simple_skeleton)
        buffer.push(pose)
        buffer.push(pose)

        buffer.clear()

        assert buffer.count == 0

    def test_buffer_wrong_skeleton_fails(self, simple_skeleton):
        """Test pushing pose with wrong skeleton raises error."""
        buffer = PoseBuffer(simple_skeleton, capacity=4)

        other_skeleton = Skeleton(name="other")
        other_skeleton.add_bone(Bone(index=0, name="root"))
        other_pose = Pose(other_skeleton)

        with pytest.raises(ValueError, match="same skeleton"):
            buffer.push(other_pose)

    def test_buffer_len(self, simple_skeleton):
        """Test buffer length."""
        buffer = PoseBuffer(simple_skeleton, capacity=4)
        pose = Pose(simple_skeleton)
        buffer.push(pose)
        buffer.push(pose)

        assert len(buffer) == 2


class TestPoseWorldMatrices:
    """Tests for pose world matrix computation."""

    def test_get_world_matrices(self):
        """Test getting world matrices from pose."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(
            index=0, name="root",
            local_bind_pose=Transform(translation=Vec3(0, 0, 0))
        ))
        skeleton.add_bone(Bone(
            index=1, name="child", parent_index=0,
            local_bind_pose=Transform(translation=Vec3(1, 0, 0))
        ))
        skeleton._rebuild_caches()

        pose = Pose(skeleton)
        matrices = pose.get_world_matrices()

        assert len(matrices) == 2

    def test_get_skinning_matrices(self):
        """Test getting skinning matrices from pose."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton._rebuild_caches()
        skeleton.compute_inverse_bind_poses()

        pose = Pose(skeleton)
        matrices = pose.get_skinning_matrices()

        assert len(matrices) == 1


class TestQuaternionSlerpEdgeCases:
    """Tests for T-SKEL-1.2: SLERP edge cases."""

    def test_slerp_near_identity(self):
        """Test SLERP with near-identity quaternions."""
        a = BoneTransform(rotation=Quat.identity())
        # Very small rotation
        small_angle = 0.001
        b = BoneTransform(rotation=Quat(0, 0, math.sin(small_angle/2), math.cos(small_angle/2)))

        result = a.lerp(b, 0.5)

        # Should not produce NaN or extreme values
        assert not math.isnan(result.rotation.x)
        assert not math.isnan(result.rotation.w)

    def test_slerp_antipodality(self):
        """Test SLERP handles antipodal quaternions (q and -q)."""
        # q and -q represent the same rotation
        q = Quat(0.5, 0.5, 0.5, 0.5).normalized()
        neg_q = Quat(-0.5, -0.5, -0.5, -0.5).normalized()

        a = BoneTransform(rotation=q)
        b = BoneTransform(rotation=neg_q)

        result = a.lerp(b, 0.5)

        # Result should be valid quaternion
        length = math.sqrt(
            result.rotation.x**2 + result.rotation.y**2 +
            result.rotation.z**2 + result.rotation.w**2
        )
        assert abs(length - 1.0) < 1e-5

    def test_slerp_same_quaternion(self):
        """Test SLERP with identical quaternions."""
        q = Quat(0.5, 0.5, 0.5, 0.5).normalized()

        a = BoneTransform(rotation=q)
        b = BoneTransform(rotation=q)

        result = a.lerp(b, 0.5)

        assert abs(result.rotation.x - q.x) < 1e-6
        assert abs(result.rotation.y - q.y) < 1e-6
        assert abs(result.rotation.z - q.z) < 1e-6
        assert abs(result.rotation.w - q.w) < 1e-6
