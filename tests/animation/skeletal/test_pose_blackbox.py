"""
Blackbox tests for Pose and BoneTransform structures.

Tests cover pose manipulation, per-bone transforms, quaternion operations,
and transform composition without knowledge of implementation details.
"""

import math
import pytest


class TestBoneTransformCreation:
    """Tests for BoneTransform object creation."""

    def test_create_identity_transform(self):
        """Identity transform should have no translation, unit rotation, unit scale."""
        from engine.animation.skeletal.pose import BoneTransform

        transform = BoneTransform.identity()

        assert transform.translation.x == pytest.approx(0.0)
        assert transform.translation.y == pytest.approx(0.0)
        assert transform.translation.z == pytest.approx(0.0)

        assert transform.scale.x == pytest.approx(1.0)
        assert transform.scale.y == pytest.approx(1.0)
        assert transform.scale.z == pytest.approx(1.0)

    def test_create_transform_with_translation(self):
        """Transform with translation should store it correctly."""
        from engine.animation.skeletal.pose import BoneTransform
        from engine.core.math import Vec3

        transform = BoneTransform(translation=Vec3(1.0, 2.0, 3.0))

        assert transform.translation.x == pytest.approx(1.0)
        assert transform.translation.y == pytest.approx(2.0)
        assert transform.translation.z == pytest.approx(3.0)

    def test_create_transform_with_scale(self):
        """Transform with scale should store it correctly."""
        from engine.animation.skeletal.pose import BoneTransform
        from engine.core.math import Vec3

        transform = BoneTransform(scale=Vec3(2.0, 2.0, 2.0))

        assert transform.scale.x == pytest.approx(2.0)
        assert transform.scale.y == pytest.approx(2.0)
        assert transform.scale.z == pytest.approx(2.0)

    def test_create_transform_with_rotation(self):
        """Transform should accept quaternion rotation."""
        from engine.animation.skeletal.pose import BoneTransform
        from engine.core.math import Quat

        # 90 degree rotation around Y axis
        angle = math.pi / 2
        qw = math.cos(angle / 2)
        qy = math.sin(angle / 2)

        transform = BoneTransform(rotation=Quat(0.0, qy, 0.0, qw))

        assert transform.rotation.w == pytest.approx(qw, abs=1e-5)
        assert transform.rotation.y == pytest.approx(qy, abs=1e-5)

    def test_transform_components_are_independent(self):
        """Translation, rotation, scale should be independent."""
        from engine.animation.skeletal.pose import BoneTransform
        from engine.core.math import Vec3, Quat

        transform = BoneTransform(
            translation=Vec3(1.0, 0.0, 0.0),
            rotation=Quat.identity(),
            scale=Vec3(2.0, 2.0, 2.0)
        )

        assert transform.translation.x == pytest.approx(1.0)
        assert transform.scale.x == pytest.approx(2.0)


class TestBoneTransformOperations:
    """Tests for BoneTransform mathematical operations."""

    def test_transform_lerp_at_zero(self):
        """Lerp at t=0 should return first transform."""
        from engine.animation.skeletal.pose import BoneTransform
        from engine.core.math import Vec3

        t1 = BoneTransform(translation=Vec3(0.0, 0.0, 0.0))
        t2 = BoneTransform(translation=Vec3(10.0, 10.0, 10.0))

        result = t1.lerp(t2, 0.0)

        assert result.translation.x == pytest.approx(0.0)
        assert result.translation.y == pytest.approx(0.0)

    def test_transform_lerp_at_one(self):
        """Lerp at t=1 should return second transform."""
        from engine.animation.skeletal.pose import BoneTransform
        from engine.core.math import Vec3

        t1 = BoneTransform(translation=Vec3(0.0, 0.0, 0.0))
        t2 = BoneTransform(translation=Vec3(10.0, 10.0, 10.0))

        result = t1.lerp(t2, 1.0)

        assert result.translation.x == pytest.approx(10.0)
        assert result.translation.y == pytest.approx(10.0)

    def test_transform_lerp_at_half(self):
        """Lerp at t=0.5 should return midpoint."""
        from engine.animation.skeletal.pose import BoneTransform
        from engine.core.math import Vec3

        t1 = BoneTransform(translation=Vec3(0.0, 0.0, 0.0))
        t2 = BoneTransform(translation=Vec3(10.0, 10.0, 10.0))

        result = t1.lerp(t2, 0.5)

        assert result.translation.x == pytest.approx(5.0)
        assert result.translation.y == pytest.approx(5.0)

    def test_transform_copy(self):
        """Copy should be independent."""
        from engine.animation.skeletal.pose import BoneTransform
        from engine.core.math import Vec3

        original = BoneTransform(translation=Vec3(5.0, 5.0, 5.0))
        copy = original.copy()

        assert copy.translation.x == pytest.approx(5.0)
        # Modifying copy should not affect original
        copy.translation = Vec3(10.0, 10.0, 10.0)
        assert original.translation.x == pytest.approx(5.0)


class TestQuaternionSlerp:
    """Tests for quaternion spherical linear interpolation."""

    def test_slerp_identity_rotations(self):
        """Slerp between identical rotations should return same rotation."""
        from engine.animation.skeletal.pose import BoneTransform
        from engine.core.math import Quat

        t1 = BoneTransform(rotation=Quat.identity())
        t2 = BoneTransform(rotation=Quat.identity())

        result = t1.lerp(t2, 0.5)

        assert result.rotation.w == pytest.approx(1.0, abs=1e-5)

    def test_slerp_90_degree_rotation(self):
        """Slerp should interpolate through 90 degree rotation correctly."""
        from engine.animation.skeletal.pose import BoneTransform
        from engine.core.math import Quat

        # Identity
        t1 = BoneTransform(rotation=Quat.identity())

        # 90 degrees around Y
        angle = math.pi / 2
        qw = math.cos(angle / 2)
        qy = math.sin(angle / 2)
        t2 = BoneTransform(rotation=Quat(0.0, qy, 0.0, qw))

        result = t1.lerp(t2, 0.5)

        # At halfway, should be 45 degrees
        expected_angle = math.pi / 4
        expected_w = math.cos(expected_angle / 2)

        assert result.rotation.w == pytest.approx(expected_w, abs=1e-3)

    def test_slerp_produces_valid_quaternion(self):
        """Slerp should produce normalized quaternion."""
        from engine.animation.skeletal.pose import BoneTransform
        from engine.core.math import Quat

        t1 = BoneTransform(rotation=Quat.identity())
        t2 = BoneTransform(rotation=Quat(0.0, 1.0, 0.0, 0.0))  # 180 around Y

        result = t1.lerp(t2, 0.5)
        rot = result.rotation

        # Should be valid quaternion (normalized)
        magnitude = math.sqrt(rot.x**2 + rot.y**2 + rot.z**2 + rot.w**2)
        assert magnitude == pytest.approx(1.0, abs=1e-5)


class TestPoseCreation:
    """Tests for Pose object creation."""

    def test_create_pose_for_skeleton(self):
        """Pose should be created for a specific skeleton."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))

        pose = Pose(skeleton)

        assert pose.bone_count == 2

    def test_pose_starts_with_identity_transforms(self):
        """New pose should have identity transforms for all bones."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose = Pose(skeleton)
        transform = pose.get_bone_transform(0)

        assert transform.translation.x == pytest.approx(0.0)
        assert transform.translation.y == pytest.approx(0.0)
        assert transform.translation.z == pytest.approx(0.0)


class TestPoseTransformAccess:
    """Tests for accessing and modifying pose transforms."""

    def test_set_and_get_bone_transform(self):
        """Should be able to set and retrieve bone transform."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose = Pose(skeleton)
        transform = BoneTransform(translation=Vec3(1.0, 2.0, 3.0))
        pose.set_bone_transform(0, transform)

        result = pose.get_bone_transform(0)
        assert result.translation.x == pytest.approx(1.0)
        assert result.translation.y == pytest.approx(2.0)
        assert result.translation.z == pytest.approx(3.0)

    def test_transforms_are_independent(self):
        """Setting one bone's transform should not affect others."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))

        pose = Pose(skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(1.0, 0.0, 0.0)))

        spine_transform = pose.get_bone_transform(1)
        assert spine_transform.translation.x == pytest.approx(0.0)

    def test_get_bone_transform_by_name(self):
        """Should get transform by bone name."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose = Pose(skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 0.0, 0.0)))

        result = pose.get_bone_transform_by_name("root")
        assert result is not None
        assert result.translation.x == pytest.approx(5.0)

    def test_set_bone_transform_by_name(self):
        """Should set transform by bone name."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose = Pose(skeleton)
        success = pose.set_bone_transform_by_name(
            "root", BoneTransform(translation=Vec3(5.0, 0.0, 0.0))
        )

        assert success
        result = pose.get_bone_transform(0)
        assert result.translation.x == pytest.approx(5.0)


class TestPoseWorldTransform:
    """Tests for world transform computation."""

    def test_get_world_matrices(self):
        """Should compute world matrices."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0))

        pose = Pose(skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))
        pose.set_bone_transform(1, BoneTransform(translation=Vec3(5.0, 0.0, 0.0)))

        world_matrices = pose.get_world_matrices()

        assert len(world_matrices) == 2


class TestPoseCopy:
    """Tests for pose copying."""

    def test_copy_pose(self):
        """Copied pose should have same transforms."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose = Pose(skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 5.0, 5.0)))

        copy = pose.copy()

        result = copy.get_bone_transform(0)
        assert result.translation.x == pytest.approx(5.0)

    def test_copy_is_independent(self):
        """Modifying copy should not affect original."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose = Pose(skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 5.0, 5.0)))

        copy = pose.copy()
        copy.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 10.0, 10.0)))

        original = pose.get_bone_transform(0)
        assert original.translation.x == pytest.approx(5.0)


class TestPoseReset:
    """Tests for resetting pose."""

    def test_reset_to_identity(self):
        """Reset should return all transforms to identity."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose = Pose(skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 5.0, 5.0)))
        pose.reset_to_identity()

        result = pose.get_bone_transform(0)
        assert result.translation.x == pytest.approx(0.0)
        assert result.translation.y == pytest.approx(0.0)
        assert result.translation.z == pytest.approx(0.0)


class TestPoseValidation:
    """Tests for pose validation and error handling."""

    def test_invalid_bone_index_raises(self):
        """Accessing invalid bone index should raise error."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose = Pose(skeleton)

        with pytest.raises(IndexError):
            pose.get_bone_transform(10)

    def test_negative_bone_index_raises(self):
        """Negative bone index should raise error."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose = Pose(skeleton)

        with pytest.raises(IndexError):
            pose.get_bone_transform(-1)


class TestLerpPoses:
    """Tests for pose interpolation."""

    def test_lerp_poses_at_zero(self):
        """Lerp at 0 should return first pose."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform, lerp_poses
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose_a = Pose(skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))

        pose_b = Pose(skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        result = lerp_poses(pose_a, pose_b, 0.0)
        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(0.0)

    def test_lerp_poses_at_one(self):
        """Lerp at 1 should return second pose."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform, lerp_poses
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose_a = Pose(skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))

        pose_b = Pose(skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        result = lerp_poses(pose_a, pose_b, 1.0)
        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(10.0)

    def test_lerp_poses_at_half(self):
        """Lerp at 0.5 should return midpoint."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform, lerp_poses
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose_a = Pose(skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))

        pose_b = Pose(skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        result = lerp_poses(pose_a, pose_b, 0.5)
        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(5.0)


class TestBlendMultiplePoses:
    """Tests for blending multiple poses."""

    def test_blend_multiple_poses(self):
        """Should blend multiple poses with weights."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform, blend_multiple_poses
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        pose1 = Pose(skeleton)
        pose1.set_bone_transform(0, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))

        pose2 = Pose(skeleton)
        pose2.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        result = blend_multiple_poses([pose1, pose2], [0.5, 0.5])
        transform = result.get_bone_transform(0)
        assert transform.translation.x == pytest.approx(5.0)

    def test_blend_empty_raises(self):
        """Blending empty list should raise error."""
        from engine.animation.skeletal.pose import blend_multiple_poses

        with pytest.raises(ValueError):
            blend_multiple_poses([], [])


class TestAdditivePose:
    """Tests for additive pose operations."""

    def test_additive_blend(self):
        """Should apply additive pose on top of base."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform, additive_blend
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        base = Pose(skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        additive = Pose(skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 0.0, 0.0)))

        result = additive_blend(base, additive, 1.0)
        transform = result.get_bone_transform(0)
        # Base 10 + additive 5 = 15
        assert transform.translation.x == pytest.approx(15.0)

    def test_additive_with_weight(self):
        """Weighted additive should scale the addition."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform, additive_blend
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        base = Pose(skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(0.0, 0.0, 0.0)))

        additive = Pose(skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(10.0, 0.0, 0.0)))

        result = additive_blend(base, additive, 0.5)
        transform = result.get_bone_transform(0)
        # 0 + 10 * 0.5 = 5
        assert transform.translation.x == pytest.approx(5.0)


class TestPoseBuffer:
    """Tests for PoseBuffer."""

    def test_pose_buffer_push_and_get(self):
        """Should push and get poses."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform, PoseBuffer
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        buffer = PoseBuffer(skeleton, capacity=4)

        pose = Pose(skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(5.0, 0.0, 0.0)))
        buffer.push(pose)

        assert buffer.count == 1
        retrieved = buffer.get(0)
        assert retrieved.get_bone_transform(0).translation.x == pytest.approx(5.0)

    def test_pose_buffer_capacity(self):
        """Buffer should evict oldest when at capacity."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform, PoseBuffer
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        buffer = PoseBuffer(skeleton, capacity=2)

        for i in range(3):
            pose = Pose(skeleton)
            pose.set_bone_transform(0, BoneTransform(translation=Vec3(float(i), 0.0, 0.0)))
            buffer.push(pose)

        assert buffer.count == 2
        # Oldest (0) should be evicted
        oldest = buffer.get(0)
        assert oldest.get_bone_transform(0).translation.x == pytest.approx(1.0)

    def test_pose_buffer_get_latest(self):
        """Should get most recent pose."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone
        from engine.animation.skeletal.pose import Pose, BoneTransform, PoseBuffer
        from engine.core.math import Vec3

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        buffer = PoseBuffer(skeleton, capacity=4)

        for i in range(3):
            pose = Pose(skeleton)
            pose.set_bone_transform(0, BoneTransform(translation=Vec3(float(i), 0.0, 0.0)))
            buffer.push(pose)

        latest = buffer.get_latest()
        assert latest.get_bone_transform(0).translation.x == pytest.approx(2.0)
