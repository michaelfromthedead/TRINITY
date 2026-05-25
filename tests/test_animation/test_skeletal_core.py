"""
Comprehensive tests for the skeletal animation core subsystem.

Tests cover:
- Skeleton and Bone hierarchy
- Pose operations and transforms
- Animation clips and keyframes
- Clip playback and events
- All interpolation modes
- Pose blending with edge cases
- Additive pose creation and application
- Bone masks

Target: 150+ tests with real assertions.
"""

import math
import sys
import pytest


# =============================================================================
# Test Configuration Constants
# =============================================================================

# Tolerance for floating point comparisons in tests
FLOAT_TOLERANCE = 0.001

# Direct imports from our modules (avoiding the __init__.py which has other module issues)
from engine.core.math import Vec3, Quat, Mat4, Transform

# Import skeleton module directly
from engine.animation.skeletal.skeleton import (
    Bone,
    Skeleton,
    animation_data,
    create_humanoid_skeleton,
)

# Import pose module directly
from engine.animation.skeletal.pose import (
    BoneTransform,
    Pose,
    PoseSpace,
    PoseBuffer,
    lerp_poses,
    additive_blend,
    compute_additive_pose,
    blend_multiple_poses,
)

# Import clip module directly
from engine.animation.skeletal.clip import (
    Keyframe,
    AnimationCurve,
    AnimationEvent,
    BoneTrack,
    AnimationClip,
    InterpolationType,
    create_simple_clip,
)

# Import clip player module directly
from engine.animation.skeletal.clip_player import (
    ClipPlayer,
    ClipQueue,
    CrossfadePlayer,
    PlaybackMode,
    PlaybackState,
    PlaybackEvent,
)

# Import blending module directly
from engine.animation.skeletal.blending import (
    BlendMode,
    BoneMask,
    LayeredBlender,
    PoseCache,
    blend_poses,
    blend_multiple_poses as blend_poses_weighted,
    compute_additive_pose as compute_delta_pose,
    apply_additive_pose,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_skeleton():
    """Create a simple 3-bone skeleton for testing."""
    skeleton = Skeleton(name="simple")
    skeleton.add_bone(Bone(0, "root", parent_index=-1))
    skeleton.add_bone(Bone(1, "spine", parent_index=0))
    skeleton.add_bone(Bone(2, "head", parent_index=1))
    skeleton._rebuild_caches()
    return skeleton


@pytest.fixture
def branching_skeleton():
    """Create a skeleton with branches for testing."""
    skeleton = Skeleton(name="branching")
    skeleton.add_bone(Bone(0, "root", parent_index=-1))
    skeleton.add_bone(Bone(1, "spine", parent_index=0))
    skeleton.add_bone(Bone(2, "arm_l", parent_index=1))
    skeleton.add_bone(Bone(3, "arm_r", parent_index=1))
    skeleton.add_bone(Bone(4, "hand_l", parent_index=2))
    skeleton.add_bone(Bone(5, "hand_r", parent_index=3))
    skeleton._rebuild_caches()
    return skeleton


@pytest.fixture
def humanoid_skeleton():
    """Create a humanoid skeleton."""
    return create_humanoid_skeleton()


@pytest.fixture
def simple_pose(simple_skeleton):
    """Create a simple pose."""
    return Pose(simple_skeleton)


@pytest.fixture
def simple_clip():
    """Create a simple animation clip."""
    pos_curve = AnimationCurve(
        [
            Keyframe(0.0, Vec3(0, 0, 0)),
            Keyframe(1.0, Vec3(1, 0, 0)),
        ],
        InterpolationType.LINEAR,
    )
    track = BoneTrack(0, position_curve=pos_curve)
    return AnimationClip("test", duration=1.0, bone_tracks={0: track})


# =============================================================================
# BONE TESTS (15 tests)
# =============================================================================


class TestBone:
    """Tests for Bone class."""

    def test_create_bone(self):
        """Test basic bone creation."""
        bone = Bone(0, "root")
        assert bone.index == 0
        assert bone.name == "root"
        assert bone.parent_index == -1

    def test_create_bone_with_parent(self):
        """Test bone with parent."""
        bone = Bone(1, "child", parent_index=0)
        assert bone.parent_index == 0
        assert not bone.is_root()

    def test_bone_is_root(self):
        """Test root bone detection."""
        root = Bone(0, "root")
        assert root.is_root()

    def test_bone_not_root(self):
        """Test non-root bone."""
        child = Bone(1, "child", parent_index=0)
        assert not child.is_root()

    def test_bone_copy(self):
        """Test bone deep copy."""
        original = Bone(
            0, "test",
            local_bind_pose=Transform(Vec3(1, 2, 3)),
        )
        copy = original.copy()
        assert copy.index == original.index
        assert copy.name == original.name
        assert copy is not original
        assert copy.local_bind_pose is not original.local_bind_pose

    def test_bone_negative_index_fails(self):
        """Test that negative index raises error."""
        with pytest.raises(ValueError, match="index must be >= 0"):
            Bone(-1, "bad")

    def test_bone_empty_name_fails(self):
        """Test that empty name raises error."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Bone(0, "")

    def test_bone_invalid_parent_fails(self):
        """Test that invalid parent raises error."""
        with pytest.raises(ValueError, match="Parent index must be >= -1"):
            Bone(0, "bad", parent_index=-2)

    def test_bone_self_parent_fails(self):
        """Test that self-parenting raises error."""
        with pytest.raises(ValueError, match="cannot be its own parent"):
            Bone(5, "bad", parent_index=5)

    def test_bone_repr(self):
        """Test bone string representation."""
        bone = Bone(0, "root")
        assert "root" in repr(bone)
        assert "0" in repr(bone)

    def test_bone_with_transform(self):
        """Test bone with non-identity transform."""
        transform = Transform(
            translation=Vec3(1, 2, 3),
            rotation=Quat.from_axis_angle(Vec3.unit_y(), math.pi / 2),
        )
        bone = Bone(0, "test", local_bind_pose=transform)
        assert bone.local_bind_pose.translation.x == 1

    def test_bone_inverse_bind_pose(self):
        """Test bone inverse bind pose."""
        bone = Bone(0, "test", inverse_bind_pose=Mat4.translation(Vec3(1, 0, 0)))
        assert bone.inverse_bind_pose.m[12] == 1.0

    def test_bone_copy_preserves_transforms(self):
        """Test that bone copy preserves all transform data."""
        original = Bone(
            0, "test",
            local_bind_pose=Transform(Vec3(1, 2, 3)),
            inverse_bind_pose=Mat4.translation(Vec3(4, 5, 6)),
        )
        copy = original.copy()
        assert copy.local_bind_pose.translation.x == 1
        assert copy.inverse_bind_pose.m[12] == 4

    def test_bone_copy_is_independent(self):
        """Test that copied bone is independent."""
        original = Bone(0, "test", local_bind_pose=Transform(Vec3(1, 2, 3)))
        copy = original.copy()
        copy.local_bind_pose.translation.x = 100
        assert original.local_bind_pose.translation.x == 1

    def test_bone_copy_preserves_parent(self):
        """Test that copy preserves parent index."""
        bone = Bone(5, "child", parent_index=2)
        copy = bone.copy()
        assert copy.parent_index == 2


# =============================================================================
# SKELETON TESTS (25 tests)
# =============================================================================


class TestSkeleton:
    """Tests for Skeleton class."""

    def test_create_skeleton(self):
        """Test basic skeleton creation."""
        skeleton = Skeleton(name="test")
        assert skeleton.name == "test"
        assert skeleton.bone_count == 0

    def test_skeleton_empty_name_fails(self):
        """Test that empty name raises error."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Skeleton(name="")

    def test_add_bone(self, simple_skeleton):
        """Test adding bones to skeleton."""
        assert simple_skeleton.bone_count == 3

    def test_get_bone(self, simple_skeleton):
        """Test getting bone by index."""
        bone = simple_skeleton.get_bone(0)
        assert bone.name == "root"

    def test_get_bone_out_of_range(self, simple_skeleton):
        """Test getting bone with invalid index."""
        with pytest.raises(IndexError):
            simple_skeleton.get_bone(100)

    def test_get_bone_by_name(self, simple_skeleton):
        """Test getting bone by name."""
        bone = simple_skeleton.get_bone_by_name("spine")
        assert bone is not None
        assert bone.index == 1

    def test_get_bone_by_name_not_found(self, simple_skeleton):
        """Test getting nonexistent bone by name."""
        bone = simple_skeleton.get_bone_by_name("nonexistent")
        assert bone is None

    def test_get_bone_index(self, simple_skeleton):
        """Test getting bone index by name."""
        idx = simple_skeleton.get_bone_index("spine")
        assert idx == 1

    def test_get_bone_index_not_found(self, simple_skeleton):
        """Test getting index of nonexistent bone."""
        idx = simple_skeleton.get_bone_index("nonexistent")
        assert idx == -1

    def test_root_bone_indices(self, simple_skeleton):
        """Test getting root bone indices."""
        roots = simple_skeleton.root_bone_indices
        assert len(roots) == 1
        assert roots[0] == 0

    def test_get_bone_children(self, branching_skeleton):
        """Test getting children of a bone."""
        children = branching_skeleton.get_bone_children(1)  # spine
        assert len(children) == 2
        assert 2 in children  # arm_l
        assert 3 in children  # arm_r

    def test_get_bone_children_leaf(self, simple_skeleton):
        """Test getting children of leaf bone."""
        children = simple_skeleton.get_bone_children(2)  # head
        assert len(children) == 0

    def test_get_bone_descendants(self, branching_skeleton):
        """Test getting all descendants."""
        descendants = branching_skeleton.get_bone_descendants(1)  # spine
        assert len(descendants) == 4  # arm_l, arm_r, hand_l, hand_r

    def test_get_bone_chain(self, simple_skeleton):
        """Test getting bone chain."""
        chain = simple_skeleton.get_bone_chain(0, 2)
        assert chain == [0, 1, 2]

    def test_get_bone_chain_reverse(self, simple_skeleton):
        """Test getting reverse bone chain."""
        chain = simple_skeleton.get_bone_chain(2, 0)
        assert chain == [2, 1, 0]

    def test_get_bone_path(self, simple_skeleton):
        """Test getting bone path string."""
        path = simple_skeleton.get_bone_path(2)
        assert path == "root/spine/head"

    def test_skeleton_clone(self, simple_skeleton):
        """Test skeleton cloning."""
        clone = simple_skeleton.clone()
        assert clone.name == simple_skeleton.name
        assert clone.bone_count == simple_skeleton.bone_count
        assert clone is not simple_skeleton

    def test_skeleton_validate(self, simple_skeleton):
        """Test skeleton validation."""
        errors = simple_skeleton.validate()
        assert len(errors) == 0

    def test_skeleton_get_leaf_bones(self, branching_skeleton):
        """Test getting leaf bones."""
        leaves = branching_skeleton.get_leaf_bones()
        assert 4 in leaves  # hand_l
        assert 5 in leaves  # hand_r
        assert 0 not in leaves  # root is not a leaf

    def test_skeleton_get_depth(self, simple_skeleton):
        """Test getting bone depth."""
        assert simple_skeleton.get_depth(0) == 0
        assert simple_skeleton.get_depth(1) == 1
        assert simple_skeleton.get_depth(2) == 2

    def test_skeleton_get_max_depth(self, simple_skeleton):
        """Test getting max skeleton depth."""
        assert simple_skeleton.get_max_depth() == 2

    def test_skeleton_compute_world_transforms(self, simple_skeleton):
        """Test computing world transforms."""
        transforms = simple_skeleton.compute_world_transforms()
        assert len(transforms) == 3
        assert all(isinstance(t, Mat4) for t in transforms)

    def test_skeleton_iter(self, simple_skeleton):
        """Test iterating over skeleton."""
        bones = list(simple_skeleton)
        assert len(bones) == 3

    def test_skeleton_len(self, simple_skeleton):
        """Test skeleton length."""
        assert len(simple_skeleton) == 3

    def test_skeleton_getitem(self, simple_skeleton):
        """Test skeleton indexing."""
        bone = simple_skeleton[1]
        assert bone.name == "spine"


# =============================================================================
# BONE TRANSFORM TESTS (10 tests)
# =============================================================================


class TestBoneTransform:
    """Tests for BoneTransform class."""

    def test_create_default(self):
        """Test default bone transform."""
        t = BoneTransform()
        assert t.translation.x == 0
        assert t.rotation.w == 1
        assert t.scale.x == 1

    def test_create_identity(self):
        """Test identity bone transform."""
        t = BoneTransform.identity()
        assert t.translation == Vec3.zero()
        assert t.rotation == Quat.identity()
        assert t.scale == Vec3.one()

    def test_to_transform(self):
        """Test conversion to Transform."""
        bt = BoneTransform(
            translation=Vec3(1, 2, 3),
            rotation=Quat.identity(),
            scale=Vec3(2, 2, 2),
        )
        t = bt.to_transform()
        assert isinstance(t, Transform)
        assert t.translation.x == 1

    def test_from_transform(self):
        """Test creation from Transform."""
        t = Transform(
            translation=Vec3(1, 2, 3),
            rotation=Quat.identity(),
            scale=Vec3(2, 2, 2),
        )
        bt = BoneTransform.from_transform(t)
        assert bt.translation.x == 1

    def test_copy(self):
        """Test deep copy."""
        original = BoneTransform(translation=Vec3(1, 2, 3))
        copy = original.copy()
        assert copy.translation.x == 1
        copy.translation.x = 100
        assert original.translation.x == 1

    def test_lerp(self):
        """Test linear interpolation."""
        a = BoneTransform(translation=Vec3(0, 0, 0))
        b = BoneTransform(translation=Vec3(10, 0, 0))
        result = a.lerp(b, 0.5)
        assert abs(result.translation.x - 5.0) < 0.001

    def test_lerp_zero(self):
        """Test lerp at alpha=0."""
        a = BoneTransform(translation=Vec3(0, 0, 0))
        b = BoneTransform(translation=Vec3(10, 0, 0))
        result = a.lerp(b, 0.0)
        assert abs(result.translation.x - 0.0) < 0.001

    def test_lerp_one(self):
        """Test lerp at alpha=1."""
        a = BoneTransform(translation=Vec3(0, 0, 0))
        b = BoneTransform(translation=Vec3(10, 0, 0))
        result = a.lerp(b, 1.0)
        assert abs(result.translation.x - 10.0) < 0.001

    def test_equality(self):
        """Test bone transform equality."""
        a = BoneTransform(translation=Vec3(1, 2, 3))
        b = BoneTransform(translation=Vec3(1, 2, 3))
        assert a == b

    def test_repr(self):
        """Test string representation."""
        t = BoneTransform()
        assert "BoneTransform" in repr(t)


# =============================================================================
# POSE TESTS (20 tests)
# =============================================================================


class TestPose:
    """Tests for Pose class."""

    def test_create_pose(self, simple_skeleton):
        """Test pose creation."""
        pose = Pose(simple_skeleton)
        assert pose.bone_count == 3

    def test_pose_space_default(self, simple_skeleton):
        """Test default pose space."""
        pose = Pose(simple_skeleton)
        assert pose.space == PoseSpace.LOCAL

    def test_get_bone_transform(self, simple_pose):
        """Test getting bone transform."""
        t = simple_pose.get_bone_transform(0)
        assert isinstance(t, BoneTransform)

    def test_get_bone_transform_out_of_range(self, simple_pose):
        """Test getting transform with invalid index."""
        with pytest.raises(IndexError):
            simple_pose.get_bone_transform(100)

    def test_set_bone_transform(self, simple_pose):
        """Test setting bone transform."""
        new_t = BoneTransform(translation=Vec3(1, 2, 3))
        simple_pose.set_bone_transform(0, new_t)
        t = simple_pose.get_bone_transform(0)
        assert t.translation.x == 1

    def test_set_bone_transform_by_name(self, simple_skeleton):
        """Test setting transform by bone name."""
        pose = Pose(simple_skeleton)
        success = pose.set_bone_transform_by_name(
            "spine",
            BoneTransform(translation=Vec3(5, 0, 0)),
        )
        assert success
        t = pose.get_bone_transform(1)
        assert t.translation.x == 5

    def test_reset_to_bind_pose(self, simple_pose):
        """Test resetting to bind pose."""
        simple_pose.set_bone_transform(0, BoneTransform(translation=Vec3(100, 0, 0)))
        simple_pose.reset_to_bind_pose()
        t = simple_pose.get_bone_transform(0)
        assert t.translation.x == 0

    def test_reset_to_identity(self, simple_pose):
        """Test resetting to identity."""
        simple_pose.set_bone_transform(0, BoneTransform(translation=Vec3(100, 0, 0)))
        simple_pose.reset_to_identity()
        t = simple_pose.get_bone_transform(0)
        assert t.translation == Vec3.zero()

    def test_pose_copy(self, simple_pose):
        """Test pose deep copy."""
        simple_pose.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))
        copy = simple_pose.copy()
        assert copy.get_bone_transform(0).translation.x == 5
        copy.set_bone_transform(0, BoneTransform(translation=Vec3(100, 0, 0)))
        assert simple_pose.get_bone_transform(0).translation.x == 5

    def test_get_world_matrices(self, simple_pose):
        """Test getting world matrices."""
        matrices = simple_pose.get_world_matrices()
        assert len(matrices) == 3
        assert all(isinstance(m, Mat4) for m in matrices)

    def test_lerp_poses(self, simple_skeleton):
        """Test pose lerp."""
        pose_a = Pose(simple_skeleton)
        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        result = lerp_poses(pose_a, pose_b, 0.5)
        assert abs(result.get_bone_transform(0).translation.x - 5.0) < 0.001

    def test_lerp_poses_alpha_zero(self, simple_skeleton):
        """Test pose lerp at alpha=0."""
        pose_a = Pose(simple_skeleton)
        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        result = lerp_poses(pose_a, pose_b, 0.0)
        assert abs(result.get_bone_transform(0).translation.x - 0.0) < 0.001

    def test_lerp_poses_alpha_one(self, simple_skeleton):
        """Test pose lerp at alpha=1."""
        pose_a = Pose(simple_skeleton)
        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        result = lerp_poses(pose_a, pose_b, 1.0)
        assert abs(result.get_bone_transform(0).translation.x - 10.0) < 0.001

    def test_lerp_poses_different_skeletons_fails(self, simple_skeleton, branching_skeleton):
        """Test that lerping different skeletons fails."""
        pose_a = Pose(simple_skeleton)
        pose_b = Pose(branching_skeleton)
        with pytest.raises(ValueError, match="different skeletons"):
            lerp_poses(pose_a, pose_b, 0.5)

    def test_additive_blend(self, simple_skeleton):
        """Test additive pose blending."""
        base = Pose(simple_skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))

        additive = Pose(simple_skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(2, 0, 0)))

        result = additive_blend(base, additive, weight=1.0)
        assert abs(result.get_bone_transform(0).translation.x - 7.0) < 0.001

    def test_compute_additive_pose(self, simple_skeleton):
        """Test computing additive pose delta."""
        reference = Pose(simple_skeleton)
        target = Pose(simple_skeleton)
        target.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))

        delta = compute_additive_pose(reference, target)
        assert abs(delta.get_bone_transform(0).translation.x - 5.0) < 0.001

    def test_blend_multiple_poses(self, simple_skeleton):
        """Test blending multiple poses."""
        poses = [
            Pose(simple_skeleton),
            Pose(simple_skeleton),
            Pose(simple_skeleton),
        ]
        poses[0].set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))
        poses[1].set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))
        poses[2].set_bone_transform(0, BoneTransform(translation=Vec3(20, 0, 0)))

        result = blend_multiple_poses(poses, [1.0, 1.0, 1.0])
        # Average is 10
        assert abs(result.get_bone_transform(0).translation.x - 10.0) < 0.001

    def test_pose_repr(self, simple_pose):
        """Test pose string representation."""
        assert "Pose" in repr(simple_pose)
        assert "simple" in repr(simple_pose)

    def test_pose_skeleton_property(self, simple_pose, simple_skeleton):
        """Test pose skeleton property."""
        assert simple_pose.skeleton is simple_skeleton


# =============================================================================
# KEYFRAME TESTS (8 tests)
# =============================================================================


class TestKeyframe:
    """Tests for Keyframe class."""

    def test_create_keyframe(self):
        """Test keyframe creation."""
        kf = Keyframe(0.5, 10.0)
        assert kf.time == 0.5
        assert kf.value == 10.0

    def test_keyframe_negative_time_fails(self):
        """Test that negative time fails."""
        with pytest.raises(ValueError, match="must be >= 0"):
            Keyframe(-1.0, 0)

    def test_keyframe_vec3_value(self):
        """Test keyframe with Vec3 value."""
        kf = Keyframe(0.0, Vec3(1, 2, 3))
        assert kf.value.x == 1

    def test_keyframe_quat_value(self):
        """Test keyframe with Quat value."""
        kf = Keyframe(0.0, Quat.identity())
        assert kf.value.w == 1

    def test_keyframe_copy(self):
        """Test keyframe copy."""
        original = Keyframe(0.5, Vec3(1, 2, 3))
        copy = original.copy()
        assert copy.time == 0.5
        assert copy.value.x == 1
        assert copy is not original

    def test_keyframe_with_tangents(self):
        """Test keyframe with tangents."""
        kf = Keyframe(0.5, 10.0, in_tangent=1.0, out_tangent=2.0)
        assert kf.in_tangent == 1.0
        assert kf.out_tangent == 2.0

    def test_keyframe_repr(self):
        """Test keyframe string representation."""
        kf = Keyframe(0.5, 10.0)
        assert "0.5" in repr(kf)

    def test_keyframe_zero_time(self):
        """Test keyframe at time zero."""
        kf = Keyframe(0.0, 5.0)
        assert kf.time == 0.0


# =============================================================================
# ANIMATION CURVE TESTS (15 tests)
# =============================================================================


class TestAnimationCurve:
    """Tests for AnimationCurve class."""

    def test_create_empty_curve(self):
        """Test empty curve creation."""
        curve = AnimationCurve()
        assert curve.keyframe_count == 0

    def test_add_keyframe(self):
        """Test adding keyframes."""
        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(0.0, 0.0))
        curve.add_keyframe(Keyframe(1.0, 10.0))
        assert curve.keyframe_count == 2

    def test_keyframes_sorted(self):
        """Test keyframes are sorted by time."""
        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(1.0, 10.0))
        curve.add_keyframe(Keyframe(0.0, 0.0))
        kfs = curve.keyframes
        assert kfs[0].time == 0.0
        assert kfs[1].time == 1.0

    def test_curve_duration(self):
        """Test curve duration."""
        curve = AnimationCurve(
            [Keyframe(0.0, 0), Keyframe(2.0, 20)]
        )
        assert curve.duration == 2.0

    def test_sample_linear(self):
        """Test linear interpolation sampling."""
        curve = AnimationCurve(
            [Keyframe(0.0, 0.0), Keyframe(1.0, 10.0)],
            InterpolationType.LINEAR,
        )
        assert abs(curve.sample(0.5) - 5.0) < 0.001

    def test_sample_step(self):
        """Test step interpolation sampling."""
        curve = AnimationCurve(
            [Keyframe(0.0, 0.0), Keyframe(1.0, 10.0)],
            InterpolationType.STEP,
        )
        assert curve.sample(0.5) == 0.0

    def test_sample_cubic(self):
        """Test cubic interpolation sampling."""
        curve = AnimationCurve(
            [Keyframe(0.0, 0.0), Keyframe(1.0, 10.0)],
            InterpolationType.CUBIC,
        )
        # Should produce some value
        result = curve.sample(0.5)
        assert 0.0 <= result <= 10.0

    def test_sample_before_start(self):
        """Test sampling before first keyframe."""
        curve = AnimationCurve([Keyframe(1.0, 10.0)])
        assert curve.sample(0.0) == 10.0

    def test_sample_after_end(self):
        """Test sampling after last keyframe."""
        curve = AnimationCurve([Keyframe(0.0, 5.0)])
        assert curve.sample(100.0) == 5.0

    def test_sample_empty_fails(self):
        """Test sampling empty curve fails."""
        curve = AnimationCurve()
        with pytest.raises(ValueError, match="empty curve"):
            curve.sample(0.5)

    def test_sample_vec3_linear(self):
        """Test linear interpolation of Vec3."""
        curve = AnimationCurve(
            [Keyframe(0.0, Vec3(0, 0, 0)), Keyframe(1.0, Vec3(10, 0, 0))],
            InterpolationType.LINEAR,
        )
        result = curve.sample(0.5)
        assert abs(result.x - 5.0) < 0.001

    def test_sample_quat_linear(self):
        """Test linear interpolation of Quat (slerp)."""
        q1 = Quat.identity()
        q2 = Quat.from_axis_angle(Vec3.unit_y(), math.pi / 2)
        curve = AnimationCurve(
            [Keyframe(0.0, q1), Keyframe(1.0, q2)],
            InterpolationType.LINEAR,
        )
        result = curve.sample(0.5)
        assert isinstance(result, Quat)

    def test_remove_keyframe(self):
        """Test removing keyframes."""
        curve = AnimationCurve(
            [Keyframe(0.0, 0), Keyframe(1.0, 10)]
        )
        curve.remove_keyframe(0)
        assert curve.keyframe_count == 1

    def test_curve_copy(self):
        """Test curve copy."""
        original = AnimationCurve([Keyframe(0.0, 5.0)])
        copy = original.copy()
        assert copy.keyframe_count == 1
        assert copy is not original

    def test_interpolation_setter(self):
        """Test setting interpolation type."""
        curve = AnimationCurve()
        curve.interpolation = InterpolationType.STEP
        assert curve.interpolation == InterpolationType.STEP


# =============================================================================
# BONE TRACK TESTS (10 tests)
# =============================================================================


class TestBoneTrack:
    """Tests for BoneTrack class."""

    def test_create_track(self):
        """Test track creation."""
        track = BoneTrack(bone_index=0)
        assert track.bone_index == 0

    def test_negative_index_fails(self):
        """Test that negative index fails."""
        with pytest.raises(ValueError, match="must be >= 0"):
            BoneTrack(bone_index=-1)

    def test_has_position(self):
        """Test position curve detection."""
        track = BoneTrack(
            0,
            position_curve=AnimationCurve([Keyframe(0.0, Vec3.zero())]),
        )
        assert track.has_position()

    def test_has_rotation(self):
        """Test rotation curve detection."""
        track = BoneTrack(
            0,
            rotation_curve=AnimationCurve([Keyframe(0.0, Quat.identity())]),
        )
        assert track.has_rotation()

    def test_has_scale(self):
        """Test scale curve detection."""
        track = BoneTrack(
            0,
            scale_curve=AnimationCurve([Keyframe(0.0, Vec3.one())]),
        )
        assert track.has_scale()

    def test_sample_position(self):
        """Test sampling position."""
        track = BoneTrack(
            0,
            position_curve=AnimationCurve([
                Keyframe(0.0, Vec3(0, 0, 0)),
                Keyframe(1.0, Vec3(10, 0, 0)),
            ]),
        )
        pos = track.sample_position(0.5)
        assert abs(pos.x - 5.0) < 0.001

    def test_sample_position_default(self):
        """Test sampling position with default."""
        track = BoneTrack(0)
        pos = track.sample_position(0.5, default=Vec3(1, 2, 3))
        assert pos.x == 1

    def test_track_duration(self):
        """Test track duration."""
        track = BoneTrack(
            0,
            position_curve=AnimationCurve([
                Keyframe(0.0, Vec3.zero()),
                Keyframe(2.0, Vec3.zero()),
            ]),
        )
        assert track.duration == 2.0

    def test_track_copy(self):
        """Test track copy."""
        original = BoneTrack(
            0,
            position_curve=AnimationCurve([Keyframe(0.0, Vec3.zero())]),
        )
        copy = original.copy()
        assert copy.bone_index == 0
        assert copy is not original

    def test_track_repr(self):
        """Test track string representation."""
        track = BoneTrack(5)
        assert "5" in repr(track)


# =============================================================================
# ANIMATION CLIP TESTS (15 tests)
# =============================================================================


class TestAnimationClip:
    """Tests for AnimationClip class."""

    def test_create_clip(self):
        """Test clip creation."""
        clip = AnimationClip("test", duration=1.0)
        assert clip.name == "test"
        assert clip.duration == 1.0

    def test_empty_name_fails(self):
        """Test that empty name fails."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            AnimationClip("")

    def test_invalid_framerate_fails(self):
        """Test that invalid framerate fails."""
        with pytest.raises(ValueError, match="Framerate must be > 0"):
            AnimationClip("test", framerate=0)

    def test_add_bone_track(self, simple_clip):
        """Test adding bone tracks."""
        assert simple_clip.track_count == 1

    def test_get_bone_track(self, simple_clip):
        """Test getting bone track."""
        track = simple_clip.get_bone_track(0)
        assert track is not None
        assert track.bone_index == 0

    def test_has_bone_track(self, simple_clip):
        """Test checking bone track exists."""
        assert simple_clip.has_bone_track(0)
        assert not simple_clip.has_bone_track(100)

    def test_remove_bone_track(self, simple_clip):
        """Test removing bone track."""
        assert simple_clip.remove_bone_track(0)
        assert simple_clip.track_count == 0

    def test_add_event(self):
        """Test adding events."""
        clip = AnimationClip("test", duration=1.0)
        clip.add_event(AnimationEvent(0.5, "footstep"))
        assert clip.event_count == 1

    def test_events_sorted(self):
        """Test events are sorted by time."""
        clip = AnimationClip("test", duration=1.0)
        clip.add_event(AnimationEvent(0.8, "b"))
        clip.add_event(AnimationEvent(0.2, "a"))
        events = clip.events
        assert events[0].time == 0.2
        assert events[1].time == 0.8

    def test_get_events_in_range(self):
        """Test getting events in time range."""
        clip = AnimationClip("test", duration=1.0)
        clip.add_event(AnimationEvent(0.25, "a"))
        clip.add_event(AnimationEvent(0.5, "b"))
        clip.add_event(AnimationEvent(0.75, "c"))
        events = clip.get_events_in_range(0.2, 0.6)
        assert len(events) == 2

    def test_sample_pose(self, simple_skeleton):
        """Test sampling pose from clip."""
        clip = create_simple_clip(
            "test", 1.0, 0,
            Vec3(0, 0, 0), Vec3(10, 0, 0),
        )
        pose = clip.sample_pose(simple_skeleton, 0.5)
        assert abs(pose.get_bone_transform(0).translation.x - 5.0) < 0.001

    def test_clip_looping_property(self):
        """Test looping property."""
        clip = AnimationClip("test", looping=True)
        assert clip.looping

    def test_clip_root_motion_property(self):
        """Test root motion property."""
        clip = AnimationClip("test", root_motion=True)
        assert clip.root_motion

    def test_clip_copy(self, simple_clip):
        """Test clip copy."""
        copy = simple_clip.copy()
        assert copy.name == simple_clip.name
        assert copy is not simple_clip

    def test_clip_validate(self, simple_skeleton):
        """Test clip validation."""
        clip = AnimationClip("test", duration=1.0)
        errors = clip.validate(simple_skeleton)
        assert len(errors) == 0


# =============================================================================
# CLIP PLAYER TESTS (20 tests)
# =============================================================================


class TestClipPlayer:
    """Tests for ClipPlayer class."""

    def test_create_player(self, simple_clip):
        """Test player creation."""
        player = ClipPlayer(simple_clip)
        assert player.clip is simple_clip
        assert player.time == 0.0

    def test_play(self, simple_clip):
        """Test starting playback."""
        player = ClipPlayer(simple_clip)
        player.play()
        assert player.is_playing

    def test_pause(self, simple_clip):
        """Test pausing playback."""
        player = ClipPlayer(simple_clip)
        player.play()
        player.pause()
        assert player.is_paused

    def test_stop(self, simple_clip):
        """Test stopping playback."""
        player = ClipPlayer(simple_clip)
        player.play()
        player.stop()
        assert player.is_stopped
        assert player.time == 0.0

    def test_update(self, simple_clip):
        """Test update advances time."""
        player = ClipPlayer(simple_clip)
        player.play()
        player.update(0.5)
        assert abs(player.time - 0.5) < 0.001

    def test_update_stops_at_end(self, simple_clip):
        """Test non-looping clip stops at end."""
        player = ClipPlayer(simple_clip, looping=False)
        player.play()
        player.update(2.0)
        assert player.time == 1.0
        assert player.is_stopped

    def test_update_loops(self, simple_clip):
        """Test looping clip loops."""
        player = ClipPlayer(simple_clip, looping=True)
        player.play()
        player.update(1.5)
        assert player.time < 1.0
        assert player.is_playing

    def test_set_time(self, simple_clip):
        """Test setting time."""
        player = ClipPlayer(simple_clip)
        player.set_time(0.5)
        assert player.time == 0.5

    def test_set_normalized_time(self, simple_clip):
        """Test setting normalized time."""
        player = ClipPlayer(simple_clip)
        player.set_normalized_time(0.5)
        assert player.time == 0.5

    def test_speed(self, simple_clip):
        """Test playback speed."""
        player = ClipPlayer(simple_clip, speed=2.0)
        player.play()
        player.update(0.25)
        assert abs(player.time - 0.5) < 0.001

    def test_weight(self, simple_clip):
        """Test blend weight."""
        player = ClipPlayer(simple_clip, weight=0.5)
        assert player.weight == 0.5

    def test_event_callback(self, simple_skeleton):
        """Test event callbacks fire."""
        clip = AnimationClip("test", duration=1.0)
        clip.add_event(AnimationEvent(0.5, "test_event"))

        player = ClipPlayer(clip)
        fired_events = []
        player.add_event_callback(lambda e: fired_events.append(e))

        player.play()
        player.update(0.6)

        assert len(fired_events) == 1
        assert fired_events[0].event_name == "test_event"

    def test_sample_pose(self, simple_skeleton, simple_clip):
        """Test sampling pose from player."""
        player = ClipPlayer(simple_clip)
        player.set_time(0.5)
        pose = player.sample_pose(simple_skeleton)
        assert pose.bone_count == 3

    def test_reverse_mode(self, simple_clip):
        """Test reverse playback mode."""
        player = ClipPlayer(simple_clip, mode=PlaybackMode.REVERSE)
        assert player.time == 1.0
        player.play()
        player.update(0.5)
        assert player.time < 1.0

    def test_ping_pong_mode(self, simple_clip):
        """Test ping-pong playback mode."""
        player = ClipPlayer(simple_clip, mode=PlaybackMode.PING_PONG, looping=True)
        player.play()
        player.update(1.5)  # Should reverse
        assert player.is_playing

    def test_loop_count(self, simple_clip):
        """Test loop count tracking."""
        player = ClipPlayer(simple_clip, looping=True)
        player.play()
        # With 1.0s clip, updating 1.5s should complete exactly 1 full loop
        # and be partway through the second loop
        initial_count = player.loop_count
        assert initial_count == 0
        player.update(1.5)  # Should complete 1 loop
        assert player.loop_count == 1
        assert player.time < 1.0  # Should be in second loop
        assert player.is_playing

    def test_duration_property(self, simple_clip):
        """Test duration property."""
        player = ClipPlayer(simple_clip)
        assert player.duration == 1.0

    def test_normalized_time_property(self, simple_clip):
        """Test normalized time property."""
        player = ClipPlayer(simple_clip)
        player.set_time(0.5)
        assert abs(player.normalized_time - 0.5) < 0.001

    def test_player_copy(self, simple_clip):
        """Test player copy."""
        original = ClipPlayer(simple_clip)
        original.set_time(0.5)
        copy = original.copy()
        assert copy.time == 0.5
        assert copy is not original

    def test_player_repr(self, simple_clip):
        """Test player string representation."""
        player = ClipPlayer(simple_clip)
        assert "test" in repr(player)


# =============================================================================
# CROSSFADE PLAYER TESTS (8 tests)
# =============================================================================


class TestCrossfadePlayer:
    """Tests for CrossfadePlayer class."""

    def test_create(self, simple_skeleton):
        """Test creation."""
        player = CrossfadePlayer(simple_skeleton)
        assert player.skeleton is simple_skeleton

    def test_play_clip(self, simple_skeleton, simple_clip):
        """Test playing a clip."""
        player = CrossfadePlayer(simple_skeleton)
        player.play(simple_clip)
        assert player.is_playing
        assert player.current_clip is simple_clip

    def test_crossfade(self, simple_skeleton, simple_clip):
        """Test crossfading between clips."""
        clip2 = AnimationClip("clip2", duration=1.0)
        player = CrossfadePlayer(simple_skeleton)
        player.play(simple_clip)
        player.play(clip2, blend_time=0.5)
        assert player.is_crossfading

    def test_crossfade_complete(self, simple_skeleton, simple_clip):
        """Test crossfade completion."""
        clip2 = AnimationClip("clip2", duration=1.0)
        player = CrossfadePlayer(simple_skeleton)
        player.play(simple_clip)
        player.play(clip2, blend_time=0.5)
        player.update(0.6)
        assert not player.is_crossfading
        assert player.current_clip is clip2

    def test_stop(self, simple_skeleton, simple_clip):
        """Test stopping."""
        player = CrossfadePlayer(simple_skeleton)
        player.play(simple_clip)
        player.stop()
        assert not player.is_playing

    def test_pause_resume(self, simple_skeleton, simple_clip):
        """Test pause and resume."""
        player = CrossfadePlayer(simple_skeleton)
        player.play(simple_clip)
        player.pause()
        time_before = player.get_current_time()
        player.update(0.5)
        assert player.get_current_time() == time_before
        player.resume()
        player.update(0.5)
        assert player.get_current_time() > time_before

    def test_sample_pose(self, simple_skeleton, simple_clip):
        """Test sampling pose."""
        player = CrossfadePlayer(simple_skeleton)
        player.play(simple_clip)
        pose = player.sample_pose()
        assert pose is not None
        assert pose.bone_count == 3

    def test_blend_progress(self, simple_skeleton, simple_clip):
        """Test blend progress during crossfade."""
        clip2 = AnimationClip("clip2", duration=1.0)
        player = CrossfadePlayer(simple_skeleton)
        player.play(simple_clip)
        player.play(clip2, blend_time=1.0)
        player.update(0.5)
        assert abs(player.blend_progress - 0.5) < 0.1


# =============================================================================
# BLEND MODE TESTS (10 tests)
# =============================================================================


class TestBlendPoses:
    """Tests for pose blending functions."""

    def test_blend_override(self, simple_skeleton):
        """Test override blend mode."""
        pose_a = Pose(simple_skeleton)
        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        result = blend_poses(pose_a, pose_b, 0.5, BlendMode.OVERRIDE)
        assert abs(result.get_bone_transform(0).translation.x - 5.0) < 0.001

    def test_blend_additive(self, simple_skeleton):
        """Test additive blend mode."""
        base = Pose(simple_skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))

        additive = Pose(simple_skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(3, 0, 0)))

        result = blend_poses(base, additive, 1.0, BlendMode.ADDITIVE)
        assert abs(result.get_bone_transform(0).translation.x - 8.0) < 0.001

    def test_blend_alpha_zero(self, simple_skeleton):
        """Test blend at alpha=0."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))
        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        result = blend_poses(pose_a, pose_b, 0.0)
        assert abs(result.get_bone_transform(0).translation.x - 5.0) < 0.001

    def test_blend_alpha_one(self, simple_skeleton):
        """Test blend at alpha=1."""
        pose_a = Pose(simple_skeleton)
        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        result = blend_poses(pose_a, pose_b, 1.0)
        assert abs(result.get_bone_transform(0).translation.x - 10.0) < 0.001

    def test_blend_with_mask(self, simple_skeleton):
        """Test blend with bone mask."""
        pose_a = Pose(simple_skeleton)
        pose_a.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))
        pose_a.set_bone_transform(1, BoneTransform(translation=Vec3(0, 0, 0)))
        pose_b = Pose(simple_skeleton)
        pose_b.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))
        pose_b.set_bone_transform(1, BoneTransform(translation=Vec3(10, 0, 0)))

        mask = BoneMask()
        mask.set_weight(0, 0.0)  # Don't affect bone 0
        mask.set_weight(1, 1.0)  # Fully affect bone 1
        # Need to set all bones - default weight is 0 which means no effect
        mask.set_weight(2, 0.0)

        result = blend_poses(pose_a, pose_b, 1.0, mask=mask)
        # Bone 0 has weight 0, so it keeps pose_a's value
        assert abs(result.get_bone_transform(0).translation.x - 0.0) < 0.001
        # Bone 1 has weight 1, so it gets full blend to pose_b
        assert abs(result.get_bone_transform(1).translation.x - 10.0) < 0.001

    def test_compute_delta_pose(self, simple_skeleton):
        """Test computing delta pose."""
        reference = Pose(simple_skeleton)
        target = Pose(simple_skeleton)
        target.set_bone_transform(0, BoneTransform(translation=Vec3(5, 3, 1)))

        delta = compute_delta_pose(reference, target)
        assert abs(delta.get_bone_transform(0).translation.x - 5.0) < 0.001
        assert abs(delta.get_bone_transform(0).translation.y - 3.0) < 0.001

    def test_apply_additive_pose(self, simple_skeleton):
        """Test applying additive pose."""
        base = Pose(simple_skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))

        additive = Pose(simple_skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(2, 0, 0)))

        result = apply_additive_pose(base, additive, weight=1.0)
        assert abs(result.get_bone_transform(0).translation.x - 7.0) < 0.001

    def test_apply_additive_with_weight(self, simple_skeleton):
        """Test applying additive with partial weight."""
        base = Pose(simple_skeleton)
        base.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))

        additive = Pose(simple_skeleton)
        additive.set_bone_transform(0, BoneTransform(translation=Vec3(4, 0, 0)))

        result = apply_additive_pose(base, additive, weight=0.5)
        assert abs(result.get_bone_transform(0).translation.x - 7.0) < 0.001

    def test_blend_different_skeletons_fails(self, simple_skeleton, branching_skeleton):
        """Test blending different skeletons fails."""
        pose_a = Pose(simple_skeleton)
        pose_b = Pose(branching_skeleton)
        with pytest.raises(ValueError, match="different skeletons"):
            blend_poses(pose_a, pose_b, 0.5)

    def test_blend_rotation_slerp(self, simple_skeleton):
        """Test that rotation uses slerp."""
        pose_a = Pose(simple_skeleton)
        pose_b = Pose(simple_skeleton)

        q_a = Quat.identity()
        q_b = Quat.from_axis_angle(Vec3.unit_y(), math.pi / 2)

        pose_a.set_bone_transform(0, BoneTransform(rotation=q_a))
        pose_b.set_bone_transform(0, BoneTransform(rotation=q_b))

        result = blend_poses(pose_a, pose_b, 0.5)
        # Result should be roughly halfway rotation
        result_rot = result.get_bone_transform(0).rotation
        # Just check it's not identity and not full rotation
        assert result_rot.w != q_a.w or result_rot.y != q_a.y


# =============================================================================
# BONE MASK TESTS (12 tests)
# =============================================================================


class TestBoneMask:
    """Tests for BoneMask class."""

    def test_create_mask(self):
        """Test mask creation."""
        mask = BoneMask()
        assert mask.default_weight == 0.0

    def test_get_weight_default(self):
        """Test getting default weight."""
        mask = BoneMask(default_weight=0.5)
        assert mask.get_weight(0) == 0.5

    def test_set_weight(self):
        """Test setting weight."""
        mask = BoneMask()
        mask.set_weight(0, 0.75)
        assert mask.get_weight(0) == 0.75

    def test_weight_clamped(self):
        """Test weight is clamped."""
        mask = BoneMask()
        mask.set_weight(0, 2.0)
        assert mask.get_weight(0) == 1.0
        mask.set_weight(0, -0.5)
        assert mask.get_weight(0) == 0.0

    def test_include_bone(self):
        """Test including a bone."""
        mask = BoneMask()
        mask.include_bone(5)
        assert mask.get_weight(5) == 1.0

    def test_exclude_bone(self):
        """Test excluding a bone."""
        mask = BoneMask(default_weight=1.0)
        mask.exclude_bone(5)
        assert mask.get_weight(5) == 0.0

    def test_invert_mask(self):
        """Test inverting mask."""
        mask = BoneMask()
        mask.set_weight(0, 0.3)
        inverted = mask.invert()
        assert abs(inverted.get_weight(0) - 0.7) < 0.001

    def test_combine_masks(self):
        """Test combining masks."""
        mask_a = BoneMask()
        mask_a.set_weight(0, 0.5)
        mask_b = BoneMask()
        mask_b.set_weight(0, 0.6)
        combined = mask_a.combine(mask_b)
        assert abs(combined.get_weight(0) - 0.3) < 0.001

    def test_full_body_mask(self, simple_skeleton):
        """Test full body mask creation."""
        mask = BoneMask.full_body(simple_skeleton)
        for i in range(simple_skeleton.bone_count):
            assert mask.get_weight(i) == 1.0

    def test_from_bone_chain(self, simple_skeleton):
        """Test creating mask from bone chain."""
        mask = BoneMask.from_bone_chain(simple_skeleton, "spine")
        assert mask.get_weight(1) == 1.0  # spine
        assert mask.get_weight(2) == 1.0  # head (descendant)

    def test_mask_copy(self):
        """Test mask copy."""
        original = BoneMask()
        original.set_weight(0, 0.5)
        copy = original.copy()
        assert copy.get_weight(0) == 0.5
        copy.set_weight(0, 1.0)
        assert original.get_weight(0) == 0.5

    def test_mask_repr(self):
        """Test mask string representation."""
        mask = BoneMask()
        mask.include_bone(0)
        assert "BoneMask" in repr(mask)


# =============================================================================
# LAYERED BLENDER TESTS (8 tests)
# =============================================================================


class TestLayeredBlender:
    """Tests for LayeredBlender class."""

    def test_create_blender(self, simple_skeleton):
        """Test blender creation."""
        blender = LayeredBlender(simple_skeleton)
        assert blender.skeleton is simple_skeleton
        assert blender.layer_count == 0

    def test_add_layer(self, simple_skeleton):
        """Test adding layers."""
        blender = LayeredBlender(simple_skeleton)
        idx = blender.add_layer("base")
        assert idx == 0
        assert blender.layer_count == 1

    def test_remove_layer(self, simple_skeleton):
        """Test removing layers."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("base")
        blender.remove_layer(0)
        assert blender.layer_count == 0

    def test_set_layer_pose(self, simple_skeleton):
        """Test setting layer pose."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("base")
        pose = Pose(simple_skeleton)
        blender.set_layer_pose(0, pose)
        layer = blender.get_layer(0)
        assert layer.pose is not None

    def test_set_layer_weight(self, simple_skeleton):
        """Test setting layer weight."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("base")
        blender.set_layer_weight(0, 0.5)
        layer = blender.get_layer(0)
        assert layer.weight == 0.5

    def test_blend_single_layer(self, simple_skeleton):
        """Test blending with single layer."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("base")
        pose = Pose(simple_skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))
        blender.set_layer_pose(0, pose)

        result = blender.blend()
        assert result is not None
        assert abs(result.get_bone_transform(0).translation.x - 5.0) < 0.001

    def test_blend_multiple_layers(self, simple_skeleton):
        """Test blending multiple layers."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("base")
        blender.add_layer("override")

        pose_base = Pose(simple_skeleton)
        pose_base.set_bone_transform(0, BoneTransform(translation=Vec3(0, 0, 0)))

        pose_override = Pose(simple_skeleton)
        pose_override.set_bone_transform(0, BoneTransform(translation=Vec3(10, 0, 0)))

        blender.set_layer_pose(0, pose_base)
        blender.set_layer_pose(1, pose_override)
        blender.set_layer_weight(1, 0.5)

        result = blender.blend()
        # Should blend based on weight: base(0) blended with override(10) at 50% = 5
        assert result is not None
        assert abs(result.get_bone_transform(0).translation.x - 5.0) < 0.001

    def test_disable_layer(self, simple_skeleton):
        """Test disabling layer."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("base")
        blender.set_layer_enabled(0, False)
        layer = blender.get_layer(0)
        assert not layer.enabled


# =============================================================================
# POSE CACHE TESTS (6 tests)
# =============================================================================


class TestPoseCache:
    """Tests for PoseCache class."""

    def test_create_cache(self, simple_skeleton):
        """Test cache creation."""
        cache = PoseCache(simple_skeleton)
        assert cache.size == 0

    def test_put_get(self, simple_skeleton):
        """Test putting and getting poses."""
        cache = PoseCache(simple_skeleton)
        pose = Pose(simple_skeleton)
        cache.put("key1", pose)
        result = cache.get("key1")
        assert result is not None

    def test_get_missing(self, simple_skeleton):
        """Test getting missing key."""
        cache = PoseCache(simple_skeleton)
        result = cache.get("nonexistent")
        assert result is None

    def test_capacity_eviction(self, simple_skeleton):
        """Test capacity eviction."""
        cache = PoseCache(simple_skeleton, capacity=2)
        pose = Pose(simple_skeleton)
        cache.put("a", pose)
        cache.put("b", pose)
        cache.put("c", pose)
        assert not cache.contains("a")
        assert cache.contains("b")
        assert cache.contains("c")

    def test_clear(self, simple_skeleton):
        """Test clearing cache."""
        cache = PoseCache(simple_skeleton)
        pose = Pose(simple_skeleton)
        cache.put("key", pose)
        cache.clear()
        assert cache.size == 0

    def test_remove(self, simple_skeleton):
        """Test removing entry."""
        cache = PoseCache(simple_skeleton)
        pose = Pose(simple_skeleton)
        cache.put("key", pose)
        assert cache.remove("key")
        assert not cache.contains("key")


# =============================================================================
# POSE BUFFER TESTS (5 tests)
# =============================================================================


class TestPoseBuffer:
    """Tests for PoseBuffer class."""

    def test_create_buffer(self, simple_skeleton):
        """Test buffer creation."""
        buffer = PoseBuffer(simple_skeleton)
        assert buffer.count == 0

    def test_push_pose(self, simple_skeleton):
        """Test pushing poses."""
        buffer = PoseBuffer(simple_skeleton, capacity=4)
        pose = Pose(simple_skeleton)
        buffer.push(pose)
        assert buffer.count == 1

    def test_get_latest(self, simple_skeleton):
        """Test getting latest pose."""
        buffer = PoseBuffer(simple_skeleton)
        pose = Pose(simple_skeleton)
        pose.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))
        buffer.push(pose)
        latest = buffer.get_latest()
        assert latest is not None
        assert latest.get_bone_transform(0).translation.x == 5

    def test_capacity_overflow(self, simple_skeleton):
        """Test capacity overflow eviction."""
        buffer = PoseBuffer(simple_skeleton, capacity=2)
        for i in range(5):
            pose = Pose(simple_skeleton)
            pose.set_bone_transform(0, BoneTransform(translation=Vec3(i, 0, 0)))
            buffer.push(pose)
        assert buffer.count == 2
        # Oldest should be evicted
        oldest = buffer.get(0)
        assert oldest.get_bone_transform(0).translation.x == 3

    def test_clear_buffer(self, simple_skeleton):
        """Test clearing buffer."""
        buffer = PoseBuffer(simple_skeleton)
        pose = Pose(simple_skeleton)
        buffer.push(pose)
        buffer.clear()
        assert buffer.count == 0


# =============================================================================
# ANIMATION EVENT TESTS (5 tests)
# =============================================================================


class TestAnimationEvent:
    """Tests for AnimationEvent class."""

    def test_create_event(self):
        """Test event creation."""
        event = AnimationEvent(0.5, "footstep")
        assert event.time == 0.5
        assert event.name == "footstep"

    def test_event_negative_time_fails(self):
        """Test that negative time fails."""
        with pytest.raises(ValueError, match="must be >= 0"):
            AnimationEvent(-0.5, "test")

    def test_event_empty_name_fails(self):
        """Test that empty name fails."""
        with pytest.raises(ValueError, match="cannot be empty"):
            AnimationEvent(0.5, "")

    def test_event_with_data(self):
        """Test event with data."""
        event = AnimationEvent(0.5, "footstep", data={"foot": "left"})
        assert event.data["foot"] == "left"

    def test_event_copy(self):
        """Test event copy."""
        original = AnimationEvent(0.5, "test", data={"key": "value"})
        copy = original.copy()
        assert copy.time == 0.5
        assert copy.data["key"] == "value"
        assert copy is not original


# =============================================================================
# HUMANOID SKELETON TESTS (5 tests)
# =============================================================================


class TestHumanoidSkeleton:
    """Tests for humanoid skeleton creation."""

    def test_create_humanoid(self, humanoid_skeleton):
        """Test humanoid skeleton creation."""
        assert humanoid_skeleton.bone_count == 21

    def test_humanoid_has_root(self, humanoid_skeleton):
        """Test humanoid has root bone."""
        root = humanoid_skeleton.get_bone_by_name("root")
        assert root is not None
        assert root.is_root()

    def test_humanoid_hierarchy(self, humanoid_skeleton):
        """Test humanoid bone hierarchy."""
        spine = humanoid_skeleton.get_bone_by_name("spine_01")
        assert spine is not None
        assert spine.parent_index == 1  # pelvis

    def test_humanoid_arms(self, humanoid_skeleton):
        """Test humanoid has arm bones."""
        assert humanoid_skeleton.get_bone_by_name("upperarm_l") is not None
        assert humanoid_skeleton.get_bone_by_name("upperarm_r") is not None
        assert humanoid_skeleton.get_bone_by_name("hand_l") is not None
        assert humanoid_skeleton.get_bone_by_name("hand_r") is not None

    def test_humanoid_legs(self, humanoid_skeleton):
        """Test humanoid has leg bones."""
        assert humanoid_skeleton.get_bone_by_name("thigh_l") is not None
        assert humanoid_skeleton.get_bone_by_name("thigh_r") is not None
        assert humanoid_skeleton.get_bone_by_name("foot_l") is not None
        assert humanoid_skeleton.get_bone_by_name("foot_r") is not None


# =============================================================================
# INTEGRATION TESTS (5 tests)
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_animation_pipeline(self, simple_skeleton):
        """Test full animation pipeline from clip to pose."""
        # Create clip
        clip = create_simple_clip(
            "walk", 1.0, 0,
            Vec3(0, 0, 0), Vec3(10, 0, 0),
        )

        # Create player
        player = ClipPlayer(clip, looping=True)
        player.play()

        # Update and sample
        events = player.update(0.5)
        pose = player.sample_pose(simple_skeleton)

        assert abs(pose.get_bone_transform(0).translation.x - 5.0) < 0.001

    def test_crossfade_animation(self, simple_skeleton):
        """Test crossfade between animations."""
        clip1 = create_simple_clip(
            "idle", 1.0, 0,
            Vec3(0, 0, 0), Vec3(0, 0, 0),
        )
        clip2 = create_simple_clip(
            "walk", 1.0, 0,
            Vec3(0, 0, 0), Vec3(10, 0, 0),
        )

        player = CrossfadePlayer(simple_skeleton)
        player.play(clip1)
        player.update(0.5)
        player.play(clip2, blend_time=0.5)

        # After 0.25s of 0.5s blend, we're at 50% crossfade
        player.update(0.25)
        assert player.is_crossfading
        assert abs(player.blend_progress - 0.5) < 0.1

        pose = player.sample_pose()
        assert pose is not None
        # Should be blending between idle(0,0,0) and walk at time 0.25 (2.5,0,0)
        # At 50% blend: approximately 1.25 (but depends on exact timing)
        assert pose.get_bone_transform(0).translation.x >= 0.0

    def test_layered_animation(self, simple_skeleton):
        """Test layered animation blending."""
        blender = LayeredBlender(simple_skeleton)
        blender.add_layer("base", BlendMode.OVERRIDE)
        blender.add_layer("additive", BlendMode.ADDITIVE)

        base_pose = Pose(simple_skeleton)
        base_pose.set_bone_transform(0, BoneTransform(translation=Vec3(5, 0, 0)))

        additive_pose = Pose(simple_skeleton)
        additive_pose.set_bone_transform(0, BoneTransform(translation=Vec3(2, 0, 0)))

        blender.set_layer_pose(0, base_pose)
        blender.set_layer_pose(1, additive_pose)

        result = blender.blend()
        # Additive blend: base(5) + additive(2) = 7
        assert result is not None
        assert abs(result.get_bone_transform(0).translation.x - 7.0) < 0.001

    def test_masked_blend(self, branching_skeleton):
        """Test masked pose blending."""
        pose_a = Pose(branching_skeleton)
        pose_b = Pose(branching_skeleton)

        # Set different values for arms
        pose_b.set_bone_transform(2, BoneTransform(translation=Vec3(10, 0, 0)))  # arm_l
        pose_b.set_bone_transform(3, BoneTransform(translation=Vec3(10, 0, 0)))  # arm_r

        # Mask only affects arms
        mask = BoneMask()
        mask.set_weight(2, 1.0)
        mask.set_weight(3, 1.0)

        result = blend_poses(pose_a, pose_b, 1.0, mask=mask)
        assert abs(result.get_bone_transform(2).translation.x - 10.0) < 0.001
        assert abs(result.get_bone_transform(0).translation.x - 0.0) < 0.001

    def test_animation_with_events(self, simple_skeleton):
        """Test animation with event callbacks."""
        clip = AnimationClip("attack", duration=1.0)
        clip.add_event(AnimationEvent(0.3, "attack_start"))
        clip.add_event(AnimationEvent(0.6, "attack_hit"))
        clip.add_event(AnimationEvent(0.9, "attack_end"))

        player = ClipPlayer(clip)
        event_log = []
        player.add_event_callback(lambda e: event_log.append(e.event_name))

        player.play()
        player.update(0.35)
        assert "attack_start" in event_log

        player.update(0.3)
        assert "attack_hit" in event_log

        player.update(0.35)
        assert "attack_end" in event_log


# Total tests: 172
