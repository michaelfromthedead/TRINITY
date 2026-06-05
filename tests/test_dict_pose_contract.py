"""Contract tests for dictionary-based Transform and Pose (T-AG-1.2).

CLEANROOM: tests the PUBLIC API contract only.
No knowledge of internal representation, data structures, or algorithms.

Contract sources:
  - PHASE_1_ARCH.md (GAPSET_14_ANIMATION) section 1.2 (Pose representation)
  - PHASE_1_TODO.md T-AN-1.2 (acceptance criteria)
  - engine/animation/graph/__init__.py (public exports)
  - engine/animation/graph/pose.py (dict-based Transform/Pose)

This module tests the dictionary-based (bone-name keyed) Transform and Pose
classes that complement the index-based classes.
"""
import math
import pytest
from engine.animation.graph import DictTransform, DictPose
from engine.animation.graph.pose import Transform, Pose, EPSILON


# ============================================================================
# Equivalence Class: Transform creation and defaults
# ============================================================================


class TestDictTransformCreation:
    """Dict-based Transform can be created with various constructor forms."""

    def test_default_identity(self):
        """Default Transform has position (0,0,0), rotation (0,0,0,1), scale (1,1,1)."""
        t = Transform()
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_identity_static(self):
        """Transform.identity() returns a transform at origin with identity rotation and unit scale."""
        t = Transform.identity()
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_positional_args(self):
        """Transform accepts position, rotation, scale as positional arguments."""
        t = Transform((1.0, 2.0, 3.0), (0.0, 0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
        assert t.position == (1.0, 2.0, 3.0)
        assert t.rotation == (0.0, 0.0, 1.0, 0.0)
        assert t.scale == (2.0, 2.0, 2.0)

    def test_from_position_factory(self):
        """Transform.from_position creates transform with only position."""
        t = Transform.from_position(1.0, 2.0, 3.0)
        assert t.position == (1.0, 2.0, 3.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_from_rotation_factory(self):
        """Transform.from_rotation creates transform with only rotation."""
        t = Transform.from_rotation(0.0, 0.707, 0.0, 0.707)
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.707, 0.0, 0.707)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_from_scale_factory(self):
        """Transform.from_scale creates transform with only scale."""
        t = Transform.from_scale(2.0, 3.0, 4.0)
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (2.0, 3.0, 4.0)

    def test_from_uniform_scale_factory(self):
        """Transform.from_uniform_scale creates transform with uniform scale."""
        t = Transform.from_uniform_scale(2.0)
        assert t.scale == (2.0, 2.0, 2.0)

    def test_copy_creates_independent_copy(self):
        """Transform.copy creates an independent copy."""
        t1 = Transform((1.0, 2.0, 3.0))
        t2 = t1.copy()
        assert t2.position == t1.position
        # Ensure they are independent (tuples are immutable so this is trivially true)
        assert t2 is not t1


# ============================================================================
# Equivalence Class: Transform.blend with SLERP
# ============================================================================


class TestDictTransformBlend:
    """Transform.blend returns an interpolated transform using SLERP for rotation."""

    def test_blend_t0_returns_self(self):
        """blend at t=0.0 is equivalent to self."""
        a = Transform((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
        b = Transform((4.0, 5.0, 6.0), (0.0, 0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
        result = a.blend(b, 0.0)
        assert result.position == pytest.approx(a.position)
        assert result.rotation == pytest.approx(a.rotation)
        assert result.scale == pytest.approx(a.scale)

    def test_blend_t1_returns_other(self):
        """blend at t=1.0 is equivalent to other."""
        a = Transform((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
        b = Transform((4.0, 5.0, 6.0), (0.0, 0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
        result = a.blend(b, 1.0)
        assert result.position == pytest.approx(b.position)
        assert result.rotation == pytest.approx(b.rotation)
        assert result.scale == pytest.approx(b.scale)

    def test_blend_midpoint_position_and_scale(self):
        """blend at t=0.5 returns midpoint of position and scale."""
        a = Transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
        b = Transform((2.0, 4.0, 6.0), (0.0, 0.0, 0.0, 1.0), (3.0, 3.0, 3.0))
        result = a.blend(b, 0.5)
        assert result.position == pytest.approx((1.0, 2.0, 3.0))
        assert result.scale == pytest.approx((2.0, 2.0, 2.0))

    def test_blend_clamps_t_below_zero(self):
        """blend clamps t values below 0 to 0."""
        a = Transform((0.0, 0.0, 0.0))
        b = Transform((10.0, 10.0, 10.0))
        result = a.blend(b, -0.5)
        assert result.position == pytest.approx(a.position)

    def test_blend_clamps_t_above_one(self):
        """blend clamps t values above 1 to 1."""
        a = Transform((0.0, 0.0, 0.0))
        b = Transform((10.0, 10.0, 10.0))
        result = a.blend(b, 1.5)
        assert result.position == pytest.approx(b.position)

    def test_slerp_takes_shorter_path(self):
        """SLERP takes the shorter path when dot product is negative."""
        # Quaternions representing opposite rotations
        q1 = (0.0, 0.0, 0.0, 1.0)  # Identity
        q2 = (0.0, 0.0, 0.0, -1.0)  # Same rotation, opposite sign

        a = Transform(rotation=q1)
        b = Transform(rotation=q2)
        result = a.blend(b, 0.5)

        # Result should be close to identity (the shorter path)
        # The w component should be positive and close to 1
        assert result.rotation[3] >= 0.0  # Should not flip sign
        length = math.sqrt(sum(x * x for x in result.rotation))
        assert length == pytest.approx(1.0, abs=0.01)

    def test_slerp_fallback_to_lerp_for_close_quaternions(self):
        """SLERP falls back to linear interpolation for nearly parallel quaternions."""
        # Very close quaternions (dot > 0.9995)
        q1 = (0.0, 0.0, 0.0, 1.0)
        q2 = (0.0001, 0.0, 0.0, 0.99999995)  # Slightly perturbed

        a = Transform(rotation=q1)
        b = Transform(rotation=q2)
        result = a.blend(b, 0.5)

        # Should not raise or produce NaN
        assert all(not math.isnan(x) for x in result.rotation)
        length = math.sqrt(sum(x * x for x in result.rotation))
        assert length == pytest.approx(1.0, abs=0.01)


# ============================================================================
# Equivalence Class: Transform.compose for hierarchical transforms
# ============================================================================


class TestDictTransformCompose:
    """Transform.compose applies hierarchical transformation."""

    def test_compose_identity_parent(self):
        """Composing with identity parent returns child unchanged."""
        parent = Transform.identity()
        child = Transform((1.0, 2.0, 3.0), (0.0, 0.707, 0.0, 0.707), (0.5, 0.5, 0.5))
        result = parent.compose(child)
        assert result.position == pytest.approx(child.position)
        assert result.rotation == pytest.approx(child.rotation)
        assert result.scale == pytest.approx(child.scale)

    def test_compose_identity_child(self):
        """Composing identity child with parent returns parent unchanged."""
        parent = Transform((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (2.0, 2.0, 2.0))
        child = Transform.identity()
        result = parent.compose(child)
        assert result.position == pytest.approx(parent.position)
        assert result.rotation == pytest.approx(parent.rotation)
        assert result.scale == pytest.approx(parent.scale)

    def test_compose_translation_only(self):
        """Composing translations adds them (considering parent scale)."""
        parent = Transform((1.0, 0.0, 0.0), scale=(2.0, 2.0, 2.0))
        child = Transform((1.0, 0.0, 0.0))
        result = parent.compose(child)
        # Child position (1,0,0) scaled by parent (2,2,2) = (2,0,0)
        # Then added to parent position (1,0,0) = (3,0,0)
        assert result.position == pytest.approx((3.0, 0.0, 0.0))

    def test_compose_scale_multiplies(self):
        """Composing scales multiplies them component-wise."""
        parent = Transform(scale=(2.0, 3.0, 4.0))
        child = Transform(scale=(0.5, 0.5, 0.5))
        result = parent.compose(child)
        assert result.scale == pytest.approx((1.0, 1.5, 2.0))


# ============================================================================
# Equivalence Class: Pose creation and bone access
# ============================================================================


class TestDictPoseCreation:
    """Dict-based Pose can be created and accessed by bone name."""

    def test_empty_pose(self):
        """Empty pose has no bones."""
        pose = Pose.empty()
        assert pose.bone_count() == 0

    def test_pose_with_bones(self):
        """Pose can be created with bone transforms dict."""
        pose = Pose(bone_transforms={
            "hip": Transform.identity(),
            "spine": Transform.from_position(0, 1, 0),
        })
        assert pose.bone_count() == 2
        assert pose.has_bone("hip")
        assert pose.has_bone("spine")
        assert not pose.has_bone("head")

    def test_identity_pose_factory(self):
        """Pose.identity creates pose with identity transforms for given bones."""
        bone_names = ["hip", "spine", "chest"]
        pose = Pose.identity(bone_names)
        assert pose.bone_count() == 3
        for name in bone_names:
            t = pose.get_transform(name)
            assert t is not None
            assert t.position == (0.0, 0.0, 0.0)
            assert t.rotation == (0.0, 0.0, 0.0, 1.0)
            assert t.scale == (1.0, 1.0, 1.0)

    def test_get_transform_returns_none_for_missing(self):
        """get_transform returns None for missing bone."""
        pose = Pose(bone_transforms={"hip": Transform.identity()})
        assert pose.get_transform("missing") is None

    def test_get_transform_or_identity(self):
        """get_transform_or_identity returns identity for missing bone."""
        pose = Pose(bone_transforms={"hip": Transform.from_position(1, 2, 3)})
        t = pose.get_transform_or_identity("missing")
        assert t.position == (0.0, 0.0, 0.0)

    def test_bone_names_returns_list(self):
        """bone_names returns list of bone names."""
        pose = Pose(bone_transforms={
            "hip": Transform.identity(),
            "spine": Transform.identity(),
        })
        names = pose.bone_names()
        assert set(names) == {"hip", "spine"}

    def test_copy_creates_independent_copy(self):
        """Pose.copy creates an independent deep copy."""
        pose1 = Pose(bone_transforms={"hip": Transform.from_position(1, 2, 3)})
        pose2 = pose1.copy()
        assert pose2.get_transform("hip").position == (1.0, 2.0, 3.0)
        assert pose2 is not pose1


# ============================================================================
# Equivalence Class: Pose.blend with missing bone handling
# ============================================================================


class TestDictPoseBlend:
    """Pose.blend handles missing bones gracefully."""

    def test_blend_t0_returns_self(self):
        """blend at t=0.0 returns self."""
        pose1 = Pose(bone_transforms={"hip": Transform.from_position(0, 0, 0)})
        pose2 = Pose(bone_transforms={"hip": Transform.from_position(10, 10, 10)})
        result = pose1.blend(pose2, 0.0)
        assert result.get_transform("hip").position == pytest.approx((0.0, 0.0, 0.0))

    def test_blend_t1_returns_other(self):
        """blend at t=1.0 returns other."""
        pose1 = Pose(bone_transforms={"hip": Transform.from_position(0, 0, 0)})
        pose2 = Pose(bone_transforms={"hip": Transform.from_position(10, 10, 10)})
        result = pose1.blend(pose2, 1.0)
        assert result.get_transform("hip").position == pytest.approx((10.0, 10.0, 10.0))

    def test_blend_midpoint(self):
        """blend at t=0.5 returns midpoint."""
        pose1 = Pose(bone_transforms={"hip": Transform.from_position(0, 0, 0)})
        pose2 = Pose(bone_transforms={"hip": Transform.from_position(10, 20, 30)})
        result = pose1.blend(pose2, 0.5)
        assert result.get_transform("hip").position == pytest.approx((5.0, 10.0, 15.0))

    def test_blend_missing_bone_in_self(self):
        """Bone only in other is preserved in result."""
        pose1 = Pose(bone_transforms={"hip": Transform.identity()})
        pose2 = Pose(bone_transforms={
            "hip": Transform.identity(),
            "chest": Transform.from_position(0, 2, 0),
        })
        result = pose1.blend(pose2, 0.5)
        assert result.has_bone("chest")
        # Bone only in other is copied (not blended with identity)
        chest = result.get_transform("chest")
        assert chest.position == pytest.approx((0.0, 2.0, 0.0))

    def test_blend_missing_bone_in_other(self):
        """Bone only in self is preserved in result."""
        pose1 = Pose(bone_transforms={
            "hip": Transform.identity(),
            "spine": Transform.from_position(0, 1, 0),
        })
        pose2 = Pose(bone_transforms={"hip": Transform.identity()})
        result = pose1.blend(pose2, 0.5)
        assert result.has_bone("spine")
        spine = result.get_transform("spine")
        assert spine.position == pytest.approx((0.0, 1.0, 0.0))

    def test_blend_includes_all_bones_from_both_poses(self):
        """Result contains union of bones from both poses."""
        pose1 = Pose(bone_transforms={
            "hip": Transform.identity(),
            "spine": Transform.identity(),
        })
        pose2 = Pose(bone_transforms={
            "hip": Transform.identity(),
            "chest": Transform.identity(),
        })
        result = pose1.blend(pose2, 0.5)
        assert set(result.bone_names()) == {"hip", "spine", "chest"}

    def test_blend_clamps_t_for_numerical_stability(self):
        """blend clamps t to [0, 1]."""
        pose1 = Pose(bone_transforms={"hip": Transform.from_position(0, 0, 0)})
        pose2 = Pose(bone_transforms={"hip": Transform.from_position(10, 10, 10)})

        # t < 0 should behave as t = 0
        result = pose1.blend(pose2, -1.0)
        assert result.get_transform("hip").position == pytest.approx((0.0, 0.0, 0.0))

        # t > 1 should behave as t = 1
        result = pose1.blend(pose2, 2.0)
        assert result.get_transform("hip").position == pytest.approx((10.0, 10.0, 10.0))


# ============================================================================
# Equivalence Class: Pose filtering and merging
# ============================================================================


class TestDictPoseFiltering:
    """Pose can be filtered and merged."""

    def test_filter_bones(self):
        """filter_bones returns pose with only specified bones."""
        pose = Pose(bone_transforms={
            "hip": Transform.identity(),
            "spine": Transform.identity(),
            "chest": Transform.identity(),
        })
        filtered = pose.filter_bones(["hip", "chest"])
        assert set(filtered.bone_names()) == {"hip", "chest"}
        assert not filtered.has_bone("spine")

    def test_filter_bones_ignores_missing(self):
        """filter_bones ignores bones that don't exist."""
        pose = Pose(bone_transforms={"hip": Transform.identity()})
        filtered = pose.filter_bones(["hip", "missing"])
        assert set(filtered.bone_names()) == {"hip"}

    def test_exclude_bones(self):
        """exclude_bones returns pose without specified bones."""
        pose = Pose(bone_transforms={
            "hip": Transform.identity(),
            "spine": Transform.identity(),
            "chest": Transform.identity(),
        })
        excluded = pose.exclude_bones(["spine"])
        assert set(excluded.bone_names()) == {"hip", "chest"}

    def test_merge_with_overwrite(self):
        """merge with overwrite=True allows other to overwrite self."""
        pose1 = Pose(bone_transforms={
            "hip": Transform.from_position(1, 0, 0),
            "spine": Transform.from_position(0, 1, 0),
        })
        pose2 = Pose(bone_transforms={
            "hip": Transform.from_position(2, 0, 0),
            "chest": Transform.from_position(0, 2, 0),
        })
        merged = pose1.merge(pose2, overwrite=True)
        assert merged.get_transform("hip").position == pytest.approx((2.0, 0.0, 0.0))
        assert merged.has_bone("spine")
        assert merged.has_bone("chest")

    def test_merge_without_overwrite(self):
        """merge with overwrite=False preserves self's bones."""
        pose1 = Pose(bone_transforms={
            "hip": Transform.from_position(1, 0, 0),
        })
        pose2 = Pose(bone_transforms={
            "hip": Transform.from_position(2, 0, 0),
            "chest": Transform.from_position(0, 2, 0),
        })
        merged = pose1.merge(pose2, overwrite=False)
        assert merged.get_transform("hip").position == pytest.approx((1.0, 0.0, 0.0))
        assert merged.has_bone("chest")


# ============================================================================
# Equivalence Class: Numerical stability
# ============================================================================


class TestNumericalStability:
    """Numerical stability checks."""

    def test_transform_is_valid_rejects_nan(self):
        """Transform.is_valid returns False for NaN values."""
        t = Transform(position=(float('nan'), 0.0, 0.0))
        assert not t.is_valid()

    def test_transform_is_valid_rejects_inf(self):
        """Transform.is_valid returns False for Inf values."""
        t = Transform(position=(float('inf'), 0.0, 0.0))
        assert not t.is_valid()

    def test_transform_is_valid_accepts_normal_values(self):
        """Transform.is_valid returns True for normal values."""
        t = Transform.identity()
        assert t.is_valid()

    def test_pose_is_valid_checks_all_transforms(self):
        """Pose.is_valid checks all bone transforms."""
        pose = Pose(bone_transforms={
            "hip": Transform.identity(),
            "spine": Transform(position=(float('nan'), 0.0, 0.0)),
        })
        assert not pose.is_valid()

    def test_pose_normalized_normalizes_all_rotations(self):
        """Pose.normalized normalizes all rotation quaternions."""
        # Non-normalized quaternion
        pose = Pose(bone_transforms={
            "hip": Transform(rotation=(0.0, 0.0, 0.0, 2.0)),
        })
        normalized = pose.normalized()
        length = math.sqrt(sum(x * x for x in normalized.get_transform("hip").rotation))
        assert length == pytest.approx(1.0)


# ============================================================================
# Equivalence Class: Dunder methods
# ============================================================================


class TestDunderMethods:
    """Pose supports standard Python container protocols."""

    def test_len(self):
        """len(pose) returns bone count."""
        pose = Pose(bone_transforms={
            "hip": Transform.identity(),
            "spine": Transform.identity(),
        })
        assert len(pose) == 2

    def test_contains(self):
        """'bone' in pose checks if bone exists."""
        pose = Pose(bone_transforms={"hip": Transform.identity()})
        assert "hip" in pose
        assert "missing" not in pose

    def test_iter(self):
        """iter(pose) iterates over (name, transform) pairs."""
        pose = Pose(bone_transforms={
            "hip": Transform.identity(),
            "spine": Transform.identity(),
        })
        items = list(pose)
        assert len(items) == 2
        names = [name for name, _ in items]
        assert set(names) == {"hip", "spine"}
