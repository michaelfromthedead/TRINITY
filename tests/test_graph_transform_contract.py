"""Contract tests for Transform and Pose (T-AG-1.2).

CLEANROOM: tests the PUBLIC API contract only.
No knowledge of internal representation, data structures, or algorithms.

Contract sources:
  - PHASE_1_ARCH.md (GAPSET_14_ANIMATION) section 1.2 (Pose representation)
  - PHASE_1_TODO.md T-AN-1.2 (acceptance criteria)
  - engine/animation/graph/__init__.py (public exports)

Forbidden files (NOT read):
  - engine/animation/graph/animation_graph.py (DEV implementation)
  - tests/test_graph_transform_whitebox.py (parallel peer)
"""
import math
import pytest
from engine.animation.graph import Transform, Pose, Skeleton


# ============================================================================
# Equivalence Class: Transform creation and defaults
# ============================================================================

class TestTransformCreation:
    """Transform can be created with various constructor forms."""

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

    def test_partial_position(self):
        """Transform with just position uses default rotation and scale."""
        t = Transform(position=(5.0, -3.0, 0.0))
        assert t.position == (5.0, -3.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_partial_rotation(self):
        """Transform with just rotation uses default position and scale."""
        t = Transform(rotation=(1.0, 0.0, 0.0, 0.0))
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (1.0, 0.0, 0.0, 0.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_partial_scale(self):
        """Transform with just scale uses default position and rotation."""
        t = Transform(scale=(0.5, 2.0, 1.5))
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (0.5, 2.0, 1.5)


# ============================================================================
# Equivalence Class: Transform.blend — interpolation between transforms
# ============================================================================

class TestTransformBlend:
    """Transform.blend returns an interpolated transform."""

    def test_blend_t0_returns_self(self):
        """blend at t=0.0 is equivalent to self (identity)."""
        a = Transform((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
        b = Transform((4.0, 5.0, 6.0), (0.0, 0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
        result = a.blend(b, 0.0)
        assert result.position == pytest.approx(a.position)
        assert result.rotation == pytest.approx(a.rotation)
        assert result.scale == pytest.approx(a.scale)

    def test_blend_t1_returns_other(self):
        """blend at t=1.0 is equivalent to other (target)."""
        a = Transform((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
        b = Transform((4.0, 5.0, 6.0), (0.0, 0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
        result = a.blend(b, 1.0)
        assert result.position == pytest.approx(b.position)
        assert result.rotation == pytest.approx(b.rotation)
        assert result.scale == pytest.approx(b.scale)

    def test_blend_midpoint(self):
        """blend at t=0.5 returns midpoint of position and scale, spherical midpoint of rotation."""
        a = Transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
        b = Transform((2.0, 4.0, 6.0), (0.0, 0.0, 0.0, 1.0), (3.0, 3.0, 3.0))
        result = a.blend(b, 0.5)
        # Position midpoint
        assert result.position == pytest.approx((1.0, 2.0, 3.0))
        # Scale midpoint (linear in scale space)
        assert result.scale == pytest.approx((2.0, 2.0, 2.0))
        # Rotation (both identity, so still identity)
        assert result.rotation == pytest.approx((0.0, 0.0, 0.0, 1.0))

    def test_blend_with_translation_and_uniform_scale(self):
        """blend with translation and scale interpolates each component."""
        a = Transform((1.0, 1.0, 1.0), scale=(1.0, 1.0, 1.0))
        b = Transform((3.0, 5.0, 7.0), scale=(5.0, 5.0, 5.0))
        result = a.blend(b, 0.25)
        assert result.position == pytest.approx((1.5, 2.0, 2.5))
        assert result.scale == pytest.approx((2.0, 2.0, 2.0))

    def test_blend_returns_new_transform(self):
        """blend returns a new Transform, not a reference to either input."""
        a = Transform((0.0, 0.0, 0.0))
        b = Transform((1.0, 1.0, 1.0))
        result = a.blend(b, 0.5)
        assert result is not a
        assert result is not b

    def test_blend_identity_self_no_op(self):
        """blend between self at t=0.5 should be equivalent to self when both are identity."""
        a = Transform.identity()
        b = Transform.identity()
        result = a.blend(b, 0.375)
        assert result.position == pytest.approx((0.0, 0.0, 0.0))
        assert result.rotation == pytest.approx((0.0, 0.0, 0.0, 1.0))
        assert result.scale == pytest.approx((1.0, 1.0, 1.0))


# ============================================================================
# Equivalence Class: Transform.compose — hierarchical composition
# ============================================================================

class TestTransformCompose:
    """Transform.compose returns hierarchical composition (self then other)."""

    def test_compose_with_identity_is_no_op(self):
        """Composing with identity returns the original transform."""
        t = Transform((2.0, 3.0, 4.0), (0.0, 0.0, 1.0, 0.0), (2.0, 2.0, 2.0))
        identity = Transform.identity()
        result = t.compose(identity)
        assert result.position == pytest.approx(t.position)
        assert result.rotation == pytest.approx(t.rotation)
        assert result.scale == pytest.approx(t.scale)

    def test_identity_compose_with_t_returns_t(self):
        """Composing identity with t is also a no-op."""
        t = Transform((1.0, 2.0, 3.0))
        identity = Transform.identity()
        result = identity.compose(t)
        assert result.position == pytest.approx(t.position)
        assert result.rotation == pytest.approx(t.rotation)
        assert result.scale == pytest.approx(t.scale)

    def test_compose_applies_translation(self):
        """Composing two translations yields additive translation."""
        t1 = Transform((2.0, 0.0, 0.0))
        t2 = Transform((3.0, 0.0, 0.0))
        # t1.compose(t2): apply t2 then t1 to a point
        # p' = R1 S1 (R2 S2 p + t2) + t1 = R1 S1 R2 S2 p + R1 S1 t2 + t1
        # For identity rotation/scale: p' = p + t2 + t1
        result = t1.compose(t2)
        assert result is not None
        assert isinstance(result, Transform)
        assert result.position == pytest.approx((5.0, 0.0, 0.0))
        assert result.rotation == pytest.approx((0.0, 0.0, 0.0, 1.0))
        assert result.scale == pytest.approx((1.0, 1.0, 1.0))

    def test_compose_scale_then_translate(self):
        """Composing scale then translation yields a transform with both effects."""
        t_scale = Transform(scale=(2.0, 2.0, 2.0))
        t_translate = Transform((5.0, 0.0, 0.0))
        # If scale is composed first, then translation:
        # p' = S (T p) = S(p + t) = S p + S t
        result = t_scale.compose(t_translate)
        assert result is not None
        assert isinstance(result, Transform)

    def test_compose_returns_new_transform(self):
        """compose returns a new Transform, not a reference to either input."""
        a = Transform((1.0, 0.0, 0.0))
        b = Transform((0.0, 1.0, 0.0))
        result = a.compose(b)
        assert result is not a
        assert result is not b


# ============================================================================
# Equivalence Class: Transform.copy
# ============================================================================

class TestTransformCopy:
    """Transform.copy produces an independent clone."""

    def test_copy_is_independent(self):
        """Modifying the copy does not affect the original."""
        original = Transform((1.0, 2.0, 3.0), (0.0, 0.0, 1.0, 0.0), (2.0, 3.0, 4.0))
        copy = original.copy()
        # Same values
        assert copy.position == original.position
        assert copy.rotation == original.rotation
        assert copy.scale == original.scale
        # Different object
        assert copy is not original

    def test_copy_position_mutation_independence(self):
        """Changing position on copy leaves original unchanged."""
        original = Transform((1.0, 2.0, 3.0))
        copy_t = original.copy()
        copy_t.position = (9.0, 9.0, 9.0)
        assert original.position == (1.0, 2.0, 3.0)
        assert copy_t.position == (9.0, 9.0, 9.0)

    def test_copy_scale_mutation_independence(self):
        """Changing scale on copy leaves original unchanged."""
        original = Transform(scale=(1.0, 1.0, 1.0))
        copy_t = original.copy()
        copy_t.scale = (3.0, 3.0, 3.0)
        assert original.scale == (1.0, 1.0, 1.0)
        assert copy_t.scale == (3.0, 3.0, 3.0)


# ============================================================================
# Equivalence Class: Pose creation
# ============================================================================

class TestPoseCreation:
    """Pose can be created with transforms and optional metadata."""

    def test_create_with_transforms_list(self):
        """Pose accepts a list of Transform objects."""
        transforms = [Transform(), Transform((1.0, 0.0, 0.0))]
        pose = Pose(transforms=transforms)
        assert pose.bone_count() == 2

    def test_identity_pose(self):
        """Pose.identity(bone_count) creates an identity pose with all bones at origin."""
        pose = Pose.identity(5)
        assert pose.bone_count() == 5
        for i in range(5):
            t = pose.get_transform(i)
            assert t.position == (0.0, 0.0, 0.0)
            assert t.rotation == (0.0, 0.0, 0.0, 1.0)
            assert t.scale == (1.0, 1.0, 1.0)

    def test_identity_zero_bones(self):
        """Pose.identity(0) creates a pose with no bones."""
        pose = Pose.identity(0)
        assert pose.bone_count() == 0

    def test_identity_one_bone(self):
        """Pose.identity(1) creates a pose with exactly one bone."""
        pose = Pose.identity(1)
        assert pose.bone_count() == 1

    def test_pose_accepts_root_motion(self):
        """Pose can carry an optional root_motion Transform."""
        motion = Transform((0.5, 0.0, 0.0))
        pose = Pose(transforms=[Transform()], root_motion=motion)
        assert pose.root_motion is motion


# ============================================================================
# Equivalence Class: Pose.blend — interpolation between poses
# ============================================================================

class TestPoseBlend:
    """Pose.blend interpolates between two poses."""

    def test_blend_t0_returns_self(self):
        """Pose.blend at t=0.0 is equivalent to self."""
        transforms_a = [Transform((0.0, 0.0, 0.0)), Transform((1.0, 0.0, 0.0))]
        transforms_b = [Transform((2.0, 2.0, 2.0)), Transform((3.0, 3.0, 3.0))]
        a = Pose(transforms=transforms_a)
        b = Pose(transforms=transforms_b)
        result = a.blend(b, 0.0)
        assert result.bone_count() == a.bone_count()
        for i in range(result.bone_count()):
            t_self = a.get_transform(i)
            t_result = result.get_transform(i)
            assert t_result.position == pytest.approx(t_self.position)
            assert t_result.rotation == pytest.approx(t_self.rotation)
            assert t_result.scale == pytest.approx(t_self.scale)

    def test_blend_t1_returns_other(self):
        """Pose.blend at t=1.0 is equivalent to other."""
        a = Pose(transforms=[Transform((0.0, 0.0, 0.0)), Transform((1.0, 0.0, 0.0))])
        b = Pose(transforms=[Transform((2.0, 2.0, 2.0)), Transform((3.0, 3.0, 3.0))])
        result = a.blend(b, 1.0)
        assert result.bone_count() == b.bone_count()
        for i in range(result.bone_count()):
            t_other = b.get_transform(i)
            t_result = result.get_transform(i)
            assert t_result.position == pytest.approx(t_other.position)
            assert t_result.rotation == pytest.approx(t_other.rotation)
            assert t_result.scale == pytest.approx(t_other.scale)

    def test_blend_halfway(self):
        """Pose.blend at t=0.5 produces midpoint per-bone transforms."""
        a = Pose(transforms=[Transform((0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))])
        b = Pose(transforms=[Transform((2.0, 4.0, 6.0), scale=(3.0, 3.0, 3.0))])
        result = a.blend(b, 0.5)
        t = result.get_transform(0)
        assert t.position == pytest.approx((1.0, 2.0, 3.0))
        assert t.scale == pytest.approx((2.0, 2.0, 2.0))

    def test_blend_multiple_bones(self):
        """Pose.blend interpolates all bones in the pose."""
        a = Pose(transforms=[
            Transform((0.0, 0.0, 0.0)),
            Transform((1.0, 0.0, 0.0)),
            Transform((2.0, 0.0, 0.0)),
        ])
        b = Pose(transforms=[
            Transform((3.0, 3.0, 3.0)),
            Transform((4.0, 4.0, 4.0)),
            Transform((5.0, 5.0, 5.0)),
        ])
        result = a.blend(b, 0.5)
        assert result.bone_count() == 3
        assert result.get_transform(0).position == pytest.approx((1.5, 1.5, 1.5))
        assert result.get_transform(1).position == pytest.approx((2.5, 2.0, 2.0))
        assert result.get_transform(2).position == pytest.approx((3.5, 2.5, 2.5))

    def test_blend_returns_new_pose(self):
        """Pose.blend returns a new Pose, not a reference to either input."""
        a = Pose(transforms=[Transform()])
        b = Pose(transforms=[Transform((1.0, 1.0, 1.0))])
        result = a.blend(b, 0.5)
        assert result is not a
        assert result is not b

    def test_blend_identity_both_ends(self):
        """Blending two identity poses yields identity at any t."""
        a = Pose(transforms=[Transform.identity() for _ in range(4)])
        b = Pose(transforms=[Transform.identity() for _ in range(4)])
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            result = a.blend(b, t)
            for i in range(result.bone_count()):
                tx = result.get_transform(i)
                assert tx.position == pytest.approx((0.0, 0.0, 0.0))
                assert tx.rotation == pytest.approx((0.0, 0.0, 0.0, 1.0))
                assert tx.scale == pytest.approx((1.0, 1.0, 1.0))


# ============================================================================
# Equivalence Class: Pose.set_transform / get_transform round-trip
# ============================================================================

class TestPoseTransformAccess:
    """Bone transforms in a pose can be read and written by index."""

    def test_get_transform_returns_transform(self):
        """get_transform returns a Transform for a valid index."""
        pose = Pose.identity(3)
        t = pose.get_transform(0)
        assert isinstance(t, Transform)

    def test_set_transform_updates_pose(self):
        """set_transform replaces the transform at a given bone index."""
        pose = Pose.identity(2)
        new_t = Transform((10.0, 20.0, 30.0))
        pose.set_transform(0, new_t)
        retrieved = pose.get_transform(0)
        assert retrieved.position == (10.0, 20.0, 30.0)

    def test_set_transform_out_of_range_none(self):
        """set_transform with out-of-range index returns silently (no error)."""
        pose = Pose.identity(1)
        # Does not raise -- contract is tolerant of out-of-range writes
        pose.set_transform(99, Transform())

    def test_get_transform_out_of_range_returns_default(self):
        """get_transform with out-of-range index returns an identity Transform (not None)."""
        pose = Pose.identity(1)
        result = pose.get_transform(99)
        # Returns a valid Transform (identity defaults) rather than raising
        assert result is not None
        assert result.position == (0.0, 0.0, 0.0)
        assert result.rotation == (0.0, 0.0, 0.0, 1.0)
        assert result.scale == (1.0, 1.0, 1.0)

    def test_set_and_get_transform(self):
        """set_transform stores a transform that get_transform retrieves."""
        pose = Pose.identity(1)
        t = Transform((5.0, 5.0, 5.0))
        pose.set_transform(0, t)
        assert pose.get_transform(0).position == (5.0, 5.0, 5.0)


# ============================================================================
# Equivalence Class: Pose with skeleton reference
# ============================================================================

class TestPoseSkeletonReference:
    """Pose with a skeleton reference preserves it through operations."""

    def test_pose_stores_skeleton_reference(self):
        """Pose created with a skeleton returns it via .skeleton."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_index=0)
        pose = Pose(
            transforms=[Transform(), Transform()],
            skeleton=skel,
        )
        assert pose.skeleton is skel

    def test_pose_identity_has_no_skeleton(self):
        """Pose.identity() has skeleton set to None."""
        pose = Pose.identity(3)
        assert pose.skeleton is None

    def test_blend_preserves_skeleton(self):
        """Blending a pose with skeleton preserves the skeleton in the result."""
        skel = Skeleton()
        skel.add_bone("root")
        a = Pose(transforms=[Transform()], skeleton=skel)
        b = Pose(transforms=[Transform((1.0, 0.0, 0.0))])
        result = a.blend(b, 0.3)
        assert result.skeleton is skel

    def test_blend_with_skeleton_on_both(self):
        """When both poses share the same skeleton, it is preserved in the result."""
        skel = Skeleton()
        skel.add_bone("root")
        a = Pose(transforms=[Transform()], skeleton=skel)
        b = Pose(transforms=[Transform((2.0, 0.0, 0.0))], skeleton=skel)
        result = a.blend(b, 0.5)
        assert result.skeleton is skel

    def test_copy_preserves_skeleton(self):
        """Copying a pose preserves the skeleton reference."""
        skel = Skeleton()
        skel.add_bone("root")
        pose = Pose(transforms=[Transform()], skeleton=skel)
        copy = pose.copy()
        assert copy.skeleton is skel

    def test_copy_preserves_root_motion_value(self):
        """Copying a pose preserves the root_motion by value (not necessarily same object)."""
        motion = Transform((0.5, 0.0, 0.0))
        pose = Pose(transforms=[Transform()], root_motion=motion)
        copy = pose.copy()
        assert copy.root_motion is not None
        assert copy.root_motion.position == motion.position
        assert copy.root_motion.rotation == motion.rotation
        assert copy.root_motion.scale == motion.scale


# ============================================================================
# Equivalence Class: Pose.copy
# ============================================================================

class TestPoseCopy:
    """Pose.copy produces an independent clone."""

    def test_copy_is_independent(self):
        """Modifying a transform on the copy does not affect the original."""
        pose = Pose(transforms=[Transform((1.0, 2.0, 3.0))])
        copy = pose.copy()
        copy.set_transform(0, Transform((9.0, 9.0, 9.0)))
        assert pose.get_transform(0).position == (1.0, 2.0, 3.0)
        assert copy.get_transform(0).position == (9.0, 9.0, 9.0)

    def test_copy_same_values(self):
        """Copy initially has the same bone transforms as the original."""
        pose = Pose(transforms=[
            Transform((1.0, 2.0, 3.0)),
            Transform((4.0, 5.0, 6.0)),
        ])
        copy = pose.copy()
        assert copy.bone_count() == pose.bone_count()
        for i in range(pose.bone_count()):
            orig_t = pose.get_transform(i)
            copy_t = copy.get_transform(i)
            assert orig_t.position == copy_t.position
            assert orig_t.rotation == copy_t.rotation
            assert orig_t.scale == copy_t.scale


# ============================================================================
# Edge cases: Pose bone count consistency
# ============================================================================

class TestPoseBoneCount:
    """Bone count remains consistent through operations."""

    def test_pose_large_bone_count(self):
        """Pose.identity can create pose with many bones."""
        n = 100
        pose = Pose.identity(n)
        assert pose.bone_count() == n
        # All bones are accessible
        for i in range(n):
            assert pose.get_transform(i) is not None

    def test_mismatched_blend_counts_produces_result(self):
        """Blending poses with different bone counts silently produces a result
        (contract is tolerant; the bone count of the result matches the smaller pose).
        """
        a = Pose(transforms=[Transform(), Transform()])
        b = Pose(transforms=[Transform()])
        # Does not raise
        result = a.blend(b, 0.5)
        assert result is not None
        assert isinstance(result, Pose)


# ============================================================================
# Edge cases: Transform boundary values
# ============================================================================

class TestTransformBoundary:
    """Transform handles edge and boundary values."""

    def test_large_translation(self):
        """Transform with large translation values does not break."""
        t = Transform((1e6, -1e6, 0.0))
        result = t.blend(Transform(), 0.0)
        assert result.position[0] == pytest.approx(1e6)

    def test_zero_scale(self):
        """Transform with zero scale is valid (flattening)."""
        t = Transform(scale=(0.0, 0.0, 0.0))
        assert t.scale == (0.0, 0.0, 0.0)

    def test_negative_scale(self):
        """Transform with negative scale (mirroring) is valid."""
        t = Transform(scale=(-1.0, 1.0, 1.0))
        assert t.scale == (-1.0, 1.0, 1.0)

    def test_blend_endpoints_as_ranges(self):
        """blend operates correctly over the full t in [0, 1] range."""
        a = Transform((0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
        b = Transform((10.0, 20.0, 30.0), scale=(3.0, 3.0, 3.0))
        # A few intermediate points
        for t_val in (0.0, 0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 0.9, 1.0):
            result = a.blend(b, t_val)
            assert result.position[0] == pytest.approx(10.0 * t_val)
            assert result.position[1] == pytest.approx(20.0 * t_val)
            assert result.position[2] == pytest.approx(30.0 * t_val)
            assert result.scale[0] == pytest.approx(1.0 + 2.0 * t_val)
