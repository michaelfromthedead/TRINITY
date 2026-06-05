"""Whitebox tests for Transform and Pose data structures.

Task: T-AG-1.2 Transform and Pose Data Structures

This module provides comprehensive whitebox testing of the pose.py module,
covering internal implementation details including:
- Transform creation and factory methods
- Transform.blend() with SLERP (including dot < 0 and threshold fallback)
- Transform.compose() for hierarchical composition
- Pose creation and bone management
- Pose.blend() with missing bones handling
- Numerical stability (epsilon checks, quaternion normalization)
- Edge cases (zero quaternions, extreme scales, empty poses)
"""

from __future__ import annotations

import math
import pytest

from engine.animation.graph.pose import (
    Transform,
    Pose,
    EPSILON,
    QUAT_NORMALIZE_EPSILON,
    Vec3,
    Quaternion,
    _slerp,
    _multiply_quaternion,
    _normalize_quaternion,
    _rotate_vector,
    _lerp_vec3,
)
from engine.animation.graph.config import get_config


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def identity_transform() -> Transform:
    """Provide an identity transform."""
    return Transform.identity()


@pytest.fixture
def sample_transform() -> Transform:
    """Provide a non-trivial transform."""
    return Transform(
        position=(1.0, 2.0, 3.0),
        rotation=(0.0, 0.707, 0.0, 0.707),  # 90 deg around Y
        scale=(2.0, 2.0, 2.0),
    )


@pytest.fixture
def sample_pose() -> Pose:
    """Provide a sample pose with multiple bones."""
    return Pose(
        bone_transforms={
            "hip": Transform.identity(),
            "spine": Transform.from_position(0.0, 1.0, 0.0),
            "head": Transform.from_position(0.0, 2.0, 0.0),
        }
    )


# =============================================================================
# TRANSFORM CREATION AND FACTORIES
# =============================================================================


class TestTransformCreation:
    """Tests for Transform creation and factory methods."""

    def test_default_transform_is_identity(self):
        """Default Transform should be identity."""
        t = Transform()
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_identity_factory(self):
        """Transform.identity() should return identity transform."""
        t = Transform.identity()
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_from_position_factory(self):
        """Transform.from_position() should only set position."""
        t = Transform.from_position(1.0, 2.0, 3.0)
        assert t.position == (1.0, 2.0, 3.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_from_rotation_factory(self):
        """Transform.from_rotation() should only set rotation."""
        t = Transform.from_rotation(0.0, 0.707, 0.0, 0.707)
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.707, 0.0, 0.707)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_from_scale_factory(self):
        """Transform.from_scale() should only set scale."""
        t = Transform.from_scale(2.0, 3.0, 4.0)
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (2.0, 3.0, 4.0)

    def test_from_uniform_scale_factory(self):
        """Transform.from_uniform_scale() should set uniform scale."""
        t = Transform.from_uniform_scale(5.0)
        assert t.scale == (5.0, 5.0, 5.0)

    def test_copy_creates_independent_instance(self, sample_transform: Transform):
        """copy() should create an independent instance."""
        copied = sample_transform.copy()
        assert copied.position == sample_transform.position
        assert copied.rotation == sample_transform.rotation
        assert copied.scale == sample_transform.scale
        # Verify independence (tuples are immutable, so this is mainly for API)
        assert copied is not sample_transform

    def test_transform_repr(self, sample_transform: Transform):
        """Transform __repr__ should be informative."""
        repr_str = repr(sample_transform)
        assert "Transform" in repr_str
        assert "pos=" in repr_str
        assert "rot=" in repr_str
        assert "scale=" in repr_str


# =============================================================================
# TRANSFORM BLENDING WITH SLERP
# =============================================================================


class TestTransformBlend:
    """Tests for Transform.blend() with SLERP rotation interpolation."""

    def test_blend_at_zero_returns_self(self):
        """Blend with t=0 should return a copy of self."""
        t1 = Transform.from_position(1.0, 0.0, 0.0)
        t2 = Transform.from_position(2.0, 0.0, 0.0)
        result = t1.blend(t2, 0.0)
        assert result.position == t1.position

    def test_blend_at_one_returns_other(self):
        """Blend with t=1 should return a copy of other."""
        t1 = Transform.from_position(1.0, 0.0, 0.0)
        t2 = Transform.from_position(2.0, 0.0, 0.0)
        result = t1.blend(t2, 1.0)
        assert result.position == t2.position

    def test_blend_at_half_interpolates_position(self):
        """Blend with t=0.5 should interpolate position."""
        t1 = Transform.from_position(0.0, 0.0, 0.0)
        t2 = Transform.from_position(2.0, 4.0, 6.0)
        result = t1.blend(t2, 0.5)
        assert result.position == pytest.approx((1.0, 2.0, 3.0), abs=1e-6)

    def test_blend_at_half_interpolates_scale(self):
        """Blend with t=0.5 should interpolate scale."""
        t1 = Transform.from_scale(1.0, 1.0, 1.0)
        t2 = Transform.from_scale(3.0, 5.0, 7.0)
        result = t1.blend(t2, 0.5)
        assert result.scale == pytest.approx((2.0, 3.0, 4.0), abs=1e-6)

    def test_blend_clamps_t_below_zero(self):
        """Blend should clamp t < 0 to 0."""
        t1 = Transform.from_position(1.0, 0.0, 0.0)
        t2 = Transform.from_position(2.0, 0.0, 0.0)
        result = t1.blend(t2, -1.0)
        assert result.position == t1.position

    def test_blend_clamps_t_above_one(self):
        """Blend should clamp t > 1 to 1."""
        t1 = Transform.from_position(1.0, 0.0, 0.0)
        t2 = Transform.from_position(2.0, 0.0, 0.0)
        result = t1.blend(t2, 2.0)
        assert result.position == t2.position

    def test_blend_early_exit_near_zero(self):
        """Blend should return self.copy() when t <= EPSILON."""
        t1 = Transform.from_position(1.0, 0.0, 0.0)
        t2 = Transform.from_position(2.0, 0.0, 0.0)
        result = t1.blend(t2, EPSILON * 0.5)
        assert result.position == t1.position

    def test_blend_early_exit_near_one(self):
        """Blend should return other.copy() when t >= 1 - EPSILON."""
        t1 = Transform.from_position(1.0, 0.0, 0.0)
        t2 = Transform.from_position(2.0, 0.0, 0.0)
        result = t1.blend(t2, 1.0 - EPSILON * 0.5)
        assert result.position == t2.position

    def test_lerp_is_alias_for_blend(self):
        """lerp() should be an alias for blend()."""
        t1 = Transform.from_position(0.0, 0.0, 0.0)
        t2 = Transform.from_position(2.0, 0.0, 0.0)
        blend_result = t1.blend(t2, 0.5)
        lerp_result = t1.lerp(t2, 0.5)
        assert blend_result.position == lerp_result.position


# =============================================================================
# SLERP INTERNALS (WHITEBOX)
# =============================================================================


class TestSlerpInternals:
    """Whitebox tests for _slerp quaternion interpolation."""

    def test_slerp_identity_to_identity(self):
        """SLERP between identical quaternions should return that quaternion."""
        q = (0.0, 0.0, 0.0, 1.0)
        result = _slerp(q, q, 0.5)
        assert result == pytest.approx(q, abs=1e-6)

    def test_slerp_negative_dot_takes_shorter_path(self):
        """SLERP should negate q2 when dot < 0 to take shorter path."""
        # q1 = identity, q2 = negated identity (same rotation, opposite sign)
        q1 = (0.0, 0.0, 0.0, 1.0)
        q2 = (0.0, 0.0, 0.0, -1.0)  # Negative w = same as identity but wrong side
        result = _slerp(q1, q2, 0.5)
        # After negation, q2 becomes q1, so result should equal q1
        assert result == pytest.approx(q1, abs=1e-6)

    def test_slerp_dot_above_threshold_uses_lerp(self):
        """SLERP should use linear interpolation when dot > threshold."""
        config = get_config()
        threshold = config.quaternion.SLERP_DOT_THRESHOLD

        # Create two quaternions with very high dot product (nearly parallel)
        q1 = (0.0, 0.0, 0.0, 1.0)
        # q2 is a very small rotation from identity
        small_angle = math.acos(threshold) * 0.5  # Half the threshold angle
        q2 = (math.sin(small_angle / 2), 0.0, 0.0, math.cos(small_angle / 2))

        # This should trigger the linear interpolation path
        result = _slerp(q1, q2, 0.5)
        # Result should be normalized
        length = math.sqrt(sum(x * x for x in result))
        assert length == pytest.approx(1.0, abs=1e-5)

    def test_slerp_90_degree_rotation(self):
        """SLERP should correctly interpolate a 90-degree rotation."""
        # Identity quaternion
        q1 = (0.0, 0.0, 0.0, 1.0)
        # 90 degrees around Y axis
        angle = math.pi / 2
        q2 = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))

        result = _slerp(q1, q2, 0.5)

        # At t=0.5, should be 45 degrees around Y
        expected_angle = math.pi / 4
        expected = (0.0, math.sin(expected_angle / 2), 0.0, math.cos(expected_angle / 2))
        assert result == pytest.approx(expected, abs=1e-5)

    def test_slerp_180_degree_rotation(self):
        """SLERP should handle 180-degree rotations."""
        # Identity quaternion
        q1 = (0.0, 0.0, 0.0, 1.0)
        # 180 degrees around Y axis
        angle = math.pi
        q2 = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))

        result = _slerp(q1, q2, 0.5)

        # At t=0.5, should be 90 degrees around Y
        expected_angle = math.pi / 2
        expected = (0.0, math.sin(expected_angle / 2), 0.0, math.cos(expected_angle / 2))
        assert result == pytest.approx(expected, abs=1e-5)


# =============================================================================
# TRANSFORM COMPOSE
# =============================================================================


class TestTransformCompose:
    """Tests for Transform.compose() hierarchical composition."""

    def test_compose_identity_with_identity(self):
        """Composing two identity transforms should give identity."""
        parent = Transform.identity()
        child = Transform.identity()
        result = parent.compose(child)
        assert result.position == (0.0, 0.0, 0.0)
        assert result.rotation == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-6)
        assert result.scale == (1.0, 1.0, 1.0)

    def test_compose_position_only(self):
        """Composing position-only transforms should add positions."""
        parent = Transform.from_position(1.0, 2.0, 3.0)
        child = Transform.from_position(4.0, 5.0, 6.0)
        result = parent.compose(child)
        # Child position is just added (no rotation, unit scale)
        assert result.position == pytest.approx((5.0, 7.0, 9.0), abs=1e-6)

    def test_compose_scale_affects_child_position(self):
        """Parent scale should affect child position."""
        parent = Transform.from_scale(2.0, 2.0, 2.0)
        child = Transform.from_position(1.0, 1.0, 1.0)
        result = parent.compose(child)
        # Child position is scaled by parent
        assert result.position == pytest.approx((2.0, 2.0, 2.0), abs=1e-6)

    def test_compose_scales_multiply(self):
        """Composed scales should multiply component-wise."""
        parent = Transform.from_scale(2.0, 3.0, 4.0)
        child = Transform.from_scale(5.0, 6.0, 7.0)
        result = parent.compose(child)
        assert result.scale == pytest.approx((10.0, 18.0, 28.0), abs=1e-6)

    def test_compose_rotation_affects_child_position(self):
        """Parent rotation should rotate child position."""
        # 90 degrees around Y axis
        angle = math.pi / 2
        parent = Transform.from_rotation(0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
        child = Transform.from_position(1.0, 0.0, 0.0)
        result = parent.compose(child)
        # X axis becomes -Z axis after 90 deg Y rotation
        assert result.position == pytest.approx((0.0, 0.0, -1.0), abs=1e-5)

    def test_compose_rotations_multiply(self):
        """Composed rotations should multiply (quaternion multiplication)."""
        # Two 90-degree rotations around Y should give 180 degrees
        angle = math.pi / 2
        q_90y = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
        parent = Transform.from_rotation(*q_90y)
        child = Transform.from_rotation(*q_90y)
        result = parent.compose(child)

        # Expected: 180 degrees around Y
        expected_angle = math.pi
        expected_q = (0.0, math.sin(expected_angle / 2), 0.0, math.cos(expected_angle / 2))
        assert result.rotation == pytest.approx(expected_q, abs=1e-5)

    def test_compose_full_hierarchy(self):
        """Full hierarchical composition with all components."""
        parent = Transform(
            position=(1.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            scale=(2.0, 2.0, 2.0),
        )
        child = Transform(
            position=(1.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            scale=(0.5, 0.5, 0.5),
        )
        result = parent.compose(child)
        # Position: 1 + (1 * 2) = 3
        assert result.position == pytest.approx((3.0, 0.0, 0.0), abs=1e-6)
        # Scale: 2 * 0.5 = 1
        assert result.scale == pytest.approx((1.0, 1.0, 1.0), abs=1e-6)


# =============================================================================
# TRANSFORM ADDITIVE OPERATIONS
# =============================================================================


class TestTransformAdditive:
    """Tests for Transform additive operations (__add__)."""

    def test_add_positions(self):
        """Adding transforms should add positions."""
        t1 = Transform.from_position(1.0, 2.0, 3.0)
        t2 = Transform.from_position(4.0, 5.0, 6.0)
        result = t1 + t2
        assert result.position == pytest.approx((5.0, 7.0, 9.0), abs=1e-6)

    def test_add_scales_multiply(self):
        """Adding transforms should multiply scales."""
        t1 = Transform.from_scale(2.0, 3.0, 4.0)
        t2 = Transform.from_scale(5.0, 6.0, 7.0)
        result = t1 + t2
        assert result.scale == pytest.approx((10.0, 18.0, 28.0), abs=1e-6)

    def test_add_rotations_multiply(self):
        """Adding transforms should multiply rotations (quaternion product)."""
        angle = math.pi / 2
        q = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
        t1 = Transform.from_rotation(*q)
        t2 = Transform.from_rotation(*q)
        result = t1 + t2
        # Two 90-deg Y rotations = 180-deg Y rotation
        expected_angle = math.pi
        expected_q = (0.0, math.sin(expected_angle / 2), 0.0, math.cos(expected_angle / 2))
        assert result.rotation == pytest.approx(expected_q, abs=1e-5)


# =============================================================================
# TRANSFORM VALIDATION
# =============================================================================


class TestTransformValidation:
    """Tests for Transform validation and normalization."""

    def test_is_valid_returns_true_for_valid_transform(self, sample_transform: Transform):
        """is_valid() should return True for valid transforms."""
        assert sample_transform.is_valid() is True

    def test_is_valid_returns_false_for_nan_position(self):
        """is_valid() should return False if position contains NaN."""
        t = Transform(position=(float("nan"), 0.0, 0.0))
        assert t.is_valid() is False

    def test_is_valid_returns_false_for_inf_position(self):
        """is_valid() should return False if position contains Inf."""
        t = Transform(position=(float("inf"), 0.0, 0.0))
        assert t.is_valid() is False

    def test_is_valid_returns_false_for_nan_rotation(self):
        """is_valid() should return False if rotation contains NaN."""
        t = Transform(rotation=(float("nan"), 0.0, 0.0, 1.0))
        assert t.is_valid() is False

    def test_is_valid_returns_false_for_nan_scale(self):
        """is_valid() should return False if scale contains NaN."""
        t = Transform(scale=(1.0, float("nan"), 1.0))
        assert t.is_valid() is False

    def test_normalized_returns_transform_with_unit_quaternion(self):
        """normalized() should return transform with normalized rotation."""
        # Non-unit quaternion
        t = Transform(rotation=(0.0, 2.0, 0.0, 2.0))
        result = t.normalized()
        length = math.sqrt(sum(x * x for x in result.rotation))
        assert length == pytest.approx(1.0, abs=1e-6)

    def test_normalized_preserves_position_and_scale(self):
        """normalized() should preserve position and scale."""
        t = Transform(
            position=(1.0, 2.0, 3.0),
            rotation=(0.0, 2.0, 0.0, 2.0),
            scale=(4.0, 5.0, 6.0),
        )
        result = t.normalized()
        assert result.position == t.position
        assert result.scale == t.scale


# =============================================================================
# QUATERNION UTILITIES
# =============================================================================


class TestQuaternionUtilities:
    """Tests for quaternion utility functions."""

    def test_normalize_quaternion_unit_length(self):
        """_normalize_quaternion should return unit length quaternion."""
        q = (1.0, 2.0, 3.0, 4.0)
        result = _normalize_quaternion(q)
        length = math.sqrt(sum(x * x for x in result))
        assert length == pytest.approx(1.0, abs=1e-6)

    def test_normalize_quaternion_zero_returns_identity(self):
        """_normalize_quaternion should return identity for zero quaternion."""
        q = (0.0, 0.0, 0.0, 0.0)
        result = _normalize_quaternion(q)
        assert result == (0.0, 0.0, 0.0, 1.0)

    def test_normalize_quaternion_near_zero_returns_identity(self):
        """_normalize_quaternion should return identity for near-zero quaternion."""
        tiny = QUAT_NORMALIZE_EPSILON * 0.1
        q = (tiny, tiny, tiny, tiny)
        result = _normalize_quaternion(q)
        assert result == (0.0, 0.0, 0.0, 1.0)

    def test_multiply_quaternion_identity(self):
        """Multiplying by identity should return the same quaternion."""
        identity = (0.0, 0.0, 0.0, 1.0)
        q = (0.5, 0.5, 0.5, 0.5)
        result = _multiply_quaternion(identity, q)
        assert result == pytest.approx(q, abs=1e-6)
        result2 = _multiply_quaternion(q, identity)
        assert result2 == pytest.approx(q, abs=1e-6)

    def test_multiply_quaternion_inverse(self):
        """q * q^-1 should give identity (approximately)."""
        # For a unit quaternion, inverse is conjugate
        # Use exact sqrt(2)/2 for unit quaternion
        s = math.sqrt(2) / 2  # Exact value for 90-degree rotation
        q = (0.0, s, 0.0, s)
        q_conj = (-q[0], -q[1], -q[2], q[3])
        result = _multiply_quaternion(q, q_conj)
        assert result == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-10)

    def test_rotate_vector_by_identity(self):
        """Rotating by identity should return the same vector."""
        v = (1.0, 2.0, 3.0)
        identity = (0.0, 0.0, 0.0, 1.0)
        result = _rotate_vector(v, identity)
        assert result == pytest.approx(v, abs=1e-6)

    def test_rotate_vector_90_degrees_around_y(self):
        """Rotating (1,0,0) by 90 degrees around Y should give (0,0,-1)."""
        v = (1.0, 0.0, 0.0)
        angle = math.pi / 2
        q = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
        result = _rotate_vector(v, q)
        assert result == pytest.approx((0.0, 0.0, -1.0), abs=1e-5)

    def test_rotate_vector_180_degrees_around_z(self):
        """Rotating (1,0,0) by 180 degrees around Z should give (-1,0,0)."""
        v = (1.0, 0.0, 0.0)
        angle = math.pi
        q = (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
        result = _rotate_vector(v, q)
        assert result == pytest.approx((-1.0, 0.0, 0.0), abs=1e-5)

    def test_lerp_vec3_at_zero(self):
        """_lerp_vec3 at t=0 should return first vector."""
        v1 = (1.0, 2.0, 3.0)
        v2 = (4.0, 5.0, 6.0)
        result = _lerp_vec3(v1, v2, 0.0)
        assert result == v1

    def test_lerp_vec3_at_one(self):
        """_lerp_vec3 at t=1 should return second vector."""
        v1 = (1.0, 2.0, 3.0)
        v2 = (4.0, 5.0, 6.0)
        result = _lerp_vec3(v1, v2, 1.0)
        assert result == v2

    def test_lerp_vec3_at_half(self):
        """_lerp_vec3 at t=0.5 should return midpoint."""
        v1 = (0.0, 0.0, 0.0)
        v2 = (2.0, 4.0, 6.0)
        result = _lerp_vec3(v1, v2, 0.5)
        assert result == pytest.approx((1.0, 2.0, 3.0), abs=1e-6)


# =============================================================================
# POSE CREATION AND BONE MANAGEMENT
# =============================================================================


class TestPoseCreation:
    """Tests for Pose creation and factory methods."""

    def test_default_pose_is_empty(self):
        """Default Pose should have no bone transforms."""
        p = Pose()
        assert len(p.bone_transforms) == 0

    def test_empty_factory(self):
        """Pose.empty() should return empty pose."""
        p = Pose.empty()
        assert len(p.bone_transforms) == 0

    def test_identity_factory_with_bones(self):
        """Pose.identity() should create identity transforms for given bones."""
        bones = ["hip", "spine", "head"]
        p = Pose.identity(bones)
        assert len(p.bone_transforms) == 3
        for bone in bones:
            assert bone in p.bone_transforms
            t = p.bone_transforms[bone]
            assert t.position == (0.0, 0.0, 0.0)
            assert t.rotation == (0.0, 0.0, 0.0, 1.0)
            assert t.scale == (1.0, 1.0, 1.0)

    def test_identity_factory_with_none_returns_empty(self):
        """Pose.identity(None) should return empty pose."""
        p = Pose.identity(None)
        assert len(p.bone_transforms) == 0

    def test_copy_creates_independent_pose(self, sample_pose: Pose):
        """copy() should create an independent pose."""
        copied = sample_pose.copy()
        assert copied is not sample_pose
        assert copied.bone_transforms is not sample_pose.bone_transforms
        for name in sample_pose.bone_transforms:
            assert name in copied.bone_transforms
            assert copied.bone_transforms[name] is not sample_pose.bone_transforms[name]


class TestPoseBoneAccessors:
    """Tests for Pose bone accessor methods."""

    def test_get_transform_existing_bone(self, sample_pose: Pose):
        """get_transform() should return transform for existing bone."""
        t = sample_pose.get_transform("hip")
        assert t is not None
        assert isinstance(t, Transform)

    def test_get_transform_missing_bone(self, sample_pose: Pose):
        """get_transform() should return None for missing bone."""
        t = sample_pose.get_transform("nonexistent")
        assert t is None

    def test_get_transform_or_identity_existing(self, sample_pose: Pose):
        """get_transform_or_identity() should return transform for existing bone."""
        t = sample_pose.get_transform_or_identity("spine")
        assert t.position == (0.0, 1.0, 0.0)

    def test_get_transform_or_identity_missing(self, sample_pose: Pose):
        """get_transform_or_identity() should return identity for missing bone."""
        t = sample_pose.get_transform_or_identity("nonexistent")
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_set_transform(self):
        """set_transform() should add/update bone transform."""
        p = Pose.empty()
        t = Transform.from_position(1.0, 2.0, 3.0)
        p.set_transform("new_bone", t)
        assert "new_bone" in p.bone_transforms
        assert p.bone_transforms["new_bone"].position == (1.0, 2.0, 3.0)

    def test_has_bone_existing(self, sample_pose: Pose):
        """has_bone() should return True for existing bone."""
        assert sample_pose.has_bone("hip") is True

    def test_has_bone_missing(self, sample_pose: Pose):
        """has_bone() should return False for missing bone."""
        assert sample_pose.has_bone("nonexistent") is False

    def test_bone_count(self, sample_pose: Pose):
        """bone_count() should return correct count."""
        assert sample_pose.bone_count() == 3

    def test_bone_names(self, sample_pose: Pose):
        """bone_names() should return list of bone names."""
        names = sample_pose.bone_names()
        assert set(names) == {"hip", "spine", "head"}


# =============================================================================
# POSE BLENDING
# =============================================================================


class TestPoseBlend:
    """Tests for Pose.blend() with missing bones handling."""

    def test_blend_at_zero_returns_self(self, sample_pose: Pose):
        """Blend with t=0 should return a copy of self."""
        other = Pose.empty()
        result = sample_pose.blend(other, 0.0)
        assert result.bone_count() == sample_pose.bone_count()
        for name in sample_pose.bone_names():
            assert name in result.bone_transforms

    def test_blend_at_one_returns_other(self, sample_pose: Pose):
        """Blend with t=1 should return a copy of other."""
        other = Pose.identity(["hip", "spine", "head"])
        result = sample_pose.blend(other, 1.0)
        for name in other.bone_names():
            assert result.bone_transforms[name].position == (0.0, 0.0, 0.0)

    def test_blend_interpolates_common_bones(self):
        """Blend should interpolate transforms for common bones."""
        p1 = Pose(bone_transforms={"hip": Transform.from_position(0.0, 0.0, 0.0)})
        p2 = Pose(bone_transforms={"hip": Transform.from_position(2.0, 4.0, 6.0)})
        result = p1.blend(p2, 0.5)
        assert result.bone_transforms["hip"].position == pytest.approx((1.0, 2.0, 3.0), abs=1e-6)

    def test_blend_preserves_unique_bones_from_self(self):
        """Blend should preserve bones that only exist in self."""
        p1 = Pose(
            bone_transforms={
                "hip": Transform.from_position(0.0, 0.0, 0.0),
                "unique_self": Transform.from_position(1.0, 1.0, 1.0),
            }
        )
        p2 = Pose(bone_transforms={"hip": Transform.from_position(2.0, 2.0, 2.0)})
        result = p1.blend(p2, 0.5)
        assert "unique_self" in result.bone_transforms
        assert result.bone_transforms["unique_self"].position == (1.0, 1.0, 1.0)

    def test_blend_preserves_unique_bones_from_other(self):
        """Blend should preserve bones that only exist in other."""
        p1 = Pose(bone_transforms={"hip": Transform.from_position(0.0, 0.0, 0.0)})
        p2 = Pose(
            bone_transforms={
                "hip": Transform.from_position(2.0, 2.0, 2.0),
                "unique_other": Transform.from_position(3.0, 3.0, 3.0),
            }
        )
        result = p1.blend(p2, 0.5)
        assert "unique_other" in result.bone_transforms
        assert result.bone_transforms["unique_other"].position == (3.0, 3.0, 3.0)

    def test_blend_clamps_t_below_zero(self):
        """Blend should clamp t < 0 to 0."""
        p1 = Pose(bone_transforms={"hip": Transform.from_position(1.0, 0.0, 0.0)})
        p2 = Pose(bone_transforms={"hip": Transform.from_position(2.0, 0.0, 0.0)})
        result = p1.blend(p2, -1.0)
        assert result.bone_transforms["hip"].position == (1.0, 0.0, 0.0)

    def test_blend_clamps_t_above_one(self):
        """Blend should clamp t > 1 to 1."""
        p1 = Pose(bone_transforms={"hip": Transform.from_position(1.0, 0.0, 0.0)})
        p2 = Pose(bone_transforms={"hip": Transform.from_position(2.0, 0.0, 0.0)})
        result = p1.blend(p2, 2.0)
        assert result.bone_transforms["hip"].position == (2.0, 0.0, 0.0)

    def test_blend_early_exit_near_zero(self):
        """Blend should return self.copy() when t <= EPSILON."""
        p1 = Pose(bone_transforms={"hip": Transform.from_position(1.0, 0.0, 0.0)})
        p2 = Pose(bone_transforms={"hip": Transform.from_position(2.0, 0.0, 0.0)})
        result = p1.blend(p2, EPSILON * 0.5)
        assert result.bone_transforms["hip"].position == (1.0, 0.0, 0.0)

    def test_blend_empty_poses(self):
        """Blending two empty poses should return empty pose."""
        p1 = Pose.empty()
        p2 = Pose.empty()
        result = p1.blend(p2, 0.5)
        assert len(result.bone_transforms) == 0

    def test_lerp_is_alias_for_blend(self):
        """lerp() should be an alias for blend()."""
        p1 = Pose(bone_transforms={"hip": Transform.from_position(0.0, 0.0, 0.0)})
        p2 = Pose(bone_transforms={"hip": Transform.from_position(2.0, 0.0, 0.0)})
        blend_result = p1.blend(p2, 0.5)
        lerp_result = p1.lerp(p2, 0.5)
        assert (
            blend_result.bone_transforms["hip"].position
            == lerp_result.bone_transforms["hip"].position
        )


# =============================================================================
# POSE ADDITIVE BLENDING
# =============================================================================


class TestPoseAdditiveBlend:
    """Tests for Pose.additive_blend()."""

    def test_additive_blend_weight_zero_returns_self(self):
        """Additive blend with weight=0 should return copy of self."""
        base = Pose(bone_transforms={"hip": Transform.from_position(1.0, 0.0, 0.0)})
        additive = Pose(bone_transforms={"hip": Transform.from_position(1.0, 0.0, 0.0)})
        result = base.additive_blend(additive, 0.0)
        assert result.bone_transforms["hip"].position == (1.0, 0.0, 0.0)

    def test_additive_blend_full_weight(self):
        """Additive blend with weight=1 should fully apply additive pose."""
        base = Pose(bone_transforms={"hip": Transform.from_position(1.0, 0.0, 0.0)})
        additive = Pose(bone_transforms={"hip": Transform.from_position(2.0, 0.0, 0.0)})
        result = base.additive_blend(additive, 1.0)
        # Position is added
        assert result.bone_transforms["hip"].position == pytest.approx((3.0, 0.0, 0.0), abs=1e-6)

    def test_additive_blend_partial_weight(self):
        """Additive blend with weight=0.5 should half-apply additive pose."""
        base = Pose(bone_transforms={"hip": Transform.from_position(0.0, 0.0, 0.0)})
        additive = Pose(bone_transforms={"hip": Transform.from_position(2.0, 0.0, 0.0)})
        result = base.additive_blend(additive, 0.5)
        # Position added: 0 + (2 * 0.5) = 1
        assert result.bone_transforms["hip"].position == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)

    def test_additive_blend_new_bone(self):
        """Additive blend should add new bones from additive pose."""
        base = Pose(bone_transforms={"hip": Transform.identity()})
        additive = Pose(bone_transforms={"new_bone": Transform.from_position(1.0, 2.0, 3.0)})
        result = base.additive_blend(additive, 1.0)
        assert "new_bone" in result.bone_transforms


# =============================================================================
# POSE FILTERING / MASKING
# =============================================================================


class TestPoseFiltering:
    """Tests for Pose filtering and masking operations."""

    def test_filter_bones_includes_requested(self, sample_pose: Pose):
        """filter_bones() should include only requested bones."""
        result = sample_pose.filter_bones(["hip", "spine"])
        assert "hip" in result.bone_transforms
        assert "spine" in result.bone_transforms
        assert "head" not in result.bone_transforms

    def test_filter_bones_ignores_missing(self, sample_pose: Pose):
        """filter_bones() should ignore bones not in pose."""
        result = sample_pose.filter_bones(["hip", "nonexistent"])
        assert "hip" in result.bone_transforms
        assert "nonexistent" not in result.bone_transforms
        assert len(result.bone_transforms) == 1

    def test_exclude_bones_removes_specified(self, sample_pose: Pose):
        """exclude_bones() should remove specified bones."""
        result = sample_pose.exclude_bones(["hip"])
        assert "hip" not in result.bone_transforms
        assert "spine" in result.bone_transforms
        assert "head" in result.bone_transforms

    def test_exclude_bones_ignores_missing(self, sample_pose: Pose):
        """exclude_bones() should ignore bones not in pose."""
        result = sample_pose.exclude_bones(["nonexistent"])
        assert len(result.bone_transforms) == 3


# =============================================================================
# POSE VALIDATION
# =============================================================================


class TestPoseValidation:
    """Tests for Pose validation and normalization."""

    def test_is_valid_returns_true_for_valid_pose(self, sample_pose: Pose):
        """is_valid() should return True for valid poses."""
        assert sample_pose.is_valid() is True

    def test_is_valid_returns_false_for_invalid_transform(self):
        """is_valid() should return False if any transform has NaN."""
        p = Pose(bone_transforms={"bad": Transform(position=(float("nan"), 0.0, 0.0))})
        assert p.is_valid() is False

    def test_normalized_normalizes_all_rotations(self):
        """normalized() should normalize all quaternion rotations."""
        p = Pose(
            bone_transforms={
                "bone1": Transform(rotation=(0.0, 2.0, 0.0, 2.0)),
                "bone2": Transform(rotation=(1.0, 1.0, 1.0, 1.0)),
            }
        )
        result = p.normalized()
        for t in result.bone_transforms.values():
            length = math.sqrt(sum(x * x for x in t.rotation))
            assert length == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# POSE MERGE
# =============================================================================


class TestPoseMerge:
    """Tests for Pose.merge()."""

    def test_merge_with_overwrite(self):
        """merge(overwrite=True) should overwrite self's bones with other's."""
        p1 = Pose(bone_transforms={"hip": Transform.from_position(1.0, 0.0, 0.0)})
        p2 = Pose(bone_transforms={"hip": Transform.from_position(2.0, 0.0, 0.0)})
        result = p1.merge(p2, overwrite=True)
        assert result.bone_transforms["hip"].position == (2.0, 0.0, 0.0)

    def test_merge_without_overwrite(self):
        """merge(overwrite=False) should keep self's bones over other's."""
        p1 = Pose(bone_transforms={"hip": Transform.from_position(1.0, 0.0, 0.0)})
        p2 = Pose(bone_transforms={"hip": Transform.from_position(2.0, 0.0, 0.0)})
        result = p1.merge(p2, overwrite=False)
        assert result.bone_transforms["hip"].position == (1.0, 0.0, 0.0)

    def test_merge_combines_unique_bones(self):
        """merge() should combine unique bones from both poses."""
        p1 = Pose(bone_transforms={"hip": Transform.identity()})
        p2 = Pose(bone_transforms={"spine": Transform.identity()})
        result = p1.merge(p2)
        assert "hip" in result.bone_transforms
        assert "spine" in result.bone_transforms


# =============================================================================
# POSE SPECIAL METHODS
# =============================================================================


class TestPoseSpecialMethods:
    """Tests for Pose special methods (__len__, __contains__, __iter__, __repr__)."""

    def test_len(self, sample_pose: Pose):
        """__len__ should return bone count."""
        assert len(sample_pose) == 3

    def test_contains_existing(self, sample_pose: Pose):
        """__contains__ should return True for existing bone."""
        assert "hip" in sample_pose

    def test_contains_missing(self, sample_pose: Pose):
        """__contains__ should return False for missing bone."""
        assert "nonexistent" not in sample_pose

    def test_iter(self, sample_pose: Pose):
        """__iter__ should iterate over (name, transform) pairs."""
        items = list(sample_pose)
        assert len(items) == 3
        names = [item[0] for item in items]
        assert set(names) == {"hip", "spine", "head"}

    def test_repr(self, sample_pose: Pose):
        """__repr__ should be informative."""
        repr_str = repr(sample_pose)
        assert "Pose" in repr_str
        assert "bones=" in repr_str


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_zero_quaternion_normalization(self):
        """Zero quaternion should normalize to identity."""
        q = (0.0, 0.0, 0.0, 0.0)
        result = _normalize_quaternion(q)
        assert result == (0.0, 0.0, 0.0, 1.0)

    def test_extreme_scales(self):
        """Transform should handle extreme scale values."""
        t = Transform.from_scale(1e10, 1e-10, 1e5)
        assert t.is_valid() is True
        assert t.scale == (1e10, 1e-10, 1e5)

    def test_very_small_blend_factor(self):
        """Blend with very small t should return self."""
        t1 = Transform.from_position(1.0, 0.0, 0.0)
        t2 = Transform.from_position(2.0, 0.0, 0.0)
        result = t1.blend(t2, 1e-10)
        assert result.position == t1.position

    def test_very_large_blend_factor(self):
        """Blend with very large t should return other."""
        t1 = Transform.from_position(1.0, 0.0, 0.0)
        t2 = Transform.from_position(2.0, 0.0, 0.0)
        result = t1.blend(t2, 1e10)
        assert result.position == t2.position

    def test_compose_with_zero_scale(self):
        """Compose should handle zero scale components."""
        parent = Transform.from_scale(0.0, 1.0, 1.0)
        child = Transform.from_position(1.0, 1.0, 1.0)
        result = parent.compose(child)
        # Zero scale should zero out child position in that component
        assert result.position[0] == pytest.approx(0.0, abs=1e-6)

    def test_blend_with_degenerate_quaternion(self):
        """Blend should handle degenerate (non-normalized) quaternions."""
        t1 = Transform(rotation=(0.0, 0.0, 0.0, 2.0))  # Non-unit quaternion
        t2 = Transform.identity()
        # Should not crash; SLERP will normalize internally
        result = t1.blend(t2, 0.5)
        assert result.is_valid()

    def test_pose_blend_with_disjoint_bones(self):
        """Pose blend with completely disjoint bone sets."""
        p1 = Pose(bone_transforms={"a": Transform.identity(), "b": Transform.identity()})
        p2 = Pose(bone_transforms={"c": Transform.identity(), "d": Transform.identity()})
        result = p1.blend(p2, 0.5)
        # All bones should be present
        assert set(result.bone_transforms.keys()) == {"a", "b", "c", "d"}

    def test_slerp_with_opposite_quaternions(self):
        """SLERP should handle quaternions representing opposite rotations."""
        # q and -q represent the same rotation
        q1 = (0.5, 0.5, 0.5, 0.5)
        q2 = (-0.5, -0.5, -0.5, -0.5)
        result = _slerp(q1, q2, 0.5)
        # Should take shortest path (essentially same as q1)
        length = math.sqrt(sum(x * x for x in result))
        assert length == pytest.approx(1.0, abs=1e-5)

    def test_rotate_vector_zero_vector(self):
        """Rotating zero vector should return zero vector."""
        v = (0.0, 0.0, 0.0)
        angle = math.pi / 4
        q = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
        result = _rotate_vector(v, q)
        assert result == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)

    def test_multiple_sequential_blends(self):
        """Sequential blends should accumulate correctly."""
        t = Transform.from_position(0.0, 0.0, 0.0)
        target = Transform.from_position(10.0, 0.0, 0.0)

        # Blend 10 times at t=0.1 each
        for _ in range(10):
            t = t.blend(target, 0.1)

        # Should approach target
        assert t.position[0] > 6.0

    def test_large_pose_blend(self):
        """Pose with many bones should blend correctly."""
        bones = [f"bone_{i}" for i in range(100)]
        p1 = Pose(
            bone_transforms={
                name: Transform.from_position(float(i), 0.0, 0.0)
                for i, name in enumerate(bones)
            }
        )
        p2 = Pose(
            bone_transforms={
                name: Transform.from_position(float(i) * 2, 0.0, 0.0)
                for i, name in enumerate(bones)
            }
        )
        result = p1.blend(p2, 0.5)
        assert len(result.bone_transforms) == 100
        # Check a sample bone
        assert result.bone_transforms["bone_10"].position[0] == pytest.approx(15.0, abs=1e-6)
