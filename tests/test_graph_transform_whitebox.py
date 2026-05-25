"""
WHITEBOX tests for engine/animation/graph/animation_graph.py — Transform and Pose.

WHITEBOX coverage plan:
  [Transform.blend]
    Path A1:  blend t <= 0.0  — early return of self.copy()
    Path A2:  blend t >= 1.0  — early return of other.copy()
    Path A3:  blend 0 < t < 1 — delegates to lerp (mid-range interpolation)

  [Transform._slerp]
    Path B1:  dot < 0          — q2 negated to take shortest arc
    Path B2:  dot > SLERP_DOT_THRESHOLD  — LERP + normalize for near-identical quaternions
    Path B3:  0 < dot <= threshold, sin_theta_0 >= threshold — full SLERP formula
    Path B4:  LERP branch with length == 0 — returns q1 fallback (defense-in-depth; mathematically unreachable in LERP branch)

  [Transform._rotate_vector]
    Path C1:  identity quaternion — vector unchanged
    Path C2:  90-degree quaternion — correct orthogonal result

  [Transform.compose]
    Path D1:  parent rotation + child position (rotated scaled child pos)
    Path D2:  non-uniform parent scale affects child position
    Path D3:  identity parent/child — leaves other unchanged

  [Transform._multiply_quaternion]
    Path E1:  identity * q == q
    Path E2:  non-identity multiplication correctness

  [Transform.__add__]
    Path F1:  additive blend — pos sum, rot multiply, scale multiply

  [Transform.copy]
    Path G1:  deep copy independence

  [Pose.blend]
    Path H1:  blend t <= 0.0 — early return of self.copy()
    Path H2:  blend t >= 1.0 — early return of other.copy()
    Path H3:  blend 0 < t < 1 — delegates to lerp

  [Pose.lerp]
    Path I1:  mismatched bone counts — padding with identity
    Path I2:  root motion — both poses have root motion, lerped together
    Path I3:  root motion — only self has root motion, propagates via copy
    Path I4:  root motion — only other has root motion, propagates via copy
    Path I5:  skeleton propagation — self.skeleton or other.skeleton

  [Pose.apply_mask]
    Path J1:  delegation to BoneMask.apply_to_pose
    Path J2:  weight_multiplier scaling

  [Pose misc]
    Path K1:  identity factory
    Path K2:  bone_count property
    Path K3:  get_transform bounds (valid index vs out-of-range)
    Path K4:  set_transform extends list when bone_index >= len
"""

from __future__ import annotations

import math
from copy import deepcopy

import pytest

from engine.animation.graph.animation_graph import (
    Bone,
    BoneMask,
    Pose,
    Skeleton,
    Transform,
)
from engine.animation.graph.config import get_config


# =========================================================================
# Constants
# =========================================================================

_SQRT2_2 = 0.7071067811865476
"""sqrt(2)/2 — used for 45-degree and 90-degree quaternions."""


# =========================================================================
# Helpers
# =========================================================================


def _quat_angle(q: tuple[float, float, float, float]) -> float:
    """Return the rotation angle (radians) of a unit quaternion."""
    w = min(1.0, max(-1.0, q[3]))
    return 2.0 * math.acos(w)


def _quat_axis(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Return the rotation axis of a unit quaternion (normalised)."""
    s = math.sqrt(1.0 - q[3] * q[3])
    if s < 1e-8:
        return (1.0, 0.0, 0.0)
    return (q[0] / s, q[1] / s, q[2] / s)


def _quat_length(q: tuple[float, float, float, float]) -> float:
    """Return the length of a quaternion."""
    return math.sqrt(sum(x * x for x in q))


# =========================================================================
# Transform.blend — early-exit paths (A1, A2, A3)
# =========================================================================


class TestTransformBlend:
    """Exercises every branch in Transform.blend."""

    def test_blend_t_zero_returns_copy_of_self(self) -> None:
        t1 = Transform(position=(1.0, 2.0, 3.0), scale=(4.0, 5.0, 6.0))
        t2 = Transform(position=(7.0, 8.0, 9.0), scale=(10.0, 11.0, 12.0))
        result = t1.blend(t2, 0.0)
        assert result.position == (1.0, 2.0, 3.0)
        assert result.rotation == (0.0, 0.0, 0.0, 1.0)
        assert result.scale == (4.0, 5.0, 6.0)

    def test_blend_t_negative_returns_copy_of_self(self) -> None:
        t1 = Transform(position=(1.0, 2.0, 3.0))
        t2 = Transform(position=(7.0, 8.0, 9.0))
        result = t1.blend(t2, -0.5)
        assert result.position == (1.0, 2.0, 3.0)

    def test_blend_t_one_returns_copy_of_other(self) -> None:
        t1 = Transform(position=(1.0, 2.0, 3.0), scale=(4.0, 5.0, 6.0))
        t2 = Transform(position=(7.0, 8.0, 9.0), scale=(10.0, 11.0, 12.0))
        result = t1.blend(t2, 1.0)
        assert result.position == (7.0, 8.0, 9.0)
        assert result.rotation == (0.0, 0.0, 0.0, 1.0)
        assert result.scale == (10.0, 11.0, 12.0)

    def test_blend_t_gt_one_returns_copy_of_other(self) -> None:
        t1 = Transform(position=(1.0, 2.0, 3.0))
        t2 = Transform(position=(7.0, 8.0, 9.0))
        result = t1.blend(t2, 2.0)
        assert result.position == (7.0, 8.0, 9.0)

    def test_blend_mid_range_equals_lerp(self) -> None:
        t1 = Transform(position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0))
        t2 = Transform(position=(10.0, 20.0, 30.0), rotation=(0.0, 0.0, 0.0, 1.0))
        blend_t = t1.blend(t2, 0.5)
        lerp_t = t1.lerp(t2, 0.5)
        assert blend_t.position == lerp_t.position == (5.0, 10.0, 15.0)
        assert blend_t.rotation == lerp_t.rotation
        assert blend_t.scale == lerp_t.scale

    def test_blend_returns_new_object_not_reference(self) -> None:
        """Verify that blend always returns a new Transform, never the original."""
        t1 = Transform(position=(1.0, 2.0, 3.0))
        t2 = Transform(position=(4.0, 5.0, 6.0))
        result = t1.blend(t2, 0.0)
        assert result is not t1
        result = t1.blend(t2, 1.0)
        assert result is not t2

    def test_blend_preserves_rotation_for_identical_rotations(self) -> None:
        """When both transforms have the same rotation, result should match."""
        t1 = Transform(rotation=(0.0, 0.0, _SQRT2_2, _SQRT2_2))
        t2 = Transform(rotation=(0.0, 0.0, _SQRT2_2, _SQRT2_2))
        result = t1.blend(t2, 0.5)
        for a, b in zip(result.rotation, t1.rotation):
            assert abs(a - b) < 1e-6


# =========================================================================
# Transform._slerp — edge cases (B1, B2, B3, B4)
# =========================================================================


class TestTransformSlerp:
    """Exercises every branch in the private _slerp helper."""

    @staticmethod
    def _slerp(
        q1: tuple[float, float, float, float],
        q2: tuple[float, float, float, float],
        t: float,
    ) -> tuple[float, float, float, float]:
        """Call the private slerp through a public path."""
        return Transform._slerp(q1, q2, t)

    # -- B1: dot < 0 -> negate q2, take shortest path --

    def test_slerp_negative_dot_negates_q2(self) -> None:
        """dot < 0 must negate q2 and proceed with positive dot."""
        q1 = (0.5, 0.5, 0.5, 0.5)
        q2 = (-0.5, -0.5, -0.5, 0.5)  # dot = -0.5 < 0
        # Call with negated q2 yields result that should match what _slerp
        # produces internally after negation.
        result = self._slerp(q1, q2, 0.5)
        # Result must be a unit quaternion
        assert abs(_quat_length(result) - 1.0) < 1e-6
        # The result should equal what we'd get with the negated q2 version
        q2_neg = (0.5, 0.5, 0.5, -0.5)
        result_neg = self._slerp(q1, q2_neg, 0.5)
        for a, b in zip(result, result_neg):
            assert abs(a - b) < 1e-10

    # -- B2: dot > SLERP_DOT_THRESHOLD -> LERP + normalize --

    def test_slerp_close_quaternions_uses_linear_path(self) -> None:
        """When |dot| > threshold, _slerp uses LERP+normalize instead of full SLERP."""
        q1 = (0.0, 0.0, 0.0, 1.0)
        # Very close to identity — dot ≈ 0.9999999 > 0.9995
        q2 = (0.001, 0.0, 0.0, math.sqrt(1.0 - 0.001 * 0.001))
        result = self._slerp(q1, q2, 0.5)
        # Must be unit length
        assert abs(_quat_length(result) - 1.0) < 1e-6
        # Must be different from both inputs
        assert result != q1
        assert result != q2
        # Near-zero-angle quaternions: w component should be close to 1
        assert abs(result[3]) > 0.99

    # -- B3: Standard SLERP path --

    def test_slerp_standard_path_45_degrees(self) -> None:
        """Full slerp at t=0.5 between 0 and 90 deg around Y gives 45 deg."""
        q1 = (0.0, 0.0, 0.0, 1.0)
        q2 = (0.0, _SQRT2_2, 0.0, _SQRT2_2)  # 90 deg around Y
        result = self._slerp(q1, q2, 0.5)
        # Unit quaternion
        assert abs(_quat_length(result) - 1.0) < 1e-6
        # Should be 45 degrees around Y
        angle = _quat_angle(result)
        expected_angle = math.radians(45.0)
        assert abs(angle - expected_angle) < 1e-4
        # Axis should be Y
        axis = _quat_axis(result)
        assert abs(axis[1] - 1.0) < 1e-4
        assert abs(axis[0]) < 1e-4
        assert abs(axis[2]) < 1e-4

    def test_slerp_standard_path_t_zero_returns_q1(self) -> None:
        """slerp with t=0 must return q1."""
        q1 = (0.0, 0.0, 0.0, 1.0)
        q2 = (0.0, _SQRT2_2, 0.0, _SQRT2_2)
        result = self._slerp(q1, q2, 0.0)
        for a, b in zip(result, q1):
            assert a == b

    def test_slerp_standard_path_t_one_returns_q2(self) -> None:
        """slerp with t=1 must return q2."""
        q1 = (0.0, 0.0, 0.0, 1.0)
        q2 = (0.0, _SQRT2_2, 0.0, _SQRT2_2)
        result = self._slerp(q1, q2, 1.0)
        for a, b in zip(result, q2):
            assert abs(a - b) < 1e-10

    # -- B2 variant: dot < 0, negation, then LERP + normalize --

    def test_slerp_negative_dot_then_linear_normalize(self) -> None:
        """After negating q2 (dot < 0), the flipped q2 enters the LERP
        branch and produces a correct normalized result."""
        q1 = (0.0, 0.0, 0.0, 1.0)
        # q2 is nearly equal to -q1: dot = -1 < 0
        # After negation in _slerp, q2 becomes (-1e-12, 0.0, 0.0, 1.0)
        # which is near-identity with dot ~ 1, entering the LERP branch.
        q2 = (1e-12, 0.0, 0.0, -1.0)
        result = self._slerp(q1, q2, 1.0)
        # The result is near identity (standard LERP + normalize of close quats)
        assert abs(_quat_length(result) - 1.0) < 1e-10
        assert abs(result[3] - 1.0) < 1e-10  # w component near 1


# =========================================================================
# Transform._rotate_vector — correctness (C1, C2)
# =========================================================================


class TestTransformRotateVector:
    """Exercises the static _rotate_vector helper."""

    @staticmethod
    def _rotate_vector(
        v: tuple[float, float, float],
        q: tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        return Transform._rotate_vector(v, q)

    def test_identity_rotation(self) -> None:
        """Identity quaternion must leave the vector unchanged."""
        result = self._rotate_vector((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        assert result == (1.0, 2.0, 3.0)

    def test_90_degrees_around_z(self) -> None:
        """90-degree rotation around Z: (1,0,0) -> (0,1,0)."""
        # q = (0, 0, sqrt2/2, sqrt2/2) = 90 deg around Z
        q = (0.0, 0.0, _SQRT2_2, _SQRT2_2)
        result = self._rotate_vector((1.0, 0.0, 0.0), q)
        assert abs(result[0] - 0.0) < 1e-6
        assert abs(result[1] - 1.0) < 1e-6
        assert abs(result[2] - 0.0) < 1e-6

    def test_180_degrees_around_x(self) -> None:
        """180-degree rotation around X: (0,1,0) -> (0,-1,0)."""
        # q = (1, 0, 0, 0) = 180 deg around X
        q = (1.0, 0.0, 0.0, 0.0)
        result = self._rotate_vector((0.0, 1.0, 0.0), q)
        assert abs(result[0] - 0.0) < 1e-6
        assert abs(result[1] - (-1.0)) < 1e-6
        assert abs(result[2] - 0.0) < 1e-6

    def test_preserves_vector_length(self) -> None:
        """Rotation must preserve the length of the vector."""
        v = (2.0, 3.0, 5.0)
        v_len = math.sqrt(sum(x * x for x in v))
        q = (0.0, _SQRT2_2, 0.0, _SQRT2_2)  # 90 deg around Y
        result = self._rotate_vector(v, q)
        result_len = math.sqrt(sum(x * x for x in result))
        assert abs(result_len / v_len - 1.0) < 1e-6


# =========================================================================
# Transform.compose — hierarchical TRS math (D1, D2, D3)
# =========================================================================


class TestTransformCompose:
    """Exercises hierarchical composition with rotation, scale, and position."""

    def test_parent_rotation_affects_child_position(self) -> None:
        """Parent rotation rotates the child's local position into world space."""
        parent = Transform(
            position=(10.0, 0.0, 0.0),
            rotation=(0.0, 0.0, _SQRT2_2, _SQRT2_2),  # 90 deg around Z
        )
        child = Transform(position=(0.0, 1.0, 0.0))
        result = parent.compose(child)
        # (0, 1, 0) rotated 90 deg Z -> (-1, 0, 0); add parent pos -> (9, 0, 0)
        assert abs(result.position[0] - 9.0) < 1e-6
        assert abs(result.position[1] - 0.0) < 1e-6
        assert abs(result.position[2] - 0.0) < 1e-6
        # Rotation = parent_rot * child_rot (child is identity)
        for a, b in zip(result.rotation, parent.rotation):
            assert abs(a - b) < 1e-6
        # Scale = parent_scale * child_scale (both (1,1,1))
        assert result.scale == (1.0, 1.0, 1.0)

    def test_non_uniform_parent_scale_scales_child_position(self) -> None:
        """Parent scale stretches child's local position before rotation."""
        parent = Transform(
            position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            scale=(2.0, 1.0, 0.5),
        )
        child = Transform(position=(1.0, 2.0, 3.0))
        result = parent.compose(child)
        # Scaled: (2, 2, 1.5); rotated by identity same; add pos (0,0,0)
        assert abs(result.position[0] - 2.0) < 1e-6
        assert abs(result.position[1] - 2.0) < 1e-6
        assert abs(result.position[2] - 1.5) < 1e-6
        # Scale = parent_scale * child_scale (child is (1,1,1))
        assert result.scale == (2.0, 1.0, 0.5)
        # Rotation = parent * identity
        assert result.rotation == (0.0, 0.0, 0.0, 1.0)

    def test_compose_with_identity_parent_equals_child(self) -> None:
        """Composing with an identity parent yields the child transform."""
        parent = Transform.identity()
        child = Transform(
            position=(5.0, 6.0, 7.0),
            rotation=(0.0, _SQRT2_2, 0.0, _SQRT2_2),
            scale=(2.0, 3.0, 4.0),
        )
        result = parent.compose(child)
        assert abs(result.position[0] - 5.0) < 1e-6
        assert abs(result.position[1] - 6.0) < 1e-6
        assert abs(result.position[2] - 7.0) < 1e-6
        for a, b in zip(result.rotation, child.rotation):
            assert abs(a - b) < 1e-6
        assert result.scale == (2.0, 3.0, 4.0)

    def test_compose_with_identity_child_equals_parent(self) -> None:
        """Composing with an identity child yields the parent transform."""
        parent = Transform(
            position=(5.0, 6.0, 7.0),
            rotation=(0.0, _SQRT2_2, 0.0, _SQRT2_2),
            scale=(2.0, 3.0, 4.0),
        )
        child = Transform.identity()
        result = parent.compose(child)
        assert abs(result.position[0] - 5.0) < 1e-6
        assert abs(result.position[1] - 6.0) < 1e-6
        assert abs(result.position[2] - 7.0) < 1e-6
        for a, b in zip(result.rotation, parent.rotation):
            assert abs(a - b) < 1e-6
        assert result.scale == (2.0, 3.0, 4.0)


# =========================================================================
# Transform._multiply_quaternion — correctness (E1, E2)
# =========================================================================


class TestTransformMultiplyQuaternion:
    """Exercises quaternion multiplication used by compose and __add__."""

    @staticmethod
    def _mul(
        q1: tuple[float, float, float, float],
        q2: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        return Transform._multiply_quaternion(q1, q2)

    def test_identity_times_q_equals_q(self) -> None:
        """q_identity * q == q"""
        result = self._mul((0.0, 0.0, 0.0, 1.0), (0.0, _SQRT2_2, 0.0, _SQRT2_2))
        for a, b in zip(result, (0.0, _SQRT2_2, 0.0, _SQRT2_2)):
            assert abs(a - b) < 1e-10

    def test_q_times_identity_equals_q(self) -> None:
        """q * q_identity == q"""
        result = self._mul((0.0, _SQRT2_2, 0.0, _SQRT2_2), (0.0, 0.0, 0.0, 1.0))
        for a, b in zip(result, (0.0, _SQRT2_2, 0.0, _SQRT2_2)):
            assert abs(a - b) < 1e-10

    def test_90z_times_90z_equals_180z(self) -> None:
        """Two 90-degree rotations around Z combine to 180 degrees around Z."""
        qz90 = (0.0, 0.0, _SQRT2_2, _SQRT2_2)
        result = self._mul(qz90, qz90)
        # 180 around Z: (0, 0, 1, 0)
        assert abs(result[0] - 0.0) < 1e-6
        assert abs(result[1] - 0.0) < 1e-6
        assert abs(result[2] - 1.0) < 1e-6
        assert abs(result[3] - 0.0) < 1e-6


# =========================================================================
# Transform.__add__ (additive blend) — (F1)
# =========================================================================


class TestTransformAdd:
    """Exercises additive blend via __add__."""

    def test_add_adds_position(self) -> None:
        t1 = Transform(position=(1.0, 2.0, 3.0))
        t2 = Transform(position=(4.0, 5.0, 6.0))
        result = t1 + t2
        assert result.position == (5.0, 7.0, 9.0)

    def test_add_multiplies_rotation(self) -> None:
        t1 = Transform(rotation=(0.0, 0.0, _SQRT2_2, _SQRT2_2))  # 90 deg Z
        t2 = Transform(rotation=(0.0, 0.0, _SQRT2_2, _SQRT2_2))  # 90 deg Z
        result = t1 + t2
        # 90 * 90 = 180 deg around Z: (0, 0, 1, 0)
        assert abs(result.rotation[0] - 0.0) < 1e-6
        assert abs(result.rotation[2] - 1.0) < 1e-6
        assert abs(result.rotation[3] - 0.0) < 1e-6

    def test_add_multiplies_scale(self) -> None:
        t1 = Transform(scale=(2.0, 3.0, 4.0))
        t2 = Transform(scale=(5.0, 6.0, 7.0))
        result = t1 + t2
        assert result.scale == (10.0, 18.0, 28.0)


# =========================================================================
# Transform.copy — deep copy (G1)
# =========================================================================


class TestTransformCopy:
    """Exercises copy and mutation isolation."""

    def test_copy_is_equal_but_independent(self) -> None:
        t1 = Transform(
            position=(1.0, 2.0, 3.0),
            rotation=(0.0, 0.0, _SQRT2_2, _SQRT2_2),
            scale=(4.0, 5.0, 6.0),
        )
        t2 = t1.copy()
        # Same values
        assert t2.position == (1.0, 2.0, 3.0)
        for a, b in zip(t2.rotation, t1.rotation):
            assert a == b
        assert t2.scale == (4.0, 5.0, 6.0)
        # Not the same object
        assert t2 is not t1

    def test_identity_returns_default_transform(self) -> None:
        t = Transform.identity()
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)


# =========================================================================
# Transform.lerp — direct call
# =========================================================================


class TestTransformLerp:
    """Exercises the lerp method that blend delegates to."""

    def test_lerp_interpolates_position_scale(self) -> None:
        t1 = Transform(position=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
        t2 = Transform(position=(10.0, 20.0, 30.0), scale=(3.0, 4.0, 5.0))
        result = t1.lerp(t2, 0.25)
        assert result.position == (2.5, 5.0, 7.5)
        assert result.scale == (1.5, 1.75, 2.0)

    def test_lerp_produces_unit_rotation(self) -> None:
        """lerp must produce a unit-length rotation quaternion."""
        t1 = Transform()
        t2 = Transform(rotation=(0.0, _SQRT2_2, 0.0, _SQRT2_2))
        result = t1.lerp(t2, 0.5)
        length = math.sqrt(sum(x * x for x in result.rotation))
        assert abs(length - 1.0) < 1e-6


# =========================================================================
# Pose.blend — early-exit paths (H1, H2, H3)
# =========================================================================


class TestPoseBlend:
    """Exercises every branch in Pose.blend."""

    def test_blend_t_zero_returns_self_copy(self) -> None:
        p1 = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])
        p2 = Pose(transforms=[Transform(position=(4.0, 5.0, 6.0))])
        result = p1.blend(p2, 0.0)
        assert result is not p1
        assert result.transforms[0].position == (1.0, 2.0, 3.0)

    def test_blend_t_negative_returns_self_copy(self) -> None:
        p1 = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])
        p2 = Pose(transforms=[Transform(position=(4.0, 5.0, 6.0))])
        result = p1.blend(p2, -0.5)
        assert result.transforms[0].position == (1.0, 2.0, 3.0)

    def test_blend_t_one_returns_other_copy(self) -> None:
        p1 = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])
        p2 = Pose(transforms=[Transform(position=(4.0, 5.0, 6.0))])
        result = p1.blend(p2, 1.0)
        assert result is not p2
        assert result.transforms[0].position == (4.0, 5.0, 6.0)

    def test_blend_t_gt_one_returns_other_copy(self) -> None:
        p1 = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])
        p2 = Pose(transforms=[Transform(position=(4.0, 5.0, 6.0))])
        result = p1.blend(p2, 2.0)
        assert result.transforms[0].position == (4.0, 5.0, 6.0)

    def test_blend_mid_range_equals_lerp(self) -> None:
        p1 = Pose(transforms=[Transform(position=(0.0, 0.0, 0.0))])
        p2 = Pose(transforms=[Transform(position=(10.0, 20.0, 30.0))])
        result_b = p1.blend(p2, 0.5)
        result_l = p1.lerp(p2, 0.5)
        assert abs(result_b.transforms[0].position[0] - 5.0) < 1e-6
        assert result_b.transforms[0].position == result_l.transforms[0].position


# =========================================================================
# Pose.lerp — bone count mismatch, root motion, skeleton (I1-I5)
# =========================================================================


class TestPoseLerp:
    """Exercises internal paths in Pose.lerp."""

    def test_mismatched_bone_counts_pads_with_identity(self) -> None:
        """When self has fewer bones, missing bones default to identity."""
        p1 = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])  # 1 bone
        p2 = Pose(transforms=[
            Transform(position=(4.0, 5.0, 6.0)),
            Transform(position=(7.0, 8.0, 9.0)),
            Transform(position=(10.0, 11.0, 12.0)),
        ])  # 3 bones
        result = p1.lerp(p2, 0.5)
        assert len(result.transforms) == 3
        # Bone 0: interpolated
        assert abs(result.transforms[0].position[0] - 2.5) < 1e-6
        # Bones 1-2: p1 defaults to identity, so lerp(identity, p2.bone, 0.5)
        assert abs(result.transforms[1].position[0] - 3.5) < 1e-6  # 0.5 * 7
        assert abs(result.transforms[2].position[0] - 5.0) < 1e-6  # 0.5 * 10

    def test_mismatched_bones_other_shorter(self) -> None:
        """When other has fewer bones, use identity for the missing indices."""
        p1 = Pose(transforms=[
            Transform(position=(4.0, 5.0, 6.0)),
            Transform(position=(7.0, 8.0, 9.0)),
        ])  # 2 bones
        p2 = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])  # 1 bone
        result = p1.lerp(p2, 0.25)
        assert len(result.transforms) == 2
        # Bone 0: interpolated
        assert abs(result.transforms[0].position[0] - 3.25) < 1e-6
        # Bone 1: lerp(p1.bone1, identity, 0.25)
        assert abs(result.transforms[1].position[0] - 5.25) < 1e-6

    def test_root_motion_both_present(self) -> None:
        """When both poses have root_motion, the result interpolates them."""
        p1 = Pose(
            transforms=[Transform.identity()],
            root_motion=Transform(position=(0.0, 0.0, 0.0)),
        )
        p2 = Pose(
            transforms=[Transform.identity()],
            root_motion=Transform(position=(10.0, 20.0, 30.0)),
        )
        result = p1.lerp(p2, 0.5)
        assert result.root_motion is not None
        assert result.root_motion.position == (5.0, 10.0, 15.0)

    def test_root_motion_only_on_self(self) -> None:
        """When only self has root_motion, it propagates via copy."""
        rm = Transform(position=(1.0, 2.0, 3.0))
        p1 = Pose(transforms=[Transform.identity()], root_motion=rm)
        p2 = Pose(transforms=[Transform.identity()])
        result = p1.lerp(p2, 0.5)
        assert result.root_motion is not None
        assert result.root_motion.position == (1.0, 2.0, 3.0)
        assert result.root_motion is not rm  # must be a copy

    def test_root_motion_only_on_other(self) -> None:
        """When only other has root_motion, it propagates via copy."""
        rm = Transform(position=(4.0, 5.0, 6.0))
        p1 = Pose(transforms=[Transform.identity()])
        p2 = Pose(transforms=[Transform.identity()], root_motion=rm)
        result = p1.lerp(p2, 0.5)
        assert result.root_motion is not None
        assert result.root_motion.position == (4.0, 5.0, 6.0)
        assert result.root_motion is not rm  # must be a copy

    def test_root_motion_neither_present(self) -> None:
        """When neither pose has root_motion, result has None."""
        p1 = Pose(transforms=[Transform.identity()])
        p2 = Pose(transforms=[Transform.identity()])
        result = p1.lerp(p2, 0.5)
        assert result.root_motion is None

    def test_skeleton_propagation_self_wins(self) -> None:
        """self.skeleton is preferred when present."""
        skel = Skeleton()
        skel.add_bone("root")
        p1 = Pose(transforms=[Transform.identity()], skeleton=skel)
        other_skel = Skeleton()
        other_skel.add_bone("other_root")
        p2 = Pose(transforms=[Transform.identity()], skeleton=other_skel)
        result = p1.lerp(p2, 0.5)
        assert result.skeleton is skel

    def test_skeleton_propagation_fallback_to_other(self) -> None:
        """When self.skeleton is None, use other.skeleton."""
        skel = Skeleton()
        skel.add_bone("root")
        p1 = Pose(transforms=[Transform.identity()])
        p2 = Pose(transforms=[Transform.identity()], skeleton=skel)
        result = p1.lerp(p2, 0.5)
        assert result.skeleton is skel

    def test_skeleton_propagation_none_when_both_none(self) -> None:
        """When neither pose has skeleton, result skeleton is None."""
        p1 = Pose(transforms=[Transform.identity()])
        p2 = Pose(transforms=[Transform.identity()])
        result = p1.lerp(p2, 0.5)
        assert result.skeleton is None

    def test_skeleton_propagates_through_blend(self) -> None:
        """Pose.blend must propagate skeleton identically to lerp."""
        skel = Skeleton()
        skel.add_bone("root")
        p1 = Pose(transforms=[Transform.identity()], skeleton=skel)
        p2 = Pose(transforms=[Transform.identity()])
        result = p1.blend(p2, 0.5)
        assert result.skeleton is skel


# =========================================================================
# Pose.apply_mask — delegation (J1, J2)
# =========================================================================


class TestPoseApplyMask:
    """Exercises apply_mask delegation to BoneMask.apply_to_pose."""

    def test_full_mask_leaves_pose_unchanged(self) -> None:
        """A mask with all weights=1 should yield the original pose interpolated at t=1."""
        pose = Pose(transforms=[
            Transform(position=(1.0, 2.0, 3.0)),
            Transform(position=(4.0, 5.0, 6.0)),
        ])
        mask = BoneMask(name="full")
        mask.set_weights([0, 1], 1.0)
        result = pose.apply_mask(mask)
        assert result.bone_count() == 2
        # At weight=1: identity.lerp(pose, 1.0) == pose
        for i in range(pose.bone_count()):
            assert result.transforms[i].position == pose.transforms[i].position

    def test_zero_mask_returns_identity(self) -> None:
        """A mask with all weights=0 should return identity poses."""
        pose = Pose(transforms=[
            Transform(position=(1.0, 2.0, 3.0)),
            Transform(position=(4.0, 5.0, 6.0)),
        ])
        mask = BoneMask(name="zero")
        # All weights default to 0.0
        result = pose.apply_mask(mask)
        for t in result.transforms:
            assert t.position == (0.0, 0.0, 0.0)
            assert t.rotation == (0.0, 0.0, 0.0, 1.0)
            assert t.scale == (1.0, 1.0, 1.0)

    def test_partial_mask_blends_some_bones(self) -> None:
        """A mask with mixed weights blends only the weighted bones."""
        pose = Pose(transforms=[
            Transform(position=(10.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
            Transform(position=(0.0, 20.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
        ])
        mask = BoneMask(name="partial")
        mask.set_weight(0, 0.5)
        mask.set_weight(1, 0.0)
        result = pose.apply_mask(mask)
        # Bone 0: identity.lerp(pose, 0.5) = (5, 0, 0)
        assert abs(result.transforms[0].position[0] - 5.0) < 1e-6
        # Bone 1: weight=0, identity
        assert result.transforms[1].position == (0.0, 0.0, 0.0)

    def test_weight_multiplier_scales_all_weights_down(self) -> None:
        """weight_multiplier must scale all bone weights."""
        pose = Pose(transforms=[
            Transform(position=(10.0, 0.0, 0.0)),
        ])
        mask = BoneMask(name="scaled")
        mask.set_weight(0, 1.0)
        # With multiplier=0.5, effective weight = 0.5
        result = pose.apply_mask(mask, weight_multiplier=0.5)
        assert abs(result.transforms[0].position[0] - 5.0) < 1e-6

    def test_weight_multiplier_can_zero_out(self) -> None:
        """multiplier=0.0 should behave like all weights are zero."""
        pose = Pose(transforms=[
            Transform(position=(10.0, 0.0, 0.0)),
        ])
        mask = BoneMask(name="full_if_not_scaled")
        mask.set_weight(0, 1.0)
        result = pose.apply_mask(mask, weight_multiplier=0.0)
        assert result.transforms[0].position == (0.0, 0.0, 0.0)

    def test_apply_mask_returns_new_pose(self) -> None:
        """apply_mask must not mutate the original pose."""
        pose = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])
        mask = BoneMask(name="immutable")
        mask.set_weight(0, 0.5)
        original_pos = pose.transforms[0].position
        result = pose.apply_mask(mask)
        assert result is not pose
        assert pose.transforms[0].position == original_pos

    def test_empty_pose_apply_mask(self) -> None:
        """Applying a mask to a pose with zero bones should return a zero-bone pose."""
        pose = Pose(transforms=[])
        mask = BoneMask(name="empty")
        result = pose.apply_mask(mask)
        assert result.bone_count() == 0


# =========================================================================
# Pose misc — identity, bone_count, get/set_transform (K1-K4)
# =========================================================================


class TestPoseIdentity:
    def test_identity_factory_creates_correct_bone_count(self) -> None:
        p = Pose.identity(5)
        assert p.bone_count() == 5
        for t in p.transforms:
            assert t.position == (0.0, 0.0, 0.0)
            assert t.rotation == (0.0, 0.0, 0.0, 1.0)
            assert t.scale == (1.0, 1.0, 1.0)

    def test_identity_zero_bones(self) -> None:
        p = Pose.identity(0)
        assert p.bone_count() == 0
        assert p.transforms == []


class TestPoseBoneCount:
    def test_empty_pose(self) -> None:
        assert Pose().bone_count() == 0

    def test_with_bones(self) -> None:
        p = Pose(transforms=[Transform.identity(), Transform.identity()])
        assert p.bone_count() == 2


class TestPoseGetTransform:
    def test_valid_index(self) -> None:
        p = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])
        t = p.get_transform(0)
        assert t.position == (1.0, 2.0, 3.0)

    def test_out_of_range_returns_identity(self) -> None:
        p = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])
        t = p.get_transform(5)
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_negative_index_returns_identity(self) -> None:
        p = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])
        t = p.get_transform(-1)
        assert t.position == (0.0, 0.0, 0.0)

    def test_empty_pose_returns_identity(self) -> None:
        p = Pose()
        t = p.get_transform(0)
        assert t.position == (0.0, 0.0, 0.0)


class TestPoseSetTransform:
    def test_set_existing_index(self) -> None:
        p = Pose(transforms=[Transform.identity()])
        p.set_transform(0, Transform(position=(1.0, 2.0, 3.0)))
        assert p.transforms[0].position == (1.0, 2.0, 3.0)

    def test_set_beyond_end_extends_with_identity(self) -> None:
        p = Pose(transforms=[Transform.identity()])
        p.set_transform(3, Transform(position=(9.0, 9.0, 9.0)))
        assert len(p.transforms) == 4
        # indices 1-2 filled with identity
        assert p.transforms[1].position == (0.0, 0.0, 0.0)
        assert p.transforms[2].rotation == (0.0, 0.0, 0.0, 1.0)
        # index 3 set correctly
        assert p.transforms[3].position == (9.0, 9.0, 9.0)

    def test_set_at_end_appends(self) -> None:
        p = Pose(transforms=[Transform.identity()])
        p.set_transform(1, Transform(position=(5.0, 5.0, 5.0)))
        assert len(p.transforms) == 2
        assert p.transforms[1].position == (5.0, 5.0, 5.0)


# =========================================================================
# Pose.copy
# =========================================================================


class TestPoseCopy:
    def test_copy_independence(self) -> None:
        t = Transform(position=(1.0, 2.0, 3.0))
        rm = Transform(position=(10.0, 20.0, 30.0))
        skel = Skeleton()
        skel.add_bone("root")
        p1 = Pose(transforms=[t], root_motion=rm, skeleton=skel)
        p2 = p1.copy()
        # Modify the copy's internal transforms
        p2.transforms[0] = Transform(position=(99.0, 99.0, 99.0))
        assert p1.transforms[0].position == (1.0, 2.0, 3.0)
        # Root motion is also deep-copied
        assert p1.root_motion is not None
        assert p1.root_motion is not p2.root_motion
        # Skeleton is shared by reference (same object)
        assert p1.skeleton is p2.skeleton

    def test_copy_no_root_motion(self) -> None:
        p1 = Pose(transforms=[Transform.identity()])
        p2 = p1.copy()
        assert p2.root_motion is None


# =========================================================================
# Pose.additive_blend
# =========================================================================


class TestPoseAdditiveBlend:
    def test_additive_blend_full_weight(self) -> None:
        base = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])
        additive = Pose(transforms=[Transform(position=(4.0, 5.0, 6.0))])
        result = base.additive_blend(additive, weight=1.0)
        # Position = base + additive
        assert result.transforms[0].position == (5.0, 7.0, 9.0)

    def test_additive_blend_scaled_weight(self) -> None:
        base = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])
        additive = Pose(transforms=[Transform(position=(10.0, 20.0, 30.0))])
        result = base.additive_blend(additive, weight=0.5)
        # additive is lerped with identity at weight=0.5 → (5, 10, 15)
        # then added to base → (6, 12, 18)
        assert result.transforms[0].position == (6.0, 12.0, 18.0)

    def test_additive_blend_mismatched_bones(self) -> None:
        """When additive has more bones, missing base bones are identity."""
        base = Pose(transforms=[Transform(position=(1.0, 2.0, 3.0))])
        additive = Pose(transforms=[
            Transform(position=(4.0, 5.0, 6.0)),
            Transform(position=(7.0, 8.0, 9.0)),
        ])
        result = base.additive_blend(additive)
        assert len(result.transforms) == 2
        # Bone 0: (1, 2, 3) + (4, 5, 6) = (5, 7, 9)
        assert result.transforms[0].position == (5.0, 7.0, 9.0)
        # Bone 1: identity + (7, 8, 9) = (7, 8, 9)
        assert result.transforms[1].position == (7.0, 8.0, 9.0)
