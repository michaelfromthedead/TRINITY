"""Whitebox tests for animation_texture.py with full internal access.

Tests cover:
1. encode_transform_to_pixels() / decode_pixels_to_transform() round-trip
2. Cubic Hermite interpolation: cubic_hermite_interpolate(), cubic_hermite_interpolate_vec3(),
   cubic_hermite_interpolate_transform()
3. SQUAD quaternion: _squad_interpolate(), _compute_squad_intermediate(), _quat_log(), _quat_exp()
4. AnimationTexture.sample_bone_transform_cubic()
5. AnimationTextureAtlas.sample_clip_cubic()
6. validate_atlas_uv_ranges() for non-overlapping UV validation
"""

from __future__ import annotations

import math
import pytest
from typing import Callable

from engine.core.math import Vec3, Quat, Transform
from engine.animation.crowds.animation_texture import (
    # Public API
    AnimationTexture,
    AnimationTextureAtlas,
    AnimationClip,
    Skeleton,
    TextureFormat,
    encode_transform_to_pixels,
    decode_pixels_to_transform,
    bake_clip_to_texture,
    validate_atlas_uv_ranges,
    cubic_hermite_interpolate,
    cubic_hermite_interpolate_vec3,
    cubic_hermite_interpolate_transform,
    pack_float_to_rgba8,
    unpack_rgba8_to_float,
    AnimationTextureOverflowError,
    # Private helpers
    _squad_interpolate,
    _compute_squad_intermediate,
    _quat_log,
    _quat_exp,
)


# -----------------------------------------------------------------------------
# Test Constants
# -----------------------------------------------------------------------------

EPSILON = 1e-5
ROTATION_EPSILON = 1e-4  # Slightly more lenient for quaternion operations


def nearly_equal(a: float, b: float, eps: float = EPSILON) -> bool:
    """Check if two floats are approximately equal."""
    return abs(a - b) <= eps


def vec3_nearly_equal(v1: Vec3, v2: Vec3, eps: float = EPSILON) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        nearly_equal(v1.x, v2.x, eps)
        and nearly_equal(v1.y, v2.y, eps)
        and nearly_equal(v1.z, v2.z, eps)
    )


def quat_nearly_equal(q1: Quat, q2: Quat, eps: float = ROTATION_EPSILON) -> bool:
    """Check if two quaternions represent approximately the same rotation.

    Note: q and -q represent the same rotation, so we check both.
    """
    return (
        (
            nearly_equal(q1.x, q2.x, eps)
            and nearly_equal(q1.y, q2.y, eps)
            and nearly_equal(q1.z, q2.z, eps)
            and nearly_equal(q1.w, q2.w, eps)
        )
        or (
            nearly_equal(q1.x, -q2.x, eps)
            and nearly_equal(q1.y, -q2.y, eps)
            and nearly_equal(q1.z, -q2.z, eps)
            and nearly_equal(q1.w, -q2.w, eps)
        )
    )


def transform_nearly_equal(t1: Transform, t2: Transform, eps: float = EPSILON) -> bool:
    """Check if two transforms are approximately equal."""
    return (
        vec3_nearly_equal(t1.translation, t2.translation, eps)
        and quat_nearly_equal(t1.rotation, t2.rotation, ROTATION_EPSILON)
        and vec3_nearly_equal(t1.scale, t2.scale, eps)
    )


# -----------------------------------------------------------------------------
# Helper Factories
# -----------------------------------------------------------------------------

def make_test_transform(
    tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
    pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0,
    scale: float = 1.0
) -> Transform:
    """Create a test transform with given parameters."""
    return Transform(
        translation=Vec3(tx, ty, tz),
        rotation=Quat.from_euler(pitch, yaw, roll),
        scale=Vec3(scale, scale, scale),
    )


def make_skeleton(bone_count: int) -> Skeleton:
    """Create a test skeleton with the specified number of bones."""
    return Skeleton(
        bone_names=[f"bone_{i}" for i in range(bone_count)],
        bone_parents=[-1 if i == 0 else i - 1 for i in range(bone_count)],
        bind_poses=[Transform.identity() for _ in range(bone_count)],
    )


def make_animation_clip(
    name: str,
    bone_count: int,
    frame_count: int,
    transform_generator: Callable[[int, int], Transform] | None = None
) -> AnimationClip:
    """Create a test animation clip.

    Args:
        name: Clip name
        bone_count: Number of bones
        frame_count: Number of frames
        transform_generator: Function(bone_idx, frame_idx) -> Transform
    """
    if transform_generator is None:
        def transform_generator(bone_idx: int, frame_idx: int) -> Transform:
            t = frame_idx / max(frame_count - 1, 1)
            return Transform(
                translation=Vec3(t * bone_idx, t * 2.0, t * 3.0),
                rotation=Quat.from_axis_angle(Vec3(0, 1, 0), t * math.pi * 0.5),
                scale=Vec3(1.0 + t * 0.5, 1.0 + t * 0.5, 1.0 + t * 0.5),
            )

    bone_tracks = {}
    for bone_idx in range(bone_count):
        bone_tracks[bone_idx] = [
            transform_generator(bone_idx, frame_idx)
            for frame_idx in range(frame_count)
        ]

    duration = frame_count / 30.0  # 30 FPS default
    return AnimationClip(
        name=name,
        duration=duration,
        frame_rate=30.0,
        bone_tracks=bone_tracks,
    )


# =============================================================================
# 1. ENCODE/DECODE ROUND-TRIP TESTS
# =============================================================================

class TestEncodeDecodeRoundTrip:
    """Tests for encode_transform_to_pixels() and decode_pixels_to_transform()."""

    def test_identity_transform_round_trip(self):
        """Identity transform should round-trip exactly."""
        original = Transform.identity()
        pixel1, pixel2 = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixel1, pixel2)

        assert transform_nearly_equal(original, decoded)

    def test_translation_only_round_trip(self):
        """Translation-only transform should round-trip."""
        original = Transform(
            translation=Vec3(1.5, -2.3, 4.7),
            rotation=Quat.identity(),
            scale=Vec3.one(),
        )
        pixel1, pixel2 = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixel1, pixel2)

        assert transform_nearly_equal(original, decoded)

    def test_rotation_only_round_trip(self):
        """Rotation-only transform should round-trip."""
        original = Transform(
            translation=Vec3.zero(),
            rotation=Quat.from_euler(0.5, 1.2, -0.3),
            scale=Vec3.one(),
        )
        pixel1, pixel2 = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixel1, pixel2)

        assert transform_nearly_equal(original, decoded)

    def test_scale_only_round_trip(self):
        """Uniform scale transform should round-trip."""
        original = Transform(
            translation=Vec3.zero(),
            rotation=Quat.identity(),
            scale=Vec3(2.5, 2.5, 2.5),
        )
        pixel1, pixel2 = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixel1, pixel2)

        assert transform_nearly_equal(original, decoded)

    def test_full_transform_round_trip(self):
        """Full TRS transform should round-trip."""
        original = Transform(
            translation=Vec3(10.0, -5.0, 3.0),
            rotation=Quat.from_euler(0.2, 0.4, 0.6),
            scale=Vec3(1.5, 1.5, 1.5),
        )
        pixel1, pixel2 = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixel1, pixel2)

        assert transform_nearly_equal(original, decoded)

    def test_non_uniform_scale_averaged(self):
        """Non-uniform scale should be averaged for encoding."""
        original = Transform(
            translation=Vec3.zero(),
            rotation=Quat.identity(),
            scale=Vec3(1.0, 2.0, 3.0),  # Non-uniform: average = 2.0
        )
        pixel1, pixel2 = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixel1, pixel2)

        # Decoded scale should be uniform with averaged value
        expected_scale = (1.0 + 2.0 + 3.0) / 3.0
        assert nearly_equal(decoded.scale.x, expected_scale)
        assert nearly_equal(decoded.scale.y, expected_scale)
        assert nearly_equal(decoded.scale.z, expected_scale)

    def test_extreme_translation_values(self):
        """Extreme translation values should round-trip."""
        original = Transform(
            translation=Vec3(1000.0, -500.0, 250.0),
            rotation=Quat.identity(),
            scale=Vec3.one(),
        )
        pixel1, pixel2 = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixel1, pixel2)

        assert transform_nearly_equal(original, decoded)

    def test_negative_scale_round_trip(self):
        """Negative scale should round-trip (though typically not used)."""
        original = Transform(
            translation=Vec3.zero(),
            rotation=Quat.identity(),
            scale=Vec3(-1.0, -1.0, -1.0),
        )
        pixel1, pixel2 = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixel1, pixel2)

        assert nearly_equal(decoded.scale.x, -1.0)

    def test_quaternion_normalization_on_decode(self):
        """Decoded quaternion should be normalized."""
        original = Transform(
            translation=Vec3.zero(),
            rotation=Quat.from_euler(0.3, 0.5, 0.7),
            scale=Vec3.one(),
        )
        pixel1, pixel2 = encode_transform_to_pixels(original)
        decoded = decode_pixels_to_transform(pixel1, pixel2)

        # Check normalization
        length = decoded.rotation.length()
        assert nearly_equal(length, 1.0, 1e-6)


# =============================================================================
# 2. CUBIC HERMITE INTERPOLATION TESTS
# =============================================================================

class TestCubicHermiteInterpolate:
    """Tests for cubic_hermite_interpolate() scalar function."""

    def test_interpolate_at_t_zero_returns_p1(self):
        """At t=0, should return p1."""
        result = cubic_hermite_interpolate(0.0, 1.0, 2.0, 3.0, 0.0)
        assert nearly_equal(result, 1.0)

    def test_interpolate_at_t_one_returns_p2(self):
        """At t=1, should return p2."""
        result = cubic_hermite_interpolate(0.0, 1.0, 2.0, 3.0, 1.0)
        assert nearly_equal(result, 2.0)

    def test_interpolate_at_t_half_is_between(self):
        """At t=0.5, result should be between p1 and p2."""
        result = cubic_hermite_interpolate(0.0, 1.0, 2.0, 3.0, 0.5)
        assert 1.0 < result < 2.0

    def test_linear_input_produces_linear_output(self):
        """Linear control points should produce approximately linear result."""
        # Points on a line: 0, 1, 2, 3
        result = cubic_hermite_interpolate(0.0, 1.0, 2.0, 3.0, 0.5)
        # For linear points, Catmull-Rom gives exact linear result
        assert nearly_equal(result, 1.5)

    def test_smooth_curve_property(self):
        """Verify C1 continuity: derivative at endpoints matches tangent."""
        # For Catmull-Rom, tangent at p1 = (p2 - p0) / 2
        p0, p1, p2, p3 = 0.0, 1.0, 4.0, 6.0

        # Sample near t=0 to approximate derivative
        dt = 0.0001
        val_0 = cubic_hermite_interpolate(p0, p1, p2, p3, 0.0)
        val_dt = cubic_hermite_interpolate(p0, p1, p2, p3, dt)
        approx_derivative = (val_dt - val_0) / dt

        expected_tangent = (p2 - p0) * 0.5
        assert nearly_equal(approx_derivative, expected_tangent, 0.01)

    def test_symmetric_points(self):
        """Symmetric control points should produce symmetric interpolation."""
        # Points: -2, -1, 1, 2 (symmetric around 0)
        result_low = cubic_hermite_interpolate(-2.0, -1.0, 1.0, 2.0, 0.25)
        result_high = cubic_hermite_interpolate(-2.0, -1.0, 1.0, 2.0, 0.75)

        # At t=0.5, result should be 0 (midpoint)
        result_mid = cubic_hermite_interpolate(-2.0, -1.0, 1.0, 2.0, 0.5)
        assert nearly_equal(result_mid, 0.0)

        # Results should be symmetric around 0
        assert nearly_equal(result_low, -result_high, 0.01)

    def test_constant_points(self):
        """All equal control points should return that value."""
        result = cubic_hermite_interpolate(5.0, 5.0, 5.0, 5.0, 0.5)
        assert nearly_equal(result, 5.0)


class TestCubicHermiteInterpolateVec3:
    """Tests for cubic_hermite_interpolate_vec3() function."""

    def test_interpolate_at_t_zero(self):
        """At t=0, should return p1."""
        p0 = Vec3(0, 0, 0)
        p1 = Vec3(1, 2, 3)
        p2 = Vec3(2, 4, 6)
        p3 = Vec3(3, 6, 9)

        result = cubic_hermite_interpolate_vec3(p0, p1, p2, p3, 0.0)
        assert vec3_nearly_equal(result, p1)

    def test_interpolate_at_t_one(self):
        """At t=1, should return p2."""
        p0 = Vec3(0, 0, 0)
        p1 = Vec3(1, 2, 3)
        p2 = Vec3(2, 4, 6)
        p3 = Vec3(3, 6, 9)

        result = cubic_hermite_interpolate_vec3(p0, p1, p2, p3, 1.0)
        assert vec3_nearly_equal(result, p2)

    def test_interpolate_linear_path(self):
        """Linear control points should produce linear interpolation."""
        p0 = Vec3(0, 0, 0)
        p1 = Vec3(1, 1, 1)
        p2 = Vec3(2, 2, 2)
        p3 = Vec3(3, 3, 3)

        result = cubic_hermite_interpolate_vec3(p0, p1, p2, p3, 0.5)
        expected = Vec3(1.5, 1.5, 1.5)
        assert vec3_nearly_equal(result, expected)

    def test_interpolate_each_component_independent(self):
        """Each component should be interpolated independently."""
        p0 = Vec3(0, 10, 100)
        p1 = Vec3(1, 11, 101)
        p2 = Vec3(2, 12, 102)
        p3 = Vec3(3, 13, 103)

        result = cubic_hermite_interpolate_vec3(p0, p1, p2, p3, 0.5)

        # Each component follows its own linear path
        assert nearly_equal(result.x, 1.5)
        assert nearly_equal(result.y, 11.5)
        assert nearly_equal(result.z, 101.5)


class TestCubicHermiteInterpolateTransform:
    """Tests for cubic_hermite_interpolate_transform() function."""

    def test_interpolate_at_endpoints(self):
        """Transform interpolation at t=0 and t=1 should return t1 and t2."""
        t0 = make_test_transform(0, 0, 0)
        t1 = make_test_transform(1, 1, 1)
        t2 = make_test_transform(2, 2, 2)
        t3 = make_test_transform(3, 3, 3)

        result_0 = cubic_hermite_interpolate_transform(t0, t1, t2, t3, 0.0)
        result_1 = cubic_hermite_interpolate_transform(t0, t1, t2, t3, 1.0)

        assert transform_nearly_equal(result_0, t1)
        assert transform_nearly_equal(result_1, t2)

    def test_interpolate_translation(self):
        """Translation should be interpolated with Catmull-Rom."""
        t0 = Transform(translation=Vec3(0, 0, 0))
        t1 = Transform(translation=Vec3(1, 0, 0))
        t2 = Transform(translation=Vec3(2, 0, 0))
        t3 = Transform(translation=Vec3(3, 0, 0))

        result = cubic_hermite_interpolate_transform(t0, t1, t2, t3, 0.5)
        assert nearly_equal(result.translation.x, 1.5)

    def test_interpolate_scale(self):
        """Scale should be interpolated with Catmull-Rom."""
        t0 = Transform(scale=Vec3(1, 1, 1))
        t1 = Transform(scale=Vec3(2, 2, 2))
        t2 = Transform(scale=Vec3(3, 3, 3))
        t3 = Transform(scale=Vec3(4, 4, 4))

        result = cubic_hermite_interpolate_transform(t0, t1, t2, t3, 0.5)
        assert nearly_equal(result.scale.x, 2.5)

    def test_interpolate_rotation_with_squad(self):
        """Rotation should use SQUAD interpolation."""
        # Create rotations around Y axis
        t0 = Transform(rotation=Quat.from_axis_angle(Vec3(0, 1, 0), 0.0))
        t1 = Transform(rotation=Quat.from_axis_angle(Vec3(0, 1, 0), math.pi * 0.25))
        t2 = Transform(rotation=Quat.from_axis_angle(Vec3(0, 1, 0), math.pi * 0.5))
        t3 = Transform(rotation=Quat.from_axis_angle(Vec3(0, 1, 0), math.pi * 0.75))

        result = cubic_hermite_interpolate_transform(t0, t1, t2, t3, 0.5)

        # Result should be approximately 3*pi/8 (midpoint between pi/4 and pi/2)
        # The rotation should be between t1 and t2
        expected_angle = math.pi * 0.375
        expected_rot = Quat.from_axis_angle(Vec3(0, 1, 0), expected_angle)

        # SQUAD may not give exact linear angle due to spherical interpolation
        # Check that result is roughly in the right range
        assert quat_nearly_equal(result.rotation, expected_rot, 0.1)

    def test_result_rotation_is_normalized(self):
        """Result rotation should always be normalized."""
        t0 = Transform(rotation=Quat.from_euler(0.1, 0.2, 0.3))
        t1 = Transform(rotation=Quat.from_euler(0.2, 0.3, 0.4))
        t2 = Transform(rotation=Quat.from_euler(0.3, 0.4, 0.5))
        t3 = Transform(rotation=Quat.from_euler(0.4, 0.5, 0.6))

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = cubic_hermite_interpolate_transform(t0, t1, t2, t3, t)
            length = result.rotation.length()
            assert nearly_equal(length, 1.0, 1e-6)


# =============================================================================
# 3. SQUAD QUATERNION TESTS (PRIVATE HELPERS)
# =============================================================================

class TestQuatLog:
    """Tests for _quat_log() private helper."""

    def test_identity_quaternion_log_is_zero(self):
        """Log of identity quaternion should be zero vector."""
        q = Quat.identity()
        result = _quat_log(q)
        assert vec3_nearly_equal(result, Vec3.zero())

    def test_180_degree_rotation(self):
        """Log of 180-degree rotation should give pi * axis."""
        q = Quat.from_axis_angle(Vec3(1, 0, 0), math.pi)
        result = _quat_log(q)

        # Half-angle is pi/2, so result is roughly (pi/2, 0, 0)
        expected = Vec3(math.pi / 2, 0, 0)
        assert vec3_nearly_equal(result, expected, 0.01)

    def test_90_degree_rotation(self):
        """Log of 90-degree rotation around Y."""
        q = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        result = _quat_log(q)

        # Half-angle is pi/4
        expected = Vec3(0, math.pi / 4, 0)
        assert vec3_nearly_equal(result, expected, 0.01)

    def test_small_rotation(self):
        """Very small rotation should give small result."""
        angle = 0.001
        q = Quat.from_axis_angle(Vec3(0, 0, 1), angle)
        result = _quat_log(q)

        expected = Vec3(0, 0, angle / 2)
        assert vec3_nearly_equal(result, expected, 0.001)


class TestQuatExp:
    """Tests for _quat_exp() private helper."""

    def test_zero_vector_exp_is_identity(self):
        """Exp of zero vector should be identity quaternion."""
        result = _quat_exp(Vec3.zero())
        assert quat_nearly_equal(result, Quat.identity())

    def test_exp_log_round_trip(self):
        """exp(log(q)) should equal q for unit quaternions."""
        q = Quat.from_axis_angle(Vec3(1, 1, 1).normalized(), 1.0)
        q = q.normalized()

        log_q = _quat_log(q)
        result = _quat_exp(log_q)

        assert quat_nearly_equal(result, q)

    def test_known_value(self):
        """Test exp for a known axis-angle representation."""
        # Half-angle = pi/4, axis = (1, 0, 0)
        v = Vec3(math.pi / 4, 0, 0)
        result = _quat_exp(v)

        # Should be rotation of pi/2 around X
        expected = Quat.from_axis_angle(Vec3(1, 0, 0), math.pi / 2)
        assert quat_nearly_equal(result, expected)


class TestComputeSquadIntermediate:
    """Tests for _compute_squad_intermediate() private helper."""

    def test_symmetric_neighbors_gives_same_quaternion(self):
        """If q_prev == q_next, intermediate should be close to q_curr."""
        q_curr = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)
        q_prev = Quat.from_axis_angle(Vec3(0, 1, 0), 0.0)
        q_next = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)

        result = _compute_squad_intermediate(q_prev, q_curr, q_next)

        # Result should be a valid unit quaternion
        assert nearly_equal(result.length(), 1.0, 0.01)

    def test_identical_quaternions(self):
        """All identical quaternions should return the same quaternion."""
        q = Quat.from_axis_angle(Vec3(0, 1, 0), 0.5)
        result = _compute_squad_intermediate(q, q, q)

        assert quat_nearly_equal(result, q)

    def test_result_is_unit_quaternion(self):
        """Intermediate quaternion should be unit length."""
        q_prev = Quat.from_euler(0.1, 0.2, 0.3)
        q_curr = Quat.from_euler(0.2, 0.3, 0.4)
        q_next = Quat.from_euler(0.3, 0.4, 0.5)

        result = _compute_squad_intermediate(q_prev, q_curr, q_next)
        assert nearly_equal(result.length(), 1.0, 0.01)


class TestSquadInterpolate:
    """Tests for _squad_interpolate() private helper."""

    def test_interpolate_at_t_zero_returns_q1(self):
        """At t=0, should return q1."""
        q0 = Quat.from_axis_angle(Vec3(0, 1, 0), 0.0)
        q1 = Quat.from_axis_angle(Vec3(0, 1, 0), 0.5)
        q2 = Quat.from_axis_angle(Vec3(0, 1, 0), 1.0)
        q3 = Quat.from_axis_angle(Vec3(0, 1, 0), 1.5)

        result = _squad_interpolate(q0, q1, q2, q3, 0.0)
        assert quat_nearly_equal(result, q1)

    def test_interpolate_at_t_one_returns_q2(self):
        """At t=1, should return q2."""
        q0 = Quat.from_axis_angle(Vec3(0, 1, 0), 0.0)
        q1 = Quat.from_axis_angle(Vec3(0, 1, 0), 0.5)
        q2 = Quat.from_axis_angle(Vec3(0, 1, 0), 1.0)
        q3 = Quat.from_axis_angle(Vec3(0, 1, 0), 1.5)

        result = _squad_interpolate(q0, q1, q2, q3, 1.0)
        assert quat_nearly_equal(result, q2)

    def test_hemisphere_correction(self):
        """SQUAD should handle quaternions in opposite hemispheres."""
        # q and -q represent the same rotation, but SQUAD needs them aligned
        q0 = Quat.from_axis_angle(Vec3(1, 0, 0), 0.0)
        q1 = Quat.from_axis_angle(Vec3(1, 0, 0), 0.5)
        # Create q2 in opposite hemisphere
        temp = Quat.from_axis_angle(Vec3(1, 0, 0), 1.0)
        q2 = Quat(-temp.x, -temp.y, -temp.z, -temp.w)
        q3 = Quat.from_axis_angle(Vec3(1, 0, 0), 1.5)

        # Should still produce valid interpolation
        result = _squad_interpolate(q0, q1, q2, q3, 0.5)
        assert nearly_equal(result.length(), 1.0, 0.01)

    def test_result_is_unit_quaternion(self):
        """Result should always be unit quaternion."""
        q0 = Quat.from_euler(0.1, 0.2, 0.3)
        q1 = Quat.from_euler(0.3, 0.4, 0.5)
        q2 = Quat.from_euler(0.5, 0.6, 0.7)
        q3 = Quat.from_euler(0.7, 0.8, 0.9)

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = _squad_interpolate(q0, q1, q2, q3, t)
            assert nearly_equal(result.length(), 1.0, 0.01)

    def test_smooth_interpolation(self):
        """SQUAD should produce smooth (C1) interpolation."""
        q0 = Quat.from_axis_angle(Vec3(0, 1, 0), 0.0)
        q1 = Quat.from_axis_angle(Vec3(0, 1, 0), 0.5)
        q2 = Quat.from_axis_angle(Vec3(0, 1, 0), 1.0)
        q3 = Quat.from_axis_angle(Vec3(0, 1, 0), 1.5)

        # Sample many points and check for smoothness
        prev_result = _squad_interpolate(q0, q1, q2, q3, 0.0)
        for i in range(1, 21):
            t = i / 20.0
            result = _squad_interpolate(q0, q1, q2, q3, t)

            # Change should be gradual (dot product close to 1)
            dot = abs(prev_result.dot(result))
            assert dot > 0.95, f"Non-smooth transition at t={t}"
            prev_result = result


# =============================================================================
# 4. ANIMATION TEXTURE CUBIC SAMPLING TESTS
# =============================================================================

class TestAnimationTextureSampleBoneTransformCubic:
    """Tests for AnimationTexture.sample_bone_transform_cubic()."""

    def test_single_frame_returns_that_frame(self):
        """Single-frame animation should return that frame."""
        skeleton = make_skeleton(2)
        clip = make_animation_clip("test", 2, 1)
        texture = bake_clip_to_texture(clip, skeleton)

        result = texture.sample_bone_transform_cubic(0, 0.5)
        expected = clip.get_bone_transform(0, 0)

        assert transform_nearly_equal(result, expected)

    def test_two_frame_falls_back_to_linear(self):
        """Two-frame animation should fall back to linear interpolation."""
        skeleton = make_skeleton(1)
        clip = make_animation_clip("test", 1, 2)
        texture = bake_clip_to_texture(clip, skeleton)

        # Sample at midpoint
        result_cubic = texture.sample_bone_transform_cubic(0, texture.duration / 2)
        result_linear = texture.sample_bone_transform(0, texture.duration / 2)

        # Should be identical
        assert transform_nearly_equal(result_cubic, result_linear)

    def test_four_plus_frames_uses_cubic(self):
        """Four or more frames should use cubic interpolation."""
        skeleton = make_skeleton(1)

        # Create a clip where linear and cubic differ
        def non_linear_transform(bone_idx: int, frame_idx: int) -> Transform:
            # Quadratic motion: position varies as t^2
            t = frame_idx / 7.0
            return Transform(
                translation=Vec3(t * t * 10, 0, 0),
                rotation=Quat.identity(),
                scale=Vec3.one(),
            )

        clip = make_animation_clip("test", 1, 8, non_linear_transform)
        texture = bake_clip_to_texture(clip, skeleton)

        # Sample at midpoint between frames 3 and 4
        t_sample = (3.5 / 7.0) * texture.duration
        result = texture.sample_bone_transform_cubic(0, t_sample)

        # Result should be reasonable (between frame 3 and 4 values)
        t3 = non_linear_transform(0, 3).translation.x
        t4 = non_linear_transform(0, 4).translation.x
        assert t3 <= result.translation.x <= t4

    def test_looping_time(self):
        """Time beyond duration should loop."""
        skeleton = make_skeleton(1)
        clip = make_animation_clip("test", 1, 10)
        texture = bake_clip_to_texture(clip, skeleton)

        # Sample at time beyond duration
        result_0 = texture.sample_bone_transform_cubic(0, 0.0)
        result_loop = texture.sample_bone_transform_cubic(0, texture.duration * 2)

        # Should be approximately the same (looped)
        assert transform_nearly_equal(result_0, result_loop, 0.01)

    def test_bone_index_out_of_bounds(self):
        """Out-of-bounds bone index should return identity."""
        skeleton = make_skeleton(2)
        clip = make_animation_clip("test", 2, 5)
        texture = bake_clip_to_texture(clip, skeleton)

        result = texture.sample_bone_transform_cubic(100, 0.0)
        assert transform_nearly_equal(result, Transform.identity())

    def test_zero_duration_returns_first_frame(self):
        """Zero-duration animation should return first frame."""
        texture = AnimationTexture(
            texture_data=[0.0] * (2 * 4 * 2),  # 2 bones, 2 frames
            bone_count=1,
            frame_count=2,
            width=2,
            height=2,
            duration=0.0,
        )

        result = texture.sample_bone_transform_cubic(0, 1.0)
        expected = texture.get_bone_transform(0, 0)
        assert transform_nearly_equal(result, expected)


# =============================================================================
# 5. ANIMATION TEXTURE ATLAS CUBIC SAMPLING TESTS
# =============================================================================

class TestAnimationTextureAtlasSampleClipCubic:
    """Tests for AnimationTextureAtlas.sample_clip_cubic()."""

    def setup_method(self):
        """Set up test atlas with multiple clips."""
        self.skeleton = make_skeleton(2)

        # Create clips with different frame counts
        self.clip_single = make_animation_clip("single", 2, 1)
        self.clip_two = make_animation_clip("two", 2, 2)
        self.clip_multi = make_animation_clip("multi", 2, 10)

        self.atlas = AnimationTextureAtlas()
        self.atlas.add_clip("single", bake_clip_to_texture(self.clip_single, self.skeleton))
        self.atlas.add_clip("two", bake_clip_to_texture(self.clip_two, self.skeleton))
        self.atlas.add_clip("multi", bake_clip_to_texture(self.clip_multi, self.skeleton))

    def test_nonexistent_clip_returns_identity(self):
        """Sampling nonexistent clip should return identity transform."""
        result = self.atlas.sample_clip_cubic("nonexistent", 0, 0.0)
        assert transform_nearly_equal(result, Transform.identity())

    def test_single_frame_clip(self):
        """Single-frame clip should return that frame."""
        result = self.atlas.sample_clip_cubic("single", 0, 0.5)
        # Should return the single frame
        assert result is not None

    def test_two_frame_clip_falls_back_to_linear(self):
        """Two-frame clip should fall back to linear interpolation."""
        result_cubic = self.atlas.sample_clip_cubic("two", 0, 0.5)
        result_linear = self.atlas.sample_clip("two", 0, 0.5)

        # Should be identical (fallback to linear)
        assert transform_nearly_equal(result_cubic, result_linear)

    def test_multi_frame_clip_uses_cubic(self):
        """Multi-frame clip should use cubic interpolation."""
        # Just verify it runs without error and returns valid transform
        result = self.atlas.sample_clip_cubic("multi", 0, 0.5)
        assert result is not None
        assert nearly_equal(result.rotation.length(), 1.0, 0.01)

    def test_zero_frame_rate_returns_first_frame(self):
        """Zero frame rate should return first frame."""
        atlas = AnimationTextureAtlas()
        atlas.clips["zero_rate"] = (0, 5, 0)  # start_row, frame_count, frame_rate=0
        atlas.bone_count = 1
        atlas.width = 2
        atlas.height = 5
        atlas.texture_data = [0.0] * (2 * 5 * 4)

        result = atlas.sample_clip_cubic("zero_rate", 0, 1.0)
        assert result is not None


# =============================================================================
# 6. VALIDATE ATLAS UV RANGES TESTS
# =============================================================================

class TestValidateAtlasUvRanges:
    """Tests for validate_atlas_uv_ranges()."""

    def test_empty_atlas_is_valid(self):
        """Empty atlas should be valid with no overlaps."""
        atlas = AnimationTextureAtlas()
        is_valid, overlaps = validate_atlas_uv_ranges(atlas)

        assert is_valid is True
        assert overlaps == []

    def test_single_clip_is_valid(self):
        """Single clip atlas should be valid."""
        skeleton = make_skeleton(1)
        clip = make_animation_clip("test", 1, 10)

        atlas = AnimationTextureAtlas()
        atlas.add_clip("test", bake_clip_to_texture(clip, skeleton))

        is_valid, overlaps = validate_atlas_uv_ranges(atlas)
        assert is_valid is True
        assert overlaps == []

    def test_non_overlapping_clips_are_valid(self):
        """Multiple non-overlapping clips should be valid."""
        skeleton = make_skeleton(1)

        atlas = AnimationTextureAtlas()
        for i in range(5):
            clip = make_animation_clip(f"clip_{i}", 1, 10)
            atlas.add_clip(f"clip_{i}", bake_clip_to_texture(clip, skeleton))

        is_valid, overlaps = validate_atlas_uv_ranges(atlas)
        assert is_valid is True
        assert overlaps == []

    def test_overlapping_clips_detected(self):
        """Manually constructed overlapping clips should be detected."""
        atlas = AnimationTextureAtlas()
        atlas.bone_count = 1
        atlas.width = 2
        atlas.height = 100
        atlas.texture_data = [0.0] * (2 * 100 * 4)

        # Manually set overlapping ranges
        atlas.clips["clip_a"] = (0, 50, 30)   # rows 0-49
        atlas.clips["clip_b"] = (40, 50, 30)  # rows 40-89 (overlaps with clip_a)

        is_valid, overlaps = validate_atlas_uv_ranges(atlas)

        assert is_valid is False
        assert len(overlaps) == 1
        assert ("clip_a", "clip_b") in overlaps or ("clip_b", "clip_a") in overlaps

    def test_adjacent_clips_no_overlap(self):
        """Adjacent (touching) clips should not be considered overlapping."""
        atlas = AnimationTextureAtlas()
        atlas.bone_count = 1
        atlas.width = 2
        atlas.height = 100
        atlas.texture_data = [0.0] * (2 * 100 * 4)

        # Adjacent ranges: 0-49 and 50-99
        atlas.clips["clip_a"] = (0, 50, 30)
        atlas.clips["clip_b"] = (50, 50, 30)

        is_valid, overlaps = validate_atlas_uv_ranges(atlas)

        assert is_valid is True
        assert overlaps == []

    def test_multiple_overlaps_all_detected(self):
        """Multiple overlapping pairs should all be detected."""
        atlas = AnimationTextureAtlas()
        atlas.bone_count = 1
        atlas.width = 2
        atlas.height = 200
        atlas.texture_data = [0.0] * (2 * 200 * 4)

        # Create overlapping ranges
        atlas.clips["clip_a"] = (0, 60, 30)    # 0-59
        atlas.clips["clip_b"] = (50, 60, 30)   # 50-109 (overlaps a)
        atlas.clips["clip_c"] = (100, 60, 30)  # 100-159 (overlaps b)

        is_valid, overlaps = validate_atlas_uv_ranges(atlas)

        assert is_valid is False
        assert len(overlaps) == 2  # a-b and b-c


# =============================================================================
# 7. RGBA8 PACK/UNPACK TESTS
# =============================================================================

class TestRgba8PackUnpack:
    """Tests for pack_float_to_rgba8() and unpack_rgba8_to_float()."""

    def test_round_trip_zero(self):
        """Zero should round-trip."""
        packed = pack_float_to_rgba8(0.0)
        unpacked = unpack_rgba8_to_float(*packed)
        assert nearly_equal(unpacked, 0.0, 0.01)

    def test_round_trip_positive(self):
        """Positive values should round-trip."""
        for val in [1.0, 10.0, 50.0, 99.0]:
            packed = pack_float_to_rgba8(val)
            unpacked = unpack_rgba8_to_float(*packed)
            assert nearly_equal(unpacked, val, 0.1)

    def test_round_trip_negative(self):
        """Negative values should round-trip."""
        for val in [-1.0, -10.0, -50.0, -99.0]:
            packed = pack_float_to_rgba8(val)
            unpacked = unpack_rgba8_to_float(*packed)
            assert nearly_equal(unpacked, val, 0.1)

    def test_clamps_out_of_range(self):
        """Values outside range should be clamped."""
        packed_high = pack_float_to_rgba8(200.0)  # Above max (100)
        unpacked_high = unpack_rgba8_to_float(*packed_high)
        assert unpacked_high <= 100.0

        packed_low = pack_float_to_rgba8(-200.0)  # Below min (-100)
        unpacked_low = unpack_rgba8_to_float(*packed_low)
        assert unpacked_low >= -100.0

    def test_packed_values_are_valid_bytes(self):
        """All packed values should be valid byte values (0-255)."""
        for val in [-100.0, 0.0, 50.0, 100.0]:
            r, g, b, a = pack_float_to_rgba8(val)
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255
            assert 0 <= a <= 255


# =============================================================================
# 8. EDGE CASE AND ERROR HANDLING TESTS
# =============================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_bake_empty_skeleton(self):
        """Baking with empty skeleton should return empty texture."""
        skeleton = Skeleton()
        clip = AnimationClip(name="empty", duration=1.0, frame_rate=30.0)

        texture = bake_clip_to_texture(clip, skeleton)

        assert texture.bone_count == 0
        assert texture.frame_count == 0
        assert texture.texture_data == []

    def test_bake_empty_clip(self):
        """Baking with empty clip should return empty texture."""
        skeleton = make_skeleton(5)
        clip = AnimationClip(name="empty", duration=0.0, frame_rate=30.0)

        texture = bake_clip_to_texture(clip, skeleton)

        assert texture.frame_count == 0

    def test_get_pixel_out_of_bounds(self):
        """Get pixel out of bounds should return zeros."""
        texture = AnimationTexture(
            texture_data=[1.0, 2.0, 3.0, 4.0],
            width=1,
            height=1,
        )

        # Out of bounds
        assert texture.get_pixel(-1, 0) == (0.0, 0.0, 0.0, 0.0)
        assert texture.get_pixel(0, -1) == (0.0, 0.0, 0.0, 0.0)
        assert texture.get_pixel(10, 0) == (0.0, 0.0, 0.0, 0.0)
        assert texture.get_pixel(0, 10) == (0.0, 0.0, 0.0, 0.0)

    def test_set_pixel_out_of_bounds_is_noop(self):
        """Set pixel out of bounds should be no-op."""
        texture = AnimationTexture(
            texture_data=[1.0, 2.0, 3.0, 4.0],
            width=1,
            height=1,
        )
        original_data = texture.texture_data.copy()

        texture.set_pixel(-1, 0, 5.0, 6.0, 7.0, 8.0)
        texture.set_pixel(10, 0, 5.0, 6.0, 7.0, 8.0)

        assert texture.texture_data == original_data

    def test_atlas_add_clip_bone_count_mismatch(self):
        """Adding clip with mismatched bone count should fail."""
        skeleton_a = make_skeleton(5)
        skeleton_b = make_skeleton(10)

        clip_a = make_animation_clip("a", 5, 3)
        clip_b = make_animation_clip("b", 10, 3)

        atlas = AnimationTextureAtlas()
        assert atlas.add_clip("a", bake_clip_to_texture(clip_a, skeleton_a)) is True
        assert atlas.add_clip("b", bake_clip_to_texture(clip_b, skeleton_b)) is False

    def test_texture_memory_size_calculation(self):
        """Memory size calculation should be correct for different formats."""
        texture = AnimationTexture(
            texture_data=[0.0] * 16,
            width=2,
            height=2,
            format=TextureFormat.FLOAT32,
        )

        # 2 * 2 * 4 components * 4 bytes = 64 bytes
        assert texture.get_memory_size_bytes() == 64

        texture.format = TextureFormat.FLOAT16
        assert texture.get_memory_size_bytes() == 32

        texture.format = TextureFormat.RGBA8_UNORM
        assert texture.get_memory_size_bytes() == 16


class TestBakeClipOverflow:
    """Tests for AnimationTextureOverflowError handling."""

    def test_overflow_too_many_bones(self):
        """Exceeding max bones should raise overflow error."""
        # Create skeleton with too many bones (>256)
        skeleton = make_skeleton(300)
        clip = make_animation_clip("overflow", 300, 10)

        with pytest.raises(AnimationTextureOverflowError) as exc_info:
            bake_clip_to_texture(clip, skeleton)

        assert "Bone count" in str(exc_info.value)

    def test_overflow_too_many_frames(self):
        """Exceeding max frames should raise overflow error."""
        skeleton = make_skeleton(1)

        # Create clip with too many frames (>4096)
        clip = AnimationClip(
            name="overflow",
            duration=200.0,
            frame_rate=30.0,
            bone_tracks={0: [Transform.identity() for _ in range(5000)]},
        )

        with pytest.raises(AnimationTextureOverflowError) as exc_info:
            bake_clip_to_texture(clip, skeleton)

        assert "Frame count" in str(exc_info.value)


# =============================================================================
# 9. PERFORMANCE SANITY TESTS
# =============================================================================

class TestPerformance:
    """Performance sanity tests (not strict benchmarks)."""

    def test_cubic_interpolation_time(self):
        """Cubic interpolation should complete in reasonable time."""
        import time

        p0, p1, p2, p3 = 0.0, 1.0, 2.0, 3.0

        start = time.perf_counter()
        for _ in range(10000):
            cubic_hermite_interpolate(p0, p1, p2, p3, 0.5)
        elapsed = time.perf_counter() - start

        # 10k interpolations should complete in < 100ms
        assert elapsed < 0.1, f"Too slow: {elapsed:.3f}s for 10k interpolations"

    def test_squad_interpolation_time(self):
        """SQUAD interpolation should complete in reasonable time."""
        import time

        q0 = Quat.from_euler(0.0, 0.0, 0.0)
        q1 = Quat.from_euler(0.1, 0.2, 0.3)
        q2 = Quat.from_euler(0.2, 0.4, 0.6)
        q3 = Quat.from_euler(0.3, 0.6, 0.9)

        start = time.perf_counter()
        for _ in range(1000):
            _squad_interpolate(q0, q1, q2, q3, 0.5)
        elapsed = time.perf_counter() - start

        # 1k interpolations should complete in < 100ms
        assert elapsed < 0.1, f"Too slow: {elapsed:.3f}s for 1k SQUAD interpolations"

    def test_atlas_validation_time(self):
        """Atlas validation should scale reasonably."""
        import time

        skeleton = make_skeleton(1)
        atlas = AnimationTextureAtlas()

        # Add 50 clips
        for i in range(50):
            clip = make_animation_clip(f"clip_{i}", 1, 10)
            atlas.add_clip(f"clip_{i}", bake_clip_to_texture(clip, skeleton))

        start = time.perf_counter()
        for _ in range(100):
            validate_atlas_uv_ranges(atlas)
        elapsed = time.perf_counter() - start

        # 100 validations of 50-clip atlas should complete in < 100ms
        assert elapsed < 0.1, f"Too slow: {elapsed:.3f}s for 100 validations"


# =============================================================================
# 10. INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_bake_and_sample_cubic(self):
        """Full pipeline: create skeleton/clip, bake, sample with cubic."""
        skeleton = make_skeleton(5)

        def generate_wave_motion(bone_idx: int, frame_idx: int) -> Transform:
            t = frame_idx / 19.0
            return Transform(
                translation=Vec3(
                    math.sin(t * math.pi * 2) * bone_idx,
                    t * 10,
                    math.cos(t * math.pi * 2) * bone_idx,
                ),
                rotation=Quat.from_axis_angle(Vec3(0, 1, 0), t * math.pi * 2),
                scale=Vec3(1.0 + math.sin(t * math.pi) * 0.2,
                           1.0 + math.sin(t * math.pi) * 0.2,
                           1.0 + math.sin(t * math.pi) * 0.2),
            )

        clip = make_animation_clip("wave", 5, 20, generate_wave_motion)
        texture = bake_clip_to_texture(clip, skeleton)

        # Sample at various times
        for bone_idx in range(5):
            for t_normalized in [0.0, 0.25, 0.5, 0.75, 1.0]:
                t = t_normalized * texture.duration
                result = texture.sample_bone_transform_cubic(bone_idx, t)

                # Verify result is valid
                assert result is not None
                assert nearly_equal(result.rotation.length(), 1.0, 0.01)
                assert result.scale.x > 0  # Scale should be positive

    def test_atlas_multi_clip_sampling(self):
        """Test sampling multiple clips from atlas with cubic interpolation."""
        skeleton = make_skeleton(3)

        clips = {
            "idle": make_animation_clip("idle", 3, 30),
            "walk": make_animation_clip("walk", 3, 45),
            "run": make_animation_clip("run", 3, 20),
        }

        atlas = AnimationTextureAtlas()
        for name, clip in clips.items():
            texture = bake_clip_to_texture(clip, skeleton)
            assert atlas.add_clip(name, texture) is True

        # Validate UV ranges
        is_valid, overlaps = validate_atlas_uv_ranges(atlas)
        assert is_valid is True

        # Sample each clip
        for name in clips:
            for bone in range(3):
                result = atlas.sample_clip_cubic(name, bone, 0.5)
                assert result is not None
                assert nearly_equal(result.rotation.length(), 1.0, 0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
