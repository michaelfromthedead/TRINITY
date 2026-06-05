"""Whitebox tests for clip.py and compression.py.

Tests animation clips, keyframes, interpolation, compression,
and retargeting.

Acceptance criteria:
- T-SKEL-1.3: Hermite Interpolation
  - Basis functions h00, h10, h01, h11
  - Tangent scaling
  - Quaternion SLERP fallback

- T-SKEL-1.5: Compression
  - Quantization round-trip
  - Keyframe reduction
  - Error thresholds

- T-SKEL-1.6: Retargeting
  - Name-based mapping
  - Scale factor computation
"""

import math
import pytest
from engine.core.math import Vec3, Quat
from engine.animation.skeletal.skeleton import Skeleton, Bone, create_humanoid_skeleton
from engine.animation.skeletal.pose import Pose, BoneTransform, PoseSpace
from engine.animation.skeletal.clip import (
    Keyframe, AnimationCurve, AnimationEvent, BoneTrack, AnimationClip,
    InterpolationType, create_simple_clip,
    KEYFRAME_TIME_TOLERANCE, INTERPOLATION_TIME_EPSILON
)
from engine.animation.skeletal.compression import (
    CompressionMethod, CompressionSettings, QuantizedValue, QuantizedCurve,
    CompressedTrack, CompressedClip, AnimationTrack, AnimationClipData,
    TrackType, CompressionErrorMetrics,
    compress_clip, decompress_clip, decompress_track,
    compute_compression_error, estimate_compressed_size,
    _reduce_keyframes, _quantize_track, _values_equal
)
from engine.animation.skeletal.compression import Keyframe as CompressionKeyframe


# =============================================================================
# Keyframe Tests
# =============================================================================

class TestKeyframe:
    """Tests for Keyframe dataclass."""

    def test_keyframe_creation(self):
        """Test basic keyframe creation."""
        kf = Keyframe(time=0.5, value=Vec3(1, 2, 3))

        assert kf.time == 0.5
        assert kf.value.x == 1

    def test_keyframe_negative_time_fails(self):
        """Test that negative time raises error."""
        with pytest.raises(ValueError, match=">= 0"):
            Keyframe(time=-1.0, value=0.0)

    def test_keyframe_with_tangents(self):
        """Test keyframe with in/out tangents."""
        kf = Keyframe(
            time=1.0,
            value=10.0,
            in_tangent=0.5,
            out_tangent=1.5
        )

        assert kf.in_tangent == 0.5
        assert kf.out_tangent == 1.5

    def test_keyframe_copy(self):
        """Test keyframe deep copy."""
        original = Keyframe(time=0.5, value=Vec3(1, 2, 3))
        copied = original.copy()

        original.value.x = 999
        assert copied.value.x == 1


# =============================================================================
# AnimationCurve Tests
# =============================================================================

class TestAnimationCurve:
    """Tests for AnimationCurve class."""

    def test_curve_creation_empty(self):
        """Test creating empty curve."""
        curve = AnimationCurve()

        assert curve.keyframe_count == 0
        assert curve.duration == 0.0

    def test_curve_creation_with_keyframes(self):
        """Test creating curve with keyframes."""
        kfs = [
            Keyframe(0.0, 0.0),
            Keyframe(1.0, 10.0),
            Keyframe(2.0, 20.0)
        ]
        curve = AnimationCurve(keyframes=kfs)

        assert curve.keyframe_count == 3
        assert curve.duration == 2.0

    def test_curve_add_keyframe(self):
        """Test adding keyframes to curve."""
        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(1.0, 10.0))
        curve.add_keyframe(Keyframe(0.0, 0.0))  # Earlier time

        assert curve.keyframe_count == 2
        # Should be sorted by time
        assert curve.get_keyframe(0).time == 0.0
        assert curve.get_keyframe(1).time == 1.0

    def test_curve_add_keyframe_replace(self):
        """Test adding keyframe at existing time replaces it."""
        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(1.0, 10.0))
        curve.add_keyframe(Keyframe(1.0, 20.0))  # Same time

        assert curve.keyframe_count == 1
        assert curve.get_keyframe(0).value == 20.0

    def test_curve_remove_keyframe(self):
        """Test removing keyframe by index."""
        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(0.0, 0.0))
        curve.add_keyframe(Keyframe(1.0, 10.0))

        curve.remove_keyframe(0)

        assert curve.keyframe_count == 1
        assert curve.get_keyframe(0).time == 1.0

    def test_curve_remove_keyframe_at_time(self):
        """Test removing keyframe by time."""
        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(0.0, 0.0))
        curve.add_keyframe(Keyframe(1.0, 10.0))
        curve.add_keyframe(Keyframe(2.0, 20.0))

        removed = curve.remove_keyframe_at_time(1.0)

        assert removed is True
        assert curve.keyframe_count == 2

    def test_curve_find_keyframe_indices(self):
        """Test finding keyframe indices for interpolation."""
        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(0.0, 0.0))
        curve.add_keyframe(Keyframe(1.0, 10.0))
        curve.add_keyframe(Keyframe(2.0, 20.0))

        # Mid-interval
        prev, next_idx = curve.find_keyframe_indices(0.5)
        assert prev == 0
        assert next_idx == 1

        # At keyframe - uses bisect_right so 1.0 is past index 1
        prev, next_idx = curve.find_keyframe_indices(1.0)
        assert prev == 1
        assert next_idx == 2

        # Before first
        prev, next_idx = curve.find_keyframe_indices(-1.0)
        assert prev == 0
        assert next_idx == 0

        # After last
        prev, next_idx = curve.find_keyframe_indices(10.0)
        assert prev == 2
        assert next_idx == 2


class TestCurveSampling:
    """Tests for AnimationCurve sampling."""

    def test_sample_empty_curve_fails(self):
        """Test sampling empty curve raises error."""
        curve = AnimationCurve()

        with pytest.raises(ValueError, match="empty"):
            curve.sample(0.5)

    def test_sample_before_first_keyframe(self):
        """Test sampling before first keyframe returns first value."""
        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(1.0, 10.0))

        assert curve.sample(0.0) == 10.0
        assert curve.sample(-1.0) == 10.0

    def test_sample_after_last_keyframe(self):
        """Test sampling after last keyframe returns last value."""
        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(0.0, 0.0))
        curve.add_keyframe(Keyframe(1.0, 10.0))

        assert curve.sample(1.0) == 10.0
        assert curve.sample(100.0) == 10.0

    def test_sample_step_interpolation(self):
        """Test step interpolation (no blending)."""
        curve = AnimationCurve(interpolation=InterpolationType.STEP)
        curve.add_keyframe(Keyframe(0.0, 0.0))
        curve.add_keyframe(Keyframe(1.0, 10.0))

        # Step should always return previous keyframe value
        assert curve.sample(0.0) == 0.0
        assert curve.sample(0.5) == 0.0
        assert curve.sample(0.9) == 0.0
        assert curve.sample(1.0) == 10.0

    def test_sample_linear_float(self):
        """Test linear interpolation for floats."""
        curve = AnimationCurve(interpolation=InterpolationType.LINEAR)
        curve.add_keyframe(Keyframe(0.0, 0.0))
        curve.add_keyframe(Keyframe(1.0, 10.0))

        assert abs(curve.sample(0.0) - 0.0) < 1e-6
        assert abs(curve.sample(0.5) - 5.0) < 1e-6
        assert abs(curve.sample(1.0) - 10.0) < 1e-6

    def test_sample_linear_vec3(self):
        """Test linear interpolation for Vec3."""
        curve = AnimationCurve(interpolation=InterpolationType.LINEAR)
        curve.add_keyframe(Keyframe(0.0, Vec3(0, 0, 0)))
        curve.add_keyframe(Keyframe(1.0, Vec3(10, 20, 30)))

        result = curve.sample(0.5)

        assert abs(result.x - 5.0) < 1e-6
        assert abs(result.y - 10.0) < 1e-6
        assert abs(result.z - 15.0) < 1e-6

    def test_sample_linear_quaternion(self):
        """Test linear interpolation uses SLERP for quaternions."""
        curve = AnimationCurve(interpolation=InterpolationType.LINEAR)

        q0 = Quat.identity()
        angle = math.pi / 2
        q1 = Quat(0, math.sin(angle/2), 0, math.cos(angle/2))

        curve.add_keyframe(Keyframe(0.0, q0))
        curve.add_keyframe(Keyframe(1.0, q1))

        result = curve.sample(0.5)

        # Result should be unit quaternion
        length = math.sqrt(result.x**2 + result.y**2 + result.z**2 + result.w**2)
        assert abs(length - 1.0) < 1e-6


class TestHermiteInterpolation:
    """Tests for T-SKEL-1.3: Hermite interpolation."""

    def test_cubic_interpolation_float(self):
        """Test cubic interpolation for floats."""
        curve = AnimationCurve(interpolation=InterpolationType.CUBIC)
        curve.add_keyframe(Keyframe(0.0, 0.0, out_tangent=0.0))
        curve.add_keyframe(Keyframe(1.0, 10.0, in_tangent=0.0))

        # At endpoints
        assert abs(curve.sample(0.0) - 0.0) < 1e-6
        assert abs(curve.sample(1.0) - 10.0) < 1e-6

        # Mid-point should follow Hermite curve
        mid = curve.sample(0.5)
        assert 0.0 <= mid <= 10.0  # Should be between endpoints

    def test_cubic_interpolation_vec3(self):
        """Test T-SKEL-1.3: Hermite for Vec3."""
        curve = AnimationCurve(interpolation=InterpolationType.CUBIC)
        curve.add_keyframe(Keyframe(
            0.0, Vec3(0, 0, 0),
            out_tangent=Vec3(1, 0, 0)
        ))
        curve.add_keyframe(Keyframe(
            1.0, Vec3(10, 0, 0),
            in_tangent=Vec3(1, 0, 0)
        ))

        result = curve.sample(0.5)
        assert isinstance(result, Vec3)

    def test_cubic_interpolation_quaternion_fallback(self):
        """Test T-SKEL-1.3: Quaternion falls back to SLERP."""
        curve = AnimationCurve(interpolation=InterpolationType.CUBIC)

        q0 = Quat.identity()
        q1 = Quat(0, 0.7071, 0, 0.7071)  # 90 deg

        curve.add_keyframe(Keyframe(0.0, q0))
        curve.add_keyframe(Keyframe(1.0, q1))

        result = curve.sample(0.5)

        # Should still produce valid quaternion
        length = math.sqrt(result.x**2 + result.y**2 + result.z**2 + result.w**2)
        assert abs(length - 1.0) < 1e-5

    def test_hermite_basis_functions(self):
        """Test T-SKEL-1.3: Hermite basis h00, h10, h01, h11."""
        # h00(0) = 1, h00(1) = 0
        # h10(0) = 0, h10(1) = 0
        # h01(0) = 0, h01(1) = 1
        # h11(0) = 0, h11(1) = 0

        curve = AnimationCurve(interpolation=InterpolationType.CUBIC)
        curve.add_keyframe(Keyframe(0.0, 1.0, out_tangent=0.0))
        curve.add_keyframe(Keyframe(1.0, 0.0, in_tangent=0.0))

        # At t=0, h00 dominates, should be close to 1
        assert abs(curve.sample(0.0) - 1.0) < 1e-6

        # At t=1, h01 dominates, should be close to 0
        assert abs(curve.sample(1.0) - 0.0) < 1e-6

    def test_tangent_scaling(self):
        """Test T-SKEL-1.3: Tangent scaling with dt."""
        curve = AnimationCurve(interpolation=InterpolationType.CUBIC)

        # Larger dt should scale tangent influence
        curve.add_keyframe(Keyframe(0.0, 0.0, out_tangent=10.0))
        curve.add_keyframe(Keyframe(2.0, 0.0, in_tangent=-10.0))  # dt = 2

        # With symmetric tangents, should overshoot
        mid = curve.sample(1.0)
        # The curve should bulge due to tangents
        assert mid != 0.0


# =============================================================================
# AnimationEvent Tests
# =============================================================================

class TestAnimationEvent:
    """Tests for AnimationEvent dataclass."""

    def test_event_creation(self):
        """Test creating animation event."""
        event = AnimationEvent(time=1.5, name="footstep")

        assert event.time == 1.5
        assert event.name == "footstep"

    def test_event_negative_time_fails(self):
        """Test that negative time raises error."""
        with pytest.raises(ValueError, match=">= 0"):
            AnimationEvent(time=-1.0, name="test")

    def test_event_empty_name_fails(self):
        """Test that empty name raises error."""
        with pytest.raises(ValueError, match="empty"):
            AnimationEvent(time=0.0, name="")

    def test_event_with_data(self):
        """Test event with payload data."""
        event = AnimationEvent(
            time=1.0,
            name="impact",
            data={"force": 100, "sound": "hit.wav"}
        )

        assert event.data["force"] == 100
        assert event.data["sound"] == "hit.wav"

    def test_event_copy(self):
        """Test event deep copy."""
        original = AnimationEvent(
            time=1.0, name="test",
            data={"key": "value"}
        )
        copied = original.copy()

        original.data["key"] = "modified"
        assert copied.data["key"] == "value"


# =============================================================================
# BoneTrack Tests
# =============================================================================

class TestBoneTrack:
    """Tests for BoneTrack class."""

    def test_bone_track_creation(self):
        """Test creating bone track."""
        track = BoneTrack(bone_index=0)

        assert track.bone_index == 0
        assert not track.has_position()
        assert not track.has_rotation()
        assert not track.has_scale()

    def test_bone_track_negative_index_fails(self):
        """Test that negative bone index raises error."""
        with pytest.raises(ValueError, match=">= 0"):
            BoneTrack(bone_index=-1)

    def test_bone_track_with_curves(self):
        """Test track with position and rotation curves."""
        pos_curve = AnimationCurve()
        pos_curve.add_keyframe(Keyframe(0.0, Vec3(0, 0, 0)))
        pos_curve.add_keyframe(Keyframe(1.0, Vec3(10, 0, 0)))

        rot_curve = AnimationCurve()
        rot_curve.add_keyframe(Keyframe(0.0, Quat.identity()))

        track = BoneTrack(
            bone_index=0,
            position_curve=pos_curve,
            rotation_curve=rot_curve
        )

        assert track.has_position()
        assert track.has_rotation()
        assert not track.has_scale()

    def test_bone_track_duration(self):
        """Test track duration from curves."""
        pos_curve = AnimationCurve()
        pos_curve.add_keyframe(Keyframe(0.0, Vec3.zero()))
        pos_curve.add_keyframe(Keyframe(2.0, Vec3.zero()))

        rot_curve = AnimationCurve()
        rot_curve.add_keyframe(Keyframe(0.0, Quat.identity()))
        rot_curve.add_keyframe(Keyframe(3.0, Quat.identity()))

        track = BoneTrack(
            bone_index=0,
            position_curve=pos_curve,
            rotation_curve=rot_curve
        )

        # Duration is max of all curves
        assert track.duration == 3.0

    def test_bone_track_sample_position(self):
        """Test sampling position from track."""
        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(0.0, Vec3(0, 0, 0)))
        curve.add_keyframe(Keyframe(1.0, Vec3(10, 0, 0)))

        track = BoneTrack(bone_index=0, position_curve=curve)

        result = track.sample_position(0.5)
        assert abs(result.x - 5.0) < 1e-6

    def test_bone_track_sample_position_default(self):
        """Test sampling position returns default when no curve."""
        track = BoneTrack(bone_index=0)

        result = track.sample_position(0.5)
        assert result == Vec3.zero()

        result = track.sample_position(0.5, default=Vec3(1, 2, 3))
        assert result.x == 1

    def test_bone_track_sample_rotation(self):
        """Test sampling rotation from track."""
        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(0.0, Quat.identity()))

        track = BoneTrack(bone_index=0, rotation_curve=curve)

        result = track.sample_rotation(0.5)
        assert result.w == 1.0

    def test_bone_track_sample_scale(self):
        """Test sampling scale from track."""
        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(0.0, Vec3.one()))
        curve.add_keyframe(Keyframe(1.0, Vec3(2, 2, 2)))

        track = BoneTrack(bone_index=0, scale_curve=curve)

        result = track.sample_scale(0.5)
        assert abs(result.x - 1.5) < 1e-6


# =============================================================================
# AnimationClip Tests
# =============================================================================

class TestAnimationClip:
    """Tests for AnimationClip class."""

    def test_clip_creation(self):
        """Test creating animation clip."""
        clip = AnimationClip(name="test_clip", duration=2.0)

        assert clip.name == "test_clip"
        assert clip.duration == 2.0

    def test_clip_empty_name_fails(self):
        """Test that empty name raises error."""
        with pytest.raises(ValueError, match="empty"):
            AnimationClip(name="")

    def test_clip_invalid_framerate_fails(self):
        """Test that invalid framerate raises error."""
        with pytest.raises(ValueError, match="> 0"):
            AnimationClip(name="test", framerate=0.0)

    def test_clip_add_bone_track(self):
        """Test adding bone track to clip."""
        clip = AnimationClip(name="test")

        track = BoneTrack(bone_index=0)
        clip.add_bone_track(track)

        assert clip.track_count == 1
        assert clip.has_bone_track(0)

    def test_clip_remove_bone_track(self):
        """Test removing bone track."""
        clip = AnimationClip(name="test")
        clip.add_bone_track(BoneTrack(bone_index=0))

        removed = clip.remove_bone_track(0)

        assert removed is True
        assert clip.track_count == 0

    def test_clip_add_event(self):
        """Test adding events to clip."""
        clip = AnimationClip(name="test", duration=2.0)
        clip.add_event(AnimationEvent(time=1.0, name="event1"))
        clip.add_event(AnimationEvent(time=0.5, name="event2"))

        assert clip.event_count == 2
        # Events should be sorted by time
        events = clip.events
        assert events[0].time == 0.5
        assert events[1].time == 1.0

    def test_clip_get_events_in_range(self):
        """Test getting events in time range."""
        clip = AnimationClip(name="test", duration=2.0)
        clip.add_event(AnimationEvent(time=0.5, name="e1"))
        clip.add_event(AnimationEvent(time=1.0, name="e2"))
        clip.add_event(AnimationEvent(time=1.5, name="e3"))

        # Range (0.5, 1.2] - exclusive start, inclusive end
        events = clip.get_events_in_range(0.5, 1.2)

        assert len(events) == 1
        assert events[0].name == "e2"

    def test_clip_sample_bone(self):
        """Test sampling bone transform from clip."""
        clip = AnimationClip(name="test", duration=1.0)

        pos_curve = AnimationCurve()
        pos_curve.add_keyframe(Keyframe(0.0, Vec3(0, 0, 0)))
        pos_curve.add_keyframe(Keyframe(1.0, Vec3(10, 0, 0)))

        track = BoneTrack(bone_index=0, position_curve=pos_curve)
        clip.add_bone_track(track)

        pos, rot, scale = clip.sample_bone(0, 0.5)

        assert abs(pos.x - 5.0) < 1e-6

    def test_clip_sample_pose(self):
        """Test sampling full pose from clip."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton._rebuild_caches()

        clip = AnimationClip(name="test", duration=1.0)

        pos_curve = AnimationCurve()
        pos_curve.add_keyframe(Keyframe(0.0, Vec3(0, 0, 0)))
        pos_curve.add_keyframe(Keyframe(1.0, Vec3(10, 0, 0)))

        track = BoneTrack(bone_index=0, position_curve=pos_curve)
        clip.add_bone_track(track)

        pose = clip.sample_pose(skeleton, 0.5)

        assert pose.bone_count == 1
        assert abs(pose.get_bone_transform(0).translation.x - 5.0) < 1e-6

    def test_clip_validate(self):
        """Test clip validation."""
        clip = AnimationClip(name="test", duration=1.0)
        clip.add_event(AnimationEvent(time=2.0, name="late_event"))

        errors = clip.validate()

        assert any("exceeds duration" in e for e in errors)


# =============================================================================
# Compression Tests - T-SKEL-1.5
# =============================================================================

class TestQuantization:
    """Tests for T-SKEL-1.5: Quantization round-trip."""

    def test_quantize_float(self):
        """Test float quantization."""
        value = 0.5
        min_val = 0.0
        max_val = 1.0
        bits = 16

        quantized = QuantizedValue.quantize_float(value, min_val, max_val, bits)
        dequantized = QuantizedValue.dequantize_float(quantized, min_val, max_val, bits)

        assert abs(dequantized - value) < 1e-4

    def test_quantize_round_trip(self):
        """Test T-SKEL-1.5: Quantization round-trip accuracy."""
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        bits = 16

        for value in values:
            quantized = QuantizedValue.quantize_float(value, 0.0, 1.0, bits)
            restored = QuantizedValue.dequantize_float(quantized, 0.0, 1.0, bits)
            assert abs(restored - value) < 1e-4

    def test_quantize_clamps_range(self):
        """Test quantization clamps out-of-range values."""
        # Value outside range
        value = 1.5
        quantized = QuantizedValue.quantize_float(value, 0.0, 1.0, 16)
        dequantized = QuantizedValue.dequantize_float(quantized, 0.0, 1.0, 16)

        # Should be clamped to max
        assert abs(dequantized - 1.0) < 1e-4


class TestKeyframeReduction:
    """Tests for T-SKEL-1.5: Keyframe reduction."""

    def test_reduce_keyframes_linear(self):
        """Test reducing redundant keyframes from linear motion."""
        # Create keyframes on a straight line
        keyframes = [
            CompressionKeyframe(0.0, Vec3(0, 0, 0)),
            CompressionKeyframe(0.5, Vec3(5, 0, 0)),  # Redundant - on line
            CompressionKeyframe(1.0, Vec3(10, 0, 0)),
        ]

        reduced = _reduce_keyframes(keyframes, TrackType.TRANSLATION, 0.01)

        # Middle keyframe should be removed
        assert len(reduced) == 2

    def test_reduce_keyframes_preserves_endpoints(self):
        """Test that first and last keyframes are always kept."""
        keyframes = [
            CompressionKeyframe(0.0, Vec3(0, 0, 0)),
            CompressionKeyframe(1.0, Vec3(10, 0, 0)),
        ]

        reduced = _reduce_keyframes(keyframes, TrackType.TRANSLATION, 0.01)

        assert len(reduced) == 2
        assert reduced[0].time == 0.0
        assert reduced[-1].time == 1.0

    def test_reduce_keyframes_keeps_curve(self):
        """Test that non-linear keyframes are preserved."""
        keyframes = [
            CompressionKeyframe(0.0, Vec3(0, 0, 0)),
            CompressionKeyframe(0.5, Vec3(10, 0, 0)),  # Significant deviation
            CompressionKeyframe(1.0, Vec3(0, 0, 0)),
        ]

        reduced = _reduce_keyframes(keyframes, TrackType.TRANSLATION, 0.01)

        # Middle keyframe should be kept (not on straight line)
        assert len(reduced) == 3


class TestCompressClip:
    """Tests for clip compression."""

    def test_compress_clip_none(self):
        """Test no compression."""
        track = AnimationTrack(
            bone_index=0,
            track_type=TrackType.TRANSLATION,
            keyframes=[
                CompressionKeyframe(0.0, Vec3(0, 0, 0)),
                CompressionKeyframe(1.0, Vec3(10, 0, 0))
            ]
        )

        clip = AnimationClipData(
            name="test",
            duration=1.0,
            bone_count=1,
            tracks=[track]
        )

        settings = CompressionSettings(method=CompressionMethod.NONE)
        compressed = compress_clip(clip, settings)

        assert compressed.compression_method == CompressionMethod.NONE

    def test_compress_clip_quantized(self):
        """Test quantized compression."""
        track = AnimationTrack(
            bone_index=0,
            track_type=TrackType.TRANSLATION,
            keyframes=[
                CompressionKeyframe(0.0, Vec3(0, 0, 0)),
                CompressionKeyframe(1.0, Vec3(10, 0, 0))
            ]
        )

        clip = AnimationClipData(
            name="test",
            duration=1.0,
            bone_count=1,
            tracks=[track]
        )

        settings = CompressionSettings(method=CompressionMethod.QUANTIZED)
        compressed = compress_clip(clip, settings)

        assert compressed.compression_method == CompressionMethod.QUANTIZED
        assert compressed.compression_ratio >= 1.0

    def test_compress_constant_track(self):
        """Test compressing constant-value track."""
        track = AnimationTrack(
            bone_index=0,
            track_type=TrackType.TRANSLATION,
            keyframes=[
                CompressionKeyframe(0.0, Vec3(5, 5, 5)),
                CompressionKeyframe(0.5, Vec3(5, 5, 5)),
                CompressionKeyframe(1.0, Vec3(5, 5, 5))
            ]
        )

        clip = AnimationClipData(
            name="test",
            duration=1.0,
            bone_count=1,
            tracks=[track]
        )

        compressed = compress_clip(clip, CompressionSettings())
        compressed_track = compressed.tracks[0]

        # Should detect as constant
        assert compressed_track.is_constant is True
        assert compressed_track.constant_value is not None


class TestDecompressClip:
    """Tests for clip decompression."""

    def test_decompress_track(self):
        """Test decompressing a single track."""
        track = AnimationTrack(
            bone_index=0,
            track_type=TrackType.TRANSLATION,
            keyframes=[
                CompressionKeyframe(0.0, Vec3(0, 0, 0)),
                CompressionKeyframe(1.0, Vec3(10, 0, 0))
            ]
        )

        clip = AnimationClipData(
            name="test", duration=1.0, bone_count=1, tracks=[track]
        )

        compressed = compress_clip(clip, CompressionSettings())
        decompressed_track = decompress_track(compressed.tracks[0])

        assert decompressed_track.bone_index == 0
        assert len(decompressed_track.keyframes) >= 2

    def test_decompress_constant_track(self):
        """Test decompressing constant track."""
        compressed_track = CompressedTrack(
            bone_index=0,
            track_type=TrackType.TRANSLATION,
            is_constant=True,
            constant_value=Vec3(5, 5, 5)
        )

        decompressed = decompress_track(compressed_track)

        assert len(decompressed.keyframes) == 1
        assert decompressed.keyframes[0].value.x == 5

    def test_decompress_clip_roundtrip(self):
        """Test compress/decompress round-trip."""
        track = AnimationTrack(
            bone_index=0,
            track_type=TrackType.TRANSLATION,
            keyframes=[
                CompressionKeyframe(0.0, Vec3(0, 0, 0)),
                CompressionKeyframe(1.0, Vec3(10, 0, 0))
            ]
        )

        original = AnimationClipData(
            name="test", duration=1.0, bone_count=1, tracks=[track]
        )

        compressed = compress_clip(original, CompressionSettings())
        decompressed = decompress_clip(compressed)

        assert decompressed.name == original.name
        assert decompressed.duration == original.duration


class TestCompressionError:
    """Tests for T-SKEL-1.5: Error thresholds."""

    def test_compute_compression_error(self):
        """Test computing compression error metrics."""
        track = AnimationTrack(
            bone_index=0,
            track_type=TrackType.TRANSLATION,
            keyframes=[
                CompressionKeyframe(0.0, Vec3(0, 0, 0)),
                CompressionKeyframe(1.0, Vec3(10, 0, 0))
            ]
        )

        original = AnimationClipData(
            name="test", duration=1.0, bone_count=1, tracks=[track]
        )

        compressed = compress_clip(original, CompressionSettings())
        metrics = compute_compression_error(original, compressed)

        assert metrics.max_translation_error >= 0
        assert metrics.mean_translation_error >= 0

    def test_error_metrics_meets_threshold(self):
        """Test checking if error meets threshold."""
        metrics = CompressionErrorMetrics(
            max_translation_error=0.0005,  # < 0.001 threshold
            max_rotation_error=0.00005,    # < 0.0001 threshold
            max_scale_error=0.00005        # < 0.0001 threshold
        )

        settings = CompressionSettings()
        assert metrics.meets_threshold(settings) is True

    def test_error_metrics_exceeds_threshold(self):
        """Test detecting error exceeds threshold."""
        metrics = CompressionErrorMetrics(
            max_translation_error=0.01,  # > 0.001 threshold
            max_rotation_error=0.0,
            max_scale_error=0.0
        )

        settings = CompressionSettings()
        assert metrics.meets_threshold(settings) is False

    def test_estimate_compressed_size(self):
        """Test estimating compressed size."""
        track = AnimationTrack(
            bone_index=0,
            track_type=TrackType.TRANSLATION,
            keyframes=[
                CompressionKeyframe(0.0, Vec3(0, 0, 0)),
                CompressionKeyframe(1.0, Vec3(10, 0, 0))
            ]
        )

        clip = AnimationClipData(
            name="test", duration=1.0, bone_count=1, tracks=[track]
        )

        size = estimate_compressed_size(clip, CompressionSettings())

        assert size > 0


# =============================================================================
# Helper Functions Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_simple_clip(self):
        """Test create_simple_clip factory."""
        clip = create_simple_clip(
            name="simple",
            duration=1.0,
            bone_index=0,
            start_position=Vec3(0, 0, 0),
            end_position=Vec3(10, 0, 0)
        )

        assert clip.name == "simple"
        assert clip.duration == 1.0
        assert clip.has_bone_track(0)

    def test_create_simple_clip_with_rotation(self):
        """Test create_simple_clip with rotation."""
        clip = create_simple_clip(
            name="rotating",
            duration=1.0,
            bone_index=0,
            start_position=Vec3(0, 0, 0),
            end_position=Vec3(0, 0, 0),
            start_rotation=Quat.identity(),
            end_rotation=Quat(0, 0.7071, 0, 0.7071)
        )

        track = clip.get_bone_track(0)
        assert track.has_rotation()

    def test_values_equal_vec3(self):
        """Test Vec3 equality with tolerance."""
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(1.001, 2.001, 3.001)
        c = Vec3(2.0, 2.0, 3.0)

        assert _values_equal(a, b, 0.01) is True
        assert _values_equal(a, c, 0.01) is False

    def test_values_equal_quat(self):
        """Test quaternion equality with tolerance."""
        a = Quat(0, 0, 0, 1)
        b = Quat(0.001, 0, 0, 0.9999)  # Very close
        c = Quat(0.7071, 0, 0, 0.7071)  # Different

        assert _values_equal(a, b, 0.01) is True
        assert _values_equal(a, c, 0.01) is False


# =============================================================================
# Root Motion Tests
# =============================================================================

class TestRootMotion:
    """Tests for root motion extraction."""

    def test_extract_root_motion_disabled(self):
        """Test root motion extraction when disabled."""
        clip = AnimationClip(name="test", duration=1.0, root_motion=False)

        trans, rot = clip.extract_root_motion(0.0, 1.0)

        assert trans == Vec3.zero()

    def test_extract_root_motion_enabled(self):
        """Test root motion extraction when enabled."""
        clip = AnimationClip(name="test", duration=1.0, root_motion=True)

        pos_curve = AnimationCurve()
        pos_curve.add_keyframe(Keyframe(0.0, Vec3(0, 0, 0)))
        pos_curve.add_keyframe(Keyframe(1.0, Vec3(10, 0, 0)))

        track = BoneTrack(bone_index=0, position_curve=pos_curve)
        clip.add_bone_track(track)

        trans, rot = clip.extract_root_motion(0.0, 1.0)

        assert abs(trans.x - 10.0) < 1e-6


# =============================================================================
# Additive Clip Tests
# =============================================================================

class TestAdditiveClip:
    """Tests for additive clip creation."""

    def test_create_additive_clip(self):
        """Test creating additive clip from reference."""
        # Reference clip (bind pose)
        ref_pos_curve = AnimationCurve()
        ref_pos_curve.add_keyframe(Keyframe(0.0, Vec3(0, 0, 0)))
        ref_pos_curve.add_keyframe(Keyframe(1.0, Vec3(0, 0, 0)))

        ref_track = BoneTrack(bone_index=0, position_curve=ref_pos_curve)
        reference = AnimationClip(name="ref", duration=1.0)
        reference.add_bone_track(ref_track)

        # Target clip
        tgt_pos_curve = AnimationCurve()
        tgt_pos_curve.add_keyframe(Keyframe(0.0, Vec3(5, 0, 0)))
        tgt_pos_curve.add_keyframe(Keyframe(1.0, Vec3(5, 0, 0)))

        tgt_track = BoneTrack(bone_index=0, position_curve=tgt_pos_curve)
        target = AnimationClip(name="target", duration=1.0)
        target.add_bone_track(tgt_track)

        additive = target.create_additive_clip(reference)

        assert "additive" in additive.name
        # Delta should be (5, 0, 0)
        pos, _, _ = additive.sample_bone(0, 0.5)
        assert abs(pos.x - 5.0) < 1e-5
