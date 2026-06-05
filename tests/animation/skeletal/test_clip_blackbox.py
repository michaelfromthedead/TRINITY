"""
Blackbox tests for AnimationClip and keyframe operations.

Tests cover animation clip creation, keyframe sampling, interpolation modes,
and looping behavior without knowledge of implementation details.
"""

import math
import pytest


class TestAnimationCurve:
    """Tests for AnimationCurve."""

    def test_create_empty_curve(self):
        """Empty curve should have zero duration."""
        from engine.animation.skeletal.clip import AnimationCurve

        curve = AnimationCurve()
        assert curve.duration == pytest.approx(0.0)
        assert curve.keyframe_count == 0

    def test_add_keyframe(self):
        """Should add keyframe to curve."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=0.0, value=0.0))
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))

        assert curve.keyframe_count == 2
        assert curve.duration == pytest.approx(1.0)

    def test_keyframes_sorted(self):
        """Keyframes should be sorted by time."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))
        curve.add_keyframe(Keyframe(time=0.0, value=0.0))
        curve.add_keyframe(Keyframe(time=0.5, value=5.0))

        keyframes = curve.keyframes
        assert keyframes[0].time == pytest.approx(0.0)
        assert keyframes[1].time == pytest.approx(0.5)
        assert keyframes[2].time == pytest.approx(1.0)

    def test_sample_at_keyframe_time(self):
        """Sampling at exact keyframe should return that value."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=0.0, value=0.0))
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))

        assert curve.sample(0.0) == pytest.approx(0.0)
        assert curve.sample(1.0) == pytest.approx(10.0)

    def test_sample_between_keyframes(self):
        """Sampling between keyframes should interpolate."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=0.0, value=0.0))
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))

        assert curve.sample(0.5) == pytest.approx(5.0)

    def test_sample_before_first_keyframe(self):
        """Sampling before first keyframe returns first value."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=1.0, value=5.0))
        curve.add_keyframe(Keyframe(time=2.0, value=10.0))

        assert curve.sample(0.0) == pytest.approx(5.0)

    def test_sample_after_last_keyframe(self):
        """Sampling after last keyframe returns last value."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=0.0, value=0.0))
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))

        assert curve.sample(2.0) == pytest.approx(10.0)


class TestInterpolationTypes:
    """Tests for different interpolation types."""

    def test_linear_interpolation(self):
        """Linear interpolation should be linear."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe, InterpolationType

        curve = AnimationCurve(interpolation=InterpolationType.LINEAR)
        curve.add_keyframe(Keyframe(time=0.0, value=0.0))
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert curve.sample(t) == pytest.approx(t * 10.0)

    def test_step_interpolation(self):
        """Step interpolation should hold previous value."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe, InterpolationType

        curve = AnimationCurve(interpolation=InterpolationType.STEP)
        curve.add_keyframe(Keyframe(time=0.0, value=0.0))
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))

        assert curve.sample(0.5) == pytest.approx(0.0)
        assert curve.sample(0.99) == pytest.approx(0.0)

    def test_cubic_interpolation(self):
        """Cubic interpolation should smooth values."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe, InterpolationType

        curve = AnimationCurve(interpolation=InterpolationType.CUBIC)
        curve.add_keyframe(Keyframe(time=0.0, value=0.0))
        curve.add_keyframe(Keyframe(time=0.5, value=5.0))
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))

        # Values should be reasonable
        val = curve.sample(0.25)
        assert 0.0 < val < 5.0


class TestVec3Curves:
    """Tests for Vec3 animation curves."""

    def test_vec3_linear_interpolation(self):
        """Vec3 should interpolate linearly."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe
        from engine.core.math import Vec3

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=0.0, value=Vec3(0.0, 0.0, 0.0)))
        curve.add_keyframe(Keyframe(time=1.0, value=Vec3(10.0, 20.0, 30.0)))

        result = curve.sample(0.5)
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(10.0)
        assert result.z == pytest.approx(15.0)


class TestQuatCurves:
    """Tests for quaternion animation curves."""

    def test_quat_slerp_interpolation(self):
        """Quaternions should use SLERP."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe
        from engine.core.math import Quat

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=0.0, value=Quat.identity()))

        # 90 degree rotation around Y
        angle = math.pi / 2
        qw = math.cos(angle / 2)
        qy = math.sin(angle / 2)
        curve.add_keyframe(Keyframe(time=1.0, value=Quat(0.0, qy, 0.0, qw)))

        result = curve.sample(0.5)
        # Should be valid quaternion
        magnitude = math.sqrt(result.x**2 + result.y**2 + result.z**2 + result.w**2)
        assert magnitude == pytest.approx(1.0, abs=1e-5)


class TestAnimationEvent:
    """Tests for animation events."""

    def test_create_event(self):
        """Should create event with time and name."""
        from engine.animation.skeletal.clip import AnimationEvent

        event = AnimationEvent(time=0.5, name="footstep")
        assert event.time == pytest.approx(0.5)
        assert event.name == "footstep"

    def test_event_with_data(self):
        """Should store event data."""
        from engine.animation.skeletal.clip import AnimationEvent

        event = AnimationEvent(
            time=0.5,
            name="sound",
            data={"sound_id": "step_01"}
        )
        assert event.data["sound_id"] == "step_01"

    def test_event_negative_time_raises(self):
        """Event time must be >= 0."""
        from engine.animation.skeletal.clip import AnimationEvent

        with pytest.raises(ValueError):
            AnimationEvent(time=-1.0, name="invalid")

    def test_event_empty_name_raises(self):
        """Event name cannot be empty."""
        from engine.animation.skeletal.clip import AnimationEvent

        with pytest.raises(ValueError):
            AnimationEvent(time=0.0, name="")


class TestBoneTrack:
    """Tests for BoneTrack."""

    def test_create_bone_track(self):
        """Should create track for bone index."""
        from engine.animation.skeletal.clip import BoneTrack

        track = BoneTrack(bone_index=0)
        assert track.bone_index == 0

    def test_bone_track_with_curves(self):
        """Should accept position, rotation, scale curves."""
        from engine.animation.skeletal.clip import BoneTrack, AnimationCurve, Keyframe
        from engine.core.math import Vec3, Quat

        pos_curve = AnimationCurve()
        pos_curve.add_keyframe(Keyframe(time=0.0, value=Vec3(0.0, 0.0, 0.0)))

        rot_curve = AnimationCurve()
        rot_curve.add_keyframe(Keyframe(time=0.0, value=Quat.identity()))

        scale_curve = AnimationCurve()
        scale_curve.add_keyframe(Keyframe(time=0.0, value=Vec3(1.0, 1.0, 1.0)))

        track = BoneTrack(
            bone_index=0,
            position_curve=pos_curve,
            rotation_curve=rot_curve,
            scale_curve=scale_curve
        )

        assert track.has_position()
        assert track.has_rotation()
        assert track.has_scale()

    def test_bone_track_sample(self):
        """Should sample all curves."""
        from engine.animation.skeletal.clip import BoneTrack, AnimationCurve, Keyframe
        from engine.core.math import Vec3

        pos_curve = AnimationCurve()
        pos_curve.add_keyframe(Keyframe(time=0.0, value=Vec3(0.0, 0.0, 0.0)))
        pos_curve.add_keyframe(Keyframe(time=1.0, value=Vec3(10.0, 0.0, 0.0)))

        track = BoneTrack(bone_index=0, position_curve=pos_curve)

        pos = track.sample_position(0.5)
        assert pos.x == pytest.approx(5.0)

    def test_bone_track_duration(self):
        """Duration should be max of all curves."""
        from engine.animation.skeletal.clip import BoneTrack, AnimationCurve, Keyframe
        from engine.core.math import Vec3

        pos_curve = AnimationCurve()
        pos_curve.add_keyframe(Keyframe(time=0.0, value=Vec3.zero()))
        pos_curve.add_keyframe(Keyframe(time=1.0, value=Vec3.zero()))

        scale_curve = AnimationCurve()
        scale_curve.add_keyframe(Keyframe(time=0.0, value=Vec3.one()))
        scale_curve.add_keyframe(Keyframe(time=2.0, value=Vec3.one()))

        track = BoneTrack(
            bone_index=0,
            position_curve=pos_curve,
            scale_curve=scale_curve
        )

        assert track.duration == pytest.approx(2.0)


class TestAnimationClipCreation:
    """Tests for AnimationClip object creation."""

    def test_create_clip_with_name(self):
        """Clip should store its name."""
        from engine.animation.skeletal.clip import AnimationClip

        clip = AnimationClip(name="walk")
        assert clip.name == "walk"

    def test_clip_with_framerate(self):
        """Clip should accept framerate setting."""
        from engine.animation.skeletal.clip import AnimationClip

        clip = AnimationClip(name="run", framerate=60.0)
        assert clip.framerate == pytest.approx(60.0)

    def test_clip_empty_name_raises(self):
        """Clip name cannot be empty."""
        from engine.animation.skeletal.clip import AnimationClip

        with pytest.raises(ValueError):
            AnimationClip(name="")

    def test_clip_invalid_framerate_raises(self):
        """Clip framerate must be > 0."""
        from engine.animation.skeletal.clip import AnimationClip

        with pytest.raises(ValueError):
            AnimationClip(name="test", framerate=0.0)


class TestAnimationClipTracks:
    """Tests for clip track management."""

    def test_add_bone_track(self):
        """Should add bone track to clip."""
        from engine.animation.skeletal.clip import AnimationClip, BoneTrack

        clip = AnimationClip(name="test")
        track = BoneTrack(bone_index=0)
        clip.add_bone_track(track)

        assert 0 in clip.bone_tracks

    def test_get_bone_track(self):
        """Should retrieve bone track."""
        from engine.animation.skeletal.clip import AnimationClip, BoneTrack

        clip = AnimationClip(name="test")
        track = BoneTrack(bone_index=0)
        clip.add_bone_track(track)

        retrieved = clip.get_bone_track(0)
        assert retrieved is not None
        assert retrieved.bone_index == 0

    def test_multiple_bone_tracks(self):
        """Should support multiple bone tracks."""
        from engine.animation.skeletal.clip import AnimationClip, BoneTrack

        clip = AnimationClip(name="test")
        clip.add_bone_track(BoneTrack(bone_index=0))
        clip.add_bone_track(BoneTrack(bone_index=1))
        clip.add_bone_track(BoneTrack(bone_index=2))

        assert len(clip.bone_tracks) == 3


class TestAnimationClipDuration:
    """Tests for clip duration."""

    def test_explicit_duration(self):
        """Clip should accept explicit duration."""
        from engine.animation.skeletal.clip import AnimationClip

        clip = AnimationClip(name="test", duration=5.0)
        assert clip.duration == pytest.approx(5.0)

    def test_auto_duration_from_tracks(self):
        """Duration should auto-calculate from tracks."""
        from engine.animation.skeletal.clip import AnimationClip, BoneTrack, AnimationCurve, Keyframe
        from engine.core.math import Vec3

        clip = AnimationClip(name="test")

        pos_curve = AnimationCurve()
        pos_curve.add_keyframe(Keyframe(time=0.0, value=Vec3.zero()))
        pos_curve.add_keyframe(Keyframe(time=2.5, value=Vec3.zero()))

        track = BoneTrack(bone_index=0, position_curve=pos_curve)
        clip.add_bone_track(track)

        assert clip.duration == pytest.approx(2.5)


class TestAnimationClipEvents:
    """Tests for clip events."""

    def test_add_event(self):
        """Should add event to clip."""
        from engine.animation.skeletal.clip import AnimationClip, AnimationEvent

        clip = AnimationClip(name="test")
        event = AnimationEvent(time=0.5, name="footstep")
        clip.add_event(event)

        events = clip.events
        assert len(events) >= 1

    def test_get_events_in_range(self):
        """Should get events in time range."""
        from engine.animation.skeletal.clip import AnimationClip, AnimationEvent

        clip = AnimationClip(name="test")
        clip.add_event(AnimationEvent(time=0.0, name="start"))
        clip.add_event(AnimationEvent(time=0.5, name="mid"))
        clip.add_event(AnimationEvent(time=1.0, name="end"))

        events = clip.get_events_in_range(0.25, 0.75)
        event_names = [e.name for e in events]
        assert "mid" in event_names


class TestAnimationClipLooping:
    """Tests for looping behavior."""

    def test_clip_looping_flag(self):
        """Clip should store looping flag."""
        from engine.animation.skeletal.clip import AnimationClip

        clip = AnimationClip(name="test", looping=True)
        assert clip.looping

        clip.looping = False
        assert not clip.looping


class TestAnimationClipRootMotion:
    """Tests for root motion flag."""

    def test_clip_root_motion_flag(self):
        """Clip should store root motion flag."""
        from engine.animation.skeletal.clip import AnimationClip

        clip = AnimationClip(name="test", root_motion=True)
        assert clip.root_motion

        clip.root_motion = False
        assert not clip.root_motion


class TestKeyframeValidation:
    """Tests for keyframe validation."""

    def test_keyframe_negative_time_raises(self):
        """Keyframe time must be >= 0."""
        from engine.animation.skeletal.clip import Keyframe

        with pytest.raises(ValueError):
            Keyframe(time=-1.0, value=0.0)

    def test_keyframe_copy(self):
        """Keyframe copy should be independent."""
        from engine.animation.skeletal.clip import Keyframe

        original = Keyframe(time=1.0, value=10.0)
        copy = original.copy()

        assert copy.time == pytest.approx(1.0)
        assert copy.value == pytest.approx(10.0)


class TestAnimationCurveRemoval:
    """Tests for removing keyframes."""

    def test_remove_keyframe_by_index(self):
        """Should remove keyframe by index."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=0.0, value=0.0))
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))

        curve.remove_keyframe(0)
        assert curve.keyframe_count == 1

    def test_remove_keyframe_at_time(self):
        """Should remove keyframe at time."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=0.0, value=0.0))
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))

        removed = curve.remove_keyframe_at_time(0.0)
        assert removed
        assert curve.keyframe_count == 1


class TestAnimationCurveCopy:
    """Tests for curve copying."""

    def test_curve_copy(self):
        """Curve copy should be independent."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe

        original = AnimationCurve()
        original.add_keyframe(Keyframe(time=0.0, value=0.0))
        original.add_keyframe(Keyframe(time=1.0, value=10.0))

        copy = original.copy()
        copy.add_keyframe(Keyframe(time=2.0, value=20.0))

        assert original.keyframe_count == 2
        assert copy.keyframe_count == 3


class TestBoneTrackSamplingDefaults:
    """Tests for bone track sampling with defaults."""

    def test_sample_position_with_default(self):
        """Should return default when no curve."""
        from engine.animation.skeletal.clip import BoneTrack
        from engine.core.math import Vec3

        track = BoneTrack(bone_index=0)

        result = track.sample_position(0.5, default=Vec3(1.0, 2.0, 3.0))
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)

    def test_sample_rotation_with_default(self):
        """Should return default when no curve."""
        from engine.animation.skeletal.clip import BoneTrack
        from engine.core.math import Quat

        track = BoneTrack(bone_index=0)

        result = track.sample_rotation(0.5)
        # Should return identity by default
        assert result.w == pytest.approx(1.0, abs=1e-5)

    def test_sample_scale_with_default(self):
        """Should return default when no curve."""
        from engine.animation.skeletal.clip import BoneTrack

        track = BoneTrack(bone_index=0)

        result = track.sample_scale(0.5)
        # Should return (1, 1, 1) by default
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(1.0)
        assert result.z == pytest.approx(1.0)


class TestBoneTrackCopy:
    """Tests for bone track copying."""

    def test_bone_track_copy(self):
        """Track copy should be independent."""
        from engine.animation.skeletal.clip import BoneTrack, AnimationCurve, Keyframe
        from engine.core.math import Vec3

        pos_curve = AnimationCurve()
        pos_curve.add_keyframe(Keyframe(time=0.0, value=Vec3.zero()))

        original = BoneTrack(bone_index=0, position_curve=pos_curve)
        copy = original.copy()

        assert copy.bone_index == 0
        assert copy.has_position()


class TestAnimationClipCopy:
    """Tests for clip copying."""

    def test_clip_copy(self):
        """Clip copy should be independent."""
        from engine.animation.skeletal.clip import AnimationClip, BoneTrack

        original = AnimationClip(name="original")
        original.add_bone_track(BoneTrack(bone_index=0))

        copy = original.copy()

        assert copy.name == "original"
        assert 0 in copy.bone_tracks


class TestFindKeyframeIndices:
    """Tests for finding keyframe indices."""

    def test_find_indices_before_start(self):
        """Should find indices before first keyframe."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))
        curve.add_keyframe(Keyframe(time=2.0, value=20.0))

        # Time before first keyframe
        prev, next_idx = curve.find_keyframe_indices(0.0)
        assert prev == 0
        assert next_idx == 0

    def test_find_indices_at_end(self):
        """Should find indices at end."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=0.0, value=0.0))
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))

        # Time at or after last keyframe
        prev, next_idx = curve.find_keyframe_indices(2.0)
        assert prev == 1
        assert next_idx == 1

    def test_find_indices_between(self):
        """Should find surrounding indices."""
        from engine.animation.skeletal.clip import AnimationCurve, Keyframe

        curve = AnimationCurve()
        curve.add_keyframe(Keyframe(time=0.0, value=0.0))
        curve.add_keyframe(Keyframe(time=1.0, value=10.0))

        prev, next_idx = curve.find_keyframe_indices(0.5)
        assert prev == 0
        assert next_idx == 1
