"""Tests for animation sequencer with tracks, keyframes, and timeline."""

import pytest

from engine.core.math import Quat, Transform, Vec3
from engine.tooling.animation_tools.sequencer import (
    AnimationSequencer,
    AudioTrack,
    CameraTrack,
    EventKeyframe,
    EventTrack,
    Keyframe,
    PlaybackMode,
    PropertyKeyframe,
    PropertyTrack,
    SequencerPlayback,
    SkeletalTrack,
    Timeline,
    TimelineRange,
    TrackType,
    TransformKeyframe,
    TransformTrack,
)


# =============================================================================
# KEYFRAME TESTS
# =============================================================================


class TestKeyframe:
    def test_basic_keyframe(self):
        kf = Keyframe(time=1.0, value=5)
        assert kf.time == 1.0
        assert kf.value == 5
        assert kf.interpolation == "linear"

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="time must be >= 0"):
            Keyframe(time=-1.0)

    def test_keyframe_copy(self):
        kf = Keyframe(time=2.0, value=10, interpolation="bezier")
        copy = kf.copy()
        assert copy.time == kf.time
        assert copy.value == kf.value
        assert copy is not kf


class TestTransformKeyframe:
    def test_basic_transform_keyframe(self):
        kf = TransformKeyframe(
            time=0.5,
            position=Vec3(1, 2, 3),
            rotation=Quat.identity(),
            scale=Vec3(1, 1, 1),
        )
        assert kf.time == 0.5
        assert kf.position.x == 1
        assert kf.position.y == 2
        assert kf.position.z == 3

    def test_to_transform(self):
        kf = TransformKeyframe(
            time=0.0,
            position=Vec3(5, 0, 0),
        )
        transform = kf.to_transform()
        assert transform.translation.x == 5

    def test_copy_transform_keyframe(self):
        kf = TransformKeyframe(time=1.0, position=Vec3(1, 2, 3))
        copy = kf.copy()
        assert copy.time == kf.time
        assert copy.position.x == kf.position.x
        assert copy is not kf


class TestEventKeyframe:
    def test_basic_event_keyframe(self):
        kf = EventKeyframe(
            time=1.0,
            event_name="shoot",
            event_data={"damage": 10},
        )
        assert kf.time == 1.0
        assert kf.event_name == "shoot"
        assert kf.event_data["damage"] == 10

    def test_copy_event_keyframe(self):
        kf = EventKeyframe(time=2.0, event_name="jump")
        copy = kf.copy()
        assert copy.event_name == "jump"
        assert copy is not kf


class TestPropertyKeyframe:
    def test_basic_property_keyframe(self):
        kf = PropertyKeyframe(
            time=0.5,
            property_name="alpha",
            property_value=0.5,
        )
        assert kf.property_name == "alpha"
        assert kf.property_value == 0.5

    def test_copy_property_keyframe(self):
        kf = PropertyKeyframe(time=1.0, property_name="speed", property_value=2.0)
        copy = kf.copy()
        assert copy.property_name == "speed"


# =============================================================================
# TIMELINE TESTS
# =============================================================================


class TestTimelineRange:
    def test_basic_range(self):
        r = TimelineRange(start=0.0, end=5.0)
        assert r.start == 0.0
        assert r.end == 5.0
        assert r.duration == 5.0

    def test_contains(self):
        r = TimelineRange(start=1.0, end=3.0)
        assert r.contains(1.5)
        assert r.contains(1.0)
        assert r.contains(3.0)
        assert not r.contains(0.5)
        assert not r.contains(3.5)

    def test_overlaps(self):
        r1 = TimelineRange(start=0.0, end=2.0)
        r2 = TimelineRange(start=1.0, end=3.0)
        r3 = TimelineRange(start=3.0, end=5.0)
        assert r1.overlaps(r2)
        assert not r1.overlaps(r3)

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            TimelineRange(start=-1.0, end=1.0)
        with pytest.raises(ValueError):
            TimelineRange(start=5.0, end=2.0)


class TestTimeline:
    def test_basic_timeline(self):
        timeline = Timeline(duration=10.0, frame_rate=30.0)
        assert timeline.duration == 10.0
        assert timeline.frame_rate == 30.0
        assert timeline.total_frames == 300

    def test_frame_conversion(self):
        timeline = Timeline(duration=5.0, frame_rate=24.0)
        assert timeline.time_to_frame(1.0) == 24
        assert abs(timeline.frame_to_time(24) - 1.0) < 0.001

    def test_snap_to_frame(self):
        timeline = Timeline(duration=10.0, frame_rate=30.0)
        snapped = timeline.snap_to_frame(0.55)
        # Should snap to frame 17 = 0.5666...
        assert abs(snapped - 17 / 30) < 0.001

    def test_markers(self):
        timeline = Timeline(duration=10.0)
        timeline.add_marker("start", 0.0)
        timeline.add_marker("end", 10.0)
        assert timeline.get_marker("start") == 0.0
        assert timeline.get_marker("end") == 10.0
        assert timeline.remove_marker("start")
        assert timeline.get_marker("start") is None

    def test_ranges(self):
        timeline = Timeline(duration=10.0)
        r = TimelineRange(start=2.0, end=5.0)
        timeline.add_range(r)
        assert len(timeline.ranges) == 1
        ranges_at_3 = timeline.get_ranges_at(3.0)
        assert len(ranges_at_3) == 1

    def test_loop_range(self):
        timeline = Timeline(duration=10.0)
        timeline.set_loop_range(2.0, 8.0)
        assert timeline.loop_range is not None
        assert timeline.loop_range.start == 2.0
        timeline.clear_loop_range()
        assert timeline.loop_range is None

    def test_invalid_duration_raises(self):
        with pytest.raises(ValueError):
            Timeline(duration=0)
        with pytest.raises(ValueError):
            Timeline(duration=-5)


# =============================================================================
# TRACK TESTS
# =============================================================================


class TestTransformTrack:
    def test_basic_transform_track(self):
        track = TransformTrack("Transform")
        assert track.name == "Transform"
        assert track.track_type == TrackType.TRANSFORM
        assert track.keyframe_count == 0

    def test_add_keyframes(self):
        track = TransformTrack("Track1")
        kf1 = TransformKeyframe(time=0.0, position=Vec3(0, 0, 0))
        kf2 = TransformKeyframe(time=1.0, position=Vec3(10, 0, 0))
        track.add_keyframe(kf1)
        track.add_keyframe(kf2)
        assert track.keyframe_count == 2

    def test_evaluate_interpolation(self):
        track = TransformTrack("Track1")
        track.add_keyframe(TransformKeyframe(time=0.0, position=Vec3(0, 0, 0)))
        track.add_keyframe(TransformKeyframe(time=1.0, position=Vec3(10, 0, 0)))

        transform = track.evaluate(0.5)
        assert abs(transform.translation.x - 5.0) < 0.001

    def test_muted_track(self):
        track = TransformTrack("Track1")
        track.muted = True
        assert track.muted

    def test_locked_track_prevents_modification(self):
        track = TransformTrack("Track1")
        track.locked = True
        with pytest.raises(RuntimeError):
            track.add_keyframe(TransformKeyframe(time=0.0))


class TestSkeletalTrack:
    def test_basic_skeletal_track(self):
        track = SkeletalTrack("Bone_Track", "spine_01")
        assert track.bone_name == "spine_01"
        assert track.track_type == TrackType.SKELETAL


class TestCameraTrack:
    def test_basic_camera_track(self):
        track = CameraTrack("Camera")
        assert track.track_type == TrackType.CAMERA

    def test_fov_keyframes(self):
        track = CameraTrack("Camera")
        track.add_fov_keyframe(0.0, 60.0)
        track.add_fov_keyframe(1.0, 90.0)
        assert abs(track.evaluate_fov(0.5) - 75.0) < 0.001


class TestEventTrack:
    def test_basic_event_track(self):
        track = EventTrack("Events")
        assert track.track_type == TrackType.EVENT

    def test_event_evaluation(self):
        track = EventTrack("Events")
        track.add_keyframe(EventKeyframe(time=1.0, event_name="jump"))
        # Should return event at exact time
        event = track.evaluate(1.0)
        assert event is not None
        assert event["name"] == "jump"

    def test_events_in_range(self):
        track = EventTrack("Events")
        track.add_keyframe(EventKeyframe(time=1.0, event_name="event1"))
        track.add_keyframe(EventKeyframe(time=2.0, event_name="event2"))
        track.add_keyframe(EventKeyframe(time=5.0, event_name="event3"))
        events = track.get_events_in_range(0.5, 2.5)
        assert len(events) == 2


class TestAudioTrack:
    def test_basic_audio_track(self):
        track = AudioTrack("Audio")
        assert track.track_type == TrackType.AUDIO

    def test_audio_clips(self):
        track = AudioTrack("Audio")
        track.add_audio_clip("sounds/footstep.wav", 1.0, 0.5)
        clips = track.get_clips_at(1.2)
        assert len(clips) == 1


class TestPropertyTrack:
    def test_basic_property_track(self):
        track = PropertyTrack("Alpha", "/obj/mesh", "alpha")
        assert track.target_path == "/obj/mesh"
        assert track.property_name == "alpha"

    def test_property_interpolation(self):
        track = PropertyTrack("Alpha", "/obj", "alpha")
        track.add_keyframe(PropertyKeyframe(time=0.0, property_name="alpha", property_value=0.0))
        track.add_keyframe(PropertyKeyframe(time=1.0, property_name="alpha", property_value=1.0))
        assert abs(track.evaluate(0.5) - 0.5) < 0.001


# =============================================================================
# PLAYBACK TESTS
# =============================================================================


class TestSequencerPlayback:
    def test_basic_playback(self):
        playback = SequencerPlayback()
        assert playback.current_time == 0.0
        assert not playback.is_playing

    def test_play_pause_stop(self):
        playback = SequencerPlayback()
        playback.play()
        assert playback.is_playing
        playback.pause()
        assert not playback.is_playing
        playback.current_time = 5.0
        playback.stop()
        assert playback.current_time == 0.0

    def test_update_once_mode(self):
        playback = SequencerPlayback(mode=PlaybackMode.ONCE)
        playback.play()
        playback.update(5.0, 10.0)
        assert playback.current_time == 5.0
        playback.update(6.0, 10.0)
        # Should stop at end
        assert playback.current_time == 10.0
        assert not playback.is_playing

    def test_update_loop_mode(self):
        playback = SequencerPlayback(mode=PlaybackMode.LOOP)
        playback.play()
        playback.update(12.0, 10.0)
        # Should loop
        assert playback.current_time < 10.0
        assert playback.is_playing

    def test_seek(self):
        playback = SequencerPlayback()
        playback.seek(5.0)
        assert playback.current_time == 5.0


# =============================================================================
# SEQUENCER TESTS
# =============================================================================


class TestAnimationSequencer:
    def test_basic_sequencer(self):
        seq = AnimationSequencer("Test Sequence")
        assert seq.name == "Test Sequence"
        assert seq.track_count == 0

    def test_add_tracks(self):
        seq = AnimationSequencer()
        track1 = seq.create_transform_track("Transform")
        track2 = seq.create_event_track("Events")
        assert seq.track_count == 2

    def test_remove_track(self):
        seq = AnimationSequencer()
        track = seq.create_transform_track("Transform")
        assert seq.remove_track(track)
        assert seq.track_count == 0

    def test_get_track_by_name(self):
        seq = AnimationSequencer()
        track = seq.create_transform_track("MyTrack")
        found = seq.get_track_by_name("MyTrack")
        assert found is track

    def test_get_tracks_by_type(self):
        seq = AnimationSequencer()
        seq.create_transform_track("Transform1")
        seq.create_transform_track("Transform2")
        seq.create_event_track("Events")
        transforms = seq.get_tracks_by_type(TrackType.TRANSFORM)
        assert len(transforms) == 2

    def test_evaluate_all_tracks(self):
        seq = AnimationSequencer()
        track = seq.create_transform_track("Transform")
        track.add_keyframe(TransformKeyframe(time=0.0, position=Vec3(0, 0, 0)))
        track.add_keyframe(TransformKeyframe(time=1.0, position=Vec3(10, 0, 0)))

        result = seq.evaluate(0.5)
        assert "Transform" in result
        assert abs(result["Transform"].translation.x - 5.0) < 0.001

    def test_track_selection(self):
        seq = AnimationSequencer()
        seq.create_transform_track("Track1")
        seq.create_transform_track("Track2")
        seq.select_track(0)
        assert 0 in seq.get_selected_tracks()
        seq.select_track(1, add_to_selection=True)
        assert len(seq.get_selected_tracks()) == 2
        seq.clear_selection()
        assert len(seq.get_selected_tracks()) == 0

    def test_event_callbacks(self):
        seq = AnimationSequencer()
        track = seq.create_event_track("Events")
        track.add_keyframe(EventKeyframe(time=1.0, event_name="test_event"))

        triggered = []
        seq.register_event_callback("test_event", lambda e: triggered.append(e))

        seq.playback.play()
        seq.update(1.5)  # Should trigger event
        assert len(triggered) > 0

    def test_move_track(self):
        seq = AnimationSequencer()
        seq.create_transform_track("Track1")
        seq.create_transform_track("Track2")
        assert seq.move_track(0, 1)
        assert seq.get_track(0).name == "Track2"

    def test_to_dict(self):
        seq = AnimationSequencer("MySequence")
        seq.timeline.duration = 5.0
        data = seq.to_dict()
        assert data["name"] == "MySequence"
        assert data["duration"] == 5.0
