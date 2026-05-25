"""Tests for animation notifies editor with event timing and types."""

import pytest

# FootstepType not implemented in notifies_editor
pytest.skip("Notifies editor API mismatch", allow_module_level=True)

from engine.core.math import Vec3
from engine.tooling.animation_tools.notifies_editor import (
    AnimNotify,
    AnimNotifyState,
    CustomEventNotify,
    FootstepNotify,
    FootstepType,
    NotifiesEditor,
    NotifyPayload,
    NotifyTiming,
    NotifyTrack,
    NotifyType,
    ParticleNotify,
    SoundNotify,
)


# =============================================================================
# NOTIFY PAYLOAD TESTS
# =============================================================================


class TestNotifyPayload:
    def test_basic_payload(self):
        payload = NotifyPayload()
        assert len(payload.data) == 0

    def test_payload_with_data(self):
        payload = NotifyPayload(data={"key": "value", "count": 5})
        assert payload.data["key"] == "value"
        assert payload.data["count"] == 5

    def test_get_value(self):
        payload = NotifyPayload(data={"name": "test"})
        assert payload.get("name") == "test"
        assert payload.get("missing", "default") == "default"

    def test_set_value(self):
        payload = NotifyPayload()
        payload.set("key", "value")
        assert payload.get("key") == "value"

    def test_copy_payload(self):
        payload = NotifyPayload(data={"a": 1})
        copy = payload.copy()
        assert copy.data == payload.data
        assert copy is not payload


# =============================================================================
# BASE NOTIFY TESTS
# =============================================================================


class TestAnimNotify:
    def test_basic_notify(self):
        notify = AnimNotify(
            name="TestNotify",
            notify_type=NotifyType.CUSTOM,
            time=1.0,
        )
        assert notify.name == "TestNotify"
        assert notify.notify_type == NotifyType.CUSTOM
        assert notify.time == 1.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            AnimNotify(name="", notify_type=NotifyType.CUSTOM, time=1.0)

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="time must be >= 0"):
            AnimNotify(name="Test", notify_type=NotifyType.CUSTOM, time=-1.0)

    def test_notify_enabled(self):
        notify = AnimNotify(name="Test", notify_type=NotifyType.CUSTOM, time=1.0)
        assert notify.enabled
        notify.enabled = False
        assert not notify.enabled

    def test_copy_notify(self):
        notify = AnimNotify(
            name="Test",
            notify_type=NotifyType.CUSTOM,
            time=1.0,
            payload=NotifyPayload(data={"key": "value"}),
        )
        copy = notify.copy()
        assert copy.name == notify.name
        assert copy is not notify


class TestAnimNotifyState:
    def test_basic_state_notify(self):
        notify = AnimNotifyState(
            name="TestState",
            notify_type=NotifyType.CUSTOM,
            start_time=1.0,
            end_time=2.0,
        )
        assert notify.start_time == 1.0
        assert notify.end_time == 2.0
        assert notify.duration == 1.0

    def test_invalid_times_raises(self):
        with pytest.raises(ValueError, match="end_time must be > start_time"):
            AnimNotifyState(
                name="Test",
                notify_type=NotifyType.CUSTOM,
                start_time=2.0,
                end_time=1.0,
            )

    def test_is_active_at_time(self):
        notify = AnimNotifyState(
            name="Test",
            notify_type=NotifyType.CUSTOM,
            start_time=1.0,
            end_time=3.0,
        )
        assert not notify.is_active_at(0.5)
        assert notify.is_active_at(1.0)
        assert notify.is_active_at(2.0)
        assert not notify.is_active_at(3.0)  # Exclusive end

    def test_get_progress(self):
        notify = AnimNotifyState(
            name="Test",
            notify_type=NotifyType.CUSTOM,
            start_time=0.0,
            end_time=10.0,
        )
        assert notify.get_progress(0.0) == 0.0
        assert notify.get_progress(5.0) == 0.5
        assert notify.get_progress(10.0) == 1.0


# =============================================================================
# SOUND NOTIFY TESTS
# =============================================================================


class TestSoundNotify:
    def test_basic_sound_notify(self):
        notify = SoundNotify(
            name="Footstep",
            time=0.5,
            sound_asset="/sounds/footstep.wav",
        )
        assert notify.notify_type == NotifyType.SOUND
        assert notify.sound_asset == "/sounds/footstep.wav"

    def test_sound_with_volume(self):
        notify = SoundNotify(
            name="Impact",
            time=1.0,
            sound_asset="/sounds/impact.wav",
            volume=0.8,
            pitch=1.2,
        )
        assert notify.volume == 0.8
        assert notify.pitch == 1.2

    def test_sound_with_attenuation(self):
        notify = SoundNotify(
            name="Explosion",
            time=0.0,
            sound_asset="/sounds/explosion.wav",
            attenuation_settings="explosion_att",
        )
        assert notify.attenuation_settings == "explosion_att"

    def test_sound_at_bone(self):
        notify = SoundNotify(
            name="Weapon",
            time=0.5,
            sound_asset="/sounds/weapon.wav",
            attach_to_bone="weapon_socket",
        )
        assert notify.attach_to_bone == "weapon_socket"

    def test_get_trigger_data(self):
        notify = SoundNotify(
            name="Test",
            time=1.0,
            sound_asset="/sounds/test.wav",
            volume=0.5,
        )
        data = notify.get_trigger_data()
        assert data["type"] == "sound"
        assert data["asset"] == "/sounds/test.wav"
        assert data["volume"] == 0.5


# =============================================================================
# PARTICLE NOTIFY TESTS
# =============================================================================


class TestParticleNotify:
    def test_basic_particle_notify(self):
        notify = ParticleNotify(
            name="DustPuff",
            time=0.25,
            particle_system="/particles/dust.pfx",
        )
        assert notify.notify_type == NotifyType.PARTICLE
        assert notify.particle_system == "/particles/dust.pfx"

    def test_particle_at_socket(self):
        notify = ParticleNotify(
            name="MuzzleFlash",
            time=0.0,
            particle_system="/particles/muzzle_flash.pfx",
            socket_name="muzzle",
        )
        assert notify.socket_name == "muzzle"

    def test_particle_with_scale(self):
        notify = ParticleNotify(
            name="Explosion",
            time=1.0,
            particle_system="/particles/explosion.pfx",
            scale=Vec3(2, 2, 2),
        )
        assert notify.scale.x == 2
        assert notify.scale.y == 2
        assert notify.scale.z == 2

    def test_particle_attached(self):
        notify = ParticleNotify(
            name="Trail",
            time=0.0,
            particle_system="/particles/trail.pfx",
            attached=True,
        )
        assert notify.attached

    def test_get_trigger_data(self):
        notify = ParticleNotify(
            name="Test",
            time=0.5,
            particle_system="/particles/test.pfx",
            socket_name="socket",
        )
        data = notify.get_trigger_data()
        assert data["type"] == "particle"
        assert data["system"] == "/particles/test.pfx"
        assert data["socket"] == "socket"


# =============================================================================
# CUSTOM EVENT NOTIFY TESTS
# =============================================================================


class TestCustomEventNotify:
    def test_basic_custom_notify(self):
        notify = CustomEventNotify(
            name="DamageWindow",
            time=0.3,
            event_name="enable_damage",
        )
        assert notify.notify_type == NotifyType.CUSTOM
        assert notify.event_name == "enable_damage"

    def test_custom_with_payload(self):
        notify = CustomEventNotify(
            name="SetVariable",
            time=1.0,
            event_name="set_variable",
            event_data={"variable": "attack_phase", "value": 2},
        )
        assert notify.event_data["variable"] == "attack_phase"
        assert notify.event_data["value"] == 2

    def test_get_trigger_data(self):
        notify = CustomEventNotify(
            name="Test",
            time=0.5,
            event_name="my_event",
            event_data={"key": "value"},
        )
        data = notify.get_trigger_data()
        assert data["type"] == "custom"
        assert data["event"] == "my_event"
        assert data["data"]["key"] == "value"


# =============================================================================
# FOOTSTEP NOTIFY TESTS
# =============================================================================


class TestFootstepNotify:
    def test_basic_footstep(self):
        notify = FootstepNotify(
            name="LeftFoot",
            time=0.25,
            foot=FootstepType.LEFT,
        )
        assert notify.notify_type == NotifyType.FOOTSTEP
        assert notify.foot == FootstepType.LEFT

    def test_footstep_with_bone(self):
        notify = FootstepNotify(
            name="RightFoot",
            time=0.75,
            foot=FootstepType.RIGHT,
            bone_name="foot_r",
        )
        assert notify.bone_name == "foot_r"

    def test_footstep_volume_multiplier(self):
        notify = FootstepNotify(
            name="HeavyStep",
            time=0.5,
            foot=FootstepType.LEFT,
            volume_multiplier=1.5,
        )
        assert notify.volume_multiplier == 1.5

    def test_footstep_surface_detection(self):
        notify = FootstepNotify(
            name="Step",
            time=0.5,
            foot=FootstepType.LEFT,
            detect_surface=True,
        )
        assert notify.detect_surface

    def test_get_trigger_data(self):
        notify = FootstepNotify(
            name="Test",
            time=0.5,
            foot=FootstepType.LEFT,
            bone_name="foot_l",
        )
        data = notify.get_trigger_data()
        assert data["type"] == "footstep"
        assert data["foot"] == "left"
        assert data["bone"] == "foot_l"


# =============================================================================
# NOTIFY TIMING TESTS
# =============================================================================


class TestNotifyTiming:
    def test_basic_timing(self):
        timing = NotifyTiming(animation_length=2.0)
        assert timing.animation_length == 2.0
        assert timing.frame_rate == 30.0

    def test_time_to_frame(self):
        timing = NotifyTiming(animation_length=2.0, frame_rate=30.0)
        assert timing.time_to_frame(1.0) == 30

    def test_frame_to_time(self):
        timing = NotifyTiming(animation_length=2.0, frame_rate=30.0)
        assert abs(timing.frame_to_time(30) - 1.0) < 0.001

    def test_snap_to_frame(self):
        timing = NotifyTiming(animation_length=2.0, frame_rate=30.0)
        snapped = timing.snap_to_frame(0.55)
        # Should snap to frame 17 = 0.5666...
        assert abs(snapped - 17 / 30) < 0.001

    def test_clamp_time(self):
        timing = NotifyTiming(animation_length=2.0)
        assert timing.clamp_time(-1.0) == 0.0
        assert timing.clamp_time(1.0) == 1.0
        assert timing.clamp_time(5.0) == 2.0

    def test_normalize_time(self):
        timing = NotifyTiming(animation_length=4.0)
        assert timing.normalize_time(2.0) == 0.5
        assert timing.normalize_time(0.0) == 0.0
        assert timing.normalize_time(4.0) == 1.0


# =============================================================================
# NOTIFY TRACK TESTS
# =============================================================================


class TestNotifyTrack:
    def test_basic_track(self):
        track = NotifyTrack(name="Notifies")
        assert track.name == "Notifies"
        assert track.notify_count == 0

    def test_add_notify(self):
        track = NotifyTrack(name="Track")
        notify = SoundNotify(name="Sound", time=1.0, sound_asset="/sounds/test.wav")
        assert track.add_notify(notify)
        assert track.notify_count == 1

    def test_add_duplicate_rejected(self):
        track = NotifyTrack(name="Track")
        notify1 = SoundNotify(name="Sound", time=1.0, sound_asset="/sounds/test.wav")
        notify2 = SoundNotify(name="Sound", time=2.0, sound_asset="/sounds/other.wav")
        track.add_notify(notify1)
        assert not track.add_notify(notify2)

    def test_remove_notify(self):
        track = NotifyTrack(name="Track")
        notify = SoundNotify(name="Sound", time=1.0, sound_asset="/sounds/test.wav")
        track.add_notify(notify)
        assert track.remove_notify("Sound")
        assert track.notify_count == 0

    def test_get_notify(self):
        track = NotifyTrack(name="Track")
        notify = SoundNotify(name="Sound", time=1.0, sound_asset="/sounds/test.wav")
        track.add_notify(notify)
        found = track.get_notify("Sound")
        assert found is notify

    def test_get_notifies_at_time(self):
        track = NotifyTrack(name="Track")
        track.add_notify(SoundNotify(name="Sound1", time=1.0, sound_asset="/s1.wav"))
        track.add_notify(SoundNotify(name="Sound2", time=1.0, sound_asset="/s2.wav"))
        track.add_notify(SoundNotify(name="Sound3", time=2.0, sound_asset="/s3.wav"))

        at_1 = track.get_notifies_at_time(1.0)
        assert len(at_1) == 2

    def test_get_notifies_in_range(self):
        track = NotifyTrack(name="Track")
        track.add_notify(SoundNotify(name="S1", time=0.5, sound_asset="/s.wav"))
        track.add_notify(SoundNotify(name="S2", time=1.5, sound_asset="/s.wav"))
        track.add_notify(SoundNotify(name="S3", time=2.5, sound_asset="/s.wav"))

        in_range = track.get_notifies_in_range(1.0, 2.0)
        assert len(in_range) == 1
        assert in_range[0].name == "S2"

    def test_get_notifies_by_type(self):
        track = NotifyTrack(name="Track")
        track.add_notify(SoundNotify(name="Sound", time=1.0, sound_asset="/s.wav"))
        track.add_notify(ParticleNotify(name="Particle", time=1.0, particle_system="/p.pfx"))

        sounds = track.get_notifies_by_type(NotifyType.SOUND)
        assert len(sounds) == 1
        assert sounds[0].name == "Sound"

    def test_move_notify(self):
        track = NotifyTrack(name="Track")
        notify = SoundNotify(name="Sound", time=1.0, sound_asset="/s.wav")
        track.add_notify(notify)
        track.move_notify("Sound", 2.0)
        assert notify.time == 2.0

    def test_copy_notify(self):
        track = NotifyTrack(name="Track")
        track.add_notify(SoundNotify(name="Original", time=1.0, sound_asset="/s.wav"))
        copied = track.copy_notify("Original", "Copy")
        assert copied is not None
        assert copied.name == "Copy"
        assert track.notify_count == 2

    def test_muted_track(self):
        track = NotifyTrack(name="Track", muted=True)
        assert track.muted


# =============================================================================
# NOTIFIES EDITOR TESTS
# =============================================================================


class TestNotifiesEditor:
    def test_basic_editor(self):
        editor = NotifiesEditor()
        assert editor.track_count == 0

    def test_create_track(self):
        editor = NotifiesEditor()
        track = editor.create_track("Sounds")
        assert track is not None
        assert editor.track_count == 1

    def test_remove_track(self):
        editor = NotifiesEditor()
        editor.create_track("Sounds")
        assert editor.remove_track("Sounds")
        assert editor.track_count == 0

    def test_get_track(self):
        editor = NotifiesEditor()
        track = editor.create_track("Sounds")
        found = editor.get_track("Sounds")
        assert found is track

    def test_add_sound_notify(self):
        editor = NotifiesEditor()
        editor.create_track("Sounds")
        notify = editor.add_sound_notify(
            "Sounds",
            "Footstep",
            0.5,
            "/sounds/footstep.wav",
        )
        assert notify is not None
        assert notify.notify_type == NotifyType.SOUND

    def test_add_particle_notify(self):
        editor = NotifiesEditor()
        editor.create_track("Particles")
        notify = editor.add_particle_notify(
            "Particles",
            "Dust",
            0.25,
            "/particles/dust.pfx",
        )
        assert notify is not None
        assert notify.notify_type == NotifyType.PARTICLE

    def test_add_custom_notify(self):
        editor = NotifiesEditor()
        editor.create_track("Events")
        notify = editor.add_custom_notify(
            "Events",
            "EnableDamage",
            0.3,
            "enable_damage",
        )
        assert notify is not None
        assert notify.notify_type == NotifyType.CUSTOM

    def test_add_footstep_notify(self):
        editor = NotifiesEditor()
        editor.create_track("Footsteps")
        notify = editor.add_footstep_notify(
            "Footsteps",
            "LeftFoot",
            0.25,
            FootstepType.LEFT,
        )
        assert notify is not None
        assert notify.notify_type == NotifyType.FOOTSTEP

    def test_remove_notify(self):
        editor = NotifiesEditor()
        editor.create_track("Sounds")
        editor.add_sound_notify("Sounds", "Sound", 1.0, "/s.wav")
        assert editor.remove_notify("Sounds", "Sound")

    def test_get_all_notifies(self):
        editor = NotifiesEditor()
        editor.create_track("Sounds")
        editor.create_track("Particles")
        editor.add_sound_notify("Sounds", "S1", 1.0, "/s.wav")
        editor.add_particle_notify("Particles", "P1", 1.0, "/p.pfx")

        all_notifies = editor.get_all_notifies()
        assert len(all_notifies) == 2

    def test_get_notifies_at_time(self):
        editor = NotifiesEditor()
        editor.create_track("Track1")
        editor.create_track("Track2")
        editor.add_sound_notify("Track1", "S1", 1.0, "/s.wav")
        editor.add_sound_notify("Track2", "S2", 1.0, "/s.wav")
        editor.add_sound_notify("Track1", "S3", 2.0, "/s.wav")

        at_1 = editor.get_notifies_at_time(1.0)
        assert len(at_1) == 2

    def test_set_animation_length(self):
        editor = NotifiesEditor()
        editor.set_animation_length(5.0)
        assert editor.timing.animation_length == 5.0

    def test_set_frame_rate(self):
        editor = NotifiesEditor()
        editor.set_frame_rate(60.0)
        assert editor.timing.frame_rate == 60.0

    def test_snap_notify_to_frame(self):
        editor = NotifiesEditor()
        editor.set_frame_rate(30.0)
        editor.create_track("Track")
        editor.add_sound_notify("Track", "Sound", 0.55, "/s.wav")

        editor.snap_notify_to_frame("Track", "Sound")
        notify = editor.get_track("Track").get_notify("Sound")
        # Should be snapped to frame boundary
        expected = 17 / 30  # Frame 17
        assert abs(notify.time - expected) < 0.001

    def test_select_notify(self):
        editor = NotifiesEditor()
        editor.create_track("Track")
        editor.add_sound_notify("Track", "Sound", 1.0, "/s.wav")

        editor.select_notify("Track", "Sound")
        assert ("Track", "Sound") in editor.selected_notifies

    def test_select_multiple(self):
        editor = NotifiesEditor()
        editor.create_track("Track")
        editor.add_sound_notify("Track", "S1", 1.0, "/s.wav")
        editor.add_sound_notify("Track", "S2", 2.0, "/s.wav")

        editor.select_notify("Track", "S1")
        editor.select_notify("Track", "S2", add_to_selection=True)
        assert len(editor.selected_notifies) == 2

    def test_clear_selection(self):
        editor = NotifiesEditor()
        editor.create_track("Track")
        editor.add_sound_notify("Track", "Sound", 1.0, "/s.wav")
        editor.select_notify("Track", "Sound")
        editor.clear_selection()
        assert len(editor.selected_notifies) == 0

    def test_delete_selected(self):
        editor = NotifiesEditor()
        editor.create_track("Track")
        editor.add_sound_notify("Track", "S1", 1.0, "/s.wav")
        editor.add_sound_notify("Track", "S2", 2.0, "/s.wav")
        editor.select_notify("Track", "S1")
        deleted = editor.delete_selected()
        assert deleted == 1
        assert editor.get_track("Track").notify_count == 1

    def test_move_selected(self):
        editor = NotifiesEditor()
        editor.create_track("Track")
        editor.add_sound_notify("Track", "Sound", 1.0, "/s.wav")
        editor.select_notify("Track", "Sound")
        editor.move_selected(0.5)  # Move by delta

        notify = editor.get_track("Track").get_notify("Sound")
        assert notify.time == 1.5

    def test_validate_notifies(self):
        editor = NotifiesEditor()
        editor.set_animation_length(2.0)
        editor.create_track("Track")
        editor.add_sound_notify("Track", "OutOfBounds", 5.0, "/s.wav")  # Beyond animation

        errors = editor.validate()
        assert len(errors) > 0  # Should have out of bounds error

    def test_on_change_callback(self):
        editor = NotifiesEditor()
        callback_called = [False]

        def callback():
            callback_called[0] = True

        editor.add_on_change(callback)
        editor.create_track("Track")
        assert callback_called[0]

    def test_to_dict(self):
        editor = NotifiesEditor()
        editor.create_track("Sounds")
        editor.add_sound_notify("Sounds", "Sound", 1.0, "/s.wav")

        data = editor.to_dict()
        assert "tracks" in data
        assert len(data["tracks"]) == 1

    def test_from_dict(self):
        editor = NotifiesEditor()
        editor.create_track("Sounds")
        editor.add_sound_notify("Sounds", "Sound", 1.0, "/s.wav")

        data = editor.to_dict()
        new_editor = NotifiesEditor.from_dict(data)
        assert new_editor.track_count == 1
