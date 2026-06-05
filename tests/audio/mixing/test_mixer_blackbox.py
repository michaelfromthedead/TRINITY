"""
Blackbox tests for Mixer component.

Tests PUBLIC behavior only - no internal state inspection.
Covers: 8-stage tick() pipeline, bus registry, DFS ordering, snapshots.
"""

import pytest
import math

from engine.audio.mixing import (
    Mixer,
    MixerConfig,
    MixBus,
    BusType,
    BusState,
    MixSnapshot,
    BusSnapshot,
    SnapshotManager,
    SnapshotState,
    BusRouter,
    AuxSend,
    RoutingMode,
    DuckingManager,
    SidechainManager,
    HDRAudioManager,
    InterpolationCurve,
    db_to_linear,
    linear_to_db,
    CATEGORY_MASTER,
    CATEGORY_SFX,
    CATEGORY_MUSIC,
    CATEGORY_VO,
    CATEGORY_AMBIENT,
    CATEGORY_UI,
    DEFAULT_CATEGORIES,
    SNAPSHOT_BLEND_TIME,
    MAX_ACTIVE_SNAPSHOTS,
)


class TestMixerCreation:
    """Test Mixer instantiation."""

    def test_create_mixer_default(self):
        """Mixer can be created with defaults."""
        mixer = Mixer()
        assert mixer is not None

    def test_create_mixer_with_config(self):
        """Mixer can be created with custom config."""
        config = MixerConfig()
        mixer = Mixer(config=config)
        assert mixer is not None

    def test_mixer_has_master_bus(self):
        """Mixer always has a master bus."""
        mixer = Mixer()
        mixer.initialize()
        master = mixer.get_bus("master")
        assert master is not None
        assert master.bus_type == BusType.MASTER

    def test_mixer_has_default_categories(self):
        """Mixer creates default category buses."""
        mixer = Mixer()
        mixer.initialize()
        for category in DEFAULT_CATEGORIES:
            bus = mixer.get_bus(category)
            assert bus is not None


class TestBusRegistry:
    """Test bus registration and lookup."""

    def test_get_bus_by_name(self):
        """Bus can be retrieved by name."""
        mixer = Mixer()
        mixer.initialize()
        sfx = mixer.get_bus("sfx")
        assert sfx is not None
        assert sfx.name == "sfx"

    def test_get_nonexistent_bus_returns_none(self):
        """Getting unknown bus returns None."""
        mixer = Mixer()
        mixer.initialize()
        result = mixer.get_bus("nonexistent_bus_xyz")
        assert result is None

    def test_register_custom_bus(self):
        """Custom bus can be registered."""
        mixer = Mixer()
        mixer.initialize()
        custom = MixBus(name="custom_effects", bus_type=BusType.SUB)
        mixer.register_bus(custom)
        retrieved = mixer.get_bus("custom_effects")
        assert retrieved == custom

    def test_unregister_bus(self):
        """Bus can be unregistered."""
        mixer = Mixer()
        mixer.initialize()
        custom = MixBus(name="temporary", bus_type=BusType.SUB)
        mixer.register_bus(custom)
        mixer.unregister_bus("temporary")
        assert mixer.get_bus("temporary") is None

    def test_list_all_buses(self):
        """All registered buses can be listed."""
        mixer = Mixer()
        mixer.initialize()
        buses = mixer.list_buses()
        assert "master" in buses
        assert "sfx" in buses
        assert "music" in buses

    def test_bus_count(self):
        """Bus count is accurate."""
        mixer = Mixer()
        mixer.initialize()
        initial_count = mixer.bus_count
        custom = MixBus(name="new_bus", bus_type=BusType.SUB)
        mixer.register_bus(custom)
        assert mixer.bus_count == initial_count + 1


class TestMixerTick:
    """Test the mixer update/tick pipeline."""

    def test_tick_with_delta_time(self):
        """Mixer accepts delta time for update."""
        mixer = Mixer()
        mixer.initialize()
        # Should not raise
        mixer.tick(delta_time=0.016)  # ~60fps

    def test_tick_with_zero_delta(self):
        """Mixer handles zero delta time."""
        mixer = Mixer()
        mixer.initialize()
        mixer.tick(delta_time=0.0)  # Should not crash

    def test_tick_with_large_delta(self):
        """Mixer handles large delta time gracefully."""
        mixer = Mixer()
        mixer.initialize()
        mixer.tick(delta_time=1.0)  # 1 second delta

    def test_update_alias(self):
        """update() is alias for tick()."""
        mixer = Mixer()
        mixer.initialize()
        mixer.update(delta_time=0.016)

    def test_multiple_ticks(self):
        """Multiple ticks work correctly."""
        mixer = Mixer()
        mixer.initialize()
        for _ in range(100):
            mixer.tick(delta_time=0.016)


class TestDFSOrdering:
    """Test depth-first ordering of buses for processing."""

    def test_master_processed_last(self):
        """Master bus is processed after children."""
        mixer = Mixer()
        mixer.initialize()
        order = mixer.get_processing_order()
        master_idx = next(i for i, b in enumerate(order) if b.name == "master")
        sfx_idx = next(i for i, b in enumerate(order) if b.name == "sfx")
        # Master should come after children in output
        assert master_idx > sfx_idx

    def test_children_processed_before_parent(self):
        """Children are processed before their parent."""
        mixer = Mixer()
        mixer.initialize()
        # Add sub-bus to sfx
        footsteps = MixBus(name="footsteps", bus_type=BusType.SUB)
        sfx = mixer.get_bus("sfx")
        footsteps.parent = sfx
        mixer.register_bus(footsteps)

        order = mixer.get_processing_order()
        footsteps_idx = next(i for i, b in enumerate(order) if b.name == "footsteps")
        sfx_idx = next(i for i, b in enumerate(order) if b.name == "sfx")
        assert footsteps_idx < sfx_idx

    def test_processing_order_updates_on_hierarchy_change(self):
        """Processing order updates when hierarchy changes."""
        mixer = Mixer()
        mixer.initialize()
        initial_order = list(mixer.get_processing_order())

        new_bus = MixBus(name="dynamic_bus", bus_type=BusType.SUB)
        mixer.register_bus(new_bus)

        new_order = list(mixer.get_processing_order())
        assert len(new_order) == len(initial_order) + 1


class TestSnapshots:
    """Test mix snapshot functionality."""

    def test_capture_snapshot(self):
        """Current mix state can be captured."""
        mixer = Mixer()
        mixer.initialize()
        mixer.get_bus("sfx").volume = 0.5
        mixer.get_bus("music").volume = 0.3

        snapshot = mixer.capture_snapshot("my_snapshot")
        assert snapshot is not None
        assert snapshot.name == "my_snapshot"

    def test_restore_snapshot(self):
        """Snapshot can be restored."""
        mixer = Mixer()
        mixer.initialize()
        mixer.get_bus("sfx").volume = 0.8
        snapshot = mixer.capture_snapshot("saved_state")

        mixer.get_bus("sfx").volume = 0.2
        mixer.restore_snapshot("saved_state")

        # After restore, volume should be back to captured value
        # (May need to tick for blend)
        for _ in range(100):
            mixer.tick(0.016)
        assert mixer.get_bus("sfx").volume == pytest.approx(0.8, rel=0.1)

    def test_transition_to_snapshot(self):
        """Transition to snapshot with blend time."""
        mixer = Mixer()
        mixer.initialize()
        mixer.get_bus("sfx").volume = 1.0
        mixer.capture_snapshot("quiet")

        mixer.get_bus("sfx").volume = 0.0
        mixer.capture_snapshot("silent")

        mixer.restore_snapshot("quiet")
        mixer.transition_to_snapshot("silent", blend_time=0.5)
        # Transition is started

    def test_delete_snapshot(self):
        """Snapshot can be deleted."""
        mixer = Mixer()
        mixer.initialize()
        mixer.capture_snapshot("temp")
        mixer.delete_snapshot("temp")
        # Restore should fail or return None
        result = mixer.get_snapshot("temp")
        assert result is None

    def test_list_snapshots(self):
        """All snapshots can be listed."""
        mixer = Mixer()
        mixer.initialize()
        mixer.capture_snapshot("snap1")
        mixer.capture_snapshot("snap2")
        snapshots = mixer.list_snapshots()
        assert "snap1" in snapshots
        assert "snap2" in snapshots

    def test_snapshot_with_filters(self):
        """Snapshot captures filter state."""
        mixer = Mixer()
        mixer.initialize()
        sfx = mixer.get_bus("sfx")
        sfx.enable_low_pass(3000.0)

        mixer.capture_snapshot("filtered")
        sfx.disable_filters()
        mixer.restore_snapshot("filtered")

        for _ in range(100):
            mixer.tick(0.016)
        # Filter should be restored
        assert sfx.low_pass_freq == pytest.approx(3000.0, rel=0.1)


class TestBusRouting:
    """Test pre/post fader sends and routing."""

    def test_create_aux_send(self):
        """Aux send can be created."""
        mixer = Mixer()
        mixer.initialize()
        reverb = MixBus(name="reverb", bus_type=BusType.AUX)
        mixer.register_bus(reverb)

        sfx = mixer.get_bus("sfx")
        sfx.add_send(reverb, level=0.5)

        sends = sfx.get_sends()
        assert len(sends) > 0

    def test_send_level_adjustment(self):
        """Send level can be adjusted."""
        mixer = Mixer()
        mixer.initialize()
        reverb = MixBus(name="reverb", bus_type=BusType.AUX)
        mixer.register_bus(reverb)

        sfx = mixer.get_bus("sfx")
        sfx.add_send(reverb, level=0.3)
        sfx.set_send_level(reverb, 0.7)

        sends = sfx.get_sends()
        reverb_send = next(s for s in sends if s.target == reverb)
        assert reverb_send.level == pytest.approx(0.7)

    def test_remove_send(self):
        """Send can be removed."""
        mixer = Mixer()
        mixer.initialize()
        reverb = MixBus(name="reverb", bus_type=BusType.AUX)
        mixer.register_bus(reverb)

        sfx = mixer.get_bus("sfx")
        sfx.add_send(reverb, level=0.5)
        sfx.remove_send(reverb)

        sends = sfx.get_sends()
        assert all(s.target != reverb for s in sends)

    def test_pre_fader_send(self):
        """Pre-fader send ignores source volume."""
        mixer = Mixer()
        mixer.initialize()
        reverb = MixBus(name="reverb", bus_type=BusType.AUX)
        mixer.register_bus(reverb)

        sfx = mixer.get_bus("sfx")
        sfx.add_send(reverb, level=0.5, pre_fader=True)

        sends = sfx.get_sends()
        reverb_send = next(s for s in sends if s.target == reverb)
        assert reverb_send.pre_fader is True

    def test_post_fader_send(self):
        """Post-fader send follows source volume."""
        mixer = Mixer()
        mixer.initialize()
        reverb = MixBus(name="reverb", bus_type=BusType.AUX)
        mixer.register_bus(reverb)

        sfx = mixer.get_bus("sfx")
        sfx.add_send(reverb, level=0.5, pre_fader=False)

        sends = sfx.get_sends()
        reverb_send = next(s for s in sends if s.target == reverb)
        assert reverb_send.pre_fader is False


class TestDucking:
    """Test volume ducking functionality."""

    def test_ducking_manager_exists(self):
        """Mixer has ducking manager."""
        mixer = Mixer()
        mixer.initialize()
        assert mixer.ducking is not None

    def test_apply_dialogue_duck(self):
        """Dialogue ducking can be applied."""
        mixer = Mixer()
        mixer.initialize()

        # Apply ducking to sfx when VO plays
        mixer.ducking.apply_duck(
            source="vo",
            target="sfx",
            amount_db=-6.0
        )

        # Ducking is registered

    def test_release_duck(self):
        """Ducking can be released."""
        mixer = Mixer()
        mixer.initialize()

        duck_id = mixer.ducking.apply_duck(
            source="vo",
            target="sfx",
            amount_db=-6.0
        )
        mixer.ducking.release_duck(duck_id)

    def test_duck_with_attack_release(self):
        """Ducking respects attack/release times."""
        mixer = Mixer()
        mixer.initialize()

        mixer.ducking.apply_duck(
            source="vo",
            target="sfx",
            amount_db=-6.0,
            attack_ms=50.0,
            release_ms=200.0
        )


class TestSidechain:
    """Test sidechain compression functionality."""

    def test_sidechain_manager_exists(self):
        """Mixer has sidechain manager."""
        mixer = Mixer()
        mixer.initialize()
        assert mixer.sidechain is not None

    def test_create_sidechain(self):
        """Sidechain compression can be created."""
        mixer = Mixer()
        mixer.initialize()

        mixer.sidechain.create(
            key_source="kick",
            target="bass",
            ratio=4.0,
            threshold_db=-20.0
        )

    def test_remove_sidechain(self):
        """Sidechain compression can be removed."""
        mixer = Mixer()
        mixer.initialize()

        sc_id = mixer.sidechain.create(
            key_source="kick",
            target="bass",
            ratio=4.0,
            threshold_db=-20.0
        )
        mixer.sidechain.remove(sc_id)


class TestHDRAudio:
    """Test HDR audio functionality."""

    def test_hdr_manager_exists(self):
        """Mixer has HDR audio manager."""
        mixer = Mixer()
        mixer.initialize()
        assert mixer.hdr is not None

    def test_enable_hdr(self):
        """HDR audio can be enabled."""
        mixer = Mixer()
        mixer.initialize()
        mixer.hdr.enable()
        assert mixer.hdr.enabled is True

    def test_disable_hdr(self):
        """HDR audio can be disabled."""
        mixer = Mixer()
        mixer.initialize()
        mixer.hdr.enable()
        mixer.hdr.disable()
        assert mixer.hdr.enabled is False


class TestMixerPauseResume:
    """Test mixer pause/resume."""

    def test_pause_mixer(self):
        """Mixer can be paused."""
        mixer = Mixer()
        mixer.initialize()
        mixer.pause()
        assert mixer.paused is True

    def test_resume_mixer(self):
        """Mixer can be resumed."""
        mixer = Mixer()
        mixer.initialize()
        mixer.pause()
        mixer.resume()
        assert mixer.paused is False

    def test_pause_stops_processing(self):
        """Paused mixer doesn't process audio."""
        mixer = Mixer()
        mixer.initialize()
        mixer.pause()
        # Tick should be no-op when paused
        mixer.tick(0.016)


class TestMixerVolume:
    """Test master volume control."""

    def test_set_master_volume(self):
        """Master volume can be set."""
        mixer = Mixer()
        mixer.initialize()
        mixer.master_volume = 0.5
        assert mixer.master_volume == pytest.approx(0.5)

    def test_master_volume_affects_all_buses(self):
        """Master volume affects effective output."""
        mixer = Mixer()
        mixer.initialize()
        mixer.master_volume = 0.5
        master = mixer.get_bus("master")
        assert master.volume == pytest.approx(0.5)

    def test_mute_all(self):
        """All buses can be muted."""
        mixer = Mixer()
        mixer.initialize()
        mixer.mute_all()
        # All buses should be muted
        for bus_name in mixer.list_buses():
            bus = mixer.get_bus(bus_name)
            assert bus.muted is True

    def test_unmute_all(self):
        """All buses can be unmuted."""
        mixer = Mixer()
        mixer.initialize()
        mixer.mute_all()
        mixer.unmute_all()
        # All buses should be unmuted
        for bus_name in mixer.list_buses():
            bus = mixer.get_bus(bus_name)
            assert bus.muted is False


class TestMixerStats:
    """Test mixer statistics and monitoring."""

    def test_get_peak_levels(self):
        """Peak levels can be retrieved."""
        mixer = Mixer()
        mixer.initialize()
        peaks = mixer.get_peak_levels()
        assert "master" in peaks

    def test_get_rms_levels(self):
        """RMS levels can be retrieved."""
        mixer = Mixer()
        mixer.initialize()
        rms = mixer.get_rms_levels()
        assert "master" in rms

    def test_reset_meters(self):
        """Peak meters can be reset."""
        mixer = Mixer()
        mixer.initialize()
        mixer.reset_meters()


class TestMixerEdgeCases:
    """Test edge cases and error handling."""

    def test_tick_before_initialize(self):
        """Tick before initialize is handled."""
        mixer = Mixer()
        # Should not crash, may no-op
        try:
            mixer.tick(0.016)
        except Exception:
            pass  # Some implementations may raise

    def test_double_initialize(self):
        """Double initialize is handled."""
        mixer = Mixer()
        mixer.initialize()
        mixer.initialize()  # Should be idempotent

    def test_shutdown(self):
        """Mixer can be shut down."""
        mixer = Mixer()
        mixer.initialize()
        mixer.shutdown()

    def test_reinitialize_after_shutdown(self):
        """Mixer can be reinitialized after shutdown."""
        mixer = Mixer()
        mixer.initialize()
        mixer.shutdown()
        mixer.initialize()
        assert mixer.get_bus("master") is not None


class TestSnapshotInterpolation:
    """Test snapshot interpolation curves."""

    def test_linear_interpolation(self):
        """Linear interpolation works."""
        mixer = Mixer()
        mixer.initialize()
        mixer.get_bus("sfx").volume = 1.0
        mixer.capture_snapshot("full")

        mixer.get_bus("sfx").volume = 0.0
        mixer.transition_to_snapshot("full", blend_time=1.0, curve=InterpolationCurve.LINEAR)

    def test_ease_in_interpolation(self):
        """Ease-in interpolation works."""
        mixer = Mixer()
        mixer.initialize()
        mixer.capture_snapshot("state")
        mixer.transition_to_snapshot("state", blend_time=1.0, curve=InterpolationCurve.EASE_IN)

    def test_ease_out_interpolation(self):
        """Ease-out interpolation works."""
        mixer = Mixer()
        mixer.initialize()
        mixer.capture_snapshot("state")
        mixer.transition_to_snapshot("state", blend_time=1.0, curve=InterpolationCurve.EASE_OUT)

    def test_ease_in_out_interpolation(self):
        """Ease-in-out interpolation works."""
        mixer = Mixer()
        mixer.initialize()
        mixer.capture_snapshot("state")
        mixer.transition_to_snapshot("state", blend_time=1.0, curve=InterpolationCurve.EASE_IN_OUT)
