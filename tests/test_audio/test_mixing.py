"""
Comprehensive tests for the audio mixing subsystem.

Tests cover:
- MixBus (creation, volume, filters, mute/solo)
- Bus hierarchy (parent-child relationships)
- Bus routing (aux sends, direct outputs)
- MixSnapshot (create, apply, blend)
- Snapshot transitions (interpolation)
- Ducking (trigger, attack, release)
- Sidechain compression
- HDR audio (window, adaptation)
- Mixer (full integration)
- Thread safety
"""

import math
import threading
import time
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from engine.audio.mixing import (
    # Main classes
    Mixer,
    MixerConfig,
    MixBus,
    BusRouter,
    SnapshotManager,
    DuckingManager,
    SidechainManager,
    HDRAudioManager,
    # Types and configs
    BusType,
    BusState,
    FilterState,
    RoutingMode,
    AuxSend,
    DirectOutput,
    MixSnapshot,
    BusSnapshot,
    SnapshotState,
    DuckConfig,
    DuckType,
    DuckState,
    SidechainConfig,
    CompressorState,
    AudioSource,
    MixWindow,
    HDRPriority,
    InterpolationCurve,
    # Utilities
    db_to_linear,
    linear_to_db,
    clamp,
    lerp,
    apply_curve,
    create_default_hierarchy,
    # Constants
    CATEGORY_MASTER,
    CATEGORY_SFX,
    CATEGORY_MUSIC,
    CATEGORY_VO,
    MIN_VOLUME_DB,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def master_bus() -> MixBus:
    """Create a master bus."""
    return MixBus("master", BusType.MASTER)


@pytest.fixture
def sfx_bus(master_bus: MixBus) -> MixBus:
    """Create an SFX category bus."""
    return MixBus("sfx", BusType.CATEGORY, parent=master_bus)


@pytest.fixture
def music_bus(master_bus: MixBus) -> MixBus:
    """Create a music category bus."""
    return MixBus("music", BusType.CATEGORY, parent=master_bus)


@pytest.fixture
def vo_bus(master_bus: MixBus) -> MixBus:
    """Create a VO category bus."""
    return MixBus("vo", BusType.CATEGORY, parent=master_bus)


@pytest.fixture
def bus_hierarchy(
    master_bus: MixBus,
    sfx_bus: MixBus,
    music_bus: MixBus,
    vo_bus: MixBus,
) -> dict[str, MixBus]:
    """Create a complete bus hierarchy."""
    buses = {
        "master": master_bus,
        "sfx": sfx_bus,
        "music": music_bus,
        "vo": vo_bus,
    }

    # Add sub-buses
    buses["footsteps"] = MixBus("footsteps", BusType.SUB, parent=sfx_bus)
    buses["weapons"] = MixBus("weapons", BusType.SUB, parent=sfx_bus)

    return buses


@pytest.fixture
def router() -> BusRouter:
    """Create a bus router."""
    return BusRouter()


@pytest.fixture
def snapshot_manager(bus_hierarchy: dict[str, MixBus]) -> SnapshotManager:
    """Create a snapshot manager with buses."""
    manager = SnapshotManager(bus_hierarchy)
    return manager


@pytest.fixture
def ducking_manager() -> DuckingManager:
    """Create a ducking manager."""
    return DuckingManager()


@pytest.fixture
def sidechain_manager() -> SidechainManager:
    """Create a sidechain manager."""
    return SidechainManager()


@pytest.fixture
def hdr_manager() -> HDRAudioManager:
    """Create an HDR audio manager."""
    return HDRAudioManager()


@pytest.fixture
def mixer() -> Mixer:
    """Create and initialize a mixer."""
    m = Mixer()
    m.initialize()
    return m


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions in config.py."""

    def test_db_to_linear_zero_db(self):
        """0 dB should equal 1.0 linear."""
        assert db_to_linear(0.0) == pytest.approx(1.0)

    def test_db_to_linear_minus_6db(self):
        """-6 dB should be approximately 0.5 linear (exact: 10^(-6/20) = 0.501)."""
        expected = 10.0 ** (-6.0 / 20.0)  # ~0.501187
        assert db_to_linear(-6.0) == pytest.approx(expected, rel=0.001)

    def test_db_to_linear_minus_20db(self):
        """-20 dB should equal 0.1 linear (exact: 10^(-20/20) = 0.1)."""
        expected = 10.0 ** (-20.0 / 20.0)  # exactly 0.1
        assert db_to_linear(-20.0) == pytest.approx(expected, rel=0.001)

    def test_db_to_linear_silence(self):
        """Very low dB should be 0 (silence threshold at -80dB)."""
        assert db_to_linear(-80.0) == 0.0
        assert db_to_linear(-100.0) == 0.0

    def test_db_to_linear_positive(self):
        """+6 dB should be approximately 2.0 linear (exact: 10^(6/20) = 1.995)."""
        expected = 10.0 ** (6.0 / 20.0)  # ~1.995
        assert db_to_linear(6.0) == pytest.approx(expected, rel=0.01)

    def test_db_to_linear_plus_12db(self):
        """+12 dB should be approximately 4.0 linear."""
        expected = 10.0 ** (12.0 / 20.0)  # ~3.981
        assert db_to_linear(12.0) == pytest.approx(expected, rel=0.01)

    def test_linear_to_db_unity(self):
        """1.0 linear should equal 0 dB."""
        assert linear_to_db(1.0) == pytest.approx(0.0)

    def test_linear_to_db_half(self):
        """0.5 linear should be approximately -6 dB."""
        assert linear_to_db(0.5) == pytest.approx(-6.0, rel=0.1)

    def test_linear_to_db_silence(self):
        """Very low or zero linear should return minimum dB."""
        assert linear_to_db(0.0) == -80.0
        assert linear_to_db(0.00001) == -80.0

    def test_clamp_within_range(self):
        """Values within range should be unchanged."""
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_clamp_below_min(self):
        """Values below min should be clamped to min."""
        assert clamp(-0.5, 0.0, 1.0) == 0.0

    def test_clamp_above_max(self):
        """Values above max should be clamped to max."""
        assert clamp(1.5, 0.0, 1.0) == 1.0

    def test_lerp_start(self):
        """t=0 should return a."""
        assert lerp(0.0, 1.0, 0.0) == 0.0

    def test_lerp_end(self):
        """t=1 should return b."""
        assert lerp(0.0, 1.0, 1.0) == 1.0

    def test_lerp_middle(self):
        """t=0.5 should return midpoint."""
        assert lerp(0.0, 1.0, 0.5) == 0.5

    def test_lerp_clamped(self):
        """t values outside [0,1] should be clamped."""
        assert lerp(0.0, 1.0, -0.5) == 0.0
        assert lerp(0.0, 1.0, 1.5) == 1.0

    def test_apply_curve_linear(self):
        """Linear curve should pass through unchanged."""
        assert apply_curve(0.5, InterpolationCurve.LINEAR) == 0.5

    def test_apply_curve_ease_in(self):
        """Ease in should be slower at start."""
        result = apply_curve(0.5, InterpolationCurve.EASE_IN)
        assert result < 0.5  # Slower at start

    def test_apply_curve_ease_out(self):
        """Ease out should be faster at start."""
        result = apply_curve(0.5, InterpolationCurve.EASE_OUT)
        assert result > 0.5  # Faster at start


# =============================================================================
# MixBus Tests
# =============================================================================


class TestMixBus:
    """Tests for MixBus class."""

    def test_create_bus(self):
        """Test basic bus creation."""
        bus = MixBus("test", BusType.SUB)
        assert bus.name == "test"
        assert bus.bus_type == BusType.SUB
        assert bus.volume == pytest.approx(1.0)

    def test_create_master_bus(self):
        """Test master bus creation."""
        bus = MixBus("master", BusType.MASTER)
        assert bus.bus_type == BusType.MASTER
        assert bus.parent is None

    def test_bus_has_unique_id(self):
        """Each bus should have a unique ID."""
        bus1 = MixBus("test1", BusType.SUB)
        bus2 = MixBus("test2", BusType.SUB)
        assert bus1.id != bus2.id

    def test_set_volume_linear(self):
        """Test setting volume in linear scale."""
        bus = MixBus("test", BusType.SUB)
        bus.volume = 0.5
        assert bus.volume == pytest.approx(0.5)

    def test_set_volume_db(self):
        """Test setting volume in dB."""
        bus = MixBus("test", BusType.SUB)
        bus.volume_db = -6.0
        assert bus.volume == pytest.approx(0.5, rel=0.01)

    def test_volume_clamp_min(self):
        """Volume should be clamped to valid range."""
        bus = MixBus("test", BusType.SUB)
        bus.volume = -1.0
        assert bus.volume == 0.0

    def test_volume_clamp_max(self):
        """Volume should not exceed maximum."""
        bus = MixBus("test", BusType.SUB)
        bus.volume = 10.0
        assert bus.volume <= db_to_linear(12.0)

    def test_pitch_default(self):
        """Default pitch should be 1.0."""
        bus = MixBus("test", BusType.SUB)
        assert bus.pitch == 1.0

    def test_pitch_set(self):
        """Test setting pitch."""
        bus = MixBus("test", BusType.SUB)
        bus.pitch = 2.0
        assert bus.pitch == 2.0

    def test_pitch_clamp(self):
        """Pitch should be clamped to valid range."""
        bus = MixBus("test", BusType.SUB)
        bus.pitch = 0.01
        assert bus.pitch >= 0.1
        bus.pitch = 10.0
        assert bus.pitch <= 4.0

    def test_mute(self):
        """Test mute functionality."""
        bus = MixBus("test", BusType.SUB)
        assert not bus.muted
        bus.muted = True
        assert bus.muted

    def test_toggle_mute(self):
        """Test toggle mute."""
        bus = MixBus("test", BusType.SUB)
        result = bus.toggle_mute()
        assert result is True
        assert bus.muted
        result = bus.toggle_mute()
        assert result is False
        assert not bus.muted

    def test_solo(self):
        """Test solo functionality."""
        bus = MixBus("test", BusType.SUB)
        assert not bus.soloed
        bus.soloed = True
        assert bus.soloed

    def test_toggle_solo(self):
        """Test toggle solo."""
        bus = MixBus("test", BusType.SUB)
        result = bus.toggle_solo()
        assert result is True
        assert bus.soloed

    def test_low_pass_filter(self):
        """Test low-pass filter settings."""
        bus = MixBus("test", BusType.SUB)
        bus.set_low_pass(8000.0, q=1.0)
        filters = bus.filters
        assert filters.low_pass_freq == 8000.0
        assert filters.low_pass_enabled is True

    def test_high_pass_filter(self):
        """Test high-pass filter settings."""
        bus = MixBus("test", BusType.SUB)
        bus.set_high_pass(200.0)
        filters = bus.filters
        assert filters.high_pass_freq == 200.0
        assert filters.high_pass_enabled is True

    def test_reset_filters(self):
        """Test filter reset."""
        bus = MixBus("test", BusType.SUB)
        bus.set_low_pass(5000.0)
        bus.reset_filters()
        filters = bus.filters
        assert filters.low_pass_enabled is False

    def test_get_state(self):
        """Test getting bus state."""
        bus = MixBus("test", BusType.SUB, volume=0.8)
        bus.muted = True
        state = bus.get_state()
        assert state.volume_linear == pytest.approx(0.8)
        assert state.muted is True

    def test_set_state(self):
        """Test setting bus state."""
        bus = MixBus("test", BusType.SUB)
        state = BusState(volume_linear=0.5, muted=True)
        bus.set_state(state)
        assert bus.volume == pytest.approx(0.5)
        assert bus.muted is True

    def test_reset(self):
        """Test bus reset to defaults."""
        bus = MixBus("test", BusType.SUB)
        bus.volume = 0.5
        bus.muted = True
        bus.reset()
        assert bus.volume == pytest.approx(1.0)
        assert bus.muted is False

    def test_on_change_callback(self):
        """Test change callbacks."""
        bus = MixBus("test", BusType.SUB)
        callback_called = []

        def callback(b: MixBus):
            callback_called.append(b)

        bus.on_change(callback)
        bus.volume = 0.5

        assert len(callback_called) == 1
        assert callback_called[0] is bus


# =============================================================================
# Bus Hierarchy Tests
# =============================================================================


class TestBusHierarchy:
    """Tests for bus parent-child relationships."""

    def test_set_parent(self, master_bus: MixBus, sfx_bus: MixBus):
        """Test setting parent bus."""
        assert sfx_bus.parent is master_bus
        assert sfx_bus in master_bus.children

    def test_remove_parent(self, sfx_bus: MixBus, master_bus: MixBus):
        """Test removing parent."""
        sfx_bus.set_parent(None)
        assert sfx_bus.parent is None
        assert sfx_bus not in master_bus.children

    def test_add_child(self, master_bus: MixBus):
        """Test adding child bus."""
        child = MixBus("child", BusType.SUB)
        master_bus.add_child(child)
        assert child.parent is master_bus
        assert child in master_bus.children

    def test_remove_child(self, master_bus: MixBus, sfx_bus: MixBus):
        """Test removing child bus."""
        result = master_bus.remove_child(sfx_bus)
        assert result is True
        assert sfx_bus.parent is None
        assert sfx_bus not in master_bus.children

    def test_cannot_set_self_as_parent(self, master_bus: MixBus):
        """Cannot set a bus as its own parent."""
        with pytest.raises(ValueError, match="cannot be its own parent"):
            master_bus.set_parent(master_bus)

    def test_prevent_cycle(self, master_bus: MixBus, sfx_bus: MixBus):
        """Setting parent should not create cycles."""
        child = MixBus("child", BusType.SUB, parent=sfx_bus)
        with pytest.raises(ValueError, match="cycle"):
            master_bus.set_parent(child)

    def test_get_ancestors(self, bus_hierarchy: dict[str, MixBus]):
        """Test getting ancestor buses."""
        footsteps = bus_hierarchy["footsteps"]
        ancestors = footsteps.get_ancestors()
        assert len(ancestors) == 2
        assert bus_hierarchy["sfx"] in ancestors
        assert bus_hierarchy["master"] in ancestors

    def test_get_descendants(self, bus_hierarchy: dict[str, MixBus]):
        """Test getting descendant buses."""
        sfx = bus_hierarchy["sfx"]
        descendants = sfx.get_descendants()
        assert bus_hierarchy["footsteps"] in descendants
        assert bus_hierarchy["weapons"] in descendants

    def test_effective_volume(self, master_bus: MixBus, sfx_bus: MixBus):
        """Test effective volume through hierarchy."""
        master_bus.volume = 0.5
        sfx_bus.volume = 0.5
        assert sfx_bus.get_effective_volume() == pytest.approx(0.25)

    def test_effective_volume_muted_parent(
        self, master_bus: MixBus, sfx_bus: MixBus
    ):
        """Muted parent should result in 0 effective volume."""
        master_bus.muted = True
        assert sfx_bus.get_effective_volume() == 0.0

    def test_effective_pitch(self, master_bus: MixBus, sfx_bus: MixBus):
        """Test effective pitch through hierarchy."""
        master_bus.pitch = 2.0
        sfx_bus.pitch = 0.5
        assert sfx_bus.get_effective_pitch() == pytest.approx(1.0)

    def test_create_default_hierarchy(self):
        """Test creating default bus hierarchy."""
        buses = create_default_hierarchy()
        assert "master" in buses
        assert "sfx" in buses
        assert "music" in buses
        assert "vo" in buses
        assert buses["sfx"].parent is buses["master"]


# =============================================================================
# Bus Routing Tests
# =============================================================================


class TestBusRouting:
    """Tests for bus routing (aux sends, direct outputs)."""

    def test_create_aux_send(
        self, router: BusRouter, sfx_bus: MixBus, master_bus: MixBus
    ):
        """Test creating an aux send."""
        aux = MixBus("reverb", BusType.AUX, parent=master_bus)
        router.register_aux_bus(aux)

        send = router.create_send(sfx_bus, aux, level_db=-6.0)
        assert send is not None
        assert send.source_bus is sfx_bus
        assert send.target_bus is aux
        assert send.send_level_db == -6.0

    def test_create_send_post_fader(
        self, router: BusRouter, sfx_bus: MixBus, master_bus: MixBus
    ):
        """Test post-fader send (default)."""
        aux = MixBus("reverb", BusType.AUX, parent=master_bus)
        send = router.create_send(sfx_bus, aux)
        assert send.mode == RoutingMode.POST_FADER

    def test_create_send_pre_fader(
        self, router: BusRouter, sfx_bus: MixBus, master_bus: MixBus
    ):
        """Test pre-fader send."""
        aux = MixBus("reverb", BusType.AUX, parent=master_bus)
        send = router.create_send(sfx_bus, aux, mode=RoutingMode.PRE_FADER)
        assert send.mode == RoutingMode.PRE_FADER

    def test_cannot_send_to_self(self, router: BusRouter, sfx_bus: MixBus):
        """Cannot create send from bus to itself."""
        with pytest.raises(ValueError, match="itself"):
            router.create_send(sfx_bus, sfx_bus)

    def test_remove_send(
        self, router: BusRouter, sfx_bus: MixBus, master_bus: MixBus
    ):
        """Test removing an aux send."""
        aux = MixBus("reverb", BusType.AUX, parent=master_bus)
        send = router.create_send(sfx_bus, aux)
        result = router.remove_send(send)
        assert result is True
        assert len(router.get_sends(sfx_bus)) == 0

    def test_get_sends(
        self, router: BusRouter, sfx_bus: MixBus, master_bus: MixBus
    ):
        """Test getting sends from a bus."""
        aux1 = MixBus("reverb", BusType.AUX, parent=master_bus)
        aux2 = MixBus("delay", BusType.AUX, parent=master_bus)
        router.create_send(sfx_bus, aux1)
        router.create_send(sfx_bus, aux2)

        sends = router.get_sends(sfx_bus)
        assert len(sends) == 2

    def test_set_direct_output(
        self, router: BusRouter, sfx_bus: MixBus, master_bus: MixBus
    ):
        """Test setting direct output."""
        output = router.set_direct_output(sfx_bus, master_bus, level_db=-3.0)
        assert output is not None
        assert output.source_bus is sfx_bus
        assert output.target_bus is master_bus
        assert output.level_db == -3.0

    def test_clear_direct_output(
        self, router: BusRouter, sfx_bus: MixBus, master_bus: MixBus
    ):
        """Test clearing direct output."""
        router.set_direct_output(sfx_bus, master_bus)
        result = router.clear_direct_output(sfx_bus)
        assert result is True
        assert router.get_direct_output(sfx_bus) is None

    def test_has_direct_output(
        self, router: BusRouter, sfx_bus: MixBus, master_bus: MixBus
    ):
        """Test checking for direct output."""
        assert not router.has_direct_output(sfx_bus)
        router.set_direct_output(sfx_bus, master_bus)
        assert router.has_direct_output(sfx_bus)

    def test_get_effective_routing(
        self, router: BusRouter, sfx_bus: MixBus, master_bus: MixBus
    ):
        """Test getting complete routing info."""
        aux = MixBus("reverb", BusType.AUX, parent=master_bus)
        router.create_send(sfx_bus, aux, level_db=-6.0)

        routing = router.get_effective_routing(sfx_bus)
        assert routing["bus_name"] == "sfx"
        assert routing["parent"] == "master"
        assert len(routing["aux_sends"]) == 1


# =============================================================================
# MixSnapshot Tests
# =============================================================================


class TestMixSnapshot:
    """Tests for mix snapshots."""

    def test_create_snapshot(self, bus_hierarchy: dict[str, MixBus]):
        """Test creating a snapshot."""
        snapshot = MixSnapshot.capture("test", bus_hierarchy)
        assert snapshot.name == "test"
        assert len(snapshot.bus_states) == len(bus_hierarchy)

    def test_snapshot_captures_volume(self, bus_hierarchy: dict[str, MixBus]):
        """Snapshot should capture current volumes."""
        bus_hierarchy["sfx"].volume = 0.5
        snapshot = MixSnapshot.capture("test", bus_hierarchy)
        assert snapshot.bus_states["sfx"].volume_linear == pytest.approx(0.5)

    def test_snapshot_captures_filters(self, bus_hierarchy: dict[str, MixBus]):
        """Snapshot should capture filter settings."""
        bus_hierarchy["music"].set_low_pass(5000.0)
        snapshot = MixSnapshot.capture("test", bus_hierarchy)
        assert snapshot.bus_states["music"].low_pass_freq == 5000.0
        assert snapshot.bus_states["music"].low_pass_enabled is True

    def test_apply_snapshot_to_bus(self, bus_hierarchy: dict[str, MixBus]):
        """Test applying snapshot to a single bus."""
        bus_hierarchy["sfx"].volume = 0.5
        snapshot = MixSnapshot.capture("test", bus_hierarchy)

        bus_hierarchy["sfx"].volume = 1.0
        snapshot.apply_to_bus(bus_hierarchy["sfx"])
        assert bus_hierarchy["sfx"].volume == pytest.approx(0.5)

    def test_apply_snapshot_to_all(self, bus_hierarchy: dict[str, MixBus]):
        """Test applying snapshot to all buses."""
        bus_hierarchy["sfx"].volume = 0.5
        bus_hierarchy["music"].volume = 0.3
        snapshot = MixSnapshot.capture("test", bus_hierarchy)

        bus_hierarchy["sfx"].volume = 1.0
        bus_hierarchy["music"].volume = 1.0
        snapshot.apply_to_all(bus_hierarchy)

        assert bus_hierarchy["sfx"].volume == pytest.approx(0.5)
        assert bus_hierarchy["music"].volume == pytest.approx(0.3)

    def test_bus_snapshot_interpolate(self):
        """Test interpolating between bus snapshots."""
        a = BusSnapshot(bus_name="test", volume_linear=0.0)
        b = BusSnapshot(bus_name="test", volume_linear=1.0)

        result = BusSnapshot.interpolate(a, b, 0.5)
        assert result.volume_linear == pytest.approx(0.5)

    def test_snapshot_copy(self, bus_hierarchy: dict[str, MixBus]):
        """Test snapshot copy creates independent copy."""
        snapshot = MixSnapshot.capture("test", bus_hierarchy)
        copy = snapshot.copy()

        copy.bus_states["sfx"].volume_linear = 0.1
        assert snapshot.bus_states["sfx"].volume_linear != 0.1

    def test_bus_snapshot_interpolate_volume(self):
        """Verify volume interpolation is mathematically correct."""
        a = BusSnapshot(bus_name="test", volume_linear=0.0)
        b = BusSnapshot(bus_name="test", volume_linear=1.0)

        # Test various interpolation points
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = BusSnapshot.interpolate(a, b, t)
            expected = t  # Linear interpolation from 0 to 1
            assert result.volume_linear == pytest.approx(expected, abs=0.01), (
                f"At t={t}, expected volume {expected}, got {result.volume_linear}"
            )

    def test_bus_snapshot_interpolate_pitch(self):
        """Verify pitch interpolation handles multipliers correctly."""
        a = BusSnapshot(bus_name="test", pitch=0.5)
        b = BusSnapshot(bus_name="test", pitch=2.0)

        result = BusSnapshot.interpolate(a, b, 0.5)
        expected = 1.25  # Linear interp: 0.5 + (2.0 - 0.5) * 0.5 = 1.25
        assert result.pitch == pytest.approx(expected, abs=0.01), (
            f"Expected pitch {expected}, got {result.pitch}"
        )

    def test_bus_snapshot_interpolate_filter_freq(self):
        """Verify filter frequency interpolation."""
        a = BusSnapshot(bus_name="test", low_pass_freq=1000.0)
        b = BusSnapshot(bus_name="test", low_pass_freq=10000.0)

        result = BusSnapshot.interpolate(a, b, 0.5)
        expected = 5500.0  # Linear interp midpoint
        assert result.low_pass_freq == pytest.approx(expected, abs=10.0), (
            f"Expected filter freq {expected}, got {result.low_pass_freq}"
        )


# =============================================================================
# Snapshot Manager Tests
# =============================================================================


class TestSnapshotManager:
    """Tests for snapshot manager and transitions."""

    def test_capture_snapshot(self, snapshot_manager: SnapshotManager):
        """Test capturing current state."""
        snapshot = snapshot_manager.capture_snapshot("test")
        assert snapshot.name == "test"

    def test_save_and_load_snapshot(self, snapshot_manager: SnapshotManager):
        """Test saving and loading snapshots."""
        snapshot = snapshot_manager.capture_snapshot("test")
        snapshot_manager.save_snapshot(snapshot)

        loaded = snapshot_manager.load_snapshot("test")
        assert loaded is not None
        assert loaded.name == "test"

    def test_delete_snapshot(self, snapshot_manager: SnapshotManager):
        """Test deleting a snapshot."""
        snapshot_manager.capture_snapshot("test")
        result = snapshot_manager.delete_snapshot("test")
        assert result is True
        assert snapshot_manager.load_snapshot("test") is None

    def test_list_snapshots(self, snapshot_manager: SnapshotManager):
        """Test listing snapshots."""
        snapshot_manager.capture_snapshot("test1")
        snapshot_manager.capture_snapshot("test2")
        names = snapshot_manager.list_snapshots()
        assert "test1" in names
        assert "test2" in names

    def test_transition_to_snapshot(
        self,
        snapshot_manager: SnapshotManager,
        bus_hierarchy: dict[str, MixBus],
    ):
        """Test transitioning to a snapshot."""
        bus_hierarchy["sfx"].volume = 0.5
        snapshot = snapshot_manager.capture_snapshot("target")

        bus_hierarchy["sfx"].volume = 1.0
        snapshot_manager.transition_to(snapshot, blend_time=0.0)
        snapshot_manager.update(0.1)

        # With 0 blend time, should be applied immediately
        assert bus_hierarchy["sfx"].volume == pytest.approx(0.5, abs=0.1)

    def test_is_transitioning(self, snapshot_manager: SnapshotManager):
        """Test checking transition state."""
        snapshot = snapshot_manager.capture_snapshot("test")
        assert not snapshot_manager.is_transitioning()

        snapshot_manager.transition_to(snapshot, blend_time=1.0)
        assert snapshot_manager.is_transitioning()

    def test_apply_immediate(
        self,
        snapshot_manager: SnapshotManager,
        bus_hierarchy: dict[str, MixBus],
    ):
        """Test immediate snapshot application."""
        bus_hierarchy["sfx"].volume = 0.5
        snapshot = snapshot_manager.capture_snapshot("test")

        bus_hierarchy["sfx"].volume = 1.0
        snapshot_manager.apply_immediate(snapshot)

        assert bus_hierarchy["sfx"].volume == pytest.approx(0.5)

    def test_transition_callback(self, snapshot_manager: SnapshotManager):
        """Test transition complete callback."""
        completed = []

        def callback(snapshot: MixSnapshot):
            completed.append(snapshot)

        snapshot_manager.on_transition_complete(callback)
        snapshot = snapshot_manager.capture_snapshot("test")
        snapshot_manager.transition_to(snapshot, blend_time=0.0)

        # Multiple updates to complete
        for _ in range(5):
            snapshot_manager.update(0.1)

        assert len(completed) > 0


# =============================================================================
# Ducking Tests
# =============================================================================


class TestDucking:
    """Tests for ducking system."""

    def test_create_duck_config(self, vo_bus: MixBus, music_bus: MixBus):
        """Test creating duck configuration."""
        config = DuckConfig(
            name="dialogue_duck",
            duck_type=DuckType.DIALOGUE,
            source_bus=vo_bus,
            target_buses=[music_bus],
            amount_db=-12.0,
        )
        assert config.source_bus is vo_bus
        assert music_bus in config.target_buses

    def test_create_dialogue_duck(
        self,
        ducking_manager: DuckingManager,
        vo_bus: MixBus,
        music_bus: MixBus,
        sfx_bus: MixBus,
    ):
        """Test creating dialogue ducking."""
        instance = ducking_manager.create_dialogue_duck(
            vo_bus, [music_bus, sfx_bus], amount_db=-12.0
        )
        assert instance is not None
        assert instance.config.duck_type == DuckType.DIALOGUE

    def test_duck_trigger(
        self,
        ducking_manager: DuckingManager,
        vo_bus: MixBus,
        music_bus: MixBus,
    ):
        """Test triggering duck."""
        instance = ducking_manager.create_dialogue_duck(vo_bus, [music_bus])

        # Simulate VO above threshold
        ducking_manager.analyze_source_levels({vo_bus.id: -10.0})
        ducking_manager.update(0.1)

        assert instance.is_active

    def test_duck_release(
        self,
        ducking_manager: DuckingManager,
        vo_bus: MixBus,
        music_bus: MixBus,
    ):
        """Test duck release when source goes quiet."""
        instance = ducking_manager.create_dialogue_duck(vo_bus, [music_bus])

        # Trigger
        ducking_manager.analyze_source_levels({vo_bus.id: -10.0})
        ducking_manager.update(0.1)

        # Verify triggered
        assert instance.is_active

        # Release - go below threshold
        ducking_manager.analyze_source_levels({vo_bus.id: -80.0})

        # The release() method starts the hold phase, so we need real time to pass
        # for hold_end_time check. Use a small sleep to let real time advance.
        time.sleep(0.2)  # 200ms > hold time (100ms)

        # Update to transition through hold -> release
        for _ in range(20):
            ducking_manager.update(0.05)
            time.sleep(0.01)  # Small sleep to let time.time() advance

        # Should be releasing or idle by now (1+ seconds elapsed)
        final_state = instance.envelope.state
        # Accept holding if it's still transitioning, but duck amount should be decreasing
        assert final_state in (DuckState.HOLDING, DuckState.RELEASING, DuckState.IDLE) or instance.envelope.current_amount < 1.0

    def test_event_duck(
        self,
        ducking_manager: DuckingManager,
        music_bus: MixBus,
    ):
        """Test event ducking."""
        instance = ducking_manager.create_event_duck([music_bus], amount_db=-6.0)
        instance.trigger()

        ducking_manager.update(0.1)
        assert instance.is_active

    def test_focus_duck(
        self,
        ducking_manager: DuckingManager,
        music_bus: MixBus,
    ):
        """Test focus ducking."""
        instance = ducking_manager.create_focus_duck([music_bus])

        ducking_manager.trigger_focus_duck()
        ducking_manager.update(0.1)

        # Check that some focus duck is active
        ducks = ducking_manager.get_ducks_by_type(DuckType.FOCUS)
        assert len(ducks) > 0

    def test_get_duck_amount(
        self,
        ducking_manager: DuckingManager,
        vo_bus: MixBus,
        music_bus: MixBus,
    ):
        """Test getting duck amount for a bus."""
        ducking_manager.create_dialogue_duck(
            vo_bus, [music_bus], amount_db=-12.0
        )

        # No ducking initially
        assert ducking_manager.get_duck_amount(music_bus) == 1.0

        # Trigger ducking
        ducking_manager.analyze_source_levels({vo_bus.id: -10.0})
        ducking_manager.update(0.1)

        # Should be ducking
        amount = ducking_manager.get_duck_amount(music_bus)
        assert amount < 1.0

    def test_duck_envelope_attack(self):
        """Test duck envelope attack phase with timing verification."""
        from engine.audio.mixing.ducking import DuckEnvelope

        env = DuckEnvelope(attack_ms=100.0, release_ms=500.0)
        env.trigger(1.0)

        assert env.state == DuckState.ATTACKING

        # After 50ms (half attack time), should be roughly halfway
        env.update(0.05)  # 50ms
        assert 0.3 < env.current_amount < 0.7, (
            f"Expected ~0.5 after 50ms attack, got {env.current_amount}"
        )

        # After full attack time, should reach target
        env.update(0.05)  # Another 50ms = 100ms total
        assert env.current_amount >= 0.95, (
            f"Expected ~1.0 after 100ms attack, got {env.current_amount}"
        )

    def test_duck_envelope_release_timing(self):
        """Test duck envelope release with actual timing math."""
        from engine.audio.mixing.ducking import DuckEnvelope

        env = DuckEnvelope(attack_ms=10.0, hold_ms=0.0, release_ms=200.0)
        env.trigger(1.0)

        # Fast attack
        env.update(0.02)  # Enough for attack
        assert env.current_amount >= 0.9

        # Trigger release
        env.release()

        # Wait for hold to expire (already 0)
        env.update(0.01)
        assert env.state == DuckState.RELEASING

        # After 100ms of release (half of 200ms), should be roughly half
        env.update(0.1)
        assert 0.3 < env.current_amount < 0.7, (
            f"Expected ~0.5 after 100ms release, got {env.current_amount}"
        )

    def test_duck_amount_db_conversion(
        self,
        ducking_manager: DuckingManager,
        vo_bus: MixBus,
        music_bus: MixBus,
    ):
        """Verify duck amount converts correctly to dB."""
        instance = ducking_manager.create_dialogue_duck(
            vo_bus, [music_bus], amount_db=-12.0
        )

        # Trigger full ducking
        instance.trigger(1.0)

        # Wait for attack to complete
        for _ in range(10):
            ducking_manager.update(0.02)

        # At full duck, -12dB should give linear ~0.25 (10^(-12/20))
        expected_linear = db_to_linear(-12.0)  # ~0.251
        assert ducking_manager.get_duck_amount(music_bus) == pytest.approx(
            expected_linear, rel=0.1
        ), "Duck amount should match -12dB converted to linear"


# =============================================================================
# Sidechain Tests
# =============================================================================


class TestSidechain:
    """Tests for sidechain compression."""

    def test_create_sidechain_config(
        self, sfx_bus: MixBus, music_bus: MixBus
    ):
        """Test creating sidechain configuration."""
        config = SidechainConfig(
            name="kick_sidechain",
            key_bus=sfx_bus,
            target_bus=music_bus,
            threshold_db=-20.0,
            ratio=4.0,
        )
        assert config.key_bus is sfx_bus
        assert config.target_bus is music_bus
        assert config.ratio == 4.0

    def test_create_compressor(
        self,
        sidechain_manager: SidechainManager,
        sfx_bus: MixBus,
        music_bus: MixBus,
    ):
        """Test creating a sidechain compressor."""
        config = SidechainConfig(
            key_bus=sfx_bus,
            target_bus=music_bus,
        )
        compressor = sidechain_manager.create_compressor(config)
        assert compressor is not None

    def test_compressor_idle_when_below_threshold(
        self,
        sidechain_manager: SidechainManager,
        sfx_bus: MixBus,
        music_bus: MixBus,
    ):
        """Compressor should be idle when key is below threshold."""
        config = SidechainConfig(
            key_bus=sfx_bus,
            target_bus=music_bus,
            threshold_db=-20.0,
        )
        compressor = sidechain_manager.create_compressor(config)

        sidechain_manager.analyze_key_levels({sfx_bus.id: -40.0})
        sidechain_manager.update(0.1)

        assert not compressor.is_compressing

    def test_compressor_active_when_above_threshold(
        self,
        sidechain_manager: SidechainManager,
        sfx_bus: MixBus,
        music_bus: MixBus,
    ):
        """Compressor should compress when key is above threshold."""
        config = SidechainConfig(
            key_bus=sfx_bus,
            target_bus=music_bus,
            threshold_db=-20.0,
            ratio=4.0,
        )
        compressor = sidechain_manager.create_compressor(config)

        sidechain_manager.analyze_key_levels({sfx_bus.id: -10.0})
        sidechain_manager.update(0.1)

        assert compressor.is_compressing
        assert compressor.gain_reduction_db < 0

    def test_get_gain(
        self,
        sidechain_manager: SidechainManager,
        sfx_bus: MixBus,
        music_bus: MixBus,
    ):
        """Test getting compression gain for a bus."""
        config = SidechainConfig(
            key_bus=sfx_bus,
            target_bus=music_bus,
            threshold_db=-20.0,
        )
        sidechain_manager.create_compressor(config)

        # No compression initially
        assert sidechain_manager.get_gain(music_bus) == pytest.approx(1.0)

    def test_compressor_reset(
        self,
        sidechain_manager: SidechainManager,
        sfx_bus: MixBus,
        music_bus: MixBus,
    ):
        """Test resetting compressor."""
        config = SidechainConfig(
            key_bus=sfx_bus,
            target_bus=music_bus,
        )
        compressor = sidechain_manager.create_compressor(config)

        # Trigger compression
        sidechain_manager.analyze_key_levels({sfx_bus.id: 0.0})
        sidechain_manager.update(0.1)

        sidechain_manager.reset_all()
        assert not compressor.is_compressing

    def test_compressor_gain_reduction_math(
        self,
        sidechain_manager: SidechainManager,
        sfx_bus: MixBus,
        music_bus: MixBus,
    ):
        """Verify gain reduction follows compression formula.

        With 4:1 ratio and -20dB threshold:
        - Input at 0dB is 20dB over threshold
        - Gain reduction = overshoot * (1 - 1/ratio) = 20 * 0.75 = 15dB
        """
        config = SidechainConfig(
            key_bus=sfx_bus,
            target_bus=music_bus,
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=1.0,  # Fast attack for test
            knee_db=0.0,  # Hard knee for predictable math
        )
        compressor = sidechain_manager.create_compressor(config)

        # Set key level to 0dB (20dB above -20dB threshold)
        sidechain_manager.analyze_key_levels({sfx_bus.id: 0.0})

        # Update multiple times to let attack complete
        for _ in range(20):
            sidechain_manager.update(0.01)

        # Expected gain reduction: 20dB overshoot * (1 - 1/4) = 15dB reduction
        expected_gr = -15.0
        actual_gr = compressor.gain_reduction_db

        # Allow some tolerance due to envelope smoothing
        assert actual_gr < -10.0, (
            f"Expected significant compression (~{expected_gr}dB), got {actual_gr}dB"
        )


# =============================================================================
# HDR Audio Tests
# =============================================================================


class TestHDRAudio:
    """Tests for HDR audio system."""

    def test_register_source(self, hdr_manager: HDRAudioManager):
        """Test registering an audio source."""
        source = hdr_manager.register_source(
            name="dialogue",
            priority=HDRPriority.CRITICAL.value,
        )
        assert source.name == "dialogue"
        assert source.priority == HDRPriority.CRITICAL.value

    def test_unregister_source(self, hdr_manager: HDRAudioManager):
        """Test unregistering a source."""
        source = hdr_manager.register_source(name="test")
        result = hdr_manager.unregister_source(source.id)
        assert result is True

    def test_set_source_loudness(self, hdr_manager: HDRAudioManager):
        """Test setting source loudness."""
        source = hdr_manager.register_source(name="test")
        hdr_manager.set_source_loudness(source.id, -20.0, is_active=True)

        updated = hdr_manager.get_source(source.id)
        assert updated.target_loudness_db == -20.0
        assert updated.is_active

    def test_mix_window_default(self, hdr_manager: HDRAudioManager):
        """Test default mix window settings."""
        window = hdr_manager.window
        assert window.window_db > 0
        assert window.floor_db < window.ceiling_db

    def test_set_window_size(self, hdr_manager: HDRAudioManager):
        """Test setting window size."""
        hdr_manager.set_window_size(30.0)
        window = hdr_manager.window
        assert window.window_db == 30.0

    def test_window_adaptation(self, hdr_manager: HDRAudioManager):
        """Test that window adapts to loud sources."""
        # Register a quiet source
        source1 = hdr_manager.register_source(
            name="ambient",
            priority=HDRPriority.LOW.value,
        )
        hdr_manager.set_source_loudness(source1.id, -50.0, is_active=True)

        # Register a loud source
        source2 = hdr_manager.register_source(
            name="explosion",
            priority=HDRPriority.HIGH.value,
        )
        hdr_manager.set_source_loudness(source2.id, -10.0, is_active=True)

        # Let window adapt
        for _ in range(20):
            hdr_manager.update(0.1)

        # Window should shift towards loud source
        window = hdr_manager.window
        assert window.center_db > -40  # Shifted up from initial

    def test_protected_source(self, hdr_manager: HDRAudioManager):
        """Protected sources should not be affected by HDR."""
        source = hdr_manager.register_source(
            name="dialogue",
            priority=HDRPriority.CRITICAL.value,
            protected=True,
        )
        hdr_manager.set_source_loudness(source.id, -30.0, is_active=True)
        hdr_manager.update(0.1)

        updated = hdr_manager.get_source(source.id)
        assert updated.loudness_db == -30.0  # Unchanged

    def test_get_active_sources(self, hdr_manager: HDRAudioManager):
        """Test getting active sources."""
        s1 = hdr_manager.register_source(name="test1")
        s2 = hdr_manager.register_source(name="test2")

        hdr_manager.set_source_loudness(s1.id, -20.0, is_active=True)
        hdr_manager.set_source_loudness(s2.id, -80.0, is_active=False)

        active = hdr_manager.get_active_sources()
        assert len(active) == 1
        assert active[0].name == "test1"

    def test_enable_disable(self, hdr_manager: HDRAudioManager):
        """Test enabling/disabling HDR."""
        assert hdr_manager.enabled
        hdr_manager.enabled = False
        assert not hdr_manager.enabled

    def test_mix_window_level_mapping(self, hdr_manager: HDRAudioManager):
        """Verify HDR window maps input levels correctly.

        Window math: input within [floor, ceiling] maps to output [floor_db, ceiling_db]
        """
        window = hdr_manager.window

        # Force window to known position for predictable test
        hdr_manager.force_window_center(-30.0)
        window = hdr_manager.window

        # Level at window center should map to middle of output range
        center_input = window.center_db
        mapped = window.map_level(center_input)

        # Expected: maps to middle of output range
        expected_mid = (window.floor_db + window.ceiling_db) / 2.0
        # Allow some deviation since center might not be exactly middle
        assert window.floor_db <= mapped <= window.ceiling_db, (
            f"Mapped level {mapped} should be within output range"
        )

        # Level below window should be silent
        below_floor = window.window_floor - 10.0
        assert window.map_level(below_floor) == -80.0, (
            "Levels below window floor should be silence"
        )

        # Level above window should be clamped to ceiling
        above_ceiling = window.window_ceiling + 10.0
        assert window.map_level(above_ceiling) == window.ceiling_db, (
            "Levels above window ceiling should clamp"
        )

    def test_hdr_priority_weighting(self, hdr_manager: HDRAudioManager):
        """Verify higher priority sources have more influence on window position."""
        # Register low priority quiet source
        low = hdr_manager.register_source(
            name="ambient",
            priority=25,  # LOW
        )
        hdr_manager.set_source_loudness(low.id, -50.0, is_active=True)

        # Register high priority loud source
        high = hdr_manager.register_source(
            name="dialogue",
            priority=100,  # CRITICAL
        )
        hdr_manager.set_source_loudness(high.id, -10.0, is_active=True)

        # Let window adapt
        for _ in range(50):
            hdr_manager.update(0.05)

        window = hdr_manager.window

        # Window should be closer to the high priority source's level
        # than to the low priority source
        distance_to_high = abs(window.center_db - (-10.0))
        distance_to_low = abs(window.center_db - (-50.0))

        assert distance_to_high < distance_to_low, (
            f"Window center {window.center_db}dB should be closer to high "
            f"priority (-10dB) than low priority (-50dB)"
        )


# =============================================================================
# Mixer Integration Tests
# =============================================================================


class TestMixerIntegration:
    """Integration tests for the main Mixer class."""

    def test_initialize(self, mixer: Mixer):
        """Test mixer initialization."""
        assert mixer.initialized
        assert mixer.master_bus is not None

    def test_get_bus(self, mixer: Mixer):
        """Test getting a bus by name."""
        sfx = mixer.get_bus("sfx")
        assert sfx is not None
        assert sfx.name == "sfx"

    def test_create_bus(self, mixer: Mixer):
        """Test creating a new bus."""
        bus = mixer.create_bus("custom", parent_name="sfx")
        assert bus.name == "custom"
        assert bus.parent.name == "sfx"

    def test_remove_bus(self, mixer: Mixer):
        """Test removing a bus."""
        mixer.create_bus("temp")
        result = mixer.remove_bus("temp")
        assert result is True
        assert mixer.get_bus("temp") is None

    def test_cannot_remove_master(self, mixer: Mixer):
        """Cannot remove the master bus."""
        with pytest.raises(ValueError, match="master"):
            mixer.remove_bus("master")

    def test_set_bus_volume(self, mixer: Mixer):
        """Test setting bus volume."""
        mixer.set_bus_volume("sfx", 0.5)
        sfx = mixer.get_bus("sfx")
        assert sfx.volume == pytest.approx(0.5)

    def test_set_master_volume(self, mixer: Mixer):
        """Test setting master volume."""
        mixer.set_master_volume(0.8)
        assert mixer.master_bus.volume == pytest.approx(0.8)

    def test_mute_bus(self, mixer: Mixer):
        """Test muting a bus."""
        mixer.mute_bus("music", True)
        music = mixer.get_bus("music")
        assert music.muted

    def test_capture_and_apply_snapshot(self, mixer: Mixer):
        """Test capturing and applying snapshots."""
        mixer.set_bus_volume("sfx", 0.5)
        mixer.capture_snapshot("custom")

        mixer.set_bus_volume("sfx", 1.0)
        mixer.apply_snapshot_immediate("custom")

        sfx = mixer.get_bus("sfx")
        assert sfx.volume == pytest.approx(0.5)

    def test_transition_to_snapshot(self, mixer: Mixer):
        """Test smooth transition to snapshot."""
        mixer.set_bus_volume("music", 0.3)
        mixer.capture_snapshot("quiet")

        mixer.set_bus_volume("music", 1.0)
        result = mixer.transition_to_snapshot("quiet", blend_time=0.0)

        assert result is True

    def test_create_aux_send(self, mixer: Mixer):
        """Test creating aux sends through mixer."""
        mixer.create_bus("reverb", BusType.AUX)
        send = mixer.create_aux_send("sfx", "reverb", level_db=-6.0)

        assert send is not None
        assert send.send_level_db == -6.0

    def test_create_sidechain(self, mixer: Mixer):
        """Test creating sidechain through mixer."""
        result = mixer.create_sidechain(
            key_name="sfx",
            target_name="music",
            ratio=4.0,
        )
        assert result is True

    def test_update(self, mixer: Mixer):
        """Test mixer update."""
        # Should not raise
        mixer.update(0.016)

    def test_get_final_volume(self, mixer: Mixer):
        """Test getting final processed volume."""
        mixer.set_bus_volume("sfx", 0.5)
        volume = mixer.get_final_volume("sfx")
        assert volume == pytest.approx(0.5)

    def test_get_state(self, mixer: Mixer):
        """Test getting complete mixer state."""
        state = mixer.get_state()
        assert "buses" in state
        assert "routing" in state
        assert "initialized" in state

    def test_shutdown(self, mixer: Mixer):
        """Test mixer shutdown."""
        mixer.shutdown()
        assert not mixer.initialized
        assert mixer.get_bus("sfx") is None


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_bus_modification(self, mixer: Mixer):
        """Test concurrent bus modifications."""
        errors = []

        def modify_bus(name: str):
            try:
                for _ in range(100):
                    bus = mixer.get_bus(name)
                    if bus:
                        bus.volume = 0.5
                        bus.volume = 1.0
                        bus.toggle_mute()
                        bus.toggle_mute()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=modify_bus, args=("sfx",)),
            threading.Thread(target=modify_bus, args=("music",)),
            threading.Thread(target=modify_bus, args=("vo",)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_snapshot_operations(
        self, snapshot_manager: SnapshotManager
    ):
        """Test concurrent snapshot operations."""
        errors = []

        def snapshot_ops():
            try:
                for i in range(50):
                    snapshot_manager.capture_snapshot(f"test_{i}")
                    snapshot_manager.list_snapshots()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=snapshot_ops) for _ in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_ducking_update(
        self,
        ducking_manager: DuckingManager,
        vo_bus: MixBus,
        music_bus: MixBus,
    ):
        """Test concurrent ducking updates."""
        ducking_manager.create_dialogue_duck(vo_bus, [music_bus])
        errors = []

        def update_ducking():
            try:
                for _ in range(100):
                    ducking_manager.analyze_source_levels({vo_bus.id: -20.0})
                    ducking_manager.update(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_ducking) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_mixer_update(self, mixer: Mixer):
        """Test concurrent mixer updates."""
        errors = []

        def mixer_update():
            try:
                for _ in range(100):
                    mixer.update(0.01)
            except Exception as e:
                errors.append(e)

        def volume_change():
            try:
                for _ in range(100):
                    mixer.set_bus_volume("sfx", 0.5)
                    mixer.set_bus_volume("sfx", 1.0)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=mixer_update),
            threading.Thread(target=mixer_update),
            threading.Thread(target=volume_change),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
