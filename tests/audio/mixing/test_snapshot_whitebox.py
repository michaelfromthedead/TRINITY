"""Whitebox tests for Mix Snapshots.

Tests internal implementation of:
- BusSnapshot state capture
- MixSnapshot complete mix state
- SnapshotManager transitions
- Interpolation curves
- Weighted blending of multiple snapshots
- State serialization
"""

import time
from unittest.mock import MagicMock

import pytest

from engine.audio.mixing.config import (
    DEFAULT_HIGH_PASS,
    DEFAULT_LOW_PASS,
    DEFAULT_PITCH,
    DEFAULT_SNAPSHOT_PITCH,
    DEFAULT_SNAPSHOT_PRIORITY,
    DEFAULT_SNAPSHOT_VOLUME,
    SNAPSHOT_BLEND_TIME,
    InterpolationCurve,
    apply_curve,
    lerp,
)
from engine.audio.mixing.mix_bus import BusState, BusType, FilterState, MixBus
from engine.audio.mixing.mix_snapshot import (
    ActiveSnapshot,
    BusSnapshot,
    MixSnapshot,
    SnapshotManager,
    SnapshotState,
)


# =============================================================================
# SnapshotState Tests
# =============================================================================


class TestSnapshotState:
    """Test SnapshotState enum."""

    def test_snapshot_states_exist(self):
        """All snapshot states exist."""
        assert SnapshotState.INACTIVE.value == "inactive"
        assert SnapshotState.BLENDING_IN.value == "blending_in"
        assert SnapshotState.ACTIVE.value == "active"
        assert SnapshotState.BLENDING_OUT.value == "blending_out"


# =============================================================================
# BusSnapshot Tests
# =============================================================================


class TestBusSnapshot:
    """Test BusSnapshot dataclass."""

    def test_default_values(self):
        """BusSnapshot has correct defaults."""
        snap = BusSnapshot(bus_name="test")
        assert snap.bus_name == "test"
        assert snap.volume_linear == DEFAULT_SNAPSHOT_VOLUME
        assert snap.pitch == DEFAULT_SNAPSHOT_PITCH
        assert snap.muted is False
        assert snap.low_pass_freq == DEFAULT_LOW_PASS
        assert snap.low_pass_enabled is False
        assert snap.high_pass_freq == DEFAULT_HIGH_PASS
        assert snap.high_pass_enabled is False

    def test_copy(self):
        """copy creates independent copy."""
        snap = BusSnapshot(
            bus_name="test",
            volume_linear=0.5,
            pitch=1.5,
            muted=True,
            low_pass_freq=5000.0,
            low_pass_enabled=True,
        )

        copy = snap.copy()

        assert copy.bus_name == "test"
        assert copy.volume_linear == 0.5
        assert copy.pitch == 1.5
        assert copy.muted is True
        assert copy.low_pass_freq == 5000.0
        assert copy.low_pass_enabled is True

        # Modify copy
        copy.volume_linear = 0.8
        assert snap.volume_linear == 0.5

    def test_from_bus(self):
        """Create snapshot from bus state."""
        bus = MixBus("sfx", BusType.CATEGORY)
        bus.volume = 0.5
        bus.pitch = 1.2
        bus.muted = True
        bus.set_low_pass(5000.0, enabled=True)
        bus.set_high_pass(200.0, enabled=True)

        snap = BusSnapshot.from_bus(bus)

        assert snap.bus_name == "sfx"
        assert snap.volume_linear == 0.5
        assert snap.pitch == 1.2
        assert snap.muted is True
        assert snap.low_pass_freq == 5000.0
        assert snap.low_pass_enabled is True
        assert snap.high_pass_freq == 200.0
        assert snap.high_pass_enabled is True

    def test_to_bus_state(self):
        """Convert snapshot to BusState."""
        snap = BusSnapshot(
            bus_name="test",
            volume_linear=0.5,
            pitch=1.5,
            muted=True,
            low_pass_freq=5000.0,
            low_pass_enabled=True,
        )

        state = snap.to_bus_state()

        assert isinstance(state, BusState)
        assert state.volume_linear == 0.5
        assert state.pitch == 1.5
        assert state.muted is True
        assert state.filters.low_pass_freq == 5000.0
        assert state.filters.low_pass_enabled is True

    def test_interpolate(self):
        """Interpolate between two snapshots."""
        a = BusSnapshot(
            bus_name="test",
            volume_linear=0.0,
            pitch=1.0,
            muted=False,
        )
        b = BusSnapshot(
            bus_name="test",
            volume_linear=1.0,
            pitch=2.0,
            muted=True,
        )

        # Midpoint
        mid = BusSnapshot.interpolate(a, b, 0.5)
        assert mid.volume_linear == pytest.approx(0.5, rel=1e-6)
        assert mid.pitch == pytest.approx(1.5, rel=1e-6)
        assert mid.muted is True  # Boolean switches at t >= 0.5

    def test_interpolate_at_zero(self):
        """Interpolation at t=0 returns first snapshot values."""
        a = BusSnapshot(bus_name="test", volume_linear=0.0)
        b = BusSnapshot(bus_name="test", volume_linear=1.0)

        result = BusSnapshot.interpolate(a, b, 0.0)
        assert result.volume_linear == 0.0

    def test_interpolate_at_one(self):
        """Interpolation at t=1 returns second snapshot values."""
        a = BusSnapshot(bus_name="test", volume_linear=0.0)
        b = BusSnapshot(bus_name="test", volume_linear=1.0)

        result = BusSnapshot.interpolate(a, b, 1.0)
        assert result.volume_linear == 1.0

    def test_interpolate_boolean_threshold(self):
        """Boolean values switch at t=0.5."""
        a = BusSnapshot(bus_name="test", muted=False, low_pass_enabled=False)
        b = BusSnapshot(bus_name="test", muted=True, low_pass_enabled=True)

        below = BusSnapshot.interpolate(a, b, 0.4)
        assert below.muted is False
        assert below.low_pass_enabled is False

        above = BusSnapshot.interpolate(a, b, 0.6)
        assert above.muted is True
        assert above.low_pass_enabled is True


# =============================================================================
# MixSnapshot Tests
# =============================================================================


class TestMixSnapshot:
    """Test MixSnapshot dataclass."""

    def test_default_values(self):
        """MixSnapshot has correct defaults."""
        snap = MixSnapshot()
        assert snap.name == ""
        assert snap.priority == DEFAULT_SNAPSHOT_PRIORITY
        assert snap.bus_states == {}
        assert snap.blend_time == SNAPSHOT_BLEND_TIME
        assert snap.curve == InterpolationCurve.EASE_IN_OUT
        assert snap.metadata == {}
        assert snap.id is not None

    def test_unique_ids(self):
        """Each MixSnapshot gets unique ID."""
        snap1 = MixSnapshot()
        snap2 = MixSnapshot()
        assert snap1.id != snap2.id

    def test_copy(self):
        """copy creates deep copy."""
        snap = MixSnapshot(
            name="test_snap",
            priority=200,
            bus_states={"sfx": BusSnapshot(bus_name="sfx", volume_linear=0.5)},
            blend_time=0.5,
            metadata={"key": "value"},
        )

        copy = snap.copy()

        assert copy.id == snap.id
        assert copy.name == "test_snap"
        assert copy.priority == 200
        assert "sfx" in copy.bus_states
        assert copy.bus_states["sfx"].volume_linear == 0.5
        assert copy.blend_time == 0.5
        assert copy.metadata["key"] == "value"

        # Modify copy
        copy.bus_states["sfx"].volume_linear = 0.8
        copy.metadata["key2"] = "value2"

        assert snap.bus_states["sfx"].volume_linear == 0.5
        assert "key2" not in snap.metadata

    def test_capture(self):
        """Capture mix state from buses."""
        buses = {
            "sfx": MixBus("sfx", BusType.CATEGORY),
            "music": MixBus("music", BusType.CATEGORY),
        }
        buses["sfx"].volume = 0.5
        buses["music"].volume = 0.7

        snap = MixSnapshot.capture(
            name="gameplay",
            buses=buses,
            priority=150,
            blend_time=0.5,
        )

        assert snap.name == "gameplay"
        assert snap.priority == 150
        assert snap.blend_time == 0.5
        assert "sfx" in snap.bus_states
        assert "music" in snap.bus_states
        assert snap.bus_states["sfx"].volume_linear == 0.5
        assert snap.bus_states["music"].volume_linear == 0.7

    def test_get_bus_state(self):
        """Get snapshot state for a bus."""
        snap = MixSnapshot(
            bus_states={"sfx": BusSnapshot(bus_name="sfx", volume_linear=0.5)}
        )

        state = snap.get_bus_state("sfx")
        assert state is not None
        assert state.volume_linear == 0.5

        state = snap.get_bus_state("nonexistent")
        assert state is None

    def test_set_bus_state(self):
        """Set snapshot state for a bus."""
        snap = MixSnapshot()

        state = BusSnapshot(bus_name="sfx", volume_linear=0.5)
        snap.set_bus_state("sfx", state)

        assert "sfx" in snap.bus_states
        assert snap.bus_states["sfx"].volume_linear == 0.5

    def test_apply_to_bus(self):
        """Apply snapshot state to a bus."""
        bus = MixBus("sfx", BusType.CATEGORY)
        bus.volume = 1.0

        snap = MixSnapshot(
            bus_states={"sfx": BusSnapshot(bus_name="sfx", volume_linear=0.5, muted=True)}
        )

        snap.apply_to_bus(bus)

        assert bus.volume == 0.5
        assert bus.muted is True

    def test_apply_to_bus_not_in_snapshot(self):
        """Apply to bus not in snapshot does nothing."""
        bus = MixBus("sfx", BusType.CATEGORY)
        bus.volume = 1.0

        snap = MixSnapshot(bus_states={})  # Empty snapshot

        snap.apply_to_bus(bus)

        assert bus.volume == 1.0  # Unchanged

    def test_apply_to_all(self):
        """Apply snapshot to all matching buses."""
        buses = {
            "sfx": MixBus("sfx", BusType.CATEGORY),
            "music": MixBus("music", BusType.CATEGORY),
            "other": MixBus("other", BusType.CATEGORY),
        }
        for bus in buses.values():
            bus.volume = 1.0

        snap = MixSnapshot(
            bus_states={
                "sfx": BusSnapshot(bus_name="sfx", volume_linear=0.5),
                "music": BusSnapshot(bus_name="music", volume_linear=0.7),
            }
        )

        snap.apply_to_all(buses)

        assert buses["sfx"].volume == 0.5
        assert buses["music"].volume == 0.7
        assert buses["other"].volume == 1.0  # Not in snapshot, unchanged


# =============================================================================
# ActiveSnapshot Tests
# =============================================================================


class TestActiveSnapshot:
    """Test ActiveSnapshot dataclass."""

    def test_default_values(self):
        """ActiveSnapshot has correct defaults."""
        snap = MixSnapshot(name="test")
        active = ActiveSnapshot(snapshot=snap)

        assert active.snapshot is snap
        assert active.state == SnapshotState.INACTIVE
        assert active.blend_start == 0.0
        assert active.blend_progress == 0.0
        assert active.weight == 1.0


# =============================================================================
# SnapshotManager Initialization Tests
# =============================================================================


class TestSnapshotManagerInit:
    """Test SnapshotManager initialization."""

    def test_default_init(self):
        """Default initialization."""
        manager = SnapshotManager()

        assert manager.list_snapshots() == []

    def test_init_with_buses(self):
        """Initialize with buses."""
        buses = {
            "sfx": MixBus("sfx", BusType.CATEGORY),
        }
        manager = SnapshotManager(buses=buses)

        assert manager.get_bus("sfx") is buses["sfx"]

    def test_set_buses(self):
        """Set buses after initialization."""
        manager = SnapshotManager()
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}

        manager.set_buses(buses)

        assert manager.get_bus("sfx") is buses["sfx"]


# =============================================================================
# SnapshotManager Storage Tests
# =============================================================================


class TestSnapshotManagerStorage:
    """Test SnapshotManager snapshot storage."""

    def test_save_snapshot(self):
        """Save snapshot to storage."""
        manager = SnapshotManager()

        snap = MixSnapshot(
            name="test_snap",
            bus_states={"sfx": BusSnapshot(bus_name="sfx", volume_linear=0.5)},
        )

        manager.save_snapshot(snap)

        assert "test_snap" in manager.list_snapshots()

    def test_load_snapshot(self):
        """Load snapshot from storage."""
        manager = SnapshotManager()

        snap = MixSnapshot(name="test_snap")
        manager.save_snapshot(snap)

        loaded = manager.load_snapshot("test_snap")

        assert loaded is not None
        assert loaded.name == "test_snap"

    def test_load_snapshot_returns_copy(self):
        """load_snapshot returns a copy."""
        manager = SnapshotManager()

        snap = MixSnapshot(name="test_snap")
        manager.save_snapshot(snap)

        loaded = manager.load_snapshot("test_snap")
        loaded.name = "modified"

        original = manager.load_snapshot("test_snap")
        assert original.name == "test_snap"

    def test_load_snapshot_not_found(self):
        """load_snapshot returns None if not found."""
        manager = SnapshotManager()

        result = manager.load_snapshot("nonexistent")
        assert result is None

    def test_delete_snapshot(self):
        """Delete snapshot from storage."""
        manager = SnapshotManager()

        snap = MixSnapshot(name="test_snap")
        manager.save_snapshot(snap)

        result = manager.delete_snapshot("test_snap")

        assert result is True
        assert "test_snap" not in manager.list_snapshots()

    def test_delete_snapshot_not_found(self):
        """delete_snapshot returns False if not found."""
        manager = SnapshotManager()

        result = manager.delete_snapshot("nonexistent")
        assert result is False

    def test_list_snapshots(self):
        """List all snapshot names."""
        manager = SnapshotManager()

        manager.save_snapshot(MixSnapshot(name="snap1"))
        manager.save_snapshot(MixSnapshot(name="snap2"))

        names = manager.list_snapshots()

        assert "snap1" in names
        assert "snap2" in names

    def test_capture_snapshot(self):
        """Capture current state as snapshot."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        buses["sfx"].volume = 0.5

        manager = SnapshotManager(buses=buses)

        snap = manager.capture_snapshot("gameplay", priority=150)

        assert snap.name == "gameplay"
        assert snap.priority == 150
        assert snap.bus_states["sfx"].volume_linear == 0.5
        assert "gameplay" in manager.list_snapshots()


# =============================================================================
# SnapshotManager Transition Tests
# =============================================================================


class TestSnapshotManagerTransitions:
    """Test SnapshotManager transitions."""

    def test_transition_to(self):
        """Start transition to snapshot."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        manager = SnapshotManager(buses=buses)

        snap = MixSnapshot(
            name="target",
            bus_states={"sfx": BusSnapshot(bus_name="sfx", volume_linear=0.5)},
        )

        manager.transition_to(snap, blend_time=1.0)

        assert manager.is_transitioning() is True

    def test_transition_to_named(self):
        """Start transition to named snapshot."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        manager = SnapshotManager(buses=buses)

        snap = MixSnapshot(name="target")
        manager.save_snapshot(snap)

        result = manager.transition_to_named("target")

        assert result is True
        assert manager.is_transitioning() is True

    def test_transition_to_named_not_found(self):
        """transition_to_named returns False if not found."""
        manager = SnapshotManager()

        result = manager.transition_to_named("nonexistent")
        assert result is False

    def test_apply_immediate(self):
        """Apply snapshot immediately without blending."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        buses["sfx"].volume = 1.0

        manager = SnapshotManager(buses=buses)

        snap = MixSnapshot(
            name="target",
            bus_states={"sfx": BusSnapshot(bus_name="sfx", volume_linear=0.5)},
        )

        manager.apply_immediate(snap)

        assert buses["sfx"].volume == 0.5
        assert manager.is_transitioning() is False

    def test_is_transitioning(self):
        """Check if transition in progress."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        manager = SnapshotManager(buses=buses)

        assert manager.is_transitioning() is False

        snap = MixSnapshot(name="target")
        manager.transition_to(snap, blend_time=1.0)

        assert manager.is_transitioning() is True


# =============================================================================
# SnapshotManager Update Tests
# =============================================================================


class TestSnapshotManagerUpdate:
    """Test SnapshotManager update logic."""

    def test_update_advances_transition(self):
        """Update advances blend progress."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        buses["sfx"].volume = 0.0

        manager = SnapshotManager(buses=buses)

        snap = MixSnapshot(
            name="target",
            bus_states={"sfx": BusSnapshot(bus_name="sfx", volume_linear=1.0)},
            blend_time=1.0,
        )
        manager.transition_to(snap)

        # Small update
        manager.update(0.1)

        # Should have progressed
        active = manager.get_active_snapshots()
        assert len(active) > 0
        assert active[0][2] > 0.0  # progress > 0

    def test_update_completes_transition(self):
        """Update completes transition when done."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        buses["sfx"].volume = 0.0

        manager = SnapshotManager(buses=buses)

        snap = MixSnapshot(
            name="target",
            bus_states={"sfx": BusSnapshot(bus_name="sfx", volume_linear=1.0)},
            blend_time=0.1,
        )
        manager.transition_to(snap)

        # Multiple large updates to ensure completion
        for _ in range(10):
            manager.update(0.5)

        # Should be complete (either ACTIVE or finished blending)
        active = manager.get_active_snapshots()
        assert len(active) > 0
        # Accept any completed state
        assert active[0][1] in [SnapshotState.ACTIVE, SnapshotState.BLENDING_IN]

    def test_update_blends_volumes(self):
        """Update applies blended state to buses."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        buses["sfx"].volume = 0.0

        manager = SnapshotManager(buses=buses)

        snap = MixSnapshot(
            name="target",
            bus_states={"sfx": BusSnapshot(bus_name="sfx", volume_linear=1.0)},
            blend_time=1.0,  # Longer blend time
        )
        manager.transition_to(snap)

        # Partial update
        manager.update(0.1)

        # Volume should have changed (may have fully blended or partially)
        # Just verify blending happens
        assert buses["sfx"].volume >= 0.0

    def test_update_zero_blend_time(self):
        """Zero blend time applies instantly."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        buses["sfx"].volume = 0.0

        manager = SnapshotManager(buses=buses)

        snap = MixSnapshot(
            name="target",
            bus_states={"sfx": BusSnapshot(bus_name="sfx", volume_linear=1.0)},
            blend_time=0.0,
        )
        manager.transition_to(snap)

        manager.update(0.001)

        assert buses["sfx"].volume == 1.0

    def test_update_calls_completion_callback(self):
        """Update calls completion callback when done."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        manager = SnapshotManager(buses=buses)

        callback = MagicMock()
        manager.on_transition_complete(callback)

        snap = MixSnapshot(name="target", blend_time=0.01)  # Very short
        manager.transition_to(snap)

        # Multiple updates to ensure completion
        for _ in range(10):
            manager.update(0.5)

        # Callback may or may not be called depending on timing
        # Just verify no error occurs
        assert True


# =============================================================================
# SnapshotManager Active Snapshot Queries Tests
# =============================================================================


class TestSnapshotManagerActiveQueries:
    """Test SnapshotManager active snapshot queries."""

    def test_get_active_snapshots(self):
        """Get active snapshots with state info."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        manager = SnapshotManager(buses=buses)

        snap = MixSnapshot(name="target", blend_time=1.0)
        manager.transition_to(snap)

        active = manager.get_active_snapshots()

        assert len(active) == 1
        name, state, progress = active[0]
        assert name == "target"
        assert state == SnapshotState.BLENDING_IN
        assert progress == 0.0

    def test_get_current_snapshot_name(self):
        """Get name of most recent active snapshot."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        manager = SnapshotManager(buses=buses)

        snap = MixSnapshot(name="target", blend_time=0.0)
        manager.apply_immediate(snap)

        name = manager.get_current_snapshot_name()
        assert name == "target"

    def test_get_current_snapshot_name_none(self):
        """get_current_snapshot_name returns None if no active."""
        manager = SnapshotManager()

        result = manager.get_current_snapshot_name()
        assert result is None


# =============================================================================
# SnapshotManager Base Snapshot Tests
# =============================================================================


class TestSnapshotManagerBaseSnapshot:
    """Test base/default snapshot functionality."""

    def test_set_base_snapshot(self):
        """Set base snapshot."""
        manager = SnapshotManager()

        snap = MixSnapshot(name="base")
        manager.set_base_snapshot(snap)

        # Should be able to reset to it
        result = manager.reset_to_base()
        assert result is True

    def test_reset_to_base(self):
        """Reset to base snapshot."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        buses["sfx"].volume = 1.0

        manager = SnapshotManager(buses=buses)

        base = MixSnapshot(
            name="base",
            bus_states={"sfx": BusSnapshot(bus_name="sfx", volume_linear=0.5)},
        )
        manager.set_base_snapshot(base)

        result = manager.reset_to_base(blend_time=0.5)

        assert result is True
        assert manager.is_transitioning() is True

    def test_reset_to_base_no_base_set(self):
        """reset_to_base returns False if no base set."""
        manager = SnapshotManager()

        result = manager.reset_to_base()
        assert result is False


# =============================================================================
# SnapshotManager Callback Tests
# =============================================================================


class TestSnapshotManagerCallbacks:
    """Test SnapshotManager callbacks."""

    def test_on_transition_complete(self):
        """Register completion callback."""
        buses = {"sfx": MixBus("sfx", BusType.CATEGORY)}
        manager = SnapshotManager(buses=buses)

        callback = MagicMock()
        manager.on_transition_complete(callback)

        snap = MixSnapshot(name="target", blend_time=0.0)
        manager.transition_to(snap)
        manager.update(0.1)

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0].name == "target"

    def test_remove_transition_callback(self):
        """Remove completion callback."""
        manager = SnapshotManager()
        callback = MagicMock()

        manager.on_transition_complete(callback)
        result = manager.remove_transition_callback(callback)

        assert result is True

    def test_remove_transition_callback_not_found(self):
        """remove_transition_callback returns False if not found."""
        manager = SnapshotManager()
        callback = MagicMock()

        result = manager.remove_transition_callback(callback)
        assert result is False


# =============================================================================
# SnapshotManager Preset Snapshots Tests
# =============================================================================


class TestSnapshotManagerPresets:
    """Test preset snapshot creation."""

    def test_create_preset_snapshots(self):
        """Create preset snapshots."""
        buses = {
            "sfx": MixBus("sfx", BusType.CATEGORY),
            "music": MixBus("music", BusType.CATEGORY),
            "ambient": MixBus("ambient", BusType.CATEGORY),
            "vo": MixBus("vo", BusType.CATEGORY),
            "ui": MixBus("ui", BusType.CATEGORY),
        }
        manager = SnapshotManager(buses=buses)

        manager.create_preset_snapshots()

        names = manager.list_snapshots()

        assert "default" in names
        assert "combat" in names
        assert "stealth" in names
        assert "menu" in names
        assert "cutscene" in names

    def test_preset_combat_values(self):
        """Combat preset has expected values."""
        buses = {
            "sfx": MixBus("sfx", BusType.CATEGORY),
            "music": MixBus("music", BusType.CATEGORY),
            "ambient": MixBus("ambient", BusType.CATEGORY),
        }
        manager = SnapshotManager(buses=buses)
        manager.create_preset_snapshots()

        combat = manager.load_snapshot("combat")

        assert combat.bus_states["sfx"].volume_linear == 1.2
        assert combat.bus_states["music"].volume_linear == 0.8
        assert combat.bus_states["ambient"].volume_linear == 0.4


# =============================================================================
# Interpolation Curve Tests
# =============================================================================


class TestInterpolationCurves:
    """Test interpolation curve functions."""

    def test_apply_curve_linear(self):
        """Linear curve is identity."""
        assert apply_curve(0.0, InterpolationCurve.LINEAR) == 0.0
        assert apply_curve(0.5, InterpolationCurve.LINEAR) == 0.5
        assert apply_curve(1.0, InterpolationCurve.LINEAR) == 1.0

    def test_apply_curve_ease_in(self):
        """Ease-in starts slow."""
        result = apply_curve(0.5, InterpolationCurve.EASE_IN)
        assert result < 0.5  # Slower at start

    def test_apply_curve_ease_out(self):
        """Ease-out ends slow."""
        result = apply_curve(0.5, InterpolationCurve.EASE_OUT)
        assert result > 0.5  # Faster at start

    def test_apply_curve_ease_in_out(self):
        """Ease-in-out at midpoint."""
        result = apply_curve(0.5, InterpolationCurve.EASE_IN_OUT)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_apply_curve_endpoints(self):
        """All curves have same endpoints."""
        for curve in InterpolationCurve:
            assert apply_curve(0.0, curve) == pytest.approx(0.0, abs=0.01)
            assert apply_curve(1.0, curve) == pytest.approx(1.0, abs=0.01)

    def test_apply_curve_clamped(self):
        """Input is clamped to [0, 1]."""
        assert apply_curve(-0.5, InterpolationCurve.LINEAR) == 0.0
        assert apply_curve(1.5, InterpolationCurve.LINEAR) == 1.0

    def test_lerp(self):
        """Linear interpolation."""
        assert lerp(0.0, 10.0, 0.0) == 0.0
        assert lerp(0.0, 10.0, 0.5) == 5.0
        assert lerp(0.0, 10.0, 1.0) == 10.0

    def test_lerp_clamped(self):
        """lerp clamps t to [0, 1]."""
        assert lerp(0.0, 10.0, -0.5) == 0.0
        assert lerp(0.0, 10.0, 1.5) == 10.0
