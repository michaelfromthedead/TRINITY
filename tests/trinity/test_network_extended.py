"""
Tests for Tier 50: NETWORK_EXTENDED decorators.
"""

import pytest

from trinity.decorators.network_extended import (
    BandwidthPriorityConfig,
    InterestConfig,
    ServerReconcileConfig,
    SnapshotInterpolationConfig,
    bandwidth_priority,
    interest,
    server_reconcile,
    snapshot_interpolation,
)
from trinity.decorators.registry import Tier, registry


class TestInterest:
    """Test @interest decorator."""

    def test_basic_radius_interest(self):
        """Test basic radius-based interest."""

        @interest(type="radius", radius=100.0)
        class Entity:
            pass

        assert hasattr(Entity, "_interest")
        assert Entity._interest is True
        assert Entity._interest_type == "radius"
        assert Entity._interest_radius == 100.0
        assert Entity._interest_always_relevant_to_owner is True

        # Check config
        assert isinstance(Entity._interest_config, InterestConfig)
        assert Entity._interest_config.type == "radius"
        assert Entity._interest_config.radius == 100.0

    def test_grid_interest(self):
        """Test grid-based interest."""

        @interest(type="grid")
        class Entity:
            pass

        assert Entity._interest_type == "grid"
        assert Entity._interest_radius is None

    def test_custom_interest(self):
        """Test custom interest management."""

        @interest(type="custom", always_relevant_to_owner=False)
        class Entity:
            pass

        assert Entity._interest_type == "custom"
        assert Entity._interest_always_relevant_to_owner is False

    def test_invalid_type(self):
        """Test invalid interest type."""
        with pytest.raises(ValueError, match="Invalid interest type"):

            @interest(type="invalid")
            class Entity:
                pass

    def test_radius_without_value(self):
        """Test radius type without radius value."""
        with pytest.raises(ValueError, match="radius must be provided"):

            @interest(type="radius")
            class Entity:
                pass

    def test_radius_zero_or_negative(self):
        """Test radius with invalid value."""
        with pytest.raises(ValueError, match="radius must be > 0"):

            @interest(type="radius", radius=0)
            class Entity:
                pass

        with pytest.raises(ValueError, match="radius must be > 0"):

            @interest(type="radius", radius=-10.0)
            class Entity:
                pass

    def test_tags_and_registry(self):
        """Test that tags and registry are set."""

        @interest(type="radius", radius=50.0)
        class Entity:
            pass

        assert hasattr(Entity, "_tags")
        assert Entity._tags.get("interest") is True

        assert hasattr(Entity, "_registries")
        assert "network_extended" in Entity._registries

    def test_registry_entry(self):
        """Test that decorator is registered in the registry."""
        spec = registry.get("interest")
        assert spec is not None
        assert spec.name == "interest"
        assert spec.tier == Tier.NETWORK_EXTENDED


class TestBandwidthPriority:
    """Test @bandwidth_priority decorator."""

    def test_basic_priority(self):
        """Test basic bandwidth priority."""

        @bandwidth_priority(priority=10)
        class Entity:
            pass

        assert hasattr(Entity, "_bandwidth_priority")
        assert Entity._bandwidth_priority is True
        assert Entity._bandwidth_priority_value == 10
        assert Entity._bandwidth_max_bps is None

    def test_with_max_bps(self):
        """Test bandwidth priority with max bps limit."""

        @bandwidth_priority(priority=5, max_bps=10000)
        class Entity:
            pass

        assert Entity._bandwidth_priority_value == 5
        assert Entity._bandwidth_max_bps == 10000

        # Check config
        assert isinstance(Entity._bandwidth_priority_config, BandwidthPriorityConfig)
        assert Entity._bandwidth_priority_config.priority == 5
        assert Entity._bandwidth_priority_config.max_bps == 10000

    def test_default_priority(self):
        """Test default priority value."""

        @bandwidth_priority()
        class Entity:
            pass

        assert Entity._bandwidth_priority_value == 0

    def test_invalid_max_bps(self):
        """Test invalid max_bps values."""
        with pytest.raises(ValueError, match="max_bps must be > 0"):

            @bandwidth_priority(max_bps=0)
            class Entity:
                pass

        with pytest.raises(ValueError, match="max_bps must be > 0"):

            @bandwidth_priority(max_bps=-100)
            class Entity:
                pass

    def test_registry_entry(self):
        """Test that decorator is registered."""
        spec = registry.get("bandwidth_priority")
        assert spec is not None
        assert spec.tier == Tier.NETWORK_EXTENDED


class TestSnapshotInterpolation:
    """Test @snapshot_interpolation decorator."""

    def test_basic_interpolation(self):
        """Test basic snapshot interpolation."""

        @snapshot_interpolation()
        class Entity:
            pass

        assert hasattr(Entity, "_snapshot_interpolation")
        assert Entity._snapshot_interpolation is True
        assert Entity._snapshot_buffer_size_ms == 100.0
        assert Entity._snapshot_interp_delay_ms == 100.0

    def test_custom_values(self):
        """Test custom interpolation values."""

        @snapshot_interpolation(buffer_size_ms=200.0, interp_delay_ms=150.0)
        class Entity:
            pass

        assert Entity._snapshot_buffer_size_ms == 200.0
        assert Entity._snapshot_interp_delay_ms == 150.0

        # Check config
        config = Entity._snapshot_interpolation_config
        assert isinstance(config, SnapshotInterpolationConfig)
        assert config.buffer_size_ms == 200.0
        assert config.interp_delay_ms == 150.0

    def test_invalid_buffer_size(self):
        """Test invalid buffer size."""
        with pytest.raises(ValueError, match="buffer_size_ms must be > 0"):

            @snapshot_interpolation(buffer_size_ms=0)
            class Entity:
                pass

        with pytest.raises(ValueError, match="buffer_size_ms must be > 0"):

            @snapshot_interpolation(buffer_size_ms=-50.0)
            class Entity:
                pass

    def test_invalid_interp_delay(self):
        """Test invalid interpolation delay."""
        with pytest.raises(ValueError, match="interp_delay_ms must be > 0"):

            @snapshot_interpolation(interp_delay_ms=0)
            class Entity:
                pass

    def test_registry_entry(self):
        """Test registry entry."""
        spec = registry.get("snapshot_interpolation")
        assert spec is not None
        assert spec.tier == Tier.NETWORK_EXTENDED


class TestServerReconcile:
    """Test @server_reconcile decorator."""

    def test_basic_reconcile(self):
        """Test basic server reconciliation."""

        @server_reconcile()
        class Entity:
            pass

        assert hasattr(Entity, "_server_reconcile")
        assert Entity._server_reconcile is True
        assert Entity._server_reconcile_max_frames == 10
        assert Entity._server_reconcile_snap_threshold == 0.5

    def test_custom_values(self):
        """Test custom reconciliation values."""

        @server_reconcile(max_reconcile_frames=20, snap_threshold=1.0)
        class Entity:
            pass

        assert Entity._server_reconcile_max_frames == 20
        assert Entity._server_reconcile_snap_threshold == 1.0

        # Check config
        config = Entity._server_reconcile_config
        assert isinstance(config, ServerReconcileConfig)
        assert config.max_reconcile_frames == 20
        assert config.snap_threshold == 1.0

    def test_invalid_max_frames(self):
        """Test invalid max reconcile frames."""
        with pytest.raises(ValueError, match="max_reconcile_frames must be > 0"):

            @server_reconcile(max_reconcile_frames=0)
            class Entity:
                pass

        with pytest.raises(ValueError, match="max_reconcile_frames must be > 0"):

            @server_reconcile(max_reconcile_frames=-5)
            class Entity:
                pass

    def test_invalid_snap_threshold(self):
        """Test invalid snap threshold."""
        with pytest.raises(ValueError, match="snap_threshold must be > 0"):

            @server_reconcile(snap_threshold=0)
            class Entity:
                pass

        with pytest.raises(ValueError, match="snap_threshold must be > 0"):

            @server_reconcile(snap_threshold=-0.5)
            class Entity:
                pass

    def test_registry_entry(self):
        """Test registry entry."""
        spec = registry.get("server_reconcile")
        assert spec is not None
        assert spec.tier == Tier.NETWORK_EXTENDED


class TestDecoratorComposition:
    """Test decorator composition and stacking."""

    def test_multiple_decorators(self):
        """Test applying multiple network_extended decorators."""

        @server_reconcile(max_reconcile_frames=15)
        @snapshot_interpolation(buffer_size_ms=150.0)
        @bandwidth_priority(priority=10, max_bps=5000)
        @interest(type="radius", radius=200.0)
        class NetworkedEntity:
            pass

        # Check all decorators applied
        assert NetworkedEntity._interest is True
        assert NetworkedEntity._bandwidth_priority is True
        assert NetworkedEntity._snapshot_interpolation is True
        assert NetworkedEntity._server_reconcile is True

        # Check values
        assert NetworkedEntity._interest_radius == 200.0
        assert NetworkedEntity._bandwidth_priority_value == 10
        assert NetworkedEntity._snapshot_buffer_size_ms == 150.0
        assert NetworkedEntity._server_reconcile_max_frames == 15

    def test_applied_decorators_tracking(self):
        """Test that applied decorators are tracked."""

        @interest(type="grid")
        @bandwidth_priority(priority=5)
        class Entity:
            pass

        assert hasattr(Entity, "_applied_decorators")
        assert "interest" in Entity._applied_decorators
        assert "bandwidth_priority" in Entity._applied_decorators


class TestRegistryIntegration:
    """Test integration with decorator registry."""

    def test_all_decorators_registered(self):
        """Test that all network_extended decorators are registered."""
        tier_decorators = registry.by_tier(Tier.NETWORK_EXTENDED)
        decorator_names = {spec.name for spec in tier_decorators}

        assert "interest" in decorator_names
        assert "bandwidth_priority" in decorator_names
        assert "snapshot_interpolation" in decorator_names
        assert "server_reconcile" in decorator_names

    def test_tier_ordering(self):
        """Test that NETWORK_EXTENDED has correct tier value."""
        assert Tier.NETWORK_EXTENDED == 50
