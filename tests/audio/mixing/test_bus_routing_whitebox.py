"""Whitebox tests for BusRouter audio signal routing.

Tests internal implementation of:
- Aux send creation and management
- Direct output routing
- Pre/post-fader modes
- Send level clamping
- Routing state serialization
"""

from unittest.mock import MagicMock

import pytest

from engine.audio.mixing.bus_routing import (
    AuxSend,
    BusRouter,
    DirectOutput,
    RoutingMode,
)
from engine.audio.mixing.config import (
    DEFAULT_SEND_LEVEL,
    MAX_AUX_SENDS,
    MAX_SEND_LEVEL,
    MIN_VOLUME_DB,
    db_to_linear,
)
from engine.audio.mixing.mix_bus import BusType, MixBus


# =============================================================================
# RoutingMode Tests
# =============================================================================


class TestRoutingMode:
    """Test RoutingMode enum."""

    def test_routing_modes_exist(self):
        """All routing modes exist."""
        assert RoutingMode.PARENT.value == "parent"
        assert RoutingMode.DIRECT.value == "direct"
        assert RoutingMode.PRE_FADER.value == "pre"
        assert RoutingMode.POST_FADER.value == "post"


# =============================================================================
# AuxSend Tests
# =============================================================================


class TestAuxSend:
    """Test AuxSend dataclass."""

    def test_default_values(self):
        """AuxSend has correct defaults."""
        send = AuxSend()
        assert send.source_bus is None
        assert send.target_bus is None
        assert send.send_level_db == DEFAULT_SEND_LEVEL
        assert send.mode == RoutingMode.POST_FADER
        assert send.enabled is True
        assert send.id is not None

    def test_unique_ids(self):
        """Each AuxSend gets unique ID."""
        send1 = AuxSend()
        send2 = AuxSend()
        assert send1.id != send2.id

    def test_send_level_linear_property(self):
        """send_level_linear converts dB to linear."""
        send = AuxSend(send_level_db=0.0)
        assert send.send_level_linear == pytest.approx(1.0, rel=1e-6)

        send = AuxSend(send_level_db=-6.0)
        assert send.send_level_linear == pytest.approx(0.5012, rel=1e-3)

    def test_set_level(self):
        """set_level clamps to valid range."""
        send = AuxSend()

        send.set_level(-6.0)
        assert send.send_level_db == -6.0

        send.set_level(-100.0)
        assert send.send_level_db == MIN_VOLUME_DB

        send.set_level(100.0)
        assert send.send_level_db == MAX_SEND_LEVEL

    def test_copy(self):
        """copy creates independent copy."""
        source = MixBus("source", BusType.SUB)
        target = MixBus("target", BusType.AUX)

        send = AuxSend(
            source_bus=source,
            target_bus=target,
            send_level_db=-6.0,
            mode=RoutingMode.PRE_FADER,
            enabled=False,
        )

        copy = send.copy()

        assert copy.id == send.id
        assert copy.source_bus is source
        assert copy.target_bus is target
        assert copy.send_level_db == -6.0
        assert copy.mode == RoutingMode.PRE_FADER
        assert copy.enabled is False

        # Modify copy
        copy.send_level_db = -12.0
        assert send.send_level_db == -6.0  # Original unchanged


# =============================================================================
# DirectOutput Tests
# =============================================================================


class TestDirectOutput:
    """Test DirectOutput dataclass."""

    def test_default_values(self):
        """DirectOutput has correct defaults."""
        output = DirectOutput()
        assert output.source_bus is None
        assert output.target_bus is None
        assert output.level_db == 0.0
        assert output.enabled is True
        assert output.id is not None

    def test_level_linear_property(self):
        """level_linear converts dB to linear."""
        output = DirectOutput(level_db=0.0)
        assert output.level_linear == pytest.approx(1.0, rel=1e-6)

        output = DirectOutput(level_db=-6.0)
        assert output.level_linear == pytest.approx(0.5012, rel=1e-3)


# =============================================================================
# BusRouter Aux Bus Management Tests
# =============================================================================


class TestBusRouterAuxBusManagement:
    """Test BusRouter aux bus registration."""

    def test_register_aux_bus(self):
        """Register aux bus."""
        router = BusRouter()
        aux = MixBus("reverb", BusType.AUX)

        router.register_aux_bus(aux)

        assert aux in router.get_aux_buses()

    def test_unregister_aux_bus(self):
        """Unregister aux bus."""
        router = BusRouter()
        aux = MixBus("reverb", BusType.AUX)

        router.register_aux_bus(aux)
        router.unregister_aux_bus(aux)

        assert aux not in router.get_aux_buses()

    def test_unregister_removes_sends(self):
        """Unregistering aux bus removes sends to it."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        aux = MixBus("reverb", BusType.AUX)

        router.register_aux_bus(aux)
        router.create_send(source, aux)

        router.unregister_aux_bus(aux)

        sends = router.get_sends(source)
        assert len(sends) == 0

    def test_get_aux_buses(self):
        """Get all registered aux buses."""
        router = BusRouter()
        aux1 = MixBus("reverb", BusType.AUX)
        aux2 = MixBus("delay", BusType.AUX)

        router.register_aux_bus(aux1)
        router.register_aux_bus(aux2)

        buses = router.get_aux_buses()
        assert aux1 in buses
        assert aux2 in buses


# =============================================================================
# BusRouter Aux Send Management Tests
# =============================================================================


class TestBusRouterAuxSends:
    """Test BusRouter aux send management."""

    def test_create_send(self):
        """Create aux send."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        send = router.create_send(source, target, level_db=-6.0)

        assert send is not None
        assert send.source_bus is source
        assert send.target_bus is target
        assert send.send_level_db == -6.0
        assert send.mode == RoutingMode.POST_FADER

    def test_create_send_pre_fader(self):
        """Create pre-fader aux send."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        send = router.create_send(source, target, mode=RoutingMode.PRE_FADER)

        assert send.mode == RoutingMode.PRE_FADER

    def test_create_send_level_clamped(self):
        """Send level is clamped on creation."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        send = router.create_send(source, target, level_db=-100.0)
        assert send.send_level_db == MIN_VOLUME_DB

        send2 = router.create_send(source, MixBus("delay", BusType.AUX), level_db=100.0)
        assert send2.send_level_db == MAX_SEND_LEVEL

    def test_create_send_to_self_raises(self):
        """Cannot create send to self."""
        router = BusRouter()
        bus = MixBus("sfx", BusType.CATEGORY)

        with pytest.raises(ValueError, match="itself"):
            router.create_send(bus, bus)

    def test_create_send_duplicate_raises(self):
        """Cannot create duplicate send to same target."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        router.create_send(source, target)

        with pytest.raises(ValueError, match="already exists"):
            router.create_send(source, target)

    def test_create_send_max_sends_raises(self):
        """Cannot exceed max sends per source."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)

        for i in range(MAX_AUX_SENDS):
            target = MixBus(f"aux_{i}", BusType.AUX)
            router.create_send(source, target)

        with pytest.raises(ValueError, match="Maximum"):
            router.create_send(source, MixBus("overflow", BusType.AUX))

    def test_remove_send(self):
        """Remove aux send."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        send = router.create_send(source, target)
        result = router.remove_send(send)

        assert result is True
        assert len(router.get_sends(source)) == 0

    def test_remove_send_not_found(self):
        """Remove send returns False if not found."""
        router = BusRouter()
        send = AuxSend()

        result = router.remove_send(send)
        assert result is False

    def test_remove_send_no_source(self):
        """Remove send with no source returns False."""
        router = BusRouter()
        send = AuxSend(source_bus=None)

        result = router.remove_send(send)
        assert result is False

    def test_remove_all_sends(self):
        """Remove all sends from a source."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target1 = MixBus("reverb", BusType.AUX)
        target2 = MixBus("delay", BusType.AUX)

        router.create_send(source, target1)
        router.create_send(source, target2)

        count = router.remove_all_sends(source)

        assert count == 2
        assert len(router.get_sends(source)) == 0

    def test_get_sends(self):
        """Get sends from a source."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target1 = MixBus("reverb", BusType.AUX)
        target2 = MixBus("delay", BusType.AUX)

        router.create_send(source, target1, level_db=-6.0)
        router.create_send(source, target2, level_db=-12.0)

        sends = router.get_sends(source)

        assert len(sends) == 2
        assert all(isinstance(s, AuxSend) for s in sends)

    def test_get_sends_returns_copies(self):
        """get_sends returns copies."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        router.create_send(source, target, level_db=-6.0)

        sends = router.get_sends(source)
        sends[0].send_level_db = -12.0

        # Original unchanged
        original = router.get_sends(source)
        assert original[0].send_level_db == -6.0

    def test_get_sends_empty(self):
        """get_sends returns empty list if no sends."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)

        sends = router.get_sends(source)
        assert sends == []

    def test_get_send_by_id(self):
        """Get send by ID."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        send = router.create_send(source, target)

        found = router.get_send_by_id(send.id)

        assert found is not None
        assert found.id == send.id

    def test_get_send_by_id_not_found(self):
        """get_send_by_id returns None if not found."""
        router = BusRouter()

        result = router.get_send_by_id("nonexistent")
        assert result is None

    def test_set_send_level(self):
        """Set send level."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        send = router.create_send(source, target, level_db=-6.0)

        router.set_send_level(send, -12.0)

        sends = router.get_sends(source)
        assert sends[0].send_level_db == -12.0

    def test_set_send_level_no_source(self):
        """set_send_level with no source does nothing."""
        router = BusRouter()
        send = AuxSend(source_bus=None)

        # Should not raise
        router.set_send_level(send, -12.0)

    def test_enable_send(self):
        """Enable/disable send."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        send = router.create_send(source, target)

        router.enable_send(send, False)

        sends = router.get_sends(source)
        assert sends[0].enabled is False

        router.enable_send(send, True)

        sends = router.get_sends(source)
        assert sends[0].enabled is True


# =============================================================================
# BusRouter Direct Output Tests
# =============================================================================


class TestBusRouterDirectOutput:
    """Test BusRouter direct output management."""

    def test_set_direct_output(self):
        """Set direct output."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("submix", BusType.SUB)

        output = router.set_direct_output(source, target, level_db=-3.0)

        assert output is not None
        assert output.source_bus is source
        assert output.target_bus is target
        assert output.level_db == -3.0

    def test_set_direct_output_to_self_raises(self):
        """Cannot set direct output to self."""
        router = BusRouter()
        bus = MixBus("sfx", BusType.CATEGORY)

        with pytest.raises(ValueError, match="itself"):
            router.set_direct_output(bus, bus)

    def test_set_direct_output_replaces(self):
        """Setting direct output replaces existing."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target1 = MixBus("submix1", BusType.SUB)
        target2 = MixBus("submix2", BusType.SUB)

        router.set_direct_output(source, target1, level_db=-3.0)
        router.set_direct_output(source, target2, level_db=-6.0)

        output = router.get_direct_output(source)
        assert output.target_bus is target2
        assert output.level_db == -6.0

    def test_clear_direct_output(self):
        """Clear direct output."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("submix", BusType.SUB)

        router.set_direct_output(source, target)
        result = router.clear_direct_output(source)

        assert result is True
        assert router.get_direct_output(source) is None

    def test_clear_direct_output_not_found(self):
        """Clear direct output returns False if not found."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)

        result = router.clear_direct_output(source)
        assert result is False

    def test_get_direct_output(self):
        """Get direct output."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("submix", BusType.SUB)

        router.set_direct_output(source, target, level_db=-3.0)

        output = router.get_direct_output(source)

        assert output is not None
        assert output.target_bus is target
        assert output.level_db == -3.0

    def test_get_direct_output_not_found(self):
        """get_direct_output returns None if not found."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)

        result = router.get_direct_output(source)
        assert result is None

    def test_has_direct_output(self):
        """Check if bus has direct output."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("submix", BusType.SUB)

        assert router.has_direct_output(source) is False

        router.set_direct_output(source, target)

        assert router.has_direct_output(source) is True


# =============================================================================
# BusRouter Routing Queries Tests
# =============================================================================


class TestBusRouterRoutingQueries:
    """Test BusRouter routing query methods."""

    def test_get_effective_routing(self):
        """Get effective routing for a bus."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)
        submix = MixBus("submix", BusType.SUB)

        router.create_send(source, target, level_db=-6.0, mode=RoutingMode.POST_FADER)
        router.set_direct_output(source, submix)

        routing = router.get_effective_routing(source)

        assert routing["bus_name"] == "sfx"
        assert routing["has_direct_output"] is True
        assert routing["direct_target"] == "submix"
        assert len(routing["aux_sends"]) == 1
        assert routing["aux_sends"][0]["target"] == "reverb"
        assert routing["aux_sends"][0]["level_db"] == -6.0

    def test_get_all_sources_for_target(self):
        """Get all sources sending to a target."""
        router = BusRouter()
        source1 = MixBus("sfx", BusType.CATEGORY)
        source2 = MixBus("music", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        router.create_send(source1, target, level_db=-6.0)
        router.create_send(source2, target, level_db=-12.0)

        sources = router.get_all_sources_for_target(target)

        assert len(sources) == 2
        source_buses = [s[0] for s in sources]
        assert source1 in source_buses
        assert source2 in source_buses

    def test_get_all_sources_for_target_only_enabled(self):
        """get_all_sources_for_target only returns enabled sends."""
        router = BusRouter()
        source1 = MixBus("sfx", BusType.CATEGORY)
        source2 = MixBus("music", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        send1 = router.create_send(source1, target)
        router.create_send(source2, target)

        router.enable_send(send1, False)

        sources = router.get_all_sources_for_target(target)

        assert len(sources) == 1
        assert sources[0][0] is source2


# =============================================================================
# BusRouter State Management Tests
# =============================================================================


class TestBusRouterStateManagement:
    """Test BusRouter state management."""

    def test_get_routing_state(self):
        """Get complete routing state."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)
        submix = MixBus("submix", BusType.SUB)

        router.register_aux_bus(target)
        router.create_send(source, target, level_db=-6.0)
        router.set_direct_output(source, submix)

        state = router.get_routing_state()

        assert "aux_sends" in state
        assert "direct_outputs" in state
        assert "aux_bus_ids" in state

        assert len(state["aux_sends"]) > 0
        assert len(state["direct_outputs"]) > 0
        assert target.id in state["aux_bus_ids"]

    def test_clear(self):
        """Clear all routing."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)
        submix = MixBus("submix", BusType.SUB)

        router.register_aux_bus(target)
        router.create_send(source, target)
        router.set_direct_output(source, submix)

        router.clear()

        assert router.get_aux_buses() == []
        assert router.get_sends(source) == []
        assert router.get_direct_output(source) is None

    def test_repr(self):
        """repr shows useful info."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        router.register_aux_bus(target)
        router.create_send(source, target)

        repr_str = repr(router)

        assert "BusRouter" in repr_str
        assert "aux_sends=" in repr_str
        assert "aux_buses=" in repr_str


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestBusRouterThreadSafety:
    """Test BusRouter thread safety."""

    def test_concurrent_send_creation(self):
        """Concurrent send creation doesn't corrupt state."""
        router = BusRouter()
        sources = [MixBus(f"source_{i}", BusType.SUB) for i in range(10)]
        target = MixBus("reverb", BusType.AUX)

        def create_send(source):
            try:
                router.create_send(source, target)
            except ValueError:
                pass  # May fail if duplicate

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_send, s) for s in sources]
            for f in futures:
                f.result()

        # Should have some sends created
        total_sends = sum(len(router.get_sends(s)) for s in sources)
        assert total_sends == len(sources)

    def test_concurrent_level_changes(self):
        """Concurrent level changes don't corrupt state."""
        router = BusRouter()
        source = MixBus("sfx", BusType.CATEGORY)
        target = MixBus("reverb", BusType.AUX)

        send = router.create_send(source, target)

        def change_level(level):
            for _ in range(50):
                router.set_send_level(send, level)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(change_level, -6.0),
                executor.submit(change_level, -12.0),
                executor.submit(change_level, -18.0),
                executor.submit(change_level, 0.0),
            ]
            for f in futures:
                f.result()

        # Level should be one of the valid values
        sends = router.get_sends(source)
        assert sends[0].send_level_db in [-6.0, -12.0, -18.0, 0.0]
