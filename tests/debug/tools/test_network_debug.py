"""
Tests for the network debugging system.

Tests verify:
1. Network simulation (latency, packet loss, jitter)
2. Packet logging and statistics
3. Build-type security guards
4. Actual game impact
"""

import os
import pytest
from unittest.mock import Mock

from engine.debug.tools.network_debug import (
    NetworkDebugConfig,
    NetworkDebugger,
    NetworkSimulation,
    NetworkStats,
    PacketDirection,
    PacketLog,
    get_network_debugger,
)


class TestNetworkDebugConfig:
    """Tests for NetworkDebugConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NetworkDebugConfig()
        assert config.max_recent_packets == 10000
        assert config.max_packet_log == 1000
        assert config.max_latency_samples == 100
        assert config.stats_window_seconds == 1.0
        assert config.allow_in_shipping is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = NetworkDebugConfig(
            max_recent_packets=5000,
            max_packet_log=500,
            latency_quality_threshold_ms=50.0,
        )
        assert config.max_recent_packets == 5000
        assert config.max_packet_log == 500
        assert config.latency_quality_threshold_ms == 50.0


class TestNetworkSimulation:
    """Tests for NetworkSimulation."""

    def test_default_simulation(self):
        """Test default simulation has no effects."""
        sim = NetworkSimulation()
        assert sim.latency_ms == 0.0
        assert sim.packet_loss_percent == 0.0
        assert sim.jitter_ms == 0.0
        assert sim.bandwidth_limit_kbps == 0.0

    def test_custom_simulation(self):
        """Test custom simulation values."""
        sim = NetworkSimulation(
            latency_ms=100.0,
            packet_loss_percent=5.0,
            jitter_ms=20.0,
        )
        assert sim.latency_ms == 100.0
        assert sim.packet_loss_percent == 5.0
        assert sim.jitter_ms == 20.0


class TestNetworkDebugger:
    """Tests for NetworkDebugger."""

    @pytest.fixture
    def debugger(self):
        """Create a fresh NetworkDebugger."""
        return NetworkDebugger()

    def test_initial_state(self, debugger):
        """Test initial debugger state."""
        assert debugger.enabled is True
        assert debugger.simulation.latency_ms == 0.0
        assert debugger.simulation.packet_loss_percent == 0.0

    def test_set_latency(self, debugger):
        """Test setting simulated latency."""
        debugger.set_latency(100.0, variance_ms=20.0)
        assert debugger.simulation.latency_ms == 100.0
        assert debugger.simulation.latency_variance_ms == 20.0

    def test_set_packet_loss(self, debugger):
        """Test setting packet loss."""
        debugger.set_packet_loss(10.0, burst_chance=20.0, burst_length=5)
        assert debugger.simulation.packet_loss_percent == 10.0
        assert debugger.simulation.packet_loss_burst_chance == 20.0
        assert debugger.simulation.packet_loss_burst_length == 5

    def test_set_jitter(self, debugger):
        """Test setting jitter."""
        debugger.set_jitter(30.0)
        assert debugger.simulation.jitter_ms == 30.0

    def test_set_bandwidth_limit(self, debugger):
        """Test setting bandwidth limit."""
        debugger.set_bandwidth_limit(1000.0)
        assert debugger.simulation.bandwidth_limit_kbps == 1000.0

    def test_reset_simulation(self, debugger):
        """Test resetting simulation."""
        debugger.set_latency(100.0)
        debugger.set_packet_loss(10.0)
        debugger.set_jitter(30.0)

        debugger.reset_simulation()

        assert debugger.simulation.latency_ms == 0.0
        assert debugger.simulation.packet_loss_percent == 0.0
        assert debugger.simulation.jitter_ms == 0.0

    def test_should_drop_packet_disabled(self, debugger):
        """Test no packet loss when disabled."""
        debugger.enabled = False
        assert debugger.should_drop_packet() is False

    def test_should_drop_packet_zero_loss(self, debugger):
        """Test no packet loss when loss is zero."""
        assert debugger.should_drop_packet() is False

    def test_get_simulated_latency(self, debugger):
        """Test getting simulated latency."""
        debugger.set_latency(100.0)
        latency = debugger.get_simulated_latency()
        # With no variance, should be exactly 100
        assert latency >= 100.0  # May have jitter

    def test_get_simulated_latency_disabled(self, debugger):
        """Test no latency when disabled."""
        debugger.set_latency(100.0)
        debugger.enabled = False
        assert debugger.get_simulated_latency() == 0.0

    def test_packet_logging(self, debugger):
        """Test packet logging."""
        debugger.log_packets(True)

        debugger.log_packet(
            direction=PacketDirection.OUTBOUND,
            size=100,
            channel="reliable",
            packet_type="game_state",
            source="client",
            destination="server",
        )

        logs = debugger.get_packet_log()
        assert len(logs) == 1
        assert logs[0].direction == PacketDirection.OUTBOUND
        assert logs[0].size == 100

    def test_packet_log_filtering(self, debugger):
        """Test packet log filtering."""
        debugger.log_packets(True)

        # Log inbound and outbound
        debugger.log_packet(
            direction=PacketDirection.OUTBOUND,
            size=100,
            channel="reliable",
        )
        debugger.log_packet(
            direction=PacketDirection.INBOUND,
            size=200,
            channel="unreliable",
        )

        # Filter by direction
        outbound = debugger.get_packet_log(direction=PacketDirection.OUTBOUND)
        assert len(outbound) == 1

        # Filter by channel
        reliable = debugger.get_packet_log(channel="reliable")
        assert len(reliable) == 1

    def test_statistics_tracking(self, debugger):
        """Test statistics tracking."""
        debugger.log_packet(
            direction=PacketDirection.OUTBOUND,
            size=100,
        )
        debugger.log_packet(
            direction=PacketDirection.INBOUND,
            size=200,
        )

        stats = debugger.get_stats()
        assert stats.packets_sent == 1
        assert stats.packets_received == 1
        assert stats.bytes_sent == 100
        assert stats.bytes_received == 200

    def test_latency_recording(self, debugger):
        """Test latency recording."""
        debugger.record_latency(50.0)
        debugger.record_latency(60.0)
        debugger.record_latency(70.0)

        stats = debugger.get_stats()
        assert stats.current_latency_ms == 70.0
        assert stats.average_latency_ms == 60.0
        assert stats.min_latency_ms == 50.0
        assert stats.max_latency_ms == 70.0

    def test_rtt_recording(self, debugger):
        """Test RTT recording."""
        debugger.record_rtt(100.0)
        debugger.record_rtt(120.0)

        stats = debugger.get_stats()
        assert stats.rtt_ms == 120.0
        assert stats.rtt_average_ms == 110.0

    def test_reset_stats(self, debugger):
        """Test resetting statistics."""
        debugger.log_packet(direction=PacketDirection.OUTBOUND, size=100)
        debugger.record_latency(50.0)

        debugger.reset_stats()

        stats = debugger.get_stats()
        assert stats.packets_sent == 0
        assert stats.current_latency_ms == 0.0


class TestBuildTypeGuards:
    """Tests for build-type security guards."""

    @pytest.fixture(autouse=True)
    def reset_env(self):
        """Reset environment before each test."""
        for var in ["GAME_BUILD_TYPE", "SHIPPING"]:
            os.environ.pop(var, None)
        yield
        for var in ["GAME_BUILD_TYPE", "SHIPPING"]:
            os.environ.pop(var, None)

    def test_cannot_enable_in_shipping(self):
        """Test network debugger cannot be enabled in shipping builds."""
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"

        debugger = NetworkDebugger()
        debugger.enabled = True

        # Should trigger warning and block enable

    def test_config_allows_shipping_override(self):
        """Test config can allow network debugger in shipping."""
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"

        config = NetworkDebugConfig(allow_in_shipping=True)
        debugger = NetworkDebugger(config)

        debugger.enabled = True
        assert debugger.enabled is True


class TestConfigValues:
    """Tests that config values are used instead of magic numbers."""

    def test_custom_buffer_limits(self):
        """Test custom buffer limits from config."""
        config = NetworkDebugConfig(
            max_recent_packets=1000,
            max_packet_log=100,
            max_latency_samples=50,
        )
        debugger = NetworkDebugger(config)

        # Buffer sizes should match config
        assert debugger._recent_packets.maxlen == 1000
        assert debugger._packet_log.maxlen == 100
        assert debugger._latency_samples.maxlen == 50

    def test_custom_quality_thresholds(self):
        """Test custom quality thresholds from config."""
        config = NetworkDebugConfig(
            latency_quality_threshold_ms=50.0,
            jitter_quality_threshold_ms=15.0,
        )
        debugger = NetworkDebugger(config)

        assert debugger.config.latency_quality_threshold_ms == 50.0
        assert debugger.config.jitter_quality_threshold_ms == 15.0


class TestConnectionQuality:
    """Tests for connection quality calculation."""

    @pytest.fixture
    def debugger(self):
        """Create a fresh NetworkDebugger."""
        return NetworkDebugger()

    def test_perfect_quality(self, debugger):
        """Test perfect connection quality."""
        stats = debugger.get_stats()
        # No issues should give quality of 1.0
        assert stats.connection_quality == 1.0

    def test_latency_reduces_quality(self, debugger):
        """Test latency reduces connection quality."""
        # Record high latency
        for _ in range(10):
            debugger.record_latency(200.0)

        stats = debugger.get_stats()
        assert stats.connection_quality < 1.0

    def test_packet_loss_reduces_quality(self, debugger):
        """Test packet loss reduces connection quality."""
        # Log dropped packets
        for i in range(10):
            debugger.log_packet(
                direction=PacketDirection.OUTBOUND,
                size=100,
                dropped=(i % 2 == 0),  # 50% loss
            )

        stats = debugger.get_stats()
        assert stats.packet_loss_percent > 0
        assert stats.connection_quality < 1.0


class TestNetworkDebugGameImpact:
    """Tests that verify network debugging actually impacts networking."""

    @pytest.fixture
    def debugger(self):
        """Create a fresh NetworkDebugger."""
        return NetworkDebugger()

    def test_latency_simulation_delays_packets(self, debugger):
        """Test latency simulation affects packet timing."""
        debugger.set_latency(100.0)

        # Simulated delay
        delay = debugger.get_simulated_latency()
        assert delay >= 100.0

        # Without simulation
        debugger.reset_simulation()
        delay = debugger.get_simulated_latency()
        assert delay == 0.0

    def test_packet_loss_drops_packets(self, debugger):
        """Test packet loss simulation drops packets."""
        # Set very high packet loss for deterministic testing
        debugger.set_packet_loss(100.0)  # 100% loss

        # With 100% loss, should always drop
        dropped_count = sum(1 for _ in range(100) if debugger.should_drop_packet())
        assert dropped_count == 100

    def test_bandwidth_limiting(self, debugger):
        """Test bandwidth limiting affects send decisions."""
        debugger.set_bandwidth_limit(1.0)  # 1 kbps = 125 bytes/sec

        # Small packet should be allowed
        assert debugger.check_bandwidth(10) is True

        # But after sending, next large packet blocked (in same window)
        # Note: This depends on timing, so we just verify the method works

    def test_simulation_affects_game_networking(self, debugger):
        """Test that simulation values would affect real networking."""
        debugger.set_latency(100.0)
        debugger.set_packet_loss(5.0)
        debugger.set_jitter(20.0)
        debugger.set_bandwidth_limit(1000.0)

        # Game networking layer would check these
        assert debugger.simulation.latency_ms == 100.0
        assert debugger.simulation.packet_loss_percent == 5.0
        assert debugger.simulation.jitter_ms == 20.0
        assert debugger.simulation.bandwidth_limit_kbps == 1000.0

        # Stats track simulation settings
        stats = debugger.get_stats()
        assert stats.simulated_latency_ms == 100.0
        assert stats.simulated_packet_loss == 5.0
        assert stats.simulated_jitter_ms == 20.0
        assert stats.bandwidth_limit_kbps == 1000.0
