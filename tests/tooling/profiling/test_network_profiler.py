"""Tests for the network profiler module."""

from __future__ import annotations

import time

import pytest

from engine.tooling.profiling.network_profiler import (
    NetworkProfiler,
    NetworkProfilerState,
    PacketRecord,
    PacketDirection,
    PacketType,
    BandwidthSample,
    LatencyGraph,
    NetworkStats,
    ChannelStats,
    ActorNetworkStats,
)


class TestPacketRecord:
    """Tests for PacketRecord."""

    def test_creation(self):
        """Test basic creation."""
        record = PacketRecord(
            packet_id=1,
            timestamp=time.time(),
            direction=PacketDirection.SENT,
            packet_type=PacketType.RELIABLE,
            size_bytes=256,
        )
        assert record.packet_id == 1
        assert record.direction == PacketDirection.SENT
        assert record.size_bytes == 256

    def test_size_kb_conversion(self):
        """Test KB conversion."""
        record = PacketRecord(
            packet_id=1,
            timestamp=time.time(),
            direction=PacketDirection.SENT,
            packet_type=PacketType.RELIABLE,
            size_bytes=2048,
        )
        assert record.size_kb == pytest.approx(2.0, rel=1e-3)

    def test_with_channel(self):
        """Test packet with channel."""
        record = PacketRecord(
            packet_id=1,
            timestamp=time.time(),
            direction=PacketDirection.RECEIVED,
            packet_type=PacketType.UNRELIABLE,
            size_bytes=128,
            channel="voice",
        )
        assert record.channel == "voice"

    def test_with_actor(self):
        """Test packet with actor association."""
        record = PacketRecord(
            packet_id=1,
            timestamp=time.time(),
            direction=PacketDirection.SENT,
            packet_type=PacketType.RELIABLE,
            size_bytes=512,
            actor_id=123,
            property_name="position",
        )
        assert record.actor_id == 123
        assert record.property_name == "position"

    def test_to_dict(self):
        """Test dictionary conversion."""
        record = PacketRecord(
            packet_id=1,
            timestamp=time.time(),
            direction=PacketDirection.SENT,
            packet_type=PacketType.ORDERED,
            size_bytes=100,
            channel="game_state",
        )
        data = record.to_dict()

        assert data["packet_id"] == 1
        assert data["direction"] == "SENT"
        assert data["channel"] == "game_state"


class TestBandwidthSample:
    """Tests for BandwidthSample."""

    def test_creation(self):
        """Test basic creation."""
        sample = BandwidthSample(
            timestamp=time.time(),
            duration_seconds=1.0,
            bytes_sent=1024,
            bytes_received=2048,
        )
        assert sample.bytes_sent == 1024
        assert sample.bytes_received == 2048

    def test_kbps_calculations(self):
        """Test KB/s calculations."""
        sample = BandwidthSample(
            timestamp=time.time(),
            duration_seconds=1.0,
            bytes_sent=1024,
            bytes_received=2048,
        )
        assert sample.sent_kbps == pytest.approx(1.0, rel=1e-3)
        assert sample.received_kbps == pytest.approx(2.0, rel=1e-3)
        assert sample.total_kbps == pytest.approx(3.0, rel=1e-3)

    def test_zero_duration(self):
        """Test with zero duration."""
        sample = BandwidthSample(
            timestamp=time.time(),
            duration_seconds=0.0,
            bytes_sent=1024,
        )
        assert sample.sent_kbps == 0.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        sample = BandwidthSample(
            timestamp=time.time(),
            duration_seconds=1.0,
            bytes_sent=512,
            packets_sent=10,
        )
        data = sample.to_dict()

        assert "bytes_sent" in data
        assert "sent_kbps" in data
        assert data["packets_sent"] == 10


class TestLatencyGraph:
    """Tests for LatencyGraph."""

    def test_creation(self):
        """Test basic creation."""
        graph = LatencyGraph()
        assert len(graph.samples) == 0
        assert graph.min_rtt_ms == float("inf")

    def test_add_sample(self):
        """Test adding samples."""
        graph = LatencyGraph()

        graph.add_sample(time.time(), 10.0)
        graph.add_sample(time.time(), 20.0)
        graph.add_sample(time.time(), 15.0)

        assert len(graph.samples) == 3
        assert graph.min_rtt_ms == pytest.approx(10.0, rel=1e-3)
        assert graph.max_rtt_ms == pytest.approx(20.0, rel=1e-3)
        assert graph.avg_rtt_ms == pytest.approx(15.0, rel=1e-3)

    def test_jitter_calculation(self):
        """Test jitter calculation."""
        graph = LatencyGraph()

        # Add samples with varying latency
        graph.add_sample(time.time(), 10.0)
        graph.add_sample(time.time(), 20.0)
        graph.add_sample(time.time(), 10.0)
        graph.add_sample(time.time(), 20.0)

        assert graph.jitter_ms > 0

    def test_get_recent(self):
        """Test getting recent samples."""
        graph = LatencyGraph()

        old_time = time.time() - 20.0
        recent_time = time.time() - 2.0

        graph.samples.append((old_time, 10.0))
        graph.samples.append((recent_time, 15.0))

        recent = graph.get_recent(seconds=5.0)
        assert len(recent) == 1
        assert recent[0][1] == 15.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        graph = LatencyGraph()
        graph.add_sample(time.time(), 10.0)

        data = graph.to_dict()

        assert data["sample_count"] == 1
        assert "avg_rtt_ms" in data


class TestNetworkStats:
    """Tests for NetworkStats."""

    def test_creation(self):
        """Test basic creation."""
        stats = NetworkStats()
        assert stats.total_bytes_sent == 0
        assert stats.packet_loss_percentage == 0.0

    def test_packet_loss_calculation(self):
        """Test packet loss percentage calculation."""
        stats = NetworkStats(
            total_packets_sent=100,
            total_packets_received=100,
            packets_dropped=10,
        )
        assert stats.packet_loss_percentage == pytest.approx(5.0, rel=1e-3)

    def test_total_bytes(self):
        """Test total bytes calculation."""
        stats = NetworkStats(
            total_bytes_sent=1000,
            total_bytes_received=2000,
        )
        assert stats.total_bytes == 3000
        assert stats.total_kb == pytest.approx(3000 / 1024, rel=1e-3)

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = NetworkStats(
            total_bytes_sent=1024,
            current_rtt_ms=50.0,
        )
        data = stats.to_dict()

        assert data["total_bytes_sent"] == 1024
        assert data["current_rtt_ms"] == 50.0


class TestChannelStats:
    """Tests for ChannelStats."""

    def test_creation(self):
        """Test basic creation."""
        stats = ChannelStats(channel_name="game_state")
        assert stats.channel_name == "game_state"
        assert stats.bytes_sent == 0

    def test_total_bytes(self):
        """Test total bytes calculation."""
        stats = ChannelStats(
            channel_name="test",
            bytes_sent=500,
            bytes_received=300,
        )
        assert stats.total_bytes == 800

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = ChannelStats(
            channel_name="voice",
            packets_sent=100,
        )
        data = stats.to_dict()

        assert data["channel_name"] == "voice"
        assert data["packets_sent"] == 100


class TestActorNetworkStats:
    """Tests for ActorNetworkStats."""

    def test_creation(self):
        """Test basic creation."""
        stats = ActorNetworkStats(actor_id=123)
        assert stats.actor_id == 123
        assert stats.bytes_sent == 0

    def test_add_property_bytes(self):
        """Test adding property bytes."""
        stats = ActorNetworkStats(actor_id=1)

        stats.add_property_bytes("position", 12)
        stats.add_property_bytes("rotation", 16)
        stats.add_property_bytes("position", 12)

        assert stats.properties["position"] == 24
        assert stats.properties["rotation"] == 16

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = ActorNetworkStats(
            actor_id=42,
            bytes_sent=256,
            updates_sent=10,
        )
        stats.add_property_bytes("health", 4)

        data = stats.to_dict()

        assert data["actor_id"] == 42
        assert data["updates_sent"] == 10
        assert "health" in data["properties"]


class TestNetworkProfiler:
    """Tests for NetworkProfiler."""

    @pytest.fixture
    def profiler(self):
        """Create a fresh profiler instance."""
        return NetworkProfiler(sample_window_seconds=0.1)

    def test_initial_state(self, profiler):
        """Test initial profiler state."""
        assert profiler.state == NetworkProfilerState.DISABLED
        assert not profiler.is_enabled

    def test_enable_disable(self, profiler):
        """Test enable/disable operations."""
        profiler.enable()
        assert profiler.is_enabled

        profiler.disable()
        assert not profiler.is_enabled

    def test_pause_resume(self, profiler):
        """Test pause/resume operations."""
        profiler.enable()
        profiler.pause()
        assert profiler.state == NetworkProfilerState.PAUSED

        profiler.resume()
        assert profiler.state == NetworkProfilerState.ENABLED

    def test_record_packet_sent(self, profiler):
        """Test recording sent packet."""
        profiler.enable()

        packet_id = profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=256,
            packet_type=PacketType.RELIABLE,
            channel="game_state",
        )

        assert packet_id == 1
        stats = profiler.get_stats()
        assert stats.total_bytes_sent == 256
        assert stats.total_packets_sent == 1

    def test_record_packet_received(self, profiler):
        """Test recording received packet."""
        profiler.enable()

        profiler.record_packet(
            direction=PacketDirection.RECEIVED,
            size_bytes=512,
        )

        stats = profiler.get_stats()
        assert stats.total_bytes_received == 512
        assert stats.total_packets_received == 1

    def test_record_packet_disabled(self, profiler):
        """Test packet not recorded when disabled."""
        packet_id = profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=256,
        )

        assert packet_id == 0

    def test_rtt_tracking(self, profiler):
        """Test RTT tracking."""
        profiler.enable()

        # Send packet
        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=64,
            sequence_number=1,
        )

        # Simulate small delay
        time.sleep(0.01)

        # Receive ACK
        profiler.record_packet(
            direction=PacketDirection.RECEIVED,
            size_bytes=32,
            ack_number=1,
        )

        stats = profiler.get_stats()
        assert stats.current_rtt_ms > 0

    def test_record_rtt_direct(self, profiler):
        """Test direct RTT recording."""
        profiler.enable()

        profiler.record_rtt(50.0)
        profiler.record_rtt(60.0)

        stats = profiler.get_stats()
        assert stats.current_rtt_ms == 60.0
        assert stats.avg_rtt_ms == pytest.approx(55.0, rel=1e-3)

    def test_channel_stats(self, profiler):
        """Test per-channel statistics."""
        profiler.enable()

        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=100,
            channel="voice",
        )
        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=200,
            channel="game_state",
        )
        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=50,
            channel="voice",
        )

        channel_stats = profiler.get_channel_stats()

        assert "voice" in channel_stats
        assert channel_stats["voice"].bytes_sent == 150
        assert channel_stats["game_state"].bytes_sent == 200

    def test_actor_stats(self, profiler):
        """Test per-actor statistics."""
        profiler.enable()

        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=100,
            actor_id=1,
            property_name="position",
        )
        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=50,
            actor_id=1,
            property_name="rotation",
        )
        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=200,
            actor_id=2,
        )

        actor_stats = profiler.get_actor_stats()

        assert 1 in actor_stats
        assert actor_stats[1].bytes_sent == 150
        assert actor_stats[1].properties["position"] == 100

    def test_packet_drop_tracking(self, profiler):
        """Test packet drop tracking."""
        profiler.enable()

        packet_id = profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=64,
        )

        profiler.record_packet_drop(packet_id)

        stats = profiler.get_stats()
        assert stats.packets_dropped == 1

    def test_retransmission_tracking(self, profiler):
        """Test retransmission tracking."""
        profiler.enable()

        packet_id = profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=64,
        )

        profiler.record_retransmission(packet_id)

        stats = profiler.get_stats()
        assert stats.packets_retransmitted == 1

    def test_bandwidth_samples(self, profiler):
        """Test bandwidth sample collection."""
        profiler.enable()

        for _ in range(5):
            profiler.record_packet(
                direction=PacketDirection.SENT,
                size_bytes=1024,
            )

        # Wait for window to complete
        time.sleep(0.15)

        # Trigger window rotation
        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=512,
        )

        samples = profiler.get_bandwidth_samples()
        assert len(samples) >= 1

    def test_get_current_bandwidth(self, profiler):
        """Test getting current bandwidth."""
        profiler.enable()

        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=1024,
        )

        current = profiler.get_current_bandwidth()
        assert current.bytes_sent == 1024

    def test_get_latency_graph(self, profiler):
        """Test getting latency graph."""
        profiler.enable()

        profiler.record_rtt(10.0)
        profiler.record_rtt(15.0)

        graph = profiler.get_latency_graph()
        assert len(graph.samples) == 2

    def test_get_packets_filtered(self, profiler):
        """Test filtered packet retrieval."""
        profiler.enable()

        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=100,
            channel="voice",
        )
        profiler.record_packet(
            direction=PacketDirection.RECEIVED,
            size_bytes=200,
            channel="game_state",
        )

        sent = profiler.get_packets(direction=PacketDirection.SENT)
        assert len(sent) == 1

        voice = profiler.get_packets(channel="voice")
        assert len(voice) == 1

    def test_get_top_bandwidth_actors(self, profiler):
        """Test getting top bandwidth actors."""
        profiler.enable()

        for i in range(3):
            profiler.record_packet(
                direction=PacketDirection.SENT,
                size_bytes=(i + 1) * 100,
                actor_id=i,
            )

        top = profiler.get_top_bandwidth_actors(top_n=2)
        assert len(top) == 2
        assert top[0][0] == 2  # Highest bandwidth actor

    def test_get_top_bandwidth_channels(self, profiler):
        """Test getting top bandwidth channels."""
        profiler.enable()

        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=1000,
            channel="voice",
        )
        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=500,
            channel="game_state",
        )

        top = profiler.get_top_bandwidth_channels(top_n=2)
        assert len(top) == 2
        assert top[0][0] == "voice"

    def test_clear(self, profiler):
        """Test clearing profiler data."""
        profiler.enable()

        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=256,
        )

        profiler.clear()

        stats = profiler.get_stats()
        assert stats.total_bytes_sent == 0
        assert len(profiler.get_packets()) == 0

    def test_listener_callback(self, profiler):
        """Test packet listener callbacks."""
        profiler.enable()
        packets_received = []

        def on_packet(packet):
            packets_received.append(packet)

        profiler.add_listener(on_packet)

        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=64,
        )

        assert len(packets_received) == 1

        profiler.remove_listener(on_packet)

    def test_to_dict(self, profiler):
        """Test dictionary export."""
        profiler.enable()

        profiler.record_packet(
            direction=PacketDirection.SENT,
            size_bytes=128,
            channel="test",
        )

        data = profiler.to_dict()

        assert "state" in data
        assert "packet_count" in data
        assert data["packet_count"] == 1
