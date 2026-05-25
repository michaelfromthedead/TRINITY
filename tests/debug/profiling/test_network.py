"""Tests for network profiler."""

import time

import pytest

from engine.debug.profiling.network import (
    ConnectionStats,
    NetworkProfiler,
    NetworkStats,
    PacketDirection,
    PacketRecord,
    PacketType,
    get_default_network_profiler,
    set_default_network_profiler,
)


class TestPacketType:
    """Tests for PacketType enum."""

    def test_all_packet_types_exist(self) -> None:
        """Test all expected packet types exist."""
        expected = ["RELIABLE", "UNRELIABLE", "ORDERED", "SEQUENCED"]
        for type_name in expected:
            assert hasattr(PacketType, type_name)


class TestPacketDirection:
    """Tests for PacketDirection enum."""

    def test_all_directions_exist(self) -> None:
        """Test all expected directions exist."""
        assert hasattr(PacketDirection, "SENT")
        assert hasattr(PacketDirection, "RECEIVED")


class TestNetworkStats:
    """Tests for NetworkStats dataclass."""

    def test_total_calculations(self) -> None:
        """Test total bytes and packets calculations."""
        stats = NetworkStats(
            bytes_sent=1000,
            bytes_received=2000,
            packets_sent=10,
            packets_received=20
        )

        assert stats.total_bytes == 3000
        assert stats.total_packets == 30

    def test_bandwidth_calculations(self) -> None:
        """Test bandwidth calculations in kbps."""
        stats = NetworkStats(
            bytes_sent=1000,  # 8000 bits
            bytes_received=2000  # 16000 bits
        )

        assert stats.bandwidth_sent_kbps == 8.0
        assert stats.bandwidth_received_kbps == 16.0


class TestPacketRecord:
    """Tests for PacketRecord dataclass."""

    def test_rtt_calculation(self) -> None:
        """Test RTT calculation."""
        record = PacketRecord(
            packet_id=1,
            size=100,
            direction=PacketDirection.SENT,
            packet_type=PacketType.RELIABLE,
            timestamp=1.0,
            acknowledged=True,
            ack_timestamp=1.05  # 50ms later
        )

        assert record.rtt_ms is not None
        assert abs(record.rtt_ms - 50.0) < 0.001

    def test_rtt_not_acknowledged(self) -> None:
        """Test RTT returns None when not acknowledged."""
        record = PacketRecord(
            packet_id=1,
            size=100,
            direction=PacketDirection.SENT,
            packet_type=PacketType.RELIABLE,
            timestamp=1.0
        )

        assert record.rtt_ms is None


class TestConnectionStats:
    """Tests for ConnectionStats dataclass."""

    def test_connection_duration(self) -> None:
        """Test connection duration calculation."""
        stats = ConnectionStats(
            connection_id="test",
            remote_address="192.168.1.1:1234",
            connected_since=time.time() - 5.0,
            last_activity=time.time()
        )

        assert stats.connection_duration >= 5.0

    def test_average_rtt(self) -> None:
        """Test average RTT calculation."""
        stats = ConnectionStats(
            connection_id="test",
            remote_address="192.168.1.1:1234",
            connected_since=time.time(),
            last_activity=time.time(),
            rtt_samples=[10.0, 20.0, 30.0]
        )

        assert stats.average_rtt_ms == 20.0

    def test_average_rtt_empty(self) -> None:
        """Test average RTT with no samples."""
        stats = ConnectionStats(
            connection_id="test",
            remote_address="192.168.1.1:1234",
            connected_since=time.time(),
            last_activity=time.time()
        )

        assert stats.average_rtt_ms == 0.0


class TestNetworkProfiler:
    """Tests for NetworkProfiler class."""

    def test_track_packet_sent(self) -> None:
        """Test tracking sent packets."""
        profiler = NetworkProfiler()

        packet_id = profiler.track_packet_sent(256, PacketType.RELIABLE)

        assert packet_id > 0

        stats = profiler.get_stats()
        assert stats.bytes_sent >= 256
        assert stats.packets_sent >= 1

    def test_track_packet_received(self) -> None:
        """Test tracking received packets."""
        profiler = NetworkProfiler()

        profiler.track_packet_received(512, PacketType.UNRELIABLE)

        stats = profiler.get_stats()
        assert stats.bytes_received >= 512
        assert stats.packets_received >= 1

    def test_track_packet_ack(self) -> None:
        """Test tracking packet acknowledgment."""
        profiler = NetworkProfiler()

        packet_id = profiler.track_packet_sent(256, PacketType.RELIABLE)
        time.sleep(0.01)  # 10ms delay

        rtt = profiler.track_packet_ack(packet_id)

        assert rtt is not None
        assert rtt >= 10.0  # At least 10ms

    def test_track_packet_ack_invalid(self) -> None:
        """Test ack for non-existent packet."""
        profiler = NetworkProfiler()

        rtt = profiler.track_packet_ack(999)

        assert rtt is None

    def test_track_packet_lost(self) -> None:
        """Test tracking lost packets."""
        profiler = NetworkProfiler()

        packet_id = profiler.track_packet_sent(256, PacketType.RELIABLE)
        profiler.track_packet_lost(packet_id)

        # Packet should be removed from pending
        rtt = profiler.track_packet_ack(packet_id)
        assert rtt is None

    def test_connection_registration(self) -> None:
        """Test connection registration."""
        profiler = NetworkProfiler()

        conn = profiler.register_connection("client_1", "192.168.1.1:1234")

        assert conn.connection_id == "client_1"
        assert conn.remote_address == "192.168.1.1:1234"

        retrieved = profiler.get_connection("client_1")
        assert retrieved is not None
        assert retrieved.connection_id == "client_1"

    def test_connection_unregistration(self) -> None:
        """Test connection unregistration."""
        profiler = NetworkProfiler()

        profiler.register_connection("client_1", "192.168.1.1:1234")
        removed = profiler.unregister_connection("client_1")

        assert removed is not None
        assert profiler.get_connection("client_1") is None

    def test_connection_stats_update(self) -> None:
        """Test connection stats are updated with packets."""
        profiler = NetworkProfiler()

        conn = profiler.register_connection("client_1", "192.168.1.1:1234")

        profiler.track_packet_sent(256, connection_id="client_1")
        profiler.track_packet_received(512, connection_id="client_1")

        assert conn.stats.bytes_sent == 256
        assert conn.stats.bytes_received == 512

    def test_get_total_stats(self) -> None:
        """Test getting accumulated statistics."""
        profiler = NetworkProfiler()

        profiler.track_packet_sent(100)
        profiler.track_packet_sent(200)
        profiler.track_packet_received(300)

        total = profiler.get_total_stats()

        assert total.bytes_sent == 300
        assert total.bytes_received == 300
        assert total.packets_sent == 2
        assert total.packets_received == 1

    def test_rtt_statistics(self) -> None:
        """Test RTT averaging."""
        profiler = NetworkProfiler()

        # Send packets and acknowledge them with varying delays
        for _ in range(3):
            packet_id = profiler.track_packet_sent(100)
            time.sleep(0.01)
            profiler.track_packet_ack(packet_id)

        stats = profiler.get_stats()
        assert stats.rtt_ms >= 10.0

    def test_jitter_calculation(self) -> None:
        """Test jitter calculation."""
        profiler = NetworkProfiler()

        # Manually add RTT samples with variation
        with profiler._lock:
            profiler._rtt_samples.extend([10.0, 15.0, 12.0, 18.0])

        stats = profiler.get_stats()
        # Jitter should be non-zero due to variation
        assert stats.jitter_ms > 0

    def test_disabled_profiler(self) -> None:
        """Test disabled profiler returns 0/None."""
        profiler = NetworkProfiler(enabled=False)

        packet_id = profiler.track_packet_sent(256)
        assert packet_id == 0

        profiler.track_packet_received(512)

        rtt = profiler.track_packet_ack(1)
        assert rtt is None

    def test_reset(self) -> None:
        """Test reset clears all data."""
        profiler = NetworkProfiler()

        profiler.track_packet_sent(256)
        profiler.register_connection("test", "127.0.0.1:1234")

        profiler.reset()

        total = profiler.get_total_stats()
        assert total.bytes_sent == 0
        assert profiler.get_connection("test") is None

    def test_get_bandwidth_history(self) -> None:
        """Test getting bandwidth history."""
        profiler = NetworkProfiler()

        profiler.track_packet_sent(100)
        profiler.track_packet_received(200)

        history = profiler.get_bandwidth_history(10.0)

        # Should have at least one bucket
        assert len(history) >= 1

    def test_format_stats(self) -> None:
        """Test formatting stats as string."""
        profiler = NetworkProfiler()

        profiler.track_packet_sent(1024)
        profiler.track_packet_received(2048)

        output = profiler.format_stats()

        assert "Network Statistics" in output
        assert "Bandwidth" in output


class TestDefaultNetworkProfiler:
    """Tests for default network profiler instance."""

    def test_get_default_profiler(self) -> None:
        """Test getting default profiler instance."""
        profiler = get_default_network_profiler()
        assert profiler is not None
        assert isinstance(profiler, NetworkProfiler)

    def test_set_default_profiler(self) -> None:
        """Test setting default profiler instance."""
        original = get_default_network_profiler()
        new_profiler = NetworkProfiler()

        set_default_network_profiler(new_profiler)
        assert get_default_network_profiler() is new_profiler

        # Restore original
        set_default_network_profiler(original)
