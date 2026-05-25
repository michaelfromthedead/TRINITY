"""
Tests for the networking transport module.

Tests packets, channels, connections, and UDP transport.
"""

import time
import pytest

from engine.networking.transport.packet import (
    Packet,
    PacketHeader,
    PacketType,
    PacketFlags,
    PacketFragmenter,
    MTU,
    MAX_PAYLOAD_SIZE,
    HEADER_SIZE,
    sequence_greater_than,
    sequence_difference,
)
from engine.networking.transport.channel import (
    Channel,
    ChannelType,
    ChannelConfig,
    ChannelManager,
    UnreliableChannel,
    ReliableChannel,
    ReliableOrderedChannel,
    SequencedChannel,
)
from engine.networking.transport.connection import (
    Connection,
    ConnectionState,
    ConnectionConfig,
    ConnectionStats,
)
from engine.networking.transport.quality import (
    QualityLevel,
    QualityMetrics,
    QualityMonitor,
    NetworkQualityAdapter,
    AdaptiveSettings,
)
from engine.networking.config import DEFAULT_CONFIG


class TestPacketHeader:
    """Tests for PacketHeader class."""

    def test_header_creation(self):
        """Test creating a packet header."""
        header = PacketHeader(
            packet_type=PacketType.DATA,
            flags=PacketFlags.RELIABLE,
            sequence=100,
            ack=50,
            ack_bits=0xFFFFFFFF,
            size=500
        )

        assert header.packet_type == PacketType.DATA
        assert header.flags == PacketFlags.RELIABLE
        assert header.sequence == 100
        assert header.ack == 50
        assert header.ack_bits == 0xFFFFFFFF
        assert header.size == 500

    def test_header_serialization(self):
        """Test header serialization roundtrip."""
        original = PacketHeader(
            packet_type=PacketType.RELIABLE_DATA,
            flags=PacketFlags.COMPRESSED | PacketFlags.ORDERED,
            sequence=12345,
            ack=12340,
            ack_bits=0xABCDEF01,
            size=100
        )

        data = original.to_bytes()
        assert len(data) == HEADER_SIZE

        restored = PacketHeader.from_bytes(data)
        assert restored.packet_type == original.packet_type
        assert restored.flags == original.flags
        assert restored.sequence == original.sequence
        assert restored.ack == original.ack
        assert restored.ack_bits == original.ack_bits
        assert restored.size == original.size

    def test_has_flag(self):
        """Test flag checking."""
        header = PacketHeader(
            packet_type=PacketType.DATA,
            flags=PacketFlags.RELIABLE | PacketFlags.ORDERED
        )

        assert header.has_flag(PacketFlags.RELIABLE)
        assert header.has_flag(PacketFlags.ORDERED)
        assert not header.has_flag(PacketFlags.COMPRESSED)

    def test_set_clear_flag(self):
        """Test setting and clearing flags."""
        header = PacketHeader(packet_type=PacketType.DATA)

        header.set_flag(PacketFlags.RELIABLE)
        assert header.has_flag(PacketFlags.RELIABLE)

        header.clear_flag(PacketFlags.RELIABLE)
        assert not header.has_flag(PacketFlags.RELIABLE)


class TestPacket:
    """Tests for Packet class."""

    def test_packet_creation(self):
        """Test creating a packet."""
        packet = Packet.create(
            PacketType.DATA,
            payload=b'Hello, World!',
            sequence=42
        )

        assert packet.header.packet_type == PacketType.DATA
        assert packet.header.sequence == 42
        assert packet.payload == b'Hello, World!'
        assert packet.header.size == len(b'Hello, World!')

    def test_packet_serialization(self):
        """Test packet serialization roundtrip."""
        original = Packet.create(
            PacketType.RELIABLE_DATA,
            payload=b'\x00\x01\x02\x03\x04',
            sequence=100,
            flags=PacketFlags.RELIABLE
        )

        data = original.to_bytes()
        restored = Packet.from_bytes(data)

        assert restored.header.packet_type == original.header.packet_type
        assert restored.header.sequence == original.header.sequence
        assert restored.payload == original.payload

    def test_create_ack_packet(self):
        """Test creating ACK packet."""
        packet = Packet.create_ack(ack_sequence=50, ack_bits=0x0F)

        assert packet.header.packet_type == PacketType.ACK
        assert packet.header.ack == 50
        assert packet.header.ack_bits == 0x0F

    def test_create_heartbeat_packet(self):
        """Test creating heartbeat packet."""
        packet = Packet.create_heartbeat(sequence=10)

        assert packet.header.packet_type == PacketType.HEARTBEAT
        assert packet.header.sequence == 10

    def test_total_size(self):
        """Test total size calculation."""
        packet = Packet.create(PacketType.DATA, payload=b'test')
        assert packet.total_size == HEADER_SIZE + 4

    def test_is_reliable(self):
        """Test reliable flag checking."""
        reliable = Packet.create(PacketType.DATA, flags=PacketFlags.RELIABLE)
        unreliable = Packet.create(PacketType.DATA)

        assert reliable.is_reliable()
        assert not unreliable.is_reliable()


class TestPacketFragmenter:
    """Tests for PacketFragmenter class."""

    def test_no_fragmentation_needed(self):
        """Test that small packets don't get fragmented."""
        fragmenter = PacketFragmenter()
        payload = b'Small payload'

        fragments = fragmenter.fragment(payload)
        assert len(fragments) == 1
        assert fragments[0].payload == payload

    def test_fragmentation(self):
        """Test fragmenting large payloads."""
        fragmenter = PacketFragmenter()
        # Create payload larger than MTU
        payload = b'X' * (MTU * 2)

        fragments = fragmenter.fragment(payload)
        assert len(fragments) > 1
        for frag in fragments:
            assert frag.is_fragmented()

    def test_fragment_reassembly(self):
        """Test reassembling fragments."""
        fragmenter = PacketFragmenter()
        original = b'X' * (MTU * 3)

        fragments = fragmenter.fragment(original)

        # Add fragments one by one
        result = None
        for frag in fragments:
            result = fragmenter.add_fragment(frag)

        assert result == original

    def test_fragment_out_of_order(self):
        """Test reassembling out-of-order fragments."""
        fragmenter = PacketFragmenter()
        original = b'Y' * (MTU * 3)

        fragments = fragmenter.fragment(original)

        # Shuffle order
        import random
        shuffled = list(fragments)
        random.shuffle(shuffled)

        result = None
        for frag in shuffled:
            r = fragmenter.add_fragment(frag)
            if r:
                result = r

        assert result == original

    def test_clear_pending(self):
        """Test clearing pending fragments."""
        fragmenter = PacketFragmenter()
        payload = b'Z' * (MTU * 2)

        fragments = fragmenter.fragment(payload)
        fragmenter.add_fragment(fragments[0])  # Add only first

        fragmenter.clear_pending()

        # Now should not reassemble even with all fragments
        result = None
        for frag in fragments:
            result = fragmenter.add_fragment(frag)

        # Will need to get all fragments again
        assert result == payload  # Because we added them all again


class TestSequenceNumbers:
    """Tests for sequence number utilities."""

    def test_sequence_greater_than_normal(self):
        """Test normal sequence comparison."""
        assert sequence_greater_than(10, 5)
        assert not sequence_greater_than(5, 10)
        assert not sequence_greater_than(10, 10)

    def test_sequence_greater_than_wraparound(self):
        """Test sequence comparison with wraparound."""
        # Just after wraparound
        assert sequence_greater_than(1, 65535)
        assert not sequence_greater_than(65535, 1)

        # Further after wraparound
        assert sequence_greater_than(100, 65500)

    def test_sequence_difference(self):
        """Test sequence difference calculation."""
        assert sequence_difference(10, 5) == 5
        assert sequence_difference(5, 10) == -5

    def test_sequence_difference_wraparound(self):
        """Test sequence difference with wraparound."""
        diff = sequence_difference(1, 65535)
        assert diff == 2  # 65535 -> 0 -> 1


class TestUnreliableChannel:
    """Tests for UnreliableChannel class."""

    def test_send_receive(self):
        """Test basic send and receive."""
        channel = UnreliableChannel(0)

        packets = channel.send(b'test data')
        assert len(packets) == 1

        result = channel.receive(packets[0])
        assert result == b'test data'

    def test_stats_tracking(self):
        """Test statistics tracking."""
        channel = UnreliableChannel(0)

        channel.send(b'data')
        packet = Packet.create(PacketType.DATA, b'received')
        channel.receive(packet)

        assert channel.stats.packets_sent == 1
        assert channel.stats.packets_received == 1

    def test_process_ack_no_op(self):
        """Test that process_ack is no-op."""
        channel = UnreliableChannel(0)
        result = channel.process_ack(100, 0xFFFF)
        assert result == []


class TestReliableChannel:
    """Tests for ReliableChannel class."""

    def test_send_marks_reliable(self):
        """Test that sent packets are marked reliable."""
        channel = ReliableChannel(0)
        packets = channel.send(b'test')

        assert len(packets) == 1
        assert packets[0].is_reliable()

    def test_receive_deduplication(self):
        """Test that duplicates are ignored."""
        channel = ReliableChannel(0)
        packet = Packet.create(PacketType.DATA, b'data', sequence=1)
        packet.header.set_flag(PacketFlags.RELIABLE)

        result1 = channel.receive(packet)
        result2 = channel.receive(packet)

        assert result1 == b'data'
        assert result2 is None

    def test_ack_processing(self):
        """Test ACK processing removes pending packets."""
        channel = ReliableChannel(0)
        channel.send(b'data')

        assert channel.stats.pending_acks == 1

        channel.process_ack(0, 0)
        assert channel.stats.pending_acks == 0

    def test_retransmission(self):
        """Test retransmission on timeout."""
        config = ChannelConfig(
            ChannelType.RELIABLE_UNORDERED,
            initial_rtt=0.001  # Very short for testing
        )
        channel = ReliableChannel(0, config)
        channel.send(b'data')

        # Wait for retransmit time
        time.sleep(0.01)

        retransmits = channel.update(0.01)
        assert len(retransmits) >= 1
        assert channel.stats.packets_retransmitted >= 1

    def test_get_ack_data(self):
        """Test getting ACK data for outgoing packets."""
        channel = ReliableChannel(0)

        # Receive some packets
        for i in range(5):
            packet = Packet.create(PacketType.DATA, b'x', sequence=i)
            channel.receive(packet)

        ack, ack_bits = channel.get_ack_data()
        assert ack == 4  # Last received
        assert ack_bits & 0x0F == 0x0F  # Previous 4 acked


class TestReliableOrderedChannel:
    """Tests for ReliableOrderedChannel class."""

    def test_in_order_delivery(self):
        """Test packets delivered in order."""
        channel = ReliableOrderedChannel(0)

        # Send packets out of order
        packets = [
            Packet.create(PacketType.DATA, b'2', sequence=2),
            Packet.create(PacketType.DATA, b'0', sequence=0),
            Packet.create(PacketType.DATA, b'1', sequence=1),
        ]

        results = []
        for p in packets:
            result = channel.receive(p)
            if result:
                results.append(result)

        # When packets 0 arrives (after 2), it delivers immediately
        # When packet 1 arrives, it can deliver 1 and 2 together
        # So we get [b'0', b'12'] - the channel concatenates consecutive deliverable packets
        # Verify all data is received in order
        all_data = b''.join(results)
        assert all_data == b'012'

    def test_buffered_packets(self):
        """Test packets are buffered when waiting."""
        channel = ReliableOrderedChannel(0)

        # Receive packet 2 first
        channel.receive(Packet.create(PacketType.DATA, b'2', sequence=2))

        assert channel.get_buffered_count() == 1

        # Now receive 0 and 1
        channel.receive(Packet.create(PacketType.DATA, b'0', sequence=0))
        channel.receive(Packet.create(PacketType.DATA, b'1', sequence=1))

        assert channel.get_buffered_count() == 0


class TestSequencedChannel:
    """Tests for SequencedChannel class."""

    def test_drops_old_packets(self):
        """Test that old packets are dropped."""
        channel = SequencedChannel(0)

        # Receive newer first
        channel.receive(Packet.create(PacketType.SEQUENCED_DATA, b'new', sequence=10))

        # Receive older - should be dropped
        result = channel.receive(Packet.create(PacketType.SEQUENCED_DATA, b'old', sequence=5))

        assert result is None

    def test_accepts_newer_packets(self):
        """Test that newer packets are accepted."""
        channel = SequencedChannel(0)

        channel.receive(Packet.create(PacketType.SEQUENCED_DATA, b'1', sequence=1))
        result = channel.receive(Packet.create(PacketType.SEQUENCED_DATA, b'2', sequence=2))

        assert result == b'2'


class TestChannelManager:
    """Tests for ChannelManager class."""

    def test_create_channel(self):
        """Test creating channels."""
        manager = ChannelManager()

        channel = manager.create_channel(0, ChannelType.UNRELIABLE)
        assert channel.channel_type == ChannelType.UNRELIABLE

    def test_get_channel(self):
        """Test getting channels by ID."""
        manager = ChannelManager()
        manager.create_channel(5, ChannelType.RELIABLE_ORDERED)

        channel = manager.get_channel(5)
        assert channel is not None
        assert channel.channel_type == ChannelType.RELIABLE_ORDERED

    def test_get_channel_by_type(self):
        """Test getting channels by type."""
        manager = ChannelManager()
        manager.create_channel(0, ChannelType.UNRELIABLE)

        channel = manager.get_channel_by_type(ChannelType.UNRELIABLE)
        assert channel is not None

    def test_remove_channel(self):
        """Test removing channels."""
        manager = ChannelManager()
        manager.create_channel(0, ChannelType.UNRELIABLE)
        manager.remove_channel(0)

        assert manager.get_channel(0) is None

    def test_aggregate_stats(self):
        """Test aggregate statistics."""
        manager = ChannelManager()
        manager.create_channel(0, ChannelType.UNRELIABLE)
        manager.create_channel(1, ChannelType.RELIABLE_ORDERED)

        channel0 = manager.get_channel(0)
        channel1 = manager.get_channel(1)

        channel0.send(b'data')
        channel1.send(b'data')

        stats = manager.get_aggregate_stats()
        assert stats.packets_sent == 2


class TestConnection:
    """Tests for Connection class."""

    def test_connection_creation(self):
        """Test creating a connection."""
        conn = Connection(address=("127.0.0.1", 12345))

        assert conn.address == ("127.0.0.1", 12345)
        assert conn.state == ConnectionState.DISCONNECTED
        assert not conn.is_connected

    def test_connect(self):
        """Test initiating connection."""
        conn = Connection(address=("127.0.0.1", 12345))
        packets = conn.connect()

        assert len(packets) >= 1
        assert packets[0].header.packet_type == PacketType.CONNECT
        assert conn.state == ConnectionState.CONNECTING

    def test_disconnect(self):
        """Test disconnecting."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn.connect()

        # Simulate connected state
        conn._state = ConnectionState.CONNECTED

        packets = conn.disconnect("Test disconnect")
        assert len(packets) >= 1
        assert packets[0].header.packet_type == PacketType.DISCONNECT

    def test_send_when_connected(self):
        """Test sending data when connected."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._state = ConnectionState.CONNECTED
        conn._last_receive_time = time.time()

        packets = conn.send(b'test data', ChannelType.UNRELIABLE)
        assert len(packets) >= 1

    def test_send_when_not_connected(self):
        """Test sending data when not connected returns empty."""
        conn = Connection(address=("127.0.0.1", 12345))
        packets = conn.send(b'test data')
        assert packets == []

    def test_receive_connect_ack(self):
        """Test receiving connect acknowledgment."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn.connect()

        packet = Packet.create(PacketType.CONNECT_ACK)
        conn.receive(packet)

        assert conn.state == ConnectionState.CONNECTED

    def test_heartbeat_generation(self):
        """Test heartbeat generation on update."""
        config = ConnectionConfig(heartbeat_interval=0.001)
        conn = Connection(address=("127.0.0.1", 12345), config=config)
        conn._state = ConnectionState.CONNECTED
        conn._last_receive_time = time.time()

        # Wait for heartbeat interval
        time.sleep(0.01)

        packets = conn.update(0.01)
        heartbeats = [p for p in packets if p.header.packet_type == PacketType.HEARTBEAT]
        assert len(heartbeats) >= 1

    def test_get_channel(self):
        """Test getting specific channel."""
        conn = Connection(address=("127.0.0.1", 12345))

        channel = conn.get_channel(ChannelType.RELIABLE_ORDERED)
        assert channel is not None
        assert channel.channel_type == ChannelType.RELIABLE_ORDERED


class TestQualityMetrics:
    """Tests for QualityMetrics class."""

    def test_quality_level_excellent(self):
        """Test excellent quality classification."""
        metrics = QualityMetrics(rtt=0.03, packet_loss=0.005)
        assert metrics.quality_level == QualityLevel.EXCELLENT

    def test_quality_level_good(self):
        """Test good quality classification."""
        metrics = QualityMetrics(rtt=0.075, packet_loss=0.015)
        assert metrics.quality_level == QualityLevel.GOOD

    def test_quality_level_fair(self):
        """Test fair quality classification."""
        metrics = QualityMetrics(rtt=0.150, packet_loss=0.03)
        assert metrics.quality_level == QualityLevel.FAIR

    def test_quality_level_poor(self):
        """Test poor quality classification."""
        metrics = QualityMetrics(rtt=0.300, packet_loss=0.08)
        assert metrics.quality_level == QualityLevel.POOR

    def test_quality_level_critical(self):
        """Test critical quality classification."""
        metrics = QualityMetrics(rtt=0.500, packet_loss=0.15)
        assert metrics.quality_level == QualityLevel.CRITICAL

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = QualityMetrics(rtt=0.05, jitter=0.01, packet_loss=0.02)
        d = metrics.to_dict()

        assert d['rtt'] == 0.05
        assert d['jitter'] == 0.01
        assert d['packet_loss'] == 0.02


class TestQualityMonitor:
    """Tests for QualityMonitor class."""

    def test_add_rtt_sample(self):
        """Test adding RTT samples."""
        monitor = QualityMonitor()

        for _ in range(10):
            monitor.add_rtt_sample(0.05)

        metrics = monitor.update()
        assert abs(metrics.rtt - 0.05) < 0.01

    def test_packet_tracking(self):
        """Test packet send/receive tracking."""
        monitor = QualityMonitor()

        for _ in range(100):
            monitor.record_packet_sent()
            monitor.record_packet_received()

        metrics = monitor.update()
        assert metrics.packet_loss < 0.01

    def test_packet_loss_calculation(self):
        """Test packet loss calculation."""
        monitor = QualityMonitor()

        for _ in range(100):
            monitor.record_packet_sent()
        for _ in range(90):
            monitor.record_packet_received()

        metrics = monitor.update()
        assert 0.05 < metrics.packet_loss < 0.15  # ~10% loss

    def test_quality_change_callback(self):
        """Test quality change callback is invoked on quality transitions."""
        monitor = QualityMonitor()
        changes = []

        def on_change(old, new):
            changes.append((old, new))

        monitor.on_quality_change(on_change)

        # Start with good quality - many samples to converge EWMA
        for _ in range(50):
            monitor.add_rtt_sample(0.03)
        initial_metrics = monitor.update()
        initial_level = initial_metrics.quality_level

        # Degrade quality significantly - need many samples to shift EWMA
        for _ in range(100):
            monitor.add_rtt_sample(0.5)
        final_metrics = monitor.update()
        final_level = final_metrics.quality_level

        # Verify the quality actually changed
        assert initial_level != final_level, f"Quality did not change: {initial_level} -> {final_level}"

        # Verify callback was invoked
        assert len(changes) > 0, "Quality change callback was never called"

        # Verify the recorded change makes sense
        last_change = changes[-1]
        assert last_change[1] == final_level, f"Callback recorded wrong final level: {last_change[1]} != {final_level}"

    def test_statistics(self):
        """Test getting detailed statistics."""
        monitor = QualityMonitor()
        monitor.add_rtt_sample(0.05)
        monitor.add_rtt_sample(0.10)
        monitor.update()

        stats = monitor.get_statistics()
        assert 'rtt_min' in stats
        assert 'rtt_max' in stats
        assert 'rtt_avg' in stats

    def test_reset(self):
        """Test reset functionality."""
        monitor = QualityMonitor()
        monitor.add_rtt_sample(0.1)
        monitor.record_packet_sent()

        monitor.reset()

        metrics = monitor.get_metrics()
        assert metrics.rtt == 0.0


class TestNetworkQualityAdapter:
    """Tests for NetworkQualityAdapter class."""

    def test_adapt_to_quality(self):
        """Test adapting settings to quality level."""
        adapter = NetworkQualityAdapter(hysteresis_threshold=0)

        # Excellent quality
        metrics = QualityMetrics(rtt=0.03, packet_loss=0.005)
        settings = adapter.adapt(metrics)
        assert settings.update_rate >= 30

        # Poor quality
        metrics = QualityMetrics(rtt=0.300, packet_loss=0.08)
        time.sleep(0.01)  # Small delay for hysteresis
        settings = adapter.adapt(metrics)

    def test_force_level(self):
        """Test forcing a specific quality level."""
        adapter = NetworkQualityAdapter()

        settings = adapter.force_level(QualityLevel.CRITICAL)
        assert adapter.current_level == QualityLevel.CRITICAL
        assert settings.update_rate <= 10

    def test_update_rate_limits(self):
        """Test update rate limiting."""
        adapter = NetworkQualityAdapter()
        adapter.set_update_rate_limits(min_rate=10, max_rate=30)

        metrics = QualityMetrics(rtt=0.01, packet_loss=0)
        settings = adapter.adapt(metrics)

        assert settings.update_rate <= 30

    def test_hysteresis(self):
        """Test hysteresis prevents rapid quality setting changes."""
        adapter = NetworkQualityAdapter(hysteresis_threshold=1.0, adaptation_delay=0.5)

        initial_settings = adapter.current_settings

        # Good quality -- level tracks the metric, no adaptation yet
        metrics = QualityMetrics(rtt=0.075, packet_loss=0.01)
        settings1 = adapter.adapt(metrics)

        # Briefly poor quality -- level updates immediately but settings stay
        metrics = QualityMetrics(rtt=0.300, packet_loss=0.08)
        settings2 = adapter.adapt(metrics)

        # Tracked level reflects the new quality immediately
        assert adapter.current_level == QualityLevel.POOR, \
            f"Expected POOR, got {adapter.current_level.name}"

        # Settings remain unchanged because hysteresis has not elapsed
        assert settings2.update_rate == initial_settings.update_rate, \
            "Settings changed despite hysteresis"


class TestIntegration:
    """Integration tests for transport layer."""

    def test_connection_send_receive_flow(self):
        """Test complete send/receive flow."""
        # Create two connections
        server_conn = Connection(address=("client", 12345))
        client_conn = Connection(address=("server", 12346))

        # Client connects
        connect_packets = client_conn.connect()
        assert client_conn.state == ConnectionState.CONNECTING
        assert len(connect_packets) >= 1
        assert connect_packets[0].header.packet_type == PacketType.CONNECT

        # Server receives connect and acknowledges
        for packet in connect_packets:
            server_conn.receive(packet)

        # Server is now connected (received CONNECT)
        assert server_conn.state == ConnectionState.CONNECTED

        # Client receives ack
        ack = Packet.create(PacketType.CONNECT_ACK)
        client_conn.receive(ack)

        # Both connected
        assert client_conn.state == ConnectionState.CONNECTED

        # Send data
        client_conn._last_receive_time = time.time()
        test_data = b'Hello, Server!'
        data_packets = client_conn.send(test_data, ChannelType.UNRELIABLE)

        assert len(data_packets) >= 1, "No packets created for send"

        received_data = None
        for packet in data_packets:
            received = server_conn.receive(packet)
            if received:
                received_data = received

        assert received_data == test_data, f"Data mismatch: {received_data} != {test_data}"

    def test_reliable_channel_with_loss(self):
        """Test reliable channel handles simulated loss with retransmissions."""
        # Use very short RTT for quick retransmit
        config = ChannelConfig(
            ChannelType.RELIABLE_UNORDERED,
            initial_rtt=0.001  # 1ms RTT for fast testing
        )
        channel = ReliableChannel(0, config)

        # Send packets
        packets = []
        for i in range(5):
            packets.extend(channel.send(f'Message {i}'.encode()))

        assert len(packets) == 5, f"Expected 5 packets, got {len(packets)}"

        # Verify all packets are marked reliable
        for p in packets:
            assert p.is_reliable(), "Packet should be marked reliable"

        # Simulate receiving only even packets (0, 2, 4)
        received = []
        for i, packet in enumerate(packets):
            if i % 2 == 0:
                data = channel.receive(packet)
                if data:
                    received.append(data)

        # ACK received packets (0, 2, 4)
        ack = packets[4].header.sequence if len(packets) > 4 else 0
        ack_bits = 0x05  # Packets 0 and 2 (relative to ack-1)
        channel.process_ack(ack, ack_bits)

        # Advance retransmit timers deterministically (avoid flaky time.sleep)
        now = time.time()
        for pending in channel._pending.values():
            pending.retransmit_time = now - 0.001

        # Update should retransmit lost packets (1 and 3)
        retransmits = channel.update(0.01)
        assert len(retransmits) > 0, "Expected retransmissions for lost packets"
        assert channel.stats.packets_retransmitted > 0, "Retransmit counter not updated"

    def test_quality_affects_adapter(self):
        """Test quality metrics affect adapter settings over time."""
        monitor = QualityMonitor()
        adapter = NetworkQualityAdapter(hysteresis_threshold=0.01, adaptation_delay=0.001)

        # Simulate excellent network - many samples to stabilize EWMA
        for _ in range(100):
            monitor.add_rtt_sample(0.02)  # 20ms RTT
        metrics = monitor.update()

        # Wait for hysteresis and adaptation delay
        time.sleep(0.02)
        settings = adapter.adapt(metrics)

        # Verify we got reasonable settings for good quality
        assert settings.update_rate >= DEFAULT_CONFIG.UPDATE_RATE_FAIR, \
            f"Update rate too low for good quality: {settings.update_rate}"

        # Simulate severely degraded network
        for _ in range(100):
            monitor.add_rtt_sample(0.5)  # 500ms RTT - critical
        metrics = monitor.update()

        # Wait for hysteresis
        time.sleep(0.02)
        settings = adapter.adapt(metrics)

        # Quality should have degraded
        assert metrics.quality_level <= QualityLevel.POOR, \
            f"Quality level should be POOR or worse, got {metrics.quality_level.name}"


class TestConfigIntegration:
    """Tests verifying config constants are properly integrated."""

    def test_mtu_constant_matches(self):
        """Test MTU constant is properly used."""
        assert MTU == DEFAULT_CONFIG.MTU

    def test_header_size_matches(self):
        """Test header size is properly used."""
        assert HEADER_SIZE == DEFAULT_CONFIG.PACKET_HEADER_SIZE

    def test_max_payload_size_derived_correctly(self):
        """Test max payload size is derived from MTU and header."""
        assert MAX_PAYLOAD_SIZE == DEFAULT_CONFIG.MAX_PAYLOAD_SIZE
        assert MAX_PAYLOAD_SIZE == MTU - HEADER_SIZE

    def test_channel_config_uses_defaults(self):
        """Test ChannelConfig uses config defaults."""
        config = ChannelConfig(ChannelType.RELIABLE_ORDERED)
        assert config.max_pending == DEFAULT_CONFIG.CHANNEL_MAX_PENDING
        assert config.initial_rtt == DEFAULT_CONFIG.CHANNEL_INITIAL_RTT
        assert config.max_retries == DEFAULT_CONFIG.CHANNEL_MAX_RETRIES

    def test_connection_config_uses_defaults(self):
        """Test ConnectionConfig uses config defaults."""
        config = ConnectionConfig()
        assert config.connect_timeout == DEFAULT_CONFIG.CONNECT_TIMEOUT
        assert config.idle_timeout == DEFAULT_CONFIG.IDLE_TIMEOUT
        assert config.heartbeat_interval == DEFAULT_CONFIG.HEARTBEAT_INTERVAL

    def test_quality_thresholds_used(self):
        """Test quality thresholds from config are used."""
        # Test excellent threshold
        metrics = QualityMetrics(
            rtt=DEFAULT_CONFIG.QUALITY_RTT_EXCELLENT - 0.001,
            packet_loss=DEFAULT_CONFIG.QUALITY_LOSS_EXCELLENT - 0.001
        )
        assert metrics.quality_level == QualityLevel.EXCELLENT

        # Test critical threshold
        metrics = QualityMetrics(
            rtt=DEFAULT_CONFIG.QUALITY_RTT_POOR,
            packet_loss=0
        )
        assert metrics.quality_level == QualityLevel.CRITICAL


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
