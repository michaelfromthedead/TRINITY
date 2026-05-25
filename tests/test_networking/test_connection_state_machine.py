"""
White-box tests for connection state machine edge cases.

Tests the Connection class state transitions, connect/disconnect flows,
packet handling, and callback behavior.
"""

from __future__ import annotations

import time
from unittest import mock
import pytest

from engine.networking.transport.connection import Connection, ConnectionState
from engine.networking.transport.packet import Packet, PacketType, PacketFlags
from engine.networking.transport.channel import ChannelType


class TestConnectionStateMachine:
    """Connection state machine whitebox edge cases."""

    def test_connection_state_enum_values(self):
        """ConnectionState has expected values."""
        assert ConnectionState.DISCONNECTED == 0
        assert ConnectionState.CONNECTING == 1
        assert ConnectionState.CONNECTED == 2
        assert ConnectionState.DISCONNECTING == 3
        assert ConnectionState.FAILED == 4

    def test_connection_initial_state(self):
        """Connection starts disconnected with defaults."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.state == ConnectionState.DISCONNECTED
        assert conn.address == ("127.0.0.1", 12345)
        assert conn.rtt == 0.0
        assert conn.jitter == 0.0
        assert conn.packet_loss == 0.0
        assert conn.is_connected is False
        assert conn.stats is not None

    def test_connect_returns_connect_packet(self):
        """connect() returns CONNECT packet and transitions to CONNECTING."""
        conn = Connection(address=("127.0.0.1", 12345))
        packets = conn.connect()
        assert len(packets) == 1
        assert packets[0].header.packet_type == PacketType.CONNECT
        assert conn.state == ConnectionState.CONNECTING

    def test_double_connect_returns_empty(self):
        """connect() while already connecting returns empty list."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn.connect()
        packets = conn.connect()
        assert packets == []

    def test_connect_ack_transitions_to_connected(self):
        """CONNECT_ACK packet transitions to CONNECTED."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn.connect()
        ack_packet = Packet.create(PacketType.CONNECT_ACK, sequence=1)
        result = conn.receive(ack_packet)
        assert conn.state == ConnectionState.CONNECTED
        assert conn.is_connected is True

    def test_disconnect_cleanup(self):
        """disconnect() returns DISCONNECT packet."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn.connect()
        ack_packet = Packet.create(PacketType.CONNECT_ACK, sequence=1)
        conn.receive(ack_packet)
        assert conn.is_connected

        packets = conn.disconnect(reason="shutdown")
        assert len(packets) == 1
        assert packets[0].header.packet_type == PacketType.DISCONNECT
        assert conn.state == ConnectionState.DISCONNECTING

    def test_disconnect_when_disconnected(self):
        """disconnect() when disconnected returns empty."""
        conn = Connection(address=("127.0.0.1", 12345))
        packets = conn.disconnect("test")
        assert packets == []

    def test_disconnect_when_failed(self):
        """disconnect() when failed returns empty."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._state = ConnectionState.FAILED
        packets = conn.disconnect("test")
        assert packets == []

    def test_receive_data_before_connected(self):
        """receive() returns None when not connected."""
        conn = Connection(address=("127.0.0.1", 12345))
        data_packet = Packet.create(PacketType.DATA, b"data", sequence=1)
        result = conn.receive(data_packet)
        assert result is None

    def test_receive_disconnect_packet(self):
        """DISCONNECT packet transitions to DISCONNECTED."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._state = ConnectionState.CONNECTED
        disc_packet = Packet.create(PacketType.DISCONNECT, b"bye", sequence=1)
        result = conn.receive(disc_packet)
        assert conn.state == ConnectionState.DISCONNECTED
        assert result is None

    def test_receive_unknown_packet_type(self):
        """Unknown packet type safely returns None."""
        conn = Connection(address=("127.0.0.1", 12345))
        packet = Packet.create(PacketType.NACK, b"", sequence=1)
        result = conn.receive(packet)
        assert result is None

    def test_send_returns_packets_when_connected(self):
        """send() returns packets only when connected."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._state = ConnectionState.CONNECTED
        packets = conn.send(b"hello", ChannelType.UNRELIABLE)
        assert len(packets) >= 1

    def test_send_returns_empty_when_not_connected(self):
        """send() returns empty when not connected."""
        conn = Connection(address=("127.0.0.1", 12345))
        packets = conn.send(b"hello", ChannelType.UNRELIABLE)
        assert packets == []

    def test_update_returns_packets(self):
        """update() returns a list of packets."""
        conn = Connection(address=("127.0.0.1", 12345))
        result = conn.update(0.016)
        assert isinstance(result, list)

    def test_update_connect_timeout(self):
        """update() transitions to FAILED on connect timeout."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._config.connect_timeout = 0.1
        conn.connect()
        conn.update(0.2)
        assert conn.state == ConnectionState.FAILED

    def test_update_idle_timeout(self):
        """update() transitions to FAILED on idle timeout."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._state = ConnectionState.CONNECTED
        conn._config.idle_timeout = 0.1
        conn._last_receive_time = 0.0
        with mock.patch('time.time', return_value=10.0):
            conn.update(0.016)
        assert conn.state == ConnectionState.FAILED

    def test_update_heartbeat_sent(self):
        """update() sends heartbeat when due."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._state = ConnectionState.CONNECTED
        conn._config.heartbeat_interval = 1.0
        with mock.patch('time.time', return_value=100.0):
            conn._last_receive_time = 99.0
            conn._last_heartbeat_sent = 98.0
            packets = conn.update(0.016)
        assert len(packets) >= 1
        heartbeats = [p for p in packets if p.header.packet_type == PacketType.HEARTBEAT]
        assert len(heartbeats) == 1

    def test_update_disconnecting_timeout(self):
        """update() transitions to DISCONNECTED after disconnect timeout."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._state = ConnectionState.DISCONNECTING
        conn._config.disconnect_timeout = 0.1
        conn.update(0.2)
        assert conn.state == ConnectionState.DISCONNECTED

    def test_on_connected_callback(self):
        """on_connected callback fires when connection established."""
        conn = Connection(address=("127.0.0.1", 12345))
        callback_data = []
        conn.set_on_connected(lambda c: callback_data.append(c))
        conn._state = ConnectionState.CONNECTING
        ack_packet = Packet.create(PacketType.CONNECT_ACK, sequence=1)
        conn.receive(ack_packet)
        assert len(callback_data) == 1
        assert callback_data[0] is conn

    def test_on_disconnected_callback(self):
        """on_disconnected callback fires on disconnect."""
        conn = Connection(address=("127.0.0.1", 12345))
        callback_data = []
        conn.set_on_disconnected(lambda c, r: callback_data.append((c, r)))
        conn._state = ConnectionState.CONNECTED
        disc_packet = Packet.create(PacketType.DISCONNECT, b"bye", sequence=1)
        conn.receive(disc_packet)
        assert len(callback_data) == 1
        assert callback_data[0][0] is conn
        assert callback_data[0][1] == "bye"

    def test_get_channel_returns_channel(self):
        """get_channel returns channel by type."""
        conn = Connection(address=("127.0.0.1", 12345))
        channel = conn.get_channel(ChannelType.UNRELIABLE)
        assert channel is not None

    def test_create_custom_channel(self):
        """create_channel creates new channel."""
        conn = Connection(address=("127.0.0.1", 12345))
        from engine.networking.transport.channel import ChannelConfig
        config = ChannelConfig(ChannelType.RELIABLE_ORDERED)
        channel = conn.create_channel(99, ChannelType.RELIABLE_ORDERED, config)
        assert channel is not None
        assert channel.channel_id == 99

    def test_get_pending_ack_count(self):
        """get_pending_ack_count returns integer."""
        conn = Connection(address=("127.0.0.1", 12345))
        count = conn.get_pending_ack_count()
        assert count == 0

    def test_stats_property(self):
        """stats property returns ConnectionStats with correct defaults."""
        conn = Connection(address=("127.0.0.1", 12345))
        stats = conn.stats
        assert stats.packets_sent == 0
        assert stats.packets_received == 0
        assert stats.packets_lost == 0
        assert stats.rtt == 0.0
        assert stats.jitter == 0.0

    def test_rtt_property(self):
        """rtt property returns current RTT."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.rtt == 0.0

    def test_jitter_property(self):
        """jitter property returns current jitter."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.jitter == 0.0

    def test_packet_loss_property(self):
        """packet_loss property returns current packet loss."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.packet_loss == 0.0

    # ------------------------------------------------------------------
    # C-2: Connection.receive() state guard — DISCONNECTED returns None
    # ------------------------------------------------------------------

    def test_receive_returns_none_when_disconnected(self):
        """receive() returns None when state is DISCONNECTED (C-2 guard)."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.state == ConnectionState.DISCONNECTED
        packet = Packet.create(PacketType.DATA, b"payload", sequence=1)
        result = conn.receive(packet)
        assert result is None

    def test_receive_disconnected_does_not_update_stats(self):
        """receive() in DISCONNECTED does not increment stats."""
        conn = Connection(address=("127.0.0.1", 12345))
        before_sent = conn.stats.packets_sent
        before_rcvd = conn.stats.packets_received
        before_bytes_rcvd = conn.stats.bytes_received

        packet = Packet.create(PacketType.DATA, b"will be dropped", sequence=1)
        conn.receive(packet)

        assert conn.stats.packets_sent == before_sent
        assert conn.stats.packets_received == before_rcvd
        assert conn.stats.bytes_received == before_bytes_rcvd

    def test_receive_disconnected_does_not_update_sequence(self):
        """receive() in DISCONNECTED does not advance remote sequence."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn._remote_sequence == 0
        packet = Packet.create(PacketType.DATA, b"drop", sequence=99)
        conn.receive(packet)
        assert conn._remote_sequence == 0

    def test_receive_disconnected_does_not_update_last_receive_time(self):
        """receive() in DISCONNECTED does not touch _last_receive_time."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._last_receive_time = 42.0
        packet = Packet.create(PacketType.DATA, b"drop", sequence=1)
        conn.receive(packet)
        assert conn._last_receive_time == 42.0

    def test_receive_works_normally_when_connected(self):
        """receive() processes data normally when CONNECTED (guard does not block)."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._state = ConnectionState.CONNECTED
        data_packet = Packet.create(PacketType.DATA, b"valid", sequence=1)
        result = conn.receive(data_packet)
        assert result == b"valid"

    def test_receive_updates_stats_when_connected(self):
        """receive() increments stats normally when not DISCONNECTED."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._state = ConnectionState.CONNECTED
        payload = b"stats check"
        packet = Packet.create(PacketType.DATA, payload, sequence=1)
        conn.receive(packet)
        assert conn.stats.packets_received == 1
        assert conn.stats.bytes_received == len(payload)

    def test_receive_connect_ack_transitions_to_connected(self):
        """receive() processes CONNECT_ACK during CONNECTING (not blocked by guard)."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn.connect()
        assert conn.state == ConnectionState.CONNECTING
        ack = Packet.create(PacketType.CONNECT_ACK, sequence=1)
        conn.receive(ack)
        assert conn.state == ConnectionState.CONNECTED

    # ------------------------------------------------------------------
    # C-02-R1: DISCONNECTED guard whitelist — protocol packets pass through
    # ------------------------------------------------------------------

    def test_all_data_types_blocked_when_disconnected(self):
        """Every data-family packet type is blocked by the DISCONNECTED guard."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.state == ConnectionState.DISCONNECTED

        for idx, ptype in enumerate([
            PacketType.DATA,
            PacketType.RELIABLE_DATA,
            PacketType.SEQUENCED_DATA,
            PacketType.FRAGMENT,
        ]):
            pkt = Packet.create(ptype, b"payload", sequence=idx)
            result = conn.receive(pkt)
            assert result is None, f"{ptype.name} should be blocked, got {result}"

    def test_data_types_do_not_update_stats_when_disconnected(self):
        """Data packets blocked by guard do not increment stats or timestamps."""
        conn = Connection(address=("127.0.0.1", 12345))
        before_rcvd = conn.stats.packets_received
        before_bytes = conn.stats.bytes_received

        pkt = Packet.create(PacketType.DATA, b"payload", sequence=1)
        conn.receive(pkt)

        assert conn.stats.packets_received == before_rcvd
        assert conn.stats.bytes_received == before_bytes

    def test_connect_passes_guard_and_transitions(self):
        """CONNECT passes the DISCONNECTED guard and transitions to CONNECTED."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.state == ConnectionState.DISCONNECTED

        pkt = Packet.create(PacketType.CONNECT, sequence=1)
        result = conn.receive(pkt)

        assert conn.state == ConnectionState.CONNECTED

    def test_connect_ack_passes_guard_stays_disconnected(self):
        """CONNECT_ACK passes the guard but stays DISCONNECTED (handler only transitions from CONNECTING)."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.state == ConnectionState.DISCONNECTED

        pkt = Packet.create(PacketType.CONNECT_ACK, sequence=1)
        conn.receive(pkt)

        assert conn.state == ConnectionState.DISCONNECTED

    def test_disconnect_passes_guard_stays_disconnected(self):
        """DISCONNECT passes the guard; handler transitions to DISCONNECTED (no-op since already there)."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.state == ConnectionState.DISCONNECTED

        pkt = Packet.create(PacketType.DISCONNECT, b"bye", sequence=1)
        conn.receive(pkt)

        assert conn.state == ConnectionState.DISCONNECTED

    def test_disconnect_ack_passes_guard_returns_none(self):
        """DISCONNECT_ACK passes the guard and is handled gracefully."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.state == ConnectionState.DISCONNECTED

        pkt = Packet.create(PacketType.DISCONNECT_ACK, sequence=1)
        result = conn.receive(pkt)

        assert result is None
        # No handler for DISCONNECT_ACK; falls through to return None at end of receive()
        assert conn.state == ConnectionState.DISCONNECTED

    def test_heartbeat_passes_guard_updates_timestamp(self):
        """HEARTBEAT passes the guard and updates _last_heartbeat_received."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._last_heartbeat_received = 0.0

        pkt = Packet.create(PacketType.HEARTBEAT, sequence=1)
        conn.receive(pkt)

        assert conn._last_heartbeat_received > 0.0

    def test_heartbeat_ack_passes_guard_updates_timestamp(self):
        """HEARTBEAT_ACK passes the guard and updates _last_heartbeat_received."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._last_heartbeat_received = 0.0

        pkt = Packet.create(PacketType.HEARTBEAT_ACK, sequence=1)
        conn.receive(pkt)

        assert conn._last_heartbeat_received > 0.0

    def test_protocol_packets_increment_stats_when_disconnected(self):
        """Protocol packets that pass the guard still increment stats."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.stats.packets_received == 0

        pkt = Packet.create(PacketType.HEARTBEAT, sequence=1)
        conn.receive(pkt)

        assert conn.stats.packets_received == 1
        assert conn.stats.bytes_received == len(pkt.payload)

    def test_nack_passes_guard_returns_none(self):
        """NACK passes the guard and is handled gracefully."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.state == ConnectionState.DISCONNECTED

        pkt = Packet.create(PacketType.NACK, sequence=1)
        result = conn.receive(pkt)

        assert result is None

    def test_ack_passes_guard_returns_none(self):
        """ACK passes the guard and is handled gracefully."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.state == ConnectionState.DISCONNECTED

        pkt = Packet.create(PacketType.ACK, sequence=1)
        result = conn.receive(pkt)

        assert result is None

    def test_guard_does_not_affect_connected_state(self):
        """When CONNECTED, all data packet types work normally (guard only applies when DISCONNECTED)."""
        conn = Connection(address=("127.0.0.1", 12345))
        conn._state = ConnectionState.CONNECTED
        conn._last_receive_time = time.time()

        pkt = Packet.create(PacketType.DATA, b"valid_data", sequence=1)
        result = conn.receive(pkt)

        assert result == b"valid_data"

    def test_all_protocol_types_regression(self):
        """Regression: every known protocol PacketType has predictable behavior through DISCONNECTED guard."""
        conn = Connection(address=("127.0.0.1", 12345))
        assert conn.state == ConnectionState.DISCONNECTED

        protocol_types = [
            PacketType.CONNECT,
            PacketType.CONNECT_ACK,
            PacketType.DISCONNECT,
            PacketType.DISCONNECT_ACK,
            PacketType.HEARTBEAT,
            PacketType.HEARTBEAT_ACK,
            PacketType.ACK,
            PacketType.NACK,
        ]

        for idx, ptype in enumerate(protocol_types):
            pkt = Packet.create(ptype, b"", sequence=idx)
            # Protocol packets should not raise and should return None or have side effects
            try:
                result = conn.receive(pkt)
                # CONNECT transitions to CONNECTED; reset for next iteration
                if ptype == PacketType.CONNECT:
                    assert conn.state == ConnectionState.CONNECTED
                    conn._state = ConnectionState.DISCONNECTED
                else:
                    assert conn.state == ConnectionState.DISCONNECTED
            except Exception as e:
                pytest.fail(f"{ptype.name} raised unexpected exception: {e}")
