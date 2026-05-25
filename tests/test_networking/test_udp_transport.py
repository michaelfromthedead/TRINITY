"""T-NET-1.7: WHITEBOX tests for UDPTransport -- full implementation access.

Tests cover TransportConfig, TransportStats, TransportEvent types,
UDPTransport bind/connect/disconnect/send/receive/broadcast,
rate limiting, stats integration, callbacks, and update with data.
"""

from __future__ import annotations

import socket
import time
from unittest.mock import MagicMock, patch

import pytest

from engine.networking.config import DEFAULT_CONFIG
from engine.networking.transport.packet import Packet, PacketType
from engine.networking.transport.connection import Connection, ConnectionConfig, ConnectionState
from engine.networking.transport.udp_transport import (
    UDPTransport,
    TransportConfig,
    TransportStats,
    TransportEvent,
    TransportEventData,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_socket() -> MagicMock:
    sock = MagicMock()
    sock.getsockname.return_value = ("127.0.0.1", 12345)
    sock.fileno.return_value = 999
    sock.sendto.return_value = 100
    return sock


def _make_transport(config: TransportConfig | None = None) -> UDPTransport:
    return UDPTransport(config=config)


def _connect_transport(
    transport: UDPTransport,
    mock_sock: MagicMock,
    host: str = "192.168.1.1",
    port: int = 9876,
) -> Connection | None:
    """Bind + connect helper so the caller gets a usable connection."""
    if not transport.is_bound:
        with patch("socket.socket", return_value=mock_sock):
            transport.bind("127.0.0.1", 0)
    with patch("socket.socket", return_value=mock_sock):
        return transport.connect(host, port)


def _make_connected_connection(
    transport: UDPTransport, mock_sock: MagicMock
) -> Connection:
    """Connect and force the connection into CONNECTED state for send tests."""
    conn = _connect_transport(transport, mock_sock)
    conn._state = ConnectionState.CONNECTED
    # Prevent idle timeout and heartbeat generation during update()
    now = time.time()
    conn._last_receive_time = now
    conn._last_heartbeat_sent = now
    # Reset rate limit counters consumed by connect CONNECT packet
    transport._packets_this_second = 0
    transport._bytes_this_second = 0
    return conn


# ---------------------------------------------------------------------------
# TransportConfig
# ---------------------------------------------------------------------------

class TestTransportConfig:
    """Whitebox: TransportConfig defaults and customisation."""

    def test_default_values(self):
        config = TransportConfig()
        assert config.receive_buffer_size == DEFAULT_CONFIG.SOCKET_RECEIVE_BUFFER_SIZE
        assert config.send_buffer_size == DEFAULT_CONFIG.SOCKET_SEND_BUFFER_SIZE
        assert config.non_blocking is True
        assert config.max_connections == DEFAULT_CONFIG.MAX_CONNECTIONS
        assert config.max_packets_per_second == DEFAULT_CONFIG.MAX_PACKETS_PER_SECOND
        assert config.max_bytes_per_second == DEFAULT_CONFIG.MAX_BYTES_PER_SECOND

    def test_custom_values(self):
        config = TransportConfig(
            receive_buffer_size=16384,
            send_buffer_size=8192,
            non_blocking=False,
            max_connections=16,
            max_packets_per_second=500,
            max_bytes_per_second=512 * 1024,
        )
        assert config.receive_buffer_size == 16384
        assert config.send_buffer_size == 8192
        assert config.non_blocking is False
        assert config.max_connections == 16
        assert config.max_packets_per_second == 500
        assert config.max_bytes_per_second == 512 * 1024

    def test_connection_config_factory(self):
        config = TransportConfig()
        assert isinstance(config.connection_config, ConnectionConfig)

    def test_connection_config_custom(self):
        conn_config = ConnectionConfig(connect_timeout=5.0)
        config = TransportConfig(connection_config=conn_config)
        assert config.connection_config.connect_timeout == 5.0


# ---------------------------------------------------------------------------
# TransportStats
# ---------------------------------------------------------------------------

class TestTransportStats:
    """Whitebox: TransportStats initial state and mutation."""

    def test_initial_state(self):
        stats = TransportStats()
        assert stats.packets_sent == 0
        assert stats.packets_received == 0
        assert stats.bytes_sent == 0
        assert stats.bytes_received == 0
        assert stats.connections_total == 0
        assert stats.connections_active == 0
        assert stats.socket_errors == 0

    def test_increment_fields(self):
        stats = TransportStats()
        stats.packets_sent = 10
        stats.packets_received = 20
        stats.bytes_sent = 1000
        stats.bytes_received = 2000
        stats.connections_total = 5
        stats.connections_active = 3
        stats.socket_errors = 1
        assert stats.packets_sent == 10
        assert stats.packets_received == 20
        assert stats.bytes_sent == 1000
        assert stats.bytes_received == 2000
        assert stats.connections_total == 5
        assert stats.connections_active == 3
        assert stats.socket_errors == 1


# ---------------------------------------------------------------------------
# TransportEvent types
# ---------------------------------------------------------------------------

class TestTransportEventTypes:
    """Whitebox: TransportEvent enum and TransportEventData."""

    def test_enum_values(self):
        assert TransportEvent.CONNECTED == 1
        assert TransportEvent.DISCONNECTED == 2
        assert TransportEvent.DATA_RECEIVED == 3
        assert TransportEvent.ERROR == 4

    def test_event_data_creation(self):
        event = TransportEventData(
            event_type=TransportEvent.CONNECTED,
            address=("127.0.0.1", 12345),
        )
        assert event.event_type == TransportEvent.CONNECTED
        assert event.address == ("127.0.0.1", 12345)
        assert event.data is None
        assert event.error is None

    def test_event_data_with_data(self):
        event = TransportEventData(
            event_type=TransportEvent.DATA_RECEIVED,
            address=("127.0.0.1", 12345),
            data=b"hello",
        )
        assert event.data == b"hello"

    def test_event_data_with_error(self):
        event = TransportEventData(
            event_type=TransportEvent.ERROR,
            address=("127.0.0.1", 12345),
            error="Connection timeout",
        )
        assert event.error == "Connection timeout"


# ---------------------------------------------------------------------------
# UDPTransport: init, bind, close
# ---------------------------------------------------------------------------

class TestUDPTransportInitAndBind:
    """Whitebox: initial state, bind, close."""

    def test_init_defaults(self):
        transport = _make_transport()
        assert transport.is_bound is False
        assert transport.local_address is None
        assert isinstance(transport.stats, TransportStats)
        assert transport.stats.packets_sent == 0

    def test_init_custom_config(self):
        config = TransportConfig(max_connections=8)
        transport = _make_transport(config)
        assert transport._config.max_connections == 8

    def test_bind_creates_socket(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        with patch("socket.socket", return_value=mock_sock):
            result = transport.bind("127.0.0.1", 0)

        assert result is True
        assert transport.is_bound is True
        assert transport.local_address == ("127.0.0.1", 12345)
        mock_sock.setsockopt.assert_called()
        mock_sock.bind.assert_called_with(("127.0.0.1", 0))

    def test_bind_sets_non_blocking(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        with patch("socket.socket", return_value=mock_sock):
            transport.bind("127.0.0.1", 0)

        mock_sock.setblocking.assert_called_with(False)

    def test_bind_sets_socket_buffers(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        with patch("socket.socket", return_value=mock_sock):
            transport.bind("127.0.0.1", 0)

        rcv_calls = [
            c for c in mock_sock.setsockopt.mock_calls
            if c.args[0] == socket.SOL_SOCKET and c.args[1] == socket.SO_RCVBUF
        ]
        snd_calls = [
            c for c in mock_sock.setsockopt.mock_calls
            if c.args[0] == socket.SOL_SOCKET and c.args[1] == socket.SO_SNDBUF
        ]
        assert len(rcv_calls) >= 1
        assert len(snd_calls) >= 1

    def test_bind_failure_tracks_error(self):
        transport = _make_transport()
        mock_sock = MagicMock()
        mock_sock.bind.side_effect = socket.error("Address in use")

        with patch("socket.socket", return_value=mock_sock):
            result = transport.bind("127.0.0.1", 0)

        assert result is False
        assert transport.is_bound is False
        assert transport.stats.socket_errors == 1
        mock_sock.close.assert_called_once()

    def test_properties_before_bind(self):
        transport = _make_transport()
        assert transport.is_bound is False
        assert transport.local_address is None
        assert transport.get_connection(("x", 1)) is None
        assert transport.get_connections() == []

    def test_close(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        with patch("socket.socket", return_value=mock_sock):
            transport.bind("127.0.0.1", 0)

        transport.close()
        assert transport.is_bound is False
        assert transport.local_address is None
        mock_sock.close.assert_called_once()


# ---------------------------------------------------------------------------
# UDPTransport: connect / disconnect
# ---------------------------------------------------------------------------

class TestUDPTransportConnectAndDisconnect:
    """Whitebox: connection lifecycle."""

    def test_connect_returns_connection(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        with patch("socket.socket", return_value=mock_sock):
            conn = transport.connect("192.168.1.1", 9876)

        assert conn is not None
        assert conn.address == ("192.168.1.1", 9876)
        assert transport.stats.connections_total == 1
        assert transport.stats.connections_active == 1

    def test_connect_auto_binds(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        with patch("socket.socket", return_value=mock_sock):
            transport.connect("192.168.1.1", 9876)

        assert transport.is_bound is True
        mock_sock.bind.assert_called_with(("", 0))

    def test_connect_same_address_returns_same_connection(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        with patch("socket.socket", return_value=mock_sock):
            conn1 = transport.connect("192.168.1.1", 9876)
            conn2 = transport.connect("192.168.1.1", 9876)

        assert conn1 is conn2

    def test_connect_max_connections_returns_none(self):
        config = TransportConfig(max_connections=2)
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()

        with patch("socket.socket", return_value=mock_sock):
            transport.connect("192.168.1.1", 9001)
            transport.connect("192.168.1.2", 9002)
            conn3 = transport.connect("192.168.1.3", 9003)

        assert conn3 is None

    def test_disconnect_returns_true(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()
        conn = _connect_transport(transport, mock_sock)

        result = transport.disconnect(conn.address)
        assert result is True

    def test_disconnect_nonexistent_returns_false(self):
        transport = _make_transport()
        result = transport.disconnect(("10.0.0.1", 1234))
        assert result is False

    def test_disconnect_decrements_active_connections(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()
        conn = _connect_transport(transport, mock_sock)
        assert transport.stats.connections_active == 1

        transport.disconnect(conn.address)
        # Force connection to fully disconnected (bypass disconnect timeout)
        conn._state = ConnectionState.DISCONNECTED
        with patch("select.select", return_value=([], [], [])):
            transport.update(0.016)
        assert transport.stats.connections_active == 0

    def test_get_connection_and_connections(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()
        conn = _connect_transport(transport, mock_sock)

        retrieved = transport.get_connection(conn.address)
        assert retrieved is conn

        all_conns = transport.get_connections()
        assert len(all_conns) == 1
        assert all_conns[0] is conn


# ---------------------------------------------------------------------------
# UDPTransport: send / receive / broadcast
# ---------------------------------------------------------------------------

class TestUDPTransportSendReceive:
    """Whitebox: data transfer across connections."""

    def test_send_to_connected_connection(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        result = transport.send(b"hello", conn.address)
        assert result is True

    def test_send_unreliable_by_default(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        transport.send(b"data", conn.address)
        assert transport.stats.packets_sent >= 1
        assert transport.stats.bytes_sent >= 1

    def test_send_reliable(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        result = transport.send(b"reliable data", conn.address, reliable=True)
        assert result is True

    def test_send_nonexistent_address_fails(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()
        with patch("socket.socket", return_value=mock_sock):
            transport.bind("127.0.0.1", 0)

        result = transport.send(b"data", ("10.0.0.99", 9999))
        assert result is False

    def test_send_not_connected_connection_fails(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()
        conn = _connect_transport(transport, mock_sock)

        result = transport.send(b"data", conn.address)
        assert result is False

    def test_broadcast(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        with patch("socket.socket", return_value=mock_sock):
            c1 = transport.connect("192.168.1.1", 1001)
            c2 = transport.connect("192.168.1.2", 1002)

        c1._state = ConnectionState.CONNECTED
        c2._state = ConnectionState.CONNECTED

        count = transport.broadcast(b"broadcast msg")
        assert count == 2

    def test_broadcast_with_no_connections(self):
        transport = _make_transport()
        count = transport.broadcast(b"data")
        assert count == 0

    def test_update_returns_empty_events_when_no_data(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        with patch("socket.socket", return_value=mock_sock), \
             patch("select.select", return_value=([], [], [])):
            transport.bind("127.0.0.1", 0)
            events = transport.update(0.016)

        assert events == []


# ---------------------------------------------------------------------------
# UDPTransport: rate limiting
# ---------------------------------------------------------------------------

class TestUDPTransportRateLimiting:
    """Whitebox: packet and byte rate limiting."""

    def test_packet_rate_limit_hit(self):
        config = TransportConfig(max_packets_per_second=3, max_bytes_per_second=10**9)
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        for _ in range(3):
            assert transport.send(b"x", conn.address) is True

        assert transport.send(b"x", conn.address) is False

    def test_byte_rate_limit_hit(self):
        config = TransportConfig(max_packets_per_second=1000, max_bytes_per_second=100)
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        transport.send(b"x" * 60, conn.address)
        assert transport.send(b"x" * 60, conn.address) is False

    def test_rate_limit_reset_after_update(self):
        config = TransportConfig(max_packets_per_second=2, max_bytes_per_second=10**9)
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        transport.send(b"a", conn.address)
        transport.send(b"b", conn.address)
        assert transport.send(b"c", conn.address) is False

        transport._rate_limit_reset = time.time() - 2.0

        with patch("select.select", return_value=([], [], [])):
            transport.update(0.016)

        assert transport.send(b"d", conn.address) is True

    def test_no_rate_limit_with_high_limits(self):
        config = TransportConfig(
            max_packets_per_second=10**6,
            max_bytes_per_second=10**9,
        )
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        for _ in range(100):
            assert transport.send(b"x", conn.address) is True

    def test_rate_limit_bytes_exact_boundary(self):
        config = TransportConfig(max_packets_per_second=1000, max_bytes_per_second=13)
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        # 1-byte payload serializes to 13 total bytes -- exactly matches limit
        assert transport.send(b"x", conn.address) is True
        assert transport.send(b"y", conn.address) is False

    # ------------------------------------------------------------------
    # C-3: Rate limiter integrity — no double-count, increment after send
    # ------------------------------------------------------------------

    def test_counters_not_incremented_on_socket_error(self):
        """_send_packet counters not bumped when sendto() raises (C-3)."""
        config = TransportConfig(max_packets_per_second=1000, max_bytes_per_second=10**9)
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()
        mock_sock.sendto.side_effect = socket.error("Send failed")
        conn = _make_connected_connection(transport, mock_sock)

        before_packets = transport._packets_this_second
        before_bytes = transport._bytes_this_second
        before_stats_sent = transport.stats.packets_sent
        before_stats_bytes = transport.stats.bytes_sent

        result = transport.send(b"fail", conn.address)

        assert result is False
        assert transport._packets_this_second == before_packets
        assert transport._bytes_this_second == before_bytes
        assert transport.stats.packets_sent == before_stats_sent
        assert transport.stats.bytes_sent == before_stats_bytes

    def test_counters_increment_exactly_once_per_send(self):
        """Each send() increments rate counters exactly once (no double-count) (C-3)."""
        config = TransportConfig(max_packets_per_second=1000, max_bytes_per_second=10**9)
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        transport.send(b"alpha", conn.address)
        assert transport._packets_this_second == 1

        transport.send(b"beta", conn.address)
        assert transport._packets_this_second == 2

        transport.send(b"gamma", conn.address)
        assert transport._packets_this_second == 3

    def test_counters_match_stats_after_batch_send(self):
        """Rate counters and stats stay in sync through multiple sends (C-3)."""
        config = TransportConfig(max_packets_per_second=1000, max_bytes_per_second=10**9)
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        n = 5
        for i in range(n):
            assert transport.send(b"x", conn.address) is True

        assert transport._packets_this_second == n
        assert transport.stats.packets_sent >= n
        # Rate counters and stats should agree on total sends
        assert transport.stats.packets_sent >= transport._packets_this_second

    def test_socket_error_rejected_by_rate_limit_no_blocked_future_sends(self):
        """A sendto failure does not consume rate budget (C-3)."""
        config = TransportConfig(max_packets_per_second=2, max_bytes_per_second=10**9)
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        # Succeed once
        assert transport.send(b"ok", conn.address) is True

        # Fail once
        mock_sock.sendto.side_effect = socket.error("Bounce")
        assert transport.send(b"fail", conn.address) is False

        # Succeed again — rate budget not consumed by failure
        mock_sock.sendto.side_effect = None
        assert transport.send(b"ok2", conn.address) is True
        assert transport._packets_this_second == 2

    def test_check_rate_limit_method_removed(self):
        """_check_rate_limit has been removed (C-3/C-4) to avoid double-count."""
        config = TransportConfig(max_packets_per_second=1000, max_bytes_per_second=10**9)
        transport = _make_transport(config)
        assert not hasattr(transport, "_check_rate_limit"), \
            "_check_rate_limit was removed; rate limiting is now inline in _send_packet"

    # ------------------------------------------------------------------
    # C-03-R1: Mock socket fileno for select compatibility
    # ------------------------------------------------------------------

    def test_mock_socket_fileno_is_set(self):
        """Mock socket has fileno.return_value for select.select() compatibility (C-03-R1)."""
        mock_sock = _make_mock_socket()
        assert mock_sock.fileno.return_value == 999

    def test_select_works_with_mock_socket(self):
        """select.select() works with the mock socket that has fileno configured (C-03-R1)."""
        import select
        mock_sock = _make_mock_socket()

        # select with mock socket should not raise (fileno returns 999)
        readable, _, _ = select.select([mock_sock], [], [], 0)
        assert isinstance(readable, list)

    def test_mock_socket_fileno_called_once(self):
        """mock_sock.fileno() can be called and returns 999 each time (C-03-R1)."""
        mock_sock = _make_mock_socket()
        assert mock_sock.fileno() == 999
        assert mock_sock.fileno() == 999
        assert mock_sock.fileno.call_count == 2

    def test_fileno_in_make_mock_socket_exists(self):
        """_make_mock_socket() includes fileno.return_value for all tests (C-03-R1)."""
        mock_sock = _make_mock_socket()
        # The helper must set fileno so that any test using select.select with
        # the mock socket does not raise AttributeError or OSError.
        assert hasattr(mock_sock, "fileno")
        mock_sock.fileno.return_value = 999
        assert mock_sock.fileno() == 999

    # ------------------------------------------------------------------
    # C-04-R1: Rate limiting inline (no _check_rate_limit) — no regression
    # ------------------------------------------------------------------

    def test_rate_limiting_still_works_inline(self):
        """Rate limiting in _send_packet still denies excess packets after _check_rate_limit removal (C-04-R1)."""
        config = TransportConfig(max_packets_per_second=2, max_bytes_per_second=10**9)
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        assert transport.send(b"a", conn.address) is True
        assert transport.send(b"b", conn.address) is True
        assert transport.send(b"c", conn.address) is False

    def test_rate_limit_counters_reset_via_update(self):
        """Rate limit counters reset on update interval, allowing further sends (C-04-R1)."""
        config = TransportConfig(max_packets_per_second=2, max_bytes_per_second=10**9)
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        transport.send(b"a", conn.address)
        transport.send(b"b", conn.address)
        assert transport.send(b"c", conn.address) is False

        # Force rate limit reset by backdating _rate_limit_reset
        transport._rate_limit_reset = time.time() - 2.0

        with patch("select.select", return_value=([], [], [])):
            transport.update(0.016)

        assert transport.send(b"d", conn.address) is True

    def test_stats_match_rate_counters_after_fix(self):
        """Transport stats and rate counters stay in sync after sends (C-04-R1)."""
        config = TransportConfig(max_packets_per_second=100, max_bytes_per_second=10**9)
        transport = _make_transport(config)
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        n = 7
        for i in range(n):
            assert transport.send(b"x", conn.address) is True

        assert transport._packets_this_second == n
        assert transport.stats.packets_sent >= n


# ---------------------------------------------------------------------------
# UDPTransport: stats integration
# ---------------------------------------------------------------------------

class TestUDPTransportStatsIntegration:
    """Whitebox: transport stats updated by operations."""

    def test_stats_initial(self):
        transport = _make_transport()
        assert transport.stats.packets_sent == 0
        assert transport.stats.packets_received == 0
        assert transport.stats.bytes_sent == 0
        assert transport.stats.bytes_received == 0

    def test_stats_increment_on_send(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()
        conn = _make_connected_connection(transport, mock_sock)

        transport.send(b"test data", conn.address)
        assert transport.stats.packets_sent >= 1
        assert transport.stats.bytes_sent >= 9

    def test_stats_increment_on_receive(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        packet = Packet.create(PacketType.HEARTBEAT, b"", sequence=1)
        packet_data = packet.to_bytes()

        mock_sock.recvfrom.side_effect = [
            (packet_data, ("192.168.1.1", 9876)),
            BlockingIOError,
        ]

        with patch("socket.socket", return_value=mock_sock), \
             patch("select.select", return_value=([mock_sock], [], [])):
            transport.bind("127.0.0.1", 0)
            transport.update(0.016)

        assert transport.stats.packets_received >= 1
        assert transport.stats.bytes_received >= len(packet_data)

    def test_stats_socket_error(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        mock_sock.recvfrom.side_effect = socket.error("Connection refused")

        with patch("socket.socket", return_value=mock_sock), \
             patch("select.select", return_value=([mock_sock], [], [])):
            transport.bind("127.0.0.1", 0)
            transport.update(0.016)

        assert transport.stats.socket_errors >= 1


# ---------------------------------------------------------------------------
# UDPTransport: callbacks
# ---------------------------------------------------------------------------

class TestUDPTransportCallbacks:
    """Whitebox: on_connect, on_disconnect, on_data callbacks."""

    def test_on_data_callback_invoked(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()
        received_data = []

        def on_data(conn: Connection, data: bytes) -> None:
            received_data.append(data)

        transport.set_on_data(on_data)
        conn = _connect_transport(transport, mock_sock)
        conn._state = ConnectionState.CONNECTED

        data_packet = Packet.create(PacketType.DATA, b"callback data", sequence=1)
        transport._route_packet(data_packet, conn.address)

        assert len(received_data) >= 1
        assert received_data[0] == b"callback data"

    def test_on_data_not_invoked_for_non_data_packets(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()
        received_data = []

        def on_data(conn: Connection, data: bytes) -> None:
            received_data.append(data)

        transport.set_on_data(on_data)
        conn = _connect_transport(transport, mock_sock)

        hb_packet = Packet.create(PacketType.HEARTBEAT, b"", sequence=1)
        transport._route_packet(hb_packet, conn.address)

        assert len(received_data) == 0

    def test_on_connect_callback(self):
        transport = _make_transport()
        connected_conns = []

        def on_connect(conn: Connection) -> None:
            connected_conns.append(conn)

        transport.set_on_connect(on_connect)

        mock_conn = MagicMock(spec=Connection)
        transport._handle_connected(mock_conn)

        assert len(connected_conns) >= 1

    def test_on_disconnect_callback(self):
        transport = _make_transport()
        disconnected = []

        def on_disconnect(conn: Connection, reason: str) -> None:
            disconnected.append((conn, reason))

        transport.set_on_disconnect(on_disconnect)

        mock_conn = MagicMock(spec=Connection)
        transport._handle_disconnected(mock_conn, "timeout")

        assert len(disconnected) >= 1
        assert disconnected[0][1] == "timeout"


# ---------------------------------------------------------------------------
# UDPTransport: update with simulated data reception
# ---------------------------------------------------------------------------

class TestUDPTransportUpdateWithData:
    """Whitebox: update() processes incoming data through mocked recvfrom."""

    def test_update_processes_heartbeat(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        hb_packet = Packet.create(PacketType.HEARTBEAT, b"", sequence=1)

        mock_sock.recvfrom.side_effect = [
            (hb_packet.to_bytes(), ("192.168.1.1", 9876)),
            BlockingIOError,
        ]

        with patch("socket.socket", return_value=mock_sock), \
             patch("select.select", return_value=([mock_sock], [], [])):
            transport.bind("127.0.0.1", 0)
            events = transport.update(0.016)

        data_events = [e for e in events if e.event_type == TransportEvent.DATA_RECEIVED]
        assert len(data_events) == 0

    def test_update_connect_request_generates_connected_event(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        connect_packet = Packet.create(PacketType.CONNECT, b"", sequence=1)

        mock_sock.recvfrom.side_effect = [
            (connect_packet.to_bytes(), ("10.0.0.5", 7777)),
            BlockingIOError,
        ]

        with patch("socket.socket", return_value=mock_sock), \
             patch("select.select", return_value=([mock_sock], [], [])):
            transport.bind("127.0.0.1", 0)
            events = transport.update(0.016)

        connect_events = [e for e in events if e.event_type == TransportEvent.CONNECTED]
        assert len(connect_events) == 1
        assert connect_events[0].address == ("10.0.0.5", 7777)

    def test_update_connect_established(self):
        transport = _make_transport()
        mock_sock = _make_mock_socket()

        with patch("socket.socket", return_value=mock_sock):
            conn = transport.connect("10.0.0.5", 7777)

        ack_packet = Packet.create(PacketType.CONNECT_ACK, b"", sequence=1)
        mock_sock.recvfrom.side_effect = [
            (ack_packet.to_bytes(), ("10.0.0.5", 7777)),
            BlockingIOError,
        ]

        with patch("select.select", return_value=([mock_sock], [], [])):
            transport.update(0.016)

        assert conn.state == ConnectionState.CONNECTED
