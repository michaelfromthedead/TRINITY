"""
UDP transport layer implementation.

Provides non-blocking UDP socket handling for game networking
with connection management and packet routing.
"""

from __future__ import annotations

import logging
import select
import socket
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple, Any

from .packet import Packet, PacketType, MTU
from .connection import Connection, ConnectionConfig, ConnectionState
from ..config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class TransportConfig:
    """Configuration for UDP transport."""
    # Socket settings
    receive_buffer_size: int = DEFAULT_CONFIG.SOCKET_RECEIVE_BUFFER_SIZE
    send_buffer_size: int = DEFAULT_CONFIG.SOCKET_SEND_BUFFER_SIZE
    non_blocking: bool = True

    # Connection settings
    max_connections: int = DEFAULT_CONFIG.MAX_CONNECTIONS
    connection_config: ConnectionConfig = field(default_factory=ConnectionConfig)

    # Rate limiting
    max_packets_per_second: int = DEFAULT_CONFIG.MAX_PACKETS_PER_SECOND
    max_bytes_per_second: int = DEFAULT_CONFIG.MAX_BYTES_PER_SECOND


@dataclass
class TransportStats:
    """Statistics for the transport layer."""
    packets_sent: int = 0
    packets_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    connections_total: int = 0
    connections_active: int = 0
    socket_errors: int = 0


class TransportEvent(IntEnum):
    """Events from the transport layer."""
    CONNECTED = 1
    DISCONNECTED = 2
    DATA_RECEIVED = 3
    ERROR = 4


@dataclass
class TransportEventData:
    """Data for a transport event."""
    event_type: TransportEvent
    address: Tuple[str, int]
    data: Optional[bytes] = None
    error: Optional[str] = None


class UDPTransport:
    """
    Non-blocking UDP transport for game networking.

    Manages UDP socket, connections, and packet routing.

    Example:
        # Server
        server = UDPTransport()
        server.bind("0.0.0.0", 12345)

        while running:
            events = server.update(0.016)  # 60 FPS
            for event in events:
                if event.event_type == TransportEvent.DATA_RECEIVED:
                    process(event.data)

        # Client
        client = UDPTransport()
        client.connect("server.example.com", 12345)

        while running:
            client.send(data, server_address)
            events = client.update(0.016)
    """

    def __init__(self, config: Optional[TransportConfig] = None) -> None:
        """
        Initialize the UDP transport.

        Args:
            config: Transport configuration.
        """
        self._config = config or TransportConfig()
        self._socket: Optional[socket.socket] = None
        self._bound = False
        self._local_address: Optional[Tuple[str, int]] = None

        # Connections
        self._connections: Dict[Tuple[str, int], Connection] = {}

        # Statistics
        self._stats = TransportStats()

        # Rate limiting
        self._packets_this_second = 0
        self._bytes_this_second = 0
        self._rate_limit_reset = 0.0

        # Callbacks
        self._on_connect: Optional[Callable[[Connection], None]] = None
        self._on_disconnect: Optional[Callable[[Connection, str], None]] = None
        self._on_data: Optional[Callable[[Connection, bytes], None]] = None

    @property
    def is_bound(self) -> bool:
        """Check if socket is bound."""
        return self._bound

    @property
    def local_address(self) -> Optional[Tuple[str, int]]:
        """Get local bound address."""
        return self._local_address

    @property
    def stats(self) -> TransportStats:
        """Get transport statistics."""
        return self._stats

    def bind(self, host: str, port: int) -> bool:
        """
        Bind the socket to a local address.

        Args:
            host: Host address to bind (e.g., "0.0.0.0").
            port: Port number.

        Returns:
            True if successful.
        """
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # Set socket options
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_RCVBUF,
                self._config.receive_buffer_size
            )
            self._socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_SNDBUF,
                self._config.send_buffer_size
            )

            if self._config.non_blocking:
                self._socket.setblocking(False)

            self._socket.bind((host, port))
            self._local_address = self._socket.getsockname()
            self._bound = True

            return True

        except socket.error as e:
            logger.error("Failed to bind socket to %s:%d: %s", host, port, e)
            self._stats.socket_errors += 1
            if self._socket:
                self._socket.close()
                self._socket = None
            return False

    def close(self) -> None:
        """Close the socket and all connections."""
        # Disconnect all connections
        for conn in list(self._connections.values()):
            self._disconnect_connection(conn, "Transport closing")

        if self._socket:
            self._socket.close()
            self._socket = None

        self._bound = False
        self._local_address = None

    def connect(self, host: str, port: int) -> Optional[Connection]:
        """
        Create a connection to a remote host.

        Args:
            host: Remote host address.
            port: Remote port.

        Returns:
            The connection object, or None on failure.
        """
        # Create socket if needed
        if not self._socket:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self._config.non_blocking:
                self._socket.setblocking(False)
            self._socket.bind(('', 0))  # Bind to any available port
            self._local_address = self._socket.getsockname()
            self._bound = True

        address = (host, port)

        # Check if already connected
        if address in self._connections:
            return self._connections[address]

        # Check connection limit
        if len(self._connections) >= self._config.max_connections:
            return None

        # Create connection
        conn = Connection(
            address=address,
            config=self._config.connection_config
        )
        conn.set_on_connected(self._handle_connected)
        conn.set_on_disconnected(self._handle_disconnected)

        self._connections[address] = conn
        self._stats.connections_total += 1
        self._stats.connections_active += 1

        # Send connect packet
        packets = conn.connect()
        for packet in packets:
            self._send_packet(packet, address)

        return conn

    def disconnect(self, address: Tuple[str, int], reason: str = "User requested") -> bool:
        """
        Disconnect from a remote host.

        Args:
            address: Remote address tuple.
            reason: Disconnect reason.

        Returns:
            True if connection existed.
        """
        conn = self._connections.get(address)
        if not conn:
            return False

        self._disconnect_connection(conn, reason)
        return True

    def send(
        self,
        data: bytes,
        address: Tuple[str, int],
        reliable: bool = False
    ) -> bool:
        """
        Send data to a remote address.

        Args:
            data: Data to send.
            address: Destination address.
            reliable: Whether to use reliable delivery.

        Returns:
            True if sent successfully.
        """
        conn = self._connections.get(address)
        if not conn or not conn.is_connected:
            return False

        # Send through connection
        from .channel import ChannelType
        channel_type = ChannelType.RELIABLE_ORDERED if reliable else ChannelType.UNRELIABLE

        packets = conn.send(data, channel_type)
        for packet in packets:
            if not self._send_packet(packet, address):
                return False

        return True

    def broadcast(self, data: bytes, reliable: bool = False) -> int:
        """
        Send data to all connected clients.

        Args:
            data: Data to send.
            reliable: Whether to use reliable delivery.

        Returns:
            Number of connections sent to.
        """
        count = 0
        for address in list(self._connections.keys()):
            if self.send(data, address, reliable):
                count += 1
        return count

    def update(self, dt: float) -> List[TransportEventData]:
        """
        Update the transport layer.

        Receives packets, updates connections, and returns events.

        Args:
            dt: Time delta since last update.

        Returns:
            List of transport events.
        """
        events = []

        # Reset rate limit counter
        now = time.time()
        if now > self._rate_limit_reset:
            self._packets_this_second = 0
            self._bytes_this_second = 0
            self._rate_limit_reset = now + DEFAULT_CONFIG.RATE_LIMIT_RESET_INTERVAL

        # Receive packets
        events.extend(self._receive_packets())

        # Update connections
        for address, conn in list(self._connections.items()):
            packets = conn.update(dt)
            for packet in packets:
                self._send_packet(packet, address)

            # Check for disconnection
            if conn.state in (ConnectionState.DISCONNECTED, ConnectionState.FAILED):
                self._cleanup_connection(address)

        return events

    def _receive_packets(self) -> List[TransportEventData]:
        """Receive and process pending packets."""
        events = []

        if not self._socket:
            return events

        while True:
            try:
                # Check if data is available
                if self._config.non_blocking:
                    readable, _, _ = select.select([self._socket], [], [], 0)
                    if not readable:
                        break

                data, address = self._socket.recvfrom(MTU)
                if not data:
                    continue

                self._stats.packets_received += 1
                self._stats.bytes_received += len(data)

                # Parse packet
                try:
                    packet = Packet.from_bytes(data)
                except ValueError:
                    continue

                # Route to connection
                event = self._route_packet(packet, address)
                if event:
                    events.append(event)

            except BlockingIOError:
                break
            except socket.error as e:
                logger.warning("Socket error during receive: %s", e)
                self._stats.socket_errors += 1
                break

        return events

    def _route_packet(
        self,
        packet: Packet,
        address: Tuple[str, int]
    ) -> Optional[TransportEventData]:
        """Route a received packet to appropriate connection."""
        ptype = packet.header.packet_type

        # Handle connection requests
        if ptype == PacketType.CONNECT:
            return self._handle_connect_request(packet, address)

        # Get existing connection
        conn = self._connections.get(address)
        if not conn:
            return None

        # Process through connection
        data = conn.receive(packet)

        if data:
            # Data callback
            if self._on_data:
                self._on_data(conn, data)

            return TransportEventData(
                event_type=TransportEvent.DATA_RECEIVED,
                address=address,
                data=data
            )

        return None

    def _handle_connect_request(
        self,
        packet: Packet,
        address: Tuple[str, int]
    ) -> Optional[TransportEventData]:
        """Handle incoming connection request."""
        # Check if already connected
        if address in self._connections:
            conn = self._connections[address]
            conn.receive(packet)
            return None

        # Check connection limit
        if len(self._connections) >= self._config.max_connections:
            # Send reject
            reject = Packet.create(PacketType.DISCONNECT, b"Server full")
            self._send_packet(reject, address)
            return None

        # Accept connection
        conn = Connection(
            address=address,
            config=self._config.connection_config
        )
        conn.set_on_connected(self._handle_connected)
        conn.set_on_disconnected(self._handle_disconnected)

        self._connections[address] = conn
        self._stats.connections_total += 1
        self._stats.connections_active += 1

        # Process connect packet
        conn.receive(packet)

        # Send connect ACK
        ack = Packet.create(PacketType.CONNECT_ACK)
        self._send_packet(ack, address)

        return TransportEventData(
            event_type=TransportEvent.CONNECTED,
            address=address
        )

    def _send_packet(self, packet: Packet, address: Tuple[str, int]) -> bool:
        """Send a packet to an address."""
        if not self._socket:
            return False

        data = packet.to_bytes()

        # Check rate limit before sending (no increment yet)
        if self._packets_this_second >= self._config.max_packets_per_second:
            return False
        if self._bytes_this_second + len(data) > self._config.max_bytes_per_second:
            return False

        try:
            self._socket.sendto(data, address)
            # Increment counters AFTER successful send
            self._packets_this_second += 1
            self._bytes_this_second += len(data)
            self._stats.packets_sent += 1
            self._stats.bytes_sent += len(data)
            return True
        except socket.error as e:
            logger.warning("Failed to send packet to %s: %s", address, e)
            self._stats.socket_errors += 1
            return False

    def _disconnect_connection(self, conn: Connection, reason: str) -> None:
        """Disconnect a connection."""
        packets = conn.disconnect(reason)
        for packet in packets:
            self._send_packet(packet, conn.address)

    def _cleanup_connection(self, address: Tuple[str, int]) -> None:
        """Clean up a disconnected connection."""
        if address in self._connections:
            del self._connections[address]
            self._stats.connections_active -= 1

    def _handle_connected(self, conn: Connection) -> None:
        """Handle connection established."""
        if self._on_connect:
            self._on_connect(conn)

    def _handle_disconnected(self, conn: Connection, reason: str) -> None:
        """Handle connection lost."""
        if self._on_disconnect:
            self._on_disconnect(conn, reason)

    def get_connection(self, address: Tuple[str, int]) -> Optional[Connection]:
        """Get a connection by address."""
        return self._connections.get(address)

    def get_connections(self) -> List[Connection]:
        """Get all active connections."""
        return list(self._connections.values())

    def set_on_connect(self, callback: Callable[[Connection], None]) -> None:
        """Set callback for new connections."""
        self._on_connect = callback

    def set_on_disconnect(self, callback: Callable[[Connection, str], None]) -> None:
        """Set callback for disconnections."""
        self._on_disconnect = callback

    def set_on_data(self, callback: Callable[[Connection, bytes], None]) -> None:
        """Set callback for received data."""
        self._on_data = callback
