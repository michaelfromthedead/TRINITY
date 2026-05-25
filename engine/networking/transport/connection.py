"""
Network connection management.

Provides connection state machine, quality tracking,
and packet send/receive handling.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple, Any

from .packet import Packet, PacketType, PacketFlags, PacketHeader
from .channel import (
    Channel,
    ChannelManager,
    ChannelType,
    ChannelConfig,
    ReliableChannel,
)
from ..config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ConnectionState(IntEnum):
    """Connection state machine states."""
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    DISCONNECTING = 3
    FAILED = 4


@dataclass
class ConnectionConfig:
    """Configuration for a connection."""
    # Timeouts
    connect_timeout: float = DEFAULT_CONFIG.CONNECT_TIMEOUT
    disconnect_timeout: float = DEFAULT_CONFIG.DISCONNECT_TIMEOUT
    idle_timeout: float = DEFAULT_CONFIG.IDLE_TIMEOUT

    # Heartbeat
    heartbeat_interval: float = DEFAULT_CONFIG.HEARTBEAT_INTERVAL
    heartbeat_timeout: float = DEFAULT_CONFIG.HEARTBEAT_TIMEOUT

    # Reliability
    max_pending_packets: int = DEFAULT_CONFIG.CONNECTION_MAX_PENDING
    max_retries: int = DEFAULT_CONFIG.CHANNEL_MAX_RETRIES

    # Channels
    default_channels: bool = True


@dataclass
class ConnectionStats:
    """Statistics for a connection."""
    # Packet stats
    packets_sent: int = 0
    packets_received: int = 0
    packets_lost: int = 0
    packets_retransmitted: int = 0

    # Byte stats
    bytes_sent: int = 0
    bytes_received: int = 0

    # Quality metrics
    rtt: float = 0.0
    rtt_variance: float = 0.0
    jitter: float = 0.0
    packet_loss: float = 0.0

    # Bandwidth
    send_bandwidth: float = 0.0
    receive_bandwidth: float = 0.0

    # Timing
    connected_time: float = 0.0
    last_packet_time: float = 0.0
    last_heartbeat_time: float = 0.0


class Connection:
    """
    Represents a network connection to a remote endpoint.

    Manages connection state, channels, reliability, and quality metrics.

    Example:
        config = ConnectionConfig()
        conn = Connection(address=("127.0.0.1", 12345), config=config)

        # Connect
        conn.connect()

        # Send data
        packets = conn.send(data, ChannelType.RELIABLE_ORDERED)

        # Receive
        for packet in received_packets:
            data = conn.receive(packet)
            if data:
                process(data)

        # Update
        conn.update(delta_time)
    """

    def __init__(
        self,
        address: Tuple[str, int],
        config: Optional[ConnectionConfig] = None,
        connection_id: Optional[int] = None
    ) -> None:
        """
        Initialize the connection.

        Args:
            address: Remote (host, port) tuple.
            config: Connection configuration.
            connection_id: Optional unique identifier.
        """
        self.address = address
        self.connection_id = connection_id or id(self)
        self._config = config or ConnectionConfig()

        # State
        self._state = ConnectionState.DISCONNECTED
        self._state_time = 0.0

        # Channels
        self._channel_manager = ChannelManager()
        if self._config.default_channels:
            self._setup_default_channels()

        # Statistics
        self._stats = ConnectionStats()

        # Timing
        self._last_send_time = 0.0
        self._last_receive_time = 0.0
        self._last_heartbeat_sent = 0.0
        self._last_heartbeat_received = 0.0
        self._connect_start_time = 0.0

        # Sequence tracking for ACKs
        self._local_sequence = 0
        self._remote_sequence = 0
        self._received_sequences: set = set()

        # RTT tracking
        self._rtt_samples: List[float] = []
        self._max_rtt_samples = DEFAULT_CONFIG.MAX_RTT_SAMPLES

        # Callbacks
        self._on_connected: Optional[Callable[['Connection'], None]] = None
        self._on_disconnected: Optional[Callable[['Connection', str], None]] = None

    def _setup_default_channels(self) -> None:
        """Create default channels."""
        # Unreliable for frequent updates
        self._channel_manager.create_channel(0, ChannelType.UNRELIABLE)
        # Reliable ordered for important game events
        self._channel_manager.create_channel(1, ChannelType.RELIABLE_ORDERED)
        # Reliable unordered for less critical reliable data
        self._channel_manager.create_channel(2, ChannelType.RELIABLE_UNORDERED)
        # Sequenced for state that only needs latest
        self._channel_manager.create_channel(3, ChannelType.SEQUENCED)

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def stats(self) -> ConnectionStats:
        """Get connection statistics."""
        return self._stats

    @property
    def rtt(self) -> float:
        """Get current RTT estimate."""
        return self._stats.rtt

    @property
    def jitter(self) -> float:
        """Get current jitter estimate."""
        return self._stats.jitter

    @property
    def packet_loss(self) -> float:
        """Get packet loss ratio (0-1)."""
        return self._stats.packet_loss

    def connect(self) -> List[Packet]:
        """
        Initiate connection.

        Returns:
            List of packets to send.
        """
        if self._state != ConnectionState.DISCONNECTED:
            return []

        self._state = ConnectionState.CONNECTING
        self._connect_start_time = time.time()
        self._state_time = 0.0

        # Create connect packet
        packet = Packet.create(
            PacketType.CONNECT,
            sequence=self._get_next_sequence(),
            flags=PacketFlags.RELIABLE
        )

        return [packet]

    def disconnect(self, reason: str = "User requested") -> List[Packet]:
        """
        Initiate disconnection.

        Args:
            reason: Disconnect reason.

        Returns:
            List of packets to send.
        """
        if self._state in (ConnectionState.DISCONNECTED, ConnectionState.FAILED):
            return []

        self._state = ConnectionState.DISCONNECTING
        self._state_time = 0.0

        # Create disconnect packet
        packet = Packet.create(
            PacketType.DISCONNECT,
            payload=reason.encode('utf-8'),
            sequence=self._get_next_sequence()
        )

        return [packet]

    def send(
        self,
        data: bytes,
        channel_type: ChannelType = ChannelType.UNRELIABLE
    ) -> List[Packet]:
        """
        Send data over a channel.

        Args:
            data: The data to send.
            channel_type: Which channel to use.

        Returns:
            List of packets to transmit.
        """
        if not self.is_connected:
            return []

        channel = self._channel_manager.get_channel_by_type(channel_type)
        if not channel:
            # Fall back to unreliable
            channel = self._channel_manager.get_channel(0)

        if not channel:
            return []

        packets = channel.send(data)

        # Add ACK information to packets
        for packet in packets:
            self._add_ack_info(packet)

        self._last_send_time = time.time()
        self._stats.packets_sent += len(packets)
        self._stats.bytes_sent += sum(len(p.payload) for p in packets)

        return packets

    def receive(self, packet: Packet) -> Optional[bytes]:
        """
        Process a received packet.

        Args:
            packet: The received packet.

        Returns:
            Data if ready to deliver, None otherwise.
        """
        if self._state == ConnectionState.DISCONNECTED:
            # Only block data-family packets when disconnected.
            # Allow protocol packets (CONNECT, CONNECT_ACK, DISCONNECT,
            # DISCONNECT_ACK, HEARTBEAT, HEARTBEAT_ACK) to pass through
            # so that connection state machine can transition properly.
            ptype = packet.header.packet_type
            if ptype in (
                PacketType.DATA,
                PacketType.RELIABLE_DATA,
                PacketType.SEQUENCED_DATA,
                PacketType.FRAGMENT,
            ):
                return None

        now = time.time()
        self._last_receive_time = now
        self._stats.packets_received += 1
        self._stats.bytes_received += len(packet.payload)

        # Track sequence for ACKs
        seq = packet.header.sequence
        self._received_sequences.add(seq)
        if len(self._received_sequences) > DEFAULT_CONFIG.MAX_RECEIVED_SEQUENCES:
            # Keep only recent
            min_seq = seq - (DEFAULT_CONFIG.MAX_RECEIVED_SEQUENCES // 2)
            self._received_sequences = {s for s in self._received_sequences if s > min_seq}

        if seq > self._remote_sequence:
            self._remote_sequence = seq

        # Process ACK info
        self._process_ack_info(packet)

        # Handle packet by type
        ptype = packet.header.packet_type

        if ptype == PacketType.CONNECT:
            return self._handle_connect(packet)
        elif ptype == PacketType.CONNECT_ACK:
            return self._handle_connect_ack(packet)
        elif ptype == PacketType.DISCONNECT:
            return self._handle_disconnect(packet)
        elif ptype == PacketType.HEARTBEAT:
            return self._handle_heartbeat(packet)
        elif ptype == PacketType.HEARTBEAT_ACK:
            return self._handle_heartbeat_ack(packet)
        elif ptype in (PacketType.DATA, PacketType.RELIABLE_DATA, PacketType.SEQUENCED_DATA, PacketType.FRAGMENT):
            return self._handle_data(packet)
        elif ptype == PacketType.ACK:
            # ACK-only packet, already processed
            return None

        return None

    def _handle_connect(self, packet: Packet) -> Optional[bytes]:
        """Handle incoming connect request."""
        if self._state == ConnectionState.DISCONNECTED:
            self._state = ConnectionState.CONNECTED
            self._stats.connected_time = time.time()
            if self._on_connected:
                self._on_connected(self)
        return None

    def _handle_connect_ack(self, packet: Packet) -> Optional[bytes]:
        """Handle connect acknowledgment."""
        if self._state == ConnectionState.CONNECTING:
            self._state = ConnectionState.CONNECTED
            self._stats.connected_time = time.time()
            if self._on_connected:
                self._on_connected(self)
        return None

    def _handle_disconnect(self, packet: Packet) -> Optional[bytes]:
        """Handle disconnect request."""
        reason = packet.payload.decode('utf-8') if packet.payload else "Remote disconnect"
        self._state = ConnectionState.DISCONNECTED
        if self._on_disconnected:
            self._on_disconnected(self, reason)
        return None

    def _handle_heartbeat(self, packet: Packet) -> Optional[bytes]:
        """Handle heartbeat request."""
        self._last_heartbeat_received = time.time()
        return None

    def _handle_heartbeat_ack(self, packet: Packet) -> Optional[bytes]:
        """Handle heartbeat acknowledgment."""
        self._last_heartbeat_received = time.time()
        return None

    def _handle_data(self, packet: Packet) -> Optional[bytes]:
        """Handle data packet."""
        # Determine channel from packet flags
        if packet.header.has_flag(PacketFlags.RELIABLE):
            if packet.header.has_flag(PacketFlags.ORDERED):
                channel = self._channel_manager.get_channel_by_type(ChannelType.RELIABLE_ORDERED)
            else:
                channel = self._channel_manager.get_channel_by_type(ChannelType.RELIABLE_UNORDERED)
        elif packet.header.packet_type == PacketType.SEQUENCED_DATA:
            channel = self._channel_manager.get_channel_by_type(ChannelType.SEQUENCED)
        else:
            channel = self._channel_manager.get_channel_by_type(ChannelType.UNRELIABLE)

        if channel:
            return channel.receive(packet)

        return packet.payload

    def update(self, dt: float) -> List[Packet]:
        """
        Update connection state.

        Args:
            dt: Time delta since last update.

        Returns:
            List of packets to send.
        """
        self._state_time += dt
        packets = []
        now = time.time()

        # State-specific updates
        if self._state == ConnectionState.CONNECTING:
            if self._state_time > self._config.connect_timeout:
                self._state = ConnectionState.FAILED
                if self._on_disconnected:
                    self._on_disconnected(self, "Connect timeout")

        elif self._state == ConnectionState.CONNECTED:
            # Check for timeout
            if now - self._last_receive_time > self._config.idle_timeout:
                self._state = ConnectionState.FAILED
                if self._on_disconnected:
                    self._on_disconnected(self, "Idle timeout")
            else:
                # Send heartbeat if needed
                if now - self._last_heartbeat_sent > self._config.heartbeat_interval:
                    packets.append(self._create_heartbeat())
                    self._last_heartbeat_sent = now

        elif self._state == ConnectionState.DISCONNECTING:
            if self._state_time > self._config.disconnect_timeout:
                self._state = ConnectionState.DISCONNECTED

        # Update channels (retransmissions)
        packets.extend(self._channel_manager.update(dt))

        # Update statistics
        self._update_stats(dt)

        return packets

    def _create_heartbeat(self) -> Packet:
        """Create a heartbeat packet."""
        packet = Packet.create_heartbeat(self._get_next_sequence())
        self._add_ack_info(packet)
        return packet

    def _get_next_sequence(self) -> int:
        """Get next sequence number."""
        seq = self._local_sequence
        self._local_sequence = (self._local_sequence + 1) & 0xFFFF
        return seq

    def _add_ack_info(self, packet: Packet) -> None:
        """Add ACK information to outgoing packet."""
        packet.header.ack = self._remote_sequence

        # Build ACK bits
        ack_bits = 0
        for i in range(DEFAULT_CONFIG.ACK_BITS_COUNT):
            seq = (self._remote_sequence - 1 - i) & 0xFFFF
            if seq in self._received_sequences:
                ack_bits |= (1 << i)

        packet.header.ack_bits = ack_bits

    def _process_ack_info(self, packet: Packet) -> None:
        """Process ACK information from received packet."""
        ack = packet.header.ack
        ack_bits = packet.header.ack_bits

        # Notify channels
        for channel in [
            self._channel_manager.get_channel(1),
            self._channel_manager.get_channel(2),
        ]:
            if channel and isinstance(channel, ReliableChannel):
                channel.process_ack(ack, ack_bits)

    def _update_stats(self, dt: float) -> None:
        """Update connection statistics."""
        # Get channel stats
        channel_stats = self._channel_manager.get_aggregate_stats()

        self._stats.packets_sent = channel_stats.packets_sent
        self._stats.packets_received = channel_stats.packets_received
        self._stats.packets_lost = channel_stats.packets_lost
        self._stats.packets_retransmitted = channel_stats.packets_retransmitted
        self._stats.rtt = channel_stats.average_rtt

        # Calculate packet loss ratio
        total = self._stats.packets_sent
        if total > 0:
            self._stats.packet_loss = self._stats.packets_lost / total

    def get_channel(self, channel_type: ChannelType) -> Optional[Channel]:
        """Get a specific channel."""
        return self._channel_manager.get_channel_by_type(channel_type)

    def create_channel(
        self,
        channel_id: int,
        channel_type: ChannelType,
        config: Optional[ChannelConfig] = None
    ) -> Channel:
        """Create a custom channel."""
        return self._channel_manager.create_channel(channel_id, channel_type, config)

    def set_on_connected(self, callback: Callable[['Connection'], None]) -> None:
        """Set callback for when connected."""
        self._on_connected = callback

    def set_on_disconnected(self, callback: Callable[['Connection', str], None]) -> None:
        """Set callback for when disconnected."""
        self._on_disconnected = callback

    def get_pending_ack_count(self) -> int:
        """Get total number of packets awaiting ACK."""
        stats = self._channel_manager.get_aggregate_stats()
        return stats.pending_acks
