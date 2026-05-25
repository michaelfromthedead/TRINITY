"""
Network channels with different reliability guarantees.

Provides unreliable, reliable, ordered, and sequenced channels
for various network data transmission requirements.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Deque, Dict, List, Optional, Set, Tuple

from .packet import (
    Packet,
    PacketType,
    PacketFlags,
    PacketFragmenter,
    sequence_greater_than,
    sequence_difference,
)
from ..config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ChannelType(IntEnum):
    """Types of network channels."""
    UNRELIABLE = 0
    RELIABLE_UNORDERED = 1
    RELIABLE_ORDERED = 2
    SEQUENCED = 3


@dataclass
class PendingPacket:
    """A packet pending acknowledgment."""
    packet: Packet
    send_time: float
    retransmit_time: float
    retransmit_count: int = 0


@dataclass
class ChannelConfig:
    """Configuration for a channel."""
    channel_type: ChannelType
    max_pending: int = DEFAULT_CONFIG.CHANNEL_MAX_PENDING
    initial_rtt: float = DEFAULT_CONFIG.CHANNEL_INITIAL_RTT
    max_retries: int = DEFAULT_CONFIG.CHANNEL_MAX_RETRIES
    ack_timeout: float = DEFAULT_CONFIG.CHANNEL_ACK_TIMEOUT
    ordering_buffer_size: int = DEFAULT_CONFIG.CHANNEL_ORDERING_BUFFER_SIZE


class Channel(ABC):
    """
    Abstract base class for network channels.

    Channels provide different delivery guarantees:
    - Unreliable: Fire and forget, no guarantees
    - Reliable Unordered: Guaranteed delivery, any order
    - Reliable Ordered: Guaranteed delivery, in order
    - Sequenced: Latest only, drop old

    Attributes:
        channel_id: Unique identifier for this channel.
        channel_type: The type of channel.
    """

    def __init__(
        self,
        channel_id: int,
        config: Optional[ChannelConfig] = None
    ) -> None:
        """
        Initialize the channel.

        Args:
            channel_id: Unique identifier for this channel.
            config: Channel configuration.
        """
        self.channel_id = channel_id
        self._config = config or ChannelConfig(ChannelType.UNRELIABLE)
        self._local_sequence = 0
        self._remote_sequence = 0
        self._stats = ChannelStats()

    @property
    @abstractmethod
    def channel_type(self) -> ChannelType:
        """Get the channel type."""
        pass

    @abstractmethod
    def send(self, data: bytes) -> List[Packet]:
        """
        Queue data for sending.

        Args:
            data: The data to send.

        Returns:
            List of packets to transmit.
        """
        pass

    @abstractmethod
    def receive(self, packet: Packet) -> Optional[bytes]:
        """
        Process a received packet.

        Args:
            packet: The received packet.

        Returns:
            The data if ready to deliver, None otherwise.
        """
        pass

    @abstractmethod
    def process_ack(self, ack: int, ack_bits: int) -> List[Packet]:
        """
        Process acknowledgment information.

        Args:
            ack: Latest acknowledged sequence.
            ack_bits: Bitfield of previous 32 acks.

        Returns:
            List of packets to retransmit.
        """
        pass

    @abstractmethod
    def update(self, dt: float) -> List[Packet]:
        """
        Update the channel state.

        Args:
            dt: Time delta since last update.

        Returns:
            List of packets to send (retransmits, etc.).
        """
        pass

    def get_next_sequence(self) -> int:
        """Get and increment the local sequence number."""
        seq = self._local_sequence
        self._local_sequence = (self._local_sequence + 1) & 0xFFFF
        return seq

    @property
    def stats(self) -> 'ChannelStats':
        """Get channel statistics."""
        return self._stats


@dataclass
class ChannelStats:
    """Statistics for a channel."""
    packets_sent: int = 0
    packets_received: int = 0
    packets_lost: int = 0
    packets_retransmitted: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    average_rtt: float = 0.0
    pending_acks: int = 0


class UnreliableChannel(Channel):
    """
    Unreliable channel - fire and forget.

    No guarantees on delivery or ordering.
    Lowest latency, suitable for frequently updated data.
    """

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.UNRELIABLE

    def send(self, data: bytes) -> List[Packet]:
        """Send data without reliability."""
        seq = self.get_next_sequence()
        packet = Packet.create(PacketType.DATA, data, sequence=seq)
        self._stats.packets_sent += 1
        self._stats.bytes_sent += len(data)
        return [packet]

    def receive(self, packet: Packet) -> Optional[bytes]:
        """Receive and immediately deliver."""
        self._stats.packets_received += 1
        self._stats.bytes_received += len(packet.payload)
        self._remote_sequence = max(self._remote_sequence, packet.header.sequence)
        return packet.payload

    def process_ack(self, ack: int, ack_bits: int) -> List[Packet]:
        """No-op for unreliable channel."""
        return []

    def update(self, dt: float) -> List[Packet]:
        """No-op for unreliable channel."""
        return []


class ReliableChannel(Channel):
    """
    Reliable unordered channel.

    Guarantees delivery but not order.
    Uses ACK tracking and retransmission.
    """

    def __init__(
        self,
        channel_id: int,
        config: Optional[ChannelConfig] = None
    ) -> None:
        cfg = config or ChannelConfig(ChannelType.RELIABLE_UNORDERED)
        super().__init__(channel_id, cfg)

        self._pending: Dict[int, PendingPacket] = {}
        self._received: Set[int] = set()
        self._delivered: Set[int] = set()
        self._rtt_estimate = cfg.initial_rtt
        self._rtt_variance = 0.0
        self._fragmenter = PacketFragmenter()

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.RELIABLE_UNORDERED

    def send(self, data: bytes) -> List[Packet]:
        """Send data with reliability."""
        packets = self._fragmenter.fragment(data, self._local_sequence)

        result = []
        for packet in packets:
            seq = self.get_next_sequence()
            packet.header.sequence = seq
            packet.header.set_flag(PacketFlags.RELIABLE)

            now = time.time()
            self._pending[seq] = PendingPacket(
                packet=packet,
                send_time=now,
                retransmit_time=now + self._rtt_estimate * 1.5
            )
            result.append(packet)

        self._stats.packets_sent += len(result)
        self._stats.bytes_sent += sum(len(p.payload) for p in result)
        self._stats.pending_acks = len(self._pending)
        return result

    def receive(self, packet: Packet) -> Optional[bytes]:
        """Process received packet."""
        seq = packet.header.sequence

        # Skip duplicates
        if seq in self._received:
            return None

        self._received.add(seq)
        self._stats.packets_received += 1
        self._stats.bytes_received += len(packet.payload)

        # Update remote sequence
        if sequence_greater_than(seq, self._remote_sequence):
            self._remote_sequence = seq

        # Handle fragments
        if packet.is_fragmented():
            return self._fragmenter.add_fragment(packet)

        return packet.payload

    def process_ack(self, ack: int, ack_bits: int) -> List[Packet]:
        """Process acknowledgments."""
        acked_sequences = []

        # Process main ACK
        if ack in self._pending:
            acked_sequences.append(ack)

        # Process ACK bits (previous 32 packets)
        for i in range(32):
            if ack_bits & (1 << i):
                seq = (ack - 1 - i) & 0xFFFF
                if seq in self._pending:
                    acked_sequences.append(seq)

        # Remove acked packets and update RTT
        now = time.time()
        for seq in acked_sequences:
            pending = self._pending.pop(seq, None)
            if pending and pending.retransmit_count == 0:
                # Update RTT estimate (only for non-retransmitted)
                rtt = now - pending.send_time
                self._update_rtt(rtt)

        self._stats.pending_acks = len(self._pending)
        return []

    def update(self, dt: float) -> List[Packet]:
        """Check for retransmissions."""
        now = time.time()
        retransmits = []

        for seq, pending in list(self._pending.items()):
            if now >= pending.retransmit_time:
                if pending.retransmit_count >= self._config.max_retries:
                    # Give up
                    self._stats.packets_lost += 1
                    del self._pending[seq]
                else:
                    # Retransmit
                    pending.retransmit_count += 1
                    pending.retransmit_time = now + self._rtt_estimate * (1.5 ** pending.retransmit_count)
                    retransmits.append(pending.packet)
                    self._stats.packets_retransmitted += 1

        self._stats.pending_acks = len(self._pending)
        return retransmits

    def _update_rtt(self, sample: float) -> None:
        """Update RTT estimate using EWMA."""
        alpha = DEFAULT_CONFIG.RTT_SMOOTHING_ALPHA
        beta = DEFAULT_CONFIG.RTT_SMOOTHING_BETA

        diff = abs(sample - self._rtt_estimate)
        self._rtt_variance = (1 - beta) * self._rtt_variance + beta * diff
        self._rtt_estimate = (1 - alpha) * self._rtt_estimate + alpha * sample
        self._stats.average_rtt = self._rtt_estimate

    def get_ack_data(self) -> Tuple[int, int]:
        """Get ACK data to include in outgoing packets."""
        ack = self._remote_sequence
        ack_bits = 0

        for i in range(DEFAULT_CONFIG.ACK_BITS_COUNT):
            seq = (ack - 1 - i) & 0xFFFF
            if seq in self._received:
                ack_bits |= (1 << i)

        return ack, ack_bits


class ReliableOrderedChannel(ReliableChannel):
    """
    Reliable ordered channel.

    Guarantees delivery in order.
    Buffers out-of-order packets until gaps are filled.
    """

    def __init__(
        self,
        channel_id: int,
        config: Optional[ChannelConfig] = None
    ) -> None:
        cfg = config or ChannelConfig(ChannelType.RELIABLE_ORDERED)
        super().__init__(channel_id, cfg)

        self._order_buffer: Dict[int, bytes] = {}
        self._next_deliver_sequence = 0

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.RELIABLE_ORDERED

    def receive(self, packet: Packet) -> Optional[bytes]:
        """Process received packet with ordering."""
        seq = packet.header.sequence

        # Skip duplicates
        if seq in self._received or seq in self._delivered:
            return None

        self._received.add(seq)
        self._stats.packets_received += 1
        self._stats.bytes_received += len(packet.payload)

        # Handle fragments first
        if packet.is_fragmented():
            data = self._fragmenter.add_fragment(packet)
            if data is None:
                return None
        else:
            data = packet.payload

        # Update remote sequence for ACK purposes
        if sequence_greater_than(seq, self._remote_sequence):
            self._remote_sequence = seq

        # Store in order buffer
        self._order_buffer[seq] = data

        # Deliver in-order packets
        return self._deliver_ordered()

    def _deliver_ordered(self) -> Optional[bytes]:
        """Deliver packets in order."""
        result_parts = []

        while self._next_deliver_sequence in self._order_buffer:
            data = self._order_buffer.pop(self._next_deliver_sequence)
            self._delivered.add(self._next_deliver_sequence)
            result_parts.append(data)
            self._next_deliver_sequence = (self._next_deliver_sequence + 1) & 0xFFFF

        if result_parts:
            return b''.join(result_parts)
        return None

    def get_buffered_count(self) -> int:
        """Get number of packets waiting in order buffer."""
        return len(self._order_buffer)


class SequencedChannel(Channel):
    """
    Sequenced channel - latest only.

    Drops packets older than the last received.
    No reliability, just ordering by dropping old data.
    Good for data that becomes stale quickly (e.g., position updates).
    """

    def __init__(
        self,
        channel_id: int,
        config: Optional[ChannelConfig] = None
    ) -> None:
        cfg = config or ChannelConfig(ChannelType.SEQUENCED)
        super().__init__(channel_id, cfg)

        self._last_received_sequence = -1

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.SEQUENCED

    def send(self, data: bytes) -> List[Packet]:
        """Send sequenced data."""
        seq = self.get_next_sequence()
        packet = Packet.create(
            PacketType.SEQUENCED_DATA,
            data,
            sequence=seq,
            flags=PacketFlags.NONE
        )
        self._stats.packets_sent += 1
        self._stats.bytes_sent += len(data)
        return [packet]

    def receive(self, packet: Packet) -> Optional[bytes]:
        """Receive and deliver only if newer."""
        seq = packet.header.sequence

        # Drop if not newer
        if self._last_received_sequence >= 0:
            if not sequence_greater_than(seq, self._last_received_sequence):
                return None

        self._last_received_sequence = seq
        self._stats.packets_received += 1
        self._stats.bytes_received += len(packet.payload)

        return packet.payload

    def process_ack(self, ack: int, ack_bits: int) -> List[Packet]:
        """No-op for sequenced channel."""
        return []

    def update(self, dt: float) -> List[Packet]:
        """No-op for sequenced channel."""
        return []


class ChannelManager:
    """
    Manages multiple channels for a connection.

    Provides channel creation, routing, and aggregate statistics.
    """

    def __init__(self) -> None:
        """Initialize the channel manager."""
        self._channels: Dict[int, Channel] = {}
        self._type_channels: Dict[ChannelType, int] = {}

    def create_channel(
        self,
        channel_id: int,
        channel_type: ChannelType,
        config: Optional[ChannelConfig] = None
    ) -> Channel:
        """
        Create a new channel.

        Args:
            channel_id: Unique ID for the channel.
            channel_type: Type of channel to create.
            config: Optional channel configuration.

        Returns:
            The created channel.
        """
        cfg = config or ChannelConfig(channel_type)

        if channel_type == ChannelType.UNRELIABLE:
            channel = UnreliableChannel(channel_id, cfg)
        elif channel_type == ChannelType.RELIABLE_UNORDERED:
            channel = ReliableChannel(channel_id, cfg)
        elif channel_type == ChannelType.RELIABLE_ORDERED:
            channel = ReliableOrderedChannel(channel_id, cfg)
        elif channel_type == ChannelType.SEQUENCED:
            channel = SequencedChannel(channel_id, cfg)
        else:
            raise ValueError(f"Unknown channel type: {channel_type}")

        self._channels[channel_id] = channel
        self._type_channels[channel_type] = channel_id
        return channel

    def get_channel(self, channel_id: int) -> Optional[Channel]:
        """Get a channel by ID."""
        return self._channels.get(channel_id)

    def get_channel_by_type(self, channel_type: ChannelType) -> Optional[Channel]:
        """Get a channel by type."""
        channel_id = self._type_channels.get(channel_type)
        if channel_id is not None:
            return self._channels.get(channel_id)
        return None

    def remove_channel(self, channel_id: int) -> None:
        """Remove a channel."""
        channel = self._channels.pop(channel_id, None)
        if channel:
            # Remove type mapping
            for ctype, cid in list(self._type_channels.items()):
                if cid == channel_id:
                    del self._type_channels[ctype]
                    break

    def update(self, dt: float) -> List[Packet]:
        """Update all channels."""
        packets = []
        for channel in self._channels.values():
            packets.extend(channel.update(dt))
        return packets

    def get_aggregate_stats(self) -> ChannelStats:
        """Get combined statistics from all channels."""
        stats = ChannelStats()
        for channel in self._channels.values():
            cs = channel.stats
            stats.packets_sent += cs.packets_sent
            stats.packets_received += cs.packets_received
            stats.packets_lost += cs.packets_lost
            stats.packets_retransmitted += cs.packets_retransmitted
            stats.bytes_sent += cs.bytes_sent
            stats.bytes_received += cs.bytes_received
            stats.pending_acks += cs.pending_acks

        if self._channels:
            stats.average_rtt = sum(c.stats.average_rtt for c in self._channels.values()) / len(self._channels)

        return stats
