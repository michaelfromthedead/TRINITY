"""
Network packet definitions and handling.

Provides packet structure, header format, and packet types
for reliable and unreliable network transmission.
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

from ..config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Maximum Transmission Unit - safe UDP size for Internet
MTU = DEFAULT_CONFIG.MTU

# Header size in bytes
HEADER_SIZE = DEFAULT_CONFIG.PACKET_HEADER_SIZE

# Maximum payload per packet
MAX_PAYLOAD_SIZE = DEFAULT_CONFIG.MAX_PAYLOAD_SIZE


class PacketType(IntEnum):
    """Types of network packets."""
    # Core packet types
    DATA = 0
    ACK = 1
    NACK = 2

    # Connection management
    CONNECT = 10
    CONNECT_ACK = 11
    DISCONNECT = 12
    DISCONNECT_ACK = 13

    # Keep-alive
    HEARTBEAT = 20
    HEARTBEAT_ACK = 21

    # Fragmentation
    FRAGMENT = 30
    FRAGMENT_ACK = 31

    # Reliability
    RELIABLE_DATA = 40
    SEQUENCED_DATA = 41


class PacketFlags(IntEnum):
    """Packet flags for additional metadata."""
    NONE = 0x00
    COMPRESSED = 0x01
    ENCRYPTED = 0x02
    FRAGMENTED = 0x04
    RELIABLE = 0x08
    ORDERED = 0x10
    PRIORITY_HIGH = 0x20
    PRIORITY_LOW = 0x40


@dataclass
class PacketHeader:
    """
    Header for network packets.

    Format (12 bytes):
        - packet_type: 1 byte
        - flags: 1 byte
        - sequence: 2 bytes (0-65535)
        - ack: 2 bytes (last received sequence)
        - ack_bits: 4 bytes (bitfield for 32 previous acks)
        - size: 2 bytes (payload size)

    Attributes:
        packet_type: Type of the packet.
        flags: Packet flags.
        sequence: Packet sequence number.
        ack: Last received sequence from remote.
        ack_bits: Bitfield acknowledging 32 previous packets.
        size: Payload size in bytes.
    """
    packet_type: PacketType
    flags: int = 0
    sequence: int = 0
    ack: int = 0
    ack_bits: int = 0
    size: int = 0

    # Format string for struct packing (big-endian)
    _FORMAT = '!BBHHIH'

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        return struct.pack(
            self._FORMAT,
            self.packet_type,
            self.flags,
            self.sequence & 0xFFFF,
            self.ack & 0xFFFF,
            self.ack_bits & 0xFFFFFFFF,
            self.size & 0xFFFF
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'PacketHeader':
        """Deserialize header from bytes."""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Need {HEADER_SIZE} bytes for header, got {len(data)}")

        ptype, flags, seq, ack, ack_bits, size = struct.unpack(cls._FORMAT, data[:HEADER_SIZE])

        return cls(
            packet_type=PacketType(ptype) if ptype in PacketType.__members__.values() else PacketType.DATA,
            flags=flags,
            sequence=seq,
            ack=ack,
            ack_bits=ack_bits,
            size=size
        )

    def has_flag(self, flag: PacketFlags) -> bool:
        """Check if a flag is set."""
        return (self.flags & flag) != 0

    def set_flag(self, flag: PacketFlags) -> None:
        """Set a flag."""
        self.flags |= flag

    def clear_flag(self, flag: PacketFlags) -> None:
        """Clear a flag."""
        self.flags &= ~flag


@dataclass
class Packet:
    """
    Network packet with header and payload.

    Attributes:
        header: The packet header.
        payload: The packet payload bytes.
        timestamp: When the packet was created.
        retransmit_count: Number of times retransmitted.
    """
    header: PacketHeader
    payload: bytes = b''
    timestamp: float = field(default_factory=time.time)
    retransmit_count: int = 0

    @classmethod
    def create(
        cls,
        packet_type: PacketType,
        payload: bytes = b'',
        sequence: int = 0,
        flags: int = 0
    ) -> 'Packet':
        """
        Create a new packet.

        Args:
            packet_type: Type of the packet.
            payload: Payload bytes.
            sequence: Sequence number.
            flags: Packet flags.

        Returns:
            The created packet.
        """
        header = PacketHeader(
            packet_type=packet_type,
            flags=flags,
            sequence=sequence,
            size=len(payload)
        )
        return cls(header=header, payload=payload)

    @classmethod
    def create_ack(cls, ack_sequence: int, ack_bits: int = 0) -> 'Packet':
        """
        Create an ACK packet.

        Args:
            ack_sequence: Sequence number being acknowledged.
            ack_bits: Bitfield for previous 32 acks.

        Returns:
            The ACK packet.
        """
        header = PacketHeader(
            packet_type=PacketType.ACK,
            ack=ack_sequence,
            ack_bits=ack_bits
        )
        return cls(header=header)

    @classmethod
    def create_heartbeat(cls, sequence: int = 0) -> 'Packet':
        """Create a heartbeat packet."""
        header = PacketHeader(
            packet_type=PacketType.HEARTBEAT,
            sequence=sequence
        )
        return cls(header=header)

    def to_bytes(self) -> bytes:
        """Serialize the entire packet to bytes."""
        return self.header.to_bytes() + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Packet':
        """Deserialize bytes to a packet."""
        header = PacketHeader.from_bytes(data)
        payload = data[HEADER_SIZE:HEADER_SIZE + header.size]
        return cls(header=header, payload=payload)

    @property
    def total_size(self) -> int:
        """Get total packet size including header."""
        return HEADER_SIZE + len(self.payload)

    def is_reliable(self) -> bool:
        """Check if packet requires acknowledgment."""
        return self.header.has_flag(PacketFlags.RELIABLE)

    def is_fragmented(self) -> bool:
        """Check if packet is a fragment."""
        return self.header.has_flag(PacketFlags.FRAGMENTED)


@dataclass
class FragmentHeader:
    """
    Header for fragmented packets.

    Added after the main packet header for fragments.

    Attributes:
        fragment_id: Unique ID for this group of fragments.
        fragment_index: Index of this fragment (0-based).
        fragment_total: Total number of fragments.
    """
    fragment_id: int
    fragment_index: int
    fragment_total: int

    _FORMAT = '!HBB'
    SIZE = DEFAULT_CONFIG.FRAGMENT_HEADER_SIZE

    def to_bytes(self) -> bytes:
        """Serialize fragment header to bytes."""
        return struct.pack(
            self._FORMAT,
            self.fragment_id & 0xFFFF,
            self.fragment_index & 0xFF,
            self.fragment_total & 0xFF
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'FragmentHeader':
        """Deserialize fragment header from bytes."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Need {cls.SIZE} bytes for fragment header")

        frag_id, frag_idx, frag_total = struct.unpack(cls._FORMAT, data[:cls.SIZE])
        return cls(
            fragment_id=frag_id,
            fragment_index=frag_idx,
            fragment_total=frag_total
        )


class PacketFragmenter:
    """
    Handles packet fragmentation and reassembly.

    Splits large payloads into MTU-sized fragments and
    reassembles them on the receiving end.
    """

    # Size available for fragment data
    FRAGMENT_PAYLOAD_SIZE = MAX_PAYLOAD_SIZE - FragmentHeader.SIZE

    def __init__(self) -> None:
        """Initialize the fragmenter."""
        self._fragment_id = 0
        self._pending_fragments: dict[int, dict[int, bytes]] = {}
        self._fragment_totals: dict[int, int] = {}

    def fragment(self, payload: bytes, sequence: int = 0) -> List[Packet]:
        """
        Fragment a large payload into multiple packets.

        Args:
            payload: The payload to fragment.
            sequence: Base sequence number.

        Returns:
            List of fragment packets.
        """
        if len(payload) <= MAX_PAYLOAD_SIZE:
            # No fragmentation needed
            return [Packet.create(PacketType.DATA, payload, sequence)]

        # Generate fragment ID
        self._fragment_id = (self._fragment_id + 1) & 0xFFFF
        frag_id = self._fragment_id

        fragments = []
        total_fragments = (len(payload) + self.FRAGMENT_PAYLOAD_SIZE - 1) // self.FRAGMENT_PAYLOAD_SIZE

        for i in range(total_fragments):
            start = i * self.FRAGMENT_PAYLOAD_SIZE
            end = min(start + self.FRAGMENT_PAYLOAD_SIZE, len(payload))
            chunk = payload[start:end]

            # Create fragment header
            frag_header = FragmentHeader(
                fragment_id=frag_id,
                fragment_index=i,
                fragment_total=total_fragments
            )

            # Create packet with fragment header + data
            packet_payload = frag_header.to_bytes() + chunk
            packet = Packet.create(
                PacketType.FRAGMENT,
                packet_payload,
                sequence=sequence + i,
                flags=PacketFlags.FRAGMENTED | PacketFlags.RELIABLE
            )
            fragments.append(packet)

        return fragments

    def add_fragment(self, packet: Packet) -> Optional[bytes]:
        """
        Add a received fragment and attempt reassembly.

        Args:
            packet: The fragment packet.

        Returns:
            Complete payload if all fragments received, None otherwise.
        """
        if packet.header.packet_type != PacketType.FRAGMENT:
            return packet.payload

        # Parse fragment header
        frag_header = FragmentHeader.from_bytes(packet.payload)
        data = packet.payload[FragmentHeader.SIZE:]

        frag_id = frag_header.fragment_id

        # Store fragment
        if frag_id not in self._pending_fragments:
            self._pending_fragments[frag_id] = {}
            self._fragment_totals[frag_id] = frag_header.fragment_total

        self._pending_fragments[frag_id][frag_header.fragment_index] = data

        # Check if complete
        if len(self._pending_fragments[frag_id]) == self._fragment_totals[frag_id]:
            # Reassemble
            result = b''
            for i in range(self._fragment_totals[frag_id]):
                result += self._pending_fragments[frag_id][i]

            # Clean up
            del self._pending_fragments[frag_id]
            del self._fragment_totals[frag_id]

            return result

        return None

    def clear_pending(self, fragment_id: Optional[int] = None) -> None:
        """
        Clear pending fragments.

        Args:
            fragment_id: Specific fragment group to clear, or None for all.
        """
        if fragment_id is not None:
            self._pending_fragments.pop(fragment_id, None)
            self._fragment_totals.pop(fragment_id, None)
        else:
            self._pending_fragments.clear()
            self._fragment_totals.clear()


def sequence_greater_than(s1: int, s2: int, max_value: int = DEFAULT_CONFIG.MAX_SEQUENCE) -> bool:
    """
    Compare sequence numbers with wraparound handling.

    Args:
        s1: First sequence number.
        s2: Second sequence number.
        max_value: Maximum sequence value.

    Returns:
        True if s1 is greater than s2 (accounting for wraparound).
    """
    half = max_value // 2
    return ((s1 > s2) and (s1 - s2 <= half)) or ((s1 < s2) and (s2 - s1 > half))


def sequence_difference(s1: int, s2: int, max_value: int = DEFAULT_CONFIG.MAX_SEQUENCE) -> int:
    """
    Calculate difference between sequence numbers.

    Args:
        s1: First sequence number.
        s2: Second sequence number.
        max_value: Maximum sequence value.

    Returns:
        Signed difference (positive if s1 > s2).
    """
    half = max_value // 2
    diff = s1 - s2

    if diff > half:
        diff -= (max_value + 1)
    elif diff < -half:
        diff += (max_value + 1)

    return diff
