"""
Transport layer for network communication.

Provides packet management, channels with different reliability guarantees,
connection handling, and UDP transport implementation.
"""

from .packet import (
    Packet,
    PacketHeader,
    PacketType,
    MTU,
    MAX_PAYLOAD_SIZE,
)
from .channel import (
    Channel,
    ChannelType,
    ReliableChannel,
    UnreliableChannel,
    SequencedChannel,
    ReliableOrderedChannel,
)
from .connection import (
    Connection,
    ConnectionState,
    ConnectionConfig,
    ConnectionStats,
)
from .udp_transport import (
    UDPTransport,
    TransportConfig,
    TransportStats,
)
from .quality import (
    QualityLevel,
    QualityMetrics,
    QualityMonitor,
    NetworkQualityAdapter,
)

__all__ = [
    # Packet
    "Packet",
    "PacketHeader",
    "PacketType",
    "MTU",
    "MAX_PAYLOAD_SIZE",
    # Channels
    "Channel",
    "ChannelType",
    "ReliableChannel",
    "UnreliableChannel",
    "SequencedChannel",
    "ReliableOrderedChannel",
    # Connection
    "Connection",
    "ConnectionState",
    "ConnectionConfig",
    "ConnectionStats",
    # Transport
    "UDPTransport",
    "TransportConfig",
    "TransportStats",
    # Quality
    "QualityLevel",
    "QualityMetrics",
    "QualityMonitor",
    "NetworkQualityAdapter",
]
