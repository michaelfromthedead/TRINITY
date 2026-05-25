"""Network Profiler for game engine network performance analysis.

Provides bandwidth tracking, latency measurement, and packet loss detection.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

from engine.debug.profiling import config as profiling_config


class PacketType(Enum):
    """Types of network packets."""

    RELIABLE = auto()
    UNRELIABLE = auto()
    ORDERED = auto()
    SEQUENCED = auto()


class PacketDirection(Enum):
    """Direction of network packets."""

    SENT = auto()
    RECEIVED = auto()


@dataclass
class NetworkStats:
    """Network statistics for a time period."""

    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    rtt_ms: float = 0.0
    loss_percent: float = 0.0
    jitter_ms: float = 0.0

    @property
    def total_bytes(self) -> int:
        """Total bytes transferred."""
        return self.bytes_sent + self.bytes_received

    @property
    def total_packets(self) -> int:
        """Total packets transferred."""
        return self.packets_sent + self.packets_received

    @property
    def bandwidth_sent_kbps(self) -> float:
        """Sent bandwidth in kilobits per second."""
        return (self.bytes_sent * 8) / 1000

    @property
    def bandwidth_received_kbps(self) -> float:
        """Received bandwidth in kilobits per second."""
        return (self.bytes_received * 8) / 1000


@dataclass
class PacketRecord:
    """Record of a single packet."""

    packet_id: int
    size: int
    direction: PacketDirection
    packet_type: PacketType
    timestamp: float
    acknowledged: bool = False
    ack_timestamp: Optional[float] = None
    channel: int = 0

    @property
    def rtt_ms(self) -> Optional[float]:
        """Round-trip time if acknowledged."""
        if self.ack_timestamp is not None:
            return (self.ack_timestamp - self.timestamp) * 1000
        return None


@dataclass
class ConnectionStats:
    """Statistics for a specific connection."""

    connection_id: str
    remote_address: str
    connected_since: float
    last_activity: float
    stats: NetworkStats = field(default_factory=NetworkStats)
    rtt_samples: List[float] = field(default_factory=list)

    @property
    def connection_duration(self) -> float:
        """How long this connection has been active."""
        return time.time() - self.connected_since

    @property
    def average_rtt_ms(self) -> float:
        """Average RTT in milliseconds."""
        if not self.rtt_samples:
            return 0.0
        return sum(self.rtt_samples) / len(self.rtt_samples)


class NetworkProfiler:
    """Network profiler for tracking bandwidth and latency.

    Tracks packets sent and received, measures round-trip time,
    and detects packet loss.

    Example:
        profiler = NetworkProfiler()

        # Track outgoing packet
        packet_id = profiler.track_packet_sent(256, PacketType.RELIABLE)

        # Track acknowledgment
        profiler.track_packet_ack(packet_id)

        # Track incoming packet
        profiler.track_packet_received(128, PacketType.UNRELIABLE)

        # Get statistics
        stats = profiler.get_stats()
        print(f"RTT: {stats.rtt_ms}ms, Loss: {stats.loss_percent}%")
    """

    def __init__(
        self,
        enabled: bool = True,
        window_seconds: Optional[float] = None,
        history_size: Optional[int] = None
    ) -> None:
        """Initialize the network profiler.

        Args:
            enabled: Whether profiling is active.
            window_seconds: Time window for calculating statistics.
                           Defaults to profiler.network.StatsWindowSeconds CVar.
            history_size: Number of packets to keep in history.
                         Defaults to profiler.network.PacketHistorySize CVar.
        """
        self._enabled = enabled
        self._window_seconds = (
            window_seconds if window_seconds is not None
            else profiling_config.network_stats_window_seconds.value
        )
        self._history_size = (
            history_size if history_size is not None
            else profiling_config.network_packet_history_size.value
        )
        self._lock = threading.Lock()

        self._next_packet_id: int = 1
        self._sent_packets: Dict[int, PacketRecord] = {}
        self._packet_history: Deque[PacketRecord] = deque(maxlen=self._history_size)
        rtt_sample_size = profiling_config.network_rtt_sample_size.value
        self._rtt_samples: Deque[float] = deque(maxlen=rtt_sample_size)

        # Per-window statistics
        self._window_start: float = time.time()
        self._window_bytes_sent: int = 0
        self._window_bytes_received: int = 0
        self._window_packets_sent: int = 0
        self._window_packets_received: int = 0
        self._window_packets_lost: int = 0

        # Connection tracking
        self._connections: Dict[str, ConnectionStats] = {}

        # Accumulated statistics
        self._total_bytes_sent: int = 0
        self._total_bytes_received: int = 0
        self._total_packets_sent: int = 0
        self._total_packets_received: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def track_packet_sent(
        self,
        size: int,
        packet_type: PacketType = PacketType.RELIABLE,
        channel: int = 0,
        connection_id: Optional[str] = None
    ) -> int:
        """Track an outgoing packet.

        Args:
            size: Size of the packet in bytes.
            packet_type: Type of packet.
            channel: Channel/stream number.
            connection_id: Optional connection identifier.

        Returns:
            Unique packet ID for tracking acknowledgment.
        """
        if not self._enabled:
            return 0

        with self._lock:
            packet_id = self._next_packet_id
            self._next_packet_id += 1

            record = PacketRecord(
                packet_id=packet_id,
                size=size,
                direction=PacketDirection.SENT,
                packet_type=packet_type,
                timestamp=time.time(),
                channel=channel
            )

            self._sent_packets[packet_id] = record
            self._packet_history.append(record)

            self._window_bytes_sent += size
            self._window_packets_sent += 1
            self._total_bytes_sent += size
            self._total_packets_sent += 1

            # Update connection stats
            if connection_id and connection_id in self._connections:
                conn = self._connections[connection_id]
                conn.stats.bytes_sent += size
                conn.stats.packets_sent += 1
                conn.last_activity = time.time()

            self._check_window()

            return packet_id

    def track_packet_received(
        self,
        size: int,
        packet_type: PacketType = PacketType.RELIABLE,
        channel: int = 0,
        connection_id: Optional[str] = None
    ) -> None:
        """Track an incoming packet.

        Args:
            size: Size of the packet in bytes.
            packet_type: Type of packet.
            channel: Channel/stream number.
            connection_id: Optional connection identifier.
        """
        if not self._enabled:
            return

        with self._lock:
            record = PacketRecord(
                packet_id=self._next_packet_id,
                size=size,
                direction=PacketDirection.RECEIVED,
                packet_type=packet_type,
                timestamp=time.time(),
                acknowledged=True,
                channel=channel
            )
            self._next_packet_id += 1

            self._packet_history.append(record)

            self._window_bytes_received += size
            self._window_packets_received += 1
            self._total_bytes_received += size
            self._total_packets_received += 1

            # Update connection stats
            if connection_id and connection_id in self._connections:
                conn = self._connections[connection_id]
                conn.stats.bytes_received += size
                conn.stats.packets_received += 1
                conn.last_activity = time.time()

            self._check_window()

    def track_packet_ack(
        self,
        packet_id: int,
        connection_id: Optional[str] = None
    ) -> Optional[float]:
        """Track acknowledgment of a sent packet.

        Args:
            packet_id: ID of the packet being acknowledged.
            connection_id: Optional connection identifier.

        Returns:
            Round-trip time in milliseconds, or None if packet not found.
        """
        if not self._enabled or packet_id == 0:
            return None

        with self._lock:
            if packet_id not in self._sent_packets:
                return None

            record = self._sent_packets[packet_id]
            record.acknowledged = True
            record.ack_timestamp = time.time()

            rtt_ms = record.rtt_ms
            if rtt_ms is not None:
                self._rtt_samples.append(rtt_ms)

                # Update connection RTT
                if connection_id and connection_id in self._connections:
                    conn = self._connections[connection_id]
                    conn.rtt_samples.append(rtt_ms)
                    rtt_limit = profiling_config.network_rtt_sample_size.value
                    if len(conn.rtt_samples) > rtt_limit:
                        conn.rtt_samples = conn.rtt_samples[-rtt_limit:]

            # Remove from pending
            del self._sent_packets[packet_id]

            return rtt_ms

    def track_packet_lost(
        self,
        packet_id: int,
        connection_id: Optional[str] = None
    ) -> None:
        """Track a lost packet.

        Args:
            packet_id: ID of the lost packet.
            connection_id: Optional connection identifier.
        """
        if not self._enabled or packet_id == 0:
            return

        with self._lock:
            if packet_id in self._sent_packets:
                del self._sent_packets[packet_id]

            self._window_packets_lost += 1

    def register_connection(
        self,
        connection_id: str,
        remote_address: str
    ) -> ConnectionStats:
        """Register a new network connection.

        Args:
            connection_id: Unique identifier for the connection.
            remote_address: Remote endpoint address.

        Returns:
            The created ConnectionStats.
        """
        with self._lock:
            now = time.time()
            conn = ConnectionStats(
                connection_id=connection_id,
                remote_address=remote_address,
                connected_since=now,
                last_activity=now
            )
            self._connections[connection_id] = conn
            return conn

    def unregister_connection(self, connection_id: str) -> Optional[ConnectionStats]:
        """Unregister a network connection.

        Args:
            connection_id: ID of the connection to remove.

        Returns:
            The removed connection stats, or None if not found.
        """
        with self._lock:
            return self._connections.pop(connection_id, None)

    def get_connection(self, connection_id: str) -> Optional[ConnectionStats]:
        """Get stats for a specific connection.

        Args:
            connection_id: ID of the connection.

        Returns:
            Connection stats or None if not found.
        """
        with self._lock:
            return self._connections.get(connection_id)

    def _check_window(self) -> None:
        """Check if the statistics window needs to be reset."""
        now = time.time()
        if now - self._window_start >= self._window_seconds:
            self._window_start = now
            self._window_bytes_sent = 0
            self._window_bytes_received = 0
            self._window_packets_sent = 0
            self._window_packets_received = 0
            self._window_packets_lost = 0

            # Detect lost packets (unacknowledged for too long)
            timeout_multiplier = profiling_config.network_packet_timeout_multiplier.value
            timeout_threshold = now - (self._window_seconds * timeout_multiplier)
            lost_ids = [
                pid for pid, record in self._sent_packets.items()
                if record.timestamp < timeout_threshold
            ]
            for pid in lost_ids:
                self._window_packets_lost += 1
                del self._sent_packets[pid]

    def get_stats(self) -> NetworkStats:
        """Get current network statistics.

        Returns:
            NetworkStats with current values.
        """
        with self._lock:
            self._check_window()

            # Calculate RTT
            avg_rtt = 0.0
            if self._rtt_samples:
                avg_rtt = sum(self._rtt_samples) / len(self._rtt_samples)

            # Calculate loss percentage
            total_sent = self._window_packets_sent + self._window_packets_lost
            loss_percent = 0.0
            if total_sent > 0:
                loss_percent = (self._window_packets_lost / total_sent) * 100

            # Calculate jitter (variation in RTT)
            jitter = 0.0
            if len(self._rtt_samples) >= 2:
                samples = list(self._rtt_samples)
                diffs = [abs(samples[i] - samples[i-1]) for i in range(1, len(samples))]
                jitter = sum(diffs) / len(diffs) if diffs else 0.0

            return NetworkStats(
                bytes_sent=self._window_bytes_sent,
                bytes_received=self._window_bytes_received,
                packets_sent=self._window_packets_sent,
                packets_received=self._window_packets_received,
                rtt_ms=avg_rtt,
                loss_percent=loss_percent,
                jitter_ms=jitter
            )

    def get_total_stats(self) -> NetworkStats:
        """Get accumulated network statistics.

        Returns:
            NetworkStats with total values since start/reset.
        """
        with self._lock:
            avg_rtt = 0.0
            if self._rtt_samples:
                avg_rtt = sum(self._rtt_samples) / len(self._rtt_samples)

            return NetworkStats(
                bytes_sent=self._total_bytes_sent,
                bytes_received=self._total_bytes_received,
                packets_sent=self._total_packets_sent,
                packets_received=self._total_packets_received,
                rtt_ms=avg_rtt,
                loss_percent=0.0  # Can't calculate accurately for total
            )

    def get_bandwidth_history(
        self,
        duration_seconds: Optional[float] = None
    ) -> List[Tuple[float, int, int]]:
        """Get bandwidth history over time.

        Args:
            duration_seconds: How far back to look.
                             Defaults to profiler.network.BandwidthHistorySeconds CVar.

        Returns:
            List of (timestamp, bytes_sent, bytes_received) tuples.
        """
        if duration_seconds is None:
            duration_seconds = profiling_config.network_bandwidth_history_seconds.value
        with self._lock:
            cutoff = time.time() - duration_seconds

            # Group packets by time buckets
            buckets: Dict[int, Tuple[int, int]] = {}

            for record in self._packet_history:
                if record.timestamp < cutoff:
                    continue

                bucket = int(record.timestamp)
                if bucket not in buckets:
                    buckets[bucket] = (0, 0)

                sent, recv = buckets[bucket]
                if record.direction == PacketDirection.SENT:
                    buckets[bucket] = (sent + record.size, recv)
                else:
                    buckets[bucket] = (sent, recv + record.size)

            return [
                (float(ts), sent, recv)
                for ts, (sent, recv) in sorted(buckets.items())
            ]

    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._next_packet_id = 1
            self._sent_packets.clear()
            self._packet_history.clear()
            self._rtt_samples.clear()
            self._connections.clear()

            self._window_start = time.time()
            self._window_bytes_sent = 0
            self._window_bytes_received = 0
            self._window_packets_sent = 0
            self._window_packets_received = 0
            self._window_packets_lost = 0

            self._total_bytes_sent = 0
            self._total_bytes_received = 0
            self._total_packets_sent = 0
            self._total_packets_received = 0

    def format_stats(self) -> str:
        """Format current stats as a human-readable string.

        Returns:
            Formatted statistics string.
        """
        stats = self.get_stats()
        total = self.get_total_stats()

        lines = [
            "Network Statistics",
            "------------------",
            f"Current Window ({self._window_seconds}s):",
            f"  Bandwidth: {stats.bandwidth_sent_kbps:.1f} kbps up, "
            f"{stats.bandwidth_received_kbps:.1f} kbps down",
            f"  Packets: {stats.packets_sent} sent, {stats.packets_received} received",
            f"  RTT: {stats.rtt_ms:.1f}ms (jitter: {stats.jitter_ms:.1f}ms)",
            f"  Loss: {stats.loss_percent:.1f}%",
            f"",
            f"Total:",
            f"  Bytes: {total.bytes_sent / 1024:.1f} KB sent, "
            f"{total.bytes_received / 1024:.1f} KB received",
            f"  Packets: {total.packets_sent} sent, {total.packets_received} received",
        ]

        if self._connections:
            lines.append("")
            lines.append(f"Connections ({len(self._connections)}):")
            for conn in self._connections.values():
                lines.append(
                    f"  {conn.connection_id} ({conn.remote_address}): "
                    f"RTT {conn.average_rtt_ms:.1f}ms"
                )

        return "\n".join(lines)


# Global default network profiler instance
_default_network_profiler = NetworkProfiler()


def get_default_network_profiler() -> NetworkProfiler:
    """Get the global default network profiler."""
    return _default_network_profiler


def set_default_network_profiler(profiler: NetworkProfiler) -> None:
    """Set the global default network profiler."""
    global _default_network_profiler
    _default_network_profiler = profiler
