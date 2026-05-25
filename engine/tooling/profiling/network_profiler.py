"""
Network Profiler for the AI Game Engine.

Provides comprehensive network profiling with:
- Bandwidth tracking (KB/s sent/received)
- Packet inspection and statistics
- Latency monitoring and graphs
- Per-actor/per-property bandwidth breakdown
- Packet loss detection
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)


class NetworkProfilerState(Enum):
    """Network profiler operational state."""
    DISABLED = auto()
    ENABLED = auto()
    PAUSED = auto()


class PacketDirection(Enum):
    """Packet direction."""
    SENT = auto()
    RECEIVED = auto()


class PacketType(Enum):
    """Types of network packets."""
    RELIABLE = auto()
    UNRELIABLE = auto()
    ORDERED = auto()
    SEQUENCED = auto()
    BROADCAST = auto()
    MULTICAST = auto()


@dataclass(slots=True)
class PacketRecord:
    """Record of a single network packet."""
    packet_id: int
    timestamp: float
    direction: PacketDirection
    packet_type: PacketType
    size_bytes: int
    channel: str = ""
    actor_id: Optional[int] = None
    property_name: Optional[str] = None
    sequence_number: int = 0
    ack_number: int = 0
    rtt_ms: Optional[float] = None
    was_dropped: bool = False
    was_retransmitted: bool = False

    @property
    def size_kb(self) -> float:
        """Size in kilobytes."""
        return self.size_bytes / 1024

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "packet_id": self.packet_id,
            "timestamp": self.timestamp,
            "direction": self.direction.name,
            "packet_type": self.packet_type.name,
            "size_bytes": self.size_bytes,
            "size_kb": self.size_kb,
            "channel": self.channel,
            "actor_id": self.actor_id,
            "property_name": self.property_name,
            "sequence_number": self.sequence_number,
            "ack_number": self.ack_number,
            "rtt_ms": self.rtt_ms,
            "was_dropped": self.was_dropped,
            "was_retransmitted": self.was_retransmitted,
        }


@dataclass
class BandwidthSample:
    """A bandwidth sample over a time window."""
    timestamp: float
    duration_seconds: float
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0

    @property
    def sent_kbps(self) -> float:
        """Sent bandwidth in kilobytes per second."""
        if self.duration_seconds == 0:
            return 0.0
        return (self.bytes_sent / 1024) / self.duration_seconds

    @property
    def received_kbps(self) -> float:
        """Received bandwidth in kilobytes per second."""
        if self.duration_seconds == 0:
            return 0.0
        return (self.bytes_received / 1024) / self.duration_seconds

    @property
    def total_kbps(self) -> float:
        """Total bandwidth in kilobytes per second."""
        return self.sent_kbps + self.received_kbps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "packets_sent": self.packets_sent,
            "packets_received": self.packets_received,
            "sent_kbps": self.sent_kbps,
            "received_kbps": self.received_kbps,
            "total_kbps": self.total_kbps,
        }


@dataclass
class LatencyGraph:
    """Latency measurements over time."""
    samples: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, rtt_ms)
    min_rtt_ms: float = float("inf")
    max_rtt_ms: float = 0.0
    avg_rtt_ms: float = 0.0
    jitter_ms: float = 0.0

    def add_sample(self, timestamp: float, rtt_ms: float) -> None:
        """Add a latency sample."""
        self.samples.append((timestamp, rtt_ms))

        # Update statistics
        self.min_rtt_ms = min(self.min_rtt_ms, rtt_ms)
        self.max_rtt_ms = max(self.max_rtt_ms, rtt_ms)

        if len(self.samples) > 0:
            rtts = [s[1] for s in self.samples]
            self.avg_rtt_ms = sum(rtts) / len(rtts)

            # Calculate jitter (average deviation from mean)
            if len(rtts) > 1:
                deviations = [abs(r - self.avg_rtt_ms) for r in rtts]
                self.jitter_ms = sum(deviations) / len(deviations)

    def get_recent(self, seconds: float = 10.0) -> List[Tuple[float, float]]:
        """Get samples from the last N seconds."""
        cutoff = time.time() - seconds
        return [(t, r) for t, r in self.samples if t >= cutoff]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_count": len(self.samples),
            "min_rtt_ms": self.min_rtt_ms if self.min_rtt_ms != float("inf") else 0.0,
            "max_rtt_ms": self.max_rtt_ms,
            "avg_rtt_ms": self.avg_rtt_ms,
            "jitter_ms": self.jitter_ms,
        }


@dataclass
class NetworkStats:
    """Aggregated network statistics."""
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_packets_sent: int = 0
    total_packets_received: int = 0
    packets_dropped: int = 0
    packets_retransmitted: int = 0
    current_rtt_ms: float = 0.0
    avg_rtt_ms: float = 0.0
    min_rtt_ms: float = float("inf")
    max_rtt_ms: float = 0.0
    jitter_ms: float = 0.0

    @property
    def packet_loss_percentage(self) -> float:
        """Packet loss as percentage."""
        total = self.total_packets_sent + self.total_packets_received
        if total == 0:
            return 0.0
        return (self.packets_dropped / total) * 100.0

    @property
    def total_bytes(self) -> int:
        """Total bytes transferred."""
        return self.total_bytes_sent + self.total_bytes_received

    @property
    def total_kb(self) -> float:
        """Total kilobytes transferred."""
        return self.total_bytes / 1024

    @property
    def total_mb(self) -> float:
        """Total megabytes transferred."""
        return self.total_bytes / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "total_bytes": self.total_bytes,
            "total_kb": self.total_kb,
            "total_mb": self.total_mb,
            "total_packets_sent": self.total_packets_sent,
            "total_packets_received": self.total_packets_received,
            "packets_dropped": self.packets_dropped,
            "packets_retransmitted": self.packets_retransmitted,
            "packet_loss_percentage": self.packet_loss_percentage,
            "current_rtt_ms": self.current_rtt_ms,
            "avg_rtt_ms": self.avg_rtt_ms,
            "min_rtt_ms": self.min_rtt_ms if self.min_rtt_ms != float("inf") else 0.0,
            "max_rtt_ms": self.max_rtt_ms,
            "jitter_ms": self.jitter_ms,
        }


@dataclass
class ChannelStats:
    """Statistics for a network channel."""
    channel_name: str
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    packets_dropped: int = 0

    @property
    def total_bytes(self) -> int:
        return self.bytes_sent + self.bytes_received

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_name": self.channel_name,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "total_bytes": self.total_bytes,
            "packets_sent": self.packets_sent,
            "packets_received": self.packets_received,
            "packets_dropped": self.packets_dropped,
        }


@dataclass
class ActorNetworkStats:
    """Network statistics for a specific actor."""
    actor_id: int
    bytes_sent: int = 0
    bytes_received: int = 0
    updates_sent: int = 0
    updates_received: int = 0
    properties: Dict[str, int] = field(default_factory=dict)  # property -> bytes

    @property
    def total_bytes(self) -> int:
        return self.bytes_sent + self.bytes_received

    def add_property_bytes(self, property_name: str, size: int) -> None:
        """Add bytes for a property."""
        if property_name not in self.properties:
            self.properties[property_name] = 0
        self.properties[property_name] += size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "total_bytes": self.total_bytes,
            "updates_sent": self.updates_sent,
            "updates_received": self.updates_received,
            "properties": dict(self.properties),
        }


class NetworkProfiler:
    """
    Network Profiler with bandwidth and latency tracking.

    Features:
    - Bandwidth tracking (KB/s sent/received)
    - Packet inspection and statistics
    - Latency monitoring and graphs
    - Per-actor/per-property bandwidth breakdown
    - Per-channel statistics
    - Packet loss detection
    """

    __slots__ = (
        "_state",
        "_packets",
        "_lock",
        "_max_packets",
        "_packet_counter",
        "_stats",
        "_latency_graph",
        "_bandwidth_samples",
        "_channel_stats",
        "_actor_stats",
        "_sample_window",
        "_current_window_start",
        "_current_window",
        "_listeners",
        "_pending_acks",
    )

    def __init__(
        self,
        max_packets: int = 50000,
        sample_window_seconds: float = 1.0,
    ) -> None:
        """
        Initialize the network profiler.

        Args:
            max_packets: Maximum packet records to retain
            sample_window_seconds: Bandwidth sampling window
        """
        self._state = NetworkProfilerState.DISABLED
        self._packets: Deque[PacketRecord] = deque(maxlen=max_packets)
        self._lock = threading.RLock()
        self._max_packets = max_packets
        self._packet_counter = 0
        self._stats = NetworkStats()
        self._latency_graph = LatencyGraph()
        self._bandwidth_samples: List[BandwidthSample] = []
        self._channel_stats: Dict[str, ChannelStats] = {}
        self._actor_stats: Dict[int, ActorNetworkStats] = {}
        self._sample_window = sample_window_seconds
        self._current_window_start = time.time()
        self._current_window = BandwidthSample(
            timestamp=self._current_window_start,
            duration_seconds=sample_window_seconds,
        )
        self._listeners: Set[Callable[[PacketRecord], None]] = set()
        self._pending_acks: Dict[int, Tuple[float, int]] = {}  # seq -> (timestamp, size)

    @property
    def is_enabled(self) -> bool:
        """Check if profiler is enabled."""
        return self._state == NetworkProfilerState.ENABLED

    @property
    def state(self) -> NetworkProfilerState:
        """Get current profiler state."""
        return self._state

    def enable(self) -> None:
        """Enable the network profiler."""
        with self._lock:
            self._state = NetworkProfilerState.ENABLED

    def disable(self) -> None:
        """Disable the network profiler."""
        with self._lock:
            self._state = NetworkProfilerState.DISABLED

    def pause(self) -> None:
        """Pause profiling without clearing data."""
        with self._lock:
            if self._state == NetworkProfilerState.ENABLED:
                self._state = NetworkProfilerState.PAUSED

    def resume(self) -> None:
        """Resume profiling from paused state."""
        with self._lock:
            if self._state == NetworkProfilerState.PAUSED:
                self._state = NetworkProfilerState.ENABLED

    def clear(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self._packets.clear()
            self._packet_counter = 0
            self._stats = NetworkStats()
            self._latency_graph = LatencyGraph()
            self._bandwidth_samples.clear()
            self._channel_stats.clear()
            self._actor_stats.clear()
            self._current_window_start = time.time()
            self._current_window = BandwidthSample(
                timestamp=self._current_window_start,
                duration_seconds=self._sample_window,
            )
            self._pending_acks.clear()

    def add_listener(self, callback: Callable[[PacketRecord], None]) -> None:
        """Add a packet listener."""
        self._listeners.add(callback)

    def remove_listener(self, callback: Callable[[PacketRecord], None]) -> None:
        """Remove a packet listener."""
        self._listeners.discard(callback)

    def _check_window(self) -> None:
        """Check and possibly rotate the bandwidth sample window."""
        current_time = time.time()
        if current_time - self._current_window_start >= self._sample_window:
            # Finalize current window
            self._bandwidth_samples.append(self._current_window)

            # Keep only last 60 samples (1 minute of data at 1s windows)
            if len(self._bandwidth_samples) > 60:
                self._bandwidth_samples = self._bandwidth_samples[-60:]

            # Start new window
            self._current_window_start = current_time
            self._current_window = BandwidthSample(
                timestamp=current_time,
                duration_seconds=self._sample_window,
            )

    def record_packet(
        self,
        direction: PacketDirection,
        size_bytes: int,
        packet_type: PacketType = PacketType.RELIABLE,
        channel: str = "",
        actor_id: Optional[int] = None,
        property_name: Optional[str] = None,
        sequence_number: int = 0,
        ack_number: int = 0,
    ) -> int:
        """
        Record a network packet.

        Args:
            direction: Sent or received
            size_bytes: Packet size in bytes
            packet_type: Type of packet
            channel: Network channel
            actor_id: Associated actor ID
            property_name: Associated property name
            sequence_number: Packet sequence number
            ack_number: Acknowledgment number

        Returns:
            The packet ID
        """
        if self._state != NetworkProfilerState.ENABLED:
            return 0

        with self._lock:
            self._check_window()

            self._packet_counter += 1
            timestamp = time.time()

            # Calculate RTT if this is an ACK
            rtt_ms = None
            if ack_number > 0 and ack_number in self._pending_acks:
                sent_time, _ = self._pending_acks.pop(ack_number)
                rtt_ms = (timestamp - sent_time) * 1000.0
                self._latency_graph.add_sample(timestamp, rtt_ms)
                self._update_rtt_stats(rtt_ms)

            # Track sent packets for RTT calculation
            if direction == PacketDirection.SENT and sequence_number > 0:
                self._pending_acks[sequence_number] = (timestamp, size_bytes)

            record = PacketRecord(
                packet_id=self._packet_counter,
                timestamp=timestamp,
                direction=direction,
                packet_type=packet_type,
                size_bytes=size_bytes,
                channel=channel,
                actor_id=actor_id,
                property_name=property_name,
                sequence_number=sequence_number,
                ack_number=ack_number,
                rtt_ms=rtt_ms,
            )

            self._packets.append(record)
            self._update_stats(record)
            self._update_channel_stats(record)
            self._update_actor_stats(record)

            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(record)
                except Exception:
                    pass

            return self._packet_counter

    def _update_stats(self, record: PacketRecord) -> None:
        """Update global statistics."""
        if record.direction == PacketDirection.SENT:
            self._stats.total_bytes_sent += record.size_bytes
            self._stats.total_packets_sent += 1
            self._current_window.bytes_sent += record.size_bytes
            self._current_window.packets_sent += 1
        else:
            self._stats.total_bytes_received += record.size_bytes
            self._stats.total_packets_received += 1
            self._current_window.bytes_received += record.size_bytes
            self._current_window.packets_received += 1

        if record.was_dropped:
            self._stats.packets_dropped += 1

        if record.was_retransmitted:
            self._stats.packets_retransmitted += 1

    def _update_rtt_stats(self, rtt_ms: float) -> None:
        """Update RTT statistics."""
        self._stats.current_rtt_ms = rtt_ms
        self._stats.min_rtt_ms = min(self._stats.min_rtt_ms, rtt_ms)
        self._stats.max_rtt_ms = max(self._stats.max_rtt_ms, rtt_ms)
        self._stats.avg_rtt_ms = self._latency_graph.avg_rtt_ms
        self._stats.jitter_ms = self._latency_graph.jitter_ms

    def _update_channel_stats(self, record: PacketRecord) -> None:
        """Update per-channel statistics."""
        if not record.channel:
            return

        if record.channel not in self._channel_stats:
            self._channel_stats[record.channel] = ChannelStats(
                channel_name=record.channel
            )

        stats = self._channel_stats[record.channel]
        if record.direction == PacketDirection.SENT:
            stats.bytes_sent += record.size_bytes
            stats.packets_sent += 1
        else:
            stats.bytes_received += record.size_bytes
            stats.packets_received += 1

        if record.was_dropped:
            stats.packets_dropped += 1

    def _update_actor_stats(self, record: PacketRecord) -> None:
        """Update per-actor statistics."""
        if record.actor_id is None:
            return

        if record.actor_id not in self._actor_stats:
            self._actor_stats[record.actor_id] = ActorNetworkStats(
                actor_id=record.actor_id
            )

        stats = self._actor_stats[record.actor_id]
        if record.direction == PacketDirection.SENT:
            stats.bytes_sent += record.size_bytes
            stats.updates_sent += 1
        else:
            stats.bytes_received += record.size_bytes
            stats.updates_received += 1

        if record.property_name:
            stats.add_property_bytes(record.property_name, record.size_bytes)

    def record_packet_drop(self, packet_id: int) -> None:
        """Mark a packet as dropped."""
        if self._state != NetworkProfilerState.ENABLED:
            return

        with self._lock:
            for packet in self._packets:
                if packet.packet_id == packet_id:
                    packet.was_dropped = True
                    self._stats.packets_dropped += 1
                    break

    def record_retransmission(self, packet_id: int) -> None:
        """Mark a packet as retransmitted."""
        if self._state != NetworkProfilerState.ENABLED:
            return

        with self._lock:
            for packet in self._packets:
                if packet.packet_id == packet_id:
                    packet.was_retransmitted = True
                    self._stats.packets_retransmitted += 1
                    break

    def record_rtt(self, rtt_ms: float) -> None:
        """Record an RTT measurement directly."""
        if self._state != NetworkProfilerState.ENABLED:
            return

        with self._lock:
            timestamp = time.time()
            self._latency_graph.add_sample(timestamp, rtt_ms)
            self._update_rtt_stats(rtt_ms)

    def get_stats(self) -> NetworkStats:
        """Get current network statistics."""
        with self._lock:
            return NetworkStats(
                total_bytes_sent=self._stats.total_bytes_sent,
                total_bytes_received=self._stats.total_bytes_received,
                total_packets_sent=self._stats.total_packets_sent,
                total_packets_received=self._stats.total_packets_received,
                packets_dropped=self._stats.packets_dropped,
                packets_retransmitted=self._stats.packets_retransmitted,
                current_rtt_ms=self._stats.current_rtt_ms,
                avg_rtt_ms=self._stats.avg_rtt_ms,
                min_rtt_ms=self._stats.min_rtt_ms,
                max_rtt_ms=self._stats.max_rtt_ms,
                jitter_ms=self._stats.jitter_ms,
            )

    def get_latency_graph(self) -> LatencyGraph:
        """Get the latency graph."""
        with self._lock:
            graph = LatencyGraph(
                samples=list(self._latency_graph.samples),
                min_rtt_ms=self._latency_graph.min_rtt_ms,
                max_rtt_ms=self._latency_graph.max_rtt_ms,
                avg_rtt_ms=self._latency_graph.avg_rtt_ms,
                jitter_ms=self._latency_graph.jitter_ms,
            )
            return graph

    def get_bandwidth_samples(
        self,
        last_n_seconds: Optional[float] = None,
    ) -> List[BandwidthSample]:
        """
        Get bandwidth samples.

        Args:
            last_n_seconds: If provided, only return samples from last N seconds

        Returns:
            List of bandwidth samples
        """
        with self._lock:
            samples = list(self._bandwidth_samples)

        if last_n_seconds is not None:
            cutoff = time.time() - last_n_seconds
            samples = [s for s in samples if s.timestamp >= cutoff]

        return samples

    def get_current_bandwidth(self) -> BandwidthSample:
        """Get the current bandwidth sample."""
        with self._lock:
            self._check_window()
            return BandwidthSample(
                timestamp=self._current_window.timestamp,
                duration_seconds=self._current_window.duration_seconds,
                bytes_sent=self._current_window.bytes_sent,
                bytes_received=self._current_window.bytes_received,
                packets_sent=self._current_window.packets_sent,
                packets_received=self._current_window.packets_received,
            )

    def get_channel_stats(
        self,
        channel: Optional[str] = None,
    ) -> Dict[str, ChannelStats]:
        """Get per-channel statistics."""
        with self._lock:
            if channel:
                stats = self._channel_stats.get(channel)
                return {channel: stats} if stats else {}
            return dict(self._channel_stats)

    def get_actor_stats(
        self,
        actor_id: Optional[int] = None,
    ) -> Dict[int, ActorNetworkStats]:
        """Get per-actor statistics."""
        with self._lock:
            if actor_id is not None:
                stats = self._actor_stats.get(actor_id)
                return {actor_id: stats} if stats else {}
            return dict(self._actor_stats)

    def get_packets(
        self,
        direction: Optional[PacketDirection] = None,
        channel: Optional[str] = None,
        actor_id: Optional[int] = None,
        min_size: int = 0,
        last_n_seconds: Optional[float] = None,
    ) -> List[PacketRecord]:
        """
        Get packet records with optional filtering.

        Args:
            direction: Filter by direction
            channel: Filter by channel
            actor_id: Filter by actor
            min_size: Minimum packet size
            last_n_seconds: Only packets from last N seconds

        Returns:
            List of matching packets
        """
        with self._lock:
            packets = list(self._packets)

        if direction is not None:
            packets = [p for p in packets if p.direction == direction]
        if channel is not None:
            packets = [p for p in packets if p.channel == channel]
        if actor_id is not None:
            packets = [p for p in packets if p.actor_id == actor_id]
        if min_size > 0:
            packets = [p for p in packets if p.size_bytes >= min_size]
        if last_n_seconds is not None:
            cutoff = time.time() - last_n_seconds
            packets = [p for p in packets if p.timestamp >= cutoff]

        return packets

    def get_top_bandwidth_actors(
        self,
        top_n: int = 10,
    ) -> List[Tuple[int, int]]:
        """Get actors with highest bandwidth usage."""
        with self._lock:
            actors = [
                (actor_id, stats.total_bytes)
                for actor_id, stats in self._actor_stats.items()
            ]
        actors.sort(key=lambda x: x[1], reverse=True)
        return actors[:top_n]

    def get_top_bandwidth_channels(
        self,
        top_n: int = 10,
    ) -> List[Tuple[str, int]]:
        """Get channels with highest bandwidth usage."""
        with self._lock:
            channels = [
                (name, stats.total_bytes)
                for name, stats in self._channel_stats.items()
            ]
        channels.sort(key=lambda x: x[1], reverse=True)
        return channels[:top_n]

    def to_dict(self) -> Dict[str, Any]:
        """Export network profiler data as dictionary."""
        with self._lock:
            return {
                "state": self._state.name,
                "packet_count": len(self._packets),
                "stats": self._stats.to_dict(),
                "latency": self._latency_graph.to_dict(),
                "bandwidth_samples": len(self._bandwidth_samples),
                "channel_count": len(self._channel_stats),
                "actor_count": len(self._actor_stats),
            }


# Global network profiler instance
network_profiler = NetworkProfiler()
