"""
Network Debugging - Latency simulation, packet loss, jitter, bandwidth limits.

Provides tools for debugging network systems including:
- Latency simulation
- Packet loss simulation
- Jitter simulation
- Bandwidth limiting
- Packet logging
- Network statistics
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
)

logger = logging.getLogger(__name__)


class PacketDirection(Enum):
    """Direction of network packet."""
    INBOUND = auto()
    OUTBOUND = auto()


@dataclass
class PacketLog:
    """Log entry for a network packet."""
    timestamp: float
    direction: PacketDirection
    size: int
    channel: str
    packet_type: str
    source: str
    destination: str
    latency_applied: float = 0.0
    dropped: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkStats:
    """Network statistics snapshot."""
    # Bandwidth
    bytes_sent: int = 0
    bytes_received: int = 0
    bytes_per_second_sent: float = 0.0
    bytes_per_second_received: float = 0.0

    # Packets
    packets_sent: int = 0
    packets_received: int = 0
    packets_dropped: int = 0
    packets_per_second: float = 0.0

    # Latency
    current_latency_ms: float = 0.0
    average_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    jitter_ms: float = 0.0

    # RTT
    rtt_ms: float = 0.0
    rtt_average_ms: float = 0.0

    # Connection
    connection_quality: float = 1.0  # 0.0 to 1.0
    packet_loss_percent: float = 0.0

    # Simulation
    simulated_latency_ms: float = 0.0
    simulated_packet_loss: float = 0.0
    simulated_jitter_ms: float = 0.0
    bandwidth_limit_kbps: float = 0.0


@dataclass
class NetworkDebugConfig:
    """
    Configuration for network debugging system.

    All numeric constants are defined here to avoid magic numbers.
    """
    # Buffer limits
    max_recent_packets: int = 10000     # Maximum recent packets to track
    max_packet_log: int = 1000          # Maximum logged packets
    max_latency_samples: int = 100      # Latency sample buffer size
    max_rtt_samples: int = 100          # RTT sample buffer size

    # Statistics window
    stats_window_seconds: float = 1.0   # Time window for bandwidth calculation

    # Connection quality thresholds
    latency_quality_threshold_ms: float = 100.0   # Latency above this reduces quality
    latency_quality_max_penalty_ms: float = 200.0 # Additional latency for max penalty
    jitter_quality_threshold_ms: float = 30.0     # Jitter above this reduces quality
    jitter_quality_max_penalty_ms: float = 70.0   # Additional jitter for max penalty
    packet_loss_quality_divisor: float = 20.0     # Packet loss % divisor for quality

    # Quality calculation weights
    max_latency_quality_penalty: float = 0.3      # Maximum penalty from latency
    max_jitter_quality_penalty: float = 0.2       # Maximum penalty from jitter
    max_packet_loss_quality_penalty: float = 0.5  # Maximum penalty from packet loss

    # Build restrictions
    allow_in_shipping: bool = False     # Disable network debug in shipping


@dataclass
class NetworkSimulation:
    """Network simulation configuration."""
    latency_ms: float = 0.0
    latency_variance_ms: float = 0.0  # Random variance added to latency
    packet_loss_percent: float = 0.0
    packet_loss_burst_chance: float = 0.0  # Chance of burst packet loss
    packet_loss_burst_length: int = 3  # Packets in a burst
    jitter_ms: float = 0.0
    bandwidth_limit_kbps: float = 0.0  # 0 = unlimited
    duplicate_chance: float = 0.0  # Chance of duplicating packets
    reorder_chance: float = 0.0  # Chance of reordering packets
    corruption_chance: float = 0.0  # Chance of corrupting packets


class NetworkDebugger:
    """
    Debugger for network systems.

    SECURITY: This debugger is automatically disabled in shipping builds
    to prevent network manipulation exploits.

    Provides simulation and monitoring for:
    - Latency (with variance)
    - Packet loss (with burst support)
    - Jitter
    - Bandwidth limiting
    - Packet logging
    - Network statistics
    """

    def __init__(self, config: Optional[NetworkDebugConfig] = None) -> None:
        self._config = config or NetworkDebugConfig()
        self._enabled = True
        self._simulation = NetworkSimulation()
        self._logging_enabled = False

        # Check build restrictions
        self._build_allowed = self._check_build_allowed()

        # Statistics tracking - limits from config
        self._stats = NetworkStats()
        self._stats_window_seconds = self._config.stats_window_seconds
        self._recent_packets: Deque[PacketLog] = deque(maxlen=self._config.max_recent_packets)

        # Latency tracking - limits from config
        self._latency_samples: Deque[float] = deque(maxlen=self._config.max_latency_samples)
        self._rtt_samples: Deque[float] = deque(maxlen=self._config.max_rtt_samples)

        # Bandwidth tracking
        self._bytes_sent_window: Deque[Tuple[float, int]] = deque()
        self._bytes_received_window: Deque[Tuple[float, int]] = deque()

        # Burst tracking
        self._in_packet_loss_burst = False
        self._burst_remaining = 0

        # Packet log - limit from config
        self._packet_log: Deque[PacketLog] = deque(maxlen=self._config.max_packet_log)

        # Callbacks
        self._stat_callbacks: List[Callable[[NetworkStats], None]] = []

    def _check_build_allowed(self) -> bool:
        """Check if network debugging is allowed in this build."""
        import os

        if os.environ.get("GAME_BUILD_TYPE", "").upper() == "SHIPPING":
            if not self._config.allow_in_shipping:
                logger.info("NetworkDebugger disabled - shipping build")
                return False
        if os.environ.get("SHIPPING") == "1":
            if not self._config.allow_in_shipping:
                return False

        return True

    @property
    def config(self) -> NetworkDebugConfig:
        """Get the network debug configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Check if network debugging is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable network debugging."""
        if value and not self._build_allowed:
            logger.warning("Cannot enable network debugger - not allowed in this build")
            return
        self._enabled = value
        if not value:
            self._simulation = NetworkSimulation()
            self._logging_enabled = False

    @property
    def simulation(self) -> NetworkSimulation:
        """Get current simulation settings."""
        return self._simulation

    # =========================================================================
    # Simulation Control
    # =========================================================================

    def set_latency(self, ms: float, variance_ms: float = 0.0) -> None:
        """
        Set simulated network latency.

        Args:
            ms: Base latency in milliseconds
            variance_ms: Random variance added to latency
        """
        self._simulation.latency_ms = max(0.0, ms)
        self._simulation.latency_variance_ms = max(0.0, variance_ms)

        logger.info(
            "Network latency: %dms (+/- %dms)",
            int(ms),
            int(variance_ms),
        )

    def set_packet_loss(
        self,
        percent: float,
        burst_chance: float = 0.0,
        burst_length: int = 3,
    ) -> None:
        """
        Set simulated packet loss.

        Args:
            percent: Packet loss percentage (0-100)
            burst_chance: Chance of burst loss (0-100)
            burst_length: Packets in a burst
        """
        self._simulation.packet_loss_percent = max(0.0, min(100.0, percent))
        self._simulation.packet_loss_burst_chance = max(0.0, min(100.0, burst_chance))
        self._simulation.packet_loss_burst_length = max(1, burst_length)

        logger.info("Network packet loss: %.1f%%", percent)

    def set_jitter(self, ms: float) -> None:
        """
        Set simulated network jitter.

        Args:
            ms: Jitter in milliseconds (random delay variation)
        """
        self._simulation.jitter_ms = max(0.0, ms)
        logger.info("Network jitter: %dms", int(ms))

    def set_bandwidth_limit(self, kbps: float) -> None:
        """
        Set simulated bandwidth limit.

        Args:
            kbps: Bandwidth limit in kilobits per second (0 = unlimited)
        """
        self._simulation.bandwidth_limit_kbps = max(0.0, kbps)

        if kbps > 0:
            logger.info("Bandwidth limit: %.1f kbps", kbps)
        else:
            logger.info("Bandwidth limit: unlimited")

    def set_duplicate_chance(self, percent: float) -> None:
        """Set chance of packet duplication."""
        self._simulation.duplicate_chance = max(0.0, min(100.0, percent))

    def set_reorder_chance(self, percent: float) -> None:
        """Set chance of packet reordering."""
        self._simulation.reorder_chance = max(0.0, min(100.0, percent))

    def reset_simulation(self) -> None:
        """Reset all simulation settings to default."""
        self._simulation = NetworkSimulation()
        self._in_packet_loss_burst = False
        self._burst_remaining = 0
        logger.info("Network simulation reset")

    # =========================================================================
    # Packet Processing
    # =========================================================================

    def should_drop_packet(self) -> bool:
        """
        Check if a packet should be dropped based on simulation.

        Returns:
            True if packet should be dropped.
        """
        if not self._enabled:
            return False

        # Check burst loss
        if self._in_packet_loss_burst and self._burst_remaining > 0:
            self._burst_remaining -= 1
            if self._burst_remaining <= 0:
                self._in_packet_loss_burst = False
            return True

        # Check random loss
        import random
        if random.random() * 100 < self._simulation.packet_loss_percent:
            # Check for burst
            if random.random() * 100 < self._simulation.packet_loss_burst_chance:
                self._in_packet_loss_burst = True
                self._burst_remaining = self._simulation.packet_loss_burst_length - 1
            return True

        return False

    def get_simulated_latency(self) -> float:
        """
        Get the simulated latency for a packet.

        Returns:
            Latency in milliseconds.
        """
        if not self._enabled:
            return 0.0

        import random

        latency = self._simulation.latency_ms

        # Add variance
        if self._simulation.latency_variance_ms > 0:
            variance = random.uniform(
                -self._simulation.latency_variance_ms,
                self._simulation.latency_variance_ms,
            )
            latency += variance

        # Add jitter
        if self._simulation.jitter_ms > 0:
            jitter = random.uniform(0, self._simulation.jitter_ms)
            latency += jitter

        return max(0.0, latency)

    def check_bandwidth(self, bytes_to_send: int) -> bool:
        """
        Check if bandwidth allows sending bytes.

        Args:
            bytes_to_send: Number of bytes to send

        Returns:
            True if bandwidth allows sending.
        """
        if not self._enabled or self._simulation.bandwidth_limit_kbps <= 0:
            return True

        # Calculate current bandwidth usage
        current_time = time.time()
        window_start = current_time - self._stats_window_seconds

        # Clean old entries
        while self._bytes_sent_window and self._bytes_sent_window[0][0] < window_start:
            self._bytes_sent_window.popleft()

        # Calculate bytes in window
        bytes_in_window = sum(b for _, b in self._bytes_sent_window)

        # Calculate max bytes per window
        max_bytes = (self._simulation.bandwidth_limit_kbps * 1000 / 8) * self._stats_window_seconds

        return (bytes_in_window + bytes_to_send) <= max_bytes

    # =========================================================================
    # Logging
    # =========================================================================

    def log_packets(self, enabled: bool = True) -> None:
        """Enable or disable packet logging."""
        self._logging_enabled = enabled
        if enabled:
            logger.info("Packet logging enabled")
        else:
            logger.info("Packet logging disabled")

    def log_packet(
        self,
        direction: PacketDirection,
        size: int,
        channel: str = "default",
        packet_type: str = "unknown",
        source: str = "",
        destination: str = "",
        latency_applied: float = 0.0,
        dropped: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a network packet.

        Args:
            direction: Packet direction
            size: Packet size in bytes
            channel: Network channel name
            packet_type: Type of packet
            source: Source address
            destination: Destination address
            latency_applied: Simulated latency applied
            dropped: Whether packet was dropped
            metadata: Additional metadata
        """
        log_entry = PacketLog(
            timestamp=time.time(),
            direction=direction,
            size=size,
            channel=channel,
            packet_type=packet_type,
            source=source,
            destination=destination,
            latency_applied=latency_applied,
            dropped=dropped,
            metadata=metadata or {},
        )

        self._recent_packets.append(log_entry)

        if self._logging_enabled:
            self._packet_log.append(log_entry)

        # Update statistics
        self._update_stats_from_packet(log_entry)

    def get_packet_log(
        self,
        direction: Optional[PacketDirection] = None,
        channel: Optional[str] = None,
        limit: int = 100,
    ) -> List[PacketLog]:
        """
        Get logged packets with optional filtering.

        Args:
            direction: Filter by direction
            channel: Filter by channel
            limit: Maximum entries to return

        Returns:
            List of packet logs.
        """
        result = []
        for log_entry in reversed(self._packet_log):
            if direction is not None and log_entry.direction != direction:
                continue
            if channel is not None and log_entry.channel != channel:
                continue
            result.append(log_entry)
            if len(result) >= limit:
                break
        return result

    def clear_packet_log(self) -> None:
        """Clear the packet log."""
        self._packet_log.clear()

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> NetworkStats:
        """Get current network statistics."""
        self._update_stats()
        return NetworkStats(
            bytes_sent=self._stats.bytes_sent,
            bytes_received=self._stats.bytes_received,
            bytes_per_second_sent=self._stats.bytes_per_second_sent,
            bytes_per_second_received=self._stats.bytes_per_second_received,
            packets_sent=self._stats.packets_sent,
            packets_received=self._stats.packets_received,
            packets_dropped=self._stats.packets_dropped,
            packets_per_second=self._stats.packets_per_second,
            current_latency_ms=self._stats.current_latency_ms,
            average_latency_ms=self._stats.average_latency_ms,
            min_latency_ms=self._stats.min_latency_ms,
            max_latency_ms=self._stats.max_latency_ms,
            jitter_ms=self._stats.jitter_ms,
            rtt_ms=self._stats.rtt_ms,
            rtt_average_ms=self._stats.rtt_average_ms,
            connection_quality=self._stats.connection_quality,
            packet_loss_percent=self._stats.packet_loss_percent,
            simulated_latency_ms=self._simulation.latency_ms,
            simulated_packet_loss=self._simulation.packet_loss_percent,
            simulated_jitter_ms=self._simulation.jitter_ms,
            bandwidth_limit_kbps=self._simulation.bandwidth_limit_kbps,
        )

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self._latency_samples.append(latency_ms)
        self._stats.current_latency_ms = latency_ms

    def record_rtt(self, rtt_ms: float) -> None:
        """Record an RTT measurement."""
        self._rtt_samples.append(rtt_ms)
        self._stats.rtt_ms = rtt_ms

    def _update_stats_from_packet(self, packet: PacketLog) -> None:
        """Update statistics from a packet log."""
        if packet.direction == PacketDirection.OUTBOUND:
            self._stats.packets_sent += 1
            if not packet.dropped:
                self._stats.bytes_sent += packet.size
                self._bytes_sent_window.append((packet.timestamp, packet.size))
        else:
            self._stats.packets_received += 1
            if not packet.dropped:
                self._stats.bytes_received += packet.size
                self._bytes_received_window.append((packet.timestamp, packet.size))

        if packet.dropped:
            self._stats.packets_dropped += 1

    def _update_stats(self) -> None:
        """Update computed statistics."""
        current_time = time.time()
        window_start = current_time - self._stats_window_seconds

        # Clean old entries
        while self._bytes_sent_window and self._bytes_sent_window[0][0] < window_start:
            self._bytes_sent_window.popleft()
        while self._bytes_received_window and self._bytes_received_window[0][0] < window_start:
            self._bytes_received_window.popleft()

        # Calculate bandwidth
        self._stats.bytes_per_second_sent = sum(
            b for _, b in self._bytes_sent_window
        ) / self._stats_window_seconds
        self._stats.bytes_per_second_received = sum(
            b for _, b in self._bytes_received_window
        ) / self._stats_window_seconds

        # Calculate latency stats
        if self._latency_samples:
            self._stats.average_latency_ms = sum(self._latency_samples) / len(self._latency_samples)
            self._stats.min_latency_ms = min(self._latency_samples)
            self._stats.max_latency_ms = max(self._latency_samples)

            # Calculate jitter (variance in latency)
            if len(self._latency_samples) > 1:
                diffs = [
                    abs(self._latency_samples[i] - self._latency_samples[i - 1])
                    for i in range(1, len(self._latency_samples))
                ]
                self._stats.jitter_ms = sum(diffs) / len(diffs)

        # Calculate RTT stats
        if self._rtt_samples:
            self._stats.rtt_average_ms = sum(self._rtt_samples) / len(self._rtt_samples)

        # Calculate packet loss percentage
        total_packets = self._stats.packets_sent + self._stats.packets_received
        if total_packets > 0:
            self._stats.packet_loss_percent = (
                self._stats.packets_dropped / total_packets * 100
            )

        # Calculate connection quality (0-1) using config thresholds
        quality = 1.0
        cfg = self._config

        # Packet loss penalty
        if self._stats.packet_loss_percent > 0:
            penalty = self._stats.packet_loss_percent / cfg.packet_loss_quality_divisor
            quality -= min(cfg.max_packet_loss_quality_penalty, penalty)

        # Latency penalty
        if self._stats.average_latency_ms > cfg.latency_quality_threshold_ms:
            excess_latency = self._stats.average_latency_ms - cfg.latency_quality_threshold_ms
            penalty_ratio = excess_latency / cfg.latency_quality_max_penalty_ms
            quality -= min(cfg.max_latency_quality_penalty, penalty_ratio * cfg.max_latency_quality_penalty)

        # Jitter penalty
        if self._stats.jitter_ms > cfg.jitter_quality_threshold_ms:
            excess_jitter = self._stats.jitter_ms - cfg.jitter_quality_threshold_ms
            penalty_ratio = excess_jitter / cfg.jitter_quality_max_penalty_ms
            quality -= min(cfg.max_jitter_quality_penalty, penalty_ratio * cfg.max_jitter_quality_penalty)

        self._stats.connection_quality = max(0.0, quality)

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._stats = NetworkStats()
        self._latency_samples.clear()
        self._rtt_samples.clear()
        self._bytes_sent_window.clear()
        self._bytes_received_window.clear()
        self._recent_packets.clear()

    # =========================================================================
    # Callbacks
    # =========================================================================

    def add_stat_callback(
        self,
        callback: Callable[[NetworkStats], None],
    ) -> None:
        """Add a callback for stats updates."""
        self._stat_callbacks.append(callback)

    def remove_stat_callback(
        self,
        callback: Callable[[NetworkStats], None],
    ) -> bool:
        """Remove a stats callback."""
        try:
            self._stat_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def notify_stats(self) -> None:
        """Notify stat callbacks with current stats."""
        stats = self.get_stats()
        for callback in self._stat_callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.error("Stats callback error: %s", e)

    # =========================================================================
    # Console Commands
    # =========================================================================

    def cmd_net_latency(self, ms: float) -> str:
        """Console command: net.latency <ms>"""
        self.set_latency(ms)
        return f"Network latency set to {int(ms)}ms"

    def cmd_net_loss(self, percent: float) -> str:
        """Console command: net.loss <percent>"""
        self.set_packet_loss(percent)
        return f"Packet loss set to {percent:.1f}%"

    def cmd_net_jitter(self, ms: float) -> str:
        """Console command: net.jitter <ms>"""
        self.set_jitter(ms)
        return f"Jitter set to {int(ms)}ms"

    def cmd_net_bandwidth(self, kbps: float) -> str:
        """Console command: net.bandwidth <kbps>"""
        self.set_bandwidth_limit(kbps)
        if kbps > 0:
            return f"Bandwidth limited to {kbps:.1f} kbps"
        return "Bandwidth unlimited"

    def cmd_net_stats(self) -> str:
        """Console command: net.stats"""
        stats = self.get_stats()
        return (
            f"Network Stats:\n"
            f"  Sent: {stats.bytes_sent} bytes ({stats.bytes_per_second_sent:.1f} B/s)\n"
            f"  Received: {stats.bytes_received} bytes ({stats.bytes_per_second_received:.1f} B/s)\n"
            f"  Packets: {stats.packets_sent} sent, {stats.packets_received} recv, {stats.packets_dropped} dropped\n"
            f"  Latency: {stats.current_latency_ms:.1f}ms (avg: {stats.average_latency_ms:.1f}ms)\n"
            f"  RTT: {stats.rtt_ms:.1f}ms (avg: {stats.rtt_average_ms:.1f}ms)\n"
            f"  Jitter: {stats.jitter_ms:.1f}ms\n"
            f"  Loss: {stats.packet_loss_percent:.1f}%\n"
            f"  Quality: {stats.connection_quality * 100:.0f}%"
        )

    def cmd_net_reset(self) -> str:
        """Console command: net.reset"""
        self.reset_simulation()
        self.reset_stats()
        return "Network simulation and stats reset"


# =============================================================================
# Singleton instance
# =============================================================================

_network_debugger: Optional[NetworkDebugger] = None


def get_network_debugger() -> NetworkDebugger:
    """Get the global network debugger instance."""
    global _network_debugger
    if _network_debugger is None:
        _network_debugger = NetworkDebugger()
    return _network_debugger


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "get_network_debugger",
    "NetworkDebugConfig",
    "NetworkDebugger",
    "NetworkSimulation",
    "NetworkStats",
    "PacketDirection",
    "PacketLog",
]
