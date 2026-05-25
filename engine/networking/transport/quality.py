"""
Network quality monitoring and adaptation.

Provides metrics collection, quality level classification,
and adaptive settings based on network conditions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple

from ..config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class QualityLevel(IntEnum):
    """Network quality classification."""
    EXCELLENT = 4  # RTT < QUALITY_RTT_EXCELLENT, loss < QUALITY_LOSS_EXCELLENT
    GOOD = 3       # RTT < QUALITY_RTT_GOOD, loss < QUALITY_LOSS_GOOD
    FAIR = 2       # RTT < QUALITY_RTT_FAIR, loss < QUALITY_LOSS_FAIR
    POOR = 1       # RTT < QUALITY_RTT_POOR, loss < QUALITY_LOSS_POOR
    CRITICAL = 0   # RTT >= QUALITY_RTT_POOR or loss >= QUALITY_LOSS_POOR


@dataclass
class QualityMetrics:
    """
    Network quality metrics.

    Attributes:
        rtt: Round-trip time in seconds.
        rtt_variance: RTT variance for jitter calculation.
        jitter: Network jitter in seconds.
        packet_loss: Packet loss ratio (0-1).
        bandwidth_up: Upload bandwidth estimate (bytes/sec).
        bandwidth_down: Download bandwidth estimate (bytes/sec).
        timestamp: When metrics were last updated.
    """
    rtt: float = 0.0
    rtt_variance: float = 0.0
    jitter: float = 0.0
    packet_loss: float = 0.0
    bandwidth_up: float = 0.0
    bandwidth_down: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def quality_level(self) -> QualityLevel:
        """Determine quality level from metrics."""
        if self.packet_loss >= DEFAULT_CONFIG.QUALITY_LOSS_POOR or self.rtt >= DEFAULT_CONFIG.QUALITY_RTT_POOR:
            return QualityLevel.CRITICAL
        elif self.packet_loss >= DEFAULT_CONFIG.QUALITY_LOSS_FAIR or self.rtt >= DEFAULT_CONFIG.QUALITY_RTT_FAIR:
            return QualityLevel.POOR
        elif self.packet_loss >= DEFAULT_CONFIG.QUALITY_LOSS_GOOD or self.rtt >= DEFAULT_CONFIG.QUALITY_RTT_GOOD:
            return QualityLevel.FAIR
        elif self.packet_loss >= DEFAULT_CONFIG.QUALITY_LOSS_EXCELLENT or self.rtt >= DEFAULT_CONFIG.QUALITY_RTT_EXCELLENT:
            return QualityLevel.GOOD
        else:
            return QualityLevel.EXCELLENT

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'rtt': self.rtt,
            'rtt_variance': self.rtt_variance,
            'jitter': self.jitter,
            'packet_loss': self.packet_loss,
            'bandwidth_up': self.bandwidth_up,
            'bandwidth_down': self.bandwidth_down,
            'quality_level': int(self.quality_level),
        }


class QualityMonitor:
    """
    Monitors network quality over time.

    Collects RTT samples, tracks packet loss, and calculates
    smoothed quality metrics.

    Example:
        monitor = QualityMonitor()

        # Add RTT sample when ACK received
        monitor.add_rtt_sample(0.045)

        # Track packet loss
        monitor.record_packet_sent()
        monitor.record_packet_received()

        # Get current metrics
        metrics = monitor.get_metrics()
        print(f"RTT: {metrics.rtt*1000:.1f}ms, Quality: {metrics.quality_level.name}")
    """

    def __init__(
        self,
        window_size: int = DEFAULT_CONFIG.QUALITY_WINDOW_SIZE,
        smoothing_factor: float = DEFAULT_CONFIG.QUALITY_SMOOTHING_FACTOR
    ) -> None:
        """
        Initialize the quality monitor.

        Args:
            window_size: Number of samples to keep for statistics.
            smoothing_factor: EWMA smoothing factor (0-1).
        """
        self._window_size = window_size
        self._smoothing = smoothing_factor

        # RTT tracking
        self._rtt_samples: List[float] = []
        self._rtt_estimate = 0.0
        self._rtt_variance = 0.0

        # Packet loss tracking
        self._packets_sent = 0
        self._packets_received = 0
        self._packets_lost = 0
        self._loss_window_sent = 0
        self._loss_window_received = 0

        # Bandwidth tracking
        self._bytes_sent: List[Tuple[float, int]] = []
        self._bytes_received: List[Tuple[float, int]] = []
        self._bandwidth_window = DEFAULT_CONFIG.BANDWIDTH_WINDOW

        # Quality history
        self._quality_history: List[QualityLevel] = []
        self._quality_change_callbacks: List[Callable[[QualityLevel, QualityLevel], None]] = []

        # Current metrics
        self._metrics = QualityMetrics()
        self._last_update = time.time()

    def add_rtt_sample(self, rtt: float) -> None:
        """
        Add an RTT sample.

        Args:
            rtt: Round-trip time in seconds.
        """
        # Store sample
        self._rtt_samples.append(rtt)
        if len(self._rtt_samples) > self._window_size:
            self._rtt_samples.pop(0)

        # Update EWMA estimate
        if self._rtt_estimate == 0.0:
            self._rtt_estimate = rtt
        else:
            diff = abs(rtt - self._rtt_estimate)
            self._rtt_variance = (1 - self._smoothing) * self._rtt_variance + self._smoothing * diff
            self._rtt_estimate = (1 - self._smoothing) * self._rtt_estimate + self._smoothing * rtt

    def record_packet_sent(self, bytes_count: int = 0) -> None:
        """
        Record a sent packet.

        Args:
            bytes_count: Size of packet in bytes.
        """
        self._packets_sent += 1
        self._loss_window_sent += 1

        if bytes_count > 0:
            now = time.time()
            self._bytes_sent.append((now, bytes_count))
            self._prune_bandwidth_samples()

    def record_packet_received(self, bytes_count: int = 0) -> None:
        """
        Record a received packet.

        Args:
            bytes_count: Size of packet in bytes.
        """
        self._packets_received += 1
        self._loss_window_received += 1

        if bytes_count > 0:
            now = time.time()
            self._bytes_received.append((now, bytes_count))
            self._prune_bandwidth_samples()

    def record_packet_lost(self) -> None:
        """Record a lost packet."""
        self._packets_lost += 1

    def _prune_bandwidth_samples(self) -> None:
        """Remove old bandwidth samples."""
        now = time.time()
        cutoff = now - self._bandwidth_window

        self._bytes_sent = [(t, b) for t, b in self._bytes_sent if t > cutoff]
        self._bytes_received = [(t, b) for t, b in self._bytes_received if t > cutoff]

    def update(self) -> QualityMetrics:
        """
        Update and return current metrics.

        Returns:
            Current quality metrics.
        """
        now = time.time()

        # Calculate packet loss
        if self._loss_window_sent > 0:
            expected = self._loss_window_sent
            actual = self._loss_window_received
            loss = max(0, expected - actual) / expected
        else:
            loss = 0.0

        # Reset loss window periodically
        if self._loss_window_sent >= DEFAULT_CONFIG.LOSS_WINDOW_RESET_THRESHOLD:
            self._loss_window_sent = 0
            self._loss_window_received = 0

        # Calculate bandwidth
        self._prune_bandwidth_samples()
        bandwidth_up = sum(b for _, b in self._bytes_sent) / max(0.001, self._bandwidth_window)
        bandwidth_down = sum(b for _, b in self._bytes_received) / max(0.001, self._bandwidth_window)

        # Calculate jitter from RTT variance
        jitter = self._rtt_variance

        # Update metrics
        old_level = self._metrics.quality_level

        self._metrics = QualityMetrics(
            rtt=self._rtt_estimate,
            rtt_variance=self._rtt_variance,
            jitter=jitter,
            packet_loss=loss,
            bandwidth_up=bandwidth_up,
            bandwidth_down=bandwidth_down,
            timestamp=now
        )

        # Check for quality level change
        new_level = self._metrics.quality_level
        if new_level != old_level:
            self._quality_history.append(new_level)
            for callback in self._quality_change_callbacks:
                callback(old_level, new_level)

        self._last_update = now
        return self._metrics

    def get_metrics(self) -> QualityMetrics:
        """Get current metrics (without updating)."""
        return self._metrics

    def get_quality_level(self) -> QualityLevel:
        """Get current quality level."""
        return self._metrics.quality_level

    def on_quality_change(
        self,
        callback: Callable[[QualityLevel, QualityLevel], None]
    ) -> None:
        """
        Register callback for quality level changes.

        Args:
            callback: Function(old_level, new_level) to call on change.
        """
        self._quality_change_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, float]:
        """Get detailed statistics."""
        return {
            'rtt_current': self._rtt_estimate,
            'rtt_min': min(self._rtt_samples) if self._rtt_samples else 0.0,
            'rtt_max': max(self._rtt_samples) if self._rtt_samples else 0.0,
            'rtt_avg': sum(self._rtt_samples) / len(self._rtt_samples) if self._rtt_samples else 0.0,
            'jitter': self._rtt_variance,
            'packet_loss': self._metrics.packet_loss,
            'packets_sent': self._packets_sent,
            'packets_received': self._packets_received,
            'packets_lost': self._packets_lost,
            'bandwidth_up': self._metrics.bandwidth_up,
            'bandwidth_down': self._metrics.bandwidth_down,
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self._rtt_samples.clear()
        self._rtt_estimate = 0.0
        self._rtt_variance = 0.0
        self._packets_sent = 0
        self._packets_received = 0
        self._packets_lost = 0
        self._loss_window_sent = 0
        self._loss_window_received = 0
        self._bytes_sent.clear()
        self._bytes_received.clear()
        self._quality_history.clear()
        self._metrics = QualityMetrics()


@dataclass
class AdaptiveSettings:
    """Settings that can be adapted based on network quality."""
    update_rate: float = DEFAULT_CONFIG.UPDATE_RATE_FAIR  # Updates per second
    compression_level: int = DEFAULT_CONFIG.COMPRESSION_LEVEL  # zlib level (1-9)
    delta_compression: bool = True
    interpolation_delay: float = DEFAULT_CONFIG.CHANNEL_INITIAL_RTT  # seconds
    extrapolation_limit: float = DEFAULT_CONFIG.EXTRAPOLATION_LIMIT_FAIR  # seconds
    packet_aggregation: bool = False
    priority_queue: bool = True


class NetworkQualityAdapter:
    """
    Adapts network settings based on quality.

    Automatically adjusts update rates, compression, and other
    settings to maintain playability under varying conditions.

    Example:
        adapter = NetworkQualityAdapter()

        # Update with current quality
        settings = adapter.adapt(quality_metrics)

        # Apply settings
        network.set_update_rate(settings.update_rate)
        network.set_compression(settings.compression_level)
    """

    # Quality-based presets
    PRESETS = {
        QualityLevel.EXCELLENT: AdaptiveSettings(
            update_rate=DEFAULT_CONFIG.UPDATE_RATE_EXCELLENT,
            compression_level=DEFAULT_CONFIG.COMPRESSION_LEVEL_EXCELLENT,
            delta_compression=True,
            interpolation_delay=DEFAULT_CONFIG.INTERPOLATION_DELAY_EXCELLENT,
            extrapolation_limit=DEFAULT_CONFIG.EXTRAPOLATION_LIMIT_EXCELLENT,
            packet_aggregation=False,
            priority_queue=False,
        ),
        QualityLevel.GOOD: AdaptiveSettings(
            update_rate=DEFAULT_CONFIG.UPDATE_RATE_GOOD,
            compression_level=DEFAULT_CONFIG.COMPRESSION_LEVEL_GOOD,
            delta_compression=True,
            interpolation_delay=DEFAULT_CONFIG.INTERPOLATION_DELAY_GOOD,
            extrapolation_limit=DEFAULT_CONFIG.EXTRAPOLATION_LIMIT_GOOD,
            packet_aggregation=False,
            priority_queue=True,
        ),
        QualityLevel.FAIR: AdaptiveSettings(
            update_rate=DEFAULT_CONFIG.UPDATE_RATE_FAIR,
            compression_level=DEFAULT_CONFIG.COMPRESSION_LEVEL_FAIR,
            delta_compression=True,
            interpolation_delay=DEFAULT_CONFIG.INTERPOLATION_DELAY_FAIR,
            extrapolation_limit=DEFAULT_CONFIG.EXTRAPOLATION_LIMIT_FAIR,
            packet_aggregation=True,
            priority_queue=True,
        ),
        QualityLevel.POOR: AdaptiveSettings(
            update_rate=DEFAULT_CONFIG.UPDATE_RATE_POOR,
            compression_level=DEFAULT_CONFIG.COMPRESSION_LEVEL_POOR,
            delta_compression=True,
            interpolation_delay=DEFAULT_CONFIG.INTERPOLATION_DELAY_POOR,
            extrapolation_limit=DEFAULT_CONFIG.EXTRAPOLATION_LIMIT_POOR,
            packet_aggregation=True,
            priority_queue=True,
        ),
        QualityLevel.CRITICAL: AdaptiveSettings(
            update_rate=DEFAULT_CONFIG.UPDATE_RATE_CRITICAL,
            compression_level=DEFAULT_CONFIG.COMPRESSION_LEVEL_CRITICAL,
            delta_compression=True,
            interpolation_delay=DEFAULT_CONFIG.INTERPOLATION_DELAY_CRITICAL,
            extrapolation_limit=DEFAULT_CONFIG.EXTRAPOLATION_LIMIT_CRITICAL,
            packet_aggregation=True,
            priority_queue=True,
        ),
    }

    def __init__(
        self,
        hysteresis_threshold: float = DEFAULT_CONFIG.ADAPTER_HYSTERESIS_THRESHOLD,
        adaptation_delay: float = DEFAULT_CONFIG.ADAPTER_ADAPTATION_DELAY
    ) -> None:
        """
        Initialize the adapter.

        Args:
            hysteresis_threshold: Time (seconds) quality must be stable before changing.
            adaptation_delay: Minimum time between adaptations.
        """
        self._hysteresis = hysteresis_threshold
        self._adaptation_delay = adaptation_delay

        self._current_level = QualityLevel.GOOD
        self._current_settings = self.PRESETS[QualityLevel.GOOD]
        self._level_stable_since = time.time()
        self._last_adaptation = 0.0

        # Custom overrides
        self._min_update_rate = DEFAULT_CONFIG.MIN_UPDATE_RATE
        self._max_update_rate = DEFAULT_CONFIG.MAX_UPDATE_RATE

    @property
    def current_level(self) -> QualityLevel:
        """Get current quality level."""
        return self._current_level

    @property
    def current_settings(self) -> AdaptiveSettings:
        """Get current adaptive settings."""
        return self._current_settings

    def adapt(self, metrics: QualityMetrics) -> AdaptiveSettings:
        """
        Adapt settings based on metrics.

        Args:
            metrics: Current quality metrics.

        Returns:
            Adapted settings.
        """
        now = time.time()
        new_level = metrics.quality_level

        # Track level immediately on change
        if new_level != self._current_level:
            self._current_level = new_level
            self._level_stable_since = now
            return self._current_settings

        # Check hysteresis
        time_at_level = now - self._level_stable_since
        if time_at_level < self._hysteresis:
            return self._current_settings

        # Check adaptation delay
        if now - self._last_adaptation < self._adaptation_delay:
            return self._current_settings

        # Apply settings for the tracked level
        self._current_settings = self._create_settings(self._current_level, metrics)
        self._last_adaptation = now

        return self._current_settings

    def _create_settings(
        self,
        level: QualityLevel,
        metrics: QualityMetrics
    ) -> AdaptiveSettings:
        """Create settings for a quality level."""
        base = self.PRESETS[level]

        # Fine-tune based on specific metrics
        settings = AdaptiveSettings(
            update_rate=self._calculate_update_rate(base.update_rate, metrics),
            compression_level=base.compression_level,
            delta_compression=base.delta_compression,
            interpolation_delay=self._calculate_interpolation_delay(metrics),
            extrapolation_limit=base.extrapolation_limit,
            packet_aggregation=base.packet_aggregation,
            priority_queue=base.priority_queue,
        )

        return settings

    def _calculate_update_rate(
        self,
        base_rate: float,
        metrics: QualityMetrics
    ) -> float:
        """Calculate optimal update rate."""
        rate = base_rate

        # Reduce rate if bandwidth is low
        if metrics.bandwidth_up > 0:
            # Assume ~ASSUMED_BYTES_PER_UPDATE bytes per update
            max_rate = metrics.bandwidth_up / DEFAULT_CONFIG.ASSUMED_BYTES_PER_UPDATE
            rate = min(rate, max_rate)

        # Clamp to limits
        return max(self._min_update_rate, min(self._max_update_rate, rate))

    def _calculate_interpolation_delay(self, metrics: QualityMetrics) -> float:
        """Calculate optimal interpolation delay."""
        # Base on RTT + jitter buffer
        delay = metrics.rtt + metrics.jitter * 2

        # Clamp to reasonable range
        return max(DEFAULT_CONFIG.MIN_INTERPOLATION_DELAY, min(DEFAULT_CONFIG.MAX_INTERPOLATION_DELAY, delay))

    def set_update_rate_limits(self, min_rate: float, max_rate: float) -> None:
        """Set update rate limits."""
        self._min_update_rate = min_rate
        self._max_update_rate = max_rate

    def force_level(self, level: QualityLevel) -> AdaptiveSettings:
        """Force a specific quality level."""
        self._current_level = level
        self._current_settings = self.PRESETS[level]
        return self._current_settings

    def reset(self) -> None:
        """Reset to default state."""
        self._current_level = QualityLevel.GOOD
        self._current_settings = self.PRESETS[QualityLevel.GOOD]
        self._level_stable_since = time.time()
        self._last_adaptation = 0.0
