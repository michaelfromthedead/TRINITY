"""
Client view time calculation for lag compensation.

When a client fires a weapon, they see other players at a delayed
position due to network latency. This module calculates what time
the client was viewing when they took the shot, so the server can
rewind to that time for hit detection.

View time accounts for:
- Round-trip time (RTT) / 2 for one-way latency
- Interpolation delay on the client
- Jitter smoothing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import math

from engine.networking.config import (
    DEFAULT_MAX_LAG_COMPENSATION_MS,
    DEFAULT_CLIENT_INTERPOLATION_DELAY_MS,
    DEFAULT_JITTER_BUFFER_MS,
    DEFAULT_MIN_RTT_SAMPLES,
    DEFAULT_RTT_HISTORY_SIZE,
    JITTER_STANDARD_DEVIATIONS,
    DEFAULT_MAX_VIEW_TIME_DEVIATION_MS,
    DEFAULT_SUSPICIOUS_THRESHOLD,
)


@dataclass
class RTTSample:
    """A single RTT measurement sample."""

    rtt_ms: float
    """Round-trip time in milliseconds."""

    timestamp: float
    """When this sample was taken."""


@dataclass
class ViewTimeConfig:
    """Configuration for view time calculation."""

    max_lag_compensation_ms: float = DEFAULT_MAX_LAG_COMPENSATION_MS
    """Maximum lag compensation allowed (milliseconds)."""

    interpolation_delay_ms: float = DEFAULT_CLIENT_INTERPOLATION_DELAY_MS
    """Client-side interpolation delay (milliseconds)."""

    jitter_buffer_ms: float = DEFAULT_JITTER_BUFFER_MS
    """Additional buffer for jitter (milliseconds)."""

    min_rtt_samples: int = DEFAULT_MIN_RTT_SAMPLES
    """Minimum RTT samples before using average."""

    rtt_history_size: int = DEFAULT_RTT_HISTORY_SIZE
    """Number of RTT samples to keep."""


def calculate_client_view_time(
    server_time: float,
    client_rtt: float,
    interpolation_delay: float = 0.1,
) -> float:
    """
    Calculate the time the client was viewing when they acted.

    Simple calculation: client sees the world at approximately
    server_time - (RTT/2) - interpolation_delay

    Args:
        server_time: Current server time.
        client_rtt: Client's round-trip time (seconds).
        interpolation_delay: Client's interpolation delay (seconds).

    Returns:
        The server time that the client was viewing.
    """
    one_way_latency = client_rtt / 2.0
    return server_time - one_way_latency - interpolation_delay


class ViewTimeCalculator:
    """
    Calculates client view time with jitter compensation.

    Maintains a history of RTT measurements and provides smoothed
    view time calculations that account for network jitter.

    Example:
        calculator = ViewTimeCalculator(client_id=1)

        # When receiving packets, update RTT:
        calculator.add_rtt_sample(measured_rtt)

        # When client fires:
        view_time = calculator.get_interpolated_view_time(server_time)
        # Use view_time for lag compensation rewind
    """

    def __init__(
        self,
        client_id: int = 0,
        config: Optional[ViewTimeConfig] = None,
    ) -> None:
        """
        Initialize the view time calculator.

        Args:
            client_id: Identifier for the client.
            config: Configuration options.
        """
        self.client_id = client_id
        self._config = config or ViewTimeConfig()

        self._rtt_history: deque[RTTSample] = deque(
            maxlen=self._config.rtt_history_size
        )

        # Cached calculations
        self._average_rtt: float = 0.0
        self._rtt_variance: float = 0.0
        self._min_rtt: float = float('inf')
        self._max_rtt: float = 0.0

    @property
    def rtt_history(self) -> List[RTTSample]:
        """Get the RTT history."""
        return list(self._rtt_history)

    @property
    def average_rtt(self) -> float:
        """Get the average RTT in milliseconds."""
        return self._average_rtt

    @property
    def rtt_variance(self) -> float:
        """Get the RTT variance (jitter indicator)."""
        return self._rtt_variance

    @property
    def jitter(self) -> float:
        """Get estimated jitter (standard deviation of RTT)."""
        return math.sqrt(self._rtt_variance) if self._rtt_variance > 0 else 0.0

    @property
    def min_rtt(self) -> float:
        """Get minimum observed RTT."""
        return self._min_rtt if self._min_rtt != float('inf') else 0.0

    @property
    def max_rtt(self) -> float:
        """Get maximum observed RTT."""
        return self._max_rtt

    def add_rtt_sample(
        self,
        rtt_ms: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Add a new RTT measurement.

        Args:
            rtt_ms: Measured round-trip time in milliseconds.
            timestamp: Optional timestamp for this sample.
        """
        sample = RTTSample(
            rtt_ms=rtt_ms,
            timestamp=timestamp or 0.0,
        )
        self._rtt_history.append(sample)

        # Update statistics
        self._update_statistics()

    def _update_statistics(self) -> None:
        """Update cached RTT statistics."""
        if not self._rtt_history:
            return

        rtts = [s.rtt_ms for s in self._rtt_history]

        # Calculate average
        self._average_rtt = sum(rtts) / len(rtts)

        # Calculate variance
        if len(rtts) >= 2:
            variance_sum = sum((r - self._average_rtt) ** 2 for r in rtts)
            self._rtt_variance = variance_sum / (len(rtts) - 1)
        else:
            self._rtt_variance = 0.0

        # Update min/max
        self._min_rtt = min(rtts)
        self._max_rtt = max(rtts)

    def get_interpolated_view_time(
        self,
        server_time: float,
        use_jitter_compensation: bool = True,
    ) -> float:
        """
        Calculate the client's view time with jitter compensation.

        Uses smoothed RTT and adds jitter buffer for more accurate
        lag compensation.

        Args:
            server_time: Current server time.
            use_jitter_compensation: Whether to add jitter buffer.

        Returns:
            The server time the client was viewing.
        """
        # Get effective RTT (use average if enough samples)
        if len(self._rtt_history) >= self._config.min_rtt_samples:
            effective_rtt_ms = self._average_rtt
        elif self._rtt_history:
            # Use latest sample
            effective_rtt_ms = self._rtt_history[-1].rtt_ms
        else:
            # No RTT data - assume minimal latency
            effective_rtt_ms = 0.0

        # Calculate base view time
        one_way_latency_ms = effective_rtt_ms / 2.0
        interpolation_delay_ms = self._config.interpolation_delay_ms

        total_delay_ms = one_way_latency_ms + interpolation_delay_ms

        # Add jitter compensation
        if use_jitter_compensation and len(self._rtt_history) >= 2:
            jitter_buffer = min(
                self.jitter * JITTER_STANDARD_DEVIATIONS,
                self._config.jitter_buffer_ms,
            )
            total_delay_ms += jitter_buffer

        # Clamp to max lag compensation
        total_delay_ms = min(total_delay_ms, self._config.max_lag_compensation_ms)

        # Convert to seconds and calculate view time
        total_delay_s = total_delay_ms / 1000.0
        view_time = server_time - total_delay_s

        return view_time

    def get_conservative_view_time(self, server_time: float) -> float:
        """
        Get a conservative (defender-favored) view time.

        Uses minimum observed RTT for stricter hit detection.

        Args:
            server_time: Current server time.

        Returns:
            Conservative view time estimate.
        """
        one_way_latency_ms = self.min_rtt / 2.0
        interpolation_delay_ms = self._config.interpolation_delay_ms

        total_delay_s = (one_way_latency_ms + interpolation_delay_ms) / 1000.0
        return server_time - total_delay_s

    def get_liberal_view_time(self, server_time: float) -> float:
        """
        Get a liberal (shooter-favored) view time.

        Uses maximum observed RTT for more lenient hit detection.

        Args:
            server_time: Current server time.

        Returns:
            Liberal view time estimate.
        """
        one_way_latency_ms = self.max_rtt / 2.0
        interpolation_delay_ms = self._config.interpolation_delay_ms
        jitter_buffer_ms = self._config.jitter_buffer_ms

        total_delay_ms = one_way_latency_ms + interpolation_delay_ms + jitter_buffer_ms
        total_delay_ms = min(total_delay_ms, self._config.max_lag_compensation_ms)

        total_delay_s = total_delay_ms / 1000.0
        return server_time - total_delay_s

    def get_view_time_range(
        self,
        server_time: float,
    ) -> Tuple[float, float]:
        """
        Get the range of possible view times.

        Useful for checking hits within an uncertainty window.

        Args:
            server_time: Current server time.

        Returns:
            Tuple of (conservative_time, liberal_time).
        """
        return (
            self.get_conservative_view_time(server_time),
            self.get_liberal_view_time(server_time),
        )

    def get_compensation_amount(self) -> float:
        """
        Get the current lag compensation amount in milliseconds.

        Returns:
            Total compensation time in milliseconds.
        """
        if len(self._rtt_history) >= self._config.min_rtt_samples:
            effective_rtt = self._average_rtt
        elif self._rtt_history:
            effective_rtt = self._rtt_history[-1].rtt_ms
        else:
            return 0.0

        return (effective_rtt / 2.0) + self._config.interpolation_delay_ms

    def is_within_compensation_limit(self) -> bool:
        """
        Check if current latency is within acceptable limits.

        Returns:
            True if lag compensation can be applied.
        """
        return self.get_compensation_amount() <= self._config.max_lag_compensation_ms

    def reset(self) -> None:
        """Reset all RTT history and statistics."""
        self._rtt_history.clear()
        self._average_rtt = 0.0
        self._rtt_variance = 0.0
        self._min_rtt = float('inf')
        self._max_rtt = 0.0


class LagCompensationValidator:
    """
    Validates lag compensation requests for anti-cheat purposes.

    Ensures that lag compensation claims are reasonable and
    consistent with the client's connection history.
    """

    def __init__(
        self,
        max_deviation_ms: float = DEFAULT_MAX_VIEW_TIME_DEVIATION_MS,
        suspicious_threshold: int = DEFAULT_SUSPICIOUS_THRESHOLD,
    ) -> None:
        """
        Initialize the validator.

        Args:
            max_deviation_ms: Maximum allowed deviation from expected.
            suspicious_threshold: Violations before flagging.
        """
        self._max_deviation_ms = max_deviation_ms
        self._suspicious_threshold = suspicious_threshold
        self._calculators: Dict[int, ViewTimeCalculator] = {}
        self._violation_counts: Dict[int, int] = {}

    def register_client(
        self,
        client_id: int,
        config: Optional[ViewTimeConfig] = None,
    ) -> ViewTimeCalculator:
        """Register a client and create their calculator."""
        calculator = ViewTimeCalculator(client_id=client_id, config=config)
        self._calculators[client_id] = calculator
        self._violation_counts[client_id] = 0
        return calculator

    def validate_view_time_claim(
        self,
        client_id: int,
        claimed_view_time: float,
        server_time: float,
    ) -> Tuple[bool, float]:
        """
        Validate a client's view time claim.

        Args:
            client_id: The claiming client.
            claimed_view_time: Time the client claims to have been viewing.
            server_time: Current server time.

        Returns:
            Tuple of (is_valid, corrected_view_time).
        """
        calculator = self._calculators.get(client_id)
        if calculator is None:
            return False, server_time

        # Calculate expected view time
        expected = calculator.get_interpolated_view_time(server_time)
        conservative, liberal = calculator.get_view_time_range(server_time)

        # Check if claim is within acceptable range
        deviation_s = abs(claimed_view_time - expected)
        deviation_ms = deviation_s * 1000.0

        if deviation_ms <= self._max_deviation_ms:
            # Valid claim
            return True, claimed_view_time

        # Check if within liberal range
        if conservative <= claimed_view_time <= liberal:
            return True, claimed_view_time

        # Invalid claim - record violation
        self._violation_counts[client_id] = self._violation_counts.get(client_id, 0) + 1

        # Return corrected view time
        return False, expected

    def is_suspicious(self, client_id: int) -> bool:
        """Check if client has too many violations."""
        return self._violation_counts.get(client_id, 0) >= self._suspicious_threshold

    def reset_violations(self, client_id: int) -> None:
        """Reset violation count for a client."""
        self._violation_counts[client_id] = 0

    def remove_client(self, client_id: int) -> None:
        """Remove a client from tracking."""
        self._calculators.pop(client_id, None)
        self._violation_counts.pop(client_id, None)
