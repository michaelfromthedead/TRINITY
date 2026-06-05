"""Frame Budget System with Automatic Quality Adjustment (T-CC-3.10).

Provides frame budget tracking with automatic quality tier adjustment based on
performance. Integrates with the GPU timestamp profiler and quality tier system
to maintain target frame rates through dynamic quality scaling.

Key Components:
    - FrameBudget: Tracks target frame time and violation state
    - BudgetViolationDetector: Detects consecutive budget violations
    - AutoQualityAdjuster: Triggers tier changes based on performance
    - FrameBudgetManager: Coordinates all components for seamless operation

Design Goals:
    - Smooth transitions without oscillation
    - Configurable thresholds for both directions
    - Hysteresis to prevent rapid switching
    - Integration with existing quality tier system

Example:
    manager = FrameBudgetManager(quality_manager)
    manager.configure(target_fps=60, violation_threshold=10, recovery_threshold=60)

    # Each frame:
    manager.record_frame_time(frame_time_ms)

    # Quality tier is automatically adjusted when thresholds are crossed
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from trinity.types import QualityTier

if TYPE_CHECKING:
    from engine.rendering.quality.quality_manager import QualityManager


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default target frame times for common refresh rates
TARGET_FRAME_TIME_60FPS_MS = 16.67
TARGET_FRAME_TIME_30FPS_MS = 33.33
TARGET_FRAME_TIME_120FPS_MS = 8.33
TARGET_FRAME_TIME_144FPS_MS = 6.94

# Default thresholds
DEFAULT_VIOLATION_THRESHOLD = 10  # Frames before downgrade
DEFAULT_RECOVERY_THRESHOLD = 60  # Frames before upgrade
DEFAULT_SPIKE_TOLERANCE = 3  # Single-frame spikes to ignore
DEFAULT_OVER_BUDGET_MARGIN = 1.2  # 20% over budget triggers violation
DEFAULT_UNDER_BUDGET_MARGIN = 0.8  # 20% under budget allows recovery
DEFAULT_COOLDOWN_FRAMES = 30  # Frames to wait after tier change
DEFAULT_HISTORY_SIZE = 120  # Frames of history to keep


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class BudgetState(Enum):
    """State of frame budget tracking."""

    WITHIN_BUDGET = auto()  # Frame time within target
    OVER_BUDGET = auto()  # Frame time exceeds target
    UNDER_BUDGET = auto()  # Frame time significantly under target
    COOLDOWN = auto()  # Recently changed tier, in cooldown period


class TierTransitionDirection(Enum):
    """Direction of quality tier transition."""

    NONE = auto()
    DOWNGRADE = auto()  # Reduce quality
    UPGRADE = auto()  # Increase quality


class FrameTiming(NamedTuple):
    """Timing data for a single frame."""

    frame_index: int
    frame_time_ms: float
    budget_state: BudgetState
    timestamp: float


@dataclass
class BudgetViolation:
    """Record of a budget violation event."""

    start_frame: int
    end_frame: int
    consecutive_count: int
    average_overage_ms: float
    peak_overage_ms: float
    triggered_downgrade: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class TierTransition:
    """Record of a quality tier transition."""

    frame_index: int
    direction: TierTransitionDirection
    old_tier: QualityTier
    new_tier: QualityTier
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class FrameBudgetConfig:
    """Configuration for frame budget system."""

    target_fps: float = 60.0
    violation_threshold: int = DEFAULT_VIOLATION_THRESHOLD
    recovery_threshold: int = DEFAULT_RECOVERY_THRESHOLD
    spike_tolerance: int = DEFAULT_SPIKE_TOLERANCE
    over_budget_margin: float = DEFAULT_OVER_BUDGET_MARGIN
    under_budget_margin: float = DEFAULT_UNDER_BUDGET_MARGIN
    cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES
    history_size: int = DEFAULT_HISTORY_SIZE
    enabled: bool = True
    auto_adjust: bool = True

    @property
    def target_frame_time_ms(self) -> float:
        """Target frame time in milliseconds."""
        return 1000.0 / self.target_fps

    @property
    def over_budget_threshold_ms(self) -> float:
        """Frame time threshold for violation."""
        return self.target_frame_time_ms * self.over_budget_margin

    @property
    def under_budget_threshold_ms(self) -> float:
        """Frame time threshold for recovery."""
        return self.target_frame_time_ms * self.under_budget_margin


# =============================================================================
# PROTOCOLS
# =============================================================================


class QualityManagerProtocol(Protocol):
    """Protocol for quality manager interface."""

    @property
    def current_tier(self) -> QualityTier:
        """Get current quality tier."""
        ...

    @property
    def base_tier(self) -> QualityTier:
        """Get base quality tier."""
        ...

    def set_tier(self, tier: QualityTier) -> None:
        """Set quality tier."""
        ...

    def add_listener(
        self, listener: Callable[[QualityTier, QualityTier], None]
    ) -> None:
        """Add tier change listener."""
        ...


class FrameTimeSourceProtocol(Protocol):
    """Protocol for frame time data source."""

    def get_frame_time_ms(self) -> float:
        """Get most recent frame time in milliseconds."""
        ...


# =============================================================================
# FRAME BUDGET TRACKER
# =============================================================================


class FrameBudget:
    """Tracks frame time against target budget.

    Maintains current state and history of frame timings, providing
    budget state classification for each frame.

    Attributes:
        config: Budget configuration.
        current_state: Current budget state.
        frame_index: Current frame number.
    """

    def __init__(self, config: Optional[FrameBudgetConfig] = None) -> None:
        """Initialize frame budget tracker.

        Args:
            config: Budget configuration. Uses defaults if not provided.
        """
        self._config = config or FrameBudgetConfig()
        self._current_state = BudgetState.WITHIN_BUDGET
        self._frame_index = 0
        self._history: Deque[FrameTiming] = deque(maxlen=self._config.history_size)
        self._lock = threading.Lock()

        # Running statistics
        self._total_frame_time_ms = 0.0
        self._frame_count = 0
        self._min_frame_time_ms = float('inf')
        self._max_frame_time_ms = 0.0

    @property
    def config(self) -> FrameBudgetConfig:
        """Get budget configuration."""
        return self._config

    @config.setter
    def config(self, value: FrameBudgetConfig) -> None:
        """Set budget configuration."""
        with self._lock:
            self._config = value
            # Resize history if needed
            if len(self._history) > value.history_size:
                new_history: Deque[FrameTiming] = deque(maxlen=value.history_size)
                for item in list(self._history)[-value.history_size:]:
                    new_history.append(item)
                self._history = new_history
            else:
                # Just update maxlen
                old_items = list(self._history)
                self._history = deque(old_items, maxlen=value.history_size)

    @property
    def current_state(self) -> BudgetState:
        """Get current budget state."""
        return self._current_state

    @property
    def frame_index(self) -> int:
        """Get current frame index."""
        return self._frame_index

    @property
    def target_frame_time_ms(self) -> float:
        """Get target frame time in milliseconds."""
        return self._config.target_frame_time_ms

    @property
    def average_frame_time_ms(self) -> float:
        """Get average frame time in milliseconds."""
        if self._frame_count == 0:
            return 0.0
        return self._total_frame_time_ms / self._frame_count

    @property
    def min_frame_time_ms(self) -> float:
        """Get minimum recorded frame time."""
        return self._min_frame_time_ms if self._frame_count > 0 else 0.0

    @property
    def max_frame_time_ms(self) -> float:
        """Get maximum recorded frame time."""
        return self._max_frame_time_ms

    def record_frame(self, frame_time_ms: float) -> FrameTiming:
        """Record frame time and update state.

        Args:
            frame_time_ms: Frame time in milliseconds.

        Returns:
            FrameTiming with frame data and state.
        """
        with self._lock:
            # Classify budget state
            state = self._classify_state(frame_time_ms)
            self._current_state = state

            # Create timing record
            timing = FrameTiming(
                frame_index=self._frame_index,
                frame_time_ms=frame_time_ms,
                budget_state=state,
                timestamp=time.time(),
            )

            # Update history and stats
            self._history.append(timing)
            self._frame_index += 1
            self._total_frame_time_ms += frame_time_ms
            self._frame_count += 1
            self._min_frame_time_ms = min(self._min_frame_time_ms, frame_time_ms)
            self._max_frame_time_ms = max(self._max_frame_time_ms, frame_time_ms)

            return timing

    def _classify_state(self, frame_time_ms: float) -> BudgetState:
        """Classify frame time into budget state.

        Args:
            frame_time_ms: Frame time to classify.

        Returns:
            Budget state for this frame.
        """
        target = self._config.target_frame_time_ms
        over_threshold = self._config.over_budget_threshold_ms
        under_threshold = self._config.under_budget_threshold_ms

        if frame_time_ms > over_threshold:
            return BudgetState.OVER_BUDGET
        elif frame_time_ms < under_threshold:
            return BudgetState.UNDER_BUDGET
        else:
            return BudgetState.WITHIN_BUDGET

    def get_history(
        self, num_frames: Optional[int] = None
    ) -> List[FrameTiming]:
        """Get recent frame history.

        Args:
            num_frames: Number of frames to return. All if None.

        Returns:
            List of recent frame timings.
        """
        with self._lock:
            if num_frames is None:
                return list(self._history)
            return list(self._history)[-num_frames:]

    def get_recent_average(self, num_frames: int = 30) -> float:
        """Get average frame time over recent frames.

        Args:
            num_frames: Number of recent frames to average.

        Returns:
            Average frame time in milliseconds.
        """
        with self._lock:
            recent = list(self._history)[-num_frames:]
            if not recent:
                return 0.0
            return sum(t.frame_time_ms for t in recent) / len(recent)

    def get_percentile(self, percentile: float, num_frames: int = 120) -> float:
        """Get frame time percentile over recent frames.

        Args:
            percentile: Percentile (0-100).
            num_frames: Number of recent frames to consider.

        Returns:
            Frame time at given percentile.
        """
        with self._lock:
            recent = list(self._history)[-num_frames:]
            if not recent:
                return 0.0

            times = sorted(t.frame_time_ms for t in recent)
            index = int(len(times) * percentile / 100)
            index = min(index, len(times) - 1)
            return times[index]

    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self._current_state = BudgetState.WITHIN_BUDGET
            self._frame_index = 0
            self._history.clear()
            self._total_frame_time_ms = 0.0
            self._frame_count = 0
            self._min_frame_time_ms = float('inf')
            self._max_frame_time_ms = 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get budget tracking statistics.

        Returns:
            Dictionary with tracking statistics.
        """
        with self._lock:
            history = list(self._history)
            state_counts = {state: 0 for state in BudgetState}
            for timing in history:
                state_counts[timing.budget_state] += 1

            return {
                "frame_index": self._frame_index,
                "frame_count": self._frame_count,
                "current_state": self._current_state.name,
                "target_fps": self._config.target_fps,
                "target_frame_time_ms": self._config.target_frame_time_ms,
                "average_frame_time_ms": self.average_frame_time_ms,
                "min_frame_time_ms": self.min_frame_time_ms,
                "max_frame_time_ms": self.max_frame_time_ms,
                "history_size": len(history),
                "state_distribution": {
                    state.name: count for state, count in state_counts.items()
                },
            }


# =============================================================================
# BUDGET VIOLATION DETECTOR
# =============================================================================


class BudgetViolationDetector:
    """Detects consecutive budget violations.

    Tracks sequences of over-budget frames and determines when
    the violation threshold has been crossed, triggering a downgrade.

    Attributes:
        config: Budget configuration.
        consecutive_violations: Current count of consecutive violations.
    """

    def __init__(self, config: Optional[FrameBudgetConfig] = None) -> None:
        """Initialize violation detector.

        Args:
            config: Budget configuration.
        """
        self._config = config or FrameBudgetConfig()
        self._consecutive_violations = 0
        self._spike_buffer = 0  # Tolerance for isolated spikes
        self._violation_start_frame = -1
        self._total_overage_ms = 0.0
        self._peak_overage_ms = 0.0
        self._violations: List[BudgetViolation] = []
        self._lock = threading.Lock()

    @property
    def config(self) -> FrameBudgetConfig:
        """Get configuration."""
        return self._config

    @config.setter
    def config(self, value: FrameBudgetConfig) -> None:
        """Set configuration."""
        with self._lock:
            self._config = value

    @property
    def consecutive_violations(self) -> int:
        """Get current consecutive violation count."""
        return self._consecutive_violations

    @property
    def violation_threshold(self) -> int:
        """Get threshold for triggering downgrade."""
        return self._config.violation_threshold

    @property
    def is_violating(self) -> bool:
        """Check if currently in violation state."""
        return self._consecutive_violations > 0

    @property
    def threshold_crossed(self) -> bool:
        """Check if violation threshold has been crossed."""
        return self._consecutive_violations >= self._config.violation_threshold

    def process_frame(self, timing: FrameTiming) -> bool:
        """Process a frame timing and check for threshold crossing.

        Args:
            timing: Frame timing data.

        Returns:
            True if violation threshold was just crossed.
        """
        with self._lock:
            if timing.budget_state == BudgetState.OVER_BUDGET:
                return self._handle_violation(timing)
            else:
                return self._handle_non_violation(timing)

    def _handle_violation(self, timing: FrameTiming) -> bool:
        """Handle an over-budget frame.

        Args:
            timing: Frame timing data.

        Returns:
            True if threshold was just crossed.
        """
        # Calculate overage
        target = self._config.target_frame_time_ms
        overage = timing.frame_time_ms - target

        # Start new violation sequence if needed
        if self._consecutive_violations == 0:
            self._violation_start_frame = timing.frame_index
            self._total_overage_ms = 0.0
            self._peak_overage_ms = 0.0

        # Update stats
        self._consecutive_violations += 1
        self._spike_buffer = self._config.spike_tolerance
        self._total_overage_ms += overage
        self._peak_overage_ms = max(self._peak_overage_ms, overage)

        # Check if we just crossed the threshold
        if self._consecutive_violations == self._config.violation_threshold:
            # Record violation
            violation = BudgetViolation(
                start_frame=self._violation_start_frame,
                end_frame=timing.frame_index,
                consecutive_count=self._consecutive_violations,
                average_overage_ms=self._total_overage_ms / self._consecutive_violations,
                peak_overage_ms=self._peak_overage_ms,
                triggered_downgrade=True,
            )
            self._violations.append(violation)
            return True

        return False

    def _handle_non_violation(self, timing: FrameTiming) -> bool:
        """Handle a non-violation frame.

        Args:
            timing: Frame timing data.

        Returns:
            Always False (no threshold crossing).
        """
        if self._consecutive_violations > 0:
            # Use spike buffer to tolerate isolated good frames
            if self._spike_buffer > 0:
                self._spike_buffer -= 1
            else:
                # Record violation that didn't trigger downgrade
                if self._consecutive_violations > 0:
                    avg_overage = (
                        self._total_overage_ms / self._consecutive_violations
                        if self._consecutive_violations > 0
                        else 0.0
                    )
                    violation = BudgetViolation(
                        start_frame=self._violation_start_frame,
                        end_frame=timing.frame_index - 1,
                        consecutive_count=self._consecutive_violations,
                        average_overage_ms=avg_overage,
                        peak_overage_ms=self._peak_overage_ms,
                        triggered_downgrade=False,
                    )
                    self._violations.append(violation)

                # Reset violation tracking
                self._consecutive_violations = 0
                self._violation_start_frame = -1
                self._total_overage_ms = 0.0
                self._peak_overage_ms = 0.0

        return False

    def reset(self) -> None:
        """Reset violation tracking."""
        with self._lock:
            self._consecutive_violations = 0
            self._spike_buffer = 0
            self._violation_start_frame = -1
            self._total_overage_ms = 0.0
            self._peak_overage_ms = 0.0

    def get_violations(self, num_recent: Optional[int] = None) -> List[BudgetViolation]:
        """Get recorded violations.

        Args:
            num_recent: Number of recent violations to return. All if None.

        Returns:
            List of violation records.
        """
        with self._lock:
            if num_recent is None:
                return list(self._violations)
            return list(self._violations)[-num_recent:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get violation statistics.

        Returns:
            Dictionary with violation statistics.
        """
        with self._lock:
            total_violations = len(self._violations)
            triggered_count = sum(
                1 for v in self._violations if v.triggered_downgrade
            )

            return {
                "consecutive_violations": self._consecutive_violations,
                "violation_threshold": self._config.violation_threshold,
                "is_violating": self.is_violating,
                "threshold_crossed": self.threshold_crossed,
                "total_violation_events": total_violations,
                "triggered_downgrades": triggered_count,
                "spike_tolerance": self._config.spike_tolerance,
                "current_spike_buffer": self._spike_buffer,
            }


# =============================================================================
# RECOVERY TRACKER
# =============================================================================


class RecoveryTracker:
    """Tracks consecutive under-budget frames for tier recovery.

    Monitors when performance is consistently good enough to
    allow upgrading to a higher quality tier.

    Attributes:
        config: Budget configuration.
        consecutive_good_frames: Count of consecutive under-budget frames.
    """

    def __init__(self, config: Optional[FrameBudgetConfig] = None) -> None:
        """Initialize recovery tracker.

        Args:
            config: Budget configuration.
        """
        self._config = config or FrameBudgetConfig()
        self._consecutive_good_frames = 0
        self._recovery_start_frame = -1
        self._total_headroom_ms = 0.0
        self._lock = threading.Lock()

    @property
    def config(self) -> FrameBudgetConfig:
        """Get configuration."""
        return self._config

    @config.setter
    def config(self, value: FrameBudgetConfig) -> None:
        """Set configuration."""
        with self._lock:
            self._config = value

    @property
    def consecutive_good_frames(self) -> int:
        """Get count of consecutive under-budget frames."""
        return self._consecutive_good_frames

    @property
    def recovery_threshold(self) -> int:
        """Get threshold for triggering upgrade."""
        return self._config.recovery_threshold

    @property
    def is_recovering(self) -> bool:
        """Check if in recovery state."""
        return self._consecutive_good_frames > 0

    @property
    def threshold_crossed(self) -> bool:
        """Check if recovery threshold has been crossed."""
        return self._consecutive_good_frames >= self._config.recovery_threshold

    @property
    def recovery_progress(self) -> float:
        """Get progress toward recovery threshold (0-1)."""
        if self._config.recovery_threshold == 0:
            return 1.0
        return min(
            1.0,
            self._consecutive_good_frames / self._config.recovery_threshold
        )

    def process_frame(self, timing: FrameTiming) -> bool:
        """Process a frame timing and check for threshold crossing.

        Args:
            timing: Frame timing data.

        Returns:
            True if recovery threshold was just crossed.
        """
        with self._lock:
            if timing.budget_state == BudgetState.UNDER_BUDGET:
                return self._handle_good_frame(timing)
            else:
                return self._handle_non_recovery_frame(timing)

    def _handle_good_frame(self, timing: FrameTiming) -> bool:
        """Handle an under-budget frame.

        Args:
            timing: Frame timing data.

        Returns:
            True if threshold was just crossed.
        """
        target = self._config.target_frame_time_ms
        headroom = target - timing.frame_time_ms

        if self._consecutive_good_frames == 0:
            self._recovery_start_frame = timing.frame_index
            self._total_headroom_ms = 0.0

        self._consecutive_good_frames += 1
        self._total_headroom_ms += headroom

        return self._consecutive_good_frames == self._config.recovery_threshold

    def _handle_non_recovery_frame(self, timing: FrameTiming) -> bool:
        """Handle a non-recovery frame.

        Args:
            timing: Frame timing data.

        Returns:
            Always False.
        """
        # Decay instead of immediate reset for stability
        if timing.budget_state == BudgetState.WITHIN_BUDGET:
            # Within budget: slow decay
            self._consecutive_good_frames = max(
                0, self._consecutive_good_frames - 1
            )
        else:
            # Over budget: faster decay
            self._consecutive_good_frames = max(
                0, self._consecutive_good_frames - 5
            )

        if self._consecutive_good_frames == 0:
            self._recovery_start_frame = -1
            self._total_headroom_ms = 0.0

        return False

    def reset(self) -> None:
        """Reset recovery tracking."""
        with self._lock:
            self._consecutive_good_frames = 0
            self._recovery_start_frame = -1
            self._total_headroom_ms = 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics.

        Returns:
            Dictionary with recovery statistics.
        """
        with self._lock:
            avg_headroom = (
                self._total_headroom_ms / self._consecutive_good_frames
                if self._consecutive_good_frames > 0
                else 0.0
            )

            return {
                "consecutive_good_frames": self._consecutive_good_frames,
                "recovery_threshold": self._config.recovery_threshold,
                "is_recovering": self.is_recovering,
                "threshold_crossed": self.threshold_crossed,
                "recovery_progress": self.recovery_progress,
                "average_headroom_ms": avg_headroom,
            }


# =============================================================================
# AUTO QUALITY ADJUSTER
# =============================================================================


class AutoQualityAdjuster:
    """Automatically adjusts quality tier based on frame budget.

    Coordinates violation detection and recovery tracking to
    trigger smooth tier transitions with hysteresis to prevent
    oscillation.

    Attributes:
        config: Budget configuration.
        quality_manager: Quality manager for tier control.
    """

    def __init__(
        self,
        quality_manager: Optional[QualityManagerProtocol] = None,
        config: Optional[FrameBudgetConfig] = None,
    ) -> None:
        """Initialize auto quality adjuster.

        Args:
            quality_manager: Quality manager for tier control.
            config: Budget configuration.
        """
        self._quality_manager = quality_manager
        self._config = config or FrameBudgetConfig()
        self._violation_detector = BudgetViolationDetector(self._config)
        self._recovery_tracker = RecoveryTracker(self._config)

        self._cooldown_remaining = 0
        self._last_transition: Optional[TierTransition] = None
        self._transitions: List[TierTransition] = []
        self._frame_index = 0
        self._base_tier: Optional[QualityTier] = None
        self._lock = threading.Lock()

    @property
    def config(self) -> FrameBudgetConfig:
        """Get configuration."""
        return self._config

    @config.setter
    def config(self, value: FrameBudgetConfig) -> None:
        """Set configuration."""
        with self._lock:
            self._config = value
            self._violation_detector.config = value
            self._recovery_tracker.config = value

    @property
    def quality_manager(self) -> Optional[QualityManagerProtocol]:
        """Get quality manager."""
        return self._quality_manager

    @quality_manager.setter
    def quality_manager(self, value: Optional[QualityManagerProtocol]) -> None:
        """Set quality manager."""
        with self._lock:
            self._quality_manager = value
            if value is not None:
                self._base_tier = value.base_tier

    @property
    def current_tier(self) -> QualityTier:
        """Get current quality tier."""
        if self._quality_manager is not None:
            return self._quality_manager.current_tier
        return QualityTier.HIGH

    @property
    def base_tier(self) -> QualityTier:
        """Get base quality tier (maximum allowed)."""
        if self._base_tier is not None:
            return self._base_tier
        if self._quality_manager is not None:
            return self._quality_manager.base_tier
        return QualityTier.ULTRA

    @property
    def in_cooldown(self) -> bool:
        """Check if in cooldown period after tier change."""
        return self._cooldown_remaining > 0

    @property
    def cooldown_remaining(self) -> int:
        """Get remaining cooldown frames."""
        return self._cooldown_remaining

    @property
    def enabled(self) -> bool:
        """Check if auto adjustment is enabled."""
        return self._config.enabled and self._config.auto_adjust

    def process_frame(self, timing: FrameTiming) -> Optional[TierTransition]:
        """Process frame timing and potentially trigger tier change.

        Args:
            timing: Frame timing data.

        Returns:
            TierTransition if a change occurred, None otherwise.
        """
        with self._lock:
            self._frame_index = timing.frame_index

            # Handle cooldown
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                return None

            if not self.enabled:
                return None

            # Check for violations first (downgrade takes priority)
            if self._violation_detector.process_frame(timing):
                return self._try_downgrade(timing, "Budget violation threshold crossed")

            # Check for recovery
            if self._recovery_tracker.process_frame(timing):
                return self._try_upgrade(timing, "Recovery threshold crossed")

            return None

    def _try_downgrade(
        self, timing: FrameTiming, reason: str
    ) -> Optional[TierTransition]:
        """Attempt to downgrade quality tier.

        Args:
            timing: Frame timing data.
            reason: Reason for downgrade.

        Returns:
            TierTransition if successful, None otherwise.
        """
        if self._quality_manager is None:
            return None

        current = self._quality_manager.current_tier

        # Can't downgrade below LOW
        if current == QualityTier.LOW:
            return None

        new_tier = QualityTier(current.value - 1)
        return self._apply_transition(
            timing, TierTransitionDirection.DOWNGRADE, new_tier, reason
        )

    def _try_upgrade(
        self, timing: FrameTiming, reason: str
    ) -> Optional[TierTransition]:
        """Attempt to upgrade quality tier.

        Args:
            timing: Frame timing data.
            reason: Reason for upgrade.

        Returns:
            TierTransition if successful, None otherwise.
        """
        if self._quality_manager is None:
            return None

        current = self._quality_manager.current_tier

        # Can't upgrade above base tier
        if current >= self.base_tier:
            return None

        new_tier = QualityTier(current.value + 1)
        return self._apply_transition(
            timing, TierTransitionDirection.UPGRADE, new_tier, reason
        )

    def _apply_transition(
        self,
        timing: FrameTiming,
        direction: TierTransitionDirection,
        new_tier: QualityTier,
        reason: str,
    ) -> TierTransition:
        """Apply a tier transition.

        Args:
            timing: Frame timing data.
            direction: Direction of transition.
            new_tier: Target tier.
            reason: Reason for transition.

        Returns:
            TierTransition record.
        """
        old_tier = self._quality_manager.current_tier

        # Apply change
        self._quality_manager.set_tier(new_tier)

        # Create transition record
        transition = TierTransition(
            frame_index=timing.frame_index,
            direction=direction,
            old_tier=old_tier,
            new_tier=new_tier,
            reason=reason,
        )

        # Update state
        self._last_transition = transition
        self._transitions.append(transition)
        self._cooldown_remaining = self._config.cooldown_frames

        # Reset trackers
        self._violation_detector.reset()
        self._recovery_tracker.reset()

        return transition

    def force_tier(self, tier: QualityTier, reason: str = "Manual override") -> Optional[TierTransition]:
        """Force a specific quality tier.

        Args:
            tier: Target tier.
            reason: Reason for change.

        Returns:
            TierTransition if change occurred.
        """
        with self._lock:
            if self._quality_manager is None:
                return None

            old_tier = self._quality_manager.current_tier
            if old_tier == tier:
                return None

            direction = (
                TierTransitionDirection.DOWNGRADE
                if tier.value < old_tier.value
                else TierTransitionDirection.UPGRADE
            )

            self._quality_manager.set_tier(tier)

            transition = TierTransition(
                frame_index=self._frame_index,
                direction=direction,
                old_tier=old_tier,
                new_tier=tier,
                reason=reason,
            )

            self._last_transition = transition
            self._transitions.append(transition)
            self._cooldown_remaining = self._config.cooldown_frames
            self._violation_detector.reset()
            self._recovery_tracker.reset()

            return transition

    def set_base_tier(self, tier: QualityTier) -> None:
        """Set the base (maximum) quality tier.

        Args:
            tier: Maximum allowed tier.
        """
        with self._lock:
            self._base_tier = tier

    def reset(self) -> None:
        """Reset adjuster state."""
        with self._lock:
            self._cooldown_remaining = 0
            self._violation_detector.reset()
            self._recovery_tracker.reset()

    def get_transitions(
        self, num_recent: Optional[int] = None
    ) -> List[TierTransition]:
        """Get recorded tier transitions.

        Args:
            num_recent: Number of recent transitions. All if None.

        Returns:
            List of transition records.
        """
        with self._lock:
            if num_recent is None:
                return list(self._transitions)
            return list(self._transitions)[-num_recent:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get adjuster statistics.

        Returns:
            Dictionary with adjuster statistics.
        """
        with self._lock:
            downgrade_count = sum(
                1 for t in self._transitions
                if t.direction == TierTransitionDirection.DOWNGRADE
            )
            upgrade_count = sum(
                1 for t in self._transitions
                if t.direction == TierTransitionDirection.UPGRADE
            )

            return {
                "enabled": self.enabled,
                "current_tier": self.current_tier.name,
                "base_tier": self.base_tier.name,
                "in_cooldown": self.in_cooldown,
                "cooldown_remaining": self._cooldown_remaining,
                "total_transitions": len(self._transitions),
                "downgrade_count": downgrade_count,
                "upgrade_count": upgrade_count,
                "violation_stats": self._violation_detector.get_statistics(),
                "recovery_stats": self._recovery_tracker.get_statistics(),
            }


# =============================================================================
# FRAME BUDGET MANAGER
# =============================================================================


class FrameBudgetManager:
    """High-level manager coordinating frame budget tracking and quality adjustment.

    Provides a simple interface for frame budget management, coordinating
    all sub-components and providing statistics and configuration.

    Example:
        manager = FrameBudgetManager(quality_manager)
        manager.configure(target_fps=60)

        # Each frame:
        transition = manager.record_frame_time(frame_time_ms)
        if transition:
            print(f"Quality changed: {transition.old_tier} -> {transition.new_tier}")
    """

    def __init__(
        self,
        quality_manager: Optional[QualityManagerProtocol] = None,
        config: Optional[FrameBudgetConfig] = None,
    ) -> None:
        """Initialize frame budget manager.

        Args:
            quality_manager: Quality manager for tier control.
            config: Budget configuration.
        """
        self._config = config or FrameBudgetConfig()
        self._budget = FrameBudget(self._config)
        self._adjuster = AutoQualityAdjuster(quality_manager, self._config)
        self._listeners: List[Callable[[TierTransition], None]] = []
        self._lock = threading.Lock()

    @property
    def config(self) -> FrameBudgetConfig:
        """Get configuration."""
        return self._config

    @property
    def budget(self) -> FrameBudget:
        """Get frame budget tracker."""
        return self._budget

    @property
    def adjuster(self) -> AutoQualityAdjuster:
        """Get auto quality adjuster."""
        return self._adjuster

    @property
    def quality_manager(self) -> Optional[QualityManagerProtocol]:
        """Get quality manager."""
        return self._adjuster.quality_manager

    @quality_manager.setter
    def quality_manager(self, value: Optional[QualityManagerProtocol]) -> None:
        """Set quality manager."""
        self._adjuster.quality_manager = value

    @property
    def current_tier(self) -> QualityTier:
        """Get current quality tier."""
        return self._adjuster.current_tier

    @property
    def enabled(self) -> bool:
        """Check if budget system is enabled."""
        return self._config.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable budget system."""
        self._config.enabled = value

    @property
    def auto_adjust(self) -> bool:
        """Check if auto adjustment is enabled."""
        return self._config.auto_adjust

    @auto_adjust.setter
    def auto_adjust(self, value: bool) -> None:
        """Enable or disable auto adjustment."""
        self._config.auto_adjust = value

    def configure(
        self,
        target_fps: Optional[float] = None,
        violation_threshold: Optional[int] = None,
        recovery_threshold: Optional[int] = None,
        spike_tolerance: Optional[int] = None,
        over_budget_margin: Optional[float] = None,
        under_budget_margin: Optional[float] = None,
        cooldown_frames: Optional[int] = None,
    ) -> None:
        """Configure budget parameters.

        Args:
            target_fps: Target frames per second.
            violation_threshold: Frames before downgrade.
            recovery_threshold: Frames before upgrade.
            spike_tolerance: Isolated spikes to tolerate.
            over_budget_margin: Over-budget threshold multiplier.
            under_budget_margin: Under-budget threshold multiplier.
            cooldown_frames: Frames to wait after tier change.
        """
        with self._lock:
            if target_fps is not None:
                self._config.target_fps = target_fps
            if violation_threshold is not None:
                self._config.violation_threshold = violation_threshold
            if recovery_threshold is not None:
                self._config.recovery_threshold = recovery_threshold
            if spike_tolerance is not None:
                self._config.spike_tolerance = spike_tolerance
            if over_budget_margin is not None:
                self._config.over_budget_margin = over_budget_margin
            if under_budget_margin is not None:
                self._config.under_budget_margin = under_budget_margin
            if cooldown_frames is not None:
                self._config.cooldown_frames = cooldown_frames

            # Propagate to sub-components
            self._budget.config = self._config
            self._adjuster.config = self._config

    def record_frame_time(self, frame_time_ms: float) -> Optional[TierTransition]:
        """Record frame time and process for quality adjustment.

        Args:
            frame_time_ms: Frame time in milliseconds.

        Returns:
            TierTransition if quality changed, None otherwise.
        """
        if not self._config.enabled:
            return None

        with self._lock:
            # Record in budget tracker
            timing = self._budget.record_frame(frame_time_ms)

            # Process for quality adjustment
            transition = self._adjuster.process_frame(timing)

            # Notify listeners
            if transition is not None:
                self._notify_listeners(transition)

            return transition

    def force_tier(
        self, tier: QualityTier, reason: str = "Manual override"
    ) -> Optional[TierTransition]:
        """Force a specific quality tier.

        Args:
            tier: Target tier.
            reason: Reason for change.

        Returns:
            TierTransition if change occurred.
        """
        with self._lock:
            transition = self._adjuster.force_tier(tier, reason)
            if transition is not None:
                self._notify_listeners(transition)
            return transition

    def set_base_tier(self, tier: QualityTier) -> None:
        """Set the base (maximum) quality tier.

        Args:
            tier: Maximum allowed tier.
        """
        self._adjuster.set_base_tier(tier)

    def add_transition_listener(
        self, listener: Callable[[TierTransition], None]
    ) -> None:
        """Add listener for tier transitions.

        Args:
            listener: Callback function.
        """
        with self._lock:
            self._listeners.append(listener)

    def remove_transition_listener(
        self, listener: Callable[[TierTransition], None]
    ) -> None:
        """Remove tier transition listener.

        Args:
            listener: Callback to remove.
        """
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _notify_listeners(self, transition: TierTransition) -> None:
        """Notify all listeners of a tier transition.

        Args:
            transition: Transition that occurred.
        """
        for listener in self._listeners:
            try:
                listener(transition)
            except Exception:
                pass  # Don't let listener errors affect operation

    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self._budget.reset()
            self._adjuster.reset()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics.

        Returns:
            Dictionary with all statistics.
        """
        with self._lock:
            return {
                "config": {
                    "target_fps": self._config.target_fps,
                    "target_frame_time_ms": self._config.target_frame_time_ms,
                    "violation_threshold": self._config.violation_threshold,
                    "recovery_threshold": self._config.recovery_threshold,
                    "spike_tolerance": self._config.spike_tolerance,
                    "over_budget_margin": self._config.over_budget_margin,
                    "under_budget_margin": self._config.under_budget_margin,
                    "cooldown_frames": self._config.cooldown_frames,
                    "enabled": self._config.enabled,
                    "auto_adjust": self._config.auto_adjust,
                },
                "budget": self._budget.get_statistics(),
                "adjuster": self._adjuster.get_statistics(),
            }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_default_manager: Optional[FrameBudgetManager] = None
_manager_lock = threading.Lock()


def get_default_frame_budget_manager() -> FrameBudgetManager:
    """Get the default frame budget manager instance.

    Returns:
        Default FrameBudgetManager singleton.
    """
    global _default_manager
    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                _default_manager = FrameBudgetManager()
    return _default_manager


def set_default_frame_budget_manager(manager: FrameBudgetManager) -> None:
    """Set the default frame budget manager instance.

    Args:
        manager: Manager to use as default.
    """
    global _default_manager
    with _manager_lock:
        _default_manager = manager


def reset_default_frame_budget_manager() -> None:
    """Reset the default frame budget manager to None."""
    global _default_manager
    with _manager_lock:
        _default_manager = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    "FrameBudget",
    "BudgetViolationDetector",
    "RecoveryTracker",
    "AutoQualityAdjuster",
    "FrameBudgetManager",
    # Data structures
    "BudgetState",
    "TierTransitionDirection",
    "FrameTiming",
    "BudgetViolation",
    "TierTransition",
    "FrameBudgetConfig",
    # Protocols
    "QualityManagerProtocol",
    "FrameTimeSourceProtocol",
    # Singleton access
    "get_default_frame_budget_manager",
    "set_default_frame_budget_manager",
    "reset_default_frame_budget_manager",
    # Constants
    "TARGET_FRAME_TIME_60FPS_MS",
    "TARGET_FRAME_TIME_30FPS_MS",
    "TARGET_FRAME_TIME_120FPS_MS",
    "TARGET_FRAME_TIME_144FPS_MS",
    "DEFAULT_VIOLATION_THRESHOLD",
    "DEFAULT_RECOVERY_THRESHOLD",
    "DEFAULT_SPIKE_TOLERANCE",
    "DEFAULT_OVER_BUDGET_MARGIN",
    "DEFAULT_UNDER_BUDGET_MARGIN",
    "DEFAULT_COOLDOWN_FRAMES",
    "DEFAULT_HISTORY_SIZE",
]
