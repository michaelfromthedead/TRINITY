"""GI Performance Budget Configuration for TRINITY.

This module provides a quality tier system for managing GI performance vs quality
tradeoffs. Each tier defines timing budgets for various GI techniques:
    - DDGI (Dynamic Diffuse Global Illumination)
    - SSGI (Screen-Space Global Illumination)
    - SSR (Screen-Space Reflections)
    - RT (Ray Tracing)
    - Probe Updates

The GIBudgetMonitor tracks frame timing and implements hysteresis-based
automatic tier fallback when performance budgets are exceeded.

Budget Philosophy:
    - LOW tier targets mobile/low-end (0.2ms total GI)
    - MEDIUM tier targets laptops/integrated GPUs (0.8ms total)
    - HIGH tier targets desktop GPUs (2.5ms total)
    - ULTRA tier targets enthusiast GPUs (5.0ms total)
    - CINEMATIC tier targets offline rendering (12.7ms total)

References:
    - GDC 2019, "Ray Tracing in Battlefield V"
    - NVIDIA GI Best Practices 2022
    - Unreal Engine 5 Lumen Performance Guidelines
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ============================================================================
# Quality Tiers
# ============================================================================


class GIQualityTier(Enum):
    """Quality tier enumeration for GI performance scaling.

    Each tier represents a different balance between visual quality and
    GPU time budget. The renderer should select an appropriate tier based
    on target hardware and frame time requirements.
    """

    LOW = auto()  # Mobile/low-end: minimal GI
    MEDIUM = auto()  # Laptop/integrated: basic GI
    HIGH = auto()  # Desktop: full GI with optimizations
    ULTRA = auto()  # Enthusiast: maximum quality GI
    CINEMATIC = auto()  # Offline: no time budget constraints

    def __lt__(self, other: GIQualityTier) -> bool:
        """Support tier comparison for fallback logic."""
        if not isinstance(other, GIQualityTier):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: GIQualityTier) -> bool:
        """Support tier comparison for fallback logic."""
        if not isinstance(other, GIQualityTier):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: GIQualityTier) -> bool:
        """Support tier comparison for fallback logic."""
        if not isinstance(other, GIQualityTier):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: GIQualityTier) -> bool:
        """Support tier comparison for fallback logic."""
        if not isinstance(other, GIQualityTier):
            return NotImplemented
        return self.value >= other.value

    def next_lower(self) -> Optional[GIQualityTier]:
        """Get the next lower quality tier, or None if already at LOW."""
        tiers = list(GIQualityTier)
        idx = tiers.index(self)
        if idx == 0:
            return None
        return tiers[idx - 1]

    def next_higher(self) -> Optional[GIQualityTier]:
        """Get the next higher quality tier, or None if already at CINEMATIC."""
        tiers = list(GIQualityTier)
        idx = tiers.index(self)
        if idx >= len(tiers) - 1:
            return None
        return tiers[idx + 1]


# ============================================================================
# Budget Configuration
# ============================================================================


@dataclass(frozen=True)
class GIBudget:
    """Per-tier GI timing budget specification in milliseconds.

    Each budget defines time limits for individual GI techniques as well
    as the total GI budget. The sum of individual budgets may exceed the
    total budget because techniques may run in parallel or be disabled
    based on scene requirements.

    Attributes:
        tier: The quality tier this budget applies to.
        total_budget_ms: Maximum total GI time per frame.
        ddgi_budget_ms: Budget for DDGI probe updates and sampling.
        ssgi_budget_ms: Budget for screen-space GI.
        ssr_budget_ms: Budget for screen-space reflections.
        rt_budget_ms: Budget for hardware ray tracing (if available).
        probe_update_budget_ms: Budget for probe grid maintenance.
    """

    tier: GIQualityTier
    total_budget_ms: float
    ddgi_budget_ms: float
    ssgi_budget_ms: float
    ssr_budget_ms: float
    rt_budget_ms: float
    probe_update_budget_ms: float

    def __post_init__(self) -> None:
        """Validate budget values are non-negative."""
        for attr in [
            "total_budget_ms",
            "ddgi_budget_ms",
            "ssgi_budget_ms",
            "ssr_budget_ms",
            "rt_budget_ms",
            "probe_update_budget_ms",
        ]:
            value = getattr(self, attr)
            if value < 0:
                raise ValueError(f"{attr} must be non-negative, got {value}")

    def is_within_budget(self, elapsed_ms: float) -> bool:
        """Check if elapsed time is within the total budget.

        Args:
            elapsed_ms: Total GI time for the frame in milliseconds.

        Returns:
            True if elapsed time is within budget.
        """
        return elapsed_ms <= self.total_budget_ms

    def headroom_ms(self, elapsed_ms: float) -> float:
        """Calculate remaining headroom in the budget.

        Args:
            elapsed_ms: Current elapsed GI time.

        Returns:
            Positive value if under budget, negative if over.
        """
        return self.total_budget_ms - elapsed_ms

    def utilization(self, elapsed_ms: float) -> float:
        """Calculate budget utilization as a percentage.

        Args:
            elapsed_ms: Current elapsed GI time.

        Returns:
            Utilization percentage (0-100+, can exceed 100 if over budget).
        """
        if self.total_budget_ms == 0:
            return 100.0 if elapsed_ms > 0 else 0.0
        return (elapsed_ms / self.total_budget_ms) * 100.0


# Pre-defined budgets for each tier
# Values derived from industry best practices and target hardware profiles

LOW_BUDGET = GIBudget(
    tier=GIQualityTier.LOW,
    total_budget_ms=0.2,
    ddgi_budget_ms=0.0,  # Disabled on low-end
    ssgi_budget_ms=0.1,  # Minimal screen-space
    ssr_budget_ms=0.1,  # Basic reflections
    rt_budget_ms=0.0,  # No RT
    probe_update_budget_ms=0.0,  # No probes
)

MEDIUM_BUDGET = GIBudget(
    tier=GIQualityTier.MEDIUM,
    total_budget_ms=0.8,
    ddgi_budget_ms=0.3,  # Low-res probes
    ssgi_budget_ms=0.2,  # Half-res SSGI
    ssr_budget_ms=0.2,  # Standard SSR
    rt_budget_ms=0.0,  # No RT
    probe_update_budget_ms=0.1,  # Minimal probe updates
)

HIGH_BUDGET = GIBudget(
    tier=GIQualityTier.HIGH,
    total_budget_ms=2.5,
    ddgi_budget_ms=1.0,  # Full DDGI
    ssgi_budget_ms=0.5,  # Full-res SSGI
    ssr_budget_ms=0.5,  # Hi-Q SSR
    rt_budget_ms=0.0,  # No RT (software fallback)
    probe_update_budget_ms=0.5,  # Full probe updates
)

ULTRA_BUDGET = GIBudget(
    tier=GIQualityTier.ULTRA,
    total_budget_ms=5.0,
    ddgi_budget_ms=1.5,  # High-res DDGI
    ssgi_budget_ms=0.8,  # Multi-bounce SSGI
    ssr_budget_ms=0.7,  # Temporal SSR
    rt_budget_ms=1.5,  # Hardware RT shadows/reflections
    probe_update_budget_ms=0.5,  # Aggressive probe updates
)

CINEMATIC_BUDGET = GIBudget(
    tier=GIQualityTier.CINEMATIC,
    total_budget_ms=12.7,
    ddgi_budget_ms=3.0,  # Maximum quality DDGI
    ssgi_budget_ms=2.0,  # Full SSGI
    ssr_budget_ms=1.5,  # Maximum SSR
    rt_budget_ms=5.0,  # Full RT GI
    probe_update_budget_ms=1.2,  # Maximum probe quality
)

# Budget lookup table by tier
BUDGETS_BY_TIER: dict[GIQualityTier, GIBudget] = {
    GIQualityTier.LOW: LOW_BUDGET,
    GIQualityTier.MEDIUM: MEDIUM_BUDGET,
    GIQualityTier.HIGH: HIGH_BUDGET,
    GIQualityTier.ULTRA: ULTRA_BUDGET,
    GIQualityTier.CINEMATIC: CINEMATIC_BUDGET,
}


def get_budget(tier: GIQualityTier) -> GIBudget:
    """Get the budget configuration for a quality tier.

    Args:
        tier: The quality tier.

    Returns:
        The corresponding GIBudget.
    """
    return BUDGETS_BY_TIER[tier]


# ============================================================================
# Budget Monitoring
# ============================================================================


@dataclass
class GIBudgetMonitor:
    """Monitors GI timing with hysteresis-based tier fallback.

    The monitor tracks frame-to-frame timing and implements automatic
    quality tier adjustment:
        - Fallback: After N consecutive over-budget frames, drop to lower tier
        - Upgrade: After M consecutive under-budget frames, try higher tier

    Hysteresis prevents rapid tier oscillation on borderline performance.

    Attributes:
        current_tier: The current active quality tier.
        target_tier: The user's preferred maximum tier.
        consecutive_over_budget: Count of recent over-budget frames.
        consecutive_under_budget: Count of recent under-budget frames.
        hysteresis_threshold: Frames required before tier change (default: 3).
        upgrade_threshold: Frames required before upgrade attempt (default: 10).
        frame_history: Recent frame timings for analysis.
        max_history: Maximum number of frames to track.
    """

    current_tier: GIQualityTier
    target_tier: GIQualityTier
    consecutive_over_budget: int = 0
    consecutive_under_budget: int = 0
    hysteresis_threshold: int = 3
    upgrade_threshold: int = 10
    frame_history: list[float] = field(default_factory=list)
    max_history: int = 60

    def __post_init__(self) -> None:
        """Validate thresholds."""
        if self.hysteresis_threshold < 1:
            raise ValueError("hysteresis_threshold must be at least 1")
        if self.upgrade_threshold < 1:
            raise ValueError("upgrade_threshold must be at least 1")

    @property
    def current_budget(self) -> GIBudget:
        """Get the budget for the current tier."""
        return get_budget(self.current_tier)

    def record_frame(self, elapsed_ms: float) -> Optional[GIQualityTier]:
        """Record frame timing and determine if tier should change.

        Args:
            elapsed_ms: Total GI time for the frame in milliseconds.

        Returns:
            New tier if a change was triggered, None otherwise.
        """
        # Update history
        self.frame_history.append(elapsed_ms)
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)

        budget = self.current_budget

        if budget.is_within_budget(elapsed_ms):
            # Under budget
            self.consecutive_over_budget = 0
            self.consecutive_under_budget += 1

            # Check for upgrade opportunity
            if self.should_upgrade():
                new_tier = self._try_upgrade()
                if new_tier is not None:
                    return new_tier
        else:
            # Over budget
            self.consecutive_under_budget = 0
            self.consecutive_over_budget += 1

            # Check for fallback requirement
            if self.should_fallback():
                new_tier = self._do_fallback()
                if new_tier is not None:
                    return new_tier

        return None

    def should_fallback(self) -> bool:
        """Check if fallback is needed based on consecutive over-budget frames.

        Returns:
            True if fallback threshold has been reached.
        """
        return self.consecutive_over_budget >= self.hysteresis_threshold

    def should_upgrade(self) -> bool:
        """Check if upgrade should be attempted.

        Returns:
            True if upgrade threshold has been reached and current < target.
        """
        return (
            self.consecutive_under_budget >= self.upgrade_threshold
            and self.current_tier < self.target_tier
        )

    def _do_fallback(self) -> Optional[GIQualityTier]:
        """Execute tier fallback.

        Returns:
            New tier if fallback occurred, None if already at minimum.
        """
        lower = self.current_tier.next_lower()
        if lower is not None:
            self.current_tier = lower
            self._reset_counters()
            return lower
        return None

    def _try_upgrade(self) -> Optional[GIQualityTier]:
        """Attempt tier upgrade.

        Returns:
            New tier if upgrade occurred, None otherwise.
        """
        # Don't upgrade past target
        if self.current_tier >= self.target_tier:
            return None

        higher = self.current_tier.next_higher()
        if higher is not None and higher <= self.target_tier:
            self.current_tier = higher
            self._reset_counters()
            return higher
        return None

    def _reset_counters(self) -> None:
        """Reset consecutive frame counters after tier change."""
        self.consecutive_over_budget = 0
        self.consecutive_under_budget = 0

    def reset(self) -> None:
        """Full reset to target tier with cleared history."""
        self.current_tier = self.target_tier
        self._reset_counters()
        self.frame_history.clear()

    def set_target_tier(self, tier: GIQualityTier) -> None:
        """Update the target tier.

        If the new target is below current, immediately fallback.

        Args:
            tier: New target quality tier.
        """
        self.target_tier = tier
        if self.current_tier > tier:
            self.current_tier = tier
            self._reset_counters()

    def get_average_frame_time(self, window: int = 30) -> float:
        """Get average frame time over recent history.

        Args:
            window: Number of recent frames to average.

        Returns:
            Average frame time in milliseconds, or 0 if no history.
        """
        if not self.frame_history:
            return 0.0
        frames = self.frame_history[-window:]
        return sum(frames) / len(frames)

    def get_frame_time_variance(self, window: int = 30) -> float:
        """Get variance in frame time over recent history.

        Args:
            window: Number of recent frames to analyze.

        Returns:
            Variance in milliseconds squared.
        """
        if len(self.frame_history) < 2:
            return 0.0
        frames = self.frame_history[-window:]
        avg = sum(frames) / len(frames)
        return sum((f - avg) ** 2 for f in frames) / len(frames)

    def get_budget_utilization(self, window: int = 30) -> float:
        """Get average budget utilization percentage.

        Args:
            window: Number of recent frames to average.

        Returns:
            Average utilization percentage.
        """
        avg_time = self.get_average_frame_time(window)
        return self.current_budget.utilization(avg_time)


# ============================================================================
# GPU Timestamp Instrumentation
# ============================================================================


@dataclass
class GPUTimestamp:
    """Represents a GPU timestamp query pair for a named pass.

    Attributes:
        name: Descriptive name of the pass (e.g., "ddgi_update").
        start_query_index: Index of the start timestamp query.
        end_query_index: Index of the end timestamp query.
    """

    name: str
    start_query_index: int
    end_query_index: int

    def duration_from_results(self, query_results: list[float]) -> float:
        """Calculate duration from resolved query results.

        Args:
            query_results: List of resolved timestamp values (in nanoseconds).

        Returns:
            Duration in milliseconds.

        Raises:
            IndexError: If query indices are out of bounds.
        """
        start_ns = query_results[self.start_query_index]
        end_ns = query_results[self.end_query_index]
        return (end_ns - start_ns) / 1_000_000.0  # ns to ms


class GITimingInstrument:
    """Collects GPU timestamp queries for GI passes.

    This class manages timestamp query allocation and provides a simple
    interface for instrumenting GI passes. Query results are resolved
    asynchronously by the GPU backend.

    Usage:
        instrument = GITimingInstrument()
        start_idx = instrument.begin_pass("ddgi_update")
        # ... submit GPU commands ...
        end_idx = instrument.end_pass("ddgi_update")

        # After readback:
        timings = instrument.resolve(query_results)
        print(f"DDGI update: {timings['ddgi_update']:.2f}ms")

    Attributes:
        _next_query_index: Next available query index.
        _passes: Dictionary of active pass timestamps.
        _completed_passes: List of completed pass timestamps.
    """

    def __init__(self) -> None:
        """Initialize the timing instrument."""
        self._next_query_index: int = 0
        self._passes: dict[str, int] = {}  # name -> start_index
        self._completed_passes: list[GPUTimestamp] = []

    @property
    def query_count(self) -> int:
        """Get the total number of timestamp queries allocated."""
        return self._next_query_index

    def reset(self) -> None:
        """Reset for a new frame."""
        self._next_query_index = 0
        self._passes.clear()
        self._completed_passes.clear()

    def begin_pass(self, name: str) -> int:
        """Begin timing a named pass.

        Args:
            name: Unique name for this pass.

        Returns:
            The query index for the start timestamp.

        Raises:
            ValueError: If a pass with this name is already active.
        """
        if name in self._passes:
            raise ValueError(f"Pass '{name}' is already active")

        start_index = self._next_query_index
        self._next_query_index += 1
        self._passes[name] = start_index
        return start_index

    def end_pass(self, name: str) -> int:
        """End timing a named pass.

        Args:
            name: Name of the pass to end.

        Returns:
            The query index for the end timestamp.

        Raises:
            ValueError: If no pass with this name is active.
        """
        if name not in self._passes:
            raise ValueError(f"Pass '{name}' is not active")

        start_index = self._passes.pop(name)
        end_index = self._next_query_index
        self._next_query_index += 1

        self._completed_passes.append(
            GPUTimestamp(name=name, start_query_index=start_index, end_query_index=end_index)
        )
        return end_index

    def resolve(self, query_results: list[float]) -> dict[str, float]:
        """Resolve query results to pass durations.

        Args:
            query_results: List of resolved timestamp values (in nanoseconds).

        Returns:
            Dictionary mapping pass names to durations in milliseconds.

        Raises:
            ValueError: If query results don't contain enough values.
        """
        if len(query_results) < self._next_query_index:
            raise ValueError(
                f"Expected at least {self._next_query_index} query results, "
                f"got {len(query_results)}"
            )

        timings: dict[str, float] = {}
        for ts in self._completed_passes:
            timings[ts.name] = ts.duration_from_results(query_results)

        return timings

    def get_total_duration(self, query_results: list[float]) -> float:
        """Get the total duration across all passes.

        Args:
            query_results: List of resolved timestamp values.

        Returns:
            Total duration in milliseconds.
        """
        timings = self.resolve(query_results)
        return sum(timings.values())


# ============================================================================
# Utility Functions
# ============================================================================


def recommend_tier_for_target_fps(target_fps: float, gi_budget_fraction: float = 0.15) -> GIQualityTier:
    """Recommend a quality tier based on target FPS and GI budget allocation.

    Args:
        target_fps: Target frames per second.
        gi_budget_fraction: Fraction of frame time to allocate to GI (default 15%).

    Returns:
        Recommended GIQualityTier.
    """
    frame_time_ms = 1000.0 / target_fps
    gi_budget_ms = frame_time_ms * gi_budget_fraction

    # Find highest tier that fits within budget
    for tier in reversed(list(GIQualityTier)):
        budget = get_budget(tier)
        if budget.total_budget_ms <= gi_budget_ms:
            return tier

    return GIQualityTier.LOW


def estimate_probe_update_cost(
    probe_count: int,
    rays_per_probe: int = 256,
    ms_per_million_rays: float = 0.5,
) -> float:
    """Estimate probe update cost in milliseconds.

    Args:
        probe_count: Number of probes in the grid.
        rays_per_probe: Rays traced per probe per update.
        ms_per_million_rays: Baseline cost per million rays.

    Returns:
        Estimated time in milliseconds.
    """
    total_rays = probe_count * rays_per_probe
    return (total_rays / 1_000_000.0) * ms_per_million_rays
