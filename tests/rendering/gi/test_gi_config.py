"""Tests for GI performance budget configuration.

Tests cover:
    - GIQualityTier enumeration and comparison
    - GIBudget validation and budget checking
    - GIBudgetMonitor hysteresis logic
    - GPUTimestamp and GITimingInstrument
    - Tier fallback/upgrade sequences
"""

import pytest
from engine.rendering.gi.gi_config import (
    GIQualityTier,
    GIBudget,
    GIBudgetMonitor,
    GPUTimestamp,
    GITimingInstrument,
    LOW_BUDGET,
    MEDIUM_BUDGET,
    HIGH_BUDGET,
    ULTRA_BUDGET,
    CINEMATIC_BUDGET,
    BUDGETS_BY_TIER,
    get_budget,
    recommend_tier_for_target_fps,
    estimate_probe_update_cost,
)


# ============================================================================
# GIQualityTier Tests
# ============================================================================


class TestGIQualityTier:
    """Tests for GIQualityTier enumeration."""

    def test_tier_ordering(self) -> None:
        """Tiers should be ordered from LOW to CINEMATIC."""
        tiers = list(GIQualityTier)
        assert tiers == [
            GIQualityTier.LOW,
            GIQualityTier.MEDIUM,
            GIQualityTier.HIGH,
            GIQualityTier.ULTRA,
            GIQualityTier.CINEMATIC,
        ]

    def test_tier_comparison_lt(self) -> None:
        """Less-than comparison should work correctly."""
        assert GIQualityTier.LOW < GIQualityTier.MEDIUM
        assert GIQualityTier.MEDIUM < GIQualityTier.HIGH
        assert GIQualityTier.HIGH < GIQualityTier.ULTRA
        assert GIQualityTier.ULTRA < GIQualityTier.CINEMATIC
        assert not GIQualityTier.CINEMATIC < GIQualityTier.LOW

    def test_tier_comparison_le(self) -> None:
        """Less-than-or-equal comparison should work correctly."""
        assert GIQualityTier.LOW <= GIQualityTier.LOW
        assert GIQualityTier.LOW <= GIQualityTier.MEDIUM
        assert not GIQualityTier.HIGH <= GIQualityTier.MEDIUM

    def test_tier_comparison_gt(self) -> None:
        """Greater-than comparison should work correctly."""
        assert GIQualityTier.HIGH > GIQualityTier.MEDIUM
        assert GIQualityTier.CINEMATIC > GIQualityTier.LOW
        assert not GIQualityTier.LOW > GIQualityTier.MEDIUM

    def test_tier_comparison_ge(self) -> None:
        """Greater-than-or-equal comparison should work correctly."""
        assert GIQualityTier.HIGH >= GIQualityTier.HIGH
        assert GIQualityTier.HIGH >= GIQualityTier.MEDIUM
        assert not GIQualityTier.LOW >= GIQualityTier.MEDIUM

    def test_next_lower_tier(self) -> None:
        """next_lower should return the previous tier or None."""
        assert GIQualityTier.CINEMATIC.next_lower() == GIQualityTier.ULTRA
        assert GIQualityTier.ULTRA.next_lower() == GIQualityTier.HIGH
        assert GIQualityTier.HIGH.next_lower() == GIQualityTier.MEDIUM
        assert GIQualityTier.MEDIUM.next_lower() == GIQualityTier.LOW
        assert GIQualityTier.LOW.next_lower() is None

    def test_next_higher_tier(self) -> None:
        """next_higher should return the next tier or None."""
        assert GIQualityTier.LOW.next_higher() == GIQualityTier.MEDIUM
        assert GIQualityTier.MEDIUM.next_higher() == GIQualityTier.HIGH
        assert GIQualityTier.HIGH.next_higher() == GIQualityTier.ULTRA
        assert GIQualityTier.ULTRA.next_higher() == GIQualityTier.CINEMATIC
        assert GIQualityTier.CINEMATIC.next_higher() is None


# ============================================================================
# GIBudget Tests
# ============================================================================


class TestGIBudget:
    """Tests for GIBudget configuration."""

    def test_low_budget_values(self) -> None:
        """LOW budget should have correct values."""
        assert LOW_BUDGET.tier == GIQualityTier.LOW
        assert LOW_BUDGET.total_budget_ms == 0.2
        assert LOW_BUDGET.ddgi_budget_ms == 0.0  # Disabled on low-end
        assert LOW_BUDGET.rt_budget_ms == 0.0  # No RT

    def test_medium_budget_values(self) -> None:
        """MEDIUM budget should have correct values."""
        assert MEDIUM_BUDGET.tier == GIQualityTier.MEDIUM
        assert MEDIUM_BUDGET.total_budget_ms == 0.8
        assert MEDIUM_BUDGET.ddgi_budget_ms == 0.3

    def test_high_budget_values(self) -> None:
        """HIGH budget should have correct values."""
        assert HIGH_BUDGET.tier == GIQualityTier.HIGH
        assert HIGH_BUDGET.total_budget_ms == 2.5
        assert HIGH_BUDGET.ddgi_budget_ms == 1.0

    def test_ultra_budget_values(self) -> None:
        """ULTRA budget should have correct values."""
        assert ULTRA_BUDGET.tier == GIQualityTier.ULTRA
        assert ULTRA_BUDGET.total_budget_ms == 5.0
        assert ULTRA_BUDGET.rt_budget_ms == 1.5  # Has RT

    def test_cinematic_budget_values(self) -> None:
        """CINEMATIC budget should have correct values."""
        assert CINEMATIC_BUDGET.tier == GIQualityTier.CINEMATIC
        assert CINEMATIC_BUDGET.total_budget_ms == 12.7
        assert CINEMATIC_BUDGET.rt_budget_ms == 5.0

    def test_budget_progression(self) -> None:
        """Budget totals should increase with tier."""
        budgets = [LOW_BUDGET, MEDIUM_BUDGET, HIGH_BUDGET, ULTRA_BUDGET, CINEMATIC_BUDGET]
        for i in range(1, len(budgets)):
            assert budgets[i].total_budget_ms > budgets[i - 1].total_budget_ms

    def test_get_budget_all_tiers(self) -> None:
        """get_budget should return correct budget for all tiers."""
        assert get_budget(GIQualityTier.LOW) == LOW_BUDGET
        assert get_budget(GIQualityTier.MEDIUM) == MEDIUM_BUDGET
        assert get_budget(GIQualityTier.HIGH) == HIGH_BUDGET
        assert get_budget(GIQualityTier.ULTRA) == ULTRA_BUDGET
        assert get_budget(GIQualityTier.CINEMATIC) == CINEMATIC_BUDGET

    def test_is_within_budget_under(self) -> None:
        """is_within_budget should return True when under budget."""
        assert HIGH_BUDGET.is_within_budget(1.0)
        assert HIGH_BUDGET.is_within_budget(2.5)  # Exactly at budget

    def test_is_within_budget_over(self) -> None:
        """is_within_budget should return False when over budget."""
        assert not HIGH_BUDGET.is_within_budget(3.0)
        assert not HIGH_BUDGET.is_within_budget(10.0)

    def test_headroom_calculation(self) -> None:
        """headroom_ms should calculate remaining budget correctly."""
        assert HIGH_BUDGET.headroom_ms(1.0) == pytest.approx(1.5)
        assert HIGH_BUDGET.headroom_ms(2.5) == pytest.approx(0.0)
        assert HIGH_BUDGET.headroom_ms(3.0) == pytest.approx(-0.5)

    def test_utilization_calculation(self) -> None:
        """utilization should calculate percentage correctly."""
        assert HIGH_BUDGET.utilization(1.25) == pytest.approx(50.0)
        assert HIGH_BUDGET.utilization(2.5) == pytest.approx(100.0)
        assert HIGH_BUDGET.utilization(5.0) == pytest.approx(200.0)

    def test_budget_validation_negative_values(self) -> None:
        """Budget should reject negative values."""
        with pytest.raises(ValueError, match="total_budget_ms must be non-negative"):
            GIBudget(
                tier=GIQualityTier.LOW,
                total_budget_ms=-1.0,
                ddgi_budget_ms=0.0,
                ssgi_budget_ms=0.0,
                ssr_budget_ms=0.0,
                rt_budget_ms=0.0,
                probe_update_budget_ms=0.0,
            )


# ============================================================================
# GIBudgetMonitor Tests
# ============================================================================


class TestGIBudgetMonitor:
    """Tests for GIBudgetMonitor hysteresis logic."""

    def test_initial_state(self) -> None:
        """Monitor should initialize with correct state."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.HIGH,
            target_tier=GIQualityTier.ULTRA,
        )
        assert monitor.current_tier == GIQualityTier.HIGH
        assert monitor.target_tier == GIQualityTier.ULTRA
        assert monitor.consecutive_over_budget == 0
        assert monitor.consecutive_under_budget == 0

    def test_under_budget_increments_counter(self) -> None:
        """Under-budget frames should increment counter."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.HIGH,
            target_tier=GIQualityTier.ULTRA,
        )
        monitor.record_frame(1.0)  # Under 2.5ms budget
        assert monitor.consecutive_under_budget == 1
        assert monitor.consecutive_over_budget == 0

    def test_over_budget_increments_counter(self) -> None:
        """Over-budget frames should increment counter."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.HIGH,
            target_tier=GIQualityTier.ULTRA,
        )
        monitor.record_frame(5.0)  # Over 2.5ms budget
        assert monitor.consecutive_over_budget == 1
        assert monitor.consecutive_under_budget == 0

    def test_hysteresis_fallback_after_3_frames(self) -> None:
        """Fallback should trigger after 3 consecutive over-budget frames."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.HIGH,
            target_tier=GIQualityTier.HIGH,
            hysteresis_threshold=3,
        )

        # First two over-budget frames: no fallback
        assert monitor.record_frame(5.0) is None
        assert monitor.record_frame(5.0) is None
        assert monitor.current_tier == GIQualityTier.HIGH

        # Third over-budget frame: fallback triggered
        result = monitor.record_frame(5.0)
        assert result == GIQualityTier.MEDIUM
        assert monitor.current_tier == GIQualityTier.MEDIUM

    def test_hysteresis_reset_on_good_frame(self) -> None:
        """A good frame should reset the over-budget counter."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.HIGH,
            target_tier=GIQualityTier.HIGH,
            hysteresis_threshold=3,
        )

        monitor.record_frame(5.0)  # Over
        monitor.record_frame(5.0)  # Over
        assert monitor.consecutive_over_budget == 2

        monitor.record_frame(1.0)  # Under - resets counter
        assert monitor.consecutive_over_budget == 0
        assert monitor.consecutive_under_budget == 1

    def test_upgrade_after_10_frames(self) -> None:
        """Upgrade should trigger after 10 consecutive under-budget frames."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.MEDIUM,
            target_tier=GIQualityTier.ULTRA,
            upgrade_threshold=10,
        )

        # 9 under-budget frames: no upgrade
        for _ in range(9):
            result = monitor.record_frame(0.1)
            assert result is None
        assert monitor.current_tier == GIQualityTier.MEDIUM

        # 10th under-budget frame: upgrade triggered
        result = monitor.record_frame(0.1)
        assert result == GIQualityTier.HIGH
        assert monitor.current_tier == GIQualityTier.HIGH

    def test_no_upgrade_past_target(self) -> None:
        """Upgrade should not exceed target tier."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.HIGH,
            target_tier=GIQualityTier.HIGH,  # Already at target
            upgrade_threshold=10,
        )

        # Even with many under-budget frames, no upgrade
        for _ in range(20):
            result = monitor.record_frame(0.1)
            assert result is None
        assert monitor.current_tier == GIQualityTier.HIGH

    def test_fallback_sequence_ultra_to_low(self) -> None:
        """Full fallback sequence from ULTRA to LOW."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.ULTRA,
            target_tier=GIQualityTier.ULTRA,
            hysteresis_threshold=3,
        )

        expected_sequence = [
            GIQualityTier.HIGH,
            GIQualityTier.MEDIUM,
            GIQualityTier.LOW,
        ]

        for expected_tier in expected_sequence:
            # Trigger fallback with 3 over-budget frames
            for _ in range(3):
                result = monitor.record_frame(100.0)  # Way over budget
                if result is not None:
                    assert result == expected_tier
                    break

        # At LOW, no more fallback possible
        assert monitor.current_tier == GIQualityTier.LOW
        for _ in range(10):
            assert monitor.record_frame(100.0) is None

    def test_upgrade_sequence_low_to_ultra(self) -> None:
        """Full upgrade sequence from LOW to ULTRA."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.LOW,
            target_tier=GIQualityTier.ULTRA,
            upgrade_threshold=10,
        )

        expected_sequence = [
            GIQualityTier.MEDIUM,
            GIQualityTier.HIGH,
            GIQualityTier.ULTRA,
        ]

        for expected_tier in expected_sequence:
            # Trigger upgrade with 10 under-budget frames
            for _ in range(10):
                result = monitor.record_frame(0.01)  # Way under budget
                if result is not None:
                    assert result == expected_tier
                    break

        # At ULTRA (target), no more upgrades
        assert monitor.current_tier == GIQualityTier.ULTRA
        for _ in range(20):
            assert monitor.record_frame(0.01) is None

    def test_counters_reset_on_tier_change(self) -> None:
        """Counters should reset after tier change."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.HIGH,
            target_tier=GIQualityTier.HIGH,
            hysteresis_threshold=3,
        )

        # Build up over-budget counter
        monitor.record_frame(5.0)
        monitor.record_frame(5.0)
        monitor.record_frame(5.0)  # Triggers fallback

        # Counters should be reset
        assert monitor.consecutive_over_budget == 0
        assert monitor.consecutive_under_budget == 0

    def test_set_target_tier_below_current(self) -> None:
        """Setting target below current should immediately fallback."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.ULTRA,
            target_tier=GIQualityTier.ULTRA,
        )

        monitor.set_target_tier(GIQualityTier.MEDIUM)
        assert monitor.current_tier == GIQualityTier.MEDIUM
        assert monitor.target_tier == GIQualityTier.MEDIUM

    def test_reset_restores_target_tier(self) -> None:
        """Reset should restore current to target."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.LOW,
            target_tier=GIQualityTier.ULTRA,
        )
        monitor.frame_history = [1.0, 2.0, 3.0]

        monitor.reset()
        assert monitor.current_tier == GIQualityTier.ULTRA
        assert monitor.consecutive_over_budget == 0
        assert monitor.consecutive_under_budget == 0
        assert len(monitor.frame_history) == 0

    def test_average_frame_time(self) -> None:
        """get_average_frame_time should compute correctly."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.HIGH,
            target_tier=GIQualityTier.HIGH,
        )

        for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
            monitor.record_frame(t)

        assert monitor.get_average_frame_time(5) == pytest.approx(3.0)
        assert monitor.get_average_frame_time(3) == pytest.approx(4.0)  # Last 3

    def test_frame_time_variance(self) -> None:
        """get_frame_time_variance should compute correctly."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.HIGH,
            target_tier=GIQualityTier.HIGH,
        )

        # All same value: variance = 0
        for _ in range(5):
            monitor.record_frame(2.0)
        assert monitor.get_frame_time_variance(5) == pytest.approx(0.0)

    def test_budget_utilization(self) -> None:
        """get_budget_utilization should return correct percentage."""
        monitor = GIBudgetMonitor(
            current_tier=GIQualityTier.HIGH,  # 2.5ms budget
            target_tier=GIQualityTier.HIGH,
        )

        monitor.record_frame(1.25)  # 50% of budget
        assert monitor.get_budget_utilization(1) == pytest.approx(50.0)

    def test_validation_invalid_hysteresis_threshold(self) -> None:
        """Should reject hysteresis_threshold < 1."""
        with pytest.raises(ValueError, match="hysteresis_threshold must be at least 1"):
            GIBudgetMonitor(
                current_tier=GIQualityTier.HIGH,
                target_tier=GIQualityTier.HIGH,
                hysteresis_threshold=0,
            )

    def test_validation_invalid_upgrade_threshold(self) -> None:
        """Should reject upgrade_threshold < 1."""
        with pytest.raises(ValueError, match="upgrade_threshold must be at least 1"):
            GIBudgetMonitor(
                current_tier=GIQualityTier.HIGH,
                target_tier=GIQualityTier.HIGH,
                upgrade_threshold=0,
            )


# ============================================================================
# GPUTimestamp Tests
# ============================================================================


class TestGPUTimestamp:
    """Tests for GPUTimestamp."""

    def test_duration_calculation(self) -> None:
        """Duration should be calculated correctly from nanoseconds."""
        ts = GPUTimestamp(name="test_pass", start_query_index=0, end_query_index=1)

        # 1,000,000 ns = 1.0 ms
        query_results = [0.0, 1_000_000.0]
        duration = ts.duration_from_results(query_results)
        assert duration == pytest.approx(1.0)

    def test_duration_with_offset(self) -> None:
        """Duration should work with non-zero start time."""
        ts = GPUTimestamp(name="test_pass", start_query_index=0, end_query_index=1)

        query_results = [5_000_000.0, 7_000_000.0]  # 2ms duration
        duration = ts.duration_from_results(query_results)
        assert duration == pytest.approx(2.0)


# ============================================================================
# GITimingInstrument Tests
# ============================================================================


class TestGITimingInstrument:
    """Tests for GITimingInstrument."""

    def test_begin_end_pass(self) -> None:
        """begin_pass and end_pass should allocate queries correctly."""
        instrument = GITimingInstrument()

        start = instrument.begin_pass("ddgi")
        end = instrument.end_pass("ddgi")

        assert start == 0
        assert end == 1
        assert instrument.query_count == 2

    def test_multiple_passes(self) -> None:
        """Multiple passes should allocate sequential queries."""
        instrument = GITimingInstrument()

        instrument.begin_pass("ddgi")
        instrument.end_pass("ddgi")
        instrument.begin_pass("ssgi")
        instrument.end_pass("ssgi")

        assert instrument.query_count == 4

    def test_resolve_timings(self) -> None:
        """resolve should return correct durations for each pass."""
        instrument = GITimingInstrument()

        instrument.begin_pass("ddgi")
        instrument.end_pass("ddgi")
        instrument.begin_pass("ssgi")
        instrument.end_pass("ssgi")

        # Query results in nanoseconds
        query_results = [
            0.0,  # ddgi start
            1_000_000.0,  # ddgi end (1ms)
            1_500_000.0,  # ssgi start
            3_500_000.0,  # ssgi end (2ms)
        ]

        timings = instrument.resolve(query_results)
        assert timings["ddgi"] == pytest.approx(1.0)
        assert timings["ssgi"] == pytest.approx(2.0)

    def test_get_total_duration(self) -> None:
        """get_total_duration should sum all pass durations."""
        instrument = GITimingInstrument()

        instrument.begin_pass("ddgi")
        instrument.end_pass("ddgi")
        instrument.begin_pass("ssgi")
        instrument.end_pass("ssgi")

        query_results = [0.0, 1_000_000.0, 1_500_000.0, 3_500_000.0]

        total = instrument.get_total_duration(query_results)
        assert total == pytest.approx(3.0)  # 1ms + 2ms

    def test_reset_clears_state(self) -> None:
        """reset should clear all queries and passes."""
        instrument = GITimingInstrument()

        instrument.begin_pass("ddgi")
        instrument.end_pass("ddgi")
        instrument.reset()

        assert instrument.query_count == 0

    def test_duplicate_pass_name_raises(self) -> None:
        """begin_pass should raise if pass already active."""
        instrument = GITimingInstrument()

        instrument.begin_pass("ddgi")
        with pytest.raises(ValueError, match="Pass 'ddgi' is already active"):
            instrument.begin_pass("ddgi")

    def test_end_inactive_pass_raises(self) -> None:
        """end_pass should raise if pass not active."""
        instrument = GITimingInstrument()

        with pytest.raises(ValueError, match="Pass 'ddgi' is not active"):
            instrument.end_pass("ddgi")

    def test_resolve_insufficient_results_raises(self) -> None:
        """resolve should raise if query results too short."""
        instrument = GITimingInstrument()

        instrument.begin_pass("ddgi")
        instrument.end_pass("ddgi")

        with pytest.raises(ValueError, match="Expected at least 2 query results"):
            instrument.resolve([0.0])  # Only 1 result, need 2


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_recommend_tier_60fps(self) -> None:
        """60 FPS with 15% GI budget should recommend appropriate tier."""
        # 60 FPS = 16.67ms frame, 15% = 2.5ms GI budget
        tier = recommend_tier_for_target_fps(60.0, gi_budget_fraction=0.15)
        assert tier == GIQualityTier.HIGH  # 2.5ms budget fits HIGH

    def test_recommend_tier_30fps(self) -> None:
        """30 FPS should allow higher tier."""
        # 30 FPS = 33.33ms frame, 15% = 5ms GI budget
        tier = recommend_tier_for_target_fps(30.0, gi_budget_fraction=0.15)
        assert tier == GIQualityTier.ULTRA  # 5ms budget fits ULTRA

    def test_recommend_tier_120fps(self) -> None:
        """120 FPS should force lower tier."""
        # 120 FPS = 8.33ms frame, 15% = 1.25ms GI budget
        tier = recommend_tier_for_target_fps(120.0, gi_budget_fraction=0.15)
        assert tier == GIQualityTier.MEDIUM  # 0.8ms fits, 2.5ms doesn't

    def test_estimate_probe_update_cost(self) -> None:
        """Probe update cost estimation should be reasonable."""
        # 1000 probes * 256 rays = 256,000 rays
        # At 0.5ms per million rays = 0.128ms
        cost = estimate_probe_update_cost(
            probe_count=1000,
            rays_per_probe=256,
            ms_per_million_rays=0.5,
        )
        assert cost == pytest.approx(0.128)

    def test_estimate_probe_update_cost_large_grid(self) -> None:
        """Large probe grids should have proportionally higher cost."""
        small = estimate_probe_update_cost(probe_count=1000)
        large = estimate_probe_update_cost(probe_count=10000)
        assert large == pytest.approx(10.0 * small)
