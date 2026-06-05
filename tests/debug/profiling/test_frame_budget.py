"""Tests for Frame Budget System (T-CC-3.10).

Comprehensive test suite covering:
- FrameBudget tracking and state classification
- BudgetViolationDetector threshold detection
- RecoveryTracker upgrade detection
- AutoQualityAdjuster tier transitions
- FrameBudgetManager integration
- Configuration handling
- Edge cases and stress scenarios
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from engine.debug.profiling.frame_budget import (
    AutoQualityAdjuster,
    BudgetState,
    BudgetViolation,
    BudgetViolationDetector,
    FrameBudget,
    FrameBudgetConfig,
    FrameBudgetManager,
    FrameTiming,
    RecoveryTracker,
    TierTransition,
    TierTransitionDirection,
    DEFAULT_COOLDOWN_FRAMES,
    DEFAULT_HISTORY_SIZE,
    DEFAULT_OVER_BUDGET_MARGIN,
    DEFAULT_RECOVERY_THRESHOLD,
    DEFAULT_SPIKE_TOLERANCE,
    DEFAULT_UNDER_BUDGET_MARGIN,
    DEFAULT_VIOLATION_THRESHOLD,
    TARGET_FRAME_TIME_30FPS_MS,
    TARGET_FRAME_TIME_60FPS_MS,
    TARGET_FRAME_TIME_120FPS_MS,
    TARGET_FRAME_TIME_144FPS_MS,
    get_default_frame_budget_manager,
    reset_default_frame_budget_manager,
    set_default_frame_budget_manager,
)
from trinity.types import QualityTier


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def config() -> FrameBudgetConfig:
    """Create default configuration."""
    return FrameBudgetConfig()


@pytest.fixture
def config_60fps() -> FrameBudgetConfig:
    """Create 60fps configuration."""
    return FrameBudgetConfig(target_fps=60.0)


@pytest.fixture
def config_30fps() -> FrameBudgetConfig:
    """Create 30fps configuration."""
    return FrameBudgetConfig(target_fps=30.0)


@pytest.fixture
def config_fast_response() -> FrameBudgetConfig:
    """Create fast-response configuration for testing."""
    return FrameBudgetConfig(
        target_fps=60.0,
        violation_threshold=3,
        recovery_threshold=5,
        spike_tolerance=1,
        cooldown_frames=2,
    )


@pytest.fixture
def frame_budget(config: FrameBudgetConfig) -> FrameBudget:
    """Create frame budget tracker."""
    return FrameBudget(config)


@pytest.fixture
def violation_detector(config: FrameBudgetConfig) -> BudgetViolationDetector:
    """Create violation detector."""
    return BudgetViolationDetector(config)


@pytest.fixture
def recovery_tracker(config: FrameBudgetConfig) -> RecoveryTracker:
    """Create recovery tracker."""
    return RecoveryTracker(config)


@pytest.fixture
def mock_quality_manager() -> MagicMock:
    """Create mock quality manager."""
    manager = MagicMock()
    manager.current_tier = QualityTier.HIGH
    manager.base_tier = QualityTier.ULTRA
    return manager


@pytest.fixture
def adjuster(
    mock_quality_manager: MagicMock, config_fast_response: FrameBudgetConfig
) -> AutoQualityAdjuster:
    """Create auto quality adjuster."""
    return AutoQualityAdjuster(mock_quality_manager, config_fast_response)


@pytest.fixture
def manager(mock_quality_manager: MagicMock) -> FrameBudgetManager:
    """Create frame budget manager."""
    return FrameBudgetManager(mock_quality_manager)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton state between tests."""
    reset_default_frame_budget_manager()
    yield
    reset_default_frame_budget_manager()


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestFrameBudgetConfig:
    """Tests for FrameBudgetConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FrameBudgetConfig()
        assert config.target_fps == 60.0
        assert config.violation_threshold == DEFAULT_VIOLATION_THRESHOLD
        assert config.recovery_threshold == DEFAULT_RECOVERY_THRESHOLD
        assert config.spike_tolerance == DEFAULT_SPIKE_TOLERANCE
        assert config.over_budget_margin == DEFAULT_OVER_BUDGET_MARGIN
        assert config.under_budget_margin == DEFAULT_UNDER_BUDGET_MARGIN
        assert config.cooldown_frames == DEFAULT_COOLDOWN_FRAMES
        assert config.history_size == DEFAULT_HISTORY_SIZE
        assert config.enabled is True
        assert config.auto_adjust is True

    def test_target_frame_time_60fps(self):
        """Test target frame time for 60 FPS."""
        config = FrameBudgetConfig(target_fps=60.0)
        assert abs(config.target_frame_time_ms - 16.67) < 0.01

    def test_target_frame_time_30fps(self):
        """Test target frame time for 30 FPS."""
        config = FrameBudgetConfig(target_fps=30.0)
        assert abs(config.target_frame_time_ms - 33.33) < 0.01

    def test_target_frame_time_120fps(self):
        """Test target frame time for 120 FPS."""
        config = FrameBudgetConfig(target_fps=120.0)
        assert abs(config.target_frame_time_ms - 8.33) < 0.01

    def test_over_budget_threshold(self):
        """Test over-budget threshold calculation."""
        config = FrameBudgetConfig(target_fps=60.0, over_budget_margin=1.2)
        assert abs(config.over_budget_threshold_ms - 20.0) < 0.1

    def test_under_budget_threshold(self):
        """Test under-budget threshold calculation."""
        config = FrameBudgetConfig(target_fps=60.0, under_budget_margin=0.8)
        assert abs(config.under_budget_threshold_ms - 13.33) < 0.1

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        config = FrameBudgetConfig(
            violation_threshold=5,
            recovery_threshold=30,
            spike_tolerance=2,
        )
        assert config.violation_threshold == 5
        assert config.recovery_threshold == 30
        assert config.spike_tolerance == 2


class TestConstants:
    """Test module constants."""

    def test_target_frame_time_constants(self):
        """Test target frame time constants."""
        assert abs(TARGET_FRAME_TIME_60FPS_MS - 16.67) < 0.01
        assert abs(TARGET_FRAME_TIME_30FPS_MS - 33.33) < 0.01
        assert abs(TARGET_FRAME_TIME_120FPS_MS - 8.33) < 0.01
        assert abs(TARGET_FRAME_TIME_144FPS_MS - 6.94) < 0.01


# =============================================================================
# FRAME BUDGET TESTS
# =============================================================================


class TestFrameBudget:
    """Tests for FrameBudget."""

    def test_initial_state(self, frame_budget: FrameBudget):
        """Test initial budget state."""
        assert frame_budget.current_state == BudgetState.WITHIN_BUDGET
        assert frame_budget.frame_index == 0
        assert frame_budget.average_frame_time_ms == 0.0
        assert frame_budget.min_frame_time_ms == 0.0
        assert frame_budget.max_frame_time_ms == 0.0

    def test_record_within_budget(self, frame_budget: FrameBudget):
        """Test recording frame within budget."""
        timing = frame_budget.record_frame(16.0)
        assert timing.frame_time_ms == 16.0
        assert timing.budget_state == BudgetState.WITHIN_BUDGET
        assert timing.frame_index == 0
        assert frame_budget.frame_index == 1

    def test_record_over_budget(self, frame_budget: FrameBudget):
        """Test recording frame over budget."""
        timing = frame_budget.record_frame(25.0)
        assert timing.budget_state == BudgetState.OVER_BUDGET
        assert frame_budget.current_state == BudgetState.OVER_BUDGET

    def test_record_under_budget(self, frame_budget: FrameBudget):
        """Test recording frame under budget."""
        timing = frame_budget.record_frame(10.0)
        assert timing.budget_state == BudgetState.UNDER_BUDGET
        assert frame_budget.current_state == BudgetState.UNDER_BUDGET

    def test_average_frame_time(self, frame_budget: FrameBudget):
        """Test average frame time calculation."""
        frame_budget.record_frame(10.0)
        frame_budget.record_frame(20.0)
        frame_budget.record_frame(30.0)
        assert frame_budget.average_frame_time_ms == 20.0

    def test_min_max_frame_time(self, frame_budget: FrameBudget):
        """Test min/max frame time tracking."""
        frame_budget.record_frame(10.0)
        frame_budget.record_frame(50.0)
        frame_budget.record_frame(30.0)
        assert frame_budget.min_frame_time_ms == 10.0
        assert frame_budget.max_frame_time_ms == 50.0

    def test_history(self, frame_budget: FrameBudget):
        """Test frame history storage."""
        for i in range(10):
            frame_budget.record_frame(16.0 + i)

        history = frame_budget.get_history()
        assert len(history) == 10
        assert history[0].frame_time_ms == 16.0
        assert history[9].frame_time_ms == 25.0

    def test_history_limit(self, frame_budget: FrameBudget):
        """Test history respects limit."""
        frame_budget.config = FrameBudgetConfig(history_size=5)
        for i in range(10):
            frame_budget.record_frame(16.0)

        history = frame_budget.get_history()
        assert len(history) == 5

    def test_get_history_subset(self, frame_budget: FrameBudget):
        """Test getting subset of history."""
        for i in range(20):
            frame_budget.record_frame(16.0)

        recent = frame_budget.get_history(5)
        assert len(recent) == 5

    def test_recent_average(self, frame_budget: FrameBudget):
        """Test recent average calculation."""
        for i in range(50):
            frame_budget.record_frame(10.0)
        for i in range(10):
            frame_budget.record_frame(30.0)

        avg = frame_budget.get_recent_average(10)
        assert avg == 30.0

    def test_percentile(self, frame_budget: FrameBudget):
        """Test percentile calculation."""
        # Record 100 frames with values 1-100
        for i in range(1, 101):
            frame_budget.record_frame(float(i))

        p50 = frame_budget.get_percentile(50, 100)
        p90 = frame_budget.get_percentile(90, 100)
        p99 = frame_budget.get_percentile(99, 100)

        assert 49 <= p50 <= 51
        assert 89 <= p90 <= 91
        assert 98 <= p99 <= 100

    def test_reset(self, frame_budget: FrameBudget):
        """Test reset clears state."""
        for i in range(10):
            frame_budget.record_frame(25.0)

        frame_budget.reset()

        assert frame_budget.frame_index == 0
        assert frame_budget.average_frame_time_ms == 0.0
        assert frame_budget.current_state == BudgetState.WITHIN_BUDGET
        assert len(frame_budget.get_history()) == 0

    def test_statistics(self, frame_budget: FrameBudget):
        """Test statistics output."""
        frame_budget.record_frame(16.0)
        frame_budget.record_frame(25.0)

        stats = frame_budget.get_statistics()
        assert stats["frame_count"] == 2
        assert stats["target_fps"] == 60.0
        assert "state_distribution" in stats

    def test_config_update_preserves_history(self, frame_budget: FrameBudget):
        """Test config update preserves existing history."""
        for i in range(10):
            frame_budget.record_frame(16.0)

        frame_budget.config = FrameBudgetConfig(history_size=200)
        assert len(frame_budget.get_history()) == 10

    def test_config_update_trims_history(self, frame_budget: FrameBudget):
        """Test config update trims history if needed."""
        for i in range(100):
            frame_budget.record_frame(16.0)

        frame_budget.config = FrameBudgetConfig(history_size=10)
        assert len(frame_budget.get_history()) == 10


# =============================================================================
# VIOLATION DETECTOR TESTS
# =============================================================================


class TestBudgetViolationDetector:
    """Tests for BudgetViolationDetector."""

    def test_initial_state(self, violation_detector: BudgetViolationDetector):
        """Test initial detector state."""
        assert violation_detector.consecutive_violations == 0
        assert not violation_detector.is_violating
        assert not violation_detector.threshold_crossed

    def test_single_violation(self, violation_detector: BudgetViolationDetector):
        """Test single violation detection."""
        timing = FrameTiming(0, 25.0, BudgetState.OVER_BUDGET, time.time())
        violation_detector.process_frame(timing)
        assert violation_detector.consecutive_violations == 1
        assert violation_detector.is_violating
        assert not violation_detector.threshold_crossed

    def test_threshold_crossing(self):
        """Test violation threshold crossing."""
        config = FrameBudgetConfig(violation_threshold=3)
        detector = BudgetViolationDetector(config)

        for i in range(2):
            timing = FrameTiming(i, 25.0, BudgetState.OVER_BUDGET, time.time())
            result = detector.process_frame(timing)
            assert not result

        timing = FrameTiming(2, 25.0, BudgetState.OVER_BUDGET, time.time())
        result = detector.process_frame(timing)
        assert result
        assert detector.threshold_crossed

    def test_spike_tolerance(self):
        """Test spike tolerance allows isolated good frames."""
        config = FrameBudgetConfig(violation_threshold=5, spike_tolerance=2)
        detector = BudgetViolationDetector(config)

        # 3 violations
        for i in range(3):
            timing = FrameTiming(i, 25.0, BudgetState.OVER_BUDGET, time.time())
            detector.process_frame(timing)

        # 1 good frame (should use spike buffer)
        timing = FrameTiming(3, 16.0, BudgetState.WITHIN_BUDGET, time.time())
        detector.process_frame(timing)
        assert detector.consecutive_violations == 3  # Still tracking

        # Continue violations
        for i in range(2):
            timing = FrameTiming(4 + i, 25.0, BudgetState.OVER_BUDGET, time.time())
            detector.process_frame(timing)

        assert detector.consecutive_violations == 5
        assert detector.threshold_crossed

    def test_reset_on_good_frames(self):
        """Test violations reset after enough good frames."""
        config = FrameBudgetConfig(violation_threshold=10, spike_tolerance=1)
        detector = BudgetViolationDetector(config)

        # Some violations
        for i in range(3):
            timing = FrameTiming(i, 25.0, BudgetState.OVER_BUDGET, time.time())
            detector.process_frame(timing)

        # Enough good frames to reset
        for i in range(5):
            timing = FrameTiming(3 + i, 16.0, BudgetState.WITHIN_BUDGET, time.time())
            detector.process_frame(timing)

        assert detector.consecutive_violations == 0
        assert not detector.is_violating

    def test_violation_record(self):
        """Test violation records are created."""
        config = FrameBudgetConfig(violation_threshold=3, spike_tolerance=0)
        detector = BudgetViolationDetector(config)

        # Trigger threshold
        for i in range(3):
            timing = FrameTiming(i, 25.0, BudgetState.OVER_BUDGET, time.time())
            detector.process_frame(timing)

        violations = detector.get_violations()
        assert len(violations) == 1
        assert violations[0].consecutive_count == 3
        assert violations[0].triggered_downgrade

    def test_reset(self):
        """Test reset clears state."""
        config = FrameBudgetConfig(violation_threshold=10)
        detector = BudgetViolationDetector(config)

        for i in range(5):
            timing = FrameTiming(i, 25.0, BudgetState.OVER_BUDGET, time.time())
            detector.process_frame(timing)

        detector.reset()
        assert detector.consecutive_violations == 0
        assert not detector.is_violating

    def test_statistics(self, violation_detector: BudgetViolationDetector):
        """Test statistics output."""
        timing = FrameTiming(0, 25.0, BudgetState.OVER_BUDGET, time.time())
        violation_detector.process_frame(timing)

        stats = violation_detector.get_statistics()
        assert stats["consecutive_violations"] == 1
        assert stats["violation_threshold"] == DEFAULT_VIOLATION_THRESHOLD


# =============================================================================
# RECOVERY TRACKER TESTS
# =============================================================================


class TestRecoveryTracker:
    """Tests for RecoveryTracker."""

    def test_initial_state(self, recovery_tracker: RecoveryTracker):
        """Test initial tracker state."""
        assert recovery_tracker.consecutive_good_frames == 0
        assert not recovery_tracker.is_recovering
        assert not recovery_tracker.threshold_crossed
        assert recovery_tracker.recovery_progress == 0.0

    def test_single_good_frame(self, recovery_tracker: RecoveryTracker):
        """Test single good frame tracking."""
        timing = FrameTiming(0, 10.0, BudgetState.UNDER_BUDGET, time.time())
        recovery_tracker.process_frame(timing)
        assert recovery_tracker.consecutive_good_frames == 1
        assert recovery_tracker.is_recovering

    def test_threshold_crossing(self):
        """Test recovery threshold crossing."""
        config = FrameBudgetConfig(recovery_threshold=3)
        tracker = RecoveryTracker(config)

        for i in range(2):
            timing = FrameTiming(i, 10.0, BudgetState.UNDER_BUDGET, time.time())
            result = tracker.process_frame(timing)
            assert not result

        timing = FrameTiming(2, 10.0, BudgetState.UNDER_BUDGET, time.time())
        result = tracker.process_frame(timing)
        assert result
        assert tracker.threshold_crossed

    def test_decay_on_within_budget(self):
        """Test slow decay on within-budget frames."""
        config = FrameBudgetConfig(recovery_threshold=10)
        tracker = RecoveryTracker(config)

        # Build up good frames
        for i in range(5):
            timing = FrameTiming(i, 10.0, BudgetState.UNDER_BUDGET, time.time())
            tracker.process_frame(timing)

        # Within budget decays slowly
        timing = FrameTiming(5, 15.0, BudgetState.WITHIN_BUDGET, time.time())
        tracker.process_frame(timing)
        assert tracker.consecutive_good_frames == 4

    def test_fast_decay_on_over_budget(self):
        """Test fast decay on over-budget frames."""
        config = FrameBudgetConfig(recovery_threshold=20)
        tracker = RecoveryTracker(config)

        # Build up good frames
        for i in range(10):
            timing = FrameTiming(i, 10.0, BudgetState.UNDER_BUDGET, time.time())
            tracker.process_frame(timing)

        # Over budget decays faster
        timing = FrameTiming(10, 25.0, BudgetState.OVER_BUDGET, time.time())
        tracker.process_frame(timing)
        assert tracker.consecutive_good_frames == 5

    def test_recovery_progress(self):
        """Test recovery progress calculation."""
        config = FrameBudgetConfig(recovery_threshold=10)
        tracker = RecoveryTracker(config)

        for i in range(5):
            timing = FrameTiming(i, 10.0, BudgetState.UNDER_BUDGET, time.time())
            tracker.process_frame(timing)

        assert tracker.recovery_progress == 0.5

    def test_reset(self):
        """Test reset clears state."""
        config = FrameBudgetConfig(recovery_threshold=100)
        tracker = RecoveryTracker(config)

        for i in range(50):
            timing = FrameTiming(i, 10.0, BudgetState.UNDER_BUDGET, time.time())
            tracker.process_frame(timing)

        tracker.reset()
        assert tracker.consecutive_good_frames == 0
        assert not tracker.is_recovering

    def test_statistics(self, recovery_tracker: RecoveryTracker):
        """Test statistics output."""
        timing = FrameTiming(0, 10.0, BudgetState.UNDER_BUDGET, time.time())
        recovery_tracker.process_frame(timing)

        stats = recovery_tracker.get_statistics()
        assert stats["consecutive_good_frames"] == 1
        assert stats["recovery_threshold"] == DEFAULT_RECOVERY_THRESHOLD


# =============================================================================
# AUTO QUALITY ADJUSTER TESTS
# =============================================================================


class TestAutoQualityAdjuster:
    """Tests for AutoQualityAdjuster."""

    def test_initial_state(self, adjuster: AutoQualityAdjuster):
        """Test initial adjuster state."""
        assert adjuster.current_tier == QualityTier.HIGH
        assert adjuster.base_tier == QualityTier.ULTRA
        assert not adjuster.in_cooldown
        assert adjuster.enabled

    def test_downgrade_on_violations(
        self, mock_quality_manager: MagicMock, config_fast_response: FrameBudgetConfig
    ):
        """Test downgrade when violation threshold crossed."""
        adjuster = AutoQualityAdjuster(mock_quality_manager, config_fast_response)

        # Trigger violations
        for i in range(config_fast_response.violation_threshold):
            timing = FrameTiming(i, 25.0, BudgetState.OVER_BUDGET, time.time())
            result = adjuster.process_frame(timing)

            if i < config_fast_response.violation_threshold - 1:
                assert result is None
            else:
                assert result is not None
                assert result.direction == TierTransitionDirection.DOWNGRADE

        mock_quality_manager.set_tier.assert_called_once()

    def test_upgrade_on_recovery(
        self, mock_quality_manager: MagicMock, config_fast_response: FrameBudgetConfig
    ):
        """Test upgrade when recovery threshold crossed."""
        mock_quality_manager.current_tier = QualityTier.MEDIUM
        mock_quality_manager.base_tier = QualityTier.HIGH
        adjuster = AutoQualityAdjuster(mock_quality_manager, config_fast_response)
        adjuster.set_base_tier(QualityTier.HIGH)

        # Trigger recovery
        for i in range(config_fast_response.recovery_threshold):
            timing = FrameTiming(i, 10.0, BudgetState.UNDER_BUDGET, time.time())
            result = adjuster.process_frame(timing)

            if i < config_fast_response.recovery_threshold - 1:
                assert result is None
            else:
                assert result is not None
                assert result.direction == TierTransitionDirection.UPGRADE

    def test_no_downgrade_at_low(
        self, mock_quality_manager: MagicMock, config_fast_response: FrameBudgetConfig
    ):
        """Test no downgrade when already at LOW tier."""
        mock_quality_manager.current_tier = QualityTier.LOW
        adjuster = AutoQualityAdjuster(mock_quality_manager, config_fast_response)

        # Trigger violations
        for i in range(config_fast_response.violation_threshold + 5):
            timing = FrameTiming(i, 25.0, BudgetState.OVER_BUDGET, time.time())
            adjuster.process_frame(timing)

        mock_quality_manager.set_tier.assert_not_called()

    def test_no_upgrade_above_base(
        self, mock_quality_manager: MagicMock, config_fast_response: FrameBudgetConfig
    ):
        """Test no upgrade above base tier."""
        mock_quality_manager.current_tier = QualityTier.HIGH
        mock_quality_manager.base_tier = QualityTier.HIGH
        adjuster = AutoQualityAdjuster(mock_quality_manager, config_fast_response)
        adjuster.set_base_tier(QualityTier.HIGH)

        # Trigger recovery
        for i in range(config_fast_response.recovery_threshold + 5):
            timing = FrameTiming(i, 10.0, BudgetState.UNDER_BUDGET, time.time())
            adjuster.process_frame(timing)

        mock_quality_manager.set_tier.assert_not_called()

    def test_cooldown_after_transition(
        self, mock_quality_manager: MagicMock, config_fast_response: FrameBudgetConfig
    ):
        """Test cooldown prevents rapid transitions."""
        adjuster = AutoQualityAdjuster(mock_quality_manager, config_fast_response)

        # Trigger first downgrade
        for i in range(config_fast_response.violation_threshold):
            timing = FrameTiming(i, 25.0, BudgetState.OVER_BUDGET, time.time())
            adjuster.process_frame(timing)

        assert adjuster.in_cooldown
        assert adjuster.cooldown_remaining == config_fast_response.cooldown_frames

        # Violations during cooldown should not trigger
        mock_quality_manager.reset_mock()
        for i in range(config_fast_response.violation_threshold):
            frame_idx = config_fast_response.violation_threshold + i
            timing = FrameTiming(frame_idx, 25.0, BudgetState.OVER_BUDGET, time.time())
            adjuster.process_frame(timing)

        # Still in cooldown, no new tier changes
        # (cooldown is 2, threshold is 3, so we process 3 frames but cooldown ends first)

    def test_force_tier(
        self, mock_quality_manager: MagicMock, config_fast_response: FrameBudgetConfig
    ):
        """Test forcing a specific tier."""
        adjuster = AutoQualityAdjuster(mock_quality_manager, config_fast_response)

        transition = adjuster.force_tier(QualityTier.LOW, "Test override")
        assert transition is not None
        assert transition.new_tier == QualityTier.LOW
        assert transition.reason == "Test override"
        mock_quality_manager.set_tier.assert_called_with(QualityTier.LOW)

    def test_force_same_tier(
        self, mock_quality_manager: MagicMock, config_fast_response: FrameBudgetConfig
    ):
        """Test forcing same tier returns None."""
        adjuster = AutoQualityAdjuster(mock_quality_manager, config_fast_response)

        transition = adjuster.force_tier(QualityTier.HIGH)
        assert transition is None
        mock_quality_manager.set_tier.assert_not_called()

    def test_disabled_auto_adjust(
        self, mock_quality_manager: MagicMock
    ):
        """Test disabled auto adjustment."""
        config = FrameBudgetConfig(auto_adjust=False)
        adjuster = AutoQualityAdjuster(mock_quality_manager, config)

        assert not adjuster.enabled

        # Violations should not trigger changes
        for i in range(20):
            timing = FrameTiming(i, 50.0, BudgetState.OVER_BUDGET, time.time())
            adjuster.process_frame(timing)

        mock_quality_manager.set_tier.assert_not_called()

    def test_transition_history(
        self, mock_quality_manager: MagicMock, config_fast_response: FrameBudgetConfig
    ):
        """Test transition history is recorded."""
        adjuster = AutoQualityAdjuster(mock_quality_manager, config_fast_response)

        # Trigger downgrade
        for i in range(config_fast_response.violation_threshold):
            timing = FrameTiming(i, 25.0, BudgetState.OVER_BUDGET, time.time())
            adjuster.process_frame(timing)

        transitions = adjuster.get_transitions()
        assert len(transitions) == 1
        assert transitions[0].direction == TierTransitionDirection.DOWNGRADE

    def test_statistics(self, adjuster: AutoQualityAdjuster):
        """Test statistics output."""
        stats = adjuster.get_statistics()
        assert "current_tier" in stats
        assert "violation_stats" in stats
        assert "recovery_stats" in stats

    def test_reset(self, adjuster: AutoQualityAdjuster):
        """Test reset clears state."""
        # Get into a state
        for i in range(2):
            timing = FrameTiming(i, 25.0, BudgetState.OVER_BUDGET, time.time())
            adjuster.process_frame(timing)

        adjuster.reset()
        stats = adjuster.get_statistics()
        assert stats["violation_stats"]["consecutive_violations"] == 0


# =============================================================================
# FRAME BUDGET MANAGER TESTS
# =============================================================================


class TestFrameBudgetManager:
    """Tests for FrameBudgetManager."""

    def test_initial_state(self, manager: FrameBudgetManager):
        """Test initial manager state."""
        assert manager.enabled
        assert manager.auto_adjust
        assert manager.current_tier == QualityTier.HIGH

    def test_record_frame_time(self, manager: FrameBudgetManager):
        """Test recording frame times."""
        result = manager.record_frame_time(16.0)
        assert result is None  # No transition yet

        stats = manager.get_statistics()
        assert stats["budget"]["frame_count"] == 1

    def test_configure(self, manager: FrameBudgetManager):
        """Test configuration updates."""
        manager.configure(
            target_fps=30.0,
            violation_threshold=5,
            recovery_threshold=20,
        )

        assert manager.config.target_fps == 30.0
        assert manager.config.violation_threshold == 5
        assert manager.config.recovery_threshold == 20

    def test_enable_disable(self, manager: FrameBudgetManager):
        """Test enabling and disabling."""
        manager.enabled = False
        result = manager.record_frame_time(50.0)
        assert result is None

        stats = manager.get_statistics()
        assert stats["budget"]["frame_count"] == 0  # Not recorded when disabled

    def test_force_tier(self, manager: FrameBudgetManager):
        """Test forcing tier through manager."""
        transition = manager.force_tier(QualityTier.LOW)
        assert transition is not None
        assert transition.new_tier == QualityTier.LOW

    def test_set_base_tier(self, manager: FrameBudgetManager):
        """Test setting base tier."""
        manager.set_base_tier(QualityTier.MEDIUM)
        assert manager.adjuster.base_tier == QualityTier.MEDIUM

    def test_transition_listener(self, mock_quality_manager: MagicMock):
        """Test transition listeners are called."""
        manager = FrameBudgetManager(mock_quality_manager)
        manager.configure(violation_threshold=2)

        transitions_received: List[TierTransition] = []

        def on_transition(t: TierTransition):
            transitions_received.append(t)

        manager.add_transition_listener(on_transition)

        # Trigger transition
        for i in range(5):
            manager.record_frame_time(50.0)

        assert len(transitions_received) >= 1

    def test_remove_transition_listener(self, mock_quality_manager: MagicMock):
        """Test removing transition listeners."""
        manager = FrameBudgetManager(mock_quality_manager)

        call_count = [0]

        def on_transition(t: TierTransition):
            call_count[0] += 1

        manager.add_transition_listener(on_transition)
        manager.remove_transition_listener(on_transition)

        manager.force_tier(QualityTier.LOW)
        assert call_count[0] == 0

    def test_reset(self, manager: FrameBudgetManager):
        """Test reset clears all state."""
        for i in range(10):
            manager.record_frame_time(25.0)

        manager.reset()
        stats = manager.get_statistics()
        assert stats["budget"]["frame_count"] == 0

    def test_statistics(self, manager: FrameBudgetManager):
        """Test comprehensive statistics."""
        manager.record_frame_time(16.0)
        manager.record_frame_time(25.0)

        stats = manager.get_statistics()
        assert "config" in stats
        assert "budget" in stats
        assert "adjuster" in stats
        assert stats["config"]["target_fps"] == 60.0

    def test_quality_manager_property(self, manager: FrameBudgetManager):
        """Test quality manager property access."""
        assert manager.quality_manager is not None

        new_manager = MagicMock()
        new_manager.current_tier = QualityTier.MEDIUM
        new_manager.base_tier = QualityTier.HIGH
        manager.quality_manager = new_manager

        assert manager.quality_manager is new_manager


# =============================================================================
# SINGLETON TESTS
# =============================================================================


class TestSingleton:
    """Tests for singleton access."""

    def test_get_default_manager(self):
        """Test getting default manager."""
        manager = get_default_frame_budget_manager()
        assert manager is not None
        assert isinstance(manager, FrameBudgetManager)

    def test_same_instance(self):
        """Test same instance is returned."""
        manager1 = get_default_frame_budget_manager()
        manager2 = get_default_frame_budget_manager()
        assert manager1 is manager2

    def test_set_default_manager(self):
        """Test setting default manager."""
        custom_manager = FrameBudgetManager()
        set_default_frame_budget_manager(custom_manager)

        manager = get_default_frame_budget_manager()
        assert manager is custom_manager

    def test_reset_default_manager(self):
        """Test resetting default manager."""
        manager1 = get_default_frame_budget_manager()
        reset_default_frame_budget_manager()
        manager2 = get_default_frame_budget_manager()

        assert manager1 is not manager2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_downgrade_cycle(self, mock_quality_manager: MagicMock):
        """Test complete downgrade cycle."""
        config = FrameBudgetConfig(
            violation_threshold=5,
            recovery_threshold=10,
            cooldown_frames=5,
        )
        manager = FrameBudgetManager(mock_quality_manager, config)

        # Record over-budget frames until downgrade
        for i in range(10):
            result = manager.record_frame_time(30.0)
            if result is not None:
                assert result.direction == TierTransitionDirection.DOWNGRADE
                break

        assert mock_quality_manager.set_tier.called

    def test_full_recovery_cycle(self, mock_quality_manager: MagicMock):
        """Test complete recovery cycle."""
        mock_quality_manager.current_tier = QualityTier.MEDIUM
        config = FrameBudgetConfig(
            violation_threshold=10,
            recovery_threshold=5,
            cooldown_frames=2,
        )
        manager = FrameBudgetManager(mock_quality_manager, config)
        manager.set_base_tier(QualityTier.HIGH)

        # Record under-budget frames until upgrade
        for i in range(20):
            result = manager.record_frame_time(8.0)
            if result is not None:
                assert result.direction == TierTransitionDirection.UPGRADE
                break

    def test_oscillation_prevention(self, mock_quality_manager: MagicMock):
        """Test that cooldown prevents oscillation."""
        config = FrameBudgetConfig(
            violation_threshold=3,
            recovery_threshold=3,
            cooldown_frames=10,
        )
        manager = FrameBudgetManager(mock_quality_manager, config)

        transitions = []

        def track_transition(t: TierTransition):
            transitions.append(t)

        manager.add_transition_listener(track_transition)

        # Rapidly alternate between over and under budget
        for i in range(50):
            if i % 6 < 3:
                manager.record_frame_time(30.0)  # Over budget
            else:
                manager.record_frame_time(8.0)  # Under budget

        # Should not have many transitions due to cooldown
        assert len(transitions) <= 5  # Rough limit accounting for cooldown

    def test_stress_many_frames(self, mock_quality_manager: MagicMock):
        """Test processing many frames."""
        manager = FrameBudgetManager(mock_quality_manager)

        # Process 10000 frames
        for i in range(10000):
            frame_time = 16.0 + (i % 10)  # Varying frame times
            manager.record_frame_time(frame_time)

        stats = manager.get_statistics()
        assert stats["budget"]["frame_count"] == 10000

    def test_concurrent_access(self, mock_quality_manager: MagicMock):
        """Test thread-safe concurrent access."""
        manager = FrameBudgetManager(mock_quality_manager)
        errors = []

        def record_frames():
            try:
                for i in range(100):
                    manager.record_frame_time(16.0 + i % 20)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_frames) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_frame_time(self, frame_budget: FrameBudget):
        """Test handling zero frame time."""
        timing = frame_budget.record_frame(0.0)
        assert timing.budget_state == BudgetState.UNDER_BUDGET

    def test_very_large_frame_time(self, frame_budget: FrameBudget):
        """Test handling very large frame time."""
        timing = frame_budget.record_frame(10000.0)
        assert timing.budget_state == BudgetState.OVER_BUDGET

    def test_negative_frame_time(self, frame_budget: FrameBudget):
        """Test handling negative frame time (invalid but shouldn't crash)."""
        timing = frame_budget.record_frame(-5.0)
        assert timing.frame_time_ms == -5.0
        assert timing.budget_state == BudgetState.UNDER_BUDGET

    def test_no_quality_manager(self):
        """Test adjuster without quality manager."""
        adjuster = AutoQualityAdjuster(None)

        # Should not crash
        for i in range(10):
            timing = FrameTiming(i, 50.0, BudgetState.OVER_BUDGET, time.time())
            result = adjuster.process_frame(timing)
            assert result is None

    def test_config_with_zero_threshold(self):
        """Test config with zero violation threshold."""
        config = FrameBudgetConfig(violation_threshold=0)
        detector = BudgetViolationDetector(config)

        # First violation should trigger immediately
        timing = FrameTiming(0, 25.0, BudgetState.OVER_BUDGET, time.time())
        result = detector.process_frame(timing)
        # Threshold of 0 means it's always crossed when violating
        assert detector.threshold_crossed

    def test_config_with_zero_recovery(self):
        """Test config with zero recovery threshold."""
        config = FrameBudgetConfig(recovery_threshold=0)
        tracker = RecoveryTracker(config)

        # Any good frame should cross threshold
        assert tracker.threshold_crossed  # 0 threshold is always met
        assert tracker.recovery_progress == 1.0

    def test_exact_budget_frame(self, frame_budget: FrameBudget):
        """Test frame exactly at budget."""
        # Exactly at over-budget threshold
        timing = frame_budget.record_frame(20.0)  # 16.67 * 1.2 = 20.0
        assert timing.budget_state == BudgetState.WITHIN_BUDGET  # Just at threshold

    def test_listener_exception_handling(self, mock_quality_manager: MagicMock):
        """Test manager handles listener exceptions gracefully."""
        manager = FrameBudgetManager(mock_quality_manager)

        def bad_listener(t: TierTransition):
            raise ValueError("Test exception")

        manager.add_transition_listener(bad_listener)

        # Should not raise
        manager.force_tier(QualityTier.LOW)

    def test_percentile_empty_history(self, frame_budget: FrameBudget):
        """Test percentile with empty history."""
        p50 = frame_budget.get_percentile(50)
        assert p50 == 0.0

    def test_recent_average_empty_history(self, frame_budget: FrameBudget):
        """Test recent average with empty history."""
        avg = frame_budget.get_recent_average()
        assert avg == 0.0

    def test_get_history_more_than_available(self, frame_budget: FrameBudget):
        """Test getting more history than available."""
        frame_budget.record_frame(16.0)
        frame_budget.record_frame(17.0)

        history = frame_budget.get_history(100)
        assert len(history) == 2
