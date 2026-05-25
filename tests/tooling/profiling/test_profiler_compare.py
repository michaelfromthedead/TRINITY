"""Tests for the profiler comparison module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.tooling.profiling.profiler_compare import (
    ComparisonResult,
    RegressionSeverity,
    MetricComparison,
    SessionDiff,
    ComparisonReport,
    RegressionThresholds,
    RegressionDetector,
    ProfilerComparator,
)


class TestComparisonResult:
    """Tests for ComparisonResult enum."""

    def test_improved(self):
        """Test improved result."""
        assert ComparisonResult.IMPROVED is not None
        assert ComparisonResult.IMPROVED.name == "IMPROVED"

    def test_regressed(self):
        """Test regressed result."""
        assert ComparisonResult.REGRESSED is not None
        assert ComparisonResult.REGRESSED.name == "REGRESSED"

    def test_unchanged(self):
        """Test unchanged result."""
        assert ComparisonResult.UNCHANGED is not None
        assert ComparisonResult.UNCHANGED.name == "UNCHANGED"

    def test_new(self):
        """Test new result."""
        assert ComparisonResult.NEW is not None
        assert ComparisonResult.NEW.name == "NEW"

    def test_removed(self):
        """Test removed result."""
        assert ComparisonResult.REMOVED is not None
        assert ComparisonResult.REMOVED.name == "REMOVED"


class TestRegressionSeverity:
    """Tests for RegressionSeverity enum."""

    def test_severity_levels(self):
        """Test all severity levels exist."""
        assert RegressionSeverity.NONE is not None
        assert RegressionSeverity.MINOR is not None
        assert RegressionSeverity.MODERATE is not None
        assert RegressionSeverity.SEVERE is not None
        assert RegressionSeverity.CRITICAL is not None

    def test_severity_ordering(self):
        """Test severity ordering by value."""
        assert RegressionSeverity.NONE.value < RegressionSeverity.MINOR.value
        assert RegressionSeverity.MINOR.value < RegressionSeverity.MODERATE.value
        assert RegressionSeverity.MODERATE.value < RegressionSeverity.SEVERE.value
        assert RegressionSeverity.SEVERE.value < RegressionSeverity.CRITICAL.value


class TestMetricComparison:
    """Tests for MetricComparison."""

    def test_basic_creation(self):
        """Test basic creation."""
        comparison = MetricComparison(
            name="test_metric",
            baseline_value=10.0,
            current_value=15.0,
            delta=5.0,
            delta_percentage=50.0,
            result=ComparisonResult.REGRESSED,
        )
        assert comparison.name == "test_metric"
        assert comparison.baseline_value == 10.0
        assert comparison.current_value == 15.0
        assert comparison.delta == 5.0
        assert comparison.delta_percentage == 50.0
        assert comparison.result == ComparisonResult.REGRESSED

    def test_from_values_improved(self):
        """Test from_values with improvement."""
        comparison = MetricComparison.from_values(
            name="frame_time",
            baseline=20.0,
            current=15.0,
            higher_is_worse=True,
        )
        assert comparison.result == ComparisonResult.IMPROVED
        assert comparison.delta == -5.0
        assert comparison.delta_percentage == -25.0

    def test_from_values_regressed(self):
        """Test from_values with regression."""
        comparison = MetricComparison.from_values(
            name="frame_time",
            baseline=10.0,
            current=20.0,
            higher_is_worse=True,
        )
        assert comparison.result == ComparisonResult.REGRESSED
        assert comparison.delta == 10.0
        assert comparison.delta_percentage == 100.0

    def test_from_values_unchanged(self):
        """Test from_values with unchanged value."""
        comparison = MetricComparison.from_values(
            name="frame_time",
            baseline=10.0,
            current=10.05,  # Within 1% threshold
            higher_is_worse=True,
        )
        assert comparison.result == ComparisonResult.UNCHANGED

    def test_from_values_new_metric(self):
        """Test from_values with new metric."""
        comparison = MetricComparison.from_values(
            name="new_metric",
            baseline=None,
            current=10.0,
        )
        assert comparison.result == ComparisonResult.NEW
        assert comparison.baseline_value == 0.0
        assert comparison.current_value == 10.0

    def test_from_values_removed_metric(self):
        """Test from_values with removed metric."""
        comparison = MetricComparison.from_values(
            name="old_metric",
            baseline=10.0,
            current=None,
        )
        assert comparison.result == ComparisonResult.REMOVED
        assert comparison.baseline_value == 10.0
        assert comparison.current_value == 0.0

    def test_from_values_both_none(self):
        """Test from_values with both values None."""
        comparison = MetricComparison.from_values(
            name="missing",
            baseline=None,
            current=None,
        )
        assert comparison.result == ComparisonResult.UNCHANGED
        assert comparison.delta == 0.0

    def test_from_values_higher_is_better(self):
        """Test from_values when higher is better (e.g., FPS)."""
        comparison = MetricComparison.from_values(
            name="fps",
            baseline=60.0,
            current=45.0,
            higher_is_worse=False,
        )
        assert comparison.result == ComparisonResult.REGRESSED

    def test_from_values_with_severity_minor(self):
        """Test severity detection - minor."""
        thresholds = {
            RegressionSeverity.MINOR: 5.0,
            RegressionSeverity.MODERATE: 20.0,
            RegressionSeverity.SEVERE: 50.0,
            RegressionSeverity.CRITICAL: 100.0,
        }
        comparison = MetricComparison.from_values(
            name="test",
            baseline=100.0,
            current=110.0,
            regression_thresholds=thresholds,
            higher_is_worse=True,
        )
        assert comparison.severity == RegressionSeverity.MINOR

    def test_from_values_with_severity_moderate(self):
        """Test severity detection - moderate."""
        thresholds = {
            RegressionSeverity.MINOR: 5.0,
            RegressionSeverity.MODERATE: 20.0,
            RegressionSeverity.SEVERE: 50.0,
            RegressionSeverity.CRITICAL: 100.0,
        }
        comparison = MetricComparison.from_values(
            name="test",
            baseline=100.0,
            current=130.0,
            regression_thresholds=thresholds,
            higher_is_worse=True,
        )
        assert comparison.severity == RegressionSeverity.MODERATE

    def test_from_values_with_severity_severe(self):
        """Test severity detection - severe."""
        thresholds = {
            RegressionSeverity.MINOR: 5.0,
            RegressionSeverity.MODERATE: 20.0,
            RegressionSeverity.SEVERE: 50.0,
            RegressionSeverity.CRITICAL: 100.0,
        }
        comparison = MetricComparison.from_values(
            name="test",
            baseline=100.0,
            current=170.0,
            regression_thresholds=thresholds,
            higher_is_worse=True,
        )
        assert comparison.severity == RegressionSeverity.SEVERE

    def test_from_values_with_severity_critical(self):
        """Test severity detection - critical."""
        thresholds = {
            RegressionSeverity.MINOR: 5.0,
            RegressionSeverity.MODERATE: 20.0,
            RegressionSeverity.SEVERE: 50.0,
            RegressionSeverity.CRITICAL: 100.0,
        }
        comparison = MetricComparison.from_values(
            name="test",
            baseline=100.0,
            current=250.0,
            regression_thresholds=thresholds,
            higher_is_worse=True,
        )
        assert comparison.severity == RegressionSeverity.CRITICAL

    def test_from_values_zero_baseline(self):
        """Test from_values with zero baseline."""
        comparison = MetricComparison.from_values(
            name="test",
            baseline=0.0,
            current=10.0,
        )
        assert comparison.delta_percentage == 0.0  # Avoid division by zero

    def test_to_dict(self):
        """Test dictionary conversion."""
        comparison = MetricComparison(
            name="test",
            baseline_value=10.0,
            current_value=15.0,
            delta=5.0,
            delta_percentage=50.0,
            result=ComparisonResult.REGRESSED,
            severity=RegressionSeverity.MODERATE,
        )
        data = comparison.to_dict()

        assert data["name"] == "test"
        assert data["baseline_value"] == 10.0
        assert data["current_value"] == 15.0
        assert data["delta"] == 5.0
        assert data["delta_percentage"] == 50.0
        assert data["result"] == "REGRESSED"
        assert data["severity"] == "MODERATE"


class TestSessionDiff:
    """Tests for SessionDiff."""

    def test_creation(self):
        """Test basic creation."""
        diff = SessionDiff(
            baseline_id="baseline_v1",
            current_id="current_v2",
        )
        assert diff.baseline_id == "baseline_v1"
        assert diff.current_id == "current_v2"
        assert diff.overall_result == ComparisonResult.UNCHANGED
        assert diff.regression_count == 0
        assert diff.improvement_count == 0

    def test_add_cpu_comparison(self):
        """Test adding CPU comparison."""
        diff = SessionDiff(baseline_id="b", current_id="c")
        comparison = MetricComparison(
            name="cpu.update",
            baseline_value=10.0,
            current_value=15.0,
            delta=5.0,
            delta_percentage=50.0,
            result=ComparisonResult.REGRESSED,
        )

        diff.add_cpu_comparison(comparison)

        assert len(diff.cpu_comparisons) == 1
        assert diff.regression_count == 1
        assert diff.overall_result == ComparisonResult.REGRESSED

    def test_add_gpu_comparison(self):
        """Test adding GPU comparison."""
        diff = SessionDiff(baseline_id="b", current_id="c")
        comparison = MetricComparison(
            name="gpu.shadow",
            baseline_value=5.0,
            current_value=4.0,
            delta=-1.0,
            delta_percentage=-20.0,
            result=ComparisonResult.IMPROVED,
        )

        diff.add_gpu_comparison(comparison)

        assert len(diff.gpu_comparisons) == 1
        assert diff.improvement_count == 1
        assert diff.overall_result == ComparisonResult.IMPROVED

    def test_add_memory_comparison(self):
        """Test adding memory comparison."""
        diff = SessionDiff(baseline_id="b", current_id="c")
        comparison = MetricComparison(
            name="memory.peak",
            baseline_value=256.0,
            current_value=256.0,
            delta=0.0,
            delta_percentage=0.0,
            result=ComparisonResult.UNCHANGED,
        )

        diff.add_memory_comparison(comparison)

        assert len(diff.memory_comparisons) == 1
        assert diff.regression_count == 0
        assert diff.improvement_count == 0

    def test_add_frame_comparison(self):
        """Test adding frame comparison."""
        diff = SessionDiff(baseline_id="b", current_id="c")
        comparison = MetricComparison(
            name="frame.avg",
            baseline_value=16.67,
            current_value=20.0,
            delta=3.33,
            delta_percentage=20.0,
            result=ComparisonResult.REGRESSED,
        )

        diff.add_frame_comparison(comparison)

        assert len(diff.frame_comparisons) == 1

    def test_get_regressions(self):
        """Test getting regressions."""
        diff = SessionDiff(baseline_id="b", current_id="c")

        # Add various comparisons
        diff.add_cpu_comparison(MetricComparison(
            name="cpu.1",
            baseline_value=10.0,
            current_value=15.0,
            delta=5.0,
            delta_percentage=50.0,
            result=ComparisonResult.REGRESSED,
            severity=RegressionSeverity.MODERATE,
        ))
        diff.add_cpu_comparison(MetricComparison(
            name="cpu.2",
            baseline_value=10.0,
            current_value=8.0,
            delta=-2.0,
            delta_percentage=-20.0,
            result=ComparisonResult.IMPROVED,
        ))

        regressions = diff.get_regressions()
        assert len(regressions) == 1
        assert regressions[0].name == "cpu.1"

    def test_get_regressions_with_severity_filter(self):
        """Test getting regressions with severity filter."""
        diff = SessionDiff(baseline_id="b", current_id="c")

        diff.add_cpu_comparison(MetricComparison(
            name="minor_regression",
            baseline_value=10.0,
            current_value=11.0,
            delta=1.0,
            delta_percentage=10.0,
            result=ComparisonResult.REGRESSED,
            severity=RegressionSeverity.MINOR,
        ))
        diff.add_cpu_comparison(MetricComparison(
            name="severe_regression",
            baseline_value=10.0,
            current_value=20.0,
            delta=10.0,
            delta_percentage=100.0,
            result=ComparisonResult.REGRESSED,
            severity=RegressionSeverity.SEVERE,
        ))

        severe_only = diff.get_regressions(min_severity=RegressionSeverity.SEVERE)
        assert len(severe_only) == 1
        assert severe_only[0].name == "severe_regression"

    def test_get_improvements(self):
        """Test getting improvements."""
        diff = SessionDiff(baseline_id="b", current_id="c")

        diff.add_cpu_comparison(MetricComparison(
            name="improved",
            baseline_value=20.0,
            current_value=15.0,
            delta=-5.0,
            delta_percentage=-25.0,
            result=ComparisonResult.IMPROVED,
        ))
        diff.add_cpu_comparison(MetricComparison(
            name="regressed",
            baseline_value=10.0,
            current_value=15.0,
            delta=5.0,
            delta_percentage=50.0,
            result=ComparisonResult.REGRESSED,
        ))

        improvements = diff.get_improvements()
        assert len(improvements) == 1
        assert improvements[0].name == "improved"

    def test_to_dict(self):
        """Test dictionary conversion."""
        diff = SessionDiff(baseline_id="baseline", current_id="current")
        diff.add_cpu_comparison(MetricComparison(
            name="test",
            baseline_value=10.0,
            current_value=10.0,
            delta=0.0,
            delta_percentage=0.0,
            result=ComparisonResult.UNCHANGED,
        ))

        data = diff.to_dict()

        assert data["baseline_id"] == "baseline"
        assert data["current_id"] == "current"
        assert "cpu_comparisons" in data
        assert len(data["cpu_comparisons"]) == 1


class TestComparisonReport:
    """Tests for ComparisonReport."""

    def test_creation(self):
        """Test basic creation."""
        report = ComparisonReport(title="Test Report")
        assert report.title == "Test Report"
        assert report.session_diff is None
        assert report.summary == ""
        assert report.recommendations == []

    def test_with_session_diff(self):
        """Test creation with session diff."""
        diff = SessionDiff(baseline_id="b", current_id="c")
        report = ComparisonReport(
            title="Performance Report",
            session_diff=diff,
        )
        assert report.session_diff is diff

    def test_generate_summary_no_data(self):
        """Test summary generation with no data."""
        report = ComparisonReport(title="Empty Report")
        summary = report.generate_summary()
        assert "No comparison data available" in summary

    def test_generate_summary_with_data(self):
        """Test summary generation with data."""
        diff = SessionDiff(baseline_id="baseline_v1", current_id="current_v2")
        diff.add_cpu_comparison(MetricComparison(
            name="cpu.update",
            baseline_value=10.0,
            current_value=15.0,
            delta=5.0,
            delta_percentage=50.0,
            result=ComparisonResult.REGRESSED,
            severity=RegressionSeverity.MODERATE,
        ))

        report = ComparisonReport(title="Test", session_diff=diff)
        summary = report.generate_summary()

        assert "Performance Comparison Report" in summary
        assert "baseline_v1" in summary
        assert "current_v2" in summary
        assert "Regressions" in summary

    def test_add_recommendation(self):
        """Test adding recommendations."""
        report = ComparisonReport(title="Test")
        report.add_recommendation("Optimize the render loop")
        report.add_recommendation("Reduce memory allocations")

        assert len(report.recommendations) == 2
        assert "Optimize the render loop" in report.recommendations

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = ComparisonReport(title="Test Report")
        report.add_recommendation("Test recommendation")

        data = report.to_dict()

        assert data["title"] == "Test Report"
        assert "recommendations" in data
        assert len(data["recommendations"]) == 1

    def test_to_markdown(self):
        """Test Markdown generation."""
        diff = SessionDiff(baseline_id="v1", current_id="v2")
        diff.add_cpu_comparison(MetricComparison(
            name="cpu.test",
            baseline_value=10.0,
            current_value=8.0,
            delta=-2.0,
            delta_percentage=-20.0,
            result=ComparisonResult.IMPROVED,
        ))

        report = ComparisonReport(title="Perf Report", session_diff=diff)
        report.add_recommendation("Keep up the good work")

        markdown = report.to_markdown()

        assert "# Perf Report" in markdown
        assert "## Summary" in markdown
        assert "## Improvements" in markdown
        assert "## Recommendations" in markdown
        assert "Keep up the good work" in markdown


class TestRegressionThresholds:
    """Tests for RegressionThresholds."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = RegressionThresholds()
        assert thresholds.minor == 5.0
        assert thresholds.moderate == 20.0
        assert thresholds.severe == 50.0
        assert thresholds.critical == 100.0

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = RegressionThresholds(
            minor=10.0,
            moderate=30.0,
            severe=60.0,
            critical=150.0,
        )
        assert thresholds.minor == 10.0
        assert thresholds.moderate == 30.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        thresholds = RegressionThresholds()
        data = thresholds.to_dict()

        assert data[RegressionSeverity.MINOR] == 5.0
        assert data[RegressionSeverity.MODERATE] == 20.0
        assert data[RegressionSeverity.SEVERE] == 50.0
        assert data[RegressionSeverity.CRITICAL] == 100.0


class TestRegressionDetector:
    """Tests for RegressionDetector."""

    @pytest.fixture
    def detector(self):
        """Create a fresh detector instance."""
        return RegressionDetector()

    def test_detect_no_regression(self, detector):
        """Test detection with no regression."""
        baseline = {"metric1": 10.0, "metric2": 20.0}
        current = {"metric1": 10.0, "metric2": 20.0}

        comparisons = detector.detect(baseline, current)

        assert len(comparisons) == 2
        assert all(c.result == ComparisonResult.UNCHANGED for c in comparisons)

    def test_detect_regression(self, detector):
        """Test detection with regression."""
        baseline = {"metric1": 10.0}
        current = {"metric1": 20.0}

        comparisons = detector.detect(baseline, current)

        assert len(comparisons) == 1
        assert comparisons[0].result == ComparisonResult.REGRESSED

    def test_detect_improvement(self, detector):
        """Test detection with improvement."""
        baseline = {"metric1": 20.0}
        current = {"metric1": 10.0}

        comparisons = detector.detect(baseline, current, higher_is_worse=True)

        assert len(comparisons) == 1
        assert comparisons[0].result == ComparisonResult.IMPROVED

    def test_detect_new_metrics(self, detector):
        """Test detection with new metrics."""
        baseline = {"metric1": 10.0}
        current = {"metric1": 10.0, "metric2": 20.0}

        comparisons = detector.detect(baseline, current)

        new_metrics = [c for c in comparisons if c.result == ComparisonResult.NEW]
        assert len(new_metrics) == 1
        assert new_metrics[0].name == "metric2"

    def test_detect_removed_metrics(self, detector):
        """Test detection with removed metrics."""
        baseline = {"metric1": 10.0, "metric2": 20.0}
        current = {"metric1": 10.0}

        comparisons = detector.detect(baseline, current)

        removed_metrics = [c for c in comparisons if c.result == ComparisonResult.REMOVED]
        assert len(removed_metrics) == 1
        assert removed_metrics[0].name == "metric2"

    def test_has_severe_regression_true(self, detector):
        """Test has_severe_regression returns True."""
        comparisons = [
            MetricComparison(
                name="test",
                baseline_value=10.0,
                current_value=25.0,
                delta=15.0,
                delta_percentage=150.0,
                result=ComparisonResult.REGRESSED,
                severity=RegressionSeverity.CRITICAL,
            ),
        ]

        assert detector.has_severe_regression(comparisons, RegressionSeverity.SEVERE)

    def test_has_severe_regression_false(self, detector):
        """Test has_severe_regression returns False."""
        comparisons = [
            MetricComparison(
                name="test",
                baseline_value=10.0,
                current_value=11.0,
                delta=1.0,
                delta_percentage=10.0,
                result=ComparisonResult.REGRESSED,
                severity=RegressionSeverity.MINOR,
            ),
        ]

        assert not detector.has_severe_regression(comparisons, RegressionSeverity.SEVERE)

    def test_custom_thresholds(self):
        """Test detector with custom thresholds."""
        thresholds = RegressionThresholds(minor=1.0, moderate=5.0, severe=10.0, critical=20.0)
        detector = RegressionDetector(thresholds)

        baseline = {"metric1": 100.0}
        current = {"metric1": 103.0}  # 3% increase

        comparisons = detector.detect(baseline, current)

        # With minor=1.0, a 3% increase should be MINOR regression
        assert comparisons[0].result == ComparisonResult.REGRESSED
        assert comparisons[0].severity == RegressionSeverity.MINOR


class TestProfilerComparator:
    """Tests for ProfilerComparator."""

    @pytest.fixture
    def comparator(self):
        """Create a fresh comparator instance."""
        return ProfilerComparator()

    def test_compare_empty_sessions(self, comparator):
        """Test comparing empty sessions."""
        baseline = {}
        current = {}

        diff = comparator.compare(baseline, current)

        assert diff.baseline_id == "baseline"
        assert diff.current_id == "current"
        assert diff.regression_count == 0

    def test_compare_cpu_metrics(self, comparator):
        """Test comparing CPU metrics."""
        baseline = {
            "cpu": {
                "stats": {
                    "update": {"avg_time_ms": 5.0},
                },
            },
        }
        current = {
            "cpu": {
                "stats": {
                    "update": {"avg_time_ms": 10.0},
                },
            },
        }

        diff = comparator.compare(baseline, current)

        assert len(diff.cpu_comparisons) > 0
        cpu_comp = diff.cpu_comparisons[0]
        assert "cpu.update" in cpu_comp.name

    def test_compare_gpu_metrics(self, comparator):
        """Test comparing GPU metrics."""
        baseline = {
            "gpu": {
                "pass_timings": {
                    "shadow": {"gpu_time_ms": 2.0},
                },
                "draw_stats": {"total_draw_calls": 500},
            },
        }
        current = {
            "gpu": {
                "pass_timings": {
                    "shadow": {"gpu_time_ms": 3.0},
                },
                "draw_stats": {"total_draw_calls": 600},
            },
        }

        diff = comparator.compare(baseline, current)

        assert len(diff.gpu_comparisons) > 0

    def test_compare_memory_metrics(self, comparator):
        """Test comparing memory metrics."""
        baseline = {
            "memory": {
                "current_usage_mb": 256.0,
                "peak_usage_mb": 300.0,
            },
        }
        current = {
            "memory": {
                "current_usage_mb": 280.0,
                "peak_usage_mb": 350.0,
            },
        }

        diff = comparator.compare(baseline, current)

        assert len(diff.memory_comparisons) > 0

    def test_compare_frame_metrics(self, comparator):
        """Test comparing frame metrics."""
        baseline = {
            "frames": {
                "stats": {
                    "avg_frame_time_ms": 16.67,
                    "spike_count": 5,
                    "current_fps": 60.0,
                },
            },
        }
        current = {
            "frames": {
                "stats": {
                    "avg_frame_time_ms": 20.0,
                    "spike_count": 10,
                    "current_fps": 50.0,
                },
            },
        }

        diff = comparator.compare(baseline, current)

        assert len(diff.frame_comparisons) > 0

    def test_compare_with_custom_ids(self, comparator):
        """Test comparison with custom session IDs."""
        diff = comparator.compare(
            baseline={},
            current={},
            baseline_id="release_1.0",
            current_id="develop",
        )

        assert diff.baseline_id == "release_1.0"
        assert diff.current_id == "develop"

    def test_generate_report(self, comparator):
        """Test report generation."""
        baseline = {
            "cpu": {
                "stats": {
                    "update": {"avg_time_ms": 5.0},
                },
            },
        }
        current = {
            "cpu": {
                "stats": {
                    "update": {"avg_time_ms": 15.0},  # 200% increase
                },
            },
        }

        diff = comparator.compare(baseline, current)
        report = comparator.generate_report(diff, "Performance Analysis")

        assert report.title == "Performance Analysis"
        assert report.session_diff is diff
        assert report.summary != ""

    def test_generate_report_with_recommendations(self, comparator):
        """Test report generation with recommendations."""
        baseline = {
            "cpu": {"stats": {"update": {"avg_time_ms": 5.0}}},
            "gpu": {
                "pass_timings": {"shadow": {"gpu_time_ms": 2.0}},
                "draw_stats": {},
            },
            "memory": {"current_usage_mb": 100.0, "peak_usage_mb": 100.0},
        }
        current = {
            "cpu": {"stats": {"update": {"avg_time_ms": 15.0}}},  # 200% regression
            "gpu": {
                "pass_timings": {"shadow": {"gpu_time_ms": 6.0}},  # 200% regression
                "draw_stats": {},
            },
            "memory": {"current_usage_mb": 250.0, "peak_usage_mb": 250.0},  # 150% increase
        }

        diff = comparator.compare(baseline, current)
        report = comparator.generate_report(diff)

        assert len(report.recommendations) > 0

    def test_compare_files(self, comparator):
        """Test comparing session files."""
        baseline_data = {
            "cpu": {"stats": {"test": {"avg_time_ms": 5.0}}},
        }
        current_data = {
            "cpu": {"stats": {"test": {"avg_time_ms": 8.0}}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            current_path = Path(tmpdir) / "current.json"

            with open(baseline_path, "w") as f:
                json.dump(baseline_data, f)
            with open(current_path, "w") as f:
                json.dump(current_data, f)

            diff = comparator.compare_files(baseline_path, current_path)

            assert diff.baseline_id == "baseline.json"
            assert diff.current_id == "current.json"
            assert len(diff.cpu_comparisons) > 0


class TestComparisonIntegration:
    """Integration tests for comparison functionality."""

    def test_full_comparison_workflow(self):
        """Test complete comparison workflow."""
        # Create baseline and current session data
        baseline = {
            "cpu": {
                "stats": {
                    "game_update": {"avg_time_ms": 5.0},
                    "physics": {"avg_time_ms": 3.0},
                    "ai": {"avg_time_ms": 2.0},
                },
            },
            "gpu": {
                "pass_timings": {
                    "shadow": {"gpu_time_ms": 2.5},
                    "forward": {"gpu_time_ms": 5.0},
                },
                "draw_stats": {"total_draw_calls": 1000},
            },
            "memory": {
                "current_usage_mb": 256.0,
                "peak_usage_mb": 300.0,
            },
            "frames": {
                "stats": {
                    "avg_frame_time_ms": 16.67,
                    "spike_count": 2,
                    "current_fps": 60.0,
                },
            },
        }

        current = {
            "cpu": {
                "stats": {
                    "game_update": {"avg_time_ms": 7.0},  # Regression
                    "physics": {"avg_time_ms": 2.5},  # Improvement
                    "ai": {"avg_time_ms": 2.0},  # Unchanged
                },
            },
            "gpu": {
                "pass_timings": {
                    "shadow": {"gpu_time_ms": 3.0},  # Regression
                    "forward": {"gpu_time_ms": 4.5},  # Improvement
                },
                "draw_stats": {"total_draw_calls": 1200},  # Regression
            },
            "memory": {
                "current_usage_mb": 280.0,  # Regression
                "peak_usage_mb": 320.0,  # Regression
            },
            "frames": {
                "stats": {
                    "avg_frame_time_ms": 18.0,  # Regression
                    "spike_count": 5,  # Regression
                    "current_fps": 55.0,  # Regression
                },
            },
        }

        # Run comparison
        comparator = ProfilerComparator()
        diff = comparator.compare(baseline, current, "v1.0.0", "v1.1.0-dev")

        # Verify diff
        assert diff.baseline_id == "v1.0.0"
        assert diff.current_id == "v1.1.0-dev"
        assert diff.regression_count > 0
        assert diff.improvement_count > 0

        # Generate report
        report = comparator.generate_report(diff, "Version Comparison")

        # Verify report
        assert report.title == "Version Comparison"
        summary = report.generate_summary()
        assert "v1.0.0" in summary
        assert "v1.1.0-dev" in summary

        # Check Markdown output
        markdown = report.to_markdown()
        assert "# Version Comparison" in markdown
        assert "Regressions" in markdown or "Improvements" in markdown

    def test_fps_regression_detection(self):
        """Test that FPS regression is correctly detected (higher is better)."""
        baseline = {
            "frames": {"stats": {"current_fps": 60.0}},
        }
        current = {
            "frames": {"stats": {"current_fps": 45.0}},  # 25% drop
        }

        comparator = ProfilerComparator()
        diff = comparator.compare(baseline, current)

        fps_comparison = [c for c in diff.frame_comparisons if "fps" in c.name][0]
        assert fps_comparison.result == ComparisonResult.REGRESSED
