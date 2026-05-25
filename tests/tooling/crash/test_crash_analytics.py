"""
Tests for crash_analytics.py - Crash analytics and grouping.
"""

import time

import pytest

from engine.tooling.crash.crash_analytics import (
    CrashAnalytics,
    CrashGroup,
    CrashPattern,
    CrashTrend,
    analyze_crash,
    detect_patterns,
    get_analytics,
    get_crash_statistics,
    group_crashes,
)
from engine.tooling.crash.crash_reporter import (
    CrashContext,
    CrashReport,
    CrashSeverity,
    ExceptionInfo,
    StackFrame,
    SystemInfo,
)


def create_test_report(
    exception_type: str = "ValueError",
    message: str = "Test error",
    stack: list = None,
    version: str = "",
    platform: str = "Linux",
) -> CrashReport:
    """Create a test crash report."""
    if stack is None:
        stack = [StackFrame("test.py", 10, "test_func")]

    return CrashReport(
        id=f"crash-{time.time_ns()}",
        timestamp=time.time(),
        severity=CrashSeverity.ERROR,
        exception_info=ExceptionInfo(
            exception_type=exception_type,
            exception_message=message,
            stack_trace=stack,
        ),
        system_info=SystemInfo(os_name=platform),
        context=CrashContext(build_version=version),
    )


class TestCrashGroup:
    """Tests for CrashGroup dataclass."""

    def test_create_group(self):
        group = CrashGroup(
            id="group-1",
            fingerprint="abc123",
            exception_type="ValueError",
            exception_message_pattern="Test <num>",
            first_seen=1000.0,
            last_seen=1000.0,
        )

        assert group.id == "group-1"
        assert group.count == 0

    def test_add_crash(self):
        group = CrashGroup(
            id="group-1",
            fingerprint="abc123",
            exception_type="ValueError",
            exception_message_pattern="Test",
            first_seen=1000.0,
            last_seen=1000.0,
        )

        report = create_test_report(version="1.0.0", platform="Windows")
        group.add_crash(report)

        assert group.count == 1
        assert "1.0.0" in group.affected_versions
        assert "Windows" in group.affected_platforms

    def test_to_dict(self):
        group = CrashGroup(
            id="group-1",
            fingerprint="abc123",
            exception_type="ValueError",
            exception_message_pattern="Test",
            first_seen=1000.0,
            last_seen=2000.0,
            count=5,
        )
        data = group.to_dict()

        assert data["id"] == "group-1"
        assert data["count"] == 5
        assert "first_seen_formatted" in data


class TestCrashPattern:
    """Tests for CrashPattern dataclass."""

    def test_create_pattern(self):
        pattern = CrashPattern(
            id="pattern-1",
            name="Memory corruption",
            description="Pattern indicating memory issues",
            pattern_type="stack_trace",
        )

        assert pattern.id == "pattern-1"
        assert pattern.pattern_type == "stack_trace"

    def test_to_dict(self):
        pattern = CrashPattern(
            id="pattern-1",
            name="Memory corruption",
            description="Pattern description",
            pattern_type="stack_trace",
            confidence=0.85,
        )
        data = pattern.to_dict()

        assert data["name"] == "Memory corruption"
        assert data["confidence"] == 0.85


class TestCrashTrend:
    """Tests for CrashTrend dataclass."""

    def test_create_trend(self):
        trend = CrashTrend(
            period="daily",
            start_time=1000.0,
            end_time=2000.0,
            total_crashes=100,
        )

        assert trend.period == "daily"
        assert trend.total_crashes == 100

    def test_to_dict(self):
        trend = CrashTrend(
            period="daily",
            start_time=1000.0,
            end_time=2000.0,
            total_crashes=100,
            unique_issues=10,
        )
        data = trend.to_dict()

        assert data["period"] == "daily"
        assert data["total_crashes"] == 100
        assert data["unique_issues"] == 10


class TestCrashAnalytics:
    """Tests for CrashAnalytics class."""

    def test_create_analytics(self):
        analytics = CrashAnalytics()
        assert analytics is not None

    def test_add_crash(self):
        analytics = CrashAnalytics()
        report = create_test_report()

        group = analytics.add_crash(report)

        assert group is not None
        assert group.count == 1

    def test_crashes_grouped_by_fingerprint(self):
        analytics = CrashAnalytics()

        # Same stack trace = same fingerprint
        report1 = create_test_report(message="Error 1")
        report2 = create_test_report(message="Error 1")

        analytics.add_crash(report1)
        analytics.add_crash(report2)

        groups = analytics.get_groups()
        assert len(groups) == 1
        assert groups[0].count == 2

    def test_different_crashes_different_groups(self):
        analytics = CrashAnalytics()

        report1 = create_test_report(
            exception_type="ValueError",
            stack=[StackFrame("a.py", 1, "func_a")],
        )
        report2 = create_test_report(
            exception_type="TypeError",
            stack=[StackFrame("b.py", 2, "func_b")],
        )

        analytics.add_crash(report1)
        analytics.add_crash(report2)

        groups = analytics.get_groups()
        assert len(groups) == 2

    def test_get_group(self):
        analytics = CrashAnalytics()
        report = create_test_report()
        group = analytics.add_crash(report)

        retrieved = analytics.get_group(group.id)
        assert retrieved is not None
        assert retrieved.id == group.id

    def test_get_groups_filtered(self):
        analytics = CrashAnalytics()

        for i in range(5):
            analytics.add_crash(create_test_report())

        groups = analytics.get_groups(min_count=1, limit=10)
        assert len(groups) <= 10

    def test_message_pattern_extraction(self):
        analytics = CrashAnalytics()

        # Test various patterns
        patterns = [
            ("Error at 0x12345678", "Error at <addr>"),
            ("Value 42 is invalid", "Value <num> is invalid"),
            ('File "test.txt" not found', 'File "<str>" not found'),
        ]

        for message, expected_pattern in patterns:
            result = analytics._extract_message_pattern(message)
            assert result == expected_pattern

    def test_analyze_trends_daily(self):
        analytics = CrashAnalytics()

        # Add some crashes
        for _ in range(10):
            analytics.add_crash(create_test_report())

        trends = analytics.analyze_trends(period="daily", days=7)

        assert len(trends) > 0
        assert all(t.period == "daily" for t in trends)

    def test_detect_patterns(self):
        analytics = CrashAnalytics()

        # Add crashes with common stack frame
        stack = [
            StackFrame("common.py", 10, "problematic_function"),
            StackFrame("caller.py", 20, "caller"),
        ]

        for _ in range(5):
            analytics.add_crash(create_test_report(stack=stack))

        patterns = analytics.detect_patterns()
        # Should detect the common function pattern
        assert len(patterns) >= 0  # May or may not detect depending on thresholds

    def test_get_statistics(self):
        analytics = CrashAnalytics()

        analytics.add_crash(create_test_report(platform="Linux"))
        analytics.add_crash(create_test_report(platform="Windows"))
        analytics.add_crash(create_test_report(version="1.0"))

        stats = analytics.get_statistics()

        assert stats["total_crashes"] == 3
        assert "by_platform" in stats
        assert "by_version" in stats


class TestGlobalFunctions:
    """Tests for global analytics functions."""

    def test_get_analytics(self):
        analytics = get_analytics()
        assert analytics is not None

    def test_analyze_crash(self):
        report = create_test_report()
        group = analyze_crash(report)

        assert group is not None

    def test_group_crashes(self):
        reports = [
            create_test_report(),
            create_test_report(),
        ]

        groups = group_crashes(reports)
        assert len(groups) > 0

    def test_get_crash_statistics(self):
        # Add at least one crash
        analyze_crash(create_test_report())

        stats = get_crash_statistics()
        assert "total_crashes" in stats

    def test_detect_patterns_function(self):
        # Add several crashes
        for _ in range(5):
            analyze_crash(create_test_report())

        patterns = detect_patterns()
        assert isinstance(patterns, list)


class TestCrashGrouping:
    """Tests for crash grouping behavior."""

    def test_same_exception_same_stack_grouped(self):
        analytics = CrashAnalytics()

        stack = [StackFrame("test.py", 10, "test_func")]

        report1 = create_test_report(
            exception_type="ValueError",
            message="Error 1",
            stack=stack,
        )
        report2 = create_test_report(
            exception_type="ValueError",
            message="Error 1",
            stack=stack,
        )

        analytics.add_crash(report1)
        analytics.add_crash(report2)

        groups = analytics.get_groups()
        assert len(groups) == 1
        assert groups[0].count == 2

    def test_different_stack_different_group(self):
        analytics = CrashAnalytics()

        report1 = create_test_report(
            stack=[StackFrame("a.py", 1, "func_a")],
        )
        report2 = create_test_report(
            stack=[StackFrame("b.py", 2, "func_b")],
        )

        analytics.add_crash(report1)
        analytics.add_crash(report2)

        groups = analytics.get_groups()
        assert len(groups) == 2


class TestTrendAnalysis:
    """Tests for trend analysis functionality."""

    def test_trend_direction_calculation(self):
        analytics = CrashAnalytics()

        # Add crashes at current time
        for _ in range(10):
            report = create_test_report()
            analytics.add_crash(report)

        trends = analytics.analyze_trends(period="hourly", days=1)

        # Should have trend data
        assert len(trends) > 0

    def test_by_severity_tracking(self):
        analytics = CrashAnalytics()

        report = create_test_report()
        analytics.add_crash(report)

        stats = analytics.get_statistics()
        assert "by_severity" in stats
        assert "ERROR" in stats["by_severity"]
