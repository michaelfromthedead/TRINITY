"""Tests for build reports with timing, warnings, errors."""
import pytest
import json
import time
from engine.tooling.build.build_report import (
    BuildSeverity,
    BuildMessage,
    BuildTiming,
    BuildStatistics,
    BuildReport,
    ReportFormatter,
    TextReportFormatter,
    JSONReportFormatter,
    HTMLReportFormatter,
    ReportAggregator,
)


class TestBuildSeverity:
    """Tests for BuildSeverity enum."""

    def test_all_severities_exist(self):
        """Test all severity levels exist."""
        assert BuildSeverity.DEBUG
        assert BuildSeverity.INFO
        assert BuildSeverity.WARNING
        assert BuildSeverity.ERROR
        assert BuildSeverity.FATAL

    def test_severity_ordering(self):
        """Test severity levels are properly ordered."""
        assert BuildSeverity.DEBUG.value < BuildSeverity.INFO.value
        assert BuildSeverity.INFO.value < BuildSeverity.WARNING.value
        assert BuildSeverity.WARNING.value < BuildSeverity.ERROR.value
        assert BuildSeverity.ERROR.value < BuildSeverity.FATAL.value


class TestBuildMessage:
    """Tests for BuildMessage dataclass."""

    def test_message_creation(self):
        """Test creating build message."""
        msg = BuildMessage(
            severity=BuildSeverity.ERROR,
            message="Undefined symbol",
            source="main.cpp",
            line=42,
        )
        assert msg.severity == BuildSeverity.ERROR
        assert msg.line == 42

    def test_format_with_location(self):
        """Test formatting message with location."""
        msg = BuildMessage(
            severity=BuildSeverity.WARNING,
            message="Unused variable",
            source="util.cpp",
            line=10,
            column=5,
        )
        formatted = msg.format(show_location=True)
        assert "util.cpp:10:5" in formatted
        assert "warning" in formatted
        assert "Unused variable" in formatted

    def test_format_without_location(self):
        """Test formatting message without location."""
        msg = BuildMessage(
            severity=BuildSeverity.INFO,
            message="Build started",
        )
        formatted = msg.format(show_location=False)
        assert "info" in formatted
        assert "Build started" in formatted

    def test_format_with_code(self):
        """Test formatting message with error code."""
        msg = BuildMessage(
            severity=BuildSeverity.ERROR,
            message="Type mismatch",
            code="C2440",
        )
        formatted = msg.format()
        assert "[C2440]" in formatted


class TestBuildTiming:
    """Tests for BuildTiming dataclass."""

    def test_timing_creation(self):
        """Test creating timing object."""
        timing = BuildTiming(name="compile", start_time=time.time())
        assert timing.name == "compile"
        assert timing.end_time == 0.0

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        start = time.time()
        timing = BuildTiming(name="test", start_time=start)
        time.sleep(0.1)
        timing.stop()

        assert timing.elapsed >= 0.1
        assert timing.elapsed_ms >= 100

    def test_add_child_timing(self):
        """Test adding child timing."""
        parent = BuildTiming(name="build", start_time=time.time())
        child = BuildTiming(name="compile", start_time=time.time())

        parent.add_child(child)
        assert len(parent.children) == 1
        assert parent.children[0].name == "compile"

    def test_to_dict(self):
        """Test converting to dictionary."""
        timing = BuildTiming(name="link", start_time=time.time())
        timing.stop()

        data = timing.to_dict()
        assert data["name"] == "link"
        assert "elapsed_ms" in data
        assert "children" in data


class TestBuildStatistics:
    """Tests for BuildStatistics dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = BuildStatistics()
        assert stats.files_processed == 0
        assert stats.files_cached == 0

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        stats = BuildStatistics(
            files_cached=8,
            files_compiled=2,
        )
        assert stats.cache_hit_rate == 0.8

    def test_cache_hit_rate_zero_files(self):
        """Test cache hit rate with no files."""
        stats = BuildStatistics()
        assert stats.cache_hit_rate == 0.0

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        stats = BuildStatistics(
            total_size_input=1000,
            total_size_output=500,
        )
        assert stats.compression_ratio == 0.5

    def test_to_dict(self):
        """Test converting to dictionary."""
        stats = BuildStatistics(files_processed=10)
        data = stats.to_dict()
        assert data["files_processed"] == 10
        assert "cache_hit_rate" in data


class TestBuildReport:
    """Tests for BuildReport."""

    def test_report_creation(self):
        """Test creating build report."""
        report = BuildReport("TestBuild")
        assert report.name == "TestBuild"
        assert report.success is False

    def test_add_message(self):
        """Test adding messages."""
        report = BuildReport("Test")
        msg = report.add_message(
            BuildSeverity.WARNING,
            "Deprecated function",
            source="old.cpp",
        )
        assert msg in report.messages

    def test_convenience_methods(self):
        """Test convenience message methods."""
        report = BuildReport("Test")
        report.debug("Debug info")
        report.info("Info message")
        report.warning("Warning message")
        report.error("Error message")
        report.fatal("Fatal error")

        assert len(report.messages) == 5

    def test_error_count(self):
        """Test error counting."""
        report = BuildReport("Test")
        report.warning("warn1")
        report.error("err1")
        report.error("err2")
        report.fatal("fatal1")

        assert report.error_count == 3

    def test_warning_count(self):
        """Test warning counting."""
        report = BuildReport("Test")
        report.warning("warn1")
        report.warning("warn2")
        report.error("err1")

        assert report.warning_count == 2

    def test_timing_operations(self):
        """Test timing start/stop."""
        report = BuildReport("Test")
        timing = report.start_timing("compile")
        time.sleep(0.05)
        stopped = report.stop_timing()

        assert timing is stopped
        assert timing.elapsed >= 0.05

    def test_nested_timings(self):
        """Test nested timing blocks."""
        report = BuildReport("Test")
        report.start_timing("build")
        report.start_timing("compile")
        report.stop_timing()
        report.start_timing("link")
        report.stop_timing()
        report.stop_timing()

        assert len(report.timings) == 1
        assert len(report.timings[0].children) == 2

    def test_finish(self):
        """Test finishing report."""
        report = BuildReport("Test")
        report.start_timing("test")
        report.finish(success=True)

        assert report.success is True
        assert report.end_time is not None

    def test_elapsed_formatted(self):
        """Test formatted elapsed time."""
        report = BuildReport("Test")
        report.end_time = report.start_time + 65.5  # 1m 5.5s

        formatted = report.elapsed_formatted
        assert "1m" in formatted
        assert "5.50s" in formatted

    def test_get_messages_by_severity(self):
        """Test filtering messages by severity."""
        report = BuildReport("Test")
        report.warning("warn1")
        report.warning("warn2")
        report.error("err1")

        warnings = report.get_messages_by_severity(BuildSeverity.WARNING)
        assert len(warnings) == 2

    def test_get_slowest_stages(self):
        """Test getting slowest stages."""
        report = BuildReport("Test")

        t1 = report.start_timing("fast")
        time.sleep(0.01)
        report.stop_timing()

        t2 = report.start_timing("slow")
        time.sleep(0.02)
        report.stop_timing()

        slowest = report.get_slowest_stages(1)
        assert slowest[0].name == "slow"

    def test_to_dict(self):
        """Test converting to dictionary."""
        report = BuildReport("Test")
        report.info("test message")
        report.finish(success=True)

        data = report.to_dict()
        assert data["name"] == "Test"
        assert data["success"] is True
        assert len(data["messages"]) == 1

    def test_metadata(self):
        """Test setting and getting metadata."""
        report = BuildReport("Test")
        report.set_metadata("platform", "windows")
        report.set_metadata("config", "Debug")

        assert report.get_metadata("platform") == "windows"
        assert report.get_metadata("nonexistent", "default") == "default"


class TestTextReportFormatter:
    """Tests for TextReportFormatter."""

    def test_format_success(self):
        """Test formatting successful build."""
        report = BuildReport("Test")
        report.finish(success=True)

        formatter = TextReportFormatter(use_color=False)
        output = formatter.format(report)

        assert "Test" in output
        assert "SUCCESS" in output

    def test_format_failure(self):
        """Test formatting failed build."""
        report = BuildReport("Test")
        report.error("Build failed")
        report.finish(success=False)

        formatter = TextReportFormatter(use_color=False)
        output = formatter.format(report)

        assert "FAILED" in output
        assert "Build failed" in output

    def test_format_with_timing(self):
        """Test formatting with timing info."""
        report = BuildReport("Test")
        report.start_timing("compile")
        report.stop_timing()
        report.finish(success=True)

        formatter = TextReportFormatter(verbose=True, use_color=False)
        output = formatter.format(report)

        assert "compile" in output


class TestJSONReportFormatter:
    """Tests for JSONReportFormatter."""

    def test_format_pretty(self):
        """Test pretty JSON formatting."""
        report = BuildReport("Test")
        report.finish(success=True)

        formatter = JSONReportFormatter(pretty=True)
        output = formatter.format(report)

        # Should be valid JSON
        data = json.loads(output)
        assert data["name"] == "Test"
        assert data["success"] is True

    def test_format_compact(self):
        """Test compact JSON formatting."""
        report = BuildReport("Test")
        report.finish(success=True)

        formatter = JSONReportFormatter(pretty=False)
        output = formatter.format(report)

        # Should be single line
        assert "\n" not in output.strip()


class TestHTMLReportFormatter:
    """Tests for HTMLReportFormatter."""

    def test_format_html(self):
        """Test HTML formatting."""
        report = BuildReport("Test")
        report.warning("A warning")
        report.error("An error")
        report.finish(success=False)

        formatter = HTMLReportFormatter()
        output = formatter.format(report)

        assert "<!DOCTYPE html>" in output
        assert "<html>" in output
        assert "Test" in output
        assert "A warning" in output
        assert "An error" in output

    def test_html_escaping(self):
        """Test HTML special character escaping."""
        report = BuildReport("Test<>")
        report.error("Error: <script>alert('xss')</script>")
        report.finish(success=False)

        formatter = HTMLReportFormatter()
        output = formatter.format(report)

        assert "<script>" not in output
        assert "&lt;script&gt;" in output or "script" not in output


class TestReportAggregator:
    """Tests for ReportAggregator."""

    def test_add_report(self):
        """Test adding reports."""
        aggregator = ReportAggregator()
        report = BuildReport("Test")
        report.finish(success=True)

        aggregator.add(report)
        assert len(aggregator.get_all()) == 1

    def test_get_summary(self):
        """Test getting summary statistics."""
        aggregator = ReportAggregator()

        r1 = BuildReport("Build1")
        r1.warning("warn")
        r1.finish(success=True)

        r2 = BuildReport("Build2")
        r2.error("err")
        r2.finish(success=False)

        aggregator.add(r1)
        aggregator.add(r2)

        summary = aggregator.get_summary()
        assert summary["report_count"] == 2
        assert summary["success_count"] == 1
        assert summary["failure_count"] == 1
        assert summary["total_warnings"] == 1
        assert summary["total_errors"] == 1

    def test_get_slowest_builds(self):
        """Test getting slowest builds."""
        aggregator = ReportAggregator()

        r1 = BuildReport("Fast")
        r1.end_time = r1.start_time + 1.0
        r1.finish(success=True)

        r2 = BuildReport("Slow")
        r2.end_time = r2.start_time + 5.0
        r2.finish(success=True)

        aggregator.add(r1)
        aggregator.add(r2)

        slowest = aggregator.get_slowest_builds(1)
        assert slowest[0].name == "Slow"

    def test_get_failed_builds(self):
        """Test getting failed builds."""
        aggregator = ReportAggregator()

        r1 = BuildReport("Success")
        r1.finish(success=True)

        r2 = BuildReport("Failed")
        r2.finish(success=False)

        aggregator.add(r1)
        aggregator.add(r2)

        failed = aggregator.get_failed_builds()
        assert len(failed) == 1
        assert failed[0].name == "Failed"

    def test_clear(self):
        """Test clearing aggregator."""
        aggregator = ReportAggregator()
        report = BuildReport("Test")
        report.finish(success=True)

        aggregator.add(report)
        aggregator.clear()

        assert len(aggregator.get_all()) == 0
