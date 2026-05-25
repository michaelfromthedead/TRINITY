"""
Tests for test_reporting.py - Test reports with JUnit XML and HTML output.
"""

import json
import time

import pytest

from engine.tooling.testing.test_reporting import (
    ConsoleReporter,
    HTMLReporter,
    JSONReporter,
    JUnitReporter,
    TestReport,
    TestReporter,
    generate_report,
)
from engine.tooling.testing.test_runner import TestResult, TestStatus


class TestTestReport:
    """Tests for TestReport dataclass."""

    def test_create_report(self):
        report = TestReport(name="Test Report")
        assert report.name == "Test Report"
        assert report.total == 0

    def test_report_properties(self):
        report = TestReport(name="Test Report")
        report.results = [
            TestResult("t1", TestStatus.PASSED),
            TestResult("t2", TestStatus.PASSED),
            TestResult("t3", TestStatus.FAILED),
            TestResult("t4", TestStatus.SKIPPED),
        ]

        assert report.total == 4
        assert report.passed == 2
        assert report.failed == 1
        assert report.skipped == 1

    def test_success_rate(self):
        report = TestReport(name="Test Report")
        report.results = [
            TestResult("t1", TestStatus.PASSED),
            TestResult("t2", TestStatus.PASSED),
            TestResult("t3", TestStatus.FAILED),
            TestResult("t4", TestStatus.PASSED),
        ]

        assert report.success_rate == 75.0

    def test_success_property(self):
        report = TestReport(name="Test Report")
        report.results = [
            TestResult("t1", TestStatus.PASSED),
            TestResult("t2", TestStatus.PASSED),
        ]
        assert report.success is True

        report.results.append(TestResult("t3", TestStatus.FAILED))
        assert report.success is False

    def test_to_dict(self):
        report = TestReport(
            name="Test Report",
            start_time=1000.0,
            end_time=1010.0,
        )
        report.results = [TestResult("t1", TestStatus.PASSED)]

        data = report.to_dict()
        assert data["name"] == "Test Report"
        assert data["total"] == 1
        assert data["duration"] == 10.0


class TestJUnitReporter:
    """Tests for JUnitReporter class."""

    def test_generate_basic(self):
        reporter = JUnitReporter()
        report = TestReport(name="JUnit Test")
        report.results = [
            TestResult("suite.test_pass", TestStatus.PASSED, duration=0.1),
        ]

        xml = reporter.generate(report)

        assert "<?xml" in xml
        assert "testsuites" in xml
        assert "testsuite" in xml
        assert "testcase" in xml

    def test_generate_with_failure(self):
        reporter = JUnitReporter()
        report = TestReport(name="JUnit Test")
        report.results = [
            TestResult(
                "suite.test_fail",
                TestStatus.FAILED,
                message="Assertion failed",
                traceback="File test.py, line 1",
            ),
        ]

        xml = reporter.generate(report)

        assert "<failure" in xml
        assert "Assertion failed" in xml

    def test_generate_with_error(self):
        reporter = JUnitReporter()
        report = TestReport(name="JUnit Test")
        report.results = [
            TestResult(
                "suite.test_error",
                TestStatus.ERROR,
                message="RuntimeError: Something broke",
            ),
        ]

        xml = reporter.generate(report)

        assert "<error" in xml

    def test_generate_with_skipped(self):
        reporter = JUnitReporter()
        report = TestReport(name="JUnit Test")
        report.results = [
            TestResult(
                "suite.test_skip",
                TestStatus.SKIPPED,
                message="Not implemented",
            ),
        ]

        xml = reporter.generate(report)

        assert "<skipped" in xml

    def test_multiple_suites(self):
        reporter = JUnitReporter()
        report = TestReport(name="Multi Suite")
        report.results = [
            TestResult("suite1.test_a", TestStatus.PASSED),
            TestResult("suite1.test_b", TestStatus.PASSED),
            TestResult("suite2.test_c", TestStatus.PASSED),
        ]

        xml = reporter.generate(report)

        # Should have multiple testsuite elements
        assert xml.count("<testsuite") >= 2


class TestHTMLReporter:
    """Tests for HTMLReporter class."""

    def test_generate_basic(self):
        reporter = HTMLReporter()
        report = TestReport(name="HTML Test")
        report.results = [
            TestResult("test_pass", TestStatus.PASSED),
        ]

        html = reporter.generate(report)

        assert "<!DOCTYPE html>" in html
        assert "HTML Test" in html

    def test_generate_with_styling(self):
        reporter = HTMLReporter()
        report = TestReport(name="HTML Test")
        report.results = []

        html = reporter.generate(report)

        assert "<style>" in html
        assert "</style>" in html

    def test_generate_with_results_table(self):
        reporter = HTMLReporter()
        report = TestReport(name="HTML Test")
        report.results = [
            TestResult("test_one", TestStatus.PASSED),
            TestResult("test_two", TestStatus.FAILED, message="Error message"),
        ]

        html = reporter.generate(report)

        assert "<table>" in html
        assert "test_one" in html
        assert "test_two" in html

    def test_generate_with_summary(self):
        reporter = HTMLReporter()
        report = TestReport(name="HTML Test")
        report.results = [
            TestResult("t1", TestStatus.PASSED),
            TestResult("t2", TestStatus.FAILED),
        ]

        html = reporter.generate(report)

        assert "Total" in html
        assert "Passed" in html
        assert "Failed" in html


class TestConsoleReporter:
    """Tests for ConsoleReporter class."""

    def test_generate_basic(self):
        reporter = ConsoleReporter(use_color=False)
        report = TestReport(name="Console Test")
        report.results = [
            TestResult("test_pass", TestStatus.PASSED),
        ]

        output = reporter.generate(report)

        assert "Console Test" in output
        assert "Passed" in output

    def test_generate_with_color(self):
        reporter = ConsoleReporter(use_color=True)
        report = TestReport(name="Console Test")
        report.results = [
            TestResult("test_pass", TestStatus.PASSED),
        ]

        output = reporter.generate(report)

        # Should contain ANSI escape codes
        assert "\033[" in output

    def test_generate_without_color(self):
        reporter = ConsoleReporter(use_color=False)
        report = TestReport(name="Console Test")
        report.results = [
            TestResult("test_pass", TestStatus.PASSED),
        ]

        output = reporter.generate(report)

        # Should not contain ANSI escape codes
        assert "\033[" not in output

    def test_failed_tests_listed(self):
        reporter = ConsoleReporter(use_color=False)
        report = TestReport(name="Console Test")
        report.results = [
            TestResult(
                "test_fail",
                TestStatus.FAILED,
                message="Assertion error",
            ),
        ]

        output = reporter.generate(report)

        assert "Failed Tests:" in output
        assert "test_fail" in output


class TestJSONReporter:
    """Tests for JSONReporter class."""

    def test_generate_basic(self):
        reporter = JSONReporter()
        report = TestReport(name="JSON Test")
        report.results = [
            TestResult("test_one", TestStatus.PASSED),
        ]

        json_str = reporter.generate(report)
        data = json.loads(json_str)

        assert data["name"] == "JSON Test"
        assert len(data["results"]) == 1

    def test_generate_pretty(self):
        reporter = JSONReporter(pretty=True)
        report = TestReport(name="JSON Test")
        report.results = []

        json_str = reporter.generate(report)

        # Pretty JSON should have newlines
        assert "\n" in json_str

    def test_generate_compact(self):
        reporter = JSONReporter(pretty=False)
        report = TestReport(name="JSON Test")
        report.results = []

        json_str = reporter.generate(report)

        # Compact JSON should not have indentation
        assert "  " not in json_str

    def test_all_fields_included(self):
        reporter = JSONReporter()
        report = TestReport(
            name="JSON Test",
            start_time=1000.0,
            end_time=1010.0,
            environment={"os": "Linux"},
        )
        report.results = [
            TestResult("test", TestStatus.PASSED, duration=0.5),
        ]

        json_str = reporter.generate(report)
        data = json.loads(json_str)

        assert "name" in data
        assert "total" in data
        assert "passed" in data
        assert "failed" in data
        assert "duration" in data
        assert "environment" in data
        assert "results" in data


class TestGenerateReportFunction:
    """Tests for generate_report convenience function."""

    def test_generate_console_report(self):
        results = [TestResult("test", TestStatus.PASSED)]
        output = generate_report(results, name="Test", format="console")

        assert "Test" in output

    def test_generate_html_report(self):
        results = [TestResult("test", TestStatus.PASSED)]
        output = generate_report(results, name="Test", format="html")

        assert "<!DOCTYPE html>" in output

    def test_generate_junit_report(self):
        results = [TestResult("test", TestStatus.PASSED)]
        output = generate_report(results, name="Test", format="junit")

        assert "<?xml" in output

    def test_generate_json_report(self):
        results = [TestResult("test", TestStatus.PASSED)]
        output = generate_report(results, name="Test", format="json")

        data = json.loads(output)
        assert data["name"] == "Test"

    def test_generate_with_output_path(self, tmp_path):
        results = [TestResult("test", TestStatus.PASSED)]
        output_file = tmp_path / "report.html"

        generate_report(
            results,
            name="Test",
            format="html",
            output_path=str(output_file),
        )

        assert output_file.exists()
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content
