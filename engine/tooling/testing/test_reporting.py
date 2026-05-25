"""
Test reports with JUnit XML and HTML output.

Provides comprehensive test reporting in multiple formats
for CI/CD integration and human-readable output.
"""

from __future__ import annotations

import html
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union
from xml.etree import ElementTree as ET

from .test_runner import TestResult, TestStatus


@dataclass
class TestReport:
    """Complete test report with all results and metadata."""

    name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    environment: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Total duration of test run."""
        return self.end_time - self.start_time

    @property
    def total(self) -> int:
        """Total number of tests."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Number of passed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        """Number of failed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def errors(self) -> int:
        """Number of tests with errors."""
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)

    @property
    def skipped(self) -> int:
        """Number of skipped tests."""
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total == 0:
            return 100.0
        return (self.passed / self.total) * 100

    @property
    def success(self) -> bool:
        """Whether all tests passed."""
        return self.failed == 0 and self.errors == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration": self.duration,
            "success_rate": self.success_rate,
            "success": self.success,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "environment": self.environment,
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }


class TestReporter(ABC):
    """Base class for test reporters."""

    @abstractmethod
    def generate(self, report: TestReport) -> str:
        """Generate report content."""
        pass

    def write(self, report: TestReport, path: Union[str, Path]) -> None:
        """Write report to file."""
        content = self.generate(report)
        Path(path).write_text(content)

    def write_stream(self, report: TestReport, stream: TextIO) -> None:
        """Write report to stream."""
        content = self.generate(report)
        stream.write(content)


class JUnitReporter(TestReporter):
    """
    JUnit XML format reporter for CI/CD integration.

    Generates XML compatible with Jenkins, GitHub Actions, etc.
    """

    def generate(self, report: TestReport) -> str:
        """Generate JUnit XML report."""
        # Create root element
        testsuites = ET.Element("testsuites")
        testsuites.set("name", report.name)
        testsuites.set("tests", str(report.total))
        testsuites.set("failures", str(report.failed))
        testsuites.set("errors", str(report.errors))
        testsuites.set("skipped", str(report.skipped))
        testsuites.set("time", f"{report.duration:.3f}")

        # Group results by suite (based on test name prefix)
        suites: Dict[str, List[TestResult]] = {}
        for result in report.results:
            parts = result.name.rsplit(".", 1)
            suite_name = parts[0] if len(parts) > 1 else "default"
            if suite_name not in suites:
                suites[suite_name] = []
            suites[suite_name].append(result)

        # Create testsuite elements
        for suite_name, results in suites.items():
            testsuite = ET.SubElement(testsuites, "testsuite")
            testsuite.set("name", suite_name)
            testsuite.set("tests", str(len(results)))
            testsuite.set("failures", str(sum(1 for r in results if r.status == TestStatus.FAILED)))
            testsuite.set("errors", str(sum(1 for r in results if r.status == TestStatus.ERROR)))
            testsuite.set("skipped", str(sum(1 for r in results if r.status == TestStatus.SKIPPED)))
            testsuite.set("time", f"{sum(r.duration for r in results):.3f}")
            testsuite.set("timestamp", datetime.fromtimestamp(report.start_time).isoformat())

            # Add properties
            if report.environment:
                properties = ET.SubElement(testsuite, "properties")
                for key, value in report.environment.items():
                    prop = ET.SubElement(properties, "property")
                    prop.set("name", key)
                    prop.set("value", str(value))

            # Add test cases
            for result in results:
                testcase = ET.SubElement(testsuite, "testcase")
                testcase.set("name", result.name.rsplit(".", 1)[-1])
                testcase.set("classname", suite_name)
                testcase.set("time", f"{result.duration:.3f}")

                if result.status == TestStatus.FAILED:
                    failure = ET.SubElement(testcase, "failure")
                    failure.set("message", result.message)
                    failure.set("type", "AssertionError")
                    failure.text = result.traceback

                elif result.status == TestStatus.ERROR:
                    error = ET.SubElement(testcase, "error")
                    error.set("message", result.message)
                    error.set("type", "Exception")
                    error.text = result.traceback

                elif result.status == TestStatus.SKIPPED:
                    skipped = ET.SubElement(testcase, "skipped")
                    skipped.set("message", result.message)

                elif result.status == TestStatus.TIMEOUT:
                    error = ET.SubElement(testcase, "error")
                    error.set("message", result.message)
                    error.set("type", "TimeoutError")

                # Add system output
                if result.output:
                    system_out = ET.SubElement(testcase, "system-out")
                    system_out.text = result.output

        # Convert to string
        return ET.tostring(testsuites, encoding="unicode", xml_declaration=True)


class HTMLReporter(TestReporter):
    """
    HTML format reporter for human-readable output.

    Generates a standalone HTML report with styling.
    """

    def generate(self, report: TestReport) -> str:
        """Generate HTML report."""
        # CSS styles
        css = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }
        h1 { color: #333; margin-bottom: 10px; }
        .summary {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .stat-box {
            padding: 15px 25px;
            border-radius: 8px;
            text-align: center;
            min-width: 100px;
        }
        .stat-box.passed { background: #d4edda; color: #155724; }
        .stat-box.failed { background: #f8d7da; color: #721c24; }
        .stat-box.skipped { background: #fff3cd; color: #856404; }
        .stat-box.total { background: #e2e3e5; color: #383d41; }
        .stat-value { font-size: 2em; font-weight: bold; }
        .stat-label { font-size: 0.9em; opacity: 0.8; }
        .progress-bar {
            height: 20px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin: 20px 0;
        }
        .progress-fill {
            height: 100%;
            transition: width 0.3s;
        }
        .progress-fill.passed { background: #28a745; }
        .progress-fill.failed { background: #dc3545; }
        .progress-fill.skipped { background: #ffc107; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th { background: #f8f9fa; font-weight: 600; }
        tr:hover { background: #f8f9fa; }
        .status {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .status.passed { background: #d4edda; color: #155724; }
        .status.failed { background: #f8d7da; color: #721c24; }
        .status.error { background: #f8d7da; color: #721c24; }
        .status.skipped { background: #fff3cd; color: #856404; }
        .status.timeout { background: #f8d7da; color: #721c24; }
        .details { margin-top: 5px; font-size: 0.9em; color: #666; }
        .traceback {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.85em;
            overflow-x: auto;
            white-space: pre-wrap;
            margin-top: 10px;
        }
        .collapsible {
            cursor: pointer;
            user-select: none;
        }
        .collapsible:hover { text-decoration: underline; }
        .content { display: none; }
        .content.show { display: block; }
        .environment {
            margin-top: 30px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .environment h3 { margin-top: 0; }
        .env-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }
        .env-item { font-size: 0.9em; }
        .env-key { font-weight: 500; color: #666; }
        """

        # JavaScript
        js = """
        function toggleDetails(id) {
            var content = document.getElementById(id);
            content.classList.toggle('show');
        }
        """

        # Build HTML
        lines = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='UTF-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"  <title>Test Report - {html.escape(report.name)}</title>",
            f"  <style>{css}</style>",
            "</head>",
            "<body>",
            "  <div class='container'>",
            f"    <h1>Test Report: {html.escape(report.name)}</h1>",
            f"    <p>Generated: {datetime.fromtimestamp(report.end_time).strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "",
            "    <div class='summary'>",
            f"      <div class='stat-box total'><div class='stat-value'>{report.total}</div><div class='stat-label'>Total</div></div>",
            f"      <div class='stat-box passed'><div class='stat-value'>{report.passed}</div><div class='stat-label'>Passed</div></div>",
            f"      <div class='stat-box failed'><div class='stat-value'>{report.failed + report.errors}</div><div class='stat-label'>Failed</div></div>",
            f"      <div class='stat-box skipped'><div class='stat-value'>{report.skipped}</div><div class='stat-label'>Skipped</div></div>",
            "    </div>",
            "",
            "    <div class='progress-bar'>",
        ]

        # Calculate percentages
        if report.total > 0:
            passed_pct = (report.passed / report.total) * 100
            failed_pct = ((report.failed + report.errors) / report.total) * 100
            skipped_pct = (report.skipped / report.total) * 100
        else:
            passed_pct = failed_pct = skipped_pct = 0

        lines.extend([
            f"      <div class='progress-fill passed' style='width: {passed_pct}%; display: inline-block;'></div>",
            f"      <div class='progress-fill failed' style='width: {failed_pct}%; display: inline-block;'></div>",
            f"      <div class='progress-fill skipped' style='width: {skipped_pct}%; display: inline-block;'></div>",
            "    </div>",
            f"    <p>Success Rate: {report.success_rate:.1f}% | Duration: {report.duration:.2f}s</p>",
            "",
            "    <table>",
            "      <thead>",
            "        <tr>",
            "          <th>Test</th>",
            "          <th>Status</th>",
            "          <th>Duration</th>",
            "          <th>Details</th>",
            "        </tr>",
            "      </thead>",
            "      <tbody>",
        ])

        # Add test results
        for i, result in enumerate(report.results):
            status_class = result.status.name.lower()
            status_text = result.status.name

            lines.extend([
                "        <tr>",
                f"          <td>{html.escape(result.name)}</td>",
                f"          <td><span class='status {status_class}'>{status_text}</span></td>",
                f"          <td>{result.duration:.3f}s</td>",
                "          <td>",
            ])

            if result.message or result.traceback:
                lines.append(f"            <span class='collapsible' onclick='toggleDetails(\"details-{i}\")'>Show details</span>")
                lines.append(f"            <div id='details-{i}' class='content'>")
                if result.message:
                    lines.append(f"              <div class='details'>{html.escape(result.message)}</div>")
                if result.traceback:
                    lines.append(f"              <div class='traceback'>{html.escape(result.traceback)}</div>")
                lines.append("            </div>")

            lines.extend([
                "          </td>",
                "        </tr>",
            ])

        lines.extend([
            "      </tbody>",
            "    </table>",
        ])

        # Add environment info
        if report.environment:
            lines.extend([
                "",
                "    <div class='environment'>",
                "      <h3>Environment</h3>",
                "      <div class='env-grid'>",
            ])
            for key, value in report.environment.items():
                lines.append(f"        <div class='env-item'><span class='env-key'>{html.escape(key)}:</span> {html.escape(str(value))}</div>")
            lines.extend([
                "      </div>",
                "    </div>",
            ])

        lines.extend([
            "  </div>",
            f"  <script>{js}</script>",
            "</body>",
            "</html>",
        ])

        return "\n".join(lines)


class ConsoleReporter(TestReporter):
    """
    Console format reporter for terminal output.

    Generates colored text output for terminal display.
    """

    def __init__(self, use_color: bool = True):
        self.use_color = use_color

    def generate(self, report: TestReport) -> str:
        """Generate console report."""
        lines = []

        # Colors
        if self.use_color:
            GREEN = "\033[92m"
            RED = "\033[91m"
            YELLOW = "\033[93m"
            BLUE = "\033[94m"
            BOLD = "\033[1m"
            RESET = "\033[0m"
        else:
            GREEN = RED = YELLOW = BLUE = BOLD = RESET = ""

        # Header
        lines.append(f"\n{BOLD}{'=' * 60}{RESET}")
        lines.append(f"{BOLD}Test Report: {report.name}{RESET}")
        lines.append(f"{'=' * 60}")

        # Summary
        status_color = GREEN if report.success else RED
        lines.append(f"\n{BOLD}Summary:{RESET}")
        lines.append(f"  Total:   {report.total}")
        lines.append(f"  {GREEN}Passed:  {report.passed}{RESET}")
        lines.append(f"  {RED}Failed:  {report.failed}{RESET}")
        lines.append(f"  {RED}Errors:  {report.errors}{RESET}")
        lines.append(f"  {YELLOW}Skipped: {report.skipped}{RESET}")
        lines.append(f"  Duration: {report.duration:.2f}s")
        lines.append(f"  {status_color}Success Rate: {report.success_rate:.1f}%{RESET}")

        # Failed tests
        failed = [r for r in report.results if r.status in (TestStatus.FAILED, TestStatus.ERROR)]
        if failed:
            lines.append(f"\n{BOLD}{RED}Failed Tests:{RESET}")
            for result in failed:
                lines.append(f"\n  {RED}{result.name}{RESET}")
                if result.message:
                    lines.append(f"    {result.message}")
                if result.traceback:
                    for line in result.traceback.split("\n")[:5]:
                        lines.append(f"    {line}")

        # Skipped tests
        skipped = [r for r in report.results if r.status == TestStatus.SKIPPED]
        if skipped:
            lines.append(f"\n{BOLD}{YELLOW}Skipped Tests:{RESET}")
            for result in skipped:
                reason = f": {result.message}" if result.message else ""
                lines.append(f"  {YELLOW}{result.name}{reason}{RESET}")

        # Footer
        lines.append(f"\n{'=' * 60}")
        status_msg = f"{GREEN}PASSED{RESET}" if report.success else f"{RED}FAILED{RESET}"
        lines.append(f"{BOLD}Result: {status_msg}{RESET}")
        lines.append(f"{'=' * 60}\n")

        return "\n".join(lines)


class JSONReporter(TestReporter):
    """
    JSON format reporter for machine processing.

    Generates JSON output for programmatic consumption.
    """

    def __init__(self, pretty: bool = True):
        self.pretty = pretty

    def generate(self, report: TestReport) -> str:
        """Generate JSON report."""
        data = report.to_dict()

        if self.pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data)


def generate_report(
    results: List[TestResult],
    name: str = "Test Report",
    format: str = "console",
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> str:
    """
    Generate a test report in the specified format.

    Args:
        results: Test results to report
        name: Report name
        format: Output format (console, html, junit, json)
        output_path: Optional path to write report
        **kwargs: Additional arguments for reporter

    Returns:
        Generated report content
    """
    # Create report
    report = TestReport(
        name=name,
        results=results,
        start_time=kwargs.get("start_time", time.time()),
        end_time=kwargs.get("end_time", time.time()),
        environment=kwargs.get("environment", {}),
        metadata=kwargs.get("metadata", {}),
    )

    # Select reporter
    reporters = {
        "console": ConsoleReporter,
        "html": HTMLReporter,
        "junit": JUnitReporter,
        "json": JSONReporter,
    }

    reporter_class = reporters.get(format.lower(), ConsoleReporter)
    reporter = reporter_class()

    # Generate report
    content = reporter.generate(report)

    # Write to file if path provided
    if output_path:
        reporter.write(report, output_path)

    return content
