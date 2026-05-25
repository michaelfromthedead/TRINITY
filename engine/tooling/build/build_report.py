"""Build reports with timing, warnings, and errors.

Provides build reporting functionality including timing information,
warning/error collection, and various output formats.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple
import html
import json
import os
import sys
import time


class BuildSeverity(Enum):
    """Severity level for build messages."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    FATAL = auto()


@dataclass
class BuildMessage:
    """A message from the build process."""
    severity: BuildSeverity
    message: str
    source: str = ""
    line: int = 0
    column: int = 0
    code: str = ""
    category: str = ""
    timestamp: float = field(default_factory=time.time)

    def format(self, show_location: bool = True) -> str:
        """Format the message for display."""
        parts = []

        if show_location and self.source:
            location = self.source
            if self.line > 0:
                location += f":{self.line}"
                if self.column > 0:
                    location += f":{self.column}"
            parts.append(location)

        parts.append(self.severity.name.lower())

        if self.code:
            parts.append(f"[{self.code}]")

        parts.append(self.message)

        return ": ".join(parts)


@dataclass
class BuildTiming:
    """Timing information for a build stage."""
    name: str
    start_time: float
    end_time: float = 0.0
    children: List[BuildTiming] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000

    def add_child(self, timing: BuildTiming) -> None:
        """Add a child timing."""
        self.children.append(timing)

    def stop(self) -> None:
        """Stop the timer."""
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "elapsed_ms": self.elapsed_ms,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class BuildStatistics:
    """Statistics about a build."""
    files_processed: int = 0
    files_cached: int = 0
    files_compiled: int = 0
    files_linked: int = 0
    files_skipped: int = 0
    total_size_input: int = 0
    total_size_output: int = 0
    peak_memory_mb: float = 0.0
    cpu_time_seconds: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.files_cached + self.files_compiled
        if total == 0:
            return 0.0
        return self.files_cached / total

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.total_size_input == 0:
            return 1.0
        return self.total_size_output / self.total_size_input

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "files_processed": self.files_processed,
            "files_cached": self.files_cached,
            "files_compiled": self.files_compiled,
            "files_linked": self.files_linked,
            "files_skipped": self.files_skipped,
            "total_size_input": self.total_size_input,
            "total_size_output": self.total_size_output,
            "peak_memory_mb": self.peak_memory_mb,
            "cpu_time_seconds": self.cpu_time_seconds,
            "cache_hit_rate": self.cache_hit_rate,
            "compression_ratio": self.compression_ratio,
        }


class BuildReport:
    """Comprehensive build report."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.success: bool = False
        self.messages: List[BuildMessage] = []
        self.timings: List[BuildTiming] = []
        self.statistics: BuildStatistics = BuildStatistics()
        self._timing_stack: List[BuildTiming] = []
        self._metadata: Dict[str, Any] = {}

    @property
    def elapsed(self) -> float:
        """Get total elapsed time."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def elapsed_formatted(self) -> str:
        """Get formatted elapsed time."""
        elapsed = self.elapsed
        if elapsed < 60:
            return f"{elapsed:.2f}s"
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.2f}s"

    @property
    def error_count(self) -> int:
        """Get count of error messages."""
        return sum(1 for m in self.messages if m.severity in (BuildSeverity.ERROR, BuildSeverity.FATAL))

    @property
    def warning_count(self) -> int:
        """Get count of warning messages."""
        return sum(1 for m in self.messages if m.severity == BuildSeverity.WARNING)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value."""
        return self._metadata.get(key, default)

    def add_message(
        self,
        severity: BuildSeverity,
        message: str,
        source: str = "",
        line: int = 0,
        column: int = 0,
        code: str = "",
        category: str = ""
    ) -> BuildMessage:
        """Add a build message."""
        msg = BuildMessage(
            severity=severity,
            message=message,
            source=source,
            line=line,
            column=column,
            code=code,
            category=category,
        )
        self.messages.append(msg)
        return msg

    def debug(self, message: str, **kwargs) -> BuildMessage:
        """Add a debug message."""
        return self.add_message(BuildSeverity.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> BuildMessage:
        """Add an info message."""
        return self.add_message(BuildSeverity.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> BuildMessage:
        """Add a warning message."""
        return self.add_message(BuildSeverity.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> BuildMessage:
        """Add an error message."""
        return self.add_message(BuildSeverity.ERROR, message, **kwargs)

    def fatal(self, message: str, **kwargs) -> BuildMessage:
        """Add a fatal error message."""
        return self.add_message(BuildSeverity.FATAL, message, **kwargs)

    def start_timing(self, name: str) -> BuildTiming:
        """Start a timing block."""
        timing = BuildTiming(name=name, start_time=time.time())

        if self._timing_stack:
            self._timing_stack[-1].add_child(timing)
        else:
            self.timings.append(timing)

        self._timing_stack.append(timing)
        return timing

    def stop_timing(self) -> Optional[BuildTiming]:
        """Stop the current timing block."""
        if self._timing_stack:
            timing = self._timing_stack.pop()
            timing.stop()
            return timing
        return None

    def finish(self, success: bool) -> None:
        """Mark the build as finished."""
        if self.end_time is None:
            self.end_time = time.time()
        self.success = success

        # Close any open timings
        while self._timing_stack:
            self.stop_timing()

    def get_messages_by_severity(self, severity: BuildSeverity) -> List[BuildMessage]:
        """Get messages filtered by severity."""
        return [m for m in self.messages if m.severity == severity]

    def get_messages_by_source(self, source: str) -> List[BuildMessage]:
        """Get messages filtered by source file."""
        return [m for m in self.messages if m.source == source]

    def get_slowest_stages(self, count: int = 5) -> List[BuildTiming]:
        """Get the slowest build stages."""
        all_timings = self._flatten_timings(self.timings)
        all_timings.sort(key=lambda t: t.elapsed, reverse=True)
        return all_timings[:count]

    def _flatten_timings(self, timings: List[BuildTiming]) -> List[BuildTiming]:
        """Flatten timing tree to a list."""
        result = []
        for timing in timings:
            result.append(timing)
            result.extend(self._flatten_timings(timing.children))
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "success": self.success,
            "elapsed_seconds": self.elapsed,
            "elapsed_formatted": self.elapsed_formatted,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "messages": [
                {
                    "severity": m.severity.name,
                    "message": m.message,
                    "source": m.source,
                    "line": m.line,
                    "column": m.column,
                    "code": m.code,
                    "category": m.category,
                }
                for m in self.messages
            ],
            "timings": [t.to_dict() for t in self.timings],
            "statistics": self.statistics.to_dict(),
            "metadata": self._metadata,
        }


class ReportFormatter(ABC):
    """Abstract base for report formatters."""

    @abstractmethod
    def format(self, report: BuildReport) -> str:
        """Format the report as a string."""
        pass

    def write(self, report: BuildReport, output: TextIO) -> None:
        """Write the formatted report to an output stream."""
        output.write(self.format(report))


class TextReportFormatter(ReportFormatter):
    """Plain text report formatter."""

    def __init__(self, verbose: bool = False, use_color: bool = True):
        self.verbose = verbose
        self.use_color = use_color and sys.stdout.isatty()

    def _colorize(self, text: str, severity: BuildSeverity) -> str:
        """Add color to text based on severity."""
        if not self.use_color:
            return text

        colors = {
            BuildSeverity.DEBUG: "\033[90m",    # Gray
            BuildSeverity.INFO: "\033[37m",     # White
            BuildSeverity.WARNING: "\033[33m",  # Yellow
            BuildSeverity.ERROR: "\033[31m",    # Red
            BuildSeverity.FATAL: "\033[91m",    # Bright red
        }
        reset = "\033[0m"
        return f"{colors.get(severity, '')}{text}{reset}"

    def format(self, report: BuildReport) -> str:
        """Format the report as plain text."""
        lines = []

        # Header
        status = "SUCCESS" if report.success else "FAILED"
        status_color = "\033[32m" if report.success else "\033[31m"
        reset = "\033[0m" if self.use_color else ""

        if self.use_color:
            lines.append(f"Build: {report.name} - {status_color}{status}{reset}")
        else:
            lines.append(f"Build: {report.name} - {status}")

        lines.append(f"Time: {report.elapsed_formatted}")
        lines.append(f"Errors: {report.error_count}, Warnings: {report.warning_count}")
        lines.append("")

        # Statistics
        stats = report.statistics
        lines.append("Statistics:")
        lines.append(f"  Files processed: {stats.files_processed}")
        lines.append(f"  Files cached: {stats.files_cached}")
        lines.append(f"  Files compiled: {stats.files_compiled}")
        lines.append(f"  Cache hit rate: {stats.cache_hit_rate:.1%}")
        lines.append("")

        # Errors and Warnings
        if report.error_count > 0:
            lines.append("Errors:")
            for msg in report.get_messages_by_severity(BuildSeverity.ERROR):
                lines.append(f"  {self._colorize(msg.format(), msg.severity)}")
            for msg in report.get_messages_by_severity(BuildSeverity.FATAL):
                lines.append(f"  {self._colorize(msg.format(), msg.severity)}")
            lines.append("")

        if report.warning_count > 0:
            lines.append("Warnings:")
            for msg in report.get_messages_by_severity(BuildSeverity.WARNING):
                lines.append(f"  {self._colorize(msg.format(), msg.severity)}")
            lines.append("")

        # Timing breakdown
        if self.verbose and report.timings:
            lines.append("Timing Breakdown:")
            for timing in report.timings:
                self._format_timing(timing, lines, indent=2)
            lines.append("")

        return "\n".join(lines)

    def _format_timing(self, timing: BuildTiming, lines: List[str], indent: int = 0) -> None:
        """Format a timing entry recursively."""
        prefix = " " * indent
        lines.append(f"{prefix}{timing.name}: {timing.elapsed_ms:.1f}ms")
        for child in timing.children:
            self._format_timing(child, lines, indent + 2)


class JSONReportFormatter(ReportFormatter):
    """JSON report formatter."""

    def __init__(self, pretty: bool = True):
        self.pretty = pretty

    def format(self, report: BuildReport) -> str:
        """Format the report as JSON."""
        data = report.to_dict()
        if self.pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data)


class HTMLReportFormatter(ReportFormatter):
    """HTML report formatter."""

    def format(self, report: BuildReport) -> str:
        """Format the report as HTML."""
        lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "  <meta charset='utf-8'>",
            f"  <title>Build Report: {html.escape(report.name)}</title>",
            "  <style>",
            "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }",
            "    .success { color: #22c55e; }",
            "    .failed { color: #ef4444; }",
            "    .warning { color: #f59e0b; }",
            "    .error { color: #ef4444; }",
            "    .timing { margin-left: 20px; }",
            "    table { border-collapse: collapse; width: 100%; }",
            "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "    th { background-color: #f3f4f6; }",
            "    .message { font-family: monospace; font-size: 13px; }",
            "  </style>",
            "</head>",
            "<body>",
            f"  <h1>Build Report: {html.escape(report.name)}</h1>",
        ]

        # Status
        status_class = "success" if report.success else "failed"
        status_text = "SUCCESS" if report.success else "FAILED"
        lines.append(f"  <p>Status: <span class='{status_class}'><strong>{status_text}</strong></span></p>")
        lines.append(f"  <p>Duration: {report.elapsed_formatted}</p>")
        lines.append(f"  <p>Errors: {report.error_count}, Warnings: {report.warning_count}</p>")

        # Statistics
        lines.append("  <h2>Statistics</h2>")
        lines.append("  <table>")
        stats = report.statistics.to_dict()
        for key, value in stats.items():
            formatted_key = key.replace("_", " ").title()
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            lines.append(f"    <tr><td>{formatted_key}</td><td>{formatted_value}</td></tr>")
        lines.append("  </table>")

        # Messages
        if report.messages:
            lines.append("  <h2>Messages</h2>")
            lines.append("  <table>")
            lines.append("    <tr><th>Severity</th><th>Source</th><th>Message</th></tr>")
            for msg in report.messages:
                severity_class = msg.severity.name.lower()
                source = f"{html.escape(msg.source)}:{msg.line}" if msg.source else "-"
                lines.append(
                    f"    <tr><td class='{severity_class}'>{msg.severity.name}</td>"
                    f"<td>{source}</td>"
                    f"<td class='message'>{html.escape(msg.message)}</td></tr>"
                )
            lines.append("  </table>")

        # Timing
        if report.timings:
            lines.append("  <h2>Timing</h2>")
            lines.append("  <ul>")
            for timing in report.timings:
                self._format_timing_html(timing, lines)
            lines.append("  </ul>")

        lines.extend([
            "</body>",
            "</html>",
        ])

        return "\n".join(lines)

    def _format_timing_html(self, timing: BuildTiming, lines: List[str]) -> None:
        """Format timing as HTML list items."""
        lines.append(f"    <li>{html.escape(timing.name)}: {timing.elapsed_ms:.1f}ms")
        if timing.children:
            lines.append("      <ul>")
            for child in timing.children:
                self._format_timing_html(child, lines)
            lines.append("      </ul>")
        lines.append("    </li>")


class ReportAggregator:
    """Aggregates multiple build reports."""

    def __init__(self):
        self._reports: List[BuildReport] = []

    def add(self, report: BuildReport) -> None:
        """Add a report."""
        self._reports.append(report)

    def get_all(self) -> List[BuildReport]:
        """Get all reports."""
        return list(self._reports)

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate summary."""
        total_time = sum(r.elapsed for r in self._reports)
        total_errors = sum(r.error_count for r in self._reports)
        total_warnings = sum(r.warning_count for r in self._reports)
        success_count = sum(1 for r in self._reports if r.success)

        return {
            "report_count": len(self._reports),
            "success_count": success_count,
            "failure_count": len(self._reports) - success_count,
            "total_time_seconds": total_time,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "success_rate": success_count / len(self._reports) if self._reports else 0,
        }

    def get_slowest_builds(self, count: int = 5) -> List[BuildReport]:
        """Get the slowest builds."""
        sorted_reports = sorted(self._reports, key=lambda r: r.elapsed, reverse=True)
        return sorted_reports[:count]

    def get_failed_builds(self) -> List[BuildReport]:
        """Get all failed builds."""
        return [r for r in self._reports if not r.success]

    def clear(self) -> None:
        """Clear all reports."""
        self._reports.clear()
