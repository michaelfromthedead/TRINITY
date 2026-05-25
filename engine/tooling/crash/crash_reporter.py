"""
Crash reporting with stack traces and minidumps.

Provides comprehensive crash reporting infrastructure for
capturing, processing, and reporting application crashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type


class CrashSeverity(Enum):
    """Severity level of a crash."""

    INFO = auto()      # Non-fatal information
    WARNING = auto()   # Warning that might lead to issues
    ERROR = auto()     # Error that affects functionality
    CRITICAL = auto()  # Critical error, partial functionality lost
    FATAL = auto()     # Fatal error, application cannot continue


@dataclass
class SystemInfo:
    """System information at time of crash."""

    os_name: str = ""
    os_version: str = ""
    architecture: str = ""
    cpu_count: int = 0
    memory_total: int = 0
    memory_available: int = 0
    gpu_info: str = ""
    python_version: str = ""
    process_id: int = 0
    thread_id: int = 0

    @classmethod
    def capture(cls) -> "SystemInfo":
        """Capture current system information."""
        import threading

        memory_total = 0
        memory_available = 0

        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_total = memory.total
            memory_available = memory.available
        except ImportError:
            pass

        return cls(
            os_name=platform.system(),
            os_version=platform.version(),
            architecture=platform.machine(),
            cpu_count=os.cpu_count() or 0,
            memory_total=memory_total,
            memory_available=memory_available,
            python_version=sys.version,
            process_id=os.getpid(),
            thread_id=threading.current_thread().ident or 0,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "os_name": self.os_name,
            "os_version": self.os_version,
            "architecture": self.architecture,
            "cpu_count": self.cpu_count,
            "memory_total": self.memory_total,
            "memory_available": self.memory_available,
            "gpu_info": self.gpu_info,
            "python_version": self.python_version,
            "process_id": self.process_id,
            "thread_id": self.thread_id,
        }


@dataclass
class StackFrame:
    """A single frame in a stack trace."""

    filename: str
    line_number: int
    function_name: str
    code_context: str = ""
    locals: Dict[str, str] = field(default_factory=dict)
    module: str = ""
    address: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "line_number": self.line_number,
            "function_name": self.function_name,
            "code_context": self.code_context,
            "module": self.module,
            "address": self.address,
        }

    def __str__(self) -> str:
        return f'  File "{self.filename}", line {self.line_number}, in {self.function_name}\n    {self.code_context}'


@dataclass
class ExceptionInfo:
    """Information about an exception."""

    exception_type: str
    exception_message: str
    stack_trace: List[StackFrame] = field(default_factory=list)
    cause: Optional["ExceptionInfo"] = None
    context: Optional["ExceptionInfo"] = None

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        tb: Optional[Any] = None,
        capture_locals: bool = False,
    ) -> "ExceptionInfo":
        """Create from an exception."""
        if tb is None:
            tb = exc.__traceback__

        frames = []
        if tb:
            for frame_info in traceback.extract_tb(tb):
                frame = StackFrame(
                    filename=frame_info.filename,
                    line_number=frame_info.lineno,
                    function_name=frame_info.name,
                    code_context=frame_info.line or "",
                )
                frames.append(frame)

        info = cls(
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            stack_trace=frames,
        )

        # Capture chained exceptions
        if exc.__cause__:
            info.cause = cls.from_exception(exc.__cause__, capture_locals=capture_locals)
        if exc.__context__ and exc.__context__ is not exc.__cause__:
            info.context = cls.from_exception(exc.__context__, capture_locals=capture_locals)

        return info

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "stack_trace": [f.to_dict() for f in self.stack_trace],
        }
        if self.cause:
            result["cause"] = self.cause.to_dict()
        if self.context:
            result["context"] = self.context.to_dict()
        return result

    def format_traceback(self) -> str:
        """Format as readable traceback."""
        lines = ["Traceback (most recent call last):"]
        for frame in self.stack_trace:
            lines.append(str(frame))
        lines.append(f"{self.exception_type}: {self.exception_message}")
        return "\n".join(lines)


@dataclass
class CrashContext:
    """Contextual information about a crash."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    build_version: str = ""
    build_config: str = ""
    game_state: Dict[str, Any] = field(default_factory=dict)
    user_actions: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "build_version": self.build_version,
            "build_config": self.build_config,
            "game_state": self.game_state,
            "user_actions": self.user_actions,
            "logs": self.logs[-100:],  # Last 100 log lines
            "custom_data": self.custom_data,
            "tags": list(self.tags),
        }


@dataclass
class CrashReport:
    """
    Complete crash report.

    Contains all information about a crash including
    exception details, system info, and context.
    """

    id: str
    timestamp: float
    severity: CrashSeverity
    exception_info: ExceptionInfo
    system_info: SystemInfo
    context: CrashContext
    minidump_path: Optional[str] = None
    screenshot_path: Optional[str] = None
    attachments: List[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        """
        Generate a fingerprint for crash grouping.

        Crashes with the same fingerprint are likely the same issue.
        """
        components = [
            self.exception_info.exception_type,
            self.exception_info.exception_message[:100],
        ]

        # Add top stack frames
        for frame in self.exception_info.stack_trace[-3:]:
            components.append(f"{frame.filename}:{frame.function_name}")

        fingerprint_str = "|".join(components)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_formatted": datetime.fromtimestamp(self.timestamp).isoformat(),
            "severity": self.severity.name,
            "fingerprint": self.fingerprint,
            "exception": self.exception_info.to_dict(),
            "system": self.system_info.to_dict(),
            "context": self.context.to_dict(),
            "minidump_path": self.minidump_path,
            "screenshot_path": self.screenshot_path,
            "attachments": self.attachments,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, directory: str) -> str:
        """Save crash report to directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        filename = f"crash_{self.id}.json"
        filepath = path / filename

        with open(filepath, "w") as f:
            f.write(self.to_json())

        return str(filepath)


class CrashReporter:
    """
    Main crash reporter class.

    Handles crash capture, processing, and reporting.
    """

    _instance: Optional["CrashReporter"] = None

    def __init__(
        self,
        output_directory: str = "./crashes",
        max_reports: int = 100,
        auto_upload: bool = False,
        upload_url: Optional[str] = None,
    ):
        self.output_directory = Path(output_directory)
        self.max_reports = max_reports
        self.auto_upload = auto_upload
        self.upload_url = upload_url
        self._context = CrashContext()
        self._hooks: List[Callable[[CrashReport], None]] = []
        self._filters: List[Callable[[CrashReport], bool]] = []
        self._reports: List[CrashReport] = []

        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def instance(cls) -> "CrashReporter":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_context(self, **kwargs) -> None:
        """Update crash context."""
        for key, value in kwargs.items():
            if hasattr(self._context, key):
                setattr(self._context, key, value)
            else:
                self._context.custom_data[key] = value

    def add_tag(self, tag: str) -> None:
        """Add a tag to the context."""
        self._context.tags.add(tag)

    def log_action(self, action: str) -> None:
        """Log a user action for context."""
        self._context.user_actions.append(f"[{time.strftime('%H:%M:%S')}] {action}")
        # Keep last 50 actions
        self._context.user_actions = self._context.user_actions[-50:]

    def log_message(self, message: str) -> None:
        """Log a message for context."""
        self._context.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # Keep last 100 messages
        self._context.logs = self._context.logs[-100:]

    def add_hook(self, hook: Callable[[CrashReport], None]) -> None:
        """Add a hook to be called on crash."""
        self._hooks.append(hook)

    def add_filter(self, filter_func: Callable[[CrashReport], bool]) -> None:
        """Add a filter for crash reports."""
        self._filters.append(filter_func)

    def capture_exception(
        self,
        exc: BaseException,
        severity: CrashSeverity = CrashSeverity.ERROR,
        **extra_context,
    ) -> CrashReport:
        """
        Capture an exception and create a crash report.

        Args:
            exc: The exception to capture
            severity: Severity level
            **extra_context: Additional context data

        Returns:
            The created crash report
        """
        # Capture exception info
        exc_info = ExceptionInfo.from_exception(exc)

        # Capture system info
        sys_info = SystemInfo.capture()

        # Create context copy with extras
        context = CrashContext(
            user_id=self._context.user_id,
            session_id=self._context.session_id,
            build_version=self._context.build_version,
            build_config=self._context.build_config,
            game_state=self._context.game_state.copy(),
            user_actions=self._context.user_actions.copy(),
            logs=self._context.logs.copy(),
            custom_data={**self._context.custom_data, **extra_context},
            tags=self._context.tags.copy(),
        )

        # Create report
        report = CrashReport(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            severity=severity,
            exception_info=exc_info,
            system_info=sys_info,
            context=context,
        )

        # Apply filters
        for filter_func in self._filters:
            if not filter_func(report):
                return report  # Filtered out

        # Save report
        report.save(str(self.output_directory))
        self._reports.append(report)

        # Cleanup old reports
        self._cleanup_old_reports()

        # Run hooks
        for hook in self._hooks:
            try:
                hook(report)
            except Exception:
                pass  # Don't fail on hook errors

        # Auto-upload if enabled
        if self.auto_upload and self.upload_url:
            self._upload_report(report)

        return report

    def report(
        self,
        message: str,
        severity: CrashSeverity = CrashSeverity.ERROR,
        **extra_context,
    ) -> CrashReport:
        """
        Create a crash report without an exception.

        Args:
            message: Error message
            severity: Severity level
            **extra_context: Additional context data

        Returns:
            The created crash report
        """
        # Create a synthetic exception for the stack trace
        try:
            raise RuntimeError(message)
        except RuntimeError as exc:
            return self.capture_exception(exc, severity, **extra_context)

    def _cleanup_old_reports(self) -> None:
        """Remove old reports to stay within limit."""
        if len(self._reports) > self.max_reports:
            self._reports = self._reports[-self.max_reports:]

    def _upload_report(self, report: CrashReport) -> bool:
        """Upload a report to the server."""
        if not self.upload_url:
            return False

        try:
            import urllib.request

            data = report.to_json().encode()
            request = urllib.request.Request(
                self.upload_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=10)
            return True
        except Exception:
            return False

    def get_reports(self) -> List[CrashReport]:
        """Get all captured reports."""
        return self._reports.copy()

    def install_exception_handler(self) -> None:
        """Install global exception handler."""
        original_hook = sys.excepthook

        def exception_handler(exc_type, exc_value, exc_traceback):
            self.capture_exception(exc_value, CrashSeverity.FATAL)
            original_hook(exc_type, exc_value, exc_traceback)

        sys.excepthook = exception_handler


# Global instance and convenience functions

_reporter: Optional[CrashReporter] = None


def initialize_crash_reporter(
    output_directory: str = "./crashes",
    auto_upload: bool = False,
    upload_url: Optional[str] = None,
    **kwargs,
) -> CrashReporter:
    """
    Initialize the global crash reporter.

    Args:
        output_directory: Directory to save crash reports
        auto_upload: Whether to automatically upload reports
        upload_url: URL to upload reports to
        **kwargs: Additional configuration

    Returns:
        The crash reporter instance
    """
    global _reporter
    _reporter = CrashReporter(
        output_directory=output_directory,
        auto_upload=auto_upload,
        upload_url=upload_url,
        **kwargs,
    )
    return _reporter


def report_crash(
    message: str,
    severity: CrashSeverity = CrashSeverity.ERROR,
    **context,
) -> CrashReport:
    """
    Report a crash.

    Args:
        message: Error message
        severity: Severity level
        **context: Additional context

    Returns:
        The crash report
    """
    global _reporter
    if _reporter is None:
        _reporter = CrashReporter()
    return _reporter.report(message, severity, **context)


def capture_exception(
    exc: BaseException,
    severity: CrashSeverity = CrashSeverity.ERROR,
    **context,
) -> CrashReport:
    """
    Capture an exception.

    Args:
        exc: The exception
        severity: Severity level
        **context: Additional context

    Returns:
        The crash report
    """
    global _reporter
    if _reporter is None:
        _reporter = CrashReporter()
    return _reporter.capture_exception(exc, severity, **context)
