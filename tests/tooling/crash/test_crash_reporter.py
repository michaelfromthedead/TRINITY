"""
Tests for crash_reporter.py - Crash reporting with stack traces.
"""

import time

import pytest

from engine.tooling.crash.crash_reporter import (
    CrashContext,
    CrashReport,
    CrashReporter,
    CrashSeverity,
    ExceptionInfo,
    StackFrame,
    SystemInfo,
    capture_exception,
    initialize_crash_reporter,
    report_crash,
)


class TestCrashSeverity:
    """Tests for CrashSeverity enum."""

    def test_severity_values_exist(self):
        assert CrashSeverity.INFO
        assert CrashSeverity.WARNING
        assert CrashSeverity.ERROR
        assert CrashSeverity.CRITICAL
        assert CrashSeverity.FATAL


class TestSystemInfo:
    """Tests for SystemInfo dataclass."""

    def test_capture_system_info(self):
        info = SystemInfo.capture()

        assert info.os_name != ""
        assert info.python_version != ""
        assert info.process_id > 0

    def test_to_dict(self):
        info = SystemInfo.capture()
        data = info.to_dict()

        assert "os_name" in data
        assert "python_version" in data
        assert "process_id" in data


class TestStackFrame:
    """Tests for StackFrame dataclass."""

    def test_create_frame(self):
        frame = StackFrame(
            filename="test.py",
            line_number=10,
            function_name="test_func",
        )

        assert frame.filename == "test.py"
        assert frame.line_number == 10

    def test_to_dict(self):
        frame = StackFrame(
            filename="test.py",
            line_number=10,
            function_name="test_func",
            code_context="assert True",
        )
        data = frame.to_dict()

        assert data["filename"] == "test.py"
        assert data["line_number"] == 10

    def test_str_representation(self):
        frame = StackFrame(
            filename="test.py",
            line_number=10,
            function_name="test_func",
            code_context="assert True",
        )

        str_repr = str(frame)
        assert "test.py" in str_repr
        assert "10" in str_repr
        assert "test_func" in str_repr


class TestExceptionInfo:
    """Tests for ExceptionInfo dataclass."""

    def test_from_exception(self):
        try:
            raise ValueError("Test error")
        except ValueError as e:
            info = ExceptionInfo.from_exception(e)

        assert info.exception_type == "ValueError"
        assert info.exception_message == "Test error"
        assert len(info.stack_trace) > 0

    def test_chained_exception(self):
        try:
            try:
                raise ValueError("Original")
            except ValueError as e:
                raise RuntimeError("Wrapped") from e
        except RuntimeError as e:
            info = ExceptionInfo.from_exception(e)

        assert info.exception_type == "RuntimeError"
        assert info.cause is not None
        assert info.cause.exception_type == "ValueError"

    def test_to_dict(self):
        try:
            raise ValueError("Test error")
        except ValueError as e:
            info = ExceptionInfo.from_exception(e)

        data = info.to_dict()

        assert data["exception_type"] == "ValueError"
        assert data["exception_message"] == "Test error"
        assert "stack_trace" in data

    def test_format_traceback(self):
        try:
            raise ValueError("Test error")
        except ValueError as e:
            info = ExceptionInfo.from_exception(e)

        formatted = info.format_traceback()

        assert "Traceback" in formatted
        assert "ValueError: Test error" in formatted


class TestCrashContext:
    """Tests for CrashContext dataclass."""

    def test_create_context(self):
        context = CrashContext(
            user_id="user123",
            session_id="session456",
            build_version="1.0.0",
        )

        assert context.user_id == "user123"
        assert context.session_id == "session456"

    def test_to_dict(self):
        context = CrashContext(
            user_id="user123",
            build_version="1.0.0",
            tags={"critical", "ui"},
        )
        data = context.to_dict()

        assert data["user_id"] == "user123"
        assert data["build_version"] == "1.0.0"
        assert "critical" in data["tags"]


class TestCrashReport:
    """Tests for CrashReport dataclass."""

    def test_create_report(self):
        exc_info = ExceptionInfo(
            exception_type="ValueError",
            exception_message="Test",
        )
        sys_info = SystemInfo()
        context = CrashContext()

        report = CrashReport(
            id="crash-123",
            timestamp=time.time(),
            severity=CrashSeverity.ERROR,
            exception_info=exc_info,
            system_info=sys_info,
            context=context,
        )

        assert report.id == "crash-123"
        assert report.severity == CrashSeverity.ERROR

    def test_fingerprint(self):
        exc_info = ExceptionInfo(
            exception_type="ValueError",
            exception_message="Test error",
            stack_trace=[
                StackFrame("a.py", 1, "func_a"),
                StackFrame("b.py", 2, "func_b"),
            ],
        )

        report = CrashReport(
            id="crash-123",
            timestamp=time.time(),
            severity=CrashSeverity.ERROR,
            exception_info=exc_info,
            system_info=SystemInfo(),
            context=CrashContext(),
        )

        fingerprint = report.fingerprint
        assert len(fingerprint) == 32  # MD5 hex length

    def test_to_dict(self):
        exc_info = ExceptionInfo(
            exception_type="ValueError",
            exception_message="Test",
        )

        report = CrashReport(
            id="crash-123",
            timestamp=1000.0,
            severity=CrashSeverity.ERROR,
            exception_info=exc_info,
            system_info=SystemInfo(),
            context=CrashContext(),
        )

        data = report.to_dict()

        assert data["id"] == "crash-123"
        assert data["severity"] == "ERROR"
        assert "exception" in data
        assert "system" in data

    def test_to_json(self):
        exc_info = ExceptionInfo(
            exception_type="ValueError",
            exception_message="Test",
        )

        report = CrashReport(
            id="crash-123",
            timestamp=1000.0,
            severity=CrashSeverity.ERROR,
            exception_info=exc_info,
            system_info=SystemInfo(),
            context=CrashContext(),
        )

        json_str = report.to_json()

        assert '"id": "crash-123"' in json_str

    def test_save(self, tmp_path):
        exc_info = ExceptionInfo(
            exception_type="ValueError",
            exception_message="Test",
        )

        report = CrashReport(
            id="crash-123",
            timestamp=1000.0,
            severity=CrashSeverity.ERROR,
            exception_info=exc_info,
            system_info=SystemInfo(),
            context=CrashContext(),
        )

        filepath = report.save(str(tmp_path))

        assert (tmp_path / "crash_crash-123.json").exists()


class TestCrashReporter:
    """Tests for CrashReporter class."""

    def test_create_reporter(self, tmp_path):
        reporter = CrashReporter(output_directory=str(tmp_path))

        assert reporter.output_directory == tmp_path

    def test_set_context(self, tmp_path):
        reporter = CrashReporter(output_directory=str(tmp_path))
        reporter.set_context(
            user_id="user123",
            build_version="1.0.0",
        )

        assert reporter._context.user_id == "user123"
        assert reporter._context.build_version == "1.0.0"

    def test_add_tag(self, tmp_path):
        reporter = CrashReporter(output_directory=str(tmp_path))
        reporter.add_tag("critical")

        assert "critical" in reporter._context.tags

    def test_log_action(self, tmp_path):
        reporter = CrashReporter(output_directory=str(tmp_path))
        reporter.log_action("User clicked button")

        assert any("User clicked button" in a for a in reporter._context.user_actions)

    def test_log_message(self, tmp_path):
        reporter = CrashReporter(output_directory=str(tmp_path))
        reporter.log_message("Debug info")

        assert any("Debug info" in m for m in reporter._context.logs)

    def test_capture_exception(self, tmp_path):
        reporter = CrashReporter(output_directory=str(tmp_path))

        try:
            raise ValueError("Test error")
        except ValueError as e:
            report = reporter.capture_exception(e)

        assert report.exception_info.exception_type == "ValueError"
        assert len(reporter.get_reports()) == 1

    def test_report_without_exception(self, tmp_path):
        reporter = CrashReporter(output_directory=str(tmp_path))
        report = reporter.report("Something went wrong")

        assert "Something went wrong" in report.exception_info.exception_message

    def test_add_hook(self, tmp_path):
        reporter = CrashReporter(output_directory=str(tmp_path))
        hooks_called = []

        def on_crash(report):
            hooks_called.append(report.id)

        reporter.add_hook(on_crash)
        reporter.report("Test error")

        assert len(hooks_called) == 1

    def test_add_filter(self, tmp_path):
        reporter = CrashReporter(output_directory=str(tmp_path))

        # Filter out all reports
        reporter.add_filter(lambda r: False)
        reporter.report("Test error")

        # Report should still be created but might be filtered from processing
        # Filters run after creation

    def test_singleton_instance(self):
        instance1 = CrashReporter.instance()
        instance2 = CrashReporter.instance()

        assert instance1 is instance2


class TestGlobalFunctions:
    """Tests for global crash reporting functions."""

    def test_initialize_crash_reporter(self, tmp_path):
        reporter = initialize_crash_reporter(
            output_directory=str(tmp_path),
            auto_upload=False,
        )

        assert reporter is not None

    def test_report_crash(self, tmp_path):
        initialize_crash_reporter(output_directory=str(tmp_path))
        report = report_crash("Test error", severity=CrashSeverity.ERROR)

        assert report is not None
        assert "Test error" in report.exception_info.exception_message

    def test_capture_exception_function(self, tmp_path):
        initialize_crash_reporter(output_directory=str(tmp_path))

        try:
            raise ValueError("Test error")
        except ValueError as e:
            report = capture_exception(e)

        assert report.exception_info.exception_type == "ValueError"
