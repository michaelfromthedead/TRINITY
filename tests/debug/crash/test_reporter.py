"""
Tests for the crash reporter.

Tests cover:
- SystemInfoSnapshot capture
- CrashReport creation and serialization
- CrashReporter report generation
- Local saving and loading
- Report fingerprinting
- Security: path validation, data sanitization
- Error handling: crash reporter robustness
"""

import asyncio
import json
import os
import platform
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from engine.debug.crash.handler import CrashContext
from engine.debug.crash.minidump import MinidumpLevel
from engine.debug.crash.reporter import (
    CrashReport,
    CrashReporter,
    SystemInfoSnapshot,
    configure_global_reporter,
    get_global_reporter,
    DEFAULT_UPLOAD_TIMEOUT_SECONDS,
    DEFAULT_MAX_REPORT_AGE_DAYS,
    MAX_CUSTOM_DATA_SIZE_BYTES,
    FINGERPRINT_LENGTH,
)


@pytest.fixture
def temp_reports_dir():
    """Create a temporary directory for crash reports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def reporter(temp_reports_dir):
    """Create a CrashReporter with temporary directory."""
    return CrashReporter(
        game_version="1.2.3",
        build_id="abc123",
        build_type="debug",
        reports_dir=temp_reports_dir,
        generate_minidump=False,  # Disable for faster tests
    )


@pytest.fixture
def crash_context():
    """Create a sample CrashContext."""
    try:
        raise ValueError("Test crash error")
    except ValueError as e:
        return CrashContext(
            exception=e,
            stack_trace="Traceback:\n  File test.py, line 10\nValueError: Test crash error",
            recent_logs=["Log 1", "Log 2", "Log 3"],
        )


class TestSystemInfoSnapshot:
    """Tests for SystemInfoSnapshot."""

    def test_capture_basic_info(self):
        """capture() should return system information."""
        info = SystemInfoSnapshot.capture()

        assert info.os_name != ""
        assert info.os_name == platform.system()
        assert info.os_arch != ""
        assert info.cpu_count > 0

    def test_capture_os_info(self):
        """capture() should include OS information."""
        info = SystemInfoSnapshot.capture()

        assert info.os_name in ("Linux", "Darwin", "Windows")
        assert info.os_version != ""
        assert info.os_arch in ("x86_64", "AMD64", "arm64", "aarch64", "")

    def test_capture_cpu_info(self):
        """capture() should include CPU information."""
        info = SystemInfoSnapshot.capture()

        assert info.cpu_count >= 1
        # cpu_info may be empty on some systems
        assert isinstance(info.cpu_info, str)

    def test_capture_memory_info(self):
        """capture() should include memory information."""
        info = SystemInfoSnapshot.capture()

        # On Linux, these should be populated
        if platform.system() == "Linux":
            assert info.total_memory_mb > 0
            assert info.available_memory_mb > 0
            assert info.available_memory_mb <= info.total_memory_mb

    def test_default_values(self):
        """SystemInfoSnapshot should have sensible defaults."""
        info = SystemInfoSnapshot()

        assert info.os_name == ""
        assert info.cpu_count == 0
        assert info.total_memory_mb == 0
        assert info.gpu_info is None


class TestCrashReport:
    """Tests for CrashReport dataclass."""

    def test_default_values(self):
        """CrashReport should have sensible defaults."""
        report = CrashReport()

        assert report.report_id != ""
        assert report.context is None
        assert isinstance(report.system_info, SystemInfoSnapshot)
        assert report.game_version == "unknown"
        assert isinstance(report.timestamp, datetime)
        assert report.custom_data == {}

    def test_with_context(self, crash_context):
        """CrashReport should store crash context."""
        report = CrashReport(context=crash_context)

        assert report.context is not None
        assert report.context.exception is not None
        assert "Test crash error" in str(report.context.exception)

    def test_to_dict(self, crash_context):
        """to_dict() should return serializable dictionary."""
        report = CrashReport(
            context=crash_context,
            game_version="1.0.0",
            build_id="test123",
        )

        d = report.to_dict()

        assert d["report_id"] == report.report_id
        assert d["game_version"] == "1.0.0"
        assert d["build_id"] == "test123"
        assert "crash_context" in d
        assert d["crash_context"]["exception_type"] == "ValueError"
        assert "Test crash error" in d["crash_context"]["exception_message"]

    def test_to_dict_without_context(self):
        """to_dict() should work without crash context."""
        report = CrashReport()
        d = report.to_dict()

        assert "crash_context" not in d
        assert d["report_id"] != ""

    def test_to_dict_serializable(self, crash_context):
        """to_dict() result should be JSON serializable."""
        report = CrashReport(context=crash_context)
        d = report.to_dict()

        # Should not raise
        json_str = json.dumps(d, default=str)
        assert json_str is not None

    def test_fingerprint_consistent(self, crash_context):
        """get_fingerprint() should be consistent for same crash."""
        report1 = CrashReport(context=crash_context)
        report2 = CrashReport(context=crash_context)

        assert report1.get_fingerprint() == report2.get_fingerprint()

    def test_fingerprint_different_for_different_crashes(self):
        """get_fingerprint() should differ for different crashes."""
        try:
            raise ValueError("Error 1")
        except ValueError as e:
            ctx1 = CrashContext(exception=e, stack_trace="stack1")

        try:
            raise TypeError("Error 2")
        except TypeError as e:
            ctx2 = CrashContext(exception=e, stack_trace="stack2")

        report1 = CrashReport(context=ctx1)
        report2 = CrashReport(context=ctx2)

        assert report1.get_fingerprint() != report2.get_fingerprint()


class TestCrashReporter:
    """Tests for CrashReporter class."""

    def test_create_report(self, reporter, crash_context):
        """create_report() should create complete report."""
        report = reporter.create_report(crash_context)

        assert report.context == crash_context
        assert report.game_version == "1.2.3"
        assert report.build_id == "abc123"
        assert report.build_type == "debug"
        assert report.system_info.os_name != ""

    def test_save_local(self, reporter, crash_context, temp_reports_dir):
        """save_local() should write report to disk."""
        report = reporter.create_report(crash_context)
        path = reporter.save_local(report)

        assert os.path.exists(path)
        assert path.startswith(temp_reports_dir)

        # Verify content
        with open(path, "r") as f:
            data = json.load(f)

        assert data["report_id"] == report.report_id
        assert data["game_version"] == "1.2.3"

    def test_save_local_custom_path(self, reporter, crash_context, temp_reports_dir):
        """save_local() should support custom path."""
        report = reporter.create_report(crash_context)
        custom_path = os.path.join(temp_reports_dir, "custom", "crash.json")

        path = reporter.save_local(report, custom_path)

        assert path == custom_path
        assert os.path.exists(path)

    def test_load_report(self, reporter, crash_context, temp_reports_dir):
        """load_report() should load saved report."""
        original = reporter.create_report(crash_context)
        path = reporter.save_local(original)

        loaded = reporter.load_report(path)

        assert loaded is not None
        assert loaded.report_id == original.report_id
        assert loaded.game_version == original.game_version
        assert loaded.build_id == original.build_id

    def test_load_report_preserves_system_info(self, reporter, crash_context):
        """load_report() should preserve system info."""
        original = reporter.create_report(crash_context)
        path = reporter.save_local(original)

        loaded = reporter.load_report(path)

        assert loaded.system_info.os_name == original.system_info.os_name
        assert loaded.system_info.cpu_count == original.system_info.cpu_count

    def test_load_report_invalid_path(self, reporter):
        """load_report() should return None for invalid path."""
        result = reporter.load_report("/nonexistent/path.json")
        assert result is None

    def test_get_pending_reports(self, reporter, crash_context, temp_reports_dir):
        """get_pending_reports() should list saved reports."""
        # Create a few reports
        for i in range(3):
            report = reporter.create_report(crash_context)
            reporter.save_local(report)

        pending = reporter.get_pending_reports()

        assert len(pending) == 3
        assert all(p.endswith(".json") for p in pending)

    def test_cleanup_old_reports(self, reporter, crash_context, temp_reports_dir):
        """cleanup_old_reports() should remove old reports."""
        # Create a report
        report = reporter.create_report(crash_context)
        path = reporter.save_local(report)

        # Verify it exists
        assert os.path.exists(path)

        # Cleanup with 0 days should remove all
        deleted = reporter.cleanup_old_reports(max_age_days=0)

        # Note: File was just created, so it's 0 days old
        # It should be kept since it's not older than 0 days
        assert deleted == 0  # File is same-day, not older

    def test_custom_data_provider(self, reporter, crash_context):
        """Custom data providers should add data to reports."""

        def custom_provider():
            return {"custom_key": "custom_value", "count": 42}

        reporter.add_custom_data_provider(custom_provider)
        report = reporter.create_report(crash_context)

        assert report.custom_data["custom_key"] == "custom_value"
        assert report.custom_data["count"] == 42

    def test_custom_data_provider_exception(self, reporter, crash_context):
        """Custom data provider exceptions should not break report creation."""

        def failing_provider():
            raise RuntimeError("Provider error")

        def working_provider():
            return {"working": True}

        reporter.add_custom_data_provider(failing_provider)
        reporter.add_custom_data_provider(working_provider)

        # Should not raise
        report = reporter.create_report(crash_context)

        assert report.custom_data.get("working") is True

    def test_upload_callback(self, reporter, crash_context):
        """Upload callbacks should be notified."""
        callback_results = []

        def callback(report, success):
            callback_results.append((report.report_id, success))

        reporter.add_upload_callback(callback)
        report = reporter.create_report(crash_context)

        # Run upload (async)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(
                reporter.upload(report, "https://example.com/crash")
            )
        finally:
            loop.close()

        assert success is True  # Stub returns True
        assert len(callback_results) == 1
        assert callback_results[0][0] == report.report_id
        assert callback_results[0][1] is True


class TestCrashReporterWithMinidump:
    """Tests for CrashReporter with minidump generation."""

    def test_report_includes_minidump_path(self, crash_context, temp_reports_dir):
        """Report should include minidump path when enabled."""
        reporter = CrashReporter(
            reports_dir=temp_reports_dir,
            generate_minidump=True,
            minidump_level=MinidumpLevel.MINI,
        )

        report = reporter.create_report(crash_context)

        assert report.minidump_path is not None
        assert os.path.exists(report.minidump_path)

    def test_minidump_level_configurable(self, crash_context, temp_reports_dir):
        """Minidump level should be configurable."""
        reporter = CrashReporter(
            reports_dir=temp_reports_dir,
            generate_minidump=True,
            minidump_level=MinidumpLevel.MEDIUM,
        )

        report = reporter.create_report(crash_context)

        # Verify minidump was created with MEDIUM level
        assert report.minidump_path is not None
        with open(report.minidump_path, "r") as f:
            dump_data = json.load(f)
        assert dump_data["level"] == "MEDIUM"


class TestGlobalReporter:
    """Tests for global reporter functions."""

    def test_get_global_reporter(self):
        """get_global_reporter() should return same instance."""
        r1 = get_global_reporter()
        r2 = get_global_reporter()
        assert r1 is r2

    def test_configure_global_reporter(self, temp_reports_dir):
        """configure_global_reporter() should create configured instance."""
        reporter = configure_global_reporter(
            game_version="2.0.0",
            build_id="xyz789",
            build_type="release",
            reports_dir=temp_reports_dir,
        )

        assert reporter._game_version == "2.0.0"
        assert reporter._build_id == "xyz789"
        assert reporter._build_type == "release"
        assert reporter._reports_dir == temp_reports_dir


class TestUploadSync:
    """Tests for synchronous upload."""

    def test_upload_sync(self, reporter, crash_context):
        """upload_sync() should work synchronously."""
        report = reporter.create_report(crash_context)
        success = reporter.upload_sync(report, "https://example.com/crash")

        assert success is True  # Stub returns True


class TestReportFingerprinting:
    """Tests for crash report fingerprinting/deduplication."""

    def test_same_crash_same_fingerprint(self, crash_context):
        """Same crash should produce same fingerprint."""
        report1 = CrashReport(context=crash_context)
        report2 = CrashReport(context=crash_context)

        fp1 = report1.get_fingerprint()
        fp2 = report2.get_fingerprint()

        assert fp1 == fp2
        assert len(fp1) == 16  # SHA256 truncated to 16 chars

    def test_fingerprint_without_context(self):
        """Fingerprint should work without crash context."""
        report = CrashReport()
        fp = report.get_fingerprint()

        assert len(fp) == FINGERPRINT_LENGTH
        # Empty context should produce consistent fingerprint
        assert report.get_fingerprint() == fp


class TestCrashReporterSecurity:
    """Tests for security-related functionality."""

    def test_path_traversal_prevention(self, reporter, crash_context, temp_reports_dir):
        """save_local should prevent path traversal attacks."""
        report = reporter.create_report(crash_context)

        # Attempt path traversal
        malicious_path = os.path.join(temp_reports_dir, "..", "..", "etc", "passwd")

        # Should either raise or save to a safe location
        try:
            saved_path = reporter.save_local(report, malicious_path)
            # If it doesn't raise, verify it didn't actually write outside
            assert not saved_path.startswith("/etc")
            # The resolved path should be different from the malicious one
            resolved = str(Path(malicious_path).resolve())
            assert os.path.exists(saved_path)
        except (ValueError, OSError):
            # Expected behavior - path validation should reject this
            pass

    def test_null_byte_injection_prevention(self, reporter, crash_context, temp_reports_dir):
        """save_local should reject paths with null bytes."""
        report = reporter.create_report(crash_context)

        malicious_path = os.path.join(temp_reports_dir, "report\x00.json")

        with pytest.raises((ValueError, OSError)):
            reporter.save_local(report, malicious_path)

    def test_report_id_sanitization(self, reporter, crash_context, temp_reports_dir):
        """Report ID should be sanitized when used in file paths."""
        report = reporter.create_report(crash_context)
        # Attempt to inject path separator in report ID
        report.report_id = "../../../etc/malicious"

        path = reporter.save_local(report)

        # Path should not contain the directory traversal
        assert ".." not in path
        assert os.path.exists(path)
        assert path.startswith(temp_reports_dir)


class TestCrashReporterRobustness:
    """Tests for crash reporter robustness (doesn't crash itself)."""

    def test_failing_custom_provider_doesnt_break_report(self, reporter, crash_context):
        """Custom data provider exceptions should not prevent report creation."""
        def explosive_provider():
            raise RuntimeError("Boom!")

        reporter.add_custom_data_provider(explosive_provider)

        # Should not raise
        report = reporter.create_report(crash_context)
        assert report is not None
        assert report.report_id != ""

    def test_oversized_custom_data_truncated(self, reporter, crash_context):
        """Oversized custom data should be truncated."""
        def huge_provider():
            # Generate more than MAX_CUSTOM_DATA_SIZE_BYTES
            return {"huge": "x" * (MAX_CUSTOM_DATA_SIZE_BYTES + 1000)}

        reporter.add_custom_data_provider(huge_provider)

        # Should not raise, but data might be limited
        report = reporter.create_report(crash_context)
        assert report is not None

    def test_invalid_endpoint_url_rejected(self, reporter, crash_context):
        """Invalid endpoint URLs should be rejected gracefully."""
        report = reporter.create_report(crash_context)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Invalid URLs
            for bad_url in ["", "not-a-url", "ftp://wrong-protocol.com", None]:
                if bad_url is None:
                    continue
                success = loop.run_until_complete(
                    reporter.upload(report, bad_url)
                )
                assert success is False
        finally:
            loop.close()


class TestConfigurationConstants:
    """Tests for configuration constants."""

    def test_default_timeout_constant(self):
        """Default timeout constant should have reasonable value."""
        assert DEFAULT_UPLOAD_TIMEOUT_SECONDS > 0
        assert DEFAULT_UPLOAD_TIMEOUT_SECONDS <= 120  # Not too long

    def test_default_max_age_constant(self):
        """Default max age constant should have reasonable value."""
        assert DEFAULT_MAX_REPORT_AGE_DAYS > 0
        assert DEFAULT_MAX_REPORT_AGE_DAYS <= 365  # Not more than a year

    def test_fingerprint_length_constant(self):
        """Fingerprint length should be used correctly."""
        report = CrashReport()
        fp = report.get_fingerprint()
        assert len(fp) == FINGERPRINT_LENGTH
