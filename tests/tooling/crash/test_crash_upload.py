"""
Tests for crash_upload.py - Upload crash reports to server.
"""

import json
import time

import pytest

from engine.tooling.crash.crash_reporter import (
    CrashContext,
    CrashReport,
    CrashSeverity,
    ExceptionInfo,
    SystemInfo,
)
from engine.tooling.crash.crash_upload import (
    AsyncCrashUploader,
    CrashUploader,
    UploadConfig,
    UploadResult,
    UploadStatus,
    get_uploader,
    upload_crash_report,
    upload_minidump,
)


class TestUploadStatus:
    """Tests for UploadStatus enum."""

    def test_status_values_exist(self):
        assert UploadStatus.PENDING
        assert UploadStatus.IN_PROGRESS
        assert UploadStatus.SUCCESS
        assert UploadStatus.FAILED
        assert UploadStatus.RETRY


class TestUploadResult:
    """Tests for UploadResult dataclass."""

    def test_create_result(self):
        result = UploadResult(
            status=UploadStatus.SUCCESS,
            report_id="crash-123",
        )

        assert result.status == UploadStatus.SUCCESS
        assert result.report_id == "crash-123"

    def test_success_property(self):
        success = UploadResult(status=UploadStatus.SUCCESS, report_id="")
        failed = UploadResult(status=UploadStatus.FAILED, report_id="")

        assert success.success is True
        assert failed.success is False

    def test_to_dict(self):
        result = UploadResult(
            status=UploadStatus.SUCCESS,
            report_id="crash-123",
            server_id="srv-456",
            duration=1.5,
        )
        data = result.to_dict()

        assert data["status"] == "SUCCESS"
        assert data["report_id"] == "crash-123"
        assert data["server_id"] == "srv-456"


class TestUploadConfig:
    """Tests for UploadConfig dataclass."""

    def test_default_config(self):
        config = UploadConfig()

        assert config.compress is True
        assert config.timeout == 30.0
        assert config.max_retries == 3

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("CRASH_SERVER_URL", "http://crash.example.com")
        monkeypatch.setenv("CRASH_API_KEY", "test-key")
        monkeypatch.setenv("CRASH_PROJECT_ID", "project-123")

        config = UploadConfig.from_env()

        assert config.server_url == "http://crash.example.com"
        assert config.api_key == "test-key"
        assert config.project_id == "project-123"


class TestCrashUploader:
    """Tests for CrashUploader class."""

    def _create_test_report(self) -> CrashReport:
        """Create a test crash report."""
        return CrashReport(
            id="test-crash-123",
            timestamp=time.time(),
            severity=CrashSeverity.ERROR,
            exception_info=ExceptionInfo(
                exception_type="ValueError",
                exception_message="Test error",
            ),
            system_info=SystemInfo(),
            context=CrashContext(),
        )

    def test_create_uploader(self):
        config = UploadConfig(server_url="http://test.com")
        uploader = CrashUploader(config)

        assert uploader.config.server_url == "http://test.com"

    def test_compress_data(self):
        config = UploadConfig(compress=True)
        uploader = CrashUploader(config)

        original = b"test data" * 100
        compressed = uploader._compress_data(original)

        assert len(compressed) < len(original)

    def test_no_compress_when_disabled(self):
        config = UploadConfig(compress=False)
        uploader = CrashUploader(config)

        original = b"test data"
        result = uploader._compress_data(original)

        assert result == original

    def test_queue_upload(self):
        config = UploadConfig()
        uploader = CrashUploader(config)
        report = self._create_test_report()

        uploader.queue_upload(report)

        assert len(uploader._queue) == 1

    def test_callback(self):
        config = UploadConfig(server_url="")  # Empty URL will fail
        uploader = CrashUploader(config)
        callbacks_received = []

        def on_upload(result):
            callbacks_received.append(result)

        uploader.add_callback(on_upload)

        report = self._create_test_report()
        uploader.upload(report)

        assert len(callbacks_received) == 1

    def test_get_stats(self):
        config = UploadConfig()
        uploader = CrashUploader(config)
        report = self._create_test_report()

        uploader.queue_upload(report)
        stats = uploader.get_stats()

        assert stats["queued"] == 1
        assert "total_uploads" in stats


class TestUploadMinidump:
    """Tests for minidump upload functionality."""

    def test_upload_nonexistent_file(self):
        config = UploadConfig(server_url="http://test.com")
        uploader = CrashUploader(config)

        result = uploader.upload_minidump(
            "/nonexistent/path.dmp",
            "report-123",
        )

        assert result.status == UploadStatus.FAILED
        assert "not found" in result.message.lower()

    def test_upload_existing_file(self, tmp_path):
        # Create a test minidump file
        minidump = tmp_path / "test.dmp"
        minidump.write_bytes(b"MDMP" + b"\x00" * 100)

        config = UploadConfig(server_url="")  # Empty URL will fail
        uploader = CrashUploader(config)

        result = uploader.upload_minidump(str(minidump), "report-123")

        # Should fail due to empty URL, but file was found
        assert "not found" not in result.message.lower()


class TestAsyncCrashUploader:
    """Tests for AsyncCrashUploader class."""

    def test_create_async_uploader(self):
        config = UploadConfig()
        uploader = AsyncCrashUploader(config)

        assert uploader is not None

    def test_upload_async(self):
        """Test async uploader - uses asyncio.run for compatibility."""
        import asyncio

        config = UploadConfig(server_url="")  # Will fail
        uploader = AsyncCrashUploader(config)

        report = CrashReport(
            id="test-crash",
            timestamp=time.time(),
            severity=CrashSeverity.ERROR,
            exception_info=ExceptionInfo(
                exception_type="ValueError",
                exception_message="Test",
            ),
            system_info=SystemInfo(),
            context=CrashContext(),
        )

        async def run_upload():
            return await uploader.upload_async(report)

        result = asyncio.run(run_upload())
        # Should return a result (even if failed)
        assert result is not None


class TestGlobalFunctions:
    """Tests for global upload functions."""

    def test_get_uploader(self):
        uploader = get_uploader()
        assert uploader is not None

    def test_upload_crash_report_no_server(self):
        report = CrashReport(
            id="test-crash",
            timestamp=time.time(),
            severity=CrashSeverity.ERROR,
            exception_info=ExceptionInfo(
                exception_type="ValueError",
                exception_message="Test",
            ),
            system_info=SystemInfo(),
            context=CrashContext(),
        )

        # Should not raise even without a configured server
        result = upload_crash_report(report)
        assert result is not None
