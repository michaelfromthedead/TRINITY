"""
Crash reporting for the game engine.

Provides crash report generation, local storage, and remote upload
capabilities for crash diagnostics and telemetry.
"""

import asyncio
import hashlib
import json
import logging
import os
import platform
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .handler import CrashContext
from .minidump import Minidump, MinidumpLevel, generate_crash_dump

# Module-level logger
_logger = logging.getLogger(__name__)


# Configuration constants
DEFAULT_UPLOAD_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_REPORT_AGE_DAYS = 30
MAX_REPORT_SIZE_MB = 10
MAX_CUSTOM_DATA_SIZE_BYTES = 1024 * 1024  # 1 MB
MAX_RECENT_LOGS_IN_REPORT = 100
MAX_STACK_TRACE_LINES = 500
MAX_UPLOAD_RETRIES = 3
FINGERPRINT_LENGTH = 16


@dataclass
class SystemInfoSnapshot:
    """
    Snapshot of system information at crash time.

    Attributes:
        os_name: Operating system name
        os_version: Operating system version
        os_arch: System architecture
        cpu_info: CPU information string
        cpu_count: Number of CPU cores
        total_memory_mb: Total system memory in megabytes
        available_memory_mb: Available memory at crash time in megabytes
        gpu_info: GPU information (if available)
        display_info: Display/monitor information
    """
    os_name: str = ""
    os_version: str = ""
    os_arch: str = ""
    cpu_info: str = ""
    cpu_count: int = 0
    total_memory_mb: int = 0
    available_memory_mb: int = 0
    gpu_info: Optional[str] = None
    display_info: Optional[str] = None

    @classmethod
    def capture(cls) -> 'SystemInfoSnapshot':
        """
        Capture current system information.

        Returns:
            SystemInfoSnapshot with current system state
        """
        info = cls()

        # Basic OS info
        info.os_name = platform.system()
        info.os_version = platform.release()
        info.os_arch = platform.machine()
        info.cpu_info = platform.processor()
        info.cpu_count = os.cpu_count() or 0

        # Memory info (platform-specific)
        info.total_memory_mb, info.available_memory_mb = cls._get_memory_info()

        # GPU info (stub - would need platform-specific code)
        info.gpu_info = cls._get_gpu_info()

        return info

    @staticmethod
    def _get_memory_info() -> tuple[int, int]:
        """Get total and available memory in MB."""
        total_mb = 0
        available_mb = 0

        try:
            if platform.system() == 'Linux':
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            total_mb = int(line.split()[1]) // 1024
                        elif line.startswith('MemAvailable:'):
                            available_mb = int(line.split()[1]) // 1024
                        elif line.startswith('MemFree:') and available_mb == 0:
                            available_mb = int(line.split()[1]) // 1024

            elif platform.system() == 'Darwin':
                # macOS: Would use sysctl or vm_stat
                pass

            elif platform.system() == 'Windows':
                # Windows: Would use GlobalMemoryStatusEx
                pass

        except Exception as e:
            _logger.debug(f"Could not get memory info: {e}")

        return total_mb, available_mb

    @staticmethod
    def _get_gpu_info() -> Optional[str]:
        """Get GPU information (stub)."""
        # Would need platform-specific implementation
        # Linux: Parse /proc/driver/nvidia/version or lspci
        # Windows: Use WMI or DirectX
        # macOS: Use IOKit
        return None


@dataclass
class CrashReport:
    """
    Complete crash report with all diagnostic information.

    Attributes:
        report_id: Unique identifier for this report
        context: The CrashContext from the crash handler
        system_info: System information snapshot
        game_version: Version of the game/engine
        build_id: Build identifier
        build_type: Build type (debug/release/etc.)
        screenshot_path: Path to crash screenshot (if captured)
        minidump_path: Path to minidump file (if generated)
        user_description: Optional user-provided description
        custom_data: Additional custom diagnostic data
        timestamp: When the report was created
    """
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Optional[CrashContext] = None
    system_info: SystemInfoSnapshot = field(default_factory=SystemInfoSnapshot)
    game_version: str = "unknown"
    build_id: str = "unknown"
    build_type: str = "unknown"
    screenshot_path: Optional[str] = None
    minidump_path: Optional[str] = None
    user_description: Optional[str] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the report to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        result = {
            'report_id': self.report_id,
            'timestamp': self.timestamp.isoformat(),
            'game_version': self.game_version,
            'build_id': self.build_id,
            'build_type': self.build_type,
            'screenshot_path': self.screenshot_path,
            'minidump_path': self.minidump_path,
            'user_description': self.user_description,
            'custom_data': self.custom_data,
            'system_info': asdict(self.system_info),
        }

        # Add crash context if present
        if self.context:
            result['crash_context'] = {
                'exception_type': type(self.context.exception).__name__ if self.context.exception else None,
                'exception_message': str(self.context.exception) if self.context.exception else None,
                'stack_trace': self.context.stack_trace,
                'recent_logs': self.context.recent_logs,
                'thread_id': self.context.thread_id,
                'thread_name': self.context.thread_name,
                'signal_number': self.context.signal_number,
                'signal_name': self.context.signal_name,
                'crash_timestamp': self.context.timestamp.isoformat(),
            }

        return result

    def get_fingerprint(self) -> str:
        """
        Generate a fingerprint for deduplication.

        The fingerprint is based on the exception type and top of stack,
        allowing similar crashes to be grouped together.

        Returns:
            SHA256 hash fingerprint string
        """
        components = []

        if self.context:
            if self.context.exception:
                components.append(type(self.context.exception).__name__)
                components.append(str(self.context.exception))

            # Use first few lines of stack trace for fingerprint
            if self.context.stack_trace:
                # Use constant for stack trace lines in fingerprint
                from .minidump import FINGERPRINT_STACK_LINES
                lines = self.context.stack_trace.split('\n')[:FINGERPRINT_STACK_LINES]
                components.extend(lines)

            if self.context.signal_name:
                components.append(self.context.signal_name)

        combined = '\n'.join(components)
        return hashlib.sha256(combined.encode()).hexdigest()[:FINGERPRINT_LENGTH]


class CrashReporter:
    """
    Crash reporting manager.

    Handles creating crash reports, saving them locally, and
    uploading them to remote servers.

    Usage:
        >>> reporter = CrashReporter(game_version="1.2.3")
        >>> report = reporter.create_report(crash_context)
        >>> reporter.save_local(report, "/path/to/crashes")
        >>> await reporter.upload(report, "https://crashes.example.com/api")
    """

    def __init__(
        self,
        game_version: str = "unknown",
        build_id: str = "unknown",
        build_type: str = "unknown",
        reports_dir: Optional[str] = None,
        generate_minidump: bool = True,
        minidump_level: MinidumpLevel = MinidumpLevel.MEDIUM,
        capture_screenshot: bool = False,
    ):
        """
        Initialize the crash reporter.

        Args:
            game_version: Version string of the game/engine
            build_id: Build identifier (e.g., commit hash)
            build_type: Build type (debug/release/etc.)
            reports_dir: Directory to store crash reports
            generate_minidump: Whether to generate minidumps
            minidump_level: Level of detail for minidumps
            capture_screenshot: Whether to capture screenshots on crash
        """
        self._game_version = game_version
        self._build_id = build_id
        self._build_type = build_type
        self._reports_dir = reports_dir or self._default_reports_dir()
        self._generate_minidump = generate_minidump
        self._minidump_level = minidump_level
        self._capture_screenshot = capture_screenshot
        self._upload_callbacks: List[Callable[[CrashReport, bool], None]] = []
        self._custom_data_providers: List[Callable[[], Dict[str, Any]]] = []

    @staticmethod
    def _default_reports_dir() -> str:
        """Get the default crash reports directory."""
        if platform.system() == 'Windows':
            base = os.environ.get('LOCALAPPDATA', tempfile.gettempdir())
        elif platform.system() == 'Darwin':
            base = os.path.expanduser('~/Library/Application Support')
        else:
            base = os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))

        return os.path.join(base, 'GameEngine', 'CrashReports')

    def create_report(self, context: CrashContext) -> CrashReport:
        """
        Create a crash report from a CrashContext.

        This method is designed to be robust and will not raise exceptions.
        If parts of the report fail to generate, they will be skipped.

        Args:
            context: The crash context to report

        Returns:
            Complete CrashReport ready for saving/uploading
        """
        _logger.info("Creating crash report")

        # Create report with fallback for system info capture
        try:
            system_info = SystemInfoSnapshot.capture()
        except Exception as e:
            _logger.error(f"Failed to capture system info: {e}")
            system_info = SystemInfoSnapshot()

        report = CrashReport(
            context=context,
            system_info=system_info,
            game_version=self._game_version,
            build_id=self._build_id,
            build_type=self._build_type,
        )

        # Collect custom data from providers (with size limit)
        total_custom_data_size = 0
        for provider in self._custom_data_providers:
            try:
                data = provider()
                if data:
                    # Check size before adding
                    data_size = len(json.dumps(data, default=str))
                    if total_custom_data_size + data_size > MAX_CUSTOM_DATA_SIZE_BYTES:
                        _logger.warning("Custom data size limit reached, skipping provider")
                        continue
                    report.custom_data.update(data)
                    total_custom_data_size += data_size
            except Exception as e:
                _logger.error(f"Custom data provider failed: {e}")

        # Generate minidump if enabled
        if self._generate_minidump:
            try:
                dump_path = os.path.join(
                    self._reports_dir,
                    f"minidump_{report.report_id}.dmp"
                )
                Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
                generate_crash_dump(
                    dump_path,
                    self._minidump_level,
                    context.exception
                )
                report.minidump_path = dump_path
            except Exception as e:
                _logger.error(f"Failed to generate minidump: {e}")
                # Continue without minidump

        # Capture screenshot if enabled (stub)
        if self._capture_screenshot:
            try:
                report.screenshot_path = self._capture_crash_screenshot(report.report_id)
            except Exception as e:
                _logger.error(f"Failed to capture screenshot: {e}")

        _logger.info(f"Crash report created: {report.report_id}")
        return report

    def save_local(self, report: CrashReport, path: Optional[str] = None) -> str:
        """
        Save a crash report to local disk.

        Args:
            report: The crash report to save
            path: Optional specific path. If None, uses reports_dir

        Returns:
            Path to the saved report file

        Raises:
            OSError: If the file cannot be written
            ValueError: If the path is invalid
        """
        if path is None:
            # Sanitize report_id to prevent path injection
            safe_id = "".join(c for c in report.report_id if c.isalnum() or c == '-')
            path = os.path.join(
                self._reports_dir,
                f"crash_{safe_id}.json"
            )

        # Validate path
        try:
            resolved_path = Path(path).resolve()
            # Check for null bytes
            if '\x00' in str(resolved_path):
                raise ValueError("Path contains null bytes")
        except Exception as e:
            _logger.error(f"Invalid path: {e}")
            raise ValueError(f"Invalid path: {e}")

        # Ensure directory exists
        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _logger.error(f"Failed to create directory: {e}")
            raise

        # Write report with atomic operation
        temp_path = str(resolved_path) + ".tmp"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

            # Atomic rename (on most systems)
            os.replace(temp_path, str(resolved_path))

        except Exception as e:
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            raise

        _logger.info(f"Crash report saved to {resolved_path}")
        return str(resolved_path)

    async def upload(
        self,
        report: CrashReport,
        endpoint: str,
        timeout: float = DEFAULT_UPLOAD_TIMEOUT_SECONDS,
        include_minidump: bool = True,
        max_retries: int = MAX_UPLOAD_RETRIES,
    ) -> bool:
        """
        Upload a crash report to a remote server.

        This is an async stub that simulates the upload process.
        In production, this would use aiohttp or similar.

        Args:
            report: The crash report to upload
            endpoint: URL endpoint to upload to
            timeout: Upload timeout in seconds
            include_minidump: Whether to include the minidump file
            max_retries: Maximum number of retry attempts

        Returns:
            True if upload succeeded, False otherwise
        """
        _logger.info(f"Uploading crash report {report.report_id} to {endpoint}")

        # Validate endpoint URL
        if not endpoint or not endpoint.startswith(('http://', 'https://')):
            _logger.error(f"Invalid endpoint URL: {endpoint}")
            return False

        last_error = None
        for attempt in range(max_retries):
            try:
                # Prepare payload
                payload = report.to_dict()

                # Check payload size
                payload_str = json.dumps(payload, default=str)
                if len(payload_str) > MAX_REPORT_SIZE_MB * 1024 * 1024:
                    _logger.warning("Crash report exceeds size limit, truncating")
                    # Truncate stack trace if needed
                    if 'crash_context' in payload and payload['crash_context']:
                        if 'stack_trace' in payload['crash_context']:
                            lines = payload['crash_context']['stack_trace'].split('\n')
                            payload['crash_context']['stack_trace'] = '\n'.join(
                                lines[:MAX_STACK_TRACE_LINES]
                            )
                        if 'recent_logs' in payload['crash_context']:
                            payload['crash_context']['recent_logs'] = \
                                payload['crash_context']['recent_logs'][-MAX_RECENT_LOGS_IN_REPORT:]

                # Simulate network request
                # In production, this would be:
                # async with aiohttp.ClientSession() as session:
                #     async with session.post(endpoint, json=payload, timeout=timeout) as response:
                #         success = response.status == 200

                # Stub implementation - simulate async upload
                await asyncio.sleep(0.1)  # Simulate network latency
                success = True  # Simulate success

                _logger.info(f"Crash report upload {'succeeded' if success else 'failed'}")

                # Notify callbacks (in a safe manner)
                self._notify_upload_callbacks(report, success)

                return success

            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
                _logger.warning(f"Upload attempt {attempt + 1}/{max_retries} timed out")
            except asyncio.CancelledError:
                _logger.info("Upload cancelled")
                raise
            except Exception as e:
                last_error = str(e)
                _logger.warning(f"Upload attempt {attempt + 1}/{max_retries} failed: {e}")

            # Exponential backoff before retry
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)  # Cap at 30 seconds
                await asyncio.sleep(wait_time)

        _logger.error(f"Crash report upload failed after {max_retries} attempts: {last_error}")
        self._notify_upload_callbacks(report, False)
        return False

    def _notify_upload_callbacks(self, report: CrashReport, success: bool) -> None:
        """
        Safely notify all upload callbacks.

        Args:
            report: The crash report
            success: Whether upload succeeded
        """
        for callback in self._upload_callbacks:
            try:
                callback(report, success)
            except Exception as e:
                _logger.error(f"Upload callback failed: {e}")

    def upload_sync(
        self,
        report: CrashReport,
        endpoint: str,
        timeout: float = DEFAULT_UPLOAD_TIMEOUT_SECONDS,
    ) -> bool:
        """
        Synchronous wrapper for upload().

        Use this when you can't use async/await.

        Args:
            report: The crash report to upload
            endpoint: URL endpoint to upload to
            timeout: Upload timeout in seconds

        Returns:
            True if upload succeeded, False otherwise
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_running_loop()
            # If we're in a running loop, we can't use run_until_complete
            # Use asyncio.run in a new thread or just run directly
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self.upload(report, endpoint, timeout)
                )
                return future.result(timeout=timeout + 5)
        except RuntimeError:
            # No running loop, create a new one
            return asyncio.run(self.upload(report, endpoint, timeout))

    def add_upload_callback(
        self,
        callback: Callable[[CrashReport, bool], None]
    ) -> None:
        """
        Add a callback to be notified of upload results.

        Args:
            callback: Function(report, success) to call after upload
        """
        self._upload_callbacks.append(callback)

    def add_custom_data_provider(
        self,
        provider: Callable[[], Dict[str, Any]]
    ) -> None:
        """
        Add a custom data provider for crash reports.

        The provider function will be called when creating a report
        and its returned dictionary merged into custom_data.

        Args:
            provider: Function that returns diagnostic data dict
        """
        self._custom_data_providers.append(provider)

    def _capture_crash_screenshot(self, report_id: str) -> Optional[str]:
        """
        Capture a screenshot at crash time (stub).

        This would require platform-specific implementation
        or integration with the rendering system.

        Args:
            report_id: Report ID for filename

        Returns:
            Path to screenshot or None if capture failed
        """
        # Stub - would capture actual screenshot in production
        _logger.debug("Screenshot capture not implemented")
        return None

    def get_pending_reports(self) -> List[str]:
        """
        Get list of crash reports that haven't been uploaded.

        Returns:
            List of file paths to pending crash reports
        """
        reports = []

        if os.path.exists(self._reports_dir):
            for filename in os.listdir(self._reports_dir):
                if filename.startswith('crash_') and filename.endswith('.json'):
                    reports.append(os.path.join(self._reports_dir, filename))

        return reports

    def load_report(self, path: str) -> Optional[CrashReport]:
        """
        Load a crash report from disk.

        Args:
            path: Path to the crash report JSON file

        Returns:
            CrashReport or None if loading failed
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct report (simplified - context is not fully restored)
            report = CrashReport(
                report_id=data.get('report_id', str(uuid.uuid4())),
                game_version=data.get('game_version', 'unknown'),
                build_id=data.get('build_id', 'unknown'),
                build_type=data.get('build_type', 'unknown'),
                screenshot_path=data.get('screenshot_path'),
                minidump_path=data.get('minidump_path'),
                user_description=data.get('user_description'),
                custom_data=data.get('custom_data', {}),
            )

            # Restore system info
            if 'system_info' in data:
                si = data['system_info']
                report.system_info = SystemInfoSnapshot(
                    os_name=si.get('os_name', ''),
                    os_version=si.get('os_version', ''),
                    os_arch=si.get('os_arch', ''),
                    cpu_info=si.get('cpu_info', ''),
                    cpu_count=si.get('cpu_count', 0),
                    total_memory_mb=si.get('total_memory_mb', 0),
                    available_memory_mb=si.get('available_memory_mb', 0),
                    gpu_info=si.get('gpu_info'),
                    display_info=si.get('display_info'),
                )

            return report

        except Exception as e:
            _logger.error(f"Failed to load crash report: {e}")
            return None

    def cleanup_old_reports(self, max_age_days: int = DEFAULT_MAX_REPORT_AGE_DAYS) -> int:
        """
        Remove crash reports older than a certain age.

        Args:
            max_age_days: Maximum age in days to keep reports

        Returns:
            Number of reports deleted
        """
        deleted = 0
        now = datetime.now()

        if not os.path.exists(self._reports_dir):
            return 0

        for filename in os.listdir(self._reports_dir):
            filepath = os.path.join(self._reports_dir, filename)

            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                age_days = (now - mtime).days

                if age_days > max_age_days:
                    os.remove(filepath)
                    deleted += 1
                    _logger.debug(f"Deleted old crash report: {filename}")

            except Exception as e:
                _logger.error(f"Failed to check/delete {filepath}: {e}")

        if deleted > 0:
            _logger.info(f"Cleaned up {deleted} old crash reports")

        return deleted


# Global reporter instance
_global_reporter: Optional[CrashReporter] = None


def get_global_reporter() -> CrashReporter:
    """
    Get the global CrashReporter instance.

    Creates one with default settings if it doesn't exist.

    Returns:
        The global CrashReporter instance
    """
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = CrashReporter()
    return _global_reporter


def configure_global_reporter(
    game_version: str = "unknown",
    build_id: str = "unknown",
    build_type: str = "unknown",
    reports_dir: Optional[str] = None,
) -> CrashReporter:
    """
    Configure and return the global crash reporter.

    Args:
        game_version: Version string of the game/engine
        build_id: Build identifier
        build_type: Build type (debug/release)
        reports_dir: Directory to store crash reports

    Returns:
        The configured global CrashReporter instance
    """
    global _global_reporter
    _global_reporter = CrashReporter(
        game_version=game_version,
        build_id=build_id,
        build_type=build_type,
        reports_dir=reports_dir,
    )
    return _global_reporter


# Export public API
__all__ = [
    'SystemInfoSnapshot',
    'CrashReport',
    'CrashReporter',
    'get_global_reporter',
    'configure_global_reporter',
]
