"""
Upload crash reports to server.

Provides functionality for uploading crash reports and minidumps
to a crash collection server.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .crash_reporter import CrashReport


class UploadStatus(Enum):
    """Status of an upload operation."""

    PENDING = auto()
    IN_PROGRESS = auto()
    SUCCESS = auto()
    FAILED = auto()
    RETRY = auto()


@dataclass
class UploadResult:
    """Result of a crash upload operation."""

    status: UploadStatus
    report_id: str = ""
    server_id: Optional[str] = None
    message: str = ""
    url: str = ""
    duration: float = 0.0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if upload was successful."""
        return self.status == UploadStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.name,
            "report_id": self.report_id,
            "server_id": self.server_id,
            "message": self.message,
            "url": self.url,
            "duration": self.duration,
            "retries": self.retries,
            "metadata": self.metadata,
        }


@dataclass
class UploadConfig:
    """Configuration for crash uploads."""

    server_url: str = ""
    api_key: str = ""
    project_id: str = ""
    compress: bool = True
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 10
    rate_limit: float = 1.0  # Requests per second
    verify_ssl: bool = True
    custom_headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "UploadConfig":
        """Create config from environment variables."""
        return cls(
            server_url=os.environ.get("CRASH_SERVER_URL", ""),
            api_key=os.environ.get("CRASH_API_KEY", ""),
            project_id=os.environ.get("CRASH_PROJECT_ID", ""),
        )


class CrashUploader:
    """
    Handles uploading crash reports to a server.

    Features:
    - Compression
    - Retry with backoff
    - Rate limiting
    - Batch uploads
    - Async support
    """

    def __init__(self, config: Optional[UploadConfig] = None):
        self.config = config or UploadConfig()
        self._queue: List[Tuple[CrashReport, str]] = []
        self._last_upload_time: float = 0.0
        self._upload_count: int = 0
        self._callbacks: List[Callable[[UploadResult], None]] = []

    def add_callback(self, callback: Callable[[UploadResult], None]) -> None:
        """Add a callback for upload completion."""
        self._callbacks.append(callback)

    def _run_callbacks(self, result: UploadResult) -> None:
        """Run all callbacks with the result."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception:
                pass

    def _rate_limit(self) -> None:
        """Apply rate limiting."""
        if self.config.rate_limit <= 0:
            return

        min_interval = 1.0 / self.config.rate_limit
        elapsed = time.time() - self._last_upload_time

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_upload_time = time.time()

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if configured."""
        if self.config.compress:
            return gzip.compress(data)
        return data

    def _build_request(
        self,
        endpoint: str,
        data: bytes,
        content_type: str = "application/json",
    ) -> urllib.request.Request:
        """Build an HTTP request."""
        url = urllib.parse.urljoin(self.config.server_url, endpoint)

        request = urllib.request.Request(url, data=data, method="POST")

        # Add headers
        request.add_header("Content-Type", content_type)
        if self.config.compress:
            request.add_header("Content-Encoding", "gzip")
        if self.config.api_key:
            request.add_header("Authorization", f"Bearer {self.config.api_key}")
        if self.config.project_id:
            request.add_header("X-Project-ID", self.config.project_id)

        for key, value in self.config.custom_headers.items():
            request.add_header(key, value)

        return request

    def _send_request(
        self,
        request: urllib.request.Request,
    ) -> Tuple[int, bytes]:
        """Send HTTP request and return status code and body."""
        ssl_context = None
        if not self.config.verify_ssl:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout,
                context=ssl_context,
            ) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:
            raise

    def upload(
        self,
        report: CrashReport,
        endpoint: str = "/api/crashes",
    ) -> UploadResult:
        """
        Upload a crash report.

        Args:
            report: The crash report to upload
            endpoint: Server endpoint

        Returns:
            Upload result
        """
        start_time = time.time()
        result = UploadResult(
            status=UploadStatus.IN_PROGRESS,
            report_id=report.id,
        )

        # Serialize and compress
        data = report.to_json().encode("utf-8")
        data = self._compress_data(data)

        # Apply rate limiting
        self._rate_limit()

        # Retry loop
        for attempt in range(self.config.max_retries + 1):
            try:
                request = self._build_request(endpoint, data)
                status_code, response_body = self._send_request(request)

                if 200 <= status_code < 300:
                    # Success
                    response_data = json.loads(response_body.decode("utf-8"))
                    result.status = UploadStatus.SUCCESS
                    result.server_id = response_data.get("id")
                    result.url = response_data.get("url", "")
                    result.message = "Upload successful"
                    result.metadata = response_data
                    break

                elif status_code in (429, 503):
                    # Rate limited or service unavailable, retry
                    result.retries = attempt + 1
                    if attempt < self.config.max_retries:
                        time.sleep(self.config.retry_delay * (2 ** attempt))
                        continue

                    result.status = UploadStatus.FAILED
                    result.message = f"Server returned {status_code} after {attempt + 1} attempts"

                else:
                    # Other error
                    result.status = UploadStatus.FAILED
                    result.message = f"Server returned {status_code}"
                    break

            except Exception as e:
                result.retries = attempt + 1
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                    continue

                result.status = UploadStatus.FAILED
                result.message = f"Upload failed: {e}"
                break

        result.duration = time.time() - start_time
        self._upload_count += 1

        self._run_callbacks(result)
        return result

    def upload_minidump(
        self,
        minidump_path: str,
        report_id: str,
        endpoint: str = "/api/minidumps",
    ) -> UploadResult:
        """
        Upload a minidump file.

        Args:
            minidump_path: Path to minidump file
            report_id: Associated crash report ID
            endpoint: Server endpoint

        Returns:
            Upload result
        """
        start_time = time.time()
        result = UploadResult(
            status=UploadStatus.IN_PROGRESS,
            report_id=report_id,
        )

        path = Path(minidump_path)
        if not path.exists():
            result.status = UploadStatus.FAILED
            result.message = f"Minidump file not found: {minidump_path}"
            return result

        # Read and compress file
        with open(path, "rb") as f:
            data = f.read()

        data = self._compress_data(data)

        # Apply rate limiting
        self._rate_limit()

        # Upload
        for attempt in range(self.config.max_retries + 1):
            try:
                url = f"{self.config.server_url}{endpoint}"
                request = urllib.request.Request(url, data=data, method="POST")

                request.add_header("Content-Type", "application/octet-stream")
                if self.config.compress:
                    request.add_header("Content-Encoding", "gzip")
                request.add_header("X-Report-ID", report_id)
                request.add_header("X-Filename", path.name)

                if self.config.api_key:
                    request.add_header("Authorization", f"Bearer {self.config.api_key}")

                status_code, response_body = self._send_request(request)

                if 200 <= status_code < 300:
                    response_data = json.loads(response_body.decode("utf-8"))
                    result.status = UploadStatus.SUCCESS
                    result.server_id = response_data.get("id")
                    result.message = "Minidump uploaded"
                    break
                else:
                    if attempt >= self.config.max_retries:
                        result.status = UploadStatus.FAILED
                        result.message = f"Upload failed with status {status_code}"
                    else:
                        time.sleep(self.config.retry_delay * (2 ** attempt))

            except Exception as e:
                if attempt >= self.config.max_retries:
                    result.status = UploadStatus.FAILED
                    result.message = f"Upload failed: {e}"
                else:
                    time.sleep(self.config.retry_delay * (2 ** attempt))

        result.duration = time.time() - start_time
        result.retries = min(attempt, self.config.max_retries)

        self._run_callbacks(result)
        return result

    def queue_upload(self, report: CrashReport, endpoint: str = "/api/crashes") -> None:
        """Queue a report for batch upload."""
        self._queue.append((report, endpoint))

    def flush_queue(self) -> List[UploadResult]:
        """Upload all queued reports."""
        results = []

        while self._queue:
            batch = self._queue[:self.config.batch_size]
            self._queue = self._queue[self.config.batch_size:]

            for report, endpoint in batch:
                result = self.upload(report, endpoint)
                results.append(result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get upload statistics."""
        return {
            "total_uploads": self._upload_count,
            "queued": len(self._queue),
        }


# Async support

class AsyncCrashUploader(CrashUploader):
    """Async version of CrashUploader."""

    async def upload_async(
        self,
        report: CrashReport,
        endpoint: str = "/api/crashes",
    ) -> UploadResult:
        """
        Upload a crash report asynchronously.

        Args:
            report: The crash report
            endpoint: Server endpoint

        Returns:
            Upload result
        """
        # Run synchronous upload in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.upload, report, endpoint)

    async def upload_all_async(
        self,
        reports: List[CrashReport],
        endpoint: str = "/api/crashes",
        max_concurrent: int = 5,
    ) -> List[UploadResult]:
        """
        Upload multiple reports concurrently.

        Args:
            reports: Reports to upload
            endpoint: Server endpoint
            max_concurrent: Max concurrent uploads

        Returns:
            List of upload results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def upload_with_semaphore(report):
            async with semaphore:
                return await self.upload_async(report, endpoint)

        tasks = [upload_with_semaphore(report) for report in reports]
        return await asyncio.gather(*tasks)


# Global uploader and convenience functions

_uploader: Optional[CrashUploader] = None


def get_uploader() -> CrashUploader:
    """Get or create the global uploader."""
    global _uploader
    if _uploader is None:
        _uploader = CrashUploader(UploadConfig.from_env())
    return _uploader


def upload_crash_report(
    report: CrashReport,
    endpoint: str = "/api/crashes",
) -> UploadResult:
    """
    Upload a crash report.

    Args:
        report: Crash report to upload
        endpoint: Server endpoint

    Returns:
        Upload result
    """
    return get_uploader().upload(report, endpoint)


def upload_minidump(
    minidump_path: str,
    report_id: str,
    endpoint: str = "/api/minidumps",
) -> UploadResult:
    """
    Upload a minidump file.

    Args:
        minidump_path: Path to minidump
        report_id: Associated report ID
        endpoint: Server endpoint

    Returns:
        Upload result
    """
    return get_uploader().upload_minidump(minidump_path, report_id, endpoint)


async def upload_async(
    report: CrashReport,
    endpoint: str = "/api/crashes",
) -> UploadResult:
    """
    Upload a crash report asynchronously.

    Args:
        report: Crash report to upload
        endpoint: Server endpoint

    Returns:
        Upload result
    """
    uploader = AsyncCrashUploader(UploadConfig.from_env())
    return await uploader.upload_async(report, endpoint)
