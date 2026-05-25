"""
File rotation support for the game engine logging system.

Provides handlers for rotating log files by size or time,
with optional compression of rotated files.

Example:
    >>> from engine.debug.logging.rotation import RotatingFileHandler
    >>>
    >>> handler = RotatingFileHandler(
    ...     "/var/log/game.log",
    ...     max_bytes=10 * 1024 * 1024,  # 10 MB
    ...     backup_count=5,
    ...     compress=True,
    ... )
"""

from __future__ import annotations

import gzip
import re
import shutil
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from engine.core.constants import (
    LOG_FILE_MAX_SIZE,
    LOG_FILE_MAX_BACKUPS,
    LOG_FILE_ENCODING,
    LOG_ROTATION_DAILY_BACKUPS,
    LOG_ARCHIVER_DEFAULT_DAYS,
    LOG_CLEANUP_DEFAULT_DAYS,
    SECONDS_PER_MINUTE,
    SECONDS_PER_HOUR,
    SECONDS_PER_DAY,
    SECONDS_PER_WEEK,
)

if TYPE_CHECKING:
    from engine.debug.logging.logger import LogEntry


class RotatingFileHandler:
    """
    Log file handler with rotation by size.

    Rotates log files when they exceed a maximum size, keeping
    a configurable number of backup files.

    Attributes:
        path: Path to the log file
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        compress: Whether to compress rotated files

    Example:
        >>> handler = RotatingFileHandler(
        ...     "game.log",
        ...     max_bytes=1024 * 1024,  # 1 MB
        ...     backup_count=3,
        ... )
        >>> handler.write("Log entry\\n")
    """

    def __init__(
        self,
        path: str | Path,
        max_bytes: int = LOG_FILE_MAX_SIZE,
        backup_count: int = LOG_FILE_MAX_BACKUPS,
        compress: bool = False,
        mode: str = "a",
        encoding: str = LOG_FILE_ENCODING,
    ) -> None:
        """
        Initialize the rotating file handler.

        Args:
            path: Path to the log file
            max_bytes: Maximum file size before rotation (bytes)
            backup_count: Number of backup files to keep
            compress: Whether to gzip rotated files
            mode: File open mode
            encoding: File encoding
        """
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.compress = compress
        self.mode = mode
        self.encoding = encoding

        self._lock = threading.Lock()
        self._file = None
        self._current_size = 0

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open the file
        self._open()

    def _open(self) -> None:
        """Open the log file."""
        self._file = open(
            self.path,
            mode=self.mode,
            encoding=self.encoding,
        )
        self._current_size = self.path.stat().st_size if self.path.exists() else 0

    def _close(self) -> None:
        """Close the log file."""
        if self._file:
            self._file.close()
            self._file = None

    def _should_rotate(self) -> bool:
        """Check if rotation is needed."""
        return self._current_size >= self.max_bytes

    def _get_backup_path(self, index: int) -> Path:
        """Get the path for a backup file."""
        suffix = f".{index}"
        if self.compress and index > 0:
            suffix += ".gz"
        return self.path.with_suffix(self.path.suffix + suffix)

    def _rotate(self) -> None:
        """Perform file rotation."""
        self._close()

        # Delete oldest backup if it exists
        oldest = self._get_backup_path(self.backup_count)
        if oldest.exists():
            oldest.unlink()

        # Shift existing backups
        for i in range(self.backup_count - 1, 0, -1):
            src = self._get_backup_path(i)
            dst = self._get_backup_path(i + 1)
            if src.exists():
                src.rename(dst)

        # Rotate current file to .1
        if self.path.exists():
            backup1 = self._get_backup_path(1)
            if self.compress:
                # Compress to .1.gz
                with open(self.path, "rb") as f_in:
                    with gzip.open(backup1, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                self.path.unlink()
            else:
                self.path.rename(backup1)

        # Open fresh file
        self._open()

    def write(self, data: str) -> None:
        """
        Write data to the log file.

        Args:
            data: String data to write
        """
        with self._lock:
            if self._file is None:
                self._open()

            if self._should_rotate():
                self._rotate()

            self._file.write(data)
            self._current_size += len(data.encode(self.encoding))

    def flush(self) -> None:
        """Flush the file buffer."""
        with self._lock:
            if self._file:
                self._file.flush()

    def close(self) -> None:
        """Close the handler."""
        with self._lock:
            self._close()

    def get_backup_files(self) -> list[Path]:
        """
        Get list of existing backup files.

        Returns:
            List of backup file paths
        """
        backups = []
        for i in range(1, self.backup_count + 1):
            backup = self._get_backup_path(i)
            if backup.exists():
                backups.append(backup)
        return backups


class TimedRotatingFileHandler:
    """
    Log file handler with rotation by time.

    Rotates log files at specified intervals (hourly, daily, etc.),
    keeping a configurable number of backup files.

    Attributes:
        path: Path to the log file
        when: When to rotate ('h', 'd', 'w', 'midnight')
        interval: Interval multiplier
        backup_count: Number of backups to keep

    Example:
        >>> # Rotate daily at midnight
        >>> handler = TimedRotatingFileHandler(
        ...     "game.log",
        ...     when="midnight",
        ...     backup_count=7,
        ... )
        >>>
        >>> # Rotate every 6 hours
        >>> handler = TimedRotatingFileHandler(
        ...     "game.log",
        ...     when="h",
        ...     interval=6,
        ... )
    """

    # Rotation intervals in seconds
    INTERVALS = {
        "s": 1,                    # Seconds (for testing)
        "m": SECONDS_PER_MINUTE,   # Minutes
        "h": SECONDS_PER_HOUR,     # Hours
        "d": SECONDS_PER_DAY,      # Days
        "w": SECONDS_PER_WEEK,     # Weeks
        "midnight": None,          # Special case
    }

    def __init__(
        self,
        path: str | Path,
        when: Literal["s", "m", "h", "d", "w", "midnight"] = "d",
        interval: int = 1,
        backup_count: int = LOG_ROTATION_DAILY_BACKUPS,
        compress: bool = False,
        encoding: str = LOG_FILE_ENCODING,
        utc: bool = False,
    ) -> None:
        """
        Initialize the timed rotating handler.

        Args:
            path: Path to the log file
            when: When to rotate ('s', 'm', 'h', 'd', 'w', 'midnight')
            interval: Interval multiplier
            backup_count: Number of backups to keep
            compress: Whether to compress rotated files
            encoding: File encoding
            utc: Use UTC time instead of local time
        """
        if when not in self.INTERVALS:
            raise ValueError(f"Invalid when value: {when}")

        self.path = Path(path)
        self.when = when
        self.interval = interval
        self.backup_count = backup_count
        self.compress = compress
        self.encoding = encoding
        self.utc = utc

        self._lock = threading.Lock()
        self._file = None

        # Calculate rotation interval
        if when == "midnight":
            self._interval_seconds = SECONDS_PER_DAY  # Check daily
        else:
            self._interval_seconds = self.INTERVALS[when] * interval

        # Calculate next rollover time
        self._next_rollover = self._compute_rollover()

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open the file
        self._open()

    def _get_time(self) -> datetime:
        """Get current time in configured timezone."""
        if self.utc:
            return datetime.now(timezone.utc)
        return datetime.now()

    def _compute_rollover(self) -> float:
        """Calculate the next rollover time."""
        now = time.time()

        if self.when == "midnight":
            # Calculate seconds until midnight
            t = self._get_time()

            # Next midnight
            next_midnight = t.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)

            return next_midnight.timestamp()
        else:
            # Simple interval-based rollover
            return now + self._interval_seconds

    def _open(self) -> None:
        """Open the log file."""
        self._file = open(
            self.path,
            mode="a",
            encoding=self.encoding,
        )

    def _close(self) -> None:
        """Close the log file."""
        if self._file:
            self._file.close()
            self._file = None

    def _should_rotate(self) -> bool:
        """Check if rotation is needed."""
        return time.time() >= self._next_rollover

    def _get_rotation_suffix(self, t: datetime) -> str:
        """Get the suffix for a rotated file based on time."""
        if self.when in ("s", "m"):
            return t.strftime("%Y%m%d_%H%M%S")
        elif self.when == "h":
            return t.strftime("%Y%m%d_%H")
        else:  # d, w, midnight
            return t.strftime("%Y%m%d")

    def _rotate(self) -> None:
        """Perform time-based rotation."""
        self._close()

        if self.path.exists():
            # Generate timestamped backup name
            t = self._get_time()
            suffix = self._get_rotation_suffix(t)
            backup_name = f"{self.path.stem}.{suffix}{self.path.suffix}"
            backup_path = self.path.parent / backup_name

            if self.compress:
                backup_path = backup_path.with_suffix(backup_path.suffix + ".gz")
                with open(self.path, "rb") as f_in:
                    with gzip.open(backup_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                self.path.unlink()
            else:
                self.path.rename(backup_path)

        # Clean up old backups
        self._cleanup_old_backups()

        # Update rollover time
        self._next_rollover = self._compute_rollover()

        # Open new file
        self._open()

    def _cleanup_old_backups(self) -> None:
        """Remove old backup files exceeding backup_count."""
        # Find all backup files
        pattern = re.compile(
            rf"^{re.escape(self.path.stem)}\.\d{{8}}.*{re.escape(self.path.suffix)}(\.gz)?$"
        )

        backups = []
        for f in self.path.parent.iterdir():
            if pattern.match(f.name):
                backups.append(f)

        # Sort by modification time (oldest first)
        backups.sort(key=lambda f: f.stat().st_mtime)

        # Delete oldest until we're within limit
        while len(backups) > self.backup_count:
            oldest = backups.pop(0)
            oldest.unlink()

    def write(self, data: str) -> None:
        """
        Write data to the log file.

        Args:
            data: String data to write
        """
        with self._lock:
            if self._file is None:
                self._open()

            if self._should_rotate():
                self._rotate()

            self._file.write(data)

    def flush(self) -> None:
        """Flush the file buffer."""
        with self._lock:
            if self._file:
                self._file.flush()

    def close(self) -> None:
        """Close the handler."""
        with self._lock:
            self._close()


class CompressedFileReader:
    """
    Reader for compressed log files.

    Supports reading both plain text and gzipped log files.

    Example:
        >>> reader = CompressedFileReader("game.log.1.gz")
        >>> for line in reader:
        ...     print(line)
    """

    def __init__(self, path: str | Path) -> None:
        """
        Initialize the reader.

        Args:
            path: Path to the log file
        """
        self.path = Path(path)
        self._is_gzipped = self.path.suffix == ".gz"

    def __iter__(self):
        """Iterate over lines in the file."""
        if self._is_gzipped:
            with gzip.open(self.path, "rt", encoding="utf-8") as f:
                yield from f
        else:
            with open(self.path, "r", encoding="utf-8") as f:
                yield from f

    def read(self) -> str:
        """
        Read the entire file contents.

        Returns:
            File contents as string
        """
        if self._is_gzipped:
            with gzip.open(self.path, "rt", encoding="utf-8") as f:
                return f.read()
        else:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read()

    def readlines(self) -> list[str]:
        """
        Read all lines from the file.

        Returns:
            List of lines
        """
        return list(self)


class LogArchiver:
    """
    Utility for archiving and managing old log files.

    Provides methods for compressing, moving, and cleaning up
    old log files.

    Example:
        >>> archiver = LogArchiver("/var/log/game/")
        >>> archiver.archive_old(days=30)
        >>> archiver.cleanup(days=90)
    """

    def __init__(
        self,
        log_dir: str | Path,
        archive_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the archiver.

        Args:
            log_dir: Directory containing log files
            archive_dir: Directory for archived files (default: log_dir/archive)
        """
        self.log_dir = Path(log_dir)
        self.archive_dir = Path(archive_dir) if archive_dir else self.log_dir / "archive"

        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def archive_old(
        self,
        days: int = LOG_ARCHIVER_DEFAULT_DAYS,
        pattern: str = "*.log*",
        compress: bool = True,
    ) -> list[Path]:
        """
        Archive log files older than specified days.

        Args:
            days: Age threshold in days
            pattern: Glob pattern for log files
            compress: Whether to compress during archival

        Returns:
            List of archived file paths
        """
        cutoff = time.time() - (days * SECONDS_PER_DAY)
        archived = []

        for path in self.log_dir.glob(pattern):
            if path.is_file() and path.stat().st_mtime < cutoff:
                # Skip already compressed files if we're compressing
                if compress and path.suffix == ".gz":
                    continue

                # Archive the file
                dest = self.archive_dir / path.name

                if compress and path.suffix != ".gz":
                    dest = dest.with_suffix(dest.suffix + ".gz")
                    with open(path, "rb") as f_in:
                        with gzip.open(dest, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    path.unlink()
                else:
                    shutil.move(str(path), str(dest))

                archived.append(dest)

        return archived

    def cleanup(
        self,
        days: int = LOG_CLEANUP_DEFAULT_DAYS,
        pattern: str = "*",
        include_archives: bool = True,
    ) -> int:
        """
        Delete log files older than specified days.

        Args:
            days: Age threshold in days
            pattern: Glob pattern for files
            include_archives: Also clean archive directory

        Returns:
            Number of files deleted
        """
        cutoff = time.time() - (days * SECONDS_PER_DAY)
        deleted = 0

        dirs = [self.log_dir]
        if include_archives:
            dirs.append(self.archive_dir)

        for directory in dirs:
            for path in directory.glob(pattern):
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1

        return deleted

    def get_total_size(self, include_archives: bool = True) -> int:
        """
        Get total size of all log files.

        Args:
            include_archives: Include archive directory

        Returns:
            Total size in bytes
        """
        total = 0

        dirs = [self.log_dir]
        if include_archives:
            dirs.append(self.archive_dir)

        for directory in dirs:
            for path in directory.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size

        return total

    def get_statistics(self) -> dict:
        """
        Get statistics about log files.

        Returns:
            Dictionary with statistics
        """
        log_files = list(self.log_dir.glob("*"))
        archive_files = list(self.archive_dir.glob("*"))

        log_size = sum(f.stat().st_size for f in log_files if f.is_file())
        archive_size = sum(f.stat().st_size for f in archive_files if f.is_file())

        return {
            "log_files": len([f for f in log_files if f.is_file()]),
            "archive_files": len([f for f in archive_files if f.is_file()]),
            "log_size_bytes": log_size,
            "archive_size_bytes": archive_size,
            "total_size_bytes": log_size + archive_size,
        }
