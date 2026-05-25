"""File Lock Module for FlowForge Backend.

Provides file-level locking to prevent concurrent writes during code generation.
Lock files are created next to the target file with a `.flowforge.lock` extension.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class LockInfo:
    """Information about an active file lock.

    Attributes:
        pid: Process ID that holds the lock
        timestamp: Unix timestamp when the lock was acquired
        hostname: Hostname of the machine that acquired the lock
    """
    pid: int
    timestamp: float
    hostname: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pid": self.pid,
            "timestamp": self.timestamp,
            "hostname": self.hostname,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LockInfo:
        """Create LockInfo from a dictionary."""
        return cls(
            pid=data["pid"],
            timestamp=data["timestamp"],
            hostname=data["hostname"],
        )


def _lock_path(file_path: str) -> str:
    """Return the lock file path for a given file."""
    return file_path + ".flowforge.lock"


def _pid_exists(pid: int) -> bool:
    """Check whether a process with the given PID exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True
    return True


class FileLock:
    """File-level lock using a sidecar `.flowforge.lock` file.

    Supports context manager usage::

        lock = FileLock()
        with lock(path):
            # file is locked
            ...
        # lock is released

    Or manual acquire/release::

        lock = FileLock()
        if lock.acquire(path):
            try:
                ...
            finally:
                lock.release(path)
    """

    def __init__(self) -> None:
        self._held_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def acquire(self, file_path: str) -> bool:
        """Acquire a lock on *file_path*.

        Creates a ``.flowforge.lock`` file next to the target containing
        the current PID, timestamp, and hostname.

        If the file is already locked by a **live** process, returns False.
        Stale locks (PID no longer running) are automatically cleaned up
        before acquiring.

        Args:
            file_path: Absolute or relative path to the file to lock.

        Returns:
            True if the lock was acquired, False if another process holds it.
        """
        file_path = os.path.normpath(os.path.abspath(file_path))
        lock_file = _lock_path(file_path)

        # Check for existing lock
        existing = self.is_locked(file_path)
        if existing is not None:
            if existing.pid == os.getpid():
                # We already hold it
                return True
            if _pid_exists(existing.pid) and existing.hostname == socket.gethostname():
                # Another live process on this host holds the lock
                return False
            # Stale lock – remove it
            self.force_release(file_path)

        info = LockInfo(
            pid=os.getpid(),
            timestamp=time.time(),
            hostname=socket.gethostname(),
        )

        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, json.dumps(info.to_dict()).encode("utf-8"))
            finally:
                os.close(fd)
        except FileExistsError:
            # Race: someone else created the file between our check and open
            return False

        self._held_path = file_path
        return True

    def release(self, file_path: str) -> None:
        """Release a lock on *file_path*.

        Removes the ``.flowforge.lock`` file. Only removes if the lock
        is held by the current process.

        Args:
            file_path: Path to the locked file.
        """
        file_path = os.path.normpath(os.path.abspath(file_path))
        lock_file = _lock_path(file_path)

        existing = self.is_locked(file_path)
        if existing is not None and existing.pid == os.getpid():
            try:
                os.remove(lock_file)
            except FileNotFoundError:
                pass

        if self._held_path == file_path:
            self._held_path = None

    def is_locked(self, file_path: str) -> Optional[LockInfo]:
        """Check whether *file_path* is currently locked.

        Args:
            file_path: Path to the file to check.

        Returns:
            A ``LockInfo`` if the file is locked, otherwise ``None``.
        """
        file_path = os.path.normpath(os.path.abspath(file_path))
        lock_file = _lock_path(file_path)

        if not os.path.exists(lock_file):
            return None

        try:
            with open(lock_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return LockInfo.from_dict(data)
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def force_release(self, file_path: str) -> None:
        """Forcibly remove a lock file regardless of ownership.

        Useful for cleaning up stale locks where the owning process
        no longer exists.

        Args:
            file_path: Path to the locked file.
        """
        file_path = os.path.normpath(os.path.abspath(file_path))
        lock_file = _lock_path(file_path)
        try:
            os.remove(lock_file)
        except FileNotFoundError:
            pass

        if self._held_path == file_path:
            self._held_path = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __call__(self, file_path: str) -> FileLock:
        """Return self configured for use as a context manager on *file_path*."""
        self._cm_path = os.path.normpath(os.path.abspath(file_path))
        return self

    def __enter__(self) -> FileLock:
        path = getattr(self, "_cm_path", None)
        if path is None:
            raise RuntimeError("FileLock context manager requires a file path. Use: with lock(path):")
        if not self.acquire(path):
            raise RuntimeError(f"Could not acquire lock on {path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        path = getattr(self, "_cm_path", None)
        if path is not None:
            self.release(path)
            self._cm_path = None
        return None
