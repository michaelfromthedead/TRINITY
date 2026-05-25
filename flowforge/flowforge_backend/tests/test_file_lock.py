"""Tests for file_lock module."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from ..codegen.file_lock import FileLock, LockInfo, _lock_path, _pid_exists


class TestFileLockAcquireRelease:
    """Tests for basic acquire and release."""

    def test_acquire_creates_lock_file(self, tmp_path):
        """Test that acquire creates a .flowforge.lock file."""
        target = str(tmp_path / "target.py")
        open(target, "w").close()

        lock = FileLock()
        assert lock.acquire(target) is True

        lock_file = _lock_path(os.path.normpath(os.path.abspath(target)))
        assert os.path.exists(lock_file)

        lock.release(target)

    def test_release_removes_lock_file(self, tmp_path):
        """Test that release removes the lock file."""
        target = str(tmp_path / "target.py")
        open(target, "w").close()

        lock = FileLock()
        lock.acquire(target)
        lock.release(target)

        lock_file = _lock_path(os.path.normpath(os.path.abspath(target)))
        assert not os.path.exists(lock_file)

    def test_lock_file_contains_pid_and_hostname(self, tmp_path):
        """Test that the lock file contains PID, timestamp, hostname."""
        target = str(tmp_path / "target.py")
        open(target, "w").close()

        lock = FileLock()
        lock.acquire(target)

        lock_file = _lock_path(os.path.normpath(os.path.abspath(target)))
        with open(lock_file, "r") as f:
            data = json.load(f)

        assert data["pid"] == os.getpid()
        assert "timestamp" in data
        assert "hostname" in data

        lock.release(target)

    def test_acquire_same_process_succeeds(self, tmp_path):
        """Test that acquiring twice from same process succeeds."""
        target = str(tmp_path / "target.py")
        open(target, "w").close()

        lock = FileLock()
        assert lock.acquire(target) is True
        assert lock.acquire(target) is True

        lock.release(target)


class TestDoubleAcquireFails:
    """Tests for concurrent lock rejection."""

    def test_acquire_fails_if_locked_by_another_pid(self, tmp_path):
        """Test that acquire returns False when locked by another live process."""
        target = str(tmp_path / "target.py")
        open(target, "w").close()
        abs_target = os.path.normpath(os.path.abspath(target))
        lock_file = _lock_path(abs_target)

        # Simulate a lock from PID 1 (init/systemd, always alive)
        import socket
        info = {
            "pid": 1,
            "timestamp": 1000.0,
            "hostname": socket.gethostname(),
        }
        with open(lock_file, "w") as f:
            json.dump(info, f)

        lock = FileLock()
        assert lock.acquire(target) is False

        # Clean up
        os.remove(lock_file)


class TestStaleLockDetection:
    """Tests for stale lock cleanup."""

    def test_stale_lock_is_cleaned_and_acquired(self, tmp_path):
        """Test that a lock from a dead PID is cleaned up."""
        target = str(tmp_path / "target.py")
        open(target, "w").close()
        abs_target = os.path.normpath(os.path.abspath(target))
        lock_file = _lock_path(abs_target)

        # Write a lock with a PID that doesn't exist
        import socket
        info = {
            "pid": 999999999,
            "timestamp": 1000.0,
            "hostname": socket.gethostname(),
        }
        with open(lock_file, "w") as f:
            json.dump(info, f)

        lock = FileLock()
        # Should detect stale lock, clean up, and acquire
        assert lock.acquire(target) is True

        lock.release(target)

    def test_force_release_removes_any_lock(self, tmp_path):
        """Test that force_release removes lock regardless of owner."""
        target = str(tmp_path / "target.py")
        open(target, "w").close()
        abs_target = os.path.normpath(os.path.abspath(target))
        lock_file = _lock_path(abs_target)

        info = {"pid": 1, "timestamp": 1000.0, "hostname": "other-host"}
        with open(lock_file, "w") as f:
            json.dump(info, f)

        lock = FileLock()
        lock.force_release(target)

        assert not os.path.exists(lock_file)

    def test_is_locked_returns_none_when_unlocked(self, tmp_path):
        """Test is_locked returns None for unlocked files."""
        target = str(tmp_path / "target.py")
        open(target, "w").close()

        lock = FileLock()
        assert lock.is_locked(target) is None

    def test_is_locked_returns_info_when_locked(self, tmp_path):
        """Test is_locked returns LockInfo when locked."""
        target = str(tmp_path / "target.py")
        open(target, "w").close()

        lock = FileLock()
        lock.acquire(target)

        info = lock.is_locked(target)
        assert info is not None
        assert info.pid == os.getpid()

        lock.release(target)


class TestContextManager:
    """Tests for context manager support."""

    def test_context_manager_acquires_and_releases(self, tmp_path):
        """Test that the context manager acquires on enter and releases on exit."""
        target = str(tmp_path / "target.py")
        open(target, "w").close()
        abs_target = os.path.normpath(os.path.abspath(target))

        lock = FileLock()
        with lock(target):
            assert lock.is_locked(target) is not None

        assert lock.is_locked(target) is None

    def test_context_manager_releases_on_exception(self, tmp_path):
        """Test that the lock is released even if an exception occurs."""
        target = str(tmp_path / "target.py")
        open(target, "w").close()

        lock = FileLock()
        with pytest.raises(ValueError):
            with lock(target):
                raise ValueError("boom")

        assert lock.is_locked(target) is None

    def test_context_manager_raises_if_cannot_acquire(self, tmp_path):
        """Test that entering the context manager raises if lock is held."""
        target = str(tmp_path / "target.py")
        open(target, "w").close()
        abs_target = os.path.normpath(os.path.abspath(target))
        lock_file = _lock_path(abs_target)

        import socket
        info = {"pid": 1, "timestamp": 1000.0, "hostname": socket.gethostname()}
        with open(lock_file, "w") as f:
            json.dump(info, f)

        lock = FileLock()
        with pytest.raises(RuntimeError, match="Could not acquire lock"):
            with lock(target):
                pass  # pragma: no cover

        os.remove(lock_file)


class TestPidExists:
    """Tests for _pid_exists helper."""

    def test_current_pid_exists(self):
        """Test that the current process PID is detected as existing."""
        assert _pid_exists(os.getpid()) is True

    def test_invalid_pid_does_not_exist(self):
        """Test that an unlikely PID is detected as not existing."""
        assert _pid_exists(999999999) is False

    def test_zero_pid_does_not_exist(self):
        """Test that PID 0 returns False."""
        assert _pid_exists(0) is False

    def test_negative_pid_does_not_exist(self):
        """Test that negative PID returns False."""
        assert _pid_exists(-1) is False


class TestLockInfo:
    """Tests for LockInfo dataclass."""

    def test_to_dict(self):
        """Test LockInfo serialization."""
        info = LockInfo(pid=42, timestamp=1234.5, hostname="myhost")
        d = info.to_dict()
        assert d == {"pid": 42, "timestamp": 1234.5, "hostname": "myhost"}

    def test_from_dict(self):
        """Test LockInfo deserialization."""
        info = LockInfo.from_dict({"pid": 42, "timestamp": 1234.5, "hostname": "myhost"})
        assert info.pid == 42
        assert info.timestamp == 1234.5
        assert info.hostname == "myhost"
