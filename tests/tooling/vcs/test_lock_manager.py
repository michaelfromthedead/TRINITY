"""Tests for file locking manager."""
import pytest
import os
import json
import time
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from engine.tooling.vcs.lock_manager import (
    LockType,
    LockState,
    LockInfo,
    LockManager,
    BinaryFileLockManager,
)
from engine.tooling.vcs.vcs_integration import VCSError


@pytest.fixture
def mock_provider(tmp_path):
    """Create a mock VCS provider."""
    from engine.tooling.vcs.vcs_integration import VCSProvider, VCSType

    provider = MagicMock(spec=VCSProvider)
    provider.root_path = str(tmp_path)
    provider.vcs_type = VCSType.GIT
    return provider


@pytest.fixture
def lock_manager(mock_provider, tmp_path):
    """Create a lock manager with temp directory."""
    lock_dir = tmp_path / "locks"
    manager = LockManager(mock_provider, str(lock_dir))
    return manager


class TestLockType:
    """Tests for LockType enum."""

    def test_all_types_exist(self):
        """Test all lock types exist."""
        assert LockType.EXCLUSIVE
        assert LockType.SHARED
        assert LockType.INTENT


class TestLockState:
    """Tests for LockState enum."""

    def test_all_states_exist(self):
        """Test all lock states exist."""
        assert LockState.LOCKED
        assert LockState.UNLOCKED
        assert LockState.PENDING
        assert LockState.BREAKING
        assert LockState.STOLEN


class TestLockInfo:
    """Tests for LockInfo dataclass."""

    def test_lock_info_creation(self):
        """Test creating lock info."""
        lock = LockInfo(
            path="assets/texture.png",
            lock_type=LockType.EXCLUSIVE,
            state=LockState.LOCKED,
            owner="testuser",
        )
        assert lock.path == "assets/texture.png"
        assert lock.owner == "testuser"

    def test_is_expired(self):
        """Test expiration checking."""
        # Not expired - no expiry set
        lock1 = LockInfo(
            path="file.txt",
            lock_type=LockType.EXCLUSIVE,
            state=LockState.LOCKED,
            owner="user",
        )
        assert lock1.is_expired is False

        # Not expired - future expiry
        lock2 = LockInfo(
            path="file.txt",
            lock_type=LockType.EXCLUSIVE,
            state=LockState.LOCKED,
            owner="user",
            expires=time.time() + 3600,
        )
        assert lock2.is_expired is False

        # Expired
        lock3 = LockInfo(
            path="file.txt",
            lock_type=LockType.EXCLUSIVE,
            state=LockState.LOCKED,
            owner="user",
            expires=time.time() - 1,
        )
        assert lock3.is_expired is True

    def test_age_seconds(self):
        """Test age calculation."""
        lock = LockInfo(
            path="file.txt",
            lock_type=LockType.EXCLUSIVE,
            state=LockState.LOCKED,
            owner="user",
            timestamp=time.time() - 60,
        )
        assert lock.age_seconds >= 60

    def test_to_dict(self):
        """Test converting to dictionary."""
        lock = LockInfo(
            path="file.txt",
            lock_type=LockType.EXCLUSIVE,
            state=LockState.LOCKED,
            owner="user",
            reason="Editing",
        )
        data = lock.to_dict()
        assert data["path"] == "file.txt"
        assert data["owner"] == "user"
        assert data["lock_type"] == "EXCLUSIVE"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "path": "model.fbx",
            "lock_type": "EXCLUSIVE",
            "state": "LOCKED",
            "owner": "artist",
            "reason": "Rigging work",
        }
        lock = LockInfo.from_dict(data)
        assert lock.path == "model.fbx"
        assert lock.lock_type == LockType.EXCLUSIVE


class TestLockManager:
    """Tests for LockManager."""

    def test_manager_creation(self, mock_provider, tmp_path):
        """Test creating lock manager."""
        lock_dir = tmp_path / "locks"
        manager = LockManager(mock_provider, str(lock_dir))
        assert os.path.exists(str(lock_dir))

    def test_is_binary_file(self, lock_manager):
        """Test binary file detection."""
        assert lock_manager.is_binary_file("texture.png") is True
        assert lock_manager.is_binary_file("model.fbx") is True
        assert lock_manager.is_binary_file("sound.wav") is True
        assert lock_manager.is_binary_file("script.py") is False
        assert lock_manager.is_binary_file("config.json") is False

    def test_should_lock(self, lock_manager):
        """Test should_lock detection."""
        assert lock_manager.should_lock("art/character.blend") is True
        assert lock_manager.should_lock("src/main.cpp") is False

    def test_lock_file(self, lock_manager):
        """Test locking a file."""
        lock = lock_manager.lock("assets/hero.fbx", reason="Modeling")

        assert lock.path == "assets/hero.fbx"
        assert lock.state == LockState.LOCKED
        assert lock_manager.is_locked("assets/hero.fbx")

    def test_unlock_file(self, lock_manager):
        """Test unlocking a file."""
        lock_manager.lock("assets/texture.png")
        result = lock_manager.unlock("assets/texture.png")

        assert result is True
        assert not lock_manager.is_locked("assets/texture.png")

    def test_lock_already_locked(self, lock_manager):
        """Test locking already locked file."""
        lock_manager.lock("file.png")

        # Same user can re-lock
        lock2 = lock_manager.lock("file.png")
        assert lock2 is not None

    def test_lock_by_different_user(self, lock_manager):
        """Test locking file locked by another user."""
        lock_manager.lock("file.png")

        # Simulate different user
        with patch.object(lock_manager, '_current_user', 'other_user'):
            with pytest.raises(VCSError):
                lock_manager.lock("file.png")

    def test_unlock_not_owner(self, lock_manager):
        """Test unlocking file not owned."""
        lock_manager.lock("file.png")

        with patch.object(lock_manager, '_current_user', 'other_user'):
            with pytest.raises(VCSError):
                lock_manager.unlock("file.png")

    def test_unlock_force(self, lock_manager):
        """Test force unlock."""
        lock_manager.lock("file.png")

        with patch.object(lock_manager, '_current_user', 'other_user'):
            result = lock_manager.unlock("file.png", force=True)
            assert result is True

    def test_get_lock(self, lock_manager):
        """Test getting lock info."""
        lock_manager.lock("file.png", reason="Editing texture")
        lock = lock_manager.get_lock("file.png")

        assert lock is not None
        assert lock.reason == "Editing texture"

    def test_get_all_locks(self, lock_manager):
        """Test getting all locks."""
        lock_manager.lock("file1.png")
        lock_manager.lock("file2.fbx")

        locks = lock_manager.get_all_locks()
        assert len(locks) == 2

    def test_get_locks_by_user(self, lock_manager):
        """Test getting locks by user."""
        lock_manager.lock("file1.png")
        lock_manager.lock("file2.fbx")

        # Get current user's locks
        locks = lock_manager.get_my_locks()
        assert len(locks) == 2

    def test_can_edit(self, lock_manager):
        """Test can_edit check."""
        can_edit, reason = lock_manager.can_edit("unlocked.png")
        assert can_edit is True
        assert reason is None

        lock_manager.lock("locked.png")
        can_edit, reason = lock_manager.can_edit("locked.png")
        assert can_edit is True  # Current user owns it

        with patch.object(lock_manager, '_current_user', 'other_user'):
            can_edit, reason = lock_manager.can_edit("locked.png")
            assert can_edit is False

    def test_break_lock(self, lock_manager):
        """Test breaking a lock."""
        lock_manager.lock("file.png")
        result = lock_manager.break_lock("file.png", reason="Admin override")

        assert result is True
        assert not lock_manager.is_locked("file.png")

    def test_transfer_lock(self, lock_manager):
        """Test transferring a lock."""
        lock_manager.lock("file.png")
        result = lock_manager.transfer_lock("file.png", "new_owner")

        assert result is True
        lock = lock_manager.get_lock("file.png")
        assert lock.owner == "new_owner"

    def test_refresh_lock(self, lock_manager):
        """Test refreshing lock expiration."""
        lock = lock_manager.lock("file.png", expires_hours=1)
        old_expiry = lock.expires

        time.sleep(0.1)
        lock_manager.refresh_lock("file.png", extend_hours=24)

        lock = lock_manager.get_lock("file.png")
        assert lock.expires > old_expiry

    def test_lock_with_expiry(self, lock_manager):
        """Test lock with expiration."""
        lock = lock_manager.lock("file.png", expires_hours=24)
        assert lock.expires is not None
        assert lock.expires > time.time()


class TestBinaryFileLockManager:
    """Tests for BinaryFileLockManager."""

    @pytest.fixture
    def binary_manager(self, mock_provider, tmp_path):
        """Create binary file lock manager."""
        lock_dir = tmp_path / "locks"
        manager = BinaryFileLockManager(mock_provider, lock_dir=str(lock_dir))
        return manager

    def test_enable_auto_lock(self, binary_manager):
        """Test enabling/disabling auto-lock."""
        binary_manager.enable_auto_lock(True)
        assert binary_manager._auto_lock_enabled is True

        binary_manager.enable_auto_lock(False)
        assert binary_manager._auto_lock_enabled is False

    def test_add_watch_pattern(self, binary_manager):
        """Test adding watch patterns."""
        binary_manager.add_watch_pattern("*.custom")
        assert "*.custom" in binary_manager._watch_patterns

    def test_should_auto_lock(self, binary_manager):
        """Test auto-lock detection."""
        binary_manager.enable_auto_lock(True)

        assert binary_manager.should_auto_lock("texture.png") is True
        assert binary_manager.should_auto_lock("script.py") is False

        # Add custom pattern
        binary_manager.add_watch_pattern("*.level")
        assert binary_manager.should_auto_lock("map.level") is True

    def test_auto_lock_disabled(self, binary_manager):
        """Test auto-lock when disabled."""
        binary_manager.enable_auto_lock(False)
        assert binary_manager.should_auto_lock("texture.png") is False

    def test_auto_lock_on_edit(self, binary_manager):
        """Test auto-locking on edit."""
        binary_manager.enable_auto_lock(True)
        lock = binary_manager.auto_lock_on_edit("model.fbx")

        assert lock is not None
        assert binary_manager.is_locked("model.fbx")

    def test_auto_lock_on_edit_not_binary(self, binary_manager):
        """Test no auto-lock for non-binary files."""
        binary_manager.enable_auto_lock(True)
        lock = binary_manager.auto_lock_on_edit("script.py")

        assert lock is None

    def test_get_lockable_files(self, binary_manager):
        """Test getting lockable files."""
        paths = [
            "texture.png",
            "model.fbx",
            "script.py",
            "config.json",
        ]
        lockable = binary_manager.get_lockable_files(paths)

        assert "texture.png" in lockable
        assert "model.fbx" in lockable
        assert "script.py" not in lockable

    def test_lock_batch(self, binary_manager):
        """Test batch locking."""
        paths = ["tex1.png", "tex2.png", "model.fbx"]
        locks = binary_manager.lock_batch(paths, reason="Batch edit")

        assert len(locks) == 3
        for path in paths:
            assert binary_manager.is_locked(path)

    def test_unlock_batch(self, binary_manager):
        """Test batch unlocking."""
        paths = ["tex1.png", "tex2.png"]
        for path in paths:
            binary_manager.lock(path)

        unlocked = binary_manager.unlock_batch(paths)
        assert len(unlocked) == 2

    def test_cleanup_expired_locks(self, binary_manager):
        """Test cleaning up expired locks."""
        # Create expired lock
        lock = LockInfo(
            path="old.png",
            lock_type=LockType.EXCLUSIVE,
            state=LockState.LOCKED,
            owner="user",
            expires=time.time() - 1,
        )
        binary_manager._locks["old.png"] = lock

        cleaned = binary_manager.cleanup_expired_locks()
        assert "old.png" in cleaned
        assert not binary_manager.is_locked("old.png")

    def test_get_lock_report(self, binary_manager):
        """Test generating lock report."""
        binary_manager.lock("tex1.png")
        binary_manager.lock("tex2.png")
        binary_manager.lock("model.fbx")

        report = binary_manager.get_lock_report()

        assert report["total_locks"] == 3
        assert ".png" in report["by_extension"]
        assert ".fbx" in report["by_extension"]
