"""File locking for binary assets.

Provides file locking mechanisms to prevent concurrent editing of
binary files that cannot be merged (textures, meshes, audio, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json
import os
import threading
import time

from .vcs_integration import VCSProvider, VCSType, VCSError


class LockType(Enum):
    """Type of file lock."""
    EXCLUSIVE = auto()     # Only one user can edit
    SHARED = auto()        # Multiple users can read
    INTENT = auto()        # Intention to lock soon


class LockState(Enum):
    """State of a lock."""
    LOCKED = auto()
    UNLOCKED = auto()
    PENDING = auto()
    BREAKING = auto()
    STOLEN = auto()


@dataclass
class LockInfo:
    """Information about a file lock."""
    path: str
    lock_type: LockType
    state: LockState
    owner: str
    owner_email: str = ""
    timestamp: float = field(default_factory=time.time)
    expires: Optional[float] = None
    reason: str = ""
    machine: str = ""
    branch: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the lock has expired."""
        if self.expires is None:
            return False
        return time.time() > self.expires

    @property
    def age_seconds(self) -> float:
        """Get the age of the lock in seconds."""
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "lock_type": self.lock_type.name,
            "state": self.state.name,
            "owner": self.owner,
            "owner_email": self.owner_email,
            "timestamp": self.timestamp,
            "expires": self.expires,
            "reason": self.reason,
            "machine": self.machine,
            "branch": self.branch,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LockInfo:
        """Create from dictionary."""
        return cls(
            path=data["path"],
            lock_type=LockType[data["lock_type"]],
            state=LockState[data["state"]],
            owner=data["owner"],
            owner_email=data.get("owner_email", ""),
            timestamp=data.get("timestamp", time.time()),
            expires=data.get("expires"),
            reason=data.get("reason", ""),
            machine=data.get("machine", ""),
            branch=data.get("branch", ""),
            metadata=data.get("metadata", {}),
        )


class LockManager:
    """Manages file locks."""

    # Binary file extensions that should be locked
    BINARY_EXTENSIONS = {
        # Textures
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tga", ".tiff", ".exr", ".hdr", ".dds",
        ".psd", ".ai", ".svg",
        # 3D Models
        ".fbx", ".obj", ".blend", ".max", ".mb", ".ma", ".3ds", ".dae", ".gltf", ".glb",
        # Audio
        ".wav", ".mp3", ".ogg", ".flac", ".aiff", ".aac", ".wma",
        # Video
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv",
        # Archives
        ".zip", ".tar", ".gz", ".7z", ".rar",
        # Documents
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        # Fonts
        ".ttf", ".otf", ".woff", ".woff2",
        # Other binary
        ".exe", ".dll", ".so", ".dylib", ".a", ".lib",
        ".uasset", ".umap",  # Unreal Engine
    }

    def __init__(self, provider: VCSProvider, lock_dir: Optional[str] = None):
        self._provider = provider
        self._lock_dir = lock_dir or os.path.join(provider.root_path, ".locks")
        self._locks: Dict[str, LockInfo] = {}
        self._lock = threading.Lock()
        self._current_user = self._get_current_user()

        # Create lock directory if it doesn't exist
        os.makedirs(self._lock_dir, exist_ok=True)

        # Load existing locks
        self._load_locks()

    def _get_current_user(self) -> str:
        """Get the current user name."""
        if self._provider.vcs_type == VCSType.GIT:
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "config", "user.name"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                pass

        # Fall back to system user
        return os.environ.get("USER", os.environ.get("USERNAME", "unknown"))

    def _load_locks(self) -> None:
        """Load locks from storage."""
        lock_file = os.path.join(self._lock_dir, "locks.json")
        if os.path.exists(lock_file):
            try:
                with open(lock_file, "r") as f:
                    data = json.load(f)
                self._locks = {
                    path: LockInfo.from_dict(info) for path, info in data.items()
                }
            except Exception:
                self._locks = {}

    def _save_locks(self) -> None:
        """Save locks to storage."""
        os.makedirs(self._lock_dir, exist_ok=True)
        lock_file = os.path.join(self._lock_dir, "locks.json")

        data = {path: info.to_dict() for path, info in self._locks.items()}
        with open(lock_file, "w") as f:
            json.dump(data, f, indent=2)

    def is_binary_file(self, path: str) -> bool:
        """Check if a file is likely binary based on extension."""
        ext = os.path.splitext(path)[1].lower()
        return ext in self.BINARY_EXTENSIONS

    def should_lock(self, path: str) -> bool:
        """Check if a file should be locked before editing."""
        return self.is_binary_file(path)

    def lock(
        self,
        path: str,
        lock_type: LockType = LockType.EXCLUSIVE,
        reason: str = "",
        expires_hours: Optional[float] = None
    ) -> LockInfo:
        """Lock a file."""
        with self._lock:
            # Check if already locked by someone else
            if path in self._locks:
                existing = self._locks[path]
                if existing.owner != self._current_user and not existing.is_expired:
                    raise VCSError(
                        f"File '{path}' is already locked by {existing.owner}"
                    )

            # Create lock
            expires = None
            if expires_hours:
                expires = time.time() + (expires_hours * 3600)

            lock_info = LockInfo(
                path=path,
                lock_type=lock_type,
                state=LockState.LOCKED,
                owner=self._current_user,
                expires=expires,
                reason=reason,
                machine=os.uname().nodename if hasattr(os, "uname") else "",
            )

            self._locks[path] = lock_info
            self._save_locks()

            # If using Git LFS, also lock in LFS
            if self._provider.vcs_type == VCSType.GIT:
                self._git_lfs_lock(path)

            return lock_info

    def unlock(self, path: str, force: bool = False) -> bool:
        """Unlock a file."""
        with self._lock:
            if path not in self._locks:
                return True  # Already unlocked

            lock_info = self._locks[path]

            # Check ownership
            if lock_info.owner != self._current_user and not force:
                raise VCSError(
                    f"Cannot unlock '{path}': owned by {lock_info.owner}"
                )

            del self._locks[path]
            self._save_locks()

            # If using Git LFS, also unlock in LFS
            if self._provider.vcs_type == VCSType.GIT:
                self._git_lfs_unlock(path, force)

            return True

    def is_locked(self, path: str) -> bool:
        """Check if a file is locked."""
        if path not in self._locks:
            return False

        lock_info = self._locks[path]
        if lock_info.is_expired:
            # Clean up expired lock
            del self._locks[path]
            self._save_locks()
            return False

        return True

    def get_lock(self, path: str) -> Optional[LockInfo]:
        """Get lock information for a file."""
        return self._locks.get(path)

    def get_all_locks(self) -> List[LockInfo]:
        """Get all current locks."""
        # Clean up expired locks
        expired = [p for p, l in self._locks.items() if l.is_expired]
        for path in expired:
            del self._locks[path]
        if expired:
            self._save_locks()

        return list(self._locks.values())

    def get_locks_by_user(self, user: str) -> List[LockInfo]:
        """Get all locks owned by a user."""
        return [l for l in self._locks.values() if l.owner == user]

    def get_my_locks(self) -> List[LockInfo]:
        """Get all locks owned by the current user."""
        return self.get_locks_by_user(self._current_user)

    def can_edit(self, path: str) -> Tuple[bool, Optional[str]]:
        """Check if the current user can edit a file."""
        if path not in self._locks:
            return True, None

        lock_info = self._locks[path]

        if lock_info.is_expired:
            return True, None

        if lock_info.owner == self._current_user:
            return True, None

        return False, f"File is locked by {lock_info.owner}"

    def break_lock(self, path: str, reason: str = "") -> bool:
        """Forcefully break a lock (admin operation)."""
        with self._lock:
            if path not in self._locks:
                return True

            lock_info = self._locks[path]
            lock_info.state = LockState.BREAKING
            lock_info.metadata["broken_by"] = self._current_user
            lock_info.metadata["break_reason"] = reason
            lock_info.metadata["break_time"] = time.time()

            del self._locks[path]
            self._save_locks()

            if self._provider.vcs_type == VCSType.GIT:
                self._git_lfs_unlock(path, force=True)

            return True

    def transfer_lock(self, path: str, new_owner: str) -> bool:
        """Transfer a lock to another user."""
        with self._lock:
            if path not in self._locks:
                raise VCSError(f"File '{path}' is not locked")

            lock_info = self._locks[path]

            if lock_info.owner != self._current_user:
                raise VCSError(
                    f"Cannot transfer lock: owned by {lock_info.owner}"
                )

            lock_info.owner = new_owner
            lock_info.metadata["transferred_from"] = self._current_user
            lock_info.metadata["transfer_time"] = time.time()
            self._save_locks()

            return True

    def refresh_lock(self, path: str, extend_hours: float = 24) -> bool:
        """Extend a lock's expiration."""
        with self._lock:
            if path not in self._locks:
                return False

            lock_info = self._locks[path]

            if lock_info.owner != self._current_user:
                raise VCSError(
                    f"Cannot refresh lock: owned by {lock_info.owner}"
                )

            if lock_info.expires:
                lock_info.expires = time.time() + (extend_hours * 3600)
            else:
                lock_info.expires = time.time() + (extend_hours * 3600)

            self._save_locks()
            return True

    def _git_lfs_lock(self, path: str) -> bool:
        """Lock a file using Git LFS."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "lfs", "lock", path],
                cwd=self._provider.root_path,
                capture_output=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _git_lfs_unlock(self, path: str, force: bool = False) -> bool:
        """Unlock a file using Git LFS."""
        try:
            import subprocess
            cmd = ["git", "lfs", "unlock", path]
            if force:
                cmd.append("--force")

            result = subprocess.run(
                cmd,
                cwd=self._provider.root_path,
                capture_output=True,
            )
            return result.returncode == 0
        except Exception:
            return False


class BinaryFileLockManager(LockManager):
    """Specialized lock manager for binary files with auto-detection."""

    def __init__(self, provider: VCSProvider, **kwargs):
        super().__init__(provider, **kwargs)
        self._auto_lock_enabled = True
        self._watch_patterns: Set[str] = set()

    def enable_auto_lock(self, enabled: bool = True) -> None:
        """Enable or disable automatic locking of binary files."""
        self._auto_lock_enabled = enabled

    def add_watch_pattern(self, pattern: str) -> None:
        """Add a file pattern to watch for auto-locking."""
        self._watch_patterns.add(pattern)

    def remove_watch_pattern(self, pattern: str) -> None:
        """Remove a file pattern from watching."""
        self._watch_patterns.discard(pattern)

    def should_auto_lock(self, path: str) -> bool:
        """Check if a file should be auto-locked."""
        if not self._auto_lock_enabled:
            return False

        if self.is_binary_file(path):
            return True

        # Check custom patterns
        import fnmatch
        for pattern in self._watch_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True

        return False

    def auto_lock_on_edit(self, path: str) -> Optional[LockInfo]:
        """Automatically lock a file when editing begins."""
        if not self.should_auto_lock(path):
            return None

        if self.is_locked(path):
            lock = self.get_lock(path)
            if lock and lock.owner == self._current_user:
                return lock
            return None  # Can't auto-lock if someone else has it

        return self.lock(path, reason="Auto-locked for editing")

    def get_lockable_files(self, paths: List[str]) -> List[str]:
        """Filter paths to only those that can be locked."""
        lockable = []
        for path in paths:
            if self.should_lock(path):
                can_lock, reason = self.can_edit(path)
                if can_lock:
                    lockable.append(path)
        return lockable

    def lock_batch(
        self,
        paths: List[str],
        lock_type: LockType = LockType.EXCLUSIVE,
        reason: str = ""
    ) -> Dict[str, LockInfo]:
        """Lock multiple files at once."""
        results = {}
        errors = []

        for path in paths:
            try:
                lock_info = self.lock(path, lock_type, reason)
                results[path] = lock_info
            except VCSError as e:
                errors.append((path, str(e)))

        if errors:
            # Rollback successful locks
            for path in results:
                try:
                    self.unlock(path)
                except Exception:
                    pass
            raise VCSError(f"Failed to lock some files: {errors}")

        return results

    def unlock_batch(self, paths: List[str], force: bool = False) -> List[str]:
        """Unlock multiple files at once."""
        unlocked = []
        for path in paths:
            try:
                if self.unlock(path, force):
                    unlocked.append(path)
            except VCSError:
                pass
        return unlocked

    def cleanup_expired_locks(self) -> List[str]:
        """Remove all expired locks."""
        cleaned = []
        with self._lock:
            expired = [p for p, l in self._locks.items() if l.is_expired]
            for path in expired:
                del self._locks[path]
                cleaned.append(path)

            if cleaned:
                self._save_locks()

        return cleaned

    def get_lock_report(self) -> Dict[str, Any]:
        """Generate a report of current locks."""
        locks = self.get_all_locks()

        users = {}
        for lock in locks:
            if lock.owner not in users:
                users[lock.owner] = []
            users[lock.owner].append(lock.path)

        by_type = {}
        for lock in locks:
            ext = os.path.splitext(lock.path)[1].lower()
            if ext not in by_type:
                by_type[ext] = 0
            by_type[ext] += 1

        return {
            "total_locks": len(locks),
            "by_user": {u: len(p) for u, p in users.items()},
            "by_extension": by_type,
            "oldest_lock": min((l.timestamp for l in locks), default=0),
            "expiring_soon": [
                l.path for l in locks
                if l.expires and l.expires - time.time() < 3600
            ],
        }
