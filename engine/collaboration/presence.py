"""T-CC-4.5: Soft locking and presence system.

Implements advisory locking and real-time presence for collaborative editing:

- SoftLock: Advisory entity locking (doesn't block, just warns)
- LockRegistry: Tracks who has locks on which entities
- PresenceInfo: User cursor/selection state
- PresenceManager: Real-time presence tracking across editors
- LockConflictNotification: Conflict warnings for lock holders
"""
from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)


# =============================================================================
# Exceptions
# =============================================================================


class PresenceError(Exception):
    """Base exception for presence operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class LockError(PresenceError):
    """Base exception for lock operations."""

    pass


class LockNotFound(LockError):
    """Raised when a lock is not found."""

    def __init__(self, entity_id: str, user_id: Optional[str] = None):
        message = f"Lock not found for entity: {entity_id}"
        if user_id:
            message += f" (user: {user_id})"
        super().__init__(message, {"entity_id": entity_id, "user_id": user_id})
        self.entity_id = entity_id
        self.user_id = user_id


class LockConflict(LockError):
    """Raised when a lock conflict is detected (advisory warning)."""

    def __init__(
        self,
        entity_id: str,
        current_holder: str,
        requester: str,
    ):
        message = f"Lock conflict on {entity_id}: held by {current_holder}, requested by {requester}"
        super().__init__(
            message,
            {
                "entity_id": entity_id,
                "current_holder": current_holder,
                "requester": requester,
            },
        )
        self.entity_id = entity_id
        self.current_holder = current_holder
        self.requester = requester


class LockExpired(LockError):
    """Raised when attempting to use an expired lock."""

    def __init__(self, entity_id: str, user_id: str, expired_at: float):
        message = f"Lock expired for entity {entity_id} (user: {user_id})"
        super().__init__(
            message,
            {"entity_id": entity_id, "user_id": user_id, "expired_at": expired_at},
        )
        self.entity_id = entity_id
        self.user_id = user_id
        self.expired_at = expired_at


class UserNotFound(PresenceError):
    """Raised when a user is not found in presence system."""

    def __init__(self, user_id: str):
        super().__init__(f"User not found: {user_id}", {"user_id": user_id})
        self.user_id = user_id


# =============================================================================
# Enums
# =============================================================================


class LockType(Enum):
    """Types of soft locks."""

    SELECTION = auto()  # User selected the entity
    EDITING = auto()  # User is actively editing
    EXCLUSIVE = auto()  # User requests exclusive access (still advisory)


class LockPriority(Enum):
    """Priority levels for lock conflicts."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class PresenceStatus(Enum):
    """User presence status."""

    ONLINE = auto()  # Actively connected
    IDLE = auto()  # Connected but inactive
    AWAY = auto()  # Stepped away
    OFFLINE = auto()  # Disconnected


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CursorPosition:
    """Represents a cursor position in 3D space or viewport."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    viewport_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "viewport_id": self.viewport_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CursorPosition":
        """Deserialize from dictionary."""
        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0),
            viewport_id=data.get("viewport_id"),
        )

    def distance_to(self, other: "CursorPosition") -> float:
        """Calculate distance to another cursor position."""
        return (
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        ) ** 0.5


@dataclass
class Selection:
    """Represents a user's current selection."""

    entity_ids: Set[str] = field(default_factory=set)
    component_ids: Set[str] = field(default_factory=set)
    property_paths: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        # Ensure sets are actual sets
        if not isinstance(self.entity_ids, set):
            self.entity_ids = set(self.entity_ids)
        if not isinstance(self.component_ids, set):
            self.component_ids = set(self.component_ids)
        if not isinstance(self.property_paths, set):
            self.property_paths = set(self.property_paths)

    def is_empty(self) -> bool:
        """Check if selection is empty."""
        return (
            len(self.entity_ids) == 0
            and len(self.component_ids) == 0
            and len(self.property_paths) == 0
        )

    def overlaps(self, other: "Selection") -> bool:
        """Check if selections overlap."""
        return bool(
            self.entity_ids & other.entity_ids
            or self.component_ids & other.component_ids
            or self.property_paths & other.property_paths
        )

    def get_overlapping_entities(self, other: "Selection") -> Set[str]:
        """Get overlapping entity IDs."""
        return self.entity_ids & other.entity_ids

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entity_ids": list(self.entity_ids),
            "component_ids": list(self.component_ids),
            "property_paths": list(self.property_paths),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Selection":
        """Deserialize from dictionary."""
        return cls(
            entity_ids=set(data.get("entity_ids", [])),
            component_ids=set(data.get("component_ids", [])),
            property_paths=set(data.get("property_paths", [])),
        )


# =============================================================================
# Soft Lock
# =============================================================================


@dataclass
class SoftLock:
    """Advisory lock for an entity.

    Soft locks don't prevent edits - they warn other users that
    someone is working on the entity. Locks have timeouts and
    auto-release when stale.
    """

    entity_id: str
    user_id: str
    lock_type: LockType = LockType.SELECTION
    priority: LockPriority = LockPriority.NORMAL
    acquired_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    timeout_seconds: float = 60.0  # Default 60s timeout
    metadata: Dict[str, Any] = field(default_factory=dict)
    lock_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if lock has expired."""
        now = current_time or time.time()
        return (now - self.last_heartbeat) > self.timeout_seconds

    def refresh(self) -> None:
        """Refresh the lock heartbeat."""
        self.last_heartbeat = time.time()

    def time_remaining(self, current_time: Optional[float] = None) -> float:
        """Get time remaining before expiry."""
        now = current_time or time.time()
        elapsed = now - self.last_heartbeat
        return max(0.0, self.timeout_seconds - elapsed)

    def upgrade_to(self, lock_type: LockType) -> None:
        """Upgrade lock type (e.g., SELECTION -> EDITING)."""
        if lock_type.value > self.lock_type.value:
            self.lock_type = lock_type
            self.refresh()

    def downgrade_to(self, lock_type: LockType) -> None:
        """Downgrade lock type (e.g., EDITING -> SELECTION)."""
        if lock_type.value < self.lock_type.value:
            self.lock_type = lock_type
            self.refresh()

    def set_lock_type(self, lock_type: LockType) -> None:
        """Set lock type directly (upgrade or downgrade)."""
        self.lock_type = lock_type
        self.refresh()

    def can_override(self, other: "SoftLock") -> bool:
        """Check if this lock can override another (based on priority)."""
        # Higher priority can override lower
        if self.priority.value > other.priority.value:
            return True
        # Same priority, expired lock can be overridden
        if self.priority.value == other.priority.value and other.is_expired():
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entity_id": self.entity_id,
            "user_id": self.user_id,
            "lock_type": self.lock_type.name,
            "priority": self.priority.name,
            "acquired_at": self.acquired_at,
            "last_heartbeat": self.last_heartbeat,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
            "lock_id": self.lock_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoftLock":
        """Deserialize from dictionary."""
        return cls(
            entity_id=data["entity_id"],
            user_id=data["user_id"],
            lock_type=LockType[data.get("lock_type", "SELECTION")],
            priority=LockPriority[data.get("priority", "NORMAL")],
            acquired_at=data.get("acquired_at", time.time()),
            last_heartbeat=data.get("last_heartbeat", time.time()),
            timeout_seconds=data.get("timeout_seconds", 60.0),
            metadata=data.get("metadata", {}),
            lock_id=data.get("lock_id", str(uuid.uuid4())),
        )

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "SoftLock":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Lock Conflict Notification
# =============================================================================


@dataclass
class LockConflictNotification:
    """Notification about a lock conflict."""

    entity_id: str
    current_holder: str
    requester: str
    lock_type: LockType
    holder_priority: LockPriority
    requester_priority: LockPriority
    timestamp: float = field(default_factory=time.time)
    notification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message: Optional[str] = None
    acknowledged: bool = False

    def __post_init__(self) -> None:
        if self.message is None:
            self.message = (
                f"Entity '{self.entity_id}' is being edited by {self.current_holder}"
            )

    def acknowledge(self) -> None:
        """Mark notification as acknowledged."""
        self.acknowledged = True

    def is_override_allowed(self) -> bool:
        """Check if requester can override based on priority."""
        return self.requester_priority.value > self.holder_priority.value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entity_id": self.entity_id,
            "current_holder": self.current_holder,
            "requester": self.requester,
            "lock_type": self.lock_type.name,
            "holder_priority": self.holder_priority.name,
            "requester_priority": self.requester_priority.name,
            "timestamp": self.timestamp,
            "notification_id": self.notification_id,
            "message": self.message,
            "acknowledged": self.acknowledged,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LockConflictNotification":
        """Deserialize from dictionary."""
        return cls(
            entity_id=data["entity_id"],
            current_holder=data["current_holder"],
            requester=data["requester"],
            lock_type=LockType[data["lock_type"]],
            holder_priority=LockPriority[data["holder_priority"]],
            requester_priority=LockPriority[data["requester_priority"]],
            timestamp=data.get("timestamp", time.time()),
            notification_id=data.get("notification_id", str(uuid.uuid4())),
            message=data.get("message"),
            acknowledged=data.get("acknowledged", False),
        )


# =============================================================================
# Lock Registry
# =============================================================================


class LockRegistry:
    """Registry for tracking soft locks across entities.

    Thread-safe implementation for concurrent access.
    Provides lock acquisition, release, and conflict detection.
    """

    def __init__(
        self,
        default_timeout: float = 60.0,
        cleanup_interval: float = 30.0,
    ):
        """Initialize lock registry.

        Args:
            default_timeout: Default lock timeout in seconds
            cleanup_interval: Interval for automatic cleanup of expired locks
        """
        self._locks: Dict[str, SoftLock] = {}  # entity_id -> SoftLock
        self._user_locks: Dict[str, Set[str]] = {}  # user_id -> set of entity_ids
        self._conflict_handlers: List[Callable[[LockConflictNotification], None]] = []
        self._lock = threading.RLock()
        self._default_timeout = default_timeout
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    def acquire(
        self,
        entity_id: str,
        user_id: str,
        lock_type: LockType = LockType.SELECTION,
        priority: LockPriority = LockPriority.NORMAL,
        timeout: Optional[float] = None,
        force: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SoftLock, Optional[LockConflictNotification]]:
        """Acquire a soft lock on an entity.

        Args:
            entity_id: Entity to lock
            user_id: User acquiring lock
            lock_type: Type of lock
            priority: Lock priority
            timeout: Lock timeout (uses default if not specified)
            force: Force acquisition even with conflict
            metadata: Additional lock metadata

        Returns:
            Tuple of (acquired lock, conflict notification if any)
        """
        with self._lock:
            self._maybe_cleanup()

            existing = self._locks.get(entity_id)
            notification = None

            if existing and existing.user_id != user_id:
                # Check if existing lock is expired
                if existing.is_expired():
                    # Expired, can acquire
                    self._release_internal(entity_id)
                    existing = None
                else:
                    # Conflict detected
                    notification = LockConflictNotification(
                        entity_id=entity_id,
                        current_holder=existing.user_id,
                        requester=user_id,
                        lock_type=lock_type,
                        holder_priority=existing.priority,
                        requester_priority=priority,
                    )
                    self._notify_conflict(notification)

                    # Check if force or can override
                    new_lock = SoftLock(
                        entity_id=entity_id,
                        user_id=user_id,
                        lock_type=lock_type,
                        priority=priority,
                    )
                    if not force and not new_lock.can_override(existing):
                        # Return existing lock and notification
                        return existing, notification

                    # Override existing lock
                    self._release_internal(entity_id)

            # Create or upgrade lock
            if existing and existing.user_id == user_id:
                # Same user, upgrade lock
                existing.upgrade_to(lock_type)
                existing.priority = max(existing.priority, priority, key=lambda p: p.value)
                existing.refresh()
                return existing, notification

            # New lock
            lock = SoftLock(
                entity_id=entity_id,
                user_id=user_id,
                lock_type=lock_type,
                priority=priority,
                timeout_seconds=timeout or self._default_timeout,
                metadata=metadata or {},
            )
            self._locks[entity_id] = lock
            if user_id not in self._user_locks:
                self._user_locks[user_id] = set()
            self._user_locks[user_id].add(entity_id)

            return lock, notification

    def release(self, entity_id: str, user_id: str) -> bool:
        """Release a lock.

        Args:
            entity_id: Entity to unlock
            user_id: User releasing lock (must be owner)

        Returns:
            True if released, False if not found or not owner
        """
        with self._lock:
            lock = self._locks.get(entity_id)
            if lock and lock.user_id == user_id:
                self._release_internal(entity_id)
                return True
            return False

    def release_all(self, user_id: str) -> int:
        """Release all locks held by a user.

        Args:
            user_id: User whose locks to release

        Returns:
            Number of locks released
        """
        with self._lock:
            entity_ids = self._user_locks.get(user_id, set()).copy()
            for entity_id in entity_ids:
                self._release_internal(entity_id)
            return len(entity_ids)

    def _release_internal(self, entity_id: str) -> None:
        """Internal release without lock (caller must hold lock)."""
        lock = self._locks.pop(entity_id, None)
        if lock:
            user_locks = self._user_locks.get(lock.user_id)
            if user_locks:
                user_locks.discard(entity_id)
                if not user_locks:
                    del self._user_locks[lock.user_id]

    def refresh(self, entity_id: str, user_id: str) -> bool:
        """Refresh a lock's heartbeat.

        Args:
            entity_id: Entity lock to refresh
            user_id: User refreshing (must be owner)

        Returns:
            True if refreshed, False if not found or not owner
        """
        with self._lock:
            lock = self._locks.get(entity_id)
            if lock and lock.user_id == user_id:
                lock.refresh()
                return True
            return False

    def get(self, entity_id: str) -> Optional[SoftLock]:
        """Get lock for an entity."""
        with self._lock:
            lock = self._locks.get(entity_id)
            if lock and lock.is_expired():
                self._release_internal(entity_id)
                return None
            return lock

    def get_user_locks(self, user_id: str) -> List[SoftLock]:
        """Get all locks held by a user."""
        with self._lock:
            entity_ids = self._user_locks.get(user_id, set())
            locks = []
            for entity_id in list(entity_ids):
                lock = self._locks.get(entity_id)
                if lock:
                    if lock.is_expired():
                        self._release_internal(entity_id)
                    else:
                        locks.append(lock)
            return locks

    def is_locked(self, entity_id: str) -> bool:
        """Check if entity is locked."""
        return self.get(entity_id) is not None

    def is_locked_by(self, entity_id: str, user_id: str) -> bool:
        """Check if entity is locked by specific user."""
        lock = self.get(entity_id)
        return lock is not None and lock.user_id == user_id

    def get_holder(self, entity_id: str) -> Optional[str]:
        """Get user ID of lock holder."""
        lock = self.get(entity_id)
        return lock.user_id if lock else None

    def get_all_locks(self) -> Dict[str, SoftLock]:
        """Get all current locks."""
        with self._lock:
            self._cleanup_expired()
            return dict(self._locks)

    def check_conflict(
        self,
        entity_id: str,
        user_id: str,
    ) -> Optional[LockConflictNotification]:
        """Check for lock conflict without acquiring.

        Returns notification if conflict exists, None otherwise.
        """
        with self._lock:
            lock = self.get(entity_id)
            if lock and lock.user_id != user_id:
                return LockConflictNotification(
                    entity_id=entity_id,
                    current_holder=lock.user_id,
                    requester=user_id,
                    lock_type=lock.lock_type,
                    holder_priority=lock.priority,
                    requester_priority=LockPriority.NORMAL,
                )
            return None

    def add_conflict_handler(
        self,
        handler: Callable[[LockConflictNotification], None],
    ) -> None:
        """Add a conflict notification handler."""
        with self._lock:
            self._conflict_handlers.append(handler)

    def remove_conflict_handler(
        self,
        handler: Callable[[LockConflictNotification], None],
    ) -> None:
        """Remove a conflict notification handler."""
        with self._lock:
            if handler in self._conflict_handlers:
                self._conflict_handlers.remove(handler)

    def _notify_conflict(self, notification: LockConflictNotification) -> None:
        """Notify all conflict handlers."""
        for handler in self._conflict_handlers:
            try:
                handler(notification)
            except Exception:
                pass  # Don't let handler errors break the system

    def _maybe_cleanup(self) -> None:
        """Perform cleanup if interval has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = now

    def _cleanup_expired(self) -> None:
        """Remove all expired locks."""
        expired = [
            entity_id
            for entity_id, lock in self._locks.items()
            if lock.is_expired()
        ]
        for entity_id in expired:
            self._release_internal(entity_id)

    def cleanup(self) -> int:
        """Force cleanup of expired locks.

        Returns:
            Number of locks cleaned up
        """
        with self._lock:
            initial_count = len(self._locks)
            self._cleanup_expired()
            return initial_count - len(self._locks)

    def stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            locks = list(self._locks.values())
            by_type = {}
            by_priority = {}
            for lock in locks:
                by_type[lock.lock_type.name] = by_type.get(lock.lock_type.name, 0) + 1
                by_priority[lock.priority.name] = by_priority.get(lock.priority.name, 0) + 1

            return {
                "total_locks": len(locks),
                "total_users": len(self._user_locks),
                "by_type": by_type,
                "by_priority": by_priority,
                "expired_count": sum(1 for l in locks if l.is_expired()),
            }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        with self._lock:
            return {
                "locks": {k: v.to_dict() for k, v in self._locks.items()},
                "default_timeout": self._default_timeout,
                "cleanup_interval": self._cleanup_interval,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LockRegistry":
        """Deserialize from dictionary."""
        registry = cls(
            default_timeout=data.get("default_timeout", 60.0),
            cleanup_interval=data.get("cleanup_interval", 30.0),
        )
        for entity_id, lock_data in data.get("locks", {}).items():
            lock = SoftLock.from_dict(lock_data)
            registry._locks[entity_id] = lock
            if lock.user_id not in registry._user_locks:
                registry._user_locks[lock.user_id] = set()
            registry._user_locks[lock.user_id].add(entity_id)
        return registry


# =============================================================================
# Presence Info
# =============================================================================


@dataclass
class PresenceInfo:
    """Information about a user's presence in the collaborative session.

    Tracks cursor position, selection, and activity status.
    """

    user_id: str
    cursor_position: CursorPosition = field(default_factory=CursorPosition)
    selection: Selection = field(default_factory=Selection)
    status: PresenceStatus = PresenceStatus.ONLINE
    display_name: Optional[str] = None
    color: str = "#3498db"  # Default blue
    avatar_url: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_cursor(self, position: CursorPosition) -> None:
        """Update cursor position."""
        self.cursor_position = position
        self.last_activity = time.time()
        self.timestamp = time.time()
        if self.status == PresenceStatus.IDLE:
            self.status = PresenceStatus.ONLINE

    def update_selection(self, selection: Selection) -> None:
        """Update selection."""
        self.selection = selection
        self.last_activity = time.time()
        self.timestamp = time.time()
        if self.status == PresenceStatus.IDLE:
            self.status = PresenceStatus.ONLINE

    def set_status(self, status: PresenceStatus) -> None:
        """Set presence status."""
        self.status = status
        self.timestamp = time.time()

    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_activity = time.time()
        self.timestamp = time.time()

    def is_stale(self, timeout: float = 30.0) -> bool:
        """Check if presence is stale (no recent activity)."""
        return (time.time() - self.last_activity) > timeout

    def time_since_activity(self) -> float:
        """Get time since last activity."""
        return time.time() - self.last_activity

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "cursor_position": self.cursor_position.to_dict(),
            "selection": self.selection.to_dict(),
            "status": self.status.name,
            "display_name": self.display_name,
            "color": self.color,
            "avatar_url": self.avatar_url,
            "timestamp": self.timestamp,
            "last_activity": self.last_activity,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PresenceInfo":
        """Deserialize from dictionary."""
        return cls(
            user_id=data["user_id"],
            cursor_position=CursorPosition.from_dict(data.get("cursor_position", {})),
            selection=Selection.from_dict(data.get("selection", {})),
            status=PresenceStatus[data.get("status", "ONLINE")],
            display_name=data.get("display_name"),
            color=data.get("color", "#3498db"),
            avatar_url=data.get("avatar_url"),
            timestamp=data.get("timestamp", time.time()),
            last_activity=data.get("last_activity", time.time()),
            session_id=data.get("session_id", str(uuid.uuid4())),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "PresenceInfo":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Presence Manager
# =============================================================================


class PresenceManager:
    """Manager for tracking user presence in collaborative editing.

    Handles:
    - User join/leave
    - Cursor/selection updates
    - Presence broadcasting
    - Stale presence cleanup
    - Selection overlap detection
    """

    def __init__(
        self,
        idle_timeout: float = 60.0,
        offline_timeout: float = 120.0,
        cleanup_interval: float = 30.0,
    ):
        """Initialize presence manager.

        Args:
            idle_timeout: Seconds until user marked IDLE
            offline_timeout: Seconds until user marked OFFLINE
            cleanup_interval: Interval for automatic cleanup
        """
        self._users: Dict[str, PresenceInfo] = {}
        self._presence_handlers: List[Callable[[PresenceInfo, str], None]] = []
        self._lock = threading.RLock()
        self._idle_timeout = idle_timeout
        self._offline_timeout = offline_timeout
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._color_palette = [
            "#e74c3c",  # Red
            "#3498db",  # Blue
            "#2ecc71",  # Green
            "#9b59b6",  # Purple
            "#f39c12",  # Orange
            "#1abc9c",  # Teal
            "#e91e63",  # Pink
            "#00bcd4",  # Cyan
        ]
        self._next_color_index = 0

    def _assign_color(self) -> str:
        """Assign a color from the palette."""
        color = self._color_palette[self._next_color_index % len(self._color_palette)]
        self._next_color_index += 1
        return color

    def join(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        color: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PresenceInfo:
        """User joins the collaborative session.

        Args:
            user_id: Unique user identifier
            display_name: Display name for UI
            color: User color (auto-assigned if not provided)
            metadata: Additional user metadata

        Returns:
            PresenceInfo for the user
        """
        with self._lock:
            if user_id in self._users:
                # User already present, update and return
                presence = self._users[user_id]
                presence.status = PresenceStatus.ONLINE
                presence.heartbeat()
                if display_name:
                    presence.display_name = display_name
                if color:
                    presence.color = color
                if metadata:
                    presence.metadata.update(metadata)
                self._notify_presence(presence, "update")
                return presence

            presence = PresenceInfo(
                user_id=user_id,
                display_name=display_name or user_id,
                color=color or self._assign_color(),
                metadata=metadata or {},
            )
            self._users[user_id] = presence
            self._notify_presence(presence, "join")
            return presence

    def leave(self, user_id: str) -> bool:
        """User leaves the collaborative session.

        Args:
            user_id: User to remove

        Returns:
            True if user was present and removed
        """
        with self._lock:
            if user_id in self._users:
                presence = self._users.pop(user_id)
                presence.status = PresenceStatus.OFFLINE
                self._notify_presence(presence, "leave")
                return True
            return False

    def update_cursor(self, user_id: str, position: CursorPosition) -> bool:
        """Update a user's cursor position.

        Args:
            user_id: User to update
            position: New cursor position

        Returns:
            True if updated, False if user not found
        """
        with self._lock:
            presence = self._users.get(user_id)
            if presence:
                presence.update_cursor(position)
                self._notify_presence(presence, "cursor")
                return True
            return False

    def update_selection(self, user_id: str, selection: Selection) -> bool:
        """Update a user's selection.

        Args:
            user_id: User to update
            selection: New selection

        Returns:
            True if updated, False if user not found
        """
        with self._lock:
            presence = self._users.get(user_id)
            if presence:
                presence.update_selection(selection)
                self._notify_presence(presence, "selection")
                return True
            return False

    def heartbeat(self, user_id: str) -> bool:
        """Update user heartbeat.

        Args:
            user_id: User to update

        Returns:
            True if updated, False if user not found
        """
        with self._lock:
            presence = self._users.get(user_id)
            if presence:
                presence.heartbeat()
                return True
            return False

    def set_status(self, user_id: str, status: PresenceStatus) -> bool:
        """Set user presence status.

        Args:
            user_id: User to update
            status: New status

        Returns:
            True if updated, False if user not found
        """
        with self._lock:
            presence = self._users.get(user_id)
            if presence:
                presence.set_status(status)
                self._notify_presence(presence, "status")
                return True
            return False

    def get(self, user_id: str) -> Optional[PresenceInfo]:
        """Get presence info for a user."""
        with self._lock:
            return self._users.get(user_id)

    def get_all(self) -> Dict[str, PresenceInfo]:
        """Get all presence info."""
        with self._lock:
            self._maybe_cleanup()
            return dict(self._users)

    def get_online_users(self) -> List[PresenceInfo]:
        """Get all online users (not OFFLINE)."""
        with self._lock:
            self._maybe_cleanup()
            return [p for p in self._users.values() if p.status != PresenceStatus.OFFLINE]

    def get_active_users(self) -> List[PresenceInfo]:
        """Get all active users (ONLINE only)."""
        with self._lock:
            self._maybe_cleanup()
            return [p for p in self._users.values() if p.status == PresenceStatus.ONLINE]

    def get_users_at_entity(self, entity_id: str) -> List[PresenceInfo]:
        """Get users who have selected an entity."""
        with self._lock:
            return [
                p for p in self._users.values()
                if entity_id in p.selection.entity_ids
            ]

    def get_selection_overlaps(self, user_id: str) -> Dict[str, Set[str]]:
        """Get overlapping selections with other users.

        Args:
            user_id: User to check overlaps for

        Returns:
            Dict mapping other user_id to set of overlapping entity_ids
        """
        with self._lock:
            presence = self._users.get(user_id)
            if not presence:
                return {}

            overlaps = {}
            for other_id, other_presence in self._users.items():
                if other_id != user_id:
                    overlap = presence.selection.get_overlapping_entities(
                        other_presence.selection
                    )
                    if overlap:
                        overlaps[other_id] = overlap

            return overlaps

    def is_online(self, user_id: str) -> bool:
        """Check if user is online."""
        presence = self.get(user_id)
        return presence is not None and presence.status != PresenceStatus.OFFLINE

    def user_count(self) -> int:
        """Get total number of users."""
        with self._lock:
            return len(self._users)

    def online_count(self) -> int:
        """Get number of online users."""
        return len(self.get_online_users())

    def add_presence_handler(
        self,
        handler: Callable[[PresenceInfo, str], None],
    ) -> None:
        """Add a presence change handler.

        Handler receives (presence_info, event_type) where event_type is one of:
        'join', 'leave', 'cursor', 'selection', 'status', 'update'
        """
        with self._lock:
            self._presence_handlers.append(handler)

    def remove_presence_handler(
        self,
        handler: Callable[[PresenceInfo, str], None],
    ) -> None:
        """Remove a presence change handler."""
        with self._lock:
            if handler in self._presence_handlers:
                self._presence_handlers.remove(handler)

    def _notify_presence(self, presence: PresenceInfo, event_type: str) -> None:
        """Notify all presence handlers."""
        for handler in self._presence_handlers:
            try:
                handler(presence, event_type)
            except Exception:
                pass  # Don't let handler errors break the system

    def _maybe_cleanup(self) -> None:
        """Perform cleanup if interval has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self._update_stale_statuses()
            self._last_cleanup = now

    def _update_stale_statuses(self) -> None:
        """Update status of stale users."""
        now = time.time()
        for presence in list(self._users.values()):
            elapsed = now - presence.last_activity
            if elapsed > self._offline_timeout:
                if presence.status != PresenceStatus.OFFLINE:
                    presence.status = PresenceStatus.OFFLINE
                    self._notify_presence(presence, "status")
            elif elapsed > self._idle_timeout:
                if presence.status == PresenceStatus.ONLINE:
                    presence.status = PresenceStatus.IDLE
                    self._notify_presence(presence, "status")

    def cleanup(self) -> int:
        """Force cleanup and remove offline users.

        Returns:
            Number of users removed
        """
        with self._lock:
            self._update_stale_statuses()
            offline = [
                user_id for user_id, p in self._users.items()
                if p.status == PresenceStatus.OFFLINE
            ]
            for user_id in offline:
                self._users.pop(user_id, None)
            return len(offline)

    def stats(self) -> Dict[str, Any]:
        """Get presence statistics."""
        with self._lock:
            users = list(self._users.values())
            by_status = {}
            for presence in users:
                by_status[presence.status.name] = by_status.get(presence.status.name, 0) + 1

            return {
                "total_users": len(users),
                "by_status": by_status,
                "total_selections": sum(
                    len(p.selection.entity_ids) for p in users
                ),
                "idle_timeout": self._idle_timeout,
                "offline_timeout": self._offline_timeout,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        with self._lock:
            return {
                "users": {k: v.to_dict() for k, v in self._users.items()},
                "idle_timeout": self._idle_timeout,
                "offline_timeout": self._offline_timeout,
                "cleanup_interval": self._cleanup_interval,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PresenceManager":
        """Deserialize from dictionary."""
        manager = cls(
            idle_timeout=data.get("idle_timeout", 60.0),
            offline_timeout=data.get("offline_timeout", 120.0),
            cleanup_interval=data.get("cleanup_interval", 30.0),
        )
        for user_id, presence_data in data.get("users", {}).items():
            manager._users[user_id] = PresenceInfo.from_dict(presence_data)
        return manager


# =============================================================================
# Collaborative Session
# =============================================================================


class CollaborativeSession:
    """Unified session combining presence and locks.

    Provides a high-level API for collaborative editing sessions.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        lock_timeout: float = 60.0,
        idle_timeout: float = 60.0,
        offline_timeout: float = 120.0,
    ):
        """Initialize collaborative session.

        Args:
            session_id: Unique session identifier
            lock_timeout: Default lock timeout
            idle_timeout: Time until user marked idle
            offline_timeout: Time until user marked offline
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.presence = PresenceManager(
            idle_timeout=idle_timeout,
            offline_timeout=offline_timeout,
        )
        self.locks = LockRegistry(default_timeout=lock_timeout)
        self._lock = threading.RLock()
        self._created_at = time.time()

    def join(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> PresenceInfo:
        """User joins the session."""
        return self.presence.join(user_id, display_name, color)

    def leave(self, user_id: str) -> None:
        """User leaves the session."""
        self.locks.release_all(user_id)
        self.presence.leave(user_id)

    def select_entity(
        self,
        user_id: str,
        entity_id: str,
        lock_type: LockType = LockType.SELECTION,
    ) -> Tuple[bool, Optional[LockConflictNotification]]:
        """Select an entity and acquire soft lock.

        Returns:
            Tuple of (success, conflict notification if any)
        """
        with self._lock:
            presence = self.presence.get(user_id)
            if not presence:
                return False, None

            # Update selection
            presence.selection.entity_ids.add(entity_id)
            presence.heartbeat()

            # Acquire lock
            lock, conflict = self.locks.acquire(
                entity_id=entity_id,
                user_id=user_id,
                lock_type=lock_type,
            )

            return True, conflict

    def deselect_entity(self, user_id: str, entity_id: str) -> bool:
        """Deselect an entity and release lock."""
        with self._lock:
            presence = self.presence.get(user_id)
            if not presence:
                return False

            presence.selection.entity_ids.discard(entity_id)
            presence.heartbeat()
            self.locks.release(entity_id, user_id)
            return True

    def start_editing(
        self,
        user_id: str,
        entity_id: str,
    ) -> Tuple[bool, Optional[LockConflictNotification]]:
        """Start editing an entity (upgrades to EDITING lock)."""
        return self.select_entity(user_id, entity_id, LockType.EDITING)

    def stop_editing(self, user_id: str, entity_id: str) -> bool:
        """Stop editing an entity (downgrades to SELECTION lock)."""
        with self._lock:
            lock = self.locks.get(entity_id)
            if lock and lock.user_id == user_id:
                lock.downgrade_to(LockType.SELECTION)
                return True
            return False

    def get_entity_editors(self, entity_id: str) -> List[PresenceInfo]:
        """Get all users who have selected an entity."""
        return self.presence.get_users_at_entity(entity_id)

    def get_entity_lock_holder(self, entity_id: str) -> Optional[PresenceInfo]:
        """Get the user holding the lock on an entity."""
        lock = self.locks.get(entity_id)
        if lock:
            return self.presence.get(lock.user_id)
        return None

    def heartbeat(self, user_id: str) -> bool:
        """Update user heartbeat and refresh locks."""
        with self._lock:
            self.presence.heartbeat(user_id)
            for lock in self.locks.get_user_locks(user_id):
                lock.refresh()
            return True

    def get_conflicts_for_user(self, user_id: str) -> List[LockConflictNotification]:
        """Get all lock conflicts affecting a user."""
        conflicts = []
        presence = self.presence.get(user_id)
        if presence:
            for entity_id in presence.selection.entity_ids:
                conflict = self.locks.check_conflict(entity_id, user_id)
                if conflict:
                    conflicts.append(conflict)
        return conflicts

    def stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "created_at": self._created_at,
            "uptime_seconds": time.time() - self._created_at,
            "presence": self.presence.stats(),
            "locks": self.locks.stats(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self._created_at,
            "presence": self.presence.to_dict(),
            "locks": self.locks.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollaborativeSession":
        """Deserialize from dictionary."""
        session = cls(session_id=data.get("session_id"))
        session._created_at = data.get("created_at", time.time())
        session.presence = PresenceManager.from_dict(data.get("presence", {}))
        session.locks = LockRegistry.from_dict(data.get("locks", {}))
        return session
