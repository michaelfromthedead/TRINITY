"""T-CC-4.5: Tests for soft locking and presence system.

Tests cover:
- SoftLock: advisory locking with timeouts
- LockRegistry: lock acquisition, release, conflicts
- PresenceInfo: cursor, selection, status tracking
- PresenceManager: user join/leave, presence updates
- CollaborativeSession: unified presence + locks
- Lock conflict notifications
- Auto-expiry and cleanup
- Thread safety
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from engine.collaboration.presence import (
    CollaborativeSession,
    CursorPosition,
    LockConflict,
    LockConflictNotification,
    LockError,
    LockExpired,
    LockNotFound,
    LockPriority,
    LockRegistry,
    LockType,
    PresenceError,
    PresenceInfo,
    PresenceManager,
    PresenceStatus,
    Selection,
    SoftLock,
    UserNotFound,
)


# =============================================================================
# CursorPosition Tests
# =============================================================================


class TestCursorPosition:
    """Tests for CursorPosition."""

    def test_create_default(self):
        """Test creating default cursor position."""
        cursor = CursorPosition()
        assert cursor.x == 0.0
        assert cursor.y == 0.0
        assert cursor.z == 0.0
        assert cursor.viewport_id is None

    def test_create_with_values(self):
        """Test creating cursor with specific values."""
        cursor = CursorPosition(x=1.5, y=2.5, z=3.5, viewport_id="main")
        assert cursor.x == 1.5
        assert cursor.y == 2.5
        assert cursor.z == 3.5
        assert cursor.viewport_id == "main"

    def test_to_dict(self):
        """Test serialization to dict."""
        cursor = CursorPosition(x=1.0, y=2.0, z=3.0, viewport_id="vp1")
        data = cursor.to_dict()
        assert data == {
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "viewport_id": "vp1",
        }

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {"x": 1.0, "y": 2.0, "z": 3.0, "viewport_id": "vp1"}
        cursor = CursorPosition.from_dict(data)
        assert cursor.x == 1.0
        assert cursor.y == 2.0
        assert cursor.z == 3.0
        assert cursor.viewport_id == "vp1"

    def test_from_dict_defaults(self):
        """Test deserialization with missing values."""
        cursor = CursorPosition.from_dict({})
        assert cursor.x == 0.0
        assert cursor.y == 0.0
        assert cursor.z == 0.0
        assert cursor.viewport_id is None

    def test_distance_to(self):
        """Test distance calculation."""
        c1 = CursorPosition(x=0.0, y=0.0, z=0.0)
        c2 = CursorPosition(x=3.0, y=4.0, z=0.0)
        assert c1.distance_to(c2) == 5.0

    def test_distance_to_3d(self):
        """Test 3D distance calculation."""
        c1 = CursorPosition(x=1.0, y=2.0, z=3.0)
        c2 = CursorPosition(x=4.0, y=6.0, z=3.0)
        assert c1.distance_to(c2) == 5.0


# =============================================================================
# Selection Tests
# =============================================================================


class TestSelection:
    """Tests for Selection."""

    def test_create_empty(self):
        """Test creating empty selection."""
        sel = Selection()
        assert sel.is_empty()
        assert len(sel.entity_ids) == 0
        assert len(sel.component_ids) == 0
        assert len(sel.property_paths) == 0

    def test_create_with_values(self):
        """Test creating selection with values."""
        sel = Selection(
            entity_ids={"e1", "e2"},
            component_ids={"c1"},
            property_paths={"transform.position"},
        )
        assert not sel.is_empty()
        assert sel.entity_ids == {"e1", "e2"}
        assert sel.component_ids == {"c1"}
        assert sel.property_paths == {"transform.position"}

    def test_create_from_lists(self):
        """Test creating selection from lists (converted to sets)."""
        sel = Selection(
            entity_ids=["e1", "e2"],
            component_ids=["c1"],
            property_paths=["p1"],
        )
        assert sel.entity_ids == {"e1", "e2"}
        assert sel.component_ids == {"c1"}

    def test_overlaps_entities(self):
        """Test overlap detection for entities."""
        sel1 = Selection(entity_ids={"e1", "e2"})
        sel2 = Selection(entity_ids={"e2", "e3"})
        assert sel1.overlaps(sel2)

    def test_overlaps_components(self):
        """Test overlap detection for components."""
        sel1 = Selection(component_ids={"c1", "c2"})
        sel2 = Selection(component_ids={"c2", "c3"})
        assert sel1.overlaps(sel2)

    def test_no_overlap(self):
        """Test when selections don't overlap."""
        sel1 = Selection(entity_ids={"e1", "e2"})
        sel2 = Selection(entity_ids={"e3", "e4"})
        assert not sel1.overlaps(sel2)

    def test_get_overlapping_entities(self):
        """Test getting overlapping entities."""
        sel1 = Selection(entity_ids={"e1", "e2", "e3"})
        sel2 = Selection(entity_ids={"e2", "e3", "e4"})
        overlap = sel1.get_overlapping_entities(sel2)
        assert overlap == {"e2", "e3"}

    def test_to_dict(self):
        """Test serialization to dict."""
        sel = Selection(entity_ids={"e1"}, component_ids={"c1"})
        data = sel.to_dict()
        assert set(data["entity_ids"]) == {"e1"}
        assert set(data["component_ids"]) == {"c1"}

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "entity_ids": ["e1", "e2"],
            "component_ids": ["c1"],
            "property_paths": [],
        }
        sel = Selection.from_dict(data)
        assert sel.entity_ids == {"e1", "e2"}
        assert sel.component_ids == {"c1"}


# =============================================================================
# SoftLock Tests
# =============================================================================


class TestSoftLock:
    """Tests for SoftLock advisory locking."""

    def test_create_basic(self):
        """Test creating a basic lock."""
        lock = SoftLock(entity_id="entity1", user_id="user1")
        assert lock.entity_id == "entity1"
        assert lock.user_id == "user1"
        assert lock.lock_type == LockType.SELECTION
        assert lock.priority == LockPriority.NORMAL
        assert not lock.is_expired()

    def test_create_with_type(self):
        """Test creating lock with specific type."""
        lock = SoftLock(
            entity_id="entity1",
            user_id="user1",
            lock_type=LockType.EDITING,
        )
        assert lock.lock_type == LockType.EDITING

    def test_create_with_priority(self):
        """Test creating lock with priority."""
        lock = SoftLock(
            entity_id="entity1",
            user_id="user1",
            priority=LockPriority.HIGH,
        )
        assert lock.priority == LockPriority.HIGH

    def test_is_expired_false(self):
        """Test lock is not expired immediately."""
        lock = SoftLock(entity_id="e1", user_id="u1", timeout_seconds=60.0)
        assert not lock.is_expired()

    def test_is_expired_true(self):
        """Test lock expires after timeout."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            timeout_seconds=0.0,
            last_heartbeat=time.time() - 1.0,
        )
        assert lock.is_expired()

    def test_refresh(self):
        """Test refreshing lock heartbeat."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            last_heartbeat=time.time() - 100.0,
        )
        old_heartbeat = lock.last_heartbeat
        lock.refresh()
        assert lock.last_heartbeat > old_heartbeat

    def test_time_remaining(self):
        """Test time remaining calculation."""
        lock = SoftLock(entity_id="e1", user_id="u1", timeout_seconds=60.0)
        lock.refresh()
        remaining = lock.time_remaining()
        assert 59.0 < remaining <= 60.0

    def test_time_remaining_expired(self):
        """Test time remaining when expired."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            timeout_seconds=1.0,
            last_heartbeat=time.time() - 10.0,
        )
        assert lock.time_remaining() == 0.0

    def test_upgrade_to(self):
        """Test upgrading lock type."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            lock_type=LockType.SELECTION,
        )
        lock.upgrade_to(LockType.EDITING)
        assert lock.lock_type == LockType.EDITING

    def test_upgrade_no_downgrade(self):
        """Test upgrade doesn't downgrade."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            lock_type=LockType.EDITING,
        )
        lock.upgrade_to(LockType.SELECTION)
        assert lock.lock_type == LockType.EDITING

    def test_can_override_higher_priority(self):
        """Test higher priority can override."""
        lock1 = SoftLock(entity_id="e1", user_id="u1", priority=LockPriority.HIGH)
        lock2 = SoftLock(entity_id="e1", user_id="u2", priority=LockPriority.NORMAL)
        assert lock1.can_override(lock2)
        assert not lock2.can_override(lock1)

    def test_can_override_expired(self):
        """Test expired lock can be overridden."""
        lock1 = SoftLock(entity_id="e1", user_id="u1", priority=LockPriority.NORMAL)
        lock2 = SoftLock(
            entity_id="e1",
            user_id="u2",
            priority=LockPriority.NORMAL,
            timeout_seconds=0.0,
            last_heartbeat=time.time() - 10.0,
        )
        assert lock1.can_override(lock2)

    def test_to_dict(self):
        """Test serialization to dict."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            lock_type=LockType.EDITING,
            priority=LockPriority.HIGH,
            metadata={"reason": "test"},
        )
        data = lock.to_dict()
        assert data["entity_id"] == "e1"
        assert data["user_id"] == "u1"
        assert data["lock_type"] == "EDITING"
        assert data["priority"] == "HIGH"
        assert data["metadata"] == {"reason": "test"}

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "entity_id": "e1",
            "user_id": "u1",
            "lock_type": "EDITING",
            "priority": "HIGH",
            "timeout_seconds": 120.0,
        }
        lock = SoftLock.from_dict(data)
        assert lock.entity_id == "e1"
        assert lock.user_id == "u1"
        assert lock.lock_type == LockType.EDITING
        assert lock.priority == LockPriority.HIGH
        assert lock.timeout_seconds == 120.0

    def test_to_json(self):
        """Test JSON serialization."""
        lock = SoftLock(entity_id="e1", user_id="u1")
        json_str = lock.to_json()
        assert "e1" in json_str
        assert "u1" in json_str

    def test_from_json(self):
        """Test JSON deserialization."""
        lock = SoftLock(entity_id="e1", user_id="u1")
        json_str = lock.to_json()
        restored = SoftLock.from_json(json_str)
        assert restored.entity_id == "e1"
        assert restored.user_id == "u1"

    def test_downgrade_to(self):
        """Test downgrading lock type."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            lock_type=LockType.EDITING,
        )
        lock.downgrade_to(LockType.SELECTION)
        assert lock.lock_type == LockType.SELECTION

    def test_downgrade_no_upgrade(self):
        """Test downgrade doesn't upgrade."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            lock_type=LockType.SELECTION,
        )
        lock.downgrade_to(LockType.EDITING)
        assert lock.lock_type == LockType.SELECTION

    def test_set_lock_type_upgrade(self):
        """Test set_lock_type for upgrade."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            lock_type=LockType.SELECTION,
        )
        lock.set_lock_type(LockType.EDITING)
        assert lock.lock_type == LockType.EDITING

    def test_set_lock_type_downgrade(self):
        """Test set_lock_type for downgrade."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            lock_type=LockType.EDITING,
        )
        lock.set_lock_type(LockType.SELECTION)
        assert lock.lock_type == LockType.SELECTION

    def test_set_lock_type_refreshes(self):
        """Test set_lock_type refreshes heartbeat."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            last_heartbeat=time.time() - 100.0,
        )
        old_heartbeat = lock.last_heartbeat
        lock.set_lock_type(LockType.EDITING)
        assert lock.last_heartbeat > old_heartbeat

    def test_exclusive_lock_type(self):
        """Test EXCLUSIVE lock type."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            lock_type=LockType.EXCLUSIVE,
        )
        assert lock.lock_type == LockType.EXCLUSIVE
        assert lock.lock_type.value > LockType.EDITING.value


# =============================================================================
# LockConflictNotification Tests
# =============================================================================


class TestLockConflictNotification:
    """Tests for LockConflictNotification."""

    def test_create(self):
        """Test creating notification."""
        notification = LockConflictNotification(
            entity_id="e1",
            current_holder="user1",
            requester="user2",
            lock_type=LockType.EDITING,
            holder_priority=LockPriority.NORMAL,
            requester_priority=LockPriority.NORMAL,
        )
        assert notification.entity_id == "e1"
        assert notification.current_holder == "user1"
        assert notification.requester == "user2"
        assert "user1" in notification.message

    def test_acknowledge(self):
        """Test acknowledging notification."""
        notification = LockConflictNotification(
            entity_id="e1",
            current_holder="user1",
            requester="user2",
            lock_type=LockType.EDITING,
            holder_priority=LockPriority.NORMAL,
            requester_priority=LockPriority.NORMAL,
        )
        assert not notification.acknowledged
        notification.acknowledge()
        assert notification.acknowledged

    def test_is_override_allowed_true(self):
        """Test override allowed with higher priority."""
        notification = LockConflictNotification(
            entity_id="e1",
            current_holder="user1",
            requester="user2",
            lock_type=LockType.EDITING,
            holder_priority=LockPriority.NORMAL,
            requester_priority=LockPriority.HIGH,
        )
        assert notification.is_override_allowed()

    def test_is_override_allowed_false(self):
        """Test override not allowed with same/lower priority."""
        notification = LockConflictNotification(
            entity_id="e1",
            current_holder="user1",
            requester="user2",
            lock_type=LockType.EDITING,
            holder_priority=LockPriority.HIGH,
            requester_priority=LockPriority.NORMAL,
        )
        assert not notification.is_override_allowed()

    def test_to_dict(self):
        """Test serialization."""
        notification = LockConflictNotification(
            entity_id="e1",
            current_holder="user1",
            requester="user2",
            lock_type=LockType.EDITING,
            holder_priority=LockPriority.NORMAL,
            requester_priority=LockPriority.NORMAL,
        )
        data = notification.to_dict()
        assert data["entity_id"] == "e1"
        assert data["current_holder"] == "user1"
        assert data["requester"] == "user2"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "entity_id": "e1",
            "current_holder": "user1",
            "requester": "user2",
            "lock_type": "EDITING",
            "holder_priority": "NORMAL",
            "requester_priority": "HIGH",
        }
        notification = LockConflictNotification.from_dict(data)
        assert notification.entity_id == "e1"
        assert notification.is_override_allowed()


# =============================================================================
# LockRegistry Tests
# =============================================================================


class TestLockRegistry:
    """Tests for LockRegistry."""

    def test_create(self):
        """Test creating registry."""
        registry = LockRegistry()
        assert registry.stats()["total_locks"] == 0

    def test_acquire_new_lock(self):
        """Test acquiring a new lock."""
        registry = LockRegistry()
        lock, conflict = registry.acquire("entity1", "user1")
        assert lock.entity_id == "entity1"
        assert lock.user_id == "user1"
        assert conflict is None

    def test_acquire_with_type(self):
        """Test acquiring lock with type."""
        registry = LockRegistry()
        lock, _ = registry.acquire(
            "entity1",
            "user1",
            lock_type=LockType.EDITING,
        )
        assert lock.lock_type == LockType.EDITING

    def test_acquire_with_priority(self):
        """Test acquiring lock with priority."""
        registry = LockRegistry()
        lock, _ = registry.acquire(
            "entity1",
            "user1",
            priority=LockPriority.HIGH,
        )
        assert lock.priority == LockPriority.HIGH

    def test_acquire_same_user_upgrade(self):
        """Test same user upgrades lock."""
        registry = LockRegistry()
        lock1, _ = registry.acquire("entity1", "user1", lock_type=LockType.SELECTION)
        lock2, _ = registry.acquire("entity1", "user1", lock_type=LockType.EDITING)
        assert lock2.lock_type == LockType.EDITING
        assert lock1.lock_id == lock2.lock_id  # Same lock upgraded

    def test_acquire_conflict(self):
        """Test acquiring conflicting lock."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")
        lock, conflict = registry.acquire("entity1", "user2")

        assert lock.user_id == "user1"  # Returns existing lock
        assert conflict is not None
        assert conflict.current_holder == "user1"
        assert conflict.requester == "user2"

    def test_acquire_conflict_force(self):
        """Test force acquiring conflicting lock."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")
        lock, conflict = registry.acquire("entity1", "user2", force=True)

        assert lock.user_id == "user2"  # Forced acquisition
        assert conflict is not None

    def test_acquire_expired_lock(self):
        """Test acquiring lock on expired entity."""
        registry = LockRegistry(default_timeout=0.0)
        registry.acquire("entity1", "user1")
        time.sleep(0.01)
        lock, conflict = registry.acquire("entity1", "user2")

        assert lock.user_id == "user2"  # Acquired since previous expired
        assert conflict is None

    def test_release(self):
        """Test releasing a lock."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")
        assert registry.release("entity1", "user1")
        assert not registry.is_locked("entity1")

    def test_release_not_owner(self):
        """Test release fails for non-owner."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")
        assert not registry.release("entity1", "user2")
        assert registry.is_locked("entity1")

    def test_release_nonexistent(self):
        """Test releasing non-existent lock."""
        registry = LockRegistry()
        assert not registry.release("entity1", "user1")

    def test_release_all(self):
        """Test releasing all user locks."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")
        registry.acquire("entity2", "user1")
        registry.acquire("entity3", "user2")

        count = registry.release_all("user1")
        assert count == 2
        assert not registry.is_locked("entity1")
        assert not registry.is_locked("entity2")
        assert registry.is_locked("entity3")

    def test_refresh(self):
        """Test refreshing a lock."""
        registry = LockRegistry()
        lock, _ = registry.acquire("entity1", "user1")
        old_heartbeat = lock.last_heartbeat
        time.sleep(0.01)
        assert registry.refresh("entity1", "user1")
        assert lock.last_heartbeat > old_heartbeat

    def test_refresh_not_owner(self):
        """Test refresh fails for non-owner."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")
        assert not registry.refresh("entity1", "user2")

    def test_get(self):
        """Test getting a lock."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")
        lock = registry.get("entity1")
        assert lock.user_id == "user1"

    def test_get_nonexistent(self):
        """Test getting non-existent lock."""
        registry = LockRegistry()
        assert registry.get("entity1") is None

    def test_get_expired(self):
        """Test getting expired lock returns None."""
        registry = LockRegistry(default_timeout=0.0)
        registry.acquire("entity1", "user1")
        time.sleep(0.01)
        assert registry.get("entity1") is None

    def test_get_user_locks(self):
        """Test getting all locks for a user."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")
        registry.acquire("entity2", "user1")
        registry.acquire("entity3", "user2")

        locks = registry.get_user_locks("user1")
        assert len(locks) == 2
        entity_ids = {l.entity_id for l in locks}
        assert entity_ids == {"entity1", "entity2"}

    def test_is_locked(self):
        """Test checking if entity is locked."""
        registry = LockRegistry()
        assert not registry.is_locked("entity1")
        registry.acquire("entity1", "user1")
        assert registry.is_locked("entity1")

    def test_is_locked_by(self):
        """Test checking if entity is locked by specific user."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")
        assert registry.is_locked_by("entity1", "user1")
        assert not registry.is_locked_by("entity1", "user2")

    def test_get_holder(self):
        """Test getting lock holder."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")
        assert registry.get_holder("entity1") == "user1"
        assert registry.get_holder("entity2") is None

    def test_get_all_locks(self):
        """Test getting all locks."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")
        registry.acquire("entity2", "user2")

        locks = registry.get_all_locks()
        assert len(locks) == 2
        assert "entity1" in locks
        assert "entity2" in locks

    def test_check_conflict(self):
        """Test checking for conflict without acquiring."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")

        conflict = registry.check_conflict("entity1", "user2")
        assert conflict is not None
        assert conflict.current_holder == "user1"

        no_conflict = registry.check_conflict("entity1", "user1")
        assert no_conflict is None

    def test_conflict_handler(self):
        """Test conflict notification handler."""
        registry = LockRegistry()
        notifications = []

        def handler(notification):
            notifications.append(notification)

        registry.add_conflict_handler(handler)
        registry.acquire("entity1", "user1")
        registry.acquire("entity1", "user2")

        assert len(notifications) == 1
        assert notifications[0].requester == "user2"

    def test_remove_conflict_handler(self):
        """Test removing conflict handler."""
        registry = LockRegistry()
        notifications = []

        def handler(notification):
            notifications.append(notification)

        registry.add_conflict_handler(handler)
        registry.remove_conflict_handler(handler)
        registry.acquire("entity1", "user1")
        registry.acquire("entity1", "user2")

        assert len(notifications) == 0

    def test_cleanup(self):
        """Test manual cleanup."""
        registry = LockRegistry(default_timeout=0.0)
        registry.acquire("entity1", "user1")
        registry.acquire("entity2", "user2")
        time.sleep(0.01)

        cleaned = registry.cleanup()
        assert cleaned == 2
        assert not registry.is_locked("entity1")
        assert not registry.is_locked("entity2")

    def test_stats(self):
        """Test getting statistics."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1", lock_type=LockType.EDITING)
        registry.acquire("entity2", "user2", lock_type=LockType.SELECTION)

        stats = registry.stats()
        assert stats["total_locks"] == 2
        assert stats["total_users"] == 2
        assert stats["by_type"]["EDITING"] == 1
        assert stats["by_type"]["SELECTION"] == 1

    def test_to_dict(self):
        """Test serialization."""
        registry = LockRegistry(default_timeout=120.0)
        registry.acquire("entity1", "user1")

        data = registry.to_dict()
        assert "entity1" in data["locks"]
        assert data["default_timeout"] == 120.0

    def test_from_dict(self):
        """Test deserialization."""
        registry = LockRegistry()
        registry.acquire("entity1", "user1")
        data = registry.to_dict()

        restored = LockRegistry.from_dict(data)
        assert restored.is_locked("entity1")
        assert restored.is_locked_by("entity1", "user1")

    def test_thread_safety(self):
        """Test thread-safe operations."""
        registry = LockRegistry()
        errors = []

        def worker(user_id):
            try:
                for i in range(50):
                    entity_id = f"entity{i % 10}"
                    registry.acquire(entity_id, user_id)
                    registry.refresh(entity_id, user_id)
                    registry.release(entity_id, user_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"user{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# PresenceInfo Tests
# =============================================================================


class TestPresenceInfo:
    """Tests for PresenceInfo."""

    def test_create_basic(self):
        """Test creating basic presence."""
        presence = PresenceInfo(user_id="user1")
        assert presence.user_id == "user1"
        assert presence.status == PresenceStatus.ONLINE
        assert presence.cursor_position is not None
        assert presence.selection is not None

    def test_create_with_details(self):
        """Test creating presence with details."""
        presence = PresenceInfo(
            user_id="user1",
            display_name="User One",
            color="#ff0000",
        )
        assert presence.display_name == "User One"
        assert presence.color == "#ff0000"

    def test_update_cursor(self):
        """Test updating cursor position."""
        presence = PresenceInfo(user_id="user1")
        old_activity = presence.last_activity

        time.sleep(0.01)
        new_position = CursorPosition(x=1.0, y=2.0, z=3.0)
        presence.update_cursor(new_position)

        assert presence.cursor_position.x == 1.0
        assert presence.last_activity > old_activity

    def test_update_selection(self):
        """Test updating selection."""
        presence = PresenceInfo(user_id="user1")
        new_selection = Selection(entity_ids={"e1", "e2"})
        presence.update_selection(new_selection)

        assert presence.selection.entity_ids == {"e1", "e2"}

    def test_set_status(self):
        """Test setting status."""
        presence = PresenceInfo(user_id="user1")
        presence.set_status(PresenceStatus.IDLE)
        assert presence.status == PresenceStatus.IDLE

    def test_heartbeat(self):
        """Test heartbeat update."""
        presence = PresenceInfo(user_id="user1")
        old_activity = presence.last_activity

        time.sleep(0.01)
        presence.heartbeat()

        assert presence.last_activity > old_activity

    def test_is_stale(self):
        """Test staleness check."""
        presence = PresenceInfo(user_id="user1")
        presence.last_activity = time.time() - 100.0
        assert presence.is_stale(timeout=30.0)

    def test_not_stale(self):
        """Test not stale."""
        presence = PresenceInfo(user_id="user1")
        presence.heartbeat()
        assert not presence.is_stale(timeout=30.0)

    def test_time_since_activity(self):
        """Test time since activity."""
        presence = PresenceInfo(user_id="user1")
        presence.last_activity = time.time() - 5.0
        elapsed = presence.time_since_activity()
        assert 4.9 < elapsed < 5.5

    def test_update_reactivates_idle(self):
        """Test that updates reactivate idle users."""
        presence = PresenceInfo(user_id="user1")
        presence.set_status(PresenceStatus.IDLE)
        presence.update_cursor(CursorPosition(x=1.0))
        assert presence.status == PresenceStatus.ONLINE

    def test_to_dict(self):
        """Test serialization."""
        presence = PresenceInfo(
            user_id="user1",
            display_name="User One",
            color="#ff0000",
        )
        presence.update_selection(Selection(entity_ids={"e1"}))

        data = presence.to_dict()
        assert data["user_id"] == "user1"
        assert data["display_name"] == "User One"
        assert data["color"] == "#ff0000"
        assert "e1" in data["selection"]["entity_ids"]

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "user_id": "user1",
            "display_name": "User One",
            "color": "#ff0000",
            "status": "IDLE",
            "selection": {"entity_ids": ["e1"], "component_ids": [], "property_paths": []},
        }
        presence = PresenceInfo.from_dict(data)
        assert presence.user_id == "user1"
        assert presence.status == PresenceStatus.IDLE
        assert "e1" in presence.selection.entity_ids

    def test_to_json(self):
        """Test JSON serialization."""
        presence = PresenceInfo(user_id="user1")
        json_str = presence.to_json()
        assert "user1" in json_str

    def test_from_json(self):
        """Test JSON deserialization."""
        presence = PresenceInfo(user_id="user1")
        json_str = presence.to_json()
        restored = PresenceInfo.from_json(json_str)
        assert restored.user_id == "user1"


# =============================================================================
# PresenceManager Tests
# =============================================================================


class TestPresenceManager:
    """Tests for PresenceManager."""

    def test_create(self):
        """Test creating manager."""
        manager = PresenceManager()
        assert manager.user_count() == 0

    def test_join(self):
        """Test user joining."""
        manager = PresenceManager()
        presence = manager.join("user1", display_name="User One")

        assert presence.user_id == "user1"
        assert presence.display_name == "User One"
        assert manager.user_count() == 1

    def test_join_auto_color(self):
        """Test automatic color assignment."""
        manager = PresenceManager()
        p1 = manager.join("user1")
        p2 = manager.join("user2")

        assert p1.color != p2.color

    def test_join_existing(self):
        """Test joining when already present."""
        manager = PresenceManager()
        p1 = manager.join("user1", display_name="Name1")
        p2 = manager.join("user1", display_name="Name2")

        assert p2.display_name == "Name2"
        assert manager.user_count() == 1

    def test_leave(self):
        """Test user leaving."""
        manager = PresenceManager()
        manager.join("user1")
        assert manager.leave("user1")
        assert manager.user_count() == 0

    def test_leave_nonexistent(self):
        """Test leaving when not present."""
        manager = PresenceManager()
        assert not manager.leave("user1")

    def test_update_cursor(self):
        """Test cursor update."""
        manager = PresenceManager()
        manager.join("user1")
        position = CursorPosition(x=1.0, y=2.0, z=3.0)

        assert manager.update_cursor("user1", position)
        presence = manager.get("user1")
        assert presence.cursor_position.x == 1.0

    def test_update_cursor_nonexistent(self):
        """Test cursor update for non-existent user."""
        manager = PresenceManager()
        position = CursorPosition(x=1.0)
        assert not manager.update_cursor("user1", position)

    def test_update_selection(self):
        """Test selection update."""
        manager = PresenceManager()
        manager.join("user1")
        selection = Selection(entity_ids={"e1", "e2"})

        assert manager.update_selection("user1", selection)
        presence = manager.get("user1")
        assert presence.selection.entity_ids == {"e1", "e2"}

    def test_heartbeat(self):
        """Test heartbeat."""
        manager = PresenceManager()
        manager.join("user1")
        presence = manager.get("user1")
        old_activity = presence.last_activity

        time.sleep(0.01)
        assert manager.heartbeat("user1")
        assert presence.last_activity > old_activity

    def test_set_status(self):
        """Test setting status."""
        manager = PresenceManager()
        manager.join("user1")

        assert manager.set_status("user1", PresenceStatus.AWAY)
        presence = manager.get("user1")
        assert presence.status == PresenceStatus.AWAY

    def test_get(self):
        """Test getting presence."""
        manager = PresenceManager()
        manager.join("user1")
        presence = manager.get("user1")
        assert presence is not None
        assert presence.user_id == "user1"

    def test_get_nonexistent(self):
        """Test getting non-existent user."""
        manager = PresenceManager()
        assert manager.get("user1") is None

    def test_get_all(self):
        """Test getting all presence."""
        manager = PresenceManager()
        manager.join("user1")
        manager.join("user2")

        all_presence = manager.get_all()
        assert len(all_presence) == 2
        assert "user1" in all_presence
        assert "user2" in all_presence

    def test_get_online_users(self):
        """Test getting online users."""
        manager = PresenceManager()
        manager.join("user1")
        manager.join("user2")
        manager.set_status("user2", PresenceStatus.OFFLINE)

        online = manager.get_online_users()
        assert len(online) == 1
        assert online[0].user_id == "user1"

    def test_get_active_users(self):
        """Test getting active users."""
        manager = PresenceManager()
        manager.join("user1")
        manager.join("user2")
        manager.set_status("user2", PresenceStatus.IDLE)

        active = manager.get_active_users()
        assert len(active) == 1
        assert active[0].user_id == "user1"

    def test_get_users_at_entity(self):
        """Test getting users at an entity."""
        manager = PresenceManager()
        manager.join("user1")
        manager.join("user2")
        manager.update_selection("user1", Selection(entity_ids={"e1", "e2"}))
        manager.update_selection("user2", Selection(entity_ids={"e2", "e3"}))

        at_e2 = manager.get_users_at_entity("e2")
        assert len(at_e2) == 2

        at_e1 = manager.get_users_at_entity("e1")
        assert len(at_e1) == 1
        assert at_e1[0].user_id == "user1"

    def test_get_selection_overlaps(self):
        """Test getting selection overlaps."""
        manager = PresenceManager()
        manager.join("user1")
        manager.join("user2")
        manager.join("user3")
        manager.update_selection("user1", Selection(entity_ids={"e1", "e2"}))
        manager.update_selection("user2", Selection(entity_ids={"e2", "e3"}))
        manager.update_selection("user3", Selection(entity_ids={"e4"}))

        overlaps = manager.get_selection_overlaps("user1")
        assert "user2" in overlaps
        assert overlaps["user2"] == {"e2"}
        assert "user3" not in overlaps

    def test_is_online(self):
        """Test checking if user is online."""
        manager = PresenceManager()
        manager.join("user1")

        assert manager.is_online("user1")
        assert not manager.is_online("user2")

        manager.set_status("user1", PresenceStatus.OFFLINE)
        assert not manager.is_online("user1")

    def test_user_count(self):
        """Test user count."""
        manager = PresenceManager()
        assert manager.user_count() == 0
        manager.join("user1")
        assert manager.user_count() == 1
        manager.join("user2")
        assert manager.user_count() == 2

    def test_online_count(self):
        """Test online count."""
        manager = PresenceManager()
        manager.join("user1")
        manager.join("user2")
        manager.set_status("user2", PresenceStatus.OFFLINE)

        assert manager.online_count() == 1

    def test_presence_handler(self):
        """Test presence change handler."""
        manager = PresenceManager()
        events = []

        def handler(presence, event_type):
            events.append((presence.user_id, event_type))

        manager.add_presence_handler(handler)
        manager.join("user1")
        manager.leave("user1")

        assert ("user1", "join") in events
        assert ("user1", "leave") in events

    def test_remove_presence_handler(self):
        """Test removing presence handler."""
        manager = PresenceManager()
        events = []

        def handler(presence, event_type):
            events.append((presence.user_id, event_type))

        manager.add_presence_handler(handler)
        manager.remove_presence_handler(handler)
        manager.join("user1")

        assert len(events) == 0

    def test_cleanup(self):
        """Test cleanup of offline users."""
        manager = PresenceManager(offline_timeout=0.0)
        manager.join("user1")
        manager.join("user2")

        # Make users stale
        for presence in manager._users.values():
            presence.last_activity = time.time() - 200.0

        cleaned = manager.cleanup()
        assert cleaned == 2
        assert manager.user_count() == 0

    def test_stats(self):
        """Test statistics."""
        manager = PresenceManager()
        manager.join("user1")
        manager.join("user2")
        manager.set_status("user2", PresenceStatus.IDLE)
        manager.update_selection("user1", Selection(entity_ids={"e1", "e2"}))

        stats = manager.stats()
        assert stats["total_users"] == 2
        assert stats["by_status"]["ONLINE"] == 1
        assert stats["by_status"]["IDLE"] == 1
        assert stats["total_selections"] == 2

    def test_to_dict(self):
        """Test serialization."""
        manager = PresenceManager(idle_timeout=120.0)
        manager.join("user1", display_name="User One")

        data = manager.to_dict()
        assert "user1" in data["users"]
        assert data["idle_timeout"] == 120.0

    def test_from_dict(self):
        """Test deserialization."""
        manager = PresenceManager()
        manager.join("user1", display_name="User One")
        data = manager.to_dict()

        restored = PresenceManager.from_dict(data)
        assert restored.get("user1") is not None
        assert restored.get("user1").display_name == "User One"

    def test_thread_safety(self):
        """Test thread-safe operations."""
        manager = PresenceManager()
        errors = []

        def worker(user_id):
            try:
                for _ in range(50):
                    manager.join(user_id)
                    manager.heartbeat(user_id)
                    manager.update_cursor(user_id, CursorPosition(x=1.0))
                    manager.update_selection(user_id, Selection(entity_ids={"e1"}))
                    manager.leave(user_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"user{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# CollaborativeSession Tests
# =============================================================================


class TestCollaborativeSession:
    """Tests for CollaborativeSession."""

    def test_create(self):
        """Test creating session."""
        session = CollaborativeSession()
        assert session.session_id is not None
        assert session.presence is not None
        assert session.locks is not None

    def test_create_with_id(self):
        """Test creating session with specific ID."""
        session = CollaborativeSession(session_id="test-session")
        assert session.session_id == "test-session"

    def test_join(self):
        """Test user joining session."""
        session = CollaborativeSession()
        presence = session.join("user1", display_name="User One")

        assert presence.user_id == "user1"
        assert presence.display_name == "User One"

    def test_leave(self):
        """Test user leaving session."""
        session = CollaborativeSession()
        session.join("user1")
        session.locks.acquire("entity1", "user1")

        session.leave("user1")

        assert session.presence.get("user1") is None
        assert not session.locks.is_locked("entity1")

    def test_select_entity(self):
        """Test selecting an entity."""
        session = CollaborativeSession()
        session.join("user1")

        success, conflict = session.select_entity("user1", "entity1")

        assert success
        assert conflict is None
        assert session.locks.is_locked_by("entity1", "user1")
        presence = session.presence.get("user1")
        assert "entity1" in presence.selection.entity_ids

    def test_select_entity_conflict(self):
        """Test selecting entity with conflict."""
        session = CollaborativeSession()
        session.join("user1")
        session.join("user2")
        session.select_entity("user1", "entity1")

        success, conflict = session.select_entity("user2", "entity1")

        assert success  # Still succeeds (advisory)
        assert conflict is not None
        assert conflict.current_holder == "user1"

    def test_deselect_entity(self):
        """Test deselecting an entity."""
        session = CollaborativeSession()
        session.join("user1")
        session.select_entity("user1", "entity1")

        assert session.deselect_entity("user1", "entity1")

        assert not session.locks.is_locked("entity1")
        presence = session.presence.get("user1")
        assert "entity1" not in presence.selection.entity_ids

    def test_start_editing(self):
        """Test starting to edit entity."""
        session = CollaborativeSession()
        session.join("user1")

        success, conflict = session.start_editing("user1", "entity1")

        assert success
        lock = session.locks.get("entity1")
        assert lock.lock_type == LockType.EDITING

    def test_stop_editing(self):
        """Test stopping editing entity."""
        session = CollaborativeSession()
        session.join("user1")
        session.start_editing("user1", "entity1")

        assert session.stop_editing("user1", "entity1")

        lock = session.locks.get("entity1")
        assert lock.lock_type == LockType.SELECTION

    def test_get_entity_editors(self):
        """Test getting entity editors."""
        session = CollaborativeSession()
        session.join("user1")
        session.join("user2")
        session.select_entity("user1", "entity1")
        session.select_entity("user2", "entity1")

        editors = session.get_entity_editors("entity1")
        assert len(editors) == 2

    def test_get_entity_lock_holder(self):
        """Test getting lock holder."""
        session = CollaborativeSession()
        session.join("user1")
        session.select_entity("user1", "entity1")

        holder = session.get_entity_lock_holder("entity1")
        assert holder.user_id == "user1"

    def test_heartbeat(self):
        """Test session heartbeat."""
        session = CollaborativeSession()
        session.join("user1")
        session.select_entity("user1", "entity1")

        assert session.heartbeat("user1")

    def test_get_conflicts_for_user(self):
        """Test getting conflicts for a user."""
        session = CollaborativeSession()
        session.join("user1")
        session.join("user2")
        session.select_entity("user1", "entity1")
        session.select_entity("user1", "entity2")

        # User2 selects entities in presence but doesn't get locks
        presence = session.presence.get("user2")
        presence.selection.entity_ids.add("entity1")
        presence.selection.entity_ids.add("entity2")

        conflicts = session.get_conflicts_for_user("user2")
        assert len(conflicts) == 2

    def test_stats(self):
        """Test session statistics."""
        session = CollaborativeSession()
        session.join("user1")
        session.select_entity("user1", "entity1")

        stats = session.stats()
        assert stats["session_id"] is not None
        assert "presence" in stats
        assert "locks" in stats

    def test_to_dict(self):
        """Test serialization."""
        session = CollaborativeSession(session_id="test")
        session.join("user1")

        data = session.to_dict()
        assert data["session_id"] == "test"
        assert "presence" in data
        assert "locks" in data

    def test_from_dict(self):
        """Test deserialization."""
        session = CollaborativeSession(session_id="test")
        session.join("user1", display_name="User One")
        session.select_entity("user1", "entity1")
        data = session.to_dict()

        restored = CollaborativeSession.from_dict(data)
        assert restored.session_id == "test"
        assert restored.presence.get("user1") is not None
        assert restored.locks.is_locked("entity1")


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for presence exceptions."""

    def test_presence_error(self):
        """Test PresenceError."""
        error = PresenceError("Test error", {"key": "value"})
        assert str(error) == "Test error"
        assert error.details == {"key": "value"}

    def test_lock_error(self):
        """Test LockError."""
        error = LockError("Lock failed")
        assert str(error) == "Lock failed"

    def test_lock_not_found(self):
        """Test LockNotFound."""
        error = LockNotFound("entity1", "user1")
        assert error.entity_id == "entity1"
        assert error.user_id == "user1"
        assert "entity1" in str(error)

    def test_lock_conflict(self):
        """Test LockConflict."""
        error = LockConflict("entity1", "user1", "user2")
        assert error.entity_id == "entity1"
        assert error.current_holder == "user1"
        assert error.requester == "user2"

    def test_lock_expired(self):
        """Test LockExpired."""
        error = LockExpired("entity1", "user1", time.time())
        assert error.entity_id == "entity1"
        assert error.user_id == "user1"

    def test_user_not_found(self):
        """Test UserNotFound."""
        error = UserNotFound("user1")
        assert error.user_id == "user1"
        assert "user1" in str(error)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the presence system."""

    def test_full_editing_workflow(self):
        """Test complete editing workflow."""
        session = CollaborativeSession()

        # User 1 joins and starts editing
        session.join("user1", display_name="Alice")
        session.start_editing("user1", "entity1")

        # User 2 joins and tries to edit same entity
        session.join("user2", display_name="Bob")
        success, conflict = session.start_editing("user2", "entity1")

        assert success  # Advisory, doesn't block
        assert conflict is not None
        assert conflict.current_holder == "user1"

        # User 1 finishes editing
        session.deselect_entity("user1", "entity1")

        # User 2 can now get lock
        success, conflict = session.start_editing("user2", "entity1")
        assert conflict is None
        assert session.locks.is_locked_by("entity1", "user2")

    def test_multi_user_collaboration(self):
        """Test multiple users collaborating."""
        session = CollaborativeSession()

        # Multiple users join
        for i in range(5):
            session.join(f"user{i}", display_name=f"User {i}")

        # Each user selects different entities
        for i in range(5):
            session.select_entity(f"user{i}", f"entity{i}")

        # Verify each has their own lock
        for i in range(5):
            assert session.locks.is_locked_by(f"entity{i}", f"user{i}")

        # Users leave
        for i in range(5):
            session.leave(f"user{i}")

        # All locks released
        for i in range(5):
            assert not session.locks.is_locked(f"entity{i}")

    def test_presence_and_lock_sync(self):
        """Test presence and lock synchronization."""
        session = CollaborativeSession()
        session.join("user1")

        # Select multiple entities
        for i in range(5):
            session.select_entity("user1", f"entity{i}")

        presence = session.presence.get("user1")
        assert len(presence.selection.entity_ids) == 5
        assert session.locks.stats()["total_locks"] == 5

        # Deselect all
        for i in range(5):
            session.deselect_entity("user1", f"entity{i}")

        presence = session.presence.get("user1")
        assert len(presence.selection.entity_ids) == 0
        assert session.locks.stats()["total_locks"] == 0

    def test_concurrent_operations(self):
        """Test concurrent operations."""
        session = CollaborativeSession()
        errors = []

        def worker(user_id):
            try:
                session.join(user_id)
                for i in range(20):
                    entity_id = f"entity{i % 5}"
                    session.select_entity(user_id, entity_id)
                    session.heartbeat(user_id)
                    session.deselect_entity(user_id, entity_id)
                session.leave(user_id)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, f"user{i}") for i in range(5)]
            for f in futures:
                f.result()

        assert len(errors) == 0

    def test_lock_timeout_and_cleanup(self):
        """Test lock timeout and cleanup."""
        session = CollaborativeSession(lock_timeout=0.0)
        session.join("user1")
        session.select_entity("user1", "entity1")

        # Lock should expire immediately
        time.sleep(0.01)
        session.locks.cleanup()

        assert not session.locks.is_locked("entity1")

    def test_presence_status_updates(self):
        """Test automatic status updates."""
        manager = PresenceManager(idle_timeout=0.0, offline_timeout=0.0)
        manager.join("user1")

        # Make user stale
        presence = manager.get("user1")
        presence.last_activity = time.time() - 200.0

        # Trigger cleanup
        manager._update_stale_statuses()

        assert presence.status == PresenceStatus.OFFLINE


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_selection_overlap(self):
        """Test overlap with empty selection."""
        sel1 = Selection()
        sel2 = Selection(entity_ids={"e1"})
        assert not sel1.overlaps(sel2)

    def test_both_empty_selections(self):
        """Test overlap with both empty selections."""
        sel1 = Selection()
        sel2 = Selection()
        assert not sel1.overlaps(sel2)

    def test_lock_with_zero_timeout(self):
        """Test lock with zero timeout."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            timeout_seconds=0.0,
        )
        assert lock.is_expired()

    def test_lock_metadata_persistence(self):
        """Test lock metadata survives serialization."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            metadata={"custom_field": "value", "nested": {"a": 1}},
        )
        data = lock.to_dict()
        restored = SoftLock.from_dict(data)
        assert restored.metadata["custom_field"] == "value"
        assert restored.metadata["nested"]["a"] == 1

    def test_presence_manager_color_cycling(self):
        """Test color assignment cycles through palette."""
        manager = PresenceManager()
        colors = set()
        for i in range(10):
            presence = manager.join(f"user{i}")
            colors.add(presence.color)
        # Should have multiple distinct colors (palette cycles)
        assert len(colors) >= 5

    def test_session_stats_uptime_increases(self):
        """Test session uptime increases over time."""
        session = CollaborativeSession()
        stats1 = session.stats()
        time.sleep(0.05)
        stats2 = session.stats()
        assert stats2["uptime_seconds"] > stats1["uptime_seconds"]

    def test_presence_with_metadata(self):
        """Test presence with custom metadata."""
        presence = PresenceInfo(
            user_id="user1",
            metadata={"role": "admin", "team": "engineering"},
        )
        assert presence.metadata["role"] == "admin"

    def test_lock_priority_comparison(self):
        """Test lock priority comparison."""
        assert LockPriority.CRITICAL.value > LockPriority.HIGH.value
        assert LockPriority.HIGH.value > LockPriority.NORMAL.value
        assert LockPriority.NORMAL.value > LockPriority.LOW.value

    def test_cursor_position_at_origin(self):
        """Test cursor at origin."""
        c1 = CursorPosition()
        c2 = CursorPosition()
        assert c1.distance_to(c2) == 0.0

    def test_selection_with_all_types(self):
        """Test selection with all ID types."""
        sel = Selection(
            entity_ids={"e1"},
            component_ids={"c1"},
            property_paths={"transform.position.x"},
        )
        assert not sel.is_empty()

    def test_large_selection_overlap(self):
        """Test overlap detection with large selections."""
        sel1 = Selection(entity_ids={f"e{i}" for i in range(1000)})
        sel2 = Selection(entity_ids={f"e{i}" for i in range(500, 1500)})
        assert sel1.overlaps(sel2)
        overlap = sel1.get_overlapping_entities(sel2)
        assert len(overlap) == 500

    def test_session_uptime(self):
        """Test session uptime tracking."""
        session = CollaborativeSession()
        stats = session.stats()
        assert stats["uptime_seconds"] >= 0

    def test_lock_critical_priority(self):
        """Test CRITICAL priority lock."""
        lock = SoftLock(
            entity_id="e1",
            user_id="u1",
            priority=LockPriority.CRITICAL,
        )
        assert lock.priority == LockPriority.CRITICAL

    def test_presence_away_status(self):
        """Test AWAY presence status."""
        presence = PresenceInfo(user_id="user1")
        presence.set_status(PresenceStatus.AWAY)
        assert presence.status == PresenceStatus.AWAY

    def test_cursor_negative_coordinates(self):
        """Test cursor with negative coordinates."""
        cursor = CursorPosition(x=-10.0, y=-20.0, z=-30.0)
        assert cursor.x == -10.0
        assert cursor.y == -20.0
        assert cursor.z == -30.0

    def test_selection_property_paths_overlap(self):
        """Test overlap on property paths only."""
        sel1 = Selection(property_paths={"transform.position.x"})
        sel2 = Selection(property_paths={"transform.position.x", "transform.rotation"})
        assert sel1.overlaps(sel2)
