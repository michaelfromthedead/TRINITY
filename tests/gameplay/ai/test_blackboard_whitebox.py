"""
WHITEBOX Tests for the Blackboard System.

Comprehensive internal testing of blackboard with full source access.

Tests cover:
- BlackboardEntry lifecycle and TTL expiration
- BlackboardKey namespace handling
- Observer subscription and notification
- Blackboard get/set/remove/clear operations
- BlackboardScope namespaced access
- TypedBlackboard and TypedBlackboardKey
- @blackboard decorator and registry integration
- Child blackboards and parent lookup
- Edge cases: TTL expiration, max observers, key validation

Total: 50+ tests for blackboard system internals
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pytest

from engine.gameplay.ai.blackboard import (
    Blackboard,
    BlackboardEntry,
    BlackboardKey,
    BlackboardScope,
    Observer,
    TypedBlackboard,
    TypedBlackboardKey,
    VALID_SCOPES,
    blackboard,
    clear_blackboard_registry,
    create_blackboard_from_registry,
    get_all_blackboards,
    get_blackboard_metadata,
    get_blackboards_by_scope,
)
from engine.gameplay.ai.constants import (
    BLACKBOARD_DEFAULT_NAMESPACE,
    BLACKBOARD_KEY_SEPARATOR,
    BLACKBOARD_MAX_OBSERVERS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def bb():
    """Create a fresh blackboard for testing."""
    return Blackboard(name="test", enable_event_logging=False)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear blackboard registry before and after each test."""
    clear_blackboard_registry()
    yield
    clear_blackboard_registry()


# =============================================================================
# BLACKBOARD ENTRY TESTS
# =============================================================================


class TestBlackboardEntryInternals:
    """Whitebox tests for BlackboardEntry lifecycle."""

    def test_entry_default_values(self):
        """Test BlackboardEntry default initialization."""
        entry = BlackboardEntry(value=42)

        assert entry.value == 42
        assert entry.ttl is None
        assert entry.metadata == {}
        assert entry.timestamp > 0

    def test_entry_with_ttl(self):
        """Test BlackboardEntry with TTL."""
        entry = BlackboardEntry(value="data", ttl=5.0)

        assert entry.ttl == 5.0
        assert not entry.is_expired()

    def test_entry_not_expired_no_ttl(self):
        """Test entry without TTL never expires."""
        entry = BlackboardEntry(value=1, ttl=None)

        assert not entry.is_expired()
        assert not entry.is_expired(time.time() + 1000000)

    def test_entry_expires_after_ttl(self):
        """Test entry expires after TTL."""
        entry = BlackboardEntry(value=1, ttl=1.0)

        # Just created - not expired
        assert not entry.is_expired()

        # Check with future time
        future_time = entry.timestamp + 2.0
        assert entry.is_expired(future_time)

    def test_entry_with_metadata(self):
        """Test BlackboardEntry with metadata."""
        entry = BlackboardEntry(
            value="test",
            metadata={"source": "sensor", "confidence": 0.9}
        )

        assert entry.metadata["source"] == "sensor"
        assert entry.metadata["confidence"] == 0.9


# =============================================================================
# BLACKBOARD KEY TESTS
# =============================================================================


class TestBlackboardKeyInternals:
    """Whitebox tests for BlackboardKey namespace handling."""

    def test_key_default_namespace(self):
        """Test BlackboardKey uses default namespace."""
        key = BlackboardKey(name="test")

        assert key.namespace == BLACKBOARD_DEFAULT_NAMESPACE
        assert key.name == "test"

    def test_key_custom_namespace(self):
        """Test BlackboardKey with custom namespace."""
        key = BlackboardKey(name="target", namespace="combat")

        assert key.namespace == "combat"

    def test_key_full_key_format(self):
        """Test full_key property formatting."""
        key = BlackboardKey(name="position", namespace="navigation")

        expected = f"navigation{BLACKBOARD_KEY_SEPARATOR}position"
        assert key.full_key == expected

    def test_key_from_string_with_namespace(self):
        """Test parsing key with namespace from string."""
        key = BlackboardKey.from_string("combat.target")

        assert key.namespace == "combat"
        assert key.name == "target"

    def test_key_from_string_without_namespace(self):
        """Test parsing key without namespace uses default."""
        key = BlackboardKey.from_string("simple_key")

        assert key.namespace == BLACKBOARD_DEFAULT_NAMESPACE
        assert key.name == "simple_key"

    def test_key_empty_name_raises(self):
        """Test empty key name raises error."""
        with pytest.raises(ValueError):
            BlackboardKey(name="")

    def test_key_equality(self):
        """Test key equality based on full_key."""
        a = BlackboardKey(name="test", namespace="ns")
        b = BlackboardKey(name="test", namespace="ns")
        c = BlackboardKey(name="test", namespace="other")

        assert a == b
        assert a != c

    def test_key_hash(self):
        """Test key is hashable."""
        key = BlackboardKey(name="test")
        key_set = {key}

        assert key in key_set


# =============================================================================
# OBSERVER TESTS
# =============================================================================


class TestObserverInternals:
    """Whitebox tests for Observer subscription mechanics."""

    def test_observer_matches_all_keys(self):
        """Test observer without pattern matches all keys."""
        observer = Observer(callback=lambda k, o, n: None)
        key = BlackboardKey(name="any_key")

        assert observer.matches(key)

    def test_observer_matches_exact_key(self):
        """Test observer matches exact key pattern."""
        observer = Observer(
            callback=lambda k, o, n: None,
            key_pattern="target"
        )

        matches = BlackboardKey(name="target")
        no_match = BlackboardKey(name="other")

        assert observer.matches(matches)
        assert not observer.matches(no_match)

    def test_observer_matches_wildcard_pattern(self):
        """Test observer matches wildcard pattern."""
        observer = Observer(
            callback=lambda k, o, n: None,
            key_pattern="target_*"
        )

        matches = BlackboardKey(name="target_position")
        no_match = BlackboardKey(name="other_key")

        assert observer.matches(matches)
        assert not observer.matches(no_match)

    def test_observer_matches_namespace(self):
        """Test observer matches specific namespace."""
        observer = Observer(
            callback=lambda k, o, n: None,
            namespace="combat"
        )

        matches = BlackboardKey(name="target", namespace="combat")
        no_match = BlackboardKey(name="target", namespace="navigation")

        assert observer.matches(matches)
        assert not observer.matches(no_match)

    def test_observer_once_triggers_once(self):
        """Test once observer only triggers once."""
        calls = []
        observer = Observer(
            callback=lambda k, o, n: calls.append(1),
            once=True
        )

        key = BlackboardKey(name="test")

        observer.notify(key, None, 1)
        assert len(calls) == 1

        # Second notification should not match
        assert not observer.matches(key)

    def test_observer_notify_calls_callback(self):
        """Test observer notify calls callback with correct args."""
        received = []

        def callback(key, old, new):
            received.append((key, old, new))

        observer = Observer(callback=callback)
        key = BlackboardKey(name="test")

        observer.notify(key, "old_value", "new_value")

        assert len(received) == 1
        assert received[0][1] == "old_value"
        assert received[0][2] == "new_value"


# =============================================================================
# BLACKBOARD OPERATIONS TESTS
# =============================================================================


class TestBlackboardOperations:
    """Whitebox tests for Blackboard get/set/remove/clear."""

    def test_set_creates_entry(self, bb):
        """Test set creates entry in data dict."""
        bb.set("key", "value")

        key = BlackboardKey.from_string("key")
        assert key.full_key in bb._data

    def test_set_with_ttl(self, bb):
        """Test set with TTL creates expiring entry."""
        bb.set("temp", "data", ttl=5.0)

        entry = bb.get_entry("temp")
        assert entry.ttl == 5.0

    def test_set_with_metadata(self, bb):
        """Test set with metadata stores metadata."""
        bb.set("key", "value", metadata={"source": "test"})

        entry = bb.get_entry("key")
        assert entry.metadata["source"] == "test"

    def test_get_returns_value(self, bb):
        """Test get returns stored value."""
        bb.set("key", 42)
        assert bb.get("key") == 42

    def test_get_default_for_missing(self, bb):
        """Test get returns default for missing key."""
        result = bb.get("missing", default="default_value")
        assert result == "default_value"

    def test_get_removes_expired(self, bb):
        """Test get removes expired entries."""
        # Create expired entry manually
        key = BlackboardKey.from_string("expired_key")
        bb._data[key.full_key] = BlackboardEntry(
            value="old",
            ttl=0.0,
            timestamp=time.time() - 1.0
        )

        result = bb.get("expired_key", default="gone")
        assert result == "gone"
        assert key.full_key not in bb._data

    def test_has_returns_true_for_existing(self, bb):
        """Test has returns True for existing key."""
        bb.set("exists", "value")
        assert bb.has("exists")

    def test_has_returns_false_for_missing(self, bb):
        """Test has returns False for missing key."""
        assert not bb.has("missing")

    def test_has_removes_expired(self, bb):
        """Test has removes expired entries."""
        key = BlackboardKey.from_string("expired_key")
        bb._data[key.full_key] = BlackboardEntry(
            value="old",
            ttl=0.0,
            timestamp=time.time() - 1.0
        )

        assert not bb.has("expired_key")

    def test_remove_deletes_key(self, bb):
        """Test remove deletes key from data."""
        bb.set("key", "value")
        result = bb.remove("key")

        assert result is True
        assert not bb.has("key")

    def test_remove_missing_returns_false(self, bb):
        """Test remove returns False for missing key."""
        result = bb.remove("missing")
        assert result is False

    def test_clear_removes_all(self, bb):
        """Test clear removes all keys."""
        bb.set("a", 1)
        bb.set("b", 2)
        bb.set("c", 3)

        bb.clear()
        assert len(bb) == 0

    def test_clear_namespace_only(self, bb):
        """Test clear with namespace only removes that namespace."""
        bb.set(BlackboardKey(name="a", namespace="ns1"), 1)
        bb.set(BlackboardKey(name="b", namespace="ns1"), 2)
        bb.set(BlackboardKey(name="c", namespace="ns2"), 3)

        bb.clear(namespace="ns1")

        assert not bb.has(BlackboardKey(name="a", namespace="ns1"))
        assert bb.has(BlackboardKey(name="c", namespace="ns2"))


# =============================================================================
# BLACKBOARD KEY ENUMERATION TESTS
# =============================================================================


class TestBlackboardKeyEnumeration:
    """Tests for blackboard key and namespace enumeration."""

    def test_keys_returns_all_keys(self, bb):
        """Test keys returns all non-expired keys."""
        bb.set("a", 1)
        bb.set("b", 2)
        bb.set("c", 3)

        keys = bb.keys()
        assert len(keys) == 3

    def test_keys_filters_by_namespace(self, bb):
        """Test keys filters by namespace."""
        bb.set(BlackboardKey(name="a", namespace="ns1"), 1)
        bb.set(BlackboardKey(name="b", namespace="ns2"), 2)

        ns1_keys = bb.keys(namespace="ns1")
        assert len(ns1_keys) == 1
        assert ns1_keys[0].namespace == "ns1"

    def test_namespaces_returns_all_namespaces(self, bb):
        """Test namespaces returns all unique namespaces."""
        bb.set(BlackboardKey(name="a", namespace="ns1"), 1)
        bb.set(BlackboardKey(name="b", namespace="ns2"), 2)
        bb.set(BlackboardKey(name="c", namespace="ns1"), 3)

        namespaces = bb.namespaces()
        assert len(namespaces) == 2
        assert "ns1" in namespaces
        assert "ns2" in namespaces


# =============================================================================
# OBSERVER NOTIFICATION TESTS
# =============================================================================


class TestBlackboardObserverNotifications:
    """Tests for blackboard observer notifications."""

    def test_add_observer(self, bb):
        """Test adding an observer."""
        observer = bb.add_observer(lambda k, o, n: None)

        assert observer in bb._observers

    def test_remove_observer(self, bb):
        """Test removing an observer."""
        observer = bb.add_observer(lambda k, o, n: None)
        result = bb.remove_observer(observer)

        assert result is True
        assert observer not in bb._observers

    def test_remove_missing_observer_returns_false(self, bb):
        """Test removing missing observer returns False."""
        observer = Observer(callback=lambda k, o, n: None)
        result = bb.remove_observer(observer)
        assert result is False

    def test_observer_notified_on_change(self, bb):
        """Test observer is notified on value change."""
        notifications = []

        bb.add_observer(lambda k, o, n: notifications.append((o, n)))
        bb.set("key", "initial")
        bb.set("key", "changed")

        # Two notifications: None->initial, initial->changed
        assert len(notifications) == 2
        assert notifications[1] == ("initial", "changed")

    def test_observer_not_notified_same_value(self, bb):
        """Test observer not notified when value unchanged."""
        notifications = []

        bb.add_observer(lambda k, o, n: notifications.append(1))
        bb.set("key", "value")
        bb.set("key", "value")  # Same value

        # Only one notification for the initial set
        assert len(notifications) == 1

    def test_max_observers_raises(self, bb):
        """Test exceeding max observers raises error."""
        for _ in range(BLACKBOARD_MAX_OBSERVERS):
            bb.add_observer(lambda k, o, n: None)

        with pytest.raises(RuntimeError, match="Maximum number of observers"):
            bb.add_observer(lambda k, o, n: None)

    def test_once_observer_removed_after_trigger(self, bb):
        """Test once observer is removed after triggering."""
        bb.add_observer(lambda k, o, n: None, once=True)

        bb.set("key", "value1")
        bb.set("key", "value2")  # Triggers removal

        # Once observer should be removed from list
        once_observers = [o for o in bb._observers if o.once and o._triggered]
        assert len(once_observers) == 0


# =============================================================================
# BLACKBOARD SCOPE TESTS
# =============================================================================


class TestBlackboardScopeInternals:
    """Whitebox tests for BlackboardScope namespaced access."""

    def test_scope_creation(self, bb):
        """Test creating a scope."""
        scope = bb.create_scope("combat")

        assert scope._blackboard is bb
        assert scope._namespace == "combat"

    def test_scope_set_uses_namespace(self, bb):
        """Test scope set uses its namespace."""
        scope = bb.create_scope("combat")
        scope.set("target", "enemy")

        # Verify it's stored with namespace
        key = BlackboardKey(name="target", namespace="combat")
        assert bb.has(key)

    def test_scope_get_uses_namespace(self, bb):
        """Test scope get uses its namespace."""
        scope = bb.create_scope("combat")
        scope.set("target", "enemy")

        assert scope.get("target") == "enemy"

    def test_scope_has_uses_namespace(self, bb):
        """Test scope has uses its namespace."""
        scope = bb.create_scope("combat")
        scope.set("target", "enemy")

        assert scope.has("target")
        assert not scope.has("other_key")

    def test_scope_remove_uses_namespace(self, bb):
        """Test scope remove uses its namespace."""
        scope = bb.create_scope("combat")
        scope.set("target", "enemy")

        result = scope.remove("target")
        assert result is True
        assert not scope.has("target")

    def test_scope_clear_clears_namespace_only(self, bb):
        """Test scope clear only clears its namespace."""
        scope1 = bb.create_scope("ns1")
        scope2 = bb.create_scope("ns2")

        scope1.set("a", 1)
        scope2.set("b", 2)

        scope1.clear()

        assert not scope1.has("a")
        assert scope2.has("b")

    def test_scope_keys_returns_namespace_keys(self, bb):
        """Test scope keys returns only namespace keys."""
        scope = bb.create_scope("ns")
        scope.set("a", 1)
        scope.set("b", 2)

        bb.set(BlackboardKey(name="c", namespace="other"), 3)

        keys = scope.keys()
        assert len(keys) == 2

    def test_scope_len(self, bb):
        """Test scope len returns namespace entry count."""
        scope = bb.create_scope("ns")
        scope.set("a", 1)
        scope.set("b", 2)

        assert len(scope) == 2

    def test_scope_add_observer(self, bb):
        """Test scope add_observer uses namespace filter."""
        scope = bb.create_scope("combat")
        notifications = []

        scope.add_observer(lambda k, o, n: notifications.append(k.name))

        scope.set("target", "enemy")
        bb.set(BlackboardKey(name="other", namespace="navigation"), "value")

        # Only combat namespace notification
        assert len(notifications) == 1
        assert "target" in notifications


# =============================================================================
# CHILD BLACKBOARD TESTS
# =============================================================================


class TestChildBlackboardInternals:
    """Tests for child blackboard parent lookup."""

    def test_create_child_sets_parent(self, bb):
        """Test create_child sets parent reference."""
        child = bb.create_child("child_bb")

        assert child._parent is bb

    def test_child_get_checks_parent(self, bb):
        """Test child get checks parent for missing keys."""
        bb.set("parent_key", "parent_value")
        child = bb.create_child("child")

        assert child.get("parent_key") == "parent_value"

    def test_child_local_overrides_parent(self, bb):
        """Test child local value overrides parent."""
        bb.set("key", "parent_value")
        child = bb.create_child("child")
        child.set("key", "child_value")

        assert child.get("key") == "child_value"

    def test_child_has_checks_parent(self, bb):
        """Test child has checks parent."""
        bb.set("parent_key", "value")
        child = bb.create_child("child")

        assert child.has("parent_key", check_parent=True)
        assert not child.has("parent_key", check_parent=False)

    def test_get_without_parent_check(self, bb):
        """Test get with check_parent=False."""
        bb.set("parent_key", "value")
        child = bb.create_child("child")

        result = child.get("parent_key", check_parent=False, default="not_found")
        assert result == "not_found"


# =============================================================================
# TYPED BLACKBOARD TESTS
# =============================================================================


class TestTypedBlackboardInternals:
    """Tests for TypedBlackboard and TypedBlackboardKey."""

    def test_typed_key_creation(self):
        """Test TypedBlackboardKey creation."""
        key = TypedBlackboardKey[int]("count", "stats")

        assert key._key.name == "count"
        assert key._key.namespace == "stats"

    def test_typed_bb_set_typed(self):
        """Test TypedBlackboard set_typed."""
        bb = TypedBlackboard(name="test", enable_event_logging=False)
        key = TypedBlackboardKey[str]("name")

        bb.set_typed(key, "test_value")
        assert bb.get_typed(key) == "test_value"

    def test_typed_bb_get_typed_default(self):
        """Test TypedBlackboard get_typed with default."""
        bb = TypedBlackboard(name="test", enable_event_logging=False)
        key = TypedBlackboardKey[int]("missing")

        result = bb.get_typed(key, default=42)
        assert result == 42


# =============================================================================
# CLEANUP TESTS
# =============================================================================


class TestBlackboardCleanup:
    """Tests for blackboard cleanup operations."""

    def test_cleanup_expired_removes_entries(self, bb):
        """Test cleanup_expired removes expired entries."""
        # Create expired entry
        key = BlackboardKey.from_string("expired")
        bb._data[key.full_key] = BlackboardEntry(
            value="old",
            ttl=0.0,
            timestamp=time.time() - 1.0
        )

        bb.set("valid", "value")

        removed = bb.cleanup_expired()

        assert removed == 1
        assert not bb.has("expired")
        assert bb.has("valid")

    def test_cleanup_expired_returns_zero_when_none(self, bb):
        """Test cleanup_expired returns 0 when no expired entries."""
        bb.set("valid1", 1)
        bb.set("valid2", 2)

        removed = bb.cleanup_expired()
        assert removed == 0


# =============================================================================
# DECORATOR TESTS
# =============================================================================


class TestBlackboardDecorator:
    """Tests for @blackboard decorator."""

    def test_decorator_marks_class(self):
        """Test @blackboard decorator marks class."""
        @blackboard(name="combat", scope="entity")
        class CombatBlackboard(Blackboard):
            pass

        assert hasattr(CombatBlackboard, "_blackboard_registered")
        assert CombatBlackboard._blackboard_registered is True
        assert CombatBlackboard._blackboard_name == "combat"
        assert CombatBlackboard._blackboard_scope == "entity"

    def test_decorator_invalid_scope_raises(self):
        """Test @blackboard raises for invalid scope."""
        with pytest.raises(ValueError, match="Invalid blackboard scope"):
            @blackboard(name="bad", scope="invalid_scope")
            class BadBlackboard(Blackboard):
                pass

    def test_valid_scopes(self):
        """Test all valid scopes are accepted."""
        for scope in VALID_SCOPES:
            @blackboard(name=f"test_{scope}", scope=scope)
            class DynamicBlackboard(Blackboard):
                pass

            assert DynamicBlackboard._blackboard_scope == scope

    def test_decorator_with_key_types(self):
        """Test @blackboard with key_types."""
        @blackboard(name="typed", scope="entity", key_types=["target", "health"])
        class TypedBB(Blackboard):
            pass

        assert "target" in TypedBB._blackboard_key_types
        assert "health" in TypedBB._blackboard_key_types


# =============================================================================
# REGISTRY TESTS
# =============================================================================


class TestBlackboardRegistry:
    """Tests for blackboard registry functions."""

    def test_from_registry_creates_instance(self):
        """Test Blackboard.from_registry creates instance."""
        @blackboard(name="registered", scope="entity")
        class RegisteredBB(Blackboard):
            pass

        instance = Blackboard.from_registry("registered")
        assert isinstance(instance, RegisteredBB)

    def test_from_registry_missing_raises(self):
        """Test from_registry raises for missing type."""
        with pytest.raises(ValueError, match="not found"):
            Blackboard.from_registry("nonexistent")

    def test_create_blackboard_from_registry(self):
        """Test create_blackboard_from_registry helper."""
        @blackboard(name="helper_test", scope="shared")
        class HelperBB(Blackboard):
            pass

        instance = create_blackboard_from_registry("helper_test")
        assert isinstance(instance, HelperBB)


# =============================================================================
# CONTAINER PROTOCOL TESTS
# =============================================================================


class TestBlackboardContainerProtocol:
    """Tests for blackboard container protocol support."""

    def test_len(self, bb):
        """Test __len__ returns entry count."""
        bb.set("a", 1)
        bb.set("b", 2)

        assert len(bb) == 2

    def test_contains(self, bb):
        """Test __contains__ checks key existence."""
        bb.set("exists", "value")

        assert "exists" in bb
        assert "missing" not in bb

    def test_iter(self, bb):
        """Test __iter__ iterates over keys."""
        bb.set("a", 1)
        bb.set("b", 2)

        keys = list(bb)
        assert len(keys) == 2


# =============================================================================
# EDGE CASES
# =============================================================================


class TestBlackboardEdgeCases:
    """Edge case tests for blackboard system."""

    def test_empty_value(self, bb):
        """Test storing empty/None values."""
        bb.set("empty_string", "")
        bb.set("none_value", None)
        bb.set("empty_list", [])

        assert bb.get("empty_string") == ""
        assert bb.get("none_value") is None
        assert bb.get("empty_list") == []

    def test_complex_values(self, bb):
        """Test storing complex values."""
        bb.set("dict", {"nested": {"deep": "value"}})
        bb.set("list", [1, 2, [3, 4]])

        assert bb.get("dict")["nested"]["deep"] == "value"
        assert bb.get("list")[2][0] == 3

    def test_callable_value(self, bb):
        """Test storing callable values."""
        def my_func():
            return 42

        bb.set("func", my_func)
        assert bb.get("func")() == 42

    def test_very_long_key_name(self, bb):
        """Test very long key names."""
        long_name = "a" * 1000
        bb.set(long_name, "value")

        assert bb.get(long_name) == "value"

    def test_unicode_values(self, bb):
        """Test unicode values."""
        bb.set("unicode", "Hello, World!")

        assert bb.get("unicode") == "Hello, World!"

    def test_zero_ttl_expires_immediately(self, bb):
        """Test zero TTL expires immediately on next access."""
        bb.set("instant", "value", ttl=0.0)

        # Entry exists but will be expired on access
        time.sleep(0.001)  # Tiny delay
        assert bb.get("instant", default="gone") == "gone"

    def test_negative_ttl_treated_as_expired(self, bb):
        """Test negative TTL is treated as expired."""
        key = BlackboardKey.from_string("negative_ttl")
        bb._data[key.full_key] = BlackboardEntry(
            value="old",
            ttl=-1.0,
            timestamp=time.time()
        )

        assert bb.get("negative_ttl", default="gone") == "gone"

    def test_get_entry_returns_full_entry(self, bb):
        """Test get_entry returns full BlackboardEntry."""
        bb.set("key", "value", ttl=10.0, metadata={"source": "test"})

        entry = bb.get_entry("key")
        assert entry is not None
        assert entry.value == "value"
        assert entry.ttl == 10.0
        assert entry.metadata["source"] == "test"

    def test_get_entry_missing_returns_none(self, bb):
        """Test get_entry returns None for missing key."""
        entry = bb.get_entry("missing")
        assert entry is None
