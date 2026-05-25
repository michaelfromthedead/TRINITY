"""
Comprehensive unit tests for the Registry system.
Tests registration, lookup, queries, metadata, instance tracking, and thread safety.
"""
import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.registry import registry, Registry


# Fresh registry for each test
@pytest.fixture(autouse=True)
def clear_registry():
    """Store original state and restore after each test."""
    # Store original state - use the actual attribute names from the implementation
    original_types = dict(registry._types)
    original_names = dict(registry._names)
    original_metadata = dict(registry._metadata)
    original_instances = dict(registry._instances)
    yield
    # Restore
    registry._types.clear()
    registry._types.update(original_types)
    registry._names.clear()
    registry._names.update(original_names)
    registry._metadata.clear()
    registry._metadata.update(original_metadata)
    registry._instances.clear()
    registry._instances.update(original_instances)


class TestRegistration:
    """Tests for type registration functionality."""

    def test_register_class(self):
        """Verify a class can be registered and is_registered returns True."""
        class Player: pass
        registry.register(Player)
        assert registry.is_registered(Player)

    def test_register_with_custom_name(self):
        """Verify registration with a custom name works correctly."""
        class Enemy: pass
        registry.register(Enemy, name="game.Enemy")
        assert registry.get("game.Enemy") is Enemy

    def test_unregister(self):
        """Verify a class can be unregistered."""
        class Temp: pass
        registry.register(Temp)
        assert registry.is_registered(Temp)
        registry.unregister(Temp)
        assert not registry.is_registered(Temp)

    def test_register_non_class_raises_error(self):
        """Verify registering a non-class raises TypeError."""
        with pytest.raises(TypeError):
            registry.register("not a class")

    def test_register_duplicate_name_raises_error(self):
        """Verify registering duplicate names raises ValueError."""
        class First: pass
        class Second: pass
        registry.register(First, name="DuplicateName")
        with pytest.raises(ValueError):
            registry.register(Second, name="DuplicateName")

    def test_register_same_class_twice_is_idempotent(self):
        """Verify registering the same class twice does not raise an error."""
        class Singleton: pass
        registry.register(Singleton, name="Singleton")
        # Should not raise
        registry.register(Singleton, name="Singleton")
        assert registry.is_registered(Singleton)


class TestLookup:
    """Tests for type lookup functionality."""

    def test_get_by_name(self):
        """Verify getting a type by its registered name."""
        class Item: pass
        registry.register(Item, name="Item")
        assert registry.get("Item") is Item

    def test_get_name(self):
        """Verify getting the registered name for a type."""
        class Weapon: pass
        registry.register(Weapon, name="Weapon")
        assert registry.get_name(Weapon) == "Weapon"

    def test_get_unknown_returns_none(self):
        """Verify getting an unknown name returns None."""
        assert registry.get("NonExistent") is None

    def test_get_name_unknown_returns_none(self):
        """Verify getting name for unregistered type returns None."""
        class NotRegistered: pass
        assert registry.get_name(NotRegistered) is None

    def test_all_types(self):
        """Verify all_types returns all registered types."""
        class A: pass
        class B: pass
        registry.register(A, name="A")
        registry.register(B, name="B")
        all_types = registry.all_types()
        assert A in all_types
        assert B in all_types


class TestQueries:
    """Tests for type query functionality."""

    def test_subclasses(self):
        """Verify subclasses returns all registered subclasses."""
        class Entity: pass
        class Player(Entity): pass
        class Enemy(Entity): pass

        registry.register(Entity, name="Entity")
        registry.register(Player, name="Player")
        registry.register(Enemy, name="Enemy")

        subs = registry.subclasses(Entity)
        assert Player in subs
        assert Enemy in subs
        # Base class should not be in subclasses
        assert Entity not in subs

    def test_subclasses_excludes_unrelated(self):
        """Verify subclasses excludes unrelated types."""
        class Base: pass
        class Child(Base): pass
        class Unrelated: pass

        registry.register(Base, name="Base")
        registry.register(Child, name="Child")
        registry.register(Unrelated, name="Unrelated")

        subs = registry.subclasses(Base)
        assert Child in subs
        assert Unrelated not in subs

    def test_types_where(self):
        """Verify types_where filters by predicate."""
        class Small: size = 1
        class Large: size = 100

        registry.register(Small, name="Small")
        registry.register(Large, name="Large")

        large_types = registry.types_where(lambda t: getattr(t, 'size', 0) > 50)
        assert Large in large_types
        assert Small not in large_types

    def test_types_where_with_no_matches(self):
        """Verify types_where returns empty list when no matches."""
        class NoMatch: value = 5
        registry.register(NoMatch, name="NoMatch")

        result = registry.types_where(lambda t: getattr(t, 'value', 0) > 100)
        assert result == []

    def test_types_with_decorator(self):
        """Verify types_with_decorator finds decorated types."""
        class Decorated:
            _applied_decorators = {"special"}
        class NotDecorated: pass

        registry.register(Decorated, name="Decorated")
        registry.register(NotDecorated, name="NotDecorated")

        result = registry.types_with_decorator("special")
        assert Decorated in result
        assert NotDecorated not in result


class TestMetadata:
    """Tests for metadata functionality."""

    def test_set_and_get_metadata(self):
        """Verify setting and getting metadata."""
        class Config: pass
        registry.register(Config, name="Config")
        registry.set_metadata(Config, "version", "1.0")
        assert registry.get_metadata(Config, "version") == "1.0"

    def test_get_all_metadata(self):
        """Verify getting all metadata for a type."""
        class Settings: pass
        registry.register(Settings, name="Settings")
        registry.set_metadata(Settings, "a", 1)
        registry.set_metadata(Settings, "b", 2)
        meta = registry.get_all_metadata(Settings)
        assert meta.get("a") == 1
        assert meta.get("b") == 2

    def test_get_metadata_unknown_key_returns_none(self):
        """Verify getting unknown metadata key returns None."""
        class HasMeta: pass
        registry.register(HasMeta, name="HasMeta")
        assert registry.get_metadata(HasMeta, "unknown") is None

    def test_get_metadata_unregistered_type_returns_none(self):
        """Verify getting metadata for unregistered type returns None."""
        class NotRegistered: pass
        assert registry.get_metadata(NotRegistered, "any") is None

    def test_set_metadata_unregistered_raises_error(self):
        """Verify setting metadata on unregistered type raises ValueError."""
        class NotRegistered: pass
        with pytest.raises(ValueError):
            registry.set_metadata(NotRegistered, "key", "value")

    def test_get_all_metadata_unregistered_returns_empty(self):
        """Verify get_all_metadata for unregistered type returns empty dict."""
        class NotRegistered: pass
        assert registry.get_all_metadata(NotRegistered) == {}


class TestInstanceTracking:
    """Tests for instance tracking functionality."""

    def test_instance_tracking_disabled_by_default(self):
        """Verify instances are not tracked by default."""
        class NoTrack: pass
        registry.register(NoTrack, name="NoTrack")
        obj = NoTrack()
        # Should not track by default
        count = registry.instance_count(NoTrack)
        assert count == 0

    def test_instance_tracking_when_enabled(self):
        """Verify instances are tracked when enabled."""
        class Tracked: pass
        registry.register(Tracked, name="Tracked", track_instances=True)

        obj1 = Tracked()
        obj2 = Tracked()

        assert registry.instance_count(Tracked) == 2

        instances = list(registry.instances(Tracked))
        assert len(instances) == 2
        assert obj1 in instances
        assert obj2 in instances

    def test_instance_tracking_uses_weak_references(self):
        """Verify tracked instances use weak references (can be garbage collected)."""
        class WeakTracked: pass
        registry.register(WeakTracked, name="WeakTracked", track_instances=True)

        obj = WeakTracked()
        assert registry.instance_count(WeakTracked) == 1

        # Delete the reference
        del obj

        # Force garbage collection
        import gc
        gc.collect()

        # Instance should be gone
        assert registry.instance_count(WeakTracked) == 0

    def test_instances_untracked_type_returns_empty_iterator(self):
        """Verify instances() returns empty iterator for untracked types."""
        class Untracked: pass
        registry.register(Untracked, name="Untracked")
        instances = list(registry.instances(Untracked))
        assert instances == []

    def test_instance_count_unregistered_type_returns_zero(self):
        """Verify instance_count returns 0 for unregistered type."""
        class NotRegistered: pass
        assert registry.instance_count(NotRegistered) == 0


class TestDescribe:
    """Tests for the describe functionality."""

    def test_describe_registered_type(self):
        """Verify describe returns information for registered type."""
        class Describable: pass
        registry.register(Describable, name="Describable")
        description = registry.describe(Describable)
        assert "Registry: Describable" in description

    def test_describe_unregistered_type(self):
        """Verify describe returns appropriate message for unregistered type."""
        class NotRegistered: pass
        description = registry.describe(NotRegistered)
        assert "unregistered" in description.lower()

    def test_describe_with_metadata(self):
        """Verify describe includes metadata when present."""
        class WithMeta: pass
        registry.register(WithMeta, name="WithMeta")
        registry.set_metadata(WithMeta, "author", "tester")
        description = registry.describe(WithMeta)
        assert "metadata" in description.lower()

    def test_describe_with_tracked_instances(self):
        """Verify describe includes instance count for tracked types."""
        class TrackedDesc: pass
        registry.register(TrackedDesc, name="TrackedDesc", track_instances=True)
        obj = TrackedDesc()
        description = registry.describe(TrackedDesc)
        assert "instances" in description.lower()


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_registration(self):
        """Verify concurrent registration is thread-safe."""
        import threading

        classes = [type(f"ConcurrentClass{i}", (), {}) for i in range(10)]

        def register_class(cls, name):
            registry.register(cls, name=name)

        threads = [
            threading.Thread(target=register_class, args=(cls, f"ConcurrentClass{i}"))
            for i, cls in enumerate(classes)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be registered
        for i, cls in enumerate(classes):
            assert registry.is_registered(cls)

    def test_concurrent_lookup(self):
        """Verify concurrent lookup is thread-safe."""
        import threading

        class LookupTarget: pass
        registry.register(LookupTarget, name="LookupTarget")

        results = []

        def lookup():
            for _ in range(100):
                result = registry.get("LookupTarget")
                results.append(result)

        threads = [threading.Thread(target=lookup) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All lookups should succeed
        assert len(results) == 500
        assert all(r is LookupTarget for r in results)

    def test_concurrent_metadata_access(self):
        """Verify concurrent metadata access is thread-safe."""
        import threading

        class MetaTarget: pass
        registry.register(MetaTarget, name="MetaTarget")

        def write_metadata(key, value):
            for _ in range(50):
                registry.set_metadata(MetaTarget, key, value)

        def read_metadata(key):
            for _ in range(50):
                registry.get_metadata(MetaTarget, key)

        threads = [
            threading.Thread(target=write_metadata, args=(f"key{i}", f"value{i}"))
            for i in range(3)
        ]
        threads.extend([
            threading.Thread(target=read_metadata, args=(f"key{i}",))
            for i in range(3)
        ])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors


class TestNewRegistryInstance:
    """Tests for creating new Registry instances."""

    def test_new_registry_is_empty(self):
        """Verify a new Registry instance is empty."""
        new_reg = Registry()
        assert new_reg.all_types() == []

    def test_separate_registries_are_independent(self):
        """Verify separate Registry instances are independent."""
        reg1 = Registry()
        reg2 = Registry()

        class InReg1: pass
        class InReg2: pass

        reg1.register(InReg1, name="InReg1")
        reg2.register(InReg2, name="InReg2")

        assert reg1.is_registered(InReg1)
        assert not reg1.is_registered(InReg2)
        assert reg2.is_registered(InReg2)
        assert not reg2.is_registered(InReg1)
