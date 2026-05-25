"""
Comprehensive tests for EngineMeta - Base metaclass for all engine types.

Tests cover:
- Type registration in global registry
- Thread-safe concurrent class creation
- Registry filtering and access
- __repr__ formatting
- Base class name skipping
- Registry clearing
"""
import threading
from typing import Any

import pytest

from trinity.metaclasses import EngineMeta


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test to avoid pollution."""
    EngineMeta.clear_registry()
    yield
    EngineMeta.clear_registry()


def test_registration_basic():
    """Test that classes are registered in the global registry."""

    class TestEngine(metaclass=EngineMeta):
        pass

    registry = EngineMeta.get_all_types()
    qualified_name = f"{TestEngine.__module__}.TestEngine"

    assert qualified_name in registry
    assert registry[qualified_name] is TestEngine


def test_base_class_names_skipped():
    """Test that base class names in _BASE_CLASS_NAMES are not registered."""

    # These should be skipped
    class EngineBase(metaclass=EngineMeta):
        pass

    class Component(metaclass=EngineMeta):
        pass

    class System(metaclass=EngineMeta):
        pass

    registry = EngineMeta.get_all_types()

    # None of the base class names should be in registry
    assert not any("EngineBase" in name for name in registry)
    assert not any("Component" in name for name in registry)
    assert not any("System" in name for name in registry)


def test_multiple_class_registration():
    """Test that multiple classes are all registered correctly."""

    class Engine1(metaclass=EngineMeta):
        pass

    class Engine2(metaclass=EngineMeta):
        pass

    class Engine3(metaclass=EngineMeta):
        pass

    registry = EngineMeta.get_all_types()

    assert len(registry) == 3
    assert any("Engine1" in name for name in registry)
    assert any("Engine2" in name for name in registry)
    assert any("Engine3" in name for name in registry)


def test_get_all_types():
    """Test get_all_types returns copy of registry."""

    class TestEngine(metaclass=EngineMeta):
        pass

    registry1 = EngineMeta.get_all_types()
    registry2 = EngineMeta.get_all_types()

    # Should be different dict instances (defensive copy)
    assert registry1 is not registry2
    # But with same content
    assert registry1 == registry2


def test_get_types_by_metaclass():
    """Test filtering types by specific metaclass."""

    # Create a custom metaclass
    class CustomMeta(EngineMeta):
        pass

    class Engine1(metaclass=EngineMeta):
        pass

    class Engine2(metaclass=CustomMeta):
        pass

    class Engine3(metaclass=CustomMeta):
        pass

    # Get only CustomMeta instances
    custom_types = EngineMeta.get_types_by_metaclass(CustomMeta)

    assert len(custom_types) == 2
    assert any("Engine2" in name for name in custom_types)
    assert any("Engine3" in name for name in custom_types)
    assert not any("Engine1" in name for name in custom_types)


def test_get_types_by_metaclass_empty():
    """Test get_types_by_metaclass with no matching types."""

    class CustomMeta(EngineMeta):
        pass

    class Engine1(metaclass=EngineMeta):
        pass

    # No classes use CustomMeta
    custom_types = EngineMeta.get_types_by_metaclass(CustomMeta)
    assert len(custom_types) == 0


def test_clear_registry():
    """Test that clear_registry removes all registered types."""

    class Engine1(metaclass=EngineMeta):
        pass

    class Engine2(metaclass=EngineMeta):
        pass

    assert len(EngineMeta.get_all_types()) == 2

    EngineMeta.clear_registry()

    assert len(EngineMeta.get_all_types()) == 0


def test_repr_format():
    """Test that __repr__ returns clean format."""

    class TestEngine(metaclass=EngineMeta):
        pass

    repr_str = repr(TestEngine)

    # Should be <Engine 'TestEngine'> (Meta suffix removed)
    assert repr_str == "<Engine 'TestEngine'>"


def test_repr_format_custom_metaclass():
    """Test __repr__ with custom metaclass name."""

    class CustomMeta(EngineMeta):
        pass

    class TestCustom(metaclass=CustomMeta):
        pass

    repr_str = repr(TestCustom)

    # Should be <Custom 'TestCustom'> (Meta suffix removed)
    assert repr_str == "<Custom 'TestCustom'>"


def test_repr_format_no_meta_suffix():
    """Test __repr__ with metaclass that doesn't end in Meta."""

    class Special(EngineMeta):
        pass

    class TestSpecial(metaclass=Special):
        pass

    repr_str = repr(TestSpecial)

    # Should use metaclass name as-is
    assert repr_str == "<Special 'TestSpecial'>"


def test_thread_safety_concurrent_creation():
    """Test that concurrent class creation is thread-safe."""
    errors = []
    created_classes = []

    def create_class(index: int):
        try:
            # Create class with unique name
            class_name = f"ConcurrentEngine{index}"
            cls = EngineMeta(class_name, (), {})
            created_classes.append(cls)
        except Exception as e:
            errors.append(e)

    # Create 50 classes concurrently
    threads = []
    for i in range(50):
        t = threading.Thread(target=create_class, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # No errors should occur
    assert len(errors) == 0

    # All 50 classes should be created
    assert len(created_classes) == 50

    # All should be registered
    registry = EngineMeta.get_all_types()
    assert len(registry) == 50


def test_thread_safety_concurrent_access():
    """Test that concurrent registry access is thread-safe."""

    class TestEngine(metaclass=EngineMeta):
        pass

    errors = []
    results = []

    def access_registry():
        try:
            # Multiple operations
            all_types = EngineMeta.get_all_types()
            by_meta = EngineMeta.get_types_by_metaclass(EngineMeta)
            results.append((all_types, by_meta))
        except Exception as e:
            errors.append(e)

    # 20 threads accessing concurrently
    threads = [threading.Thread(target=access_registry) for _ in range(20)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) == 20

    # All results should be consistent
    first_all, first_by_meta = results[0]
    for all_types, by_meta in results:
        assert all_types == first_all
        assert by_meta == first_by_meta


def test_qualified_name_format():
    """Test that qualified names include module."""

    class TestEngine(metaclass=EngineMeta):
        pass

    registry = EngineMeta.get_all_types()

    # Should have exactly one entry
    assert len(registry) == 1

    key = list(registry.keys())[0]

    # Key should be module.ClassName format
    assert "." in key
    assert key.endswith(".TestEngine")
    assert TestEngine.__module__ in key


def test_inheritance_registration():
    """Test that subclasses are registered independently."""

    class BaseEngine(metaclass=EngineMeta):
        pass

    class DerivedEngine(BaseEngine):
        pass

    registry = EngineMeta.get_all_types()

    # Both should be registered (BaseEngine is not in _BASE_CLASS_NAMES)
    assert len(registry) == 2
    assert any("BaseEngine" in name for name in registry)
    assert any("DerivedEngine" in name for name in registry)


def test_clear_registry_preserves_metaclass():
    """Test that clearing registry doesn't break metaclass functionality."""

    class Engine1(metaclass=EngineMeta):
        pass

    EngineMeta.clear_registry()

    # Should be able to create new classes after clearing
    class Engine2(metaclass=EngineMeta):
        pass

    registry = EngineMeta.get_all_types()
    assert len(registry) == 1
    assert any("Engine2" in name for name in registry)


def test_empty_registry_operations():
    """Test registry operations on empty registry."""

    # Clear to ensure empty
    EngineMeta.clear_registry()

    all_types = EngineMeta.get_all_types()
    assert all_types == {}

    by_meta = EngineMeta.get_types_by_metaclass(EngineMeta)
    assert by_meta == {}

    # clear_registry on empty registry should not error
    EngineMeta.clear_registry()
