"""
Tests for the hot-reload core functionality.
"""
import pytest
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any

from engine.tooling.hotreload.hot_reload import (
    reloadable,
    HotReloader,
    ReloadResult,
    ReloadableClass,
    ReloadError,
    SchemaBreakingChangeError,
    _reloadable_registry,
    _instance_registry,
)


class TestReloadableDecorator:
    """Tests for the @reloadable decorator."""

    def setup_method(self):
        """Clear registries before each test."""
        _reloadable_registry.clear()
        _instance_registry.clear()

    def test_reloadable_registers_class(self):
        """Test that @reloadable registers the class."""
        @reloadable()
        class TestClass:
            pass

        full_name = f"{TestClass.__module__}.TestClass"
        assert full_name in _reloadable_registry
        assert _reloadable_registry[full_name] is TestClass

    def test_reloadable_adds_metadata(self):
        """Test that @reloadable adds metadata attributes."""
        @reloadable(preserve_state=True, allow_schema_changes=False)
        class TestClass:
            x: int = 10

        assert hasattr(TestClass, "__reloadable__")
        assert TestClass.__reloadable__ is True
        assert TestClass.__preserve_state__ is True
        assert TestClass.__allow_schema_changes__ is False
        assert hasattr(TestClass, "__schema_hash__")

    def test_reloadable_tracks_instances(self):
        """Test that instances are tracked."""
        @reloadable()
        class TestClass:
            pass

        full_name = f"{TestClass.__module__}.TestClass"

        obj1 = TestClass()
        obj2 = TestClass()

        assert full_name in _instance_registry
        # Weak refs, so we need to check if objects are tracked
        refs = _instance_registry[full_name]
        assert len(refs) == 2

    def test_reloadable_with_migration(self):
        """Test @reloadable with migration function."""
        def migrate(state, old_hash, new_hash):
            state["new_field"] = "default"
            return state

        @reloadable(migration=migrate)
        class TestClass:
            x: int = 10

        assert TestClass.__migration_fn__ is migrate

    def test_reloadable_preserves_init(self):
        """Test that original __init__ is preserved."""
        @reloadable()
        class TestClass:
            def __init__(self, value):
                self.value = value

        obj = TestClass(42)
        assert obj.value == 42


class TestHotReloader:
    """Tests for the HotReloader class."""

    def setup_method(self):
        """Create fresh reloader for each test."""
        _reloadable_registry.clear()
        _instance_registry.clear()
        self.reloader = HotReloader()

    def test_reloader_initialization(self):
        """Test HotReloader initializes correctly."""
        assert self.reloader.reload_count == 0
        assert self.reloader.enabled is True

    def test_reloader_enable_disable(self):
        """Test enabling/disabling hot reload."""
        self.reloader.enabled = False
        assert self.reloader.enabled is False

        self.reloader.enabled = True
        assert self.reloader.enabled is True

    def test_get_reloadable_classes(self):
        """Test getting registered reloadable classes."""
        @reloadable()
        class TestClass:
            x: int = 10

        classes = self.reloader.get_reloadable_classes()
        assert len(classes) == 1

        full_name = f"{TestClass.__module__}.TestClass"
        assert full_name in classes
        assert isinstance(classes[full_name], ReloadableClass)

    def test_get_instances(self):
        """Test getting instances of a reloadable class."""
        @reloadable()
        class TestClass:
            pass

        full_name = f"{TestClass.__module__}.TestClass"

        obj1 = TestClass()
        obj2 = TestClass()

        instances = self.reloader.get_instances(full_name)
        assert len(instances) == 2
        assert obj1 in instances
        assert obj2 in instances

    def test_preserve_state(self):
        """Test state preservation."""
        @reloadable()
        class TestClass:
            def __init__(self):
                self.health = 100
                self.name = "test"

        obj = TestClass()
        obj.health = 50
        obj.name = "modified"

        state = self.reloader.preserve_state(obj)

        assert "health" in state or "__type__" in state
        # Check through the serialized state
        if "__type__" in state:
            assert state.get("health") == 50
            assert state.get("name") == "modified"

    def test_restore_state(self):
        """Test state restoration."""
        @reloadable()
        class TestClass:
            def __init__(self):
                self.health = 100
                self.name = "test"

        obj = TestClass()
        state = {"health": 50, "name": "restored"}

        self.reloader.restore_state(obj, state)

        assert obj.health == 50
        assert obj.name == "restored"

    def test_reload_disabled(self):
        """Test reload when disabled."""
        self.reloader.enabled = False

        result = self.reloader.reload_module("test_module")

        assert result.success is False
        assert "disabled" in result.errors[0].lower()

    def test_reload_nonexistent_module(self):
        """Test reloading a module not in sys.modules."""
        result = self.reloader.reload_module("nonexistent_module_xyz")

        assert result.success is False
        assert "not loaded" in result.errors[0].lower()

    def test_reload_callbacks(self):
        """Test reload callbacks are invoked."""
        start_called = []
        complete_called = []
        error_called = []

        def on_start(module):
            start_called.append(module)

        def on_complete(result):
            complete_called.append(result)

        def on_error(module, exc):
            error_called.append((module, exc))

        reloader = HotReloader(
            on_reload_start=on_start,
            on_reload_complete=on_complete,
            on_reload_error=on_error,
        )

        # Reload a non-existent module to trigger callbacks
        result = reloader.reload_module("nonexistent_xyz")

        # on_start and on_complete should be called regardless
        assert len(start_called) == 1
        assert len(complete_called) == 1

    def test_clear_registry(self):
        """Test clearing the registry."""
        @reloadable()
        class TestClass:
            pass

        obj = TestClass()

        assert len(_reloadable_registry) > 0
        assert len(_instance_registry) > 0

        self.reloader.clear_registry()

        assert len(_reloadable_registry) == 0
        assert len(_instance_registry) == 0

    def test_check_schema_compatibility(self):
        """Test schema compatibility checking."""
        # Use the same class definition to avoid class rename detection
        @dataclass
        class TestClass:
            x: int = 10
            y: str = "test"

        breaking = self.reloader.check_schema_compatibility(TestClass, TestClass)
        # Same class should have no breaking changes
        assert len(breaking) == 0

    def test_check_schema_breaking_change(self):
        """Test detection of breaking schema changes."""
        @dataclass
        class OldClass:
            x: int = 10
            y: str = "test"

        @dataclass
        class NewClass:
            x: int = 10
            # y removed - breaking change

        breaking = self.reloader.check_schema_compatibility(OldClass, NewClass)
        assert len(breaking) > 0


class TestReloadResult:
    """Tests for ReloadResult."""

    def test_reload_result_success(self):
        """Test successful reload result."""
        result = ReloadResult(
            success=True,
            module_name="test_module",
            reloaded_classes=["test_module.TestClass"],
            preserved_instances=5,
            elapsed_time=0.1,
        )

        assert result.success
        assert result.module_name == "test_module"
        assert len(result.reloaded_classes) == 1
        assert result.preserved_instances == 5
        assert len(result.errors) == 0

    def test_reload_result_failure(self):
        """Test failed reload result."""
        result = ReloadResult(
            success=False,
            module_name="test_module",
            errors=["Import error", "Type error"],
        )

        assert not result.success
        assert len(result.errors) == 2


class TestSchemaBreakingChangeError:
    """Tests for SchemaBreakingChangeError."""

    def test_error_attributes(self):
        """Test error has correct attributes."""
        error = SchemaBreakingChangeError(
            class_name="TestClass",
            old_hash="abc123",
            new_hash="def456",
            breaking_changes=["Field removed: x", "Type changed: y"],
        )

        assert error.class_name == "TestClass"
        assert error.old_hash == "abc123"
        assert error.new_hash == "def456"
        assert len(error.breaking_changes) == 2

    def test_error_message(self):
        """Test error message format."""
        error = SchemaBreakingChangeError(
            class_name="TestClass",
            old_hash="abc",
            new_hash="def",
            breaking_changes=["Field removed"],
        )

        assert "TestClass" in str(error)
        assert "Field removed" in str(error)
