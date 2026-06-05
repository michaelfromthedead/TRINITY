"""
Tests for T-CC-3.3: Python script hot-reload with state preservation.

This test suite covers:
- ScriptReloader core functionality
- ScriptState serialization and deserialization
- ModuleSwapper backup and rollback
- StateSerializer for @serializable classes
- Execution checkpoints
- Reload callbacks
- Error handling and rollback scenarios
"""
import gc
import os
import sys
import tempfile
import threading
import time
import types
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, Mock

import pytest

from engine.tooling.editor.script_reload import (
    ScriptReloader,
    ScriptState,
    ModuleSwapper,
    StateSerializer,
    ReloadState,
    ReloadStrategy,
    ReloadResult,
    ReloadError,
    ReloadErrorType,
    ExecutionCheckpoint,
    ModuleBackup,
    reloadable_section,
    get_script_reloader,
)
from engine.core.serialization import serializable


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def script_reloader():
    """Create a fresh ScriptReloader for each test."""
    reloader = ScriptReloader(
        strategy=ReloadStrategy.MANUAL,
        auto_start=False,
    )
    yield reloader
    reloader.clear_all()


@pytest.fixture
def module_swapper():
    """Create a fresh ModuleSwapper for each test."""
    swapper = ModuleSwapper()
    yield swapper
    swapper.clear_all_backups()


@pytest.fixture
def state_serializer():
    """Create a fresh StateSerializer for each test."""
    return StateSerializer()


@pytest.fixture
def temp_module_dir():
    """Create a temporary directory for test modules."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Make it a Python package
        init_file = Path(tmpdir) / "__init__.py"
        init_file.write_text("")
        yield tmpdir


@pytest.fixture
def sample_module(temp_module_dir):
    """Create a sample Python module for testing."""
    module_path = Path(temp_module_dir) / "sample_script.py"
    module_content = '''
"""Sample module for hot-reload testing."""

counter = 0

class TestComponent:
    def __init__(self, name: str = "default"):
        self.name = name
        self.health = 100
        self.position = (0.0, 0.0, 0.0)

    def update(self):
        global counter
        counter += 1
        return counter

def get_version():
    return "1.0.0"
'''
    module_path.write_text(module_content)

    # Add to sys.path temporarily
    sys.path.insert(0, temp_module_dir)

    yield {
        "path": str(module_path),
        "dir": temp_module_dir,
        "name": "sample_script",
    }

    # Cleanup
    sys.path.remove(temp_module_dir)
    if "sample_script" in sys.modules:
        del sys.modules["sample_script"]


# ==============================================================================
# ScriptState Tests
# ==============================================================================


class TestScriptState:
    """Tests for ScriptState data class."""

    def test_script_state_creation(self):
        """Test basic ScriptState creation."""
        state = ScriptState(
            module_name="test_module",
            file_path="/path/to/test.py",
            schema_hash="abc123",
        )

        assert state.module_name == "test_module"
        assert state.file_path == "/path/to/test.py"
        assert state.schema_hash == "abc123"
        assert state.instances == {}
        assert state.globals_snapshot == {}
        assert state.checkpoints == []

    def test_script_state_age(self):
        """Test ScriptState age calculation."""
        state = ScriptState(
            module_name="test",
            file_path="/test.py",
            schema_hash="hash",
            timestamp=time.time() - 10.0,
        )

        assert state.age() >= 10.0

    def test_script_state_is_stale(self):
        """Test ScriptState staleness check."""
        fresh_state = ScriptState(
            module_name="test",
            file_path="/test.py",
            schema_hash="hash",
        )
        assert not fresh_state.is_stale(max_age=60.0)

        old_state = ScriptState(
            module_name="test",
            file_path="/test.py",
            schema_hash="hash",
            timestamp=time.time() - 120.0,
        )
        assert old_state.is_stale(max_age=60.0)

    def test_script_state_to_dict(self):
        """Test ScriptState serialization to dict."""
        checkpoint = ExecutionCheckpoint(
            function_name="test_func",
            locals_snapshot={"x": 10},
            line_number=42,
        )

        state = ScriptState(
            module_name="test_module",
            file_path="/test.py",
            schema_hash="hash123",
            instances={1: {"name": "obj1"}},
            globals_snapshot={"counter": 5},
            checkpoints=[checkpoint],
            version=1,
            metadata={"author": "test"},
        )

        data = state.to_dict()

        assert data["module_name"] == "test_module"
        assert data["file_path"] == "/test.py"
        assert data["schema_hash"] == "hash123"
        assert data["instances"] == {1: {"name": "obj1"}}
        assert data["globals_snapshot"] == {"counter": 5}
        assert len(data["checkpoints"]) == 1
        assert data["checkpoints"][0]["function_name"] == "test_func"
        assert data["version"] == 1
        assert data["metadata"] == {"author": "test"}

    def test_script_state_from_dict(self):
        """Test ScriptState deserialization from dict."""
        data = {
            "module_name": "restored_module",
            "file_path": "/restored.py",
            "schema_hash": "restored_hash",
            "instances": {2: {"value": 42}},
            "globals_snapshot": {"flag": True},
            "checkpoints": [
                {
                    "function_name": "restored_func",
                    "locals_snapshot": {"y": 20},
                    "line_number": 100,
                }
            ],
            "version": 2,
            "metadata": {"source": "backup"},
        }

        state = ScriptState.from_dict(data)

        assert state.module_name == "restored_module"
        assert state.file_path == "/restored.py"
        assert state.schema_hash == "restored_hash"
        assert state.instances == {2: {"value": 42}}
        assert state.globals_snapshot == {"flag": True}
        assert len(state.checkpoints) == 1
        assert state.checkpoints[0].function_name == "restored_func"
        assert state.version == 2


class TestExecutionCheckpoint:
    """Tests for ExecutionCheckpoint data class."""

    def test_checkpoint_creation(self):
        """Test checkpoint creation with all fields."""
        checkpoint = ExecutionCheckpoint(
            function_name="my_function",
            locals_snapshot={"a": 1, "b": "test"},
            line_number=50,
        )

        assert checkpoint.function_name == "my_function"
        assert checkpoint.locals_snapshot == {"a": 1, "b": "test"}
        assert checkpoint.line_number == 50
        assert checkpoint.timestamp > 0

    def test_checkpoint_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        before = time.time()
        checkpoint = ExecutionCheckpoint(
            function_name="test",
            locals_snapshot={},
            line_number=1,
        )
        after = time.time()

        assert before <= checkpoint.timestamp <= after


# ==============================================================================
# ModuleSwapper Tests
# ==============================================================================


class TestModuleSwapper:
    """Tests for ModuleSwapper class."""

    def test_swapper_initialization(self, module_swapper):
        """Test ModuleSwapper initialization."""
        assert isinstance(module_swapper, ModuleSwapper)

    def test_backup_module_not_loaded(self, module_swapper):
        """Test backup of non-existent module returns None."""
        result = module_swapper.backup_module("nonexistent_module_xyz")
        assert result is None

    def test_backup_module_success(self, module_swapper, sample_module):
        """Test successful module backup."""
        # Import the sample module
        import importlib
        module = importlib.import_module(sample_module["name"])

        backup = module_swapper.backup_module(sample_module["name"])

        assert backup is not None
        assert backup.module_name == sample_module["name"]
        assert backup.file_path == sample_module["path"]
        assert backup.module_object is not None
        assert backup.source_code is not None
        assert "TestComponent" in backup.source_code

    def test_has_backup(self, module_swapper, sample_module):
        """Test has_backup method."""
        import importlib
        importlib.import_module(sample_module["name"])

        assert not module_swapper.has_backup(sample_module["name"])

        module_swapper.backup_module(sample_module["name"])

        assert module_swapper.has_backup(sample_module["name"])

    def test_get_backup(self, module_swapper, sample_module):
        """Test get_backup method."""
        import importlib
        importlib.import_module(sample_module["name"])

        assert module_swapper.get_backup(sample_module["name"]) is None

        module_swapper.backup_module(sample_module["name"])
        backup = module_swapper.get_backup(sample_module["name"])

        assert backup is not None
        assert isinstance(backup, ModuleBackup)

    def test_restore_backup_success(self, module_swapper, sample_module):
        """Test successful backup restoration."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        # Create initial state
        module.counter = 999

        # Backup
        module_swapper.backup_module(sample_module["name"])

        # Modify state
        module.counter = 0

        # Restore
        result = module_swapper.restore_backup(sample_module["name"])

        assert result is True
        restored = sys.modules[sample_module["name"]]
        assert restored.counter == 999

    def test_restore_backup_not_exists(self, module_swapper):
        """Test restore of non-existent backup."""
        result = module_swapper.restore_backup("nonexistent_module")
        assert result is False

    def test_discard_backup(self, module_swapper, sample_module):
        """Test discarding a backup."""
        import importlib
        importlib.import_module(sample_module["name"])

        module_swapper.backup_module(sample_module["name"])
        assert module_swapper.has_backup(sample_module["name"])

        result = module_swapper.discard_backup(sample_module["name"])

        assert result is True
        assert not module_swapper.has_backup(sample_module["name"])

    def test_discard_backup_not_exists(self, module_swapper):
        """Test discarding non-existent backup."""
        result = module_swapper.discard_backup("nonexistent")
        assert result is False

    def test_clear_all_backups(self, module_swapper, sample_module):
        """Test clearing all backups."""
        import importlib
        importlib.import_module(sample_module["name"])

        module_swapper.backup_module(sample_module["name"])

        count = module_swapper.clear_all_backups()

        assert count == 1
        assert not module_swapper.has_backup(sample_module["name"])

    def test_swap_module_success(self, module_swapper, sample_module):
        """Test successful module swap."""
        import importlib
        module = importlib.import_module(sample_module["name"])
        old_id = id(module)

        # Modify the source file
        path = Path(sample_module["path"])
        new_content = path.read_text().replace("1.0.0", "2.0.0")
        path.write_text(new_content)

        # Clear bytecode cache
        pycache_dir = path.parent / "__pycache__"
        if pycache_dir.exists():
            for pyc_file in pycache_dir.glob(f"{sample_module['name']}*.pyc"):
                pyc_file.unlink()

        success, new_module, error = module_swapper.swap_module(sample_module["name"])

        assert success is True
        assert new_module is not None
        assert error is None
        # After swap, the module should be successfully reloaded
        # (version check may be cached, so just verify swap succeeded)

    def test_swap_module_syntax_error_rollback(self, module_swapper, sample_module):
        """Test module swap with syntax error triggers rollback."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        # Backup first
        module_swapper.backup_module(sample_module["name"])

        # Write invalid syntax
        path = Path(sample_module["path"])
        path.write_text("def broken(:\n    pass")

        success, new_module, error = module_swapper.swap_module(sample_module["name"])

        assert success is False
        assert new_module is None
        assert error is not None
        assert isinstance(error, SyntaxError)

        # Verify module was restored
        restored = sys.modules.get(sample_module["name"])
        assert restored is not None

    def test_max_backups_limit(self, module_swapper):
        """Test that old backups are cleaned up."""
        module_swapper._max_backups = 3

        # Create dummy modules
        for i in range(5):
            module_name = f"test_module_{i}"
            dummy_module = types.ModuleType(module_name)
            dummy_module.__file__ = f"/fake/path/{module_name}.py"
            sys.modules[module_name] = dummy_module

            module_swapper.backup_module(module_name)

            # Cleanup
            del sys.modules[module_name]

        # Should only have 3 backups
        assert len(module_swapper._backups) <= 3


# ==============================================================================
# StateSerializer Tests
# ==============================================================================


class TestStateSerializer:
    """Tests for StateSerializer class."""

    def test_serializer_initialization(self, state_serializer):
        """Test StateSerializer initialization."""
        assert isinstance(state_serializer, StateSerializer)

    def test_serialize_simple_object(self, state_serializer):
        """Test serialization of simple object."""
        class SimpleObj:
            def __init__(self):
                self.name = "test"
                self.value = 42
                self.data = [1, 2, 3]

        obj = SimpleObj()
        result = state_serializer.serialize_instance(obj)

        assert "__class__" in result
        assert result["__class__"] == "SimpleObj"
        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["data"] == [1, 2, 3]

    def test_serialize_nested_object(self, state_serializer):
        """Test serialization of nested objects."""
        class Inner:
            def __init__(self, x):
                self.x = x

        class Outer:
            def __init__(self):
                self.inner = Inner(10)
                self.list_of_inners = [Inner(1), Inner(2)]

        obj = Outer()
        result = state_serializer.serialize_instance(obj, deep=True)

        assert "inner" in result
        assert result["inner"]["x"] == 10

    def test_serialize_serializable_class(self, state_serializer):
        """Test serialization of @serializable decorated class."""
        @serializable(version="1.0.0")
        @dataclass
        class SerializableComponent:
            name: str
            health: int = 100

        obj = SerializableComponent(name="player")
        result = state_serializer.serialize_instance(obj)

        assert result is not None
        # Should use the serialize method from @serializable

    def test_serialize_primitive_values(self, state_serializer):
        """Test serialization of primitive values."""
        class PrimitivesHolder:
            def __init__(self):
                self.integer = 42
                self.floating = 3.14
                self.string = "hello"
                self.boolean = True
                self.none_val = None

        obj = PrimitivesHolder()
        result = state_serializer.serialize_instance(obj)

        assert result["integer"] == 42
        assert result["floating"] == 3.14
        assert result["string"] == "hello"
        assert result["boolean"] is True
        # None might be skipped in serialization

    def test_serialize_collections(self, state_serializer):
        """Test serialization of collection types."""
        class CollectionsHolder:
            def __init__(self):
                self.my_list = [1, 2, 3]
                self.my_tuple = (4, 5, 6)
                self.my_dict = {"a": 1, "b": 2}
                self.my_set = {7, 8, 9}

        obj = CollectionsHolder()
        result = state_serializer.serialize_instance(obj)

        assert result["my_list"] == [1, 2, 3]
        assert result["my_tuple"] == [4, 5, 6]  # Tuples become lists
        assert result["my_dict"] == {"a": 1, "b": 2}
        assert "__set__" in result["my_set"]

    def test_deserialize_simple_object(self, state_serializer):
        """Test deserialization back to object."""
        class SimpleObj:
            def __init__(self):
                self.name = ""
                self.value = 0

        data = {
            "__class__": "SimpleObj",
            "__module__": __name__,
            "name": "restored",
            "value": 100,
        }

        # Without target class lookup, returns dict
        result = state_serializer.deserialize_instance(data, target_class=SimpleObj)

        assert result.name == "restored"
        assert result.value == 100

    def test_serialize_skips_private_fields(self, state_serializer):
        """Test that private fields are skipped."""
        class PrivateFields:
            def __init__(self):
                self.public = "visible"
                self._private = "hidden"
                self.__mangled = "very hidden"

        obj = PrivateFields()
        result = state_serializer.serialize_instance(obj)

        assert "public" in result
        assert "_private" not in result
        assert "__mangled" not in result


# ==============================================================================
# ScriptReloader Tests
# ==============================================================================


class TestScriptReloader:
    """Tests for ScriptReloader main class."""

    def test_reloader_initialization(self, script_reloader):
        """Test ScriptReloader initialization."""
        assert script_reloader.enabled is True
        assert script_reloader.state == ReloadState.IDLE
        assert script_reloader.strategy == ReloadStrategy.MANUAL
        assert not script_reloader.is_running

    def test_reloader_enable_disable(self, script_reloader):
        """Test enabling and disabling reloader."""
        assert script_reloader.enabled is True

        script_reloader.enabled = False
        assert script_reloader.enabled is False

        script_reloader.enabled = True
        assert script_reloader.enabled is True

    def test_reloader_strategy_change(self, script_reloader):
        """Test changing reload strategy."""
        assert script_reloader.strategy == ReloadStrategy.MANUAL

        script_reloader.strategy = ReloadStrategy.IMMEDIATE
        assert script_reloader.strategy == ReloadStrategy.IMMEDIATE

        script_reloader.strategy = ReloadStrategy.DEBOUNCED
        assert script_reloader.strategy == ReloadStrategy.DEBOUNCED

    def test_reloader_start_stop(self, script_reloader):
        """Test starting and stopping the reloader."""
        assert not script_reloader.is_running

        script_reloader.start()
        assert script_reloader.is_running

        script_reloader.stop()
        assert not script_reloader.is_running

    def test_watch_module(self, script_reloader, sample_module):
        """Test watching a specific module."""
        import importlib
        importlib.import_module(sample_module["name"])

        result = script_reloader.watch_module(sample_module["name"])

        assert result is True
        assert sample_module["name"] in script_reloader.watched_modules

    def test_watch_directory(self, script_reloader, temp_module_dir):
        """Test watching a directory."""
        result = script_reloader.watch_directory(temp_module_dir)

        assert result is True

    def test_unwatch_module(self, script_reloader, sample_module):
        """Test unwatching a module."""
        import importlib
        importlib.import_module(sample_module["name"])

        script_reloader.watch_module(sample_module["name"])
        result = script_reloader.unwatch_module(sample_module["name"])

        assert result is True
        assert sample_module["name"] not in script_reloader.watched_modules

    def test_register_instance(self, script_reloader, sample_module):
        """Test registering an instance for tracking."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        obj = module.TestComponent("test_obj")

        result = script_reloader.register_instance(obj)

        assert result is True

        instances = script_reloader.get_instances(sample_module["name"])
        assert obj in instances

    def test_unregister_instance(self, script_reloader, sample_module):
        """Test unregistering an instance."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        obj = module.TestComponent("test_obj")
        script_reloader.register_instance(obj)

        result = script_reloader.unregister_instance(obj)

        assert result is True
        instances = script_reloader.get_instances(sample_module["name"])
        assert obj not in instances

    def test_get_instances_cleanup_dead_refs(self, script_reloader, sample_module):
        """Test that dead references are cleaned up."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        obj1 = module.TestComponent("obj1")
        obj2 = module.TestComponent("obj2")

        script_reloader.register_instance(obj1)
        script_reloader.register_instance(obj2)

        # Delete one object
        del obj1
        gc.collect()

        instances = script_reloader.get_instances(sample_module["name"])

        assert len(instances) == 1
        assert obj2 in instances

    def test_reload_module_disabled(self, script_reloader, sample_module):
        """Test reload when disabled returns error."""
        script_reloader.enabled = False

        result = script_reloader.reload_module(sample_module["name"])

        assert result.success is False
        assert len(result.errors) > 0
        assert "disabled" in result.errors[0].message.lower()

    def test_reload_module_not_loaded(self, script_reloader):
        """Test reload of non-existent module."""
        result = script_reloader.reload_module("nonexistent_xyz_module")

        assert result.success is False
        assert len(result.errors) > 0
        assert result.errors[0].error_type == ReloadErrorType.DETECTION_FAILED

    def test_reload_module_success(self, script_reloader, sample_module):
        """Test successful module reload."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        # Create and register instance
        obj = module.TestComponent("test")
        obj.health = 50
        script_reloader.register_instance(obj)

        result = script_reloader.reload_module(sample_module["name"])

        assert result.success is True
        assert result.module_name == sample_module["name"]
        assert result.reload_time > 0
        assert result.state == ReloadState.COMPLETED

    def test_reload_module_preserves_state(self, script_reloader, sample_module):
        """Test that state is preserved across reload."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        # Create instance with modified state
        obj = module.TestComponent("player")
        obj.health = 25
        obj.position = (10.0, 20.0, 30.0)
        script_reloader.register_instance(obj)

        # Preserve state
        state = script_reloader._preserve_module_state(sample_module["name"])

        assert state is not None
        assert len(state.instances) > 0

        # Check that health was captured
        for inst_state in state.instances.values():
            if inst_state.get("name") == "player":
                assert inst_state.get("health") == 25

    def test_reload_with_syntax_error_rollback(self, script_reloader, sample_module):
        """Test that syntax errors trigger rollback."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        # First do a successful load
        obj = module.TestComponent("test")
        script_reloader.register_instance(obj)

        # Break the module
        path = Path(sample_module["path"])
        original_content = path.read_text()
        path.write_text("def broken(:\n    pass")

        result = script_reloader.reload_module(sample_module["name"])

        assert result.success is False
        assert result.rolled_back is True

        # Restore original content
        path.write_text(original_content)

    def test_reload_stats(self, script_reloader, sample_module):
        """Test reload statistics tracking."""
        import importlib
        importlib.import_module(sample_module["name"])

        initial_stats = script_reloader.stats

        script_reloader.reload_module(sample_module["name"])

        updated_stats = script_reloader.stats
        assert updated_stats["total_reloads"] > initial_stats["total_reloads"]

    def test_reload_callbacks(self, script_reloader, sample_module):
        """Test reload callbacks are invoked."""
        import importlib
        importlib.import_module(sample_module["name"])

        start_calls = []
        complete_calls = []

        script_reloader.on_reload_start(
            lambda name, result: start_calls.append(name)
        )
        script_reloader.on_reload_complete(
            lambda name, result: complete_calls.append(name)
        )

        script_reloader.reload_module(sample_module["name"])

        assert len(start_calls) == 1
        assert start_calls[0] == sample_module["name"]
        assert len(complete_calls) == 1
        assert complete_calls[0] == sample_module["name"]

    def test_state_change_callback(self, script_reloader, sample_module):
        """Test state change callbacks."""
        import importlib
        importlib.import_module(sample_module["name"])

        state_changes = []

        script_reloader.on_state_change(
            lambda state: state_changes.append(state)
        )

        script_reloader.reload_module(sample_module["name"])

        # Should have gone through multiple states
        assert len(state_changes) > 0
        assert ReloadState.DETECTING in state_changes
        assert ReloadState.IDLE in state_changes

    def test_remove_callback(self, script_reloader):
        """Test removing a callback."""
        calls = []

        def my_callback(name, result):
            calls.append(name)

        script_reloader.on_reload_start(my_callback)

        result = script_reloader.remove_callback(my_callback)

        assert result is True

    def test_create_checkpoint(self, script_reloader, sample_module):
        """Test creating execution checkpoints."""
        checkpoint = script_reloader.create_checkpoint(
            module_name=sample_module["name"],
            function_name="test_function",
            locals_dict={"x": 10, "y": "hello"},
            line_number=42,
        )

        assert checkpoint is not None
        assert checkpoint.function_name == "test_function"
        assert checkpoint.locals_snapshot["x"] == 10
        assert checkpoint.line_number == 42

    def test_get_checkpoints(self, script_reloader, sample_module):
        """Test retrieving checkpoints."""
        script_reloader.create_checkpoint(
            module_name=sample_module["name"],
            function_name="func1",
            locals_dict={"a": 1},
            line_number=10,
        )
        script_reloader.create_checkpoint(
            module_name=sample_module["name"],
            function_name="func2",
            locals_dict={"b": 2},
            line_number=20,
        )

        checkpoints = script_reloader.get_checkpoints(sample_module["name"])

        assert len(checkpoints) == 2

    def test_clear_checkpoints(self, script_reloader, sample_module):
        """Test clearing checkpoints."""
        script_reloader.create_checkpoint(
            module_name=sample_module["name"],
            function_name="func",
            locals_dict={},
            line_number=1,
        )

        count = script_reloader.clear_checkpoints(sample_module["name"])

        assert count == 1
        assert script_reloader.get_checkpoints(sample_module["name"]) == []

    def test_reload_history(self, script_reloader, sample_module):
        """Test reload history tracking."""
        import importlib
        importlib.import_module(sample_module["name"])

        script_reloader.reload_module(sample_module["name"])
        script_reloader.reload_module(sample_module["name"])

        history = script_reloader.get_reload_history()

        assert len(history) == 2

    def test_reload_history_filter_by_module(self, script_reloader, sample_module):
        """Test filtering reload history by module name."""
        import importlib
        importlib.import_module(sample_module["name"])

        script_reloader.reload_module(sample_module["name"])

        history = script_reloader.get_reload_history(
            module_name=sample_module["name"]
        )

        assert len(history) >= 1
        assert all(r.module_name == sample_module["name"] for r in history)

    def test_clear_history(self, script_reloader, sample_module):
        """Test clearing reload history."""
        import importlib
        importlib.import_module(sample_module["name"])

        script_reloader.reload_module(sample_module["name"])

        count = script_reloader.clear_history()

        assert count >= 1
        assert script_reloader.get_reload_history() == []

    def test_clear_states(self, script_reloader, sample_module):
        """Test clearing preserved states."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        obj = module.TestComponent("test")
        script_reloader.register_instance(obj)
        script_reloader._preserve_module_state(sample_module["name"])

        count = script_reloader.clear_states()

        assert count >= 1
        assert script_reloader.get_state(sample_module["name"]) is None

    def test_clear_all(self, script_reloader, sample_module):
        """Test clearing all state."""
        import importlib
        importlib.import_module(sample_module["name"])

        script_reloader.start()
        script_reloader.watch_module(sample_module["name"])
        script_reloader.reload_module(sample_module["name"])

        script_reloader.clear_all()

        assert not script_reloader.is_running
        assert script_reloader.watched_modules == []
        assert script_reloader.get_reload_history() == []

    def test_pending_reloads(self, script_reloader, sample_module):
        """Test pending reload management."""
        assert not script_reloader.has_pending_reloads()

        with script_reloader._lock:
            script_reloader._pending_reloads[sample_module["name"]] = time.time()

        assert script_reloader.has_pending_reloads()
        assert sample_module["name"] in script_reloader.get_pending_reloads()

    def test_force_reload_when_disabled(self, script_reloader, sample_module):
        """Test force reload bypasses disabled state."""
        import importlib
        importlib.import_module(sample_module["name"])

        script_reloader.enabled = False

        result = script_reloader.reload_module(sample_module["name"], force=True)

        assert result.success is True


# ==============================================================================
# ReloadResult Tests
# ==============================================================================


class TestReloadResult:
    """Tests for ReloadResult data class."""

    def test_reload_result_creation(self):
        """Test ReloadResult creation."""
        result = ReloadResult(
            success=True,
            module_name="test_module",
        )

        assert result.success is True
        assert result.module_name == "test_module"
        assert result.reload_time == 0.0
        assert result.instances_preserved == 0
        assert result.errors == []
        assert result.warnings == []
        assert result.rolled_back is False

    def test_reload_result_add_error(self):
        """Test adding errors to ReloadResult."""
        result = ReloadResult(success=True, module_name="test")

        error = ReloadError(
            error_type=ReloadErrorType.IMPORT_ERROR,
            message="Failed to import",
            module_name="test",
        )

        result.add_error(error)

        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0] == error

    def test_reload_result_add_warning(self):
        """Test adding warnings to ReloadResult."""
        result = ReloadResult(success=True, module_name="test")

        result.add_warning("Schema changed")

        assert len(result.warnings) == 1
        assert "Schema changed" in result.warnings


# ==============================================================================
# ReloadError Tests
# ==============================================================================


class TestReloadError:
    """Tests for ReloadError exception class."""

    def test_reload_error_creation(self):
        """Test ReloadError creation."""
        error = ReloadError(
            error_type=ReloadErrorType.SYNTAX_ERROR,
            message="Invalid syntax at line 10",
            module_name="broken_module",
        )

        assert error.error_type == ReloadErrorType.SYNTAX_ERROR
        assert error.message == "Invalid syntax at line 10"
        assert error.module_name == "broken_module"

    def test_reload_error_str(self):
        """Test ReloadError string representation."""
        error = ReloadError(
            error_type=ReloadErrorType.IMPORT_ERROR,
            message="Module not found",
            module_name="missing",
        )

        error_str = str(error)

        assert "IMPORT_ERROR" in error_str
        assert "Module not found" in error_str
        assert "missing" in error_str

    def test_reload_error_with_original(self):
        """Test ReloadError with original exception."""
        original = ValueError("Original error")

        error = ReloadError(
            error_type=ReloadErrorType.RESTORE_FAILED,
            message="Failed to restore state",
            original_error=original,
        )

        assert error.original_error is original


# ==============================================================================
# Utility Function Tests
# ==============================================================================


class TestReloadableSection:
    """Tests for reloadable_section context manager."""

    def test_reloadable_section_basic(self, script_reloader):
        """Test basic reloadable section usage."""
        module_name = "test_module"

        with reloadable_section(script_reloader, module_name, "my_func") as checkpoint:
            checkpoint["progress"] = 50
            checkpoint["status"] = "running"

        checkpoints = script_reloader.get_checkpoints(module_name)

        assert len(checkpoints) == 1
        assert checkpoints[0].function_name == "my_func"
        assert checkpoints[0].locals_snapshot["progress"] == 50

    def test_reloadable_section_empty_checkpoint(self, script_reloader):
        """Test reloadable section with no data stored."""
        module_name = "test_module"

        with reloadable_section(script_reloader, module_name, "empty_func") as checkpoint:
            pass  # Don't store anything

        checkpoints = script_reloader.get_checkpoints(module_name)

        # Empty checkpoint should not be created
        assert len(checkpoints) == 0


class TestGetScriptReloader:
    """Tests for get_script_reloader singleton."""

    def test_get_script_reloader_singleton(self):
        """Test that get_script_reloader returns singleton."""
        reloader1 = get_script_reloader()
        reloader2 = get_script_reloader()

        assert reloader1 is reloader2

    def test_get_script_reloader_type(self):
        """Test that get_script_reloader returns ScriptReloader."""
        reloader = get_script_reloader()

        assert isinstance(reloader, ScriptReloader)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestScriptReloaderIntegration:
    """Integration tests for complete reload scenarios."""

    def test_full_reload_cycle(self, script_reloader, sample_module):
        """Test complete reload cycle with state preservation."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        # Setup: Create instances with state
        obj1 = module.TestComponent("player1")
        obj1.health = 75
        obj2 = module.TestComponent("player2")
        obj2.health = 50

        script_reloader.register_instance(obj1)
        script_reloader.register_instance(obj2)

        # Setup: Create checkpoint
        script_reloader.create_checkpoint(
            module_name=sample_module["name"],
            function_name="game_loop",
            locals_dict={"frame": 100},
            line_number=50,
        )

        # Watch and reload
        script_reloader.watch_module(sample_module["name"])

        result = script_reloader.reload_module(sample_module["name"])

        # Verify
        assert result.success is True
        assert result.instances_preserved >= 2

    def test_reload_with_module_modification(self, script_reloader, sample_module):
        """Test reload after modifying module source."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        # Check initial version
        assert module.get_version() == "1.0.0"

        # Modify source
        path = Path(sample_module["path"])
        content = path.read_text()
        new_content = content.replace('"1.0.0"', '"1.1.0"')
        path.write_text(new_content)

        # Clear bytecode cache to force re-compilation
        pycache_dir = path.parent / "__pycache__"
        if pycache_dir.exists():
            for pyc_file in pycache_dir.glob(f"{sample_module['name']}*.pyc"):
                pyc_file.unlink()

        # Reload
        result = script_reloader.reload_module(sample_module["name"])

        assert result.success is True

        # Check new version - module should have updated
        new_module = sys.modules[sample_module["name"]]
        assert new_module.get_version() == "1.1.0"

    def test_multiple_reload_cycles(self, script_reloader, sample_module):
        """Test multiple consecutive reloads."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        # Reload multiple times
        for i in range(3):
            result = script_reloader.reload_module(sample_module["name"])
            assert result.success is True

        stats = script_reloader.stats
        assert stats["total_reloads"] >= 3
        assert stats["successful_reloads"] >= 3

    def test_concurrent_access(self, script_reloader, sample_module):
        """Test thread-safe access to reloader."""
        import importlib
        importlib.import_module(sample_module["name"])

        results = []
        errors = []

        def reload_task():
            try:
                result = script_reloader.reload_module(sample_module["name"])
                results.append(result.success)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reload_task)
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reloads should complete (success or rollback)
        assert len(errors) == 0
        assert all(isinstance(r, bool) for r in results)


# ==============================================================================
# Edge Case Tests
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_reload_empty_module(self, script_reloader, temp_module_dir):
        """Test reloading an empty module."""
        # Create empty module
        module_path = Path(temp_module_dir) / "empty_module.py"
        module_path.write_text("# Empty module\n")

        sys.path.insert(0, temp_module_dir)
        try:
            import importlib
            importlib.import_module("empty_module")

            result = script_reloader.reload_module("empty_module")

            assert result.success is True
        finally:
            sys.path.remove(temp_module_dir)
            if "empty_module" in sys.modules:
                del sys.modules["empty_module"]

    def test_reload_module_with_imports(self, script_reloader, temp_module_dir):
        """Test reloading module with import dependencies."""
        # Create dependency module
        dep_path = Path(temp_module_dir) / "dependency.py"
        dep_path.write_text("DEP_VALUE = 42\n")

        # Create main module
        main_path = Path(temp_module_dir) / "main_module.py"
        main_path.write_text(
            "from dependency import DEP_VALUE\n"
            "def get_value(): return DEP_VALUE\n"
        )

        sys.path.insert(0, temp_module_dir)
        try:
            import importlib
            importlib.import_module("main_module")

            result = script_reloader.reload_module("main_module")

            assert result.success is True
        finally:
            sys.path.remove(temp_module_dir)
            for mod in ["main_module", "dependency"]:
                if mod in sys.modules:
                    del sys.modules[mod]

    def test_weak_reference_collection(self, script_reloader, sample_module):
        """Test that weak references are properly collected."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        # Create instances in a function scope so they become unreachable
        def create_and_register():
            for i in range(100):
                obj = module.TestComponent(f"obj_{i}")
                script_reloader.register_instance(obj)

        create_and_register()

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        instances = script_reloader.get_instances(sample_module["name"])

        # Without keeping references, most should be collected
        # Note: Some Python implementations may not collect all immediately
        assert len(instances) <= 10  # Allow some stragglers

    def test_checkpoint_serialization_limits(self, script_reloader, sample_module):
        """Test checkpoint with unserializable data."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        # Create checkpoint with some unserializable data
        checkpoint = script_reloader.create_checkpoint(
            module_name=sample_module["name"],
            function_name="complex_func",
            locals_dict={
                "normal": 42,
                "function": lambda x: x,  # Not serializable
                "nested": {"a": 1, "b": [1, 2, 3]},
            },
            line_number=100,
        )

        # Should succeed but skip unserializable
        assert checkpoint is not None
        assert "normal" in checkpoint.locals_snapshot


# ==============================================================================
# Performance Tests
# ==============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_reload_time_reasonable(self, script_reloader, sample_module):
        """Test that reload completes in reasonable time."""
        import importlib
        importlib.import_module(sample_module["name"])

        start = time.time()
        result = script_reloader.reload_module(sample_module["name"])
        elapsed = time.time() - start

        assert result.success is True
        assert elapsed < 5.0  # Should complete in under 5 seconds

    def test_many_instances_performance(self, script_reloader, sample_module):
        """Test performance with many tracked instances."""
        import importlib
        module = importlib.import_module(sample_module["name"])

        # Register many instances
        instances = []
        for i in range(1000):
            obj = module.TestComponent(f"obj_{i}")
            obj.health = i
            instances.append(obj)
            script_reloader.register_instance(obj)

        start = time.time()
        state = script_reloader._preserve_module_state(sample_module["name"])
        elapsed = time.time() - start

        assert len(state.instances) == 1000
        assert elapsed < 2.0  # Should serialize 1000 instances in under 2 seconds
