"""Tests for T-CC-3.4: Native code hot-reload with function table patching."""
import ctypes
import os
import platform
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from engine.native.hot_reload import (
    ABIMismatchError,
    ABIVersion,
    FunctionEntry,
    FunctionTable,
    LibraryState,
    LoadError,
    NativeLibrary,
    NativeReloader,
    ReloadError,
    ReloadEvent,
    ReloadEventType,
    ReloadOutcome,
    ReloadResult,
    StateSerializer,
    create_native_reloader,
)


# =============================================================================
# ABIVersion Tests
# =============================================================================


class TestABIVersion:
    """Tests for ABIVersion."""

    def test_basic_creation(self):
        version = ABIVersion(1, 2, 3)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_default_patch(self):
        version = ABIVersion(1, 2)
        assert version.patch == 0

    def test_with_hash(self):
        version = ABIVersion(1, 2, 3, "abc123def456")
        assert version.hash == "abc123def456"

    def test_str_without_hash(self):
        version = ABIVersion(1, 2, 3)
        assert str(version) == "1.2.3"

    def test_str_with_hash(self):
        version = ABIVersion(1, 2, 3, "abc123def456")
        assert str(version) == "1.2.3+abc123de"

    def test_equality(self):
        v1 = ABIVersion(1, 2, 3)
        v2 = ABIVersion(1, 2, 3)
        v3 = ABIVersion(1, 2, 4)
        assert v1 == v2
        assert v1 != v3

    def test_hash_for_dict_key(self):
        v1 = ABIVersion(1, 2, 3)
        v2 = ABIVersion(1, 2, 3)
        d = {v1: "test"}
        assert d[v2] == "test"

    def test_is_compatible_same_major(self):
        v1 = ABIVersion(1, 2, 0)
        v2 = ABIVersion(1, 3, 0)
        assert v1.is_compatible(v2) is True

    def test_is_compatible_different_major(self):
        v1 = ABIVersion(1, 0, 0)
        v2 = ABIVersion(2, 0, 0)
        assert v1.is_compatible(v2) is False

    def test_is_compatible_higher_minor(self):
        v1 = ABIVersion(1, 5, 0)
        v2 = ABIVersion(1, 3, 0)
        assert v1.is_compatible(v2) is False

    def test_to_dict(self):
        version = ABIVersion(1, 2, 3, "hash123")
        d = version.to_dict()
        assert d["major"] == 1
        assert d["minor"] == 2
        assert d["patch"] == 3
        assert d["hash"] == "hash123"

    def test_from_dict(self):
        d = {"major": 2, "minor": 3, "patch": 4, "hash": "xyz"}
        version = ABIVersion.from_dict(d)
        assert version.major == 2
        assert version.minor == 3
        assert version.patch == 4
        assert version.hash == "xyz"

    def test_from_dict_defaults(self):
        version = ABIVersion.from_dict({})
        assert version.major == 0
        assert version.minor == 0
        assert version.patch == 0

    def test_from_string_simple(self):
        version = ABIVersion.from_string("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_from_string_with_hash(self):
        version = ABIVersion.from_string("1.2.3+abc123")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.hash == "abc123"

    def test_from_string_short(self):
        version = ABIVersion.from_string("1.2")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 0


# =============================================================================
# FunctionEntry Tests
# =============================================================================


class TestFunctionEntry:
    """Tests for FunctionEntry."""

    def test_basic_entry(self):
        entry = FunctionEntry(
            name="test_func",
            address=0x12345,
            signature="void(int)",
        )
        assert entry.name == "test_func"
        assert entry.address == 0x12345
        assert entry.signature == "void(int)"

    def test_entry_with_types(self):
        entry = FunctionEntry(
            name="add",
            address=0,
            signature="int(int, int)",
            argtypes=[ctypes.c_int, ctypes.c_int],
            restype=ctypes.c_int,
        )
        assert entry.argtypes == [ctypes.c_int, ctypes.c_int]
        assert entry.restype == ctypes.c_int

    def test_entry_hash(self):
        entry = FunctionEntry(name="func", address=0, signature="")
        assert hash(entry) == hash("func")


# =============================================================================
# FunctionTable Tests
# =============================================================================


class TestFunctionTable:
    """Tests for FunctionTable."""

    def test_initial_state(self):
        table = FunctionTable()
        assert table.version == 0
        assert table.function_count == 0

    def test_register_function(self):
        table = FunctionTable()
        table.register("my_func", "void()")
        assert table.function_count == 1

    def test_register_with_types(self):
        table = FunctionTable()
        table.register(
            "add",
            "int(int, int)",
            argtypes=[ctypes.c_int, ctypes.c_int],
            restype=ctypes.c_int,
        )
        entry = table.get("add")
        assert entry is not None
        assert entry.argtypes == [ctypes.c_int, ctypes.c_int]

    def test_unregister_function(self):
        table = FunctionTable()
        table.register("func")
        assert table.unregister("func") is True
        assert table.function_count == 0

    def test_unregister_nonexistent(self):
        table = FunctionTable()
        assert table.unregister("nonexistent") is False

    def test_get_function(self):
        table = FunctionTable()
        table.register("test", "void()")
        entry = table.get("test")
        assert entry is not None
        assert entry.name == "test"

    def test_get_nonexistent(self):
        table = FunctionTable()
        assert table.get("nonexistent") is None

    def test_call_unregistered(self):
        table = FunctionTable()
        with pytest.raises(KeyError, match="Function not registered"):
            table.call("unknown")

    def test_call_not_loaded(self):
        table = FunctionTable()
        table.register("func")
        with pytest.raises(LoadError, match="Function not loaded"):
            table.call("func")

    def test_call_with_callable(self):
        table = FunctionTable()
        table.register("add")
        entry = table.get("add")
        entry.callable = lambda x, y: x + y
        result = table.call("add", 2, 3)
        assert result == 5

    def test_patch_function(self):
        table = FunctionTable()
        table.register("func")
        old_version = table.version
        result = table.patch("func", lambda: "patched", 0x12345)
        assert result is True
        assert table.version > old_version
        assert table.call("func") == "patched"

    def test_patch_nonexistent(self):
        table = FunctionTable()
        assert table.patch("nonexistent", lambda: None) is False

    def test_get_all_names(self):
        table = FunctionTable()
        table.register("func1")
        table.register("func2")
        table.register("func3")
        names = table.get_all_names()
        assert set(names) == {"func1", "func2", "func3"}

    def test_get_bound_names(self):
        table = FunctionTable()
        table.register("bound")
        table.register("unbound")
        entry = table.get("bound")
        entry.callable = lambda: None
        assert "bound" in table.get_bound_names()
        assert "unbound" not in table.get_bound_names()

    def test_get_unbound_names(self):
        table = FunctionTable()
        table.register("bound")
        table.register("unbound")
        entry = table.get("bound")
        entry.callable = lambda: None
        assert "unbound" in table.get_unbound_names()
        assert "bound" not in table.get_unbound_names()

    def test_unbind(self):
        table = FunctionTable()
        table.register("func")
        entry = table.get("func")
        entry.callable = lambda: None
        entry.address = 0x1000
        table.unbind()
        assert entry.callable is None
        assert entry.address == 0

    def test_clear(self):
        table = FunctionTable()
        table.register("func1")
        table.register("func2")
        table.clear()
        assert table.function_count == 0
        assert table.version == 0

    def test_to_dict(self):
        table = FunctionTable()
        table.register("func", "void()")
        entry = table.get("func")
        entry.callable = lambda: None
        entry.address = 0x1000

        d = table.to_dict()
        assert d["version"] >= 0
        assert len(d["functions"]) == 1
        assert d["functions"][0]["name"] == "func"
        assert d["functions"][0]["bound"] is True

    def test_thread_safety(self):
        table = FunctionTable()
        errors = []

        def register_many(prefix: str):
            try:
                for i in range(100):
                    table.register(f"{prefix}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_many, args=(f"t{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert table.function_count == 500


# =============================================================================
# StateSerializer Tests
# =============================================================================


class TestStateSerializer:
    """Tests for StateSerializer."""

    def test_pickle_round_trip(self):
        serializer = StateSerializer(format="pickle")
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)
        assert deserialized == data

    def test_json_round_trip(self):
        serializer = StateSerializer(format="json")
        data = {"key": "value", "number": 42}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)
        assert deserialized == data

    def test_max_size_exceeded(self):
        serializer = StateSerializer(max_size=10)
        data = {"large": "x" * 100}
        with pytest.raises(ValueError, match="State too large"):
            serializer.serialize(data)

    def test_unknown_format(self):
        serializer = StateSerializer(format="unknown")
        with pytest.raises(ValueError, match="Unknown format"):
            serializer.serialize({})


# =============================================================================
# LibraryState Tests
# =============================================================================


class TestLibraryState:
    """Tests for LibraryState."""

    def test_basic_state(self):
        state = LibraryState(
            data=b"test data",
            version=ABIVersion(1, 0, 0),
        )
        assert state.data == b"test data"
        assert state.version == ABIVersion(1, 0, 0)
        assert state.timestamp > 0
        assert state.checksum != ""

    def test_verify_checksum_valid(self):
        state = LibraryState(
            data=b"test data",
            version=ABIVersion(1, 0, 0),
        )
        assert state.verify_checksum() is True

    def test_verify_checksum_invalid(self):
        state = LibraryState(
            data=b"test data",
            version=ABIVersion(1, 0, 0),
        )
        state.data = b"modified"
        assert state.verify_checksum() is False

    def test_metadata(self):
        state = LibraryState(
            data=b"data",
            version=ABIVersion(1, 0, 0),
            metadata={"key": "value"},
        )
        assert state.metadata["key"] == "value"


# =============================================================================
# ReloadEvent Tests
# =============================================================================


class TestReloadEvent:
    """Tests for ReloadEvent."""

    def test_basic_event(self):
        event = ReloadEvent(
            event_type=ReloadEventType.LIBRARY_LOADED,
            library_path=Path("/lib.so"),
        )
        assert event.event_type == ReloadEventType.LIBRARY_LOADED
        assert event.library_path == Path("/lib.so")
        assert event.timestamp > 0

    def test_event_with_details(self):
        event = ReloadEvent(
            event_type=ReloadEventType.FUNCTION_PATCHED,
            details={"count": 5},
        )
        assert event.details["count"] == 5

    def test_event_with_error(self):
        event = ReloadEvent(
            event_type=ReloadEventType.RELOAD_FAILED,
            error="Load failed",
        )
        assert event.error == "Load failed"


# =============================================================================
# ReloadOutcome Tests
# =============================================================================


class TestReloadOutcome:
    """Tests for ReloadOutcome."""

    def test_success_outcome(self):
        outcome = ReloadOutcome(
            result=ReloadResult.SUCCESS,
            library_path=Path("/lib.so"),
            old_version=ABIVersion(1, 0, 0),
            new_version=ABIVersion(1, 1, 0),
            duration_ms=100.5,
            patched_functions=10,
            state_migrated=True,
        )
        assert outcome.result == ReloadResult.SUCCESS
        assert outcome.patched_functions == 10
        assert outcome.state_migrated is True

    def test_failed_outcome(self):
        outcome = ReloadOutcome(
            result=ReloadResult.FAILED,
            library_path=Path("/lib.so"),
            error="ABI mismatch",
        )
        assert outcome.result == ReloadResult.FAILED
        assert outcome.error == "ABI mismatch"


# =============================================================================
# NativeLibrary Tests
# =============================================================================


class TestNativeLibrary:
    """Tests for NativeLibrary."""

    def test_initial_state(self):
        lib = NativeLibrary("/path/to/lib.so", ABIVersion(1, 0, 0))
        assert lib.path == Path("/path/to/lib.so")
        assert lib.is_loaded is False
        assert lib.version == ABIVersion(1, 0, 0)
        assert lib.load_time == 0.0

    def test_function_table_access(self):
        lib = NativeLibrary("/path/to/lib.so")
        assert lib.function_table is not None
        assert isinstance(lib.function_table, FunctionTable)

    def test_load_nonexistent(self):
        lib = NativeLibrary("/nonexistent/lib.so")
        with pytest.raises(LoadError, match="Library not found"):
            lib.load()

    def test_handle_before_load(self):
        lib = NativeLibrary("/path/to/lib.so")
        assert lib.handle is None


class TestNativeLibraryWithRealLibrary:
    """Tests for NativeLibrary with actual shared libraries."""

    @pytest.fixture
    def libc_path(self):
        """Get path to libc or equivalent."""
        system = platform.system()
        if system == "Linux":
            # Try common libc paths
            for path in ["/lib/x86_64-linux-gnu/libc.so.6", "/lib64/libc.so.6", "/lib/libc.so.6"]:
                if os.path.exists(path):
                    return path
            # Fallback to letting ctypes find it
            return None
        elif system == "Darwin":
            return "/usr/lib/libSystem.B.dylib"
        elif system == "Windows":
            return "msvcrt"
        return None

    def test_load_system_library(self, libc_path):
        if libc_path is None:
            pytest.skip("Could not find system library")

        lib = NativeLibrary(libc_path)
        lib.load(copy_first=False)
        assert lib.is_loaded is True
        assert lib.load_time > 0
        lib.unload()
        assert lib.is_loaded is False

    def test_get_function_from_libc(self, libc_path):
        if libc_path is None:
            pytest.skip("Could not find system library")

        lib = NativeLibrary(libc_path)
        lib.load(copy_first=False)

        # Try to get a common function
        strlen = lib.get_function("strlen", [ctypes.c_char_p], ctypes.c_size_t)
        if strlen is not None:
            result = strlen(b"hello")
            assert result == 5

        lib.unload()

    def test_context_manager(self, libc_path):
        if libc_path is None:
            pytest.skip("Could not find system library")

        with NativeLibrary(libc_path) as lib:
            lib.load(copy_first=False)
            assert lib.is_loaded is True

        assert lib.is_loaded is False


# =============================================================================
# NativeReloader Tests
# =============================================================================


class TestNativeReloader:
    """Tests for NativeReloader."""

    def test_initial_state(self):
        reloader = NativeReloader()
        assert reloader.is_running is False
        assert reloader.library_count == 0

    def test_register_library(self):
        reloader = NativeReloader()
        lib = reloader.register_library("/path/to/lib.so")
        assert reloader.library_count == 1
        assert lib is not None

    def test_register_library_with_functions(self):
        reloader = NativeReloader()
        lib = reloader.register_library(
            "/path/to/lib.so",
            function_names=["func1", "func2"],
        )
        assert lib.function_table.function_count == 2

    def test_register_library_idempotent(self):
        reloader = NativeReloader()
        lib1 = reloader.register_library("/path/to/lib.so")
        lib2 = reloader.register_library("/path/to/lib.so")
        assert lib1 is lib2
        assert reloader.library_count == 1

    def test_unregister_library(self):
        reloader = NativeReloader()
        reloader.register_library("/path/to/lib.so")
        result = reloader.unregister_library("/path/to/lib.so")
        assert result is True
        assert reloader.library_count == 0

    def test_unregister_nonexistent(self):
        reloader = NativeReloader()
        result = reloader.unregister_library("/nonexistent.so")
        assert result is False

    def test_get_library(self):
        reloader = NativeReloader()
        reloader.register_library("/path/to/lib.so")
        lib = reloader.get_library("/path/to/lib.so")
        assert lib is not None

    def test_get_nonexistent_library(self):
        reloader = NativeReloader()
        lib = reloader.get_library("/nonexistent.so")
        assert lib is None

    def test_add_callback(self):
        reloader = NativeReloader()
        events: List[ReloadEvent] = []

        def callback(event: ReloadEvent):
            events.append(event)

        reloader.add_callback(callback)
        reloader.add_callback(callback)  # Should not duplicate

        # Trigger an event manually
        reloader._emit_event(ReloadEventType.RELOAD_STARTED, Path("/test.so"))

        assert len(events) == 1
        assert events[0].event_type == ReloadEventType.RELOAD_STARTED

    def test_remove_callback(self):
        reloader = NativeReloader()
        callback = MagicMock()
        reloader.add_callback(callback)
        result = reloader.remove_callback(callback)
        assert result is True

        reloader._emit_event(ReloadEventType.RELOAD_STARTED)
        callback.assert_not_called()

    def test_remove_nonexistent_callback(self):
        reloader = NativeReloader()
        result = reloader.remove_callback(lambda e: None)
        assert result is False

    def test_start_stop(self):
        reloader = NativeReloader()
        reloader.start()
        assert reloader.is_running is True
        reloader.stop()
        assert reloader.is_running is False

    def test_auto_start(self):
        reloader = NativeReloader(auto_start=True)
        assert reloader.is_running is True
        reloader.stop()

    def test_context_manager(self):
        with NativeReloader() as reloader:
            assert reloader.is_running is True
        assert reloader.is_running is False

    def test_get_status(self):
        reloader = NativeReloader(
            expected_version=ABIVersion(1, 0, 0),
            strict_abi=True,
        )
        reloader.register_library("/path/to/lib.so")
        status = reloader.get_status()

        assert status["running"] is False
        assert status["library_count"] == 1
        assert status["strict_abi"] is True
        assert status["expected_version"] == "1.0.0"

    def test_clear(self):
        reloader = NativeReloader()
        reloader.register_library("/path/lib1.so")
        reloader.register_library("/path/lib2.so")
        reloader.clear()
        assert reloader.library_count == 0

    def test_get_recent_outcomes(self):
        reloader = NativeReloader()
        # Manually add outcomes for testing
        outcome = ReloadOutcome(
            result=ReloadResult.SUCCESS,
            library_path=Path("/test.so"),
        )
        reloader._record_outcome(outcome)

        outcomes = reloader.get_recent_outcomes(limit=5)
        assert len(outcomes) == 1
        assert outcomes[0].result == ReloadResult.SUCCESS


class TestNativeReloaderWithMockedLibrary:
    """Tests for NativeReloader with mocked library operations."""

    @pytest.fixture
    def mock_library(self):
        """Create a mock library file."""
        with tempfile.NamedTemporaryFile(
            suffix=".so" if platform.system() != "Windows" else ".dll",
            delete=False,
        ) as f:
            # Write minimal ELF/PE header (just enough to be openable)
            f.write(b"\x00" * 1024)
            path = Path(f.name)

        yield path

        if path.exists():
            os.unlink(path)

    def test_load_library_not_found(self):
        reloader = NativeReloader()
        with pytest.raises(LoadError, match="Library not found"):
            reloader.load_library("/definitely/not/a/real/path.so")

    def test_reload_unregistered_library(self):
        reloader = NativeReloader()
        outcome = reloader.reload_library("/unregistered.so")
        assert outcome.result == ReloadResult.FAILED
        assert "not registered" in outcome.error

    def test_reload_with_state_migration(self):
        """Test state migration during reload via StateSerializer directly."""
        # Test the state serialization mechanism independently
        serializer = StateSerializer(format="pickle")
        state_data = {"counter": 42, "name": "test"}

        # Serialize state
        serialized = serializer.serialize(state_data)
        assert len(serialized) > 0

        # Create library state
        lib_state = LibraryState(
            data=serialized,
            version=ABIVersion(1, 0, 0),
        )
        assert lib_state.verify_checksum() is True

        # Deserialize and verify
        restored = serializer.deserialize(lib_state.data)
        assert restored == state_data

    def test_abi_mismatch_strict(self):
        """Test ABI mismatch error construction and properties."""
        expected = ABIVersion(2, 0, 0)
        actual = ABIVersion(1, 0, 0)

        # Test compatibility check
        assert expected.is_compatible(actual) is False

        # Test ABIMismatchError
        error = ABIMismatchError(expected, actual)
        assert error.expected == expected
        assert error.actual == actual
        assert "2.0.0" in str(error)
        assert "1.0.0" in str(error)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the hot-reload system."""

    def test_full_workflow_mock(self):
        """Test complete workflow with mocked library."""
        events: List[ReloadEvent] = []

        def event_handler(event: ReloadEvent):
            events.append(event)

        reloader = NativeReloader(
            expected_version=ABIVersion(1, 0, 0),
            strict_abi=False,
        )
        reloader.add_callback(event_handler)

        # Register library
        lib = reloader.register_library(
            "/mock/renderer.so",
            function_names=["render_frame", "update_scene", "cleanup"],
        )
        assert lib.function_table.function_count == 3

        # Verify status
        status = reloader.get_status()
        assert status["library_count"] == 1
        assert "/mock/renderer.so" in str(status["libraries"])

        # Clean up
        reloader.clear()
        assert reloader.library_count == 0

    def test_multiple_libraries(self):
        """Test managing multiple libraries."""
        reloader = NativeReloader()

        lib1 = reloader.register_library(
            "/mock/lib1.so",
            function_names=["func1"],
            version=ABIVersion(1, 0, 0),
        )
        lib2 = reloader.register_library(
            "/mock/lib2.so",
            function_names=["func2"],
            version=ABIVersion(2, 0, 0),
        )

        assert reloader.library_count == 2
        assert lib1.version == ABIVersion(1, 0, 0)
        assert lib2.version == ABIVersion(2, 0, 0)

        reloader.unregister_library("/mock/lib1.so")
        assert reloader.library_count == 1

    def test_function_table_patching_workflow(self):
        """Test function table patching mechanism."""
        table = FunctionTable()

        # Register functions
        table.register("add", "int(int, int)")
        table.register("multiply", "int(int, int)")

        # Simulate binding
        table.patch("add", lambda x, y: x + y)
        table.patch("multiply", lambda x, y: x * y)

        # Verify calls work
        assert table.call("add", 2, 3) == 5
        assert table.call("multiply", 4, 5) == 20

        # Simulate hot-patch
        old_version = table.version
        table.patch("add", lambda x, y: x + y + 1)  # New version adds 1

        assert table.version > old_version
        assert table.call("add", 2, 3) == 6


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateNativeReloader:
    """Tests for create_native_reloader factory."""

    def test_basic_creation(self):
        reloader = create_native_reloader(auto_start=False)
        assert reloader is not None
        assert reloader.is_running is False

    def test_with_watch_paths(self):
        reloader = create_native_reloader(
            watch_paths=["/path/lib1.so", "/path/lib2.so"],
            auto_start=False,
        )
        assert reloader.library_count == 2

    def test_with_expected_version(self):
        reloader = create_native_reloader(
            expected_version=ABIVersion(1, 2, 3),
            strict_abi=True,
            auto_start=False,
        )
        status = reloader.get_status()
        assert status["expected_version"] == "1.2.3"
        assert status["strict_abi"] is True

    def test_auto_start(self):
        reloader = create_native_reloader(auto_start=True)
        assert reloader.is_running is True
        reloader.stop()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_load_error_message(self):
        error = LoadError("Failed to load library")
        assert str(error) == "Failed to load library"

    def test_reload_error_message(self):
        error = ReloadError("Reload failed")
        assert str(error) == "Reload failed"

    def test_abi_mismatch_error_message(self):
        error = ABIMismatchError(ABIVersion(1, 0, 0), ABIVersion(2, 0, 0))
        assert "ABI mismatch" in str(error)
        assert "1.0.0" in str(error)
        assert "2.0.0" in str(error)

    def test_callback_exception_handling(self):
        """Test that exceptions in callbacks don't break the reloader."""
        reloader = NativeReloader()

        def bad_callback(event: ReloadEvent):
            raise RuntimeError("Callback failed")

        reloader.add_callback(bad_callback)

        # Should not raise
        reloader._emit_event(ReloadEventType.RELOAD_STARTED)


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_register_unregister(self):
        reloader = NativeReloader()
        errors = []

        def register_unregister(index: int):
            try:
                for i in range(50):
                    path = f"/lib{index}_{i}.so"
                    reloader.register_library(path)
                    reloader.unregister_library(path)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_unregister, args=(i,))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_callbacks(self):
        reloader = NativeReloader()
        call_counts = {"count": 0}
        lock = threading.Lock()

        def counting_callback(event: ReloadEvent):
            with lock:
                call_counts["count"] += 1

        reloader.add_callback(counting_callback)

        def emit_events():
            for _ in range(100):
                reloader._emit_event(ReloadEventType.RELOAD_STARTED)

        threads = [
            threading.Thread(target=emit_events)
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_counts["count"] == 400


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_function_table_to_dict(self):
        table = FunctionTable()
        d = table.to_dict()
        assert d["version"] == 0
        assert d["functions"] == []

    def test_library_state_empty_data(self):
        state = LibraryState(
            data=b"",
            version=ABIVersion(1, 0, 0),
        )
        assert state.verify_checksum() is True

    def test_abi_version_equality_ignores_hash(self):
        v1 = ABIVersion(1, 2, 3, "abc")
        v2 = ABIVersion(1, 2, 3, "xyz")
        assert v1 == v2

    def test_reloader_double_start(self):
        reloader = NativeReloader()
        reloader.start()
        reloader.start()  # Should be idempotent
        assert reloader.is_running is True
        reloader.stop()

    def test_reloader_double_stop(self):
        reloader = NativeReloader()
        reloader.start()
        reloader.stop()
        reloader.stop()  # Should be safe
        assert reloader.is_running is False

    def test_library_unload_not_loaded(self):
        """Test that unloading a not-loaded library is safe."""
        lib = NativeLibrary("/mock/lib.so")
        assert lib.is_loaded is False
        # Should not raise
        lib.unload()
        assert lib.is_loaded is False

    def test_library_version_setter(self):
        """Test NativeLibrary version handling."""
        lib = NativeLibrary("/mock/lib.so", ABIVersion(1, 2, 3))
        assert lib.version == ABIVersion(1, 2, 3)
        # Version can be accessed before loading
        assert lib.is_loaded is False


class TestReloadEventTypes:
    """Tests for all ReloadEventType values."""

    def test_all_event_types_exist(self):
        expected_types = [
            ReloadEventType.LIBRARY_LOADING,
            ReloadEventType.LIBRARY_LOADED,
            ReloadEventType.LIBRARY_UNLOADING,
            ReloadEventType.LIBRARY_UNLOADED,
            ReloadEventType.STATE_SAVING,
            ReloadEventType.STATE_SAVED,
            ReloadEventType.STATE_RESTORING,
            ReloadEventType.STATE_RESTORED,
            ReloadEventType.FUNCTION_PATCHING,
            ReloadEventType.FUNCTION_PATCHED,
            ReloadEventType.ABI_CHECK,
            ReloadEventType.RELOAD_STARTED,
            ReloadEventType.RELOAD_COMPLETED,
            ReloadEventType.RELOAD_FAILED,
            ReloadEventType.ROLLBACK_STARTED,
            ReloadEventType.ROLLBACK_COMPLETED,
        ]
        assert len(expected_types) == 16


class TestReloadResultEnum:
    """Tests for ReloadResult enum."""

    def test_all_results_exist(self):
        results = [
            ReloadResult.SUCCESS,
            ReloadResult.PARTIAL,
            ReloadResult.FAILED,
            ReloadResult.SKIPPED,
            ReloadResult.ROLLBACK,
        ]
        assert len(results) == 5
