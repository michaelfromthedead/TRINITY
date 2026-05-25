"""
Tests for state preservation functionality.
"""
import pytest
import time
from dataclasses import dataclass

from engine.tooling.hotreload.state_preservation import (
    StatePreserver,
    PreservationStrategy,
    StateSnapshot,
    PreservationConfig,
)


class TestPreservationStrategy:
    """Tests for PreservationStrategy enum."""

    def test_all_strategies_exist(self):
        """Test all expected strategies exist."""
        assert hasattr(PreservationStrategy, "SERIALIZER")
        assert hasattr(PreservationStrategy, "MIRROR")
        assert hasattr(PreservationStrategy, "PICKLE_PROTOCOL")
        assert hasattr(PreservationStrategy, "CUSTOM")
        assert hasattr(PreservationStrategy, "NONE")
        assert hasattr(PreservationStrategy, "SHALLOW_COPY")
        assert hasattr(PreservationStrategy, "DEEP_COPY")


class TestStateSnapshot:
    """Tests for StateSnapshot."""

    def test_snapshot_creation(self):
        """Test creating a state snapshot."""
        snapshot = StateSnapshot(
            obj_id=12345,
            class_name="TestClass",
            module_name="test_module",
            schema_hash="abc123",
            timestamp=time.time(),
            state={"x": 10, "y": "test"},
            strategy=PreservationStrategy.SERIALIZER,
        )

        assert snapshot.obj_id == 12345
        assert snapshot.class_name == "TestClass"
        assert snapshot.schema_hash == "abc123"
        assert "x" in snapshot.state

    def test_snapshot_age(self):
        """Test snapshot age calculation."""
        old_time = time.time() - 10
        snapshot = StateSnapshot(
            obj_id=1,
            class_name="Test",
            module_name="test",
            schema_hash="abc",
            timestamp=old_time,
            state={},
            strategy=PreservationStrategy.SERIALIZER,
        )

        assert snapshot.age() >= 10

    def test_snapshot_is_stale(self):
        """Test stale detection."""
        old_time = time.time() - 100
        snapshot = StateSnapshot(
            obj_id=1,
            class_name="Test",
            module_name="test",
            schema_hash="abc",
            timestamp=old_time,
            state={},
            strategy=PreservationStrategy.SERIALIZER,
        )

        assert snapshot.is_stale(max_age=60.0) is True
        assert snapshot.is_stale(max_age=200.0) is False


class TestPreservationConfig:
    """Tests for PreservationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PreservationConfig()

        assert config.strategy == PreservationStrategy.SERIALIZER
        assert config.include_fields is None
        assert config.exclude_fields == set()
        assert config.max_depth == 10
        assert config.preserve_refs is True
        assert config.validate_types is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = PreservationConfig(
            strategy=PreservationStrategy.MIRROR,
            include_fields={"x", "y"},
            exclude_fields={"_internal"},
            max_depth=5,
        )

        assert config.strategy == PreservationStrategy.MIRROR
        assert config.include_fields == {"x", "y"}
        assert "_internal" in config.exclude_fields


class TestStatePreserver:
    """Tests for StatePreserver."""

    def setup_method(self):
        """Create fresh preserver for each test."""
        self.preserver = StatePreserver()

    def test_preserver_initialization(self):
        """Test StatePreserver initializes correctly."""
        assert self.preserver is not None

    def test_preserve_simple_object(self):
        """Test preserving a simple object's state."""
        class SimpleClass:
            def __init__(self):
                self.x = 10
                self.y = "test"

        obj = SimpleClass()
        snapshot = self.preserver.preserve(obj)

        assert snapshot.class_name == "SimpleClass"
        assert "x" in snapshot.state
        assert "y" in snapshot.state

    def test_preserve_with_dataclass(self):
        """Test preserving a dataclass."""
        @dataclass
        class DataClass:
            x: int = 10
            y: str = "test"

        obj = DataClass()
        snapshot = self.preserver.preserve(obj)

        assert "x" in snapshot.state
        assert snapshot.state.get("x") == 10

    def test_preserve_with_mirror_strategy(self):
        """Test preserving with MIRROR strategy."""
        class TestClass:
            x: int = 10
            y: str = "test"

            def __init__(self):
                self.x = 10
                self.y = "test"

        obj = TestClass()
        snapshot = self.preserver.preserve(
            obj,
            strategy=PreservationStrategy.MIRROR,
        )

        assert snapshot.strategy == PreservationStrategy.MIRROR
        assert "x" in snapshot.state

    def test_preserve_with_none_strategy(self):
        """Test preserving with NONE strategy returns empty state."""
        class TestClass:
            def __init__(self):
                self.x = 10

        obj = TestClass()
        snapshot = self.preserver.preserve(
            obj,
            strategy=PreservationStrategy.NONE,
        )

        assert snapshot.state == {}

    def test_restore_simple_object(self):
        """Test restoring state to an object."""
        class TestClass:
            def __init__(self):
                self.x = 10
                self.y = "original"

        obj = TestClass()

        # Preserve current state
        snapshot = self.preserver.preserve(obj)

        # Modify object
        obj.x = 99
        obj.y = "modified"

        # Restore
        result = self.preserver.restore(obj, snapshot)

        assert result is True
        assert obj.x == 10
        assert obj.y == "original"

    def test_restore_without_snapshot(self):
        """Test restore with no snapshot returns False."""
        class TestClass:
            def __init__(self):
                self.x = 10

        obj = TestClass()
        result = self.preserver.restore(obj)

        assert result is False

    def test_get_snapshot(self):
        """Test getting a snapshot by object."""
        class TestClass:
            def __init__(self):
                self.x = 10

        obj = TestClass()
        self.preserver.preserve(obj)

        snapshot = self.preserver.get_snapshot(obj)
        assert snapshot is not None
        assert snapshot.obj_id == id(obj)

    def test_has_snapshot(self):
        """Test checking for snapshot existence."""
        class TestClass:
            def __init__(self):
                self.x = 10

        obj = TestClass()
        obj2 = TestClass()

        self.preserver.preserve(obj)

        assert self.preserver.has_snapshot(obj) is True
        assert self.preserver.has_snapshot(obj2) is False

    def test_clear_snapshot(self):
        """Test clearing a snapshot."""
        class TestClass:
            def __init__(self):
                self.x = 10

        obj = TestClass()
        self.preserver.preserve(obj)

        assert self.preserver.has_snapshot(obj) is True

        result = self.preserver.clear_snapshot(obj)

        assert result is True
        assert self.preserver.has_snapshot(obj) is False

    def test_clear_all(self):
        """Test clearing all snapshots."""
        class TestClass:
            def __init__(self):
                self.x = 10

        obj1 = TestClass()
        obj2 = TestClass()

        self.preserver.preserve(obj1)
        self.preserver.preserve(obj2)

        count = self.preserver.clear_all()

        assert count == 2
        assert self.preserver.has_snapshot(obj1) is False
        assert self.preserver.has_snapshot(obj2) is False

    def test_configure_class(self):
        """Test configuring preservation for a class."""
        config = PreservationConfig(
            strategy=PreservationStrategy.MIRROR,
            exclude_fields={"_private"},
        )

        self.preserver.configure("test_module.TestClass", config)

        retrieved = self.preserver.get_config("test_module.TestClass")
        assert retrieved.strategy == PreservationStrategy.MIRROR
        assert "_private" in retrieved.exclude_fields

    def test_exclude_fields(self):
        """Test that excluded fields are not preserved."""
        class TestClass:
            def __init__(self):
                self.public = 10
                self._private = "secret"

        obj = TestClass()

        config = PreservationConfig(
            exclude_fields={"_private"},
        )
        self.preserver.configure(
            f"{TestClass.__module__}.TestClass",
            config,
        )

        snapshot = self.preserver.preserve(obj)

        assert "public" in snapshot.state
        assert "_private" not in snapshot.state

    def test_include_fields(self):
        """Test that only included fields are preserved."""
        class TestClass:
            def __init__(self):
                self.x = 10
                self.y = 20
                self.z = 30

        obj = TestClass()

        config = PreservationConfig(
            include_fields={"x", "y"},
        )
        self.preserver.configure(
            f"{TestClass.__module__}.TestClass",
            config,
        )

        snapshot = self.preserver.preserve(obj)

        assert "x" in snapshot.state
        assert "y" in snapshot.state
        assert "z" not in snapshot.state


class TestCustomPreservation:
    """Tests for custom preservation strategies."""

    def setup_method(self):
        self.preserver = StatePreserver()

    def test_pickle_protocol_getstate(self):
        """Test using __getstate__ for preservation."""
        class CustomClass:
            def __init__(self):
                self.x = 10
                self._cache = {}  # Should not be saved

            def __getstate__(self):
                return {"x": self.x}

            def __setstate__(self, state):
                self.x = state["x"]
                self._cache = {}

        obj = CustomClass()
        obj.x = 42

        snapshot = self.preserver.preserve(
            obj,
            strategy=PreservationStrategy.PICKLE_PROTOCOL,
        )

        assert snapshot.state.get("x") == 42
        assert "_cache" not in snapshot.state

    def test_custom_preserve_restore(self):
        """Test using __preserve__/__restore__ methods."""
        class CustomClass:
            def __init__(self):
                self.x = 10

            def __preserve__(self):
                return {"custom_x": self.x * 2}

            def __restore__(self, state):
                self.x = state["custom_x"] // 2

        obj = CustomClass()
        obj.x = 20

        snapshot = self.preserver.preserve(
            obj,
            strategy=PreservationStrategy.CUSTOM,
        )

        assert "custom_x" in snapshot.state
        assert snapshot.state["custom_x"] == 40

        obj.x = 0
        self.preserver.restore(obj, snapshot)
        assert obj.x == 20
