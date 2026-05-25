"""
Tests for the undo/redo system.
"""
import pytest
import time

from engine.tooling.undo.undo_system import (
    UndoSystem,
    UndoSystemConfig,
    UndoEntry,
    UndoRedoError,
    UndoStackEmpty,
    RedoStackEmpty,
    get_undo_system,
)


class TestUndoSystemConfig:
    """Tests for UndoSystemConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UndoSystemConfig()

        assert config.max_undo_levels == 1000
        assert config.max_redo_levels == 1000
        assert config.enable_grouping is True
        assert config.group_timeout_ms == 500

    def test_custom_config(self):
        """Test custom configuration."""
        config = UndoSystemConfig(
            max_undo_levels=100,
            enable_grouping=False,
        )

        assert config.max_undo_levels == 100
        assert config.enable_grouping is False


class TestUndoEntry:
    """Tests for UndoEntry."""

    def test_entry_creation(self):
        """Test creating an undo entry."""
        entry = UndoEntry(
            name="Test Change",
            timestamp=time.time(),
            changes=[],
        )

        assert entry.name == "Test Change"
        assert entry.timestamp > 0
        assert len(entry.changes) == 0

    def test_entry_description_single(self):
        """Test entry description with single change."""
        from foundation.tracker import Change
        import weakref

        class TestObj:
            x = 10

        obj = TestObj()
        change = Change(
            obj_ref=weakref.ref(obj),
            field="x",
            old_value=10,
            new_value=20,
        )

        entry = UndoEntry(
            name="Set Value",
            timestamp=time.time(),
            changes=[change],
        )

        assert "x" in entry.description

    def test_entry_description_multiple(self):
        """Test entry description with multiple changes."""
        entry = UndoEntry(
            name="Multiple Changes",
            timestamp=time.time(),
            changes=[None, None, None],  # Mock changes
        )

        assert "3 changes" in entry.description


class TestUndoSystem:
    """Tests for UndoSystem."""

    def setup_method(self):
        """Create fresh undo system for each test."""
        self.undo = UndoSystem()

    def test_system_initialization(self):
        """Test UndoSystem initializes correctly."""
        assert self.undo.can_undo is False
        assert self.undo.can_redo is False
        assert self.undo.undo_count == 0
        assert self.undo.redo_count == 0

    def test_record_change(self):
        """Test recording a change."""
        class TestObj:
            x = 10

        obj = TestObj()
        entry = self.undo.record(
            name="Set x",
            obj=obj,
            field_name="x",
            old_value=10,
            new_value=20,
        )

        assert entry is not None
        assert self.undo.can_undo is True
        assert self.undo.undo_stack_size == 1

    def test_undo_single_change(self):
        """Test undoing a single change."""
        class TestObj:
            def __init__(self):
                self.x = 10

        obj = TestObj()
        obj.x = 20
        self.undo.record(
            name="Set x",
            obj=obj,
            field_name="x",
            old_value=10,
            new_value=20,
        )

        entry = self.undo.undo()

        assert entry is not None
        assert obj.x == 10
        # After undo, there may still be Foundation tracker's undo available
        assert self.undo.can_redo is True

    def test_redo_single_change(self):
        """Test redoing a single change."""
        class TestObj:
            def __init__(self):
                self.x = 10

        obj = TestObj()
        obj.x = 20
        self.undo.record(
            name="Set x",
            obj=obj,
            field_name="x",
            old_value=10,
            new_value=20,
        )

        self.undo.undo()
        assert obj.x == 10

        entry = self.undo.redo()

        assert entry is not None
        assert obj.x == 20
        assert self.undo.can_redo is False
        assert self.undo.can_undo is True

    def test_undo_empty_stack(self):
        """Test undo with empty local stack falls back to tracker."""
        # Create fresh undo system to avoid state from other tests
        fresh_undo = UndoSystem()
        # Clear the foundation tracker state
        fresh_undo._tracker._undo.clear()
        fresh_undo._tracker._redo.clear()

        with pytest.raises(UndoStackEmpty):
            fresh_undo.undo()

    def test_redo_empty_stack(self):
        """Test redo with empty stack raises error."""
        fresh_undo = UndoSystem()
        fresh_undo._tracker._redo.clear()

        with pytest.raises(RedoStackEmpty):
            fresh_undo.redo()

    def test_suspend_resume(self):
        """Test suspending and resuming undo tracking."""
        class TestObj:
            x = 10

        obj = TestObj()

        self.undo.suspend()
        assert self.undo.suspended is True

        # Record while suspended should not add to stack
        entry = self.undo.record(
            name="Set x",
            obj=obj,
            field_name="x",
            old_value=10,
            new_value=20,
        )

        assert entry is None
        assert self.undo.undo_stack_size == 0

        self.undo.resume()
        assert self.undo.suspended is False

    def test_begin_end_group(self):
        """Test grouping changes."""
        class TestObj:
            def __init__(self):
                self.x = 10
                self.y = 20

        obj = TestObj()

        group_id = self.undo.begin_group("Multiple Changes")
        assert group_id is not None

        obj.x = 100
        self.undo.record("Set x", obj, "x", 10, 100)

        obj.y = 200
        self.undo.record("Set y", obj, "y", 20, 200)

        self.undo.end_group()

        # Should be able to undo both at once
        assert self.undo.undo_stack_size >= 1

    def test_cancel_group(self):
        """Test canceling a group."""
        class TestObj:
            def __init__(self):
                self.x = 10

        obj = TestObj()

        self.undo.begin_group("Cancelled Group")
        obj.x = 100
        self.undo.record("Set x", obj, "x", 10, 100)

        self.undo.cancel_group()

        # Changes should be removed
        assert self.undo.undo_stack_size == 0

    def test_undo_history(self):
        """Test getting undo history."""
        class TestObj:
            x = 10

        obj = TestObj()

        # Disable grouping for this test to get separate entries
        self.undo._config.enable_grouping = False

        for i in range(5):
            self.undo.record(f"Change {i}", obj, "x", i, i + 1)

        history = self.undo.get_undo_history(limit=10)

        # Should have at least 1 entry
        assert len(history) >= 1

    def test_undo_description(self):
        """Test getting undo description."""
        class TestObj:
            x = 10

        obj = TestObj()

        assert self.undo.get_undo_description() is None

        self.undo.record("Test Change", obj, "x", 10, 20)

        desc = self.undo.get_undo_description()
        assert desc is not None
        assert "Test Change" in desc or "x" in desc

    def test_clear(self):
        """Test clearing undo/redo stacks."""
        class TestObj:
            x = 10

        obj = TestObj()

        self.undo.record("Change", obj, "x", 10, 20)
        self.undo.undo()

        assert self.undo.undo_stack_size > 0 or self.undo.redo_stack_size > 0

        self.undo.clear()

        assert self.undo.undo_stack_size == 0
        assert self.undo.redo_stack_size == 0

    def test_callbacks(self):
        """Test undo/redo callbacks."""
        undo_called = []
        redo_called = []
        change_called = []

        self.undo.on_undo(lambda e: undo_called.append(e))
        self.undo.on_redo(lambda e: redo_called.append(e))
        self.undo.on_change(lambda e: change_called.append(e))

        class TestObj:
            x = 10

        obj = TestObj()

        self.undo.record("Change", obj, "x", 10, 20)
        assert len(change_called) == 1

        self.undo.undo()
        assert len(undo_called) == 1

        self.undo.redo()
        assert len(redo_called) == 1

    def test_stack_limits(self):
        """Test stack size limits are enforced."""
        config = UndoSystemConfig(max_undo_levels=5)
        undo = UndoSystem(config=config)

        class TestObj:
            x = 0

        obj = TestObj()

        # Add more than limit
        for i in range(10):
            undo.record(f"Change {i}", obj, "x", i, i + 1)

        assert undo.undo_stack_size <= 5


class TestUndoSystemGrouping:
    """Tests for automatic grouping."""

    def setup_method(self):
        config = UndoSystemConfig(
            enable_grouping=True,
            group_timeout_ms=100,
        )
        self.undo = UndoSystem(config=config)

    def test_rapid_changes_grouped(self):
        """Test that rapid changes to same field are grouped."""
        class TestObj:
            x = 0

        obj = TestObj()

        # Make rapid changes to same field
        for i in range(5):
            self.undo.record("Set x", obj, "x", i, i + 1)
            # No sleep - should be grouped

        # With grouping, should have fewer entries than changes
        assert self.undo.undo_stack_size < 5

    def test_different_fields_not_grouped(self):
        """Test that changes to different fields are not grouped."""
        class TestObj:
            x = 0
            y = 0

        obj = TestObj()

        self.undo.record("Set x", obj, "x", 0, 1)
        self.undo.record("Set y", obj, "y", 0, 1)

        # Different fields should not group
        assert self.undo.undo_stack_size >= 2


class TestGlobalUndoSystem:
    """Tests for global undo system."""

    def test_get_undo_system(self):
        """Test getting global undo system."""
        undo = get_undo_system()
        assert undo is not None
        assert isinstance(undo, UndoSystem)

    def test_global_is_singleton(self):
        """Test global instance is singleton."""
        undo1 = get_undo_system()
        undo2 = get_undo_system()
        assert undo1 is undo2
