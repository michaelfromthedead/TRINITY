"""
Comprehensive tests for the Command system.

Tests cover:
- Command execution and undo
- Command batching
- Transform commands
- Create/Delete commands
- Property commands
- Reparent commands
- Composite commands
- Command manager operations
- Command merging
"""
import pytest
import sys
import time

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.editor.commands import (
    Command,
    CommandManager,
    CommandBatch,
    TransformCommand,
    CreateCommand,
    DeleteCommand,
    PropertyCommand,
    ReparentCommand,
    CompositeCommand,
)


class MockObject:
    """Mock object for command tests."""
    def __init__(self):
        self.position = (0, 0, 0)
        self.rotation = (0, 0, 0)
        self.scale = (1, 1, 1)
        self.parent = None
        self.name = "MockObject"

    def set_position(self, x, y, z):
        self.position = (x, y, z)

    def set_rotation(self, x, y, z):
        self.rotation = (x, y, z)

    def set_scale(self, x, y, z):
        self.scale = (x, y, z)

    def set_parent(self, parent):
        self.parent = parent


class MockScene:
    """Mock scene for command tests."""
    def __init__(self):
        self.objects = []

    def add(self, obj, parent=None):
        self.objects.append(obj)

    def remove(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)


class ConcreteCommand(Command):
    """Concrete command implementation for testing."""
    def __init__(self, name="test"):
        super().__init__(name)
        self.execute_count = 0
        self.undo_count = 0

    def execute(self):
        self.execute_count += 1
        self._executed = True
        return True

    def undo(self):
        self.undo_count += 1
        self._executed = False
        return True


class TestCommand:
    """Tests for Command base class."""

    def test_command_creation(self):
        """Command should be created with defaults."""
        cmd = ConcreteCommand("Test")
        assert cmd.name == "Test"
        assert cmd.executed is False
        assert cmd.timestamp == 0.0

    def test_command_execute(self):
        """Command can be executed."""
        cmd = ConcreteCommand()

        assert cmd.execute() is True
        assert cmd.executed is True
        assert cmd.execute_count == 1

    def test_command_undo(self):
        """Command can be undone."""
        cmd = ConcreteCommand()
        cmd.execute()

        assert cmd.undo() is True
        assert cmd.executed is False
        assert cmd.undo_count == 1

    def test_command_redo_default(self):
        """Command redo calls execute by default."""
        cmd = ConcreteCommand()
        cmd.execute()
        cmd.undo()

        assert cmd.redo() is True
        assert cmd.execute_count == 2


class TestTransformCommand:
    """Tests for TransformCommand class."""

    def test_transform_command_position(self):
        """TransformCommand can change position."""
        obj = MockObject()
        cmd = TransformCommand(obj, "position", (0, 0, 0), (10, 20, 30))

        assert cmd.execute() is True
        assert obj.position == (10, 20, 30)

        assert cmd.undo() is True
        assert obj.position == (0, 0, 0)

    def test_transform_command_rotation(self):
        """TransformCommand can change rotation."""
        obj = MockObject()
        cmd = TransformCommand(obj, "rotation", (0, 0, 0), (45, 90, 0))

        assert cmd.execute() is True
        assert obj.rotation == (45, 90, 0)

        assert cmd.undo() is True
        assert obj.rotation == (0, 0, 0)

    def test_transform_command_scale(self):
        """TransformCommand can change scale."""
        obj = MockObject()
        cmd = TransformCommand(obj, "scale", (1, 1, 1), (2, 2, 2))

        assert cmd.execute() is True
        assert obj.scale == (2, 2, 2)

        assert cmd.undo() is True
        assert obj.scale == (1, 1, 1)

    def test_transform_command_merge(self):
        """TransformCommands can be merged."""
        obj = MockObject()
        cmd1 = TransformCommand(obj, "position", (0, 0, 0), (5, 5, 5))
        cmd2 = TransformCommand(obj, "position", (5, 5, 5), (10, 10, 10))

        assert cmd1.can_merge(cmd2) is True

        merged = cmd1.merge(cmd2)
        assert merged is not None
        assert merged.old_value == (0, 0, 0)
        assert merged.new_value == (10, 10, 10)

    def test_transform_command_no_merge_different_type(self):
        """TransformCommands of different types don't merge."""
        obj = MockObject()
        cmd1 = TransformCommand(obj, "position", (0, 0, 0), (5, 5, 5))
        cmd2 = TransformCommand(obj, "rotation", (0, 0, 0), (45, 0, 0))

        assert cmd1.can_merge(cmd2) is False

    def test_transform_command_no_merge_different_object(self):
        """TransformCommands on different objects don't merge."""
        obj1 = MockObject()
        obj2 = MockObject()
        cmd1 = TransformCommand(obj1, "position", (0, 0, 0), (5, 5, 5))
        cmd2 = TransformCommand(obj2, "position", (0, 0, 0), (10, 10, 10))

        assert cmd1.can_merge(cmd2) is False


class TestCreateCommand:
    """Tests for CreateCommand class."""

    def test_create_command(self):
        """CreateCommand creates an object."""
        scene = MockScene()
        cmd = CreateCommand(MockObject, {}, scene=scene)

        assert cmd.execute() is True
        assert cmd.created_object is not None
        assert cmd.created_object in scene.objects

    def test_create_command_undo(self):
        """CreateCommand undo removes the object."""
        scene = MockScene()
        cmd = CreateCommand(MockObject, {}, scene=scene)

        cmd.execute()
        obj = cmd.created_object

        assert cmd.undo() is True
        assert obj not in scene.objects


class TestDeleteCommand:
    """Tests for DeleteCommand class."""

    def test_delete_command(self):
        """DeleteCommand removes objects."""
        scene = MockScene()
        obj = MockObject()
        scene.add(obj)

        cmd = DeleteCommand([obj], scene=scene)

        assert cmd.execute() is True
        assert obj not in scene.objects

    def test_delete_command_undo(self):
        """DeleteCommand undo restores objects."""
        scene = MockScene()
        obj = MockObject()
        obj.position = (10, 20, 30)
        scene.add(obj)

        cmd = DeleteCommand([obj], scene=scene)
        cmd.execute()

        assert cmd.undo() is True
        assert obj in scene.objects
        assert obj.position == (10, 20, 30)

    def test_delete_command_multiple(self):
        """DeleteCommand can delete multiple objects."""
        scene = MockScene()
        objs = [MockObject() for _ in range(3)]
        for obj in objs:
            scene.add(obj)

        cmd = DeleteCommand(objs, scene=scene)

        assert cmd.execute() is True
        for obj in objs:
            assert obj not in scene.objects


class TestPropertyCommand:
    """Tests for PropertyCommand class."""

    def test_property_command(self):
        """PropertyCommand changes property value."""
        obj = MockObject()
        cmd = PropertyCommand(obj, "name", "MockObject", "NewName")

        assert cmd.execute() is True
        assert obj.name == "NewName"

        assert cmd.undo() is True
        assert obj.name == "MockObject"

    def test_property_command_merge(self):
        """PropertyCommands on same property can merge."""
        obj = MockObject()
        cmd1 = PropertyCommand(obj, "name", "A", "B")
        cmd2 = PropertyCommand(obj, "name", "B", "C")

        assert cmd1.can_merge(cmd2) is True

        merged = cmd1.merge(cmd2)
        assert merged.old_value == "A"
        assert merged.new_value == "C"


class TestReparentCommand:
    """Tests for ReparentCommand class."""

    def test_reparent_command(self):
        """ReparentCommand changes object parent."""
        obj = MockObject()
        old_parent = MockObject()
        new_parent = MockObject()
        obj.parent = old_parent

        cmd = ReparentCommand(obj, old_parent, new_parent)

        assert cmd.execute() is True
        assert obj.parent == new_parent

        assert cmd.undo() is True
        assert obj.parent == old_parent


class TestCompositeCommand:
    """Tests for CompositeCommand class."""

    def test_composite_command_creation(self):
        """CompositeCommand can be created with commands."""
        cmd1 = ConcreteCommand("cmd1")
        cmd2 = ConcreteCommand("cmd2")
        composite = CompositeCommand([cmd1, cmd2], "composite")

        assert len(composite.commands) == 2

    def test_composite_command_execute(self):
        """CompositeCommand executes all commands."""
        cmd1 = ConcreteCommand("cmd1")
        cmd2 = ConcreteCommand("cmd2")
        composite = CompositeCommand([cmd1, cmd2])

        assert composite.execute() is True
        assert cmd1.execute_count == 1
        assert cmd2.execute_count == 1

    def test_composite_command_undo(self):
        """CompositeCommand undoes all in reverse order."""
        obj = MockObject()
        cmd1 = TransformCommand(obj, "position", (0, 0, 0), (5, 5, 5))
        cmd2 = TransformCommand(obj, "position", (5, 5, 5), (10, 10, 10))
        composite = CompositeCommand([cmd1, cmd2])

        composite.execute()
        assert obj.position == (10, 10, 10)

        composite.undo()
        # After undoing both in reverse order
        assert obj.position == (0, 0, 0)

    def test_composite_command_add(self):
        """Commands can be added to composite."""
        composite = CompositeCommand()
        cmd = ConcreteCommand()

        composite.add(cmd)
        assert cmd in composite.commands

    def test_composite_command_rollback_on_failure(self):
        """CompositeCommand rolls back on failure."""
        class FailingCommand(Command):
            def execute(self):
                return False
            def undo(self):
                return True

        cmd1 = ConcreteCommand()
        cmd2 = FailingCommand()
        composite = CompositeCommand([cmd1, cmd2])

        assert composite.execute() is False
        # First command should be undone
        assert cmd1.undo_count == 1


class TestCommandBatch:
    """Tests for CommandBatch class."""

    def test_batch_context_manager(self):
        """CommandBatch works as context manager."""
        manager = CommandManager()

        with manager.batch("Test Batch") as batch:
            obj = MockObject()
            manager.execute(TransformCommand(obj, "position", (0, 0, 0), (10, 10, 10)))
            manager.execute(TransformCommand(obj, "rotation", (0, 0, 0), (45, 0, 0)))

        # Should be one undo entry (composite)
        assert manager.can_undo is True
        # Single undo should revert both
        manager.undo()
        assert obj.position == (0, 0, 0)
        assert obj.rotation == (0, 0, 0)


class TestCommandManager:
    """Tests for CommandManager class."""

    def test_manager_creation(self):
        """CommandManager should be created empty."""
        manager = CommandManager()
        assert manager.can_undo is False
        assert manager.can_redo is False

    def test_manager_execute(self):
        """CommandManager executes commands."""
        manager = CommandManager()
        cmd = ConcreteCommand()

        assert manager.execute(cmd) is True
        assert manager.can_undo is True

    def test_manager_undo(self):
        """CommandManager can undo commands."""
        manager = CommandManager()
        cmd = ConcreteCommand()
        manager.execute(cmd)

        assert manager.undo() is True
        assert cmd.undo_count == 1
        assert manager.can_redo is True

    def test_manager_redo(self):
        """CommandManager can redo commands."""
        manager = CommandManager()
        cmd = ConcreteCommand()
        manager.execute(cmd)
        manager.undo()

        assert manager.redo() is True
        assert cmd.execute_count == 2

    def test_manager_redo_cleared_on_new_command(self):
        """New command clears redo stack."""
        manager = CommandManager()
        cmd1 = ConcreteCommand("cmd1")
        cmd2 = ConcreteCommand("cmd2")

        manager.execute(cmd1)
        manager.undo()
        assert manager.can_redo is True

        manager.execute(cmd2)
        assert manager.can_redo is False

    def test_manager_undo_empty(self):
        """Undo on empty stack returns False."""
        manager = CommandManager()
        assert manager.undo() is False

    def test_manager_redo_empty(self):
        """Redo on empty stack returns False."""
        manager = CommandManager()
        assert manager.redo() is False

    def test_manager_history_limit(self):
        """CommandManager respects history limit."""
        manager = CommandManager(max_history=5)

        for i in range(10):
            manager.execute(ConcreteCommand(f"cmd{i}"))

        assert len(manager.undo_stack) <= 5

    def test_manager_callbacks(self):
        """CommandManager triggers callbacks."""
        manager = CommandManager()
        executed = []
        undone = []
        redone = []

        manager.on_execute = lambda c: executed.append(c.name)
        manager.on_undo = lambda c: undone.append(c.name)
        manager.on_redo = lambda c: redone.append(c.name)

        cmd = ConcreteCommand("test")
        manager.execute(cmd)
        assert "test" in executed

        manager.undo()
        assert "test" in undone

        manager.redo()
        assert "test" in redone

    def test_manager_clear(self):
        """CommandManager can clear history."""
        manager = CommandManager()
        manager.execute(ConcreteCommand())
        manager.execute(ConcreteCommand())
        manager.undo()

        manager.clear()

        assert manager.can_undo is False
        assert manager.can_redo is False

    def test_manager_undo_to_command(self):
        """CommandManager can undo to specific command."""
        manager = CommandManager()
        cmd1 = ConcreteCommand("cmd1")
        cmd2 = ConcreteCommand("cmd2")
        cmd3 = ConcreteCommand("cmd3")

        manager.execute(cmd1)
        manager.execute(cmd2)
        manager.execute(cmd3)

        undone = manager.undo_to(cmd1)
        assert undone == 2
        assert manager.can_undo is True

    def test_manager_get_descriptions(self):
        """CommandManager provides undo/redo descriptions."""
        manager = CommandManager()

        assert manager.get_undo_description() is None

        cmd = ConcreteCommand("Test Command")
        cmd.description = "Test description"
        manager.execute(cmd)

        assert manager.get_undo_description() == "Test description"

        manager.undo()
        assert manager.get_redo_description() == "Test description"

    def test_manager_command_merging(self):
        """CommandManager merges consecutive commands."""
        manager = CommandManager()
        manager.merge_timeout = 2.0  # 2 second timeout

        obj = MockObject()
        cmd1 = TransformCommand(obj, "position", (0, 0, 0), (5, 5, 5))
        cmd2 = TransformCommand(obj, "position", (5, 5, 5), (10, 10, 10))

        manager.execute(cmd1)
        manager.execute(cmd2)  # Should merge with cmd1

        # Should have one merged command in stack
        assert len(manager.undo_stack) == 1

        # Single undo should revert to original
        manager.undo()
        assert obj.position == (0, 0, 0)

    def test_manager_stacks_are_copies(self):
        """Undo/redo stacks return copies."""
        manager = CommandManager()
        cmd = ConcreteCommand()
        manager.execute(cmd)

        undo = manager.undo_stack
        undo.clear()

        assert len(manager.undo_stack) == 1  # Original unchanged

    def test_manager_batch_empty_not_added(self):
        """Empty batch is not added to history."""
        manager = CommandManager()

        with manager.batch("Empty"):
            pass  # No commands

        assert manager.can_undo is False
