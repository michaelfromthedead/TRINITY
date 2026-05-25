"""
Tests for command pattern implementation.
"""
import pytest
from dataclasses import dataclass

from engine.tooling.undo.command_pattern import (
    Command,
    CompositeCommand,
    SetFieldCommand,
    CallMethodCommand,
    CreateObjectCommand,
    DeleteObjectCommand,
    CommandFactory,
)


class TestSetFieldCommand:
    """Tests for SetFieldCommand."""

    def test_execute(self):
        """Test executing a set field command."""
        class TestObj:
            def __init__(self):
                self.x = 10

        obj = TestObj()
        cmd = SetFieldCommand(obj, "x", 20)

        result = cmd.execute()

        assert result is True
        assert obj.x == 20
        assert cmd.executed is True
        assert cmd.old_value == 10

    def test_unexecute(self):
        """Test unexecuting a set field command."""
        class TestObj:
            def __init__(self):
                self.x = 10

        obj = TestObj()
        cmd = SetFieldCommand(obj, "x", 20)

        cmd.execute()
        result = cmd.unexecute()

        assert result is True
        assert obj.x == 10
        assert cmd.executed is False

    def test_can_execute(self):
        """Test can_execute check."""
        class TestObj:
            x = 10

        obj = TestObj()
        cmd = SetFieldCommand(obj, "x", 20)

        assert cmd.can_execute() is True

        cmd.execute()

        assert cmd.can_execute() is False

    def test_can_unexecute(self):
        """Test can_unexecute check."""
        class TestObj:
            x = 10

        obj = TestObj()
        cmd = SetFieldCommand(obj, "x", 20)

        assert cmd.can_unexecute() is False

        cmd.execute()

        assert cmd.can_unexecute() is True

    def test_name(self):
        """Test command name."""
        class TestObj:
            x = 10

        obj = TestObj()
        cmd = SetFieldCommand(obj, "x", 20)

        assert "x" in cmd.name

        named_cmd = SetFieldCommand(obj, "x", 20, name="Custom Name")
        assert named_cmd.name == "Custom Name"

    def test_merge_same_field(self):
        """Test merging commands on same field."""
        class TestObj:
            x = 10

        obj = TestObj()
        cmd1 = SetFieldCommand(obj, "x", 20)
        cmd1.execute()

        cmd2 = SetFieldCommand(obj, "x", 30)

        merged = cmd1.merge_with(cmd2)

        assert merged is not None
        assert merged.new_value == 30
        assert merged._old_value == 10  # Preserves original old value

    def test_merge_different_fields(self):
        """Test merging commands on different fields returns None."""
        class TestObj:
            x = 10
            y = 20

        obj = TestObj()
        cmd1 = SetFieldCommand(obj, "x", 20)
        cmd2 = SetFieldCommand(obj, "y", 30)

        merged = cmd1.merge_with(cmd2)
        assert merged is None

    def test_weak_reference(self):
        """Test that command uses weak reference."""
        class TestObj:
            x = 10

        obj = TestObj()
        cmd = SetFieldCommand(obj, "x", 20)

        assert cmd.target is obj

        del obj

        # Target should now be None
        assert cmd.target is None


class TestCompositeCommand:
    """Tests for CompositeCommand."""

    def test_execute_all(self):
        """Test executing all sub-commands."""
        class TestObj:
            def __init__(self):
                self.x = 10
                self.y = 20

        obj = TestObj()
        composite = CompositeCommand("Set Both")
        composite.add(SetFieldCommand(obj, "x", 100))
        composite.add(SetFieldCommand(obj, "y", 200))

        result = composite.execute()

        assert result is True
        assert obj.x == 100
        assert obj.y == 200
        assert composite.executed is True

    def test_unexecute_all(self):
        """Test unexecuting all sub-commands."""
        class TestObj:
            def __init__(self):
                self.x = 10
                self.y = 20

        obj = TestObj()
        composite = CompositeCommand("Set Both")
        composite.add(SetFieldCommand(obj, "x", 100))
        composite.add(SetFieldCommand(obj, "y", 200))

        composite.execute()
        result = composite.unexecute()

        assert result is True
        assert obj.x == 10
        assert obj.y == 20
        assert composite.executed is False

    def test_rollback_on_failure(self):
        """Test rollback when a command fails."""
        class TestObj:
            def __init__(self):
                self.x = 10

        obj = TestObj()

        class FailingCommand(Command):
            def execute(self):
                return False
            def unexecute(self):
                return True

        composite = CompositeCommand("Mixed")
        composite.add(SetFieldCommand(obj, "x", 100))
        composite.add(FailingCommand())

        result = composite.execute()

        assert result is False
        assert obj.x == 10  # Should be rolled back
        assert composite.executed is False

    def test_command_count(self):
        """Test command count property."""
        composite = CompositeCommand("Test")

        assert composite.command_count == 0

        class TestObj:
            x = 10

        obj = TestObj()
        composite.add(SetFieldCommand(obj, "x", 20))

        assert composite.command_count == 1


class TestCallMethodCommand:
    """Tests for CallMethodCommand."""

    def test_execute(self):
        """Test executing a method call."""
        class TestObj:
            def __init__(self):
                self.value = 10

            def increment(self, amount=1):
                self.value += amount
                return self.value

            def decrement(self, amount=1):
                self.value -= amount

        obj = TestObj()
        cmd = CallMethodCommand(
            obj,
            do_method="increment",
            undo_method="decrement",
            do_kwargs={"amount": 5},
            undo_kwargs={"amount": 5},
        )

        result = cmd.execute()

        assert result is True
        assert obj.value == 15
        assert cmd.result == 15

    def test_unexecute(self):
        """Test unexecuting a method call."""
        class TestObj:
            def __init__(self):
                self.value = 10

            def increment(self, amount=1):
                self.value += amount

            def decrement(self, amount=1):
                self.value -= amount

        obj = TestObj()
        cmd = CallMethodCommand(
            obj,
            do_method="increment",
            undo_method="decrement",
            do_kwargs={"amount": 5},
            undo_kwargs={"amount": 5},
        )

        cmd.execute()
        result = cmd.unexecute()

        assert result is True
        assert obj.value == 10

    def test_nonexistent_method(self):
        """Test calling non-existent method returns False."""
        class TestObj:
            pass

        obj = TestObj()
        cmd = CallMethodCommand(
            obj,
            do_method="nonexistent",
            undo_method="also_nonexistent",
        )

        result = cmd.execute()
        assert result is False


class TestCreateObjectCommand:
    """Tests for CreateObjectCommand."""

    def test_execute(self):
        """Test creating an object."""
        created = []

        def factory():
            obj = {"id": len(created)}
            created.append(obj)
            return obj

        def destroyer(obj):
            created.remove(obj)

        cmd = CreateObjectCommand(factory, destroyer)

        result = cmd.execute()

        assert result is True
        assert len(created) == 1
        assert cmd.created_object is created[0]

    def test_unexecute(self):
        """Test destroying the created object."""
        created = []

        def factory():
            obj = {"id": len(created)}
            created.append(obj)
            return obj

        def destroyer(obj):
            created.remove(obj)

        cmd = CreateObjectCommand(factory, destroyer)

        cmd.execute()
        assert len(created) == 1

        result = cmd.unexecute()

        assert result is True
        assert len(created) == 0
        assert cmd.created_object is None


class TestDeleteObjectCommand:
    """Tests for DeleteObjectCommand."""

    def test_execute(self):
        """Test deleting an object."""
        @dataclass
        class TestObj:
            x: int = 10

        objects = []
        obj = TestObj(x=42)
        objects.append(obj)

        def deleter(o):
            objects.remove(o)

        def restorer(o):
            objects.append(o)

        cmd = DeleteObjectCommand(obj, deleter, restorer)

        result = cmd.execute()

        assert result is True
        assert len(objects) == 0

    def test_unexecute(self):
        """Test restoring a deleted object."""
        @dataclass
        class TestObj:
            x: int = 10

        objects = []
        obj = TestObj(x=42)
        objects.append(obj)

        def deleter(o):
            objects.remove(o)

        def restorer(o):
            objects.append(o)

        cmd = DeleteObjectCommand(obj, deleter, restorer)

        cmd.execute()
        assert len(objects) == 0

        result = cmd.unexecute()

        assert result is True
        assert len(objects) == 1
        assert objects[0].x == 42


class TestCommandFactory:
    """Tests for CommandFactory."""

    def test_set_field(self):
        """Test creating a SetFieldCommand."""
        class TestObj:
            x = 10

        obj = TestObj()
        cmd = CommandFactory.set_field(obj, "x", 20)

        assert isinstance(cmd, SetFieldCommand)
        assert cmd.new_value == 20

    def test_set_fields(self):
        """Test creating a composite for multiple fields."""
        class TestObj:
            x = 10
            y = 20

        obj = TestObj()
        cmd = CommandFactory.set_fields(obj, {"x": 100, "y": 200})

        assert isinstance(cmd, CompositeCommand)
        assert cmd.command_count == 2

    def test_create_object(self):
        """Test creating a CreateObjectCommand."""
        created = []
        cmd = CommandFactory.create_object(
            factory=lambda: created.append(1) or 1,
            destroyer=lambda x: created.remove(1),
        )

        assert isinstance(cmd, CreateObjectCommand)

    def test_delete_object(self):
        """Test creating a DeleteObjectCommand."""
        class TestObj:
            x = 10

        obj = TestObj()
        cmd = CommandFactory.delete_object(
            obj=obj,
            deleter=lambda x: None,
            restorer=lambda x: None,
        )

        assert isinstance(cmd, DeleteObjectCommand)
