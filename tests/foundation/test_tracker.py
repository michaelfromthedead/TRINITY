"""
Comprehensive unit tests for the Tracker system.

Tests cover:
- Dirty tracking (mark_dirty, is_dirty, dirty_fields, mark_clean, all_dirty)
- Change events (on_change, off_change, object-specific callbacks, type callbacks)
- Transactions (begin, commit, rollback)
- Undo/Redo (undo, redo, can_undo, can_redo, stacks)
- Change coalescing within transactions
- Weak references and garbage collection
"""
import gc
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.tracker import tracker, Tracker, Change, Transaction


class SimpleObject:
    """A simple class that supports weak references (unlike bare object())."""
    pass


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset tracker state before each test."""
    tracker._dirty.clear()
    tracker._cb_global.clear()
    tracker._cb_type.clear()
    tracker._cb_obj.clear()
    tracker._undo.clear()
    tracker._redo.clear()
    tracker._txn = None
    yield


class TestDirtyTracking:
    """Tests for dirty flag tracking functionality."""

    def test_initially_not_dirty(self):
        """Objects should not be dirty initially."""
        obj = SimpleObject()
        assert tracker.is_dirty(obj) is False

    def test_mark_dirty(self):
        """Marking an object dirty should make is_dirty return True."""
        class Box:
            value = 0

        obj = Box()
        tracker.mark_dirty(obj, "value", 0, 1)
        assert tracker.is_dirty(obj) is True

    def test_dirty_fields(self):
        """dirty_fields should return all fields that were marked dirty."""
        class Point:
            x = 0
            y = 0

        obj = Point()
        tracker.mark_dirty(obj, "x", 0, 10)
        tracker.mark_dirty(obj, "y", 0, 20)

        fields = tracker.dirty_fields(obj)
        assert "x" in fields
        assert "y" in fields
        assert len(fields) == 2

    def test_dirty_fields_empty_for_clean_object(self):
        """dirty_fields should return empty set for non-dirty objects."""
        obj = SimpleObject()
        fields = tracker.dirty_fields(obj)
        assert fields == set()

    def test_mark_clean(self):
        """mark_clean should clear all dirty flags for an object."""
        obj = SimpleObject()
        tracker.mark_dirty(obj, "field1", None, 1)
        tracker.mark_dirty(obj, "field2", None, 2)
        assert tracker.is_dirty(obj)

        tracker.mark_clean(obj)
        assert tracker.is_dirty(obj) is False
        assert tracker.dirty_fields(obj) == set()

    def test_mark_clean_on_clean_object(self):
        """mark_clean on a non-dirty object should not raise."""
        obj = SimpleObject()
        tracker.mark_clean(obj)  # Should not raise
        assert tracker.is_dirty(obj) is False

    def test_all_dirty(self):
        """all_dirty should return all objects with dirty flags."""
        a = SimpleObject()
        b = SimpleObject()
        c = SimpleObject()  # Will not be marked dirty

        tracker.mark_dirty(a, "f", None, 1)
        tracker.mark_dirty(b, "f", None, 1)

        dirty = tracker.all_dirty()
        assert a in dirty
        assert b in dirty
        assert c not in dirty
        assert len(dirty) == 2

    def test_all_dirty_excludes_cleaned_objects(self):
        """all_dirty should not include objects that were marked clean."""
        obj = SimpleObject()
        tracker.mark_dirty(obj, "f", None, 1)
        tracker.mark_clean(obj)

        dirty = tracker.all_dirty()
        assert obj not in dirty


class TestChangeEvents:
    """Tests for change event subscription system."""

    def test_on_change_callback(self):
        """Global callbacks should be called on any change."""
        changes = []

        def callback(obj, field, old, new):
            changes.append((field, old, new))

        tracker.on_change(callback)

        obj = SimpleObject()
        tracker.mark_dirty(obj, "value", 10, 20)

        assert len(changes) == 1
        assert changes[0] == ("value", 10, 20)

    def test_on_change_multiple_callbacks(self):
        """Multiple global callbacks should all be called."""
        results1 = []
        results2 = []

        def cb1(obj, field, old, new):
            results1.append(field)

        def cb2(obj, field, old, new):
            results2.append(field)

        tracker.on_change(cb1)
        tracker.on_change(cb2)

        obj = SimpleObject()
        tracker.mark_dirty(obj, "test", 0, 1)

        assert results1 == ["test"]
        assert results2 == ["test"]

    def test_off_change_removes_callback(self):
        """off_change should remove a callback so it no longer receives events."""
        changes = []

        def callback(obj, field, old, new):
            changes.append(field)

        tracker.on_change(callback)
        tracker.off_change(callback)

        obj = SimpleObject()
        tracker.mark_dirty(obj, "value", 0, 1)

        assert len(changes) == 0

    def test_object_specific_callback(self):
        """Object-specific callbacks should only fire for that object."""
        target = SimpleObject()
        other = SimpleObject()
        changes = []

        def callback(obj, field, old, new):
            changes.append(field)

        tracker.on_change(target, callback)

        tracker.mark_dirty(target, "a", 0, 1)
        tracker.mark_dirty(other, "b", 0, 1)

        assert changes == ["a"]  # Only target's change

    def test_type_specific_callback(self):
        """Type-specific callbacks should fire for instances of that type."""
        class MyClass:
            pass

        class OtherClass:
            pass

        changes = []

        def callback(obj, field, old, new):
            changes.append((type(obj).__name__, field))

        tracker.on_change(MyClass, callback)

        obj1 = MyClass()
        obj2 = OtherClass()

        tracker.mark_dirty(obj1, "x", 0, 1)
        tracker.mark_dirty(obj2, "y", 0, 1)

        assert changes == [("MyClass", "x")]

    def test_on_change_duplicate_callback_not_added(self):
        """Same callback should not be added twice."""
        count = [0]

        def callback(obj, field, old, new):
            count[0] += 1

        tracker.on_change(callback)
        tracker.on_change(callback)  # Duplicate

        obj = SimpleObject()
        tracker.mark_dirty(obj, "f", 0, 1)

        assert count[0] == 1  # Called only once

    def test_on_change_requires_callback(self):
        """on_change should raise if no callback is provided."""
        with pytest.raises(ValueError, match="callback required"):
            tracker.on_change(None, None)


class TestTransactions:
    """Tests for transaction management."""

    def test_begin_commit_transaction(self):
        """begin_transaction and commit_transaction should work together."""
        tracker.begin_transaction("test")
        assert tracker.in_transaction is True
        tracker.commit_transaction()
        assert tracker.in_transaction is False

    def test_transaction_groups_changes(self):
        """Changes within a transaction should be grouped in one undo entry."""
        class Counter:
            value = 0

        obj = Counter()

        tracker.begin_transaction("increment")
        tracker.mark_dirty(obj, "value", 0, 1)
        tracker.mark_dirty(obj, "other", None, "test")
        tracker.commit_transaction()

        assert tracker.can_undo is True
        assert len(tracker.undo_stack) == 1
        assert tracker.undo_stack[0].name == "increment"

    def test_rollback_transaction(self):
        """rollback_transaction should revert changes and clear transaction."""
        class Box:
            value = 0

        obj = Box()
        obj.value = 0

        tracker.begin_transaction("test")
        obj.value = 100
        tracker.mark_dirty(obj, "value", 0, 100)
        tracker.rollback_transaction()

        assert tracker.in_transaction is False
        assert obj.value == 0  # Value should be reverted
        # Rolled back transaction should not be in undo stack
        assert len(tracker.undo_stack) == 0

    def test_nested_transactions_not_supported(self):
        """Attempting nested transactions should raise an error."""
        tracker.begin_transaction("outer")
        with pytest.raises(RuntimeError, match="Nested transactions not supported"):
            tracker.begin_transaction("inner")
        tracker.commit_transaction()

    def test_commit_without_transaction_raises(self):
        """Committing without an active transaction should raise."""
        with pytest.raises(RuntimeError, match="No active transaction"):
            tracker.commit_transaction()

    def test_rollback_without_transaction_raises(self):
        """Rolling back without an active transaction should raise."""
        with pytest.raises(RuntimeError, match="No active transaction"):
            tracker.rollback_transaction()

    def test_empty_transaction_not_added_to_undo(self):
        """Transactions with no changes should not be added to undo stack."""
        tracker.begin_transaction("empty")
        tracker.commit_transaction()

        assert len(tracker.undo_stack) == 0


class TestUndoRedo:
    """Tests for undo/redo functionality."""

    def test_undo_reverts_change(self):
        """undo should revert the object to its previous state."""
        class Box:
            value = 0

        obj = Box()
        obj.value = 0

        tracker.begin_transaction("change")
        old_val = obj.value
        obj.value = 100
        tracker.mark_dirty(obj, "value", old_val, 100)
        tracker.commit_transaction()

        assert obj.value == 100
        assert tracker.can_undo is True

        result = tracker.undo()
        assert result is True
        assert obj.value == 0  # Reverted

    def test_redo_after_undo(self):
        """redo should restore changes after undo."""
        class Box:
            value = 0

        obj = Box()
        obj.value = 0

        tracker.begin_transaction("test")
        obj.value = 50
        tracker.mark_dirty(obj, "value", 0, 50)
        tracker.commit_transaction()

        tracker.undo()
        assert obj.value == 0
        assert tracker.can_redo is True

        result = tracker.redo()
        assert result is True
        assert obj.value == 50
        assert tracker.can_undo is True

    def test_can_undo_initially_false(self):
        """can_undo should be False when there are no changes."""
        assert tracker.can_undo is False

    def test_can_redo_initially_false(self):
        """can_redo should be False when nothing has been undone."""
        assert tracker.can_redo is False

    def test_undo_returns_false_when_empty(self):
        """undo should return False when there is nothing to undo."""
        result = tracker.undo()
        assert result is False

    def test_redo_returns_false_when_empty(self):
        """redo should return False when there is nothing to redo."""
        result = tracker.redo()
        assert result is False

    def test_new_change_clears_redo_stack(self):
        """Making a new change after undo should clear the redo stack."""
        class Box:
            value = 0

        obj = Box()

        # First change
        tracker.begin_transaction("first")
        obj.value = 10
        tracker.mark_dirty(obj, "value", 0, 10)
        tracker.commit_transaction()

        # Undo it
        tracker.undo()
        assert tracker.can_redo is True

        # New change
        tracker.begin_transaction("second")
        obj.value = 20
        tracker.mark_dirty(obj, "value", 0, 20)
        tracker.commit_transaction()

        # Redo stack should be cleared
        assert tracker.can_redo is False

    def test_undo_stack_property(self):
        """undo_stack should return copy of internal undo stack."""
        obj = SimpleObject()
        tracker.begin_transaction("tx1")
        tracker.mark_dirty(obj, "f", 0, 1)
        tracker.commit_transaction()

        stack = tracker.undo_stack
        assert len(stack) == 1
        assert stack[0].name == "tx1"

        # Verify it's a copy
        stack.clear()
        assert len(tracker.undo_stack) == 1

    def test_redo_stack_property(self):
        """redo_stack should return copy of internal redo stack."""
        obj = SimpleObject()
        tracker.begin_transaction("tx1")
        tracker.mark_dirty(obj, "f", 0, 1)
        tracker.commit_transaction()

        tracker.undo()

        stack = tracker.redo_stack
        assert len(stack) == 1
        assert stack[0].name == "tx1"


class TestChangeCoalescing:
    """Tests for change coalescing within transactions."""

    def test_multiple_changes_same_field_coalesced(self):
        """Multiple changes to same field in one transaction should coalesce."""
        class Counter:
            n = 0

        obj = Counter()

        tracker.begin_transaction("many changes")
        tracker.mark_dirty(obj, "n", 0, 1)
        tracker.mark_dirty(obj, "n", 1, 2)
        tracker.mark_dirty(obj, "n", 2, 3)
        tracker.commit_transaction()

        # Should have one transaction
        assert len(tracker.undo_stack) == 1
        tx = tracker.undo_stack[0]

        # Coalesced: should preserve original old_value (0) and final new_value (3)
        assert len(tx.changes) == 1
        change = tx.changes[0]
        assert change.field == "n"
        assert change.old_value == 0
        assert change.new_value == 3

    def test_different_fields_not_coalesced(self):
        """Changes to different fields should not be coalesced."""
        class Point:
            x = 0
            y = 0

        obj = Point()

        tracker.begin_transaction("move")
        tracker.mark_dirty(obj, "x", 0, 10)
        tracker.mark_dirty(obj, "y", 0, 20)
        tracker.commit_transaction()

        tx = tracker.undo_stack[0]
        assert len(tx.changes) == 2

    def test_different_objects_not_coalesced(self):
        """Changes to different objects should not be coalesced."""
        class Box:
            value = 0

        obj1 = Box()
        obj2 = Box()

        tracker.begin_transaction("update both")
        tracker.mark_dirty(obj1, "value", 0, 1)
        tracker.mark_dirty(obj2, "value", 0, 2)
        tracker.commit_transaction()

        tx = tracker.undo_stack[0]
        assert len(tx.changes) == 2


class TestWeakReferences:
    """Tests for weak reference handling and garbage collection."""

    def test_change_object_property(self):
        """Change.obj property should return the object or None if GC'd."""
        class Temp:
            pass

        obj = Temp()
        import weakref
        change = Change(weakref.ref(obj), "field", 0, 1)

        assert change.obj is obj

    def test_dead_objects_cleaned_from_dirty_set(self):
        """Dead objects should be cleaned when all_dirty is called."""
        class Temp:
            pass

        # Create and dirty an object
        obj = Temp()
        tracker.mark_dirty(obj, "f", 0, 1)
        assert len(tracker.all_dirty()) == 1

        # Delete reference and force GC
        del obj
        gc.collect()

        # all_dirty should clean up dead references
        dirty = tracker.all_dirty()
        assert len(dirty) == 0

    def test_cleanup_removes_object_callbacks(self):
        """When an object is GC'd, its callbacks should be cleaned up."""
        class Temp:
            pass

        obj = Temp()
        oid = id(obj)

        def callback(o, f, old, new):
            pass

        tracker.on_change(obj, callback)
        assert oid in tracker._cb_obj

        # Trigger cleanup manually (simulating GC)
        tracker._cleanup(oid)
        assert oid not in tracker._cb_obj


class TestChange:
    """Tests for the Change dataclass."""

    def test_change_creation(self):
        """Change should store field, old_value, and new_value."""
        import weakref

        obj = SimpleObject()
        change = Change(weakref.ref(obj), "field", 10, 20)

        assert change.field == "field"
        assert change.old_value == 10
        assert change.new_value == 20
        assert change.obj is obj


class TestTransaction:
    """Tests for the Transaction dataclass."""

    def test_transaction_creation(self):
        """Transaction should have name, changes list, and timestamp."""
        tx = Transaction(name="test_tx")

        assert tx.name == "test_tx"
        assert tx.changes == []
        assert tx.timestamp > 0

    def test_transaction_with_changes(self):
        """Transaction can be created with changes."""
        import weakref

        obj = SimpleObject()
        change = Change(weakref.ref(obj), "f", 0, 1)
        tx = Transaction(name="with_changes", changes=[change])

        assert len(tx.changes) == 1
        assert tx.changes[0].field == "f"


class TestTrackerInstance:
    """Tests for the global tracker singleton."""

    def test_tracker_is_singleton(self):
        """The global tracker should be a Tracker instance."""
        assert isinstance(tracker, Tracker)

    def test_fresh_tracker_instance(self):
        """A fresh Tracker instance should have empty state."""
        fresh = Tracker()

        assert fresh.can_undo is False
        assert fresh.can_redo is False
        assert fresh.in_transaction is False
        assert len(fresh.undo_stack) == 0
        assert len(fresh.redo_stack) == 0


class TestUndoWithApply:
    """Tests verifying that undo/redo actually apply changes to objects."""

    def test_undo_sets_object_dict(self):
        """undo should set values in object __dict__."""
        class Entity:
            def __init__(self):
                self.health = 100

        e = Entity()

        tracker.begin_transaction("damage")
        old = e.health
        e.health = 50
        tracker.mark_dirty(e, "health", old, 50)
        tracker.commit_transaction()

        tracker.undo()
        assert e.health == 100

    def test_redo_sets_object_dict(self):
        """redo should restore values in object __dict__."""
        class Entity:
            def __init__(self):
                self.health = 100

        e = Entity()

        tracker.begin_transaction("damage")
        old = e.health
        e.health = 50
        tracker.mark_dirty(e, "health", old, 50)
        tracker.commit_transaction()

        tracker.undo()
        assert e.health == 100

        tracker.redo()
        assert e.health == 50

    def test_multiple_undo_redo(self):
        """Multiple undo/redo operations should work correctly."""
        class Counter:
            def __init__(self):
                self.value = 0

        c = Counter()

        # Change 1: 0 -> 10
        tracker.begin_transaction("set to 10")
        old = c.value
        c.value = 10
        tracker.mark_dirty(c, "value", old, 10)
        tracker.commit_transaction()

        # Change 2: 10 -> 20
        tracker.begin_transaction("set to 20")
        old = c.value
        c.value = 20
        tracker.mark_dirty(c, "value", old, 20)
        tracker.commit_transaction()

        assert c.value == 20

        tracker.undo()
        assert c.value == 10

        tracker.undo()
        assert c.value == 0

        tracker.redo()
        assert c.value == 10

        tracker.redo()
        assert c.value == 20
