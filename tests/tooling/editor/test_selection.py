"""
Comprehensive tests for the Selection system.

Tests cover:
- Selection operations (select, add, remove, toggle, clear)
- Selection filters by type and custom functions
- Selection sets (named selections)
- Marquee (box) selection
- Selection groups
- Selection history (undo/redo selection)
- Picking results
"""
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.editor.selection import (
    SelectionManager,
    Selection,
    SelectionSet,
    SelectionFilter,
    MarqueeSelection,
    SelectionGroup,
    SelectionOperation,
    PickingResult,
)


class MockObject:
    """Mock object for selection tests."""
    def __init__(self, name: str = ""):
        self.name = name


class TestPickingResult:
    """Tests for PickingResult class."""

    def test_picking_result_hit(self):
        """PickingResult with hit."""
        obj = MockObject("target")
        result = PickingResult(
            hit=True,
            object=obj,
            position=(1.0, 2.0, 3.0),
            normal=(0.0, 1.0, 0.0),
            distance=5.0
        )

        assert result.hit is True
        assert result.object == obj
        assert result.position == (1.0, 2.0, 3.0)
        assert result.distance == 5.0

    def test_picking_result_miss(self):
        """PickingResult miss factory."""
        result = PickingResult.miss()

        assert result.hit is False
        assert result.object is None
        assert result.distance == float('inf')


class TestSelectionFilter:
    """Tests for SelectionFilter class."""

    def test_filter_creation(self):
        """SelectionFilter should be created with defaults."""
        filter = SelectionFilter()
        assert filter.enabled is True
        assert len(filter.allowed_types) == 0
        assert len(filter.excluded_types) == 0

    def test_filter_allow_type(self):
        """Filter can allow specific types."""
        filter = SelectionFilter()
        filter.allow_type(MockObject)

        obj = MockObject()
        assert filter.accepts(obj) is True

        class OtherClass:
            pass
        assert filter.accepts(OtherClass()) is False

    def test_filter_exclude_type(self):
        """Filter can exclude specific types."""
        filter = SelectionFilter()
        filter.exclude_type(MockObject)

        obj = MockObject()
        assert filter.accepts(obj) is False

    def test_filter_custom_function(self):
        """Filter can use custom function."""
        filter = SelectionFilter()
        filter.set_custom_filter(lambda obj: hasattr(obj, "name") and obj.name.startswith("A"))

        obj1 = MockObject("Apple")
        obj2 = MockObject("Banana")

        assert filter.accepts(obj1) is True
        assert filter.accepts(obj2) is False

    def test_filter_disabled(self):
        """Disabled filter accepts all."""
        filter = SelectionFilter()
        filter.exclude_type(MockObject)
        filter.enabled = False

        obj = MockObject()
        assert filter.accepts(obj) is True

    def test_filter_reset(self):
        """Filter can be reset."""
        filter = SelectionFilter()
        filter.allow_type(MockObject)
        filter.exclude_type(str)
        filter.set_custom_filter(lambda x: True)
        filter.enabled = False

        filter.reset()

        assert len(filter.allowed_types) == 0
        assert len(filter.excluded_types) == 0
        assert filter.custom_filter is None
        assert filter.enabled is True

    def test_filter_chaining(self):
        """Filter methods can be chained."""
        filter = SelectionFilter()
        result = filter.allow_type(MockObject).set_custom_filter(lambda x: True)

        assert result == filter


class TestSelection:
    """Tests for Selection class."""

    def test_selection_creation(self):
        """Selection should be created empty."""
        selection = Selection()
        assert selection.empty is True
        assert selection.count == 0
        assert selection.primary is None

    def test_selection_select_set(self):
        """SET operation replaces selection."""
        selection = Selection()
        obj1 = MockObject("1")
        obj2 = MockObject("2")

        selection.select(obj1, SelectionOperation.SET)
        assert selection.count == 1
        assert selection.primary == obj1

        selection.select(obj2, SelectionOperation.SET)
        assert selection.count == 1
        assert selection.primary == obj2
        assert obj1 not in selection

    def test_selection_select_add(self):
        """ADD operation adds to selection."""
        selection = Selection()
        obj1 = MockObject("1")
        obj2 = MockObject("2")

        selection.select(obj1, SelectionOperation.SET)
        selection.select(obj2, SelectionOperation.ADD)

        assert selection.count == 2
        assert obj1 in selection
        assert obj2 in selection

    def test_selection_select_remove(self):
        """REMOVE operation removes from selection."""
        selection = Selection()
        obj1 = MockObject("1")
        obj2 = MockObject("2")

        selection.select(obj1, SelectionOperation.SET)
        selection.select(obj2, SelectionOperation.ADD)
        selection.select(obj1, SelectionOperation.REMOVE)

        assert selection.count == 1
        assert obj1 not in selection
        assert obj2 in selection

    def test_selection_select_toggle(self):
        """TOGGLE operation toggles selection."""
        selection = Selection()
        obj = MockObject("1")

        selection.select(obj, SelectionOperation.TOGGLE)
        assert obj in selection

        selection.select(obj, SelectionOperation.TOGGLE)
        assert obj not in selection

    def test_selection_clear(self):
        """Clear removes all items."""
        selection = Selection()
        obj1 = MockObject("1")
        obj2 = MockObject("2")

        selection.select(obj1, SelectionOperation.SET)
        selection.select(obj2, SelectionOperation.ADD)

        assert selection.clear() is True
        assert selection.empty is True
        assert selection.primary is None

    def test_selection_select_all(self):
        """Select all replaces with multiple items."""
        selection = Selection()
        objs = [MockObject(str(i)) for i in range(5)]

        selection.select_all(objs)

        assert selection.count == 5
        for obj in objs:
            assert obj in selection

    def test_selection_add_all(self):
        """Add all adds multiple items."""
        selection = Selection()
        obj1 = MockObject("1")
        objs = [MockObject(str(i)) for i in range(3)]

        selection.select(obj1, SelectionOperation.SET)
        selection.add_all(objs)

        assert selection.count == 4

    def test_selection_remove_all(self):
        """Remove all removes multiple items."""
        selection = Selection()
        objs = [MockObject(str(i)) for i in range(5)]

        selection.select_all(objs)
        selection.remove_all(objs[:2])

        assert selection.count == 3

    def test_selection_set_primary(self):
        """Primary can be set to any selected item."""
        selection = Selection()
        obj1 = MockObject("1")
        obj2 = MockObject("2")

        selection.select_all([obj1, obj2])

        assert selection.set_primary(obj2) is True
        assert selection.primary == obj2

    def test_selection_set_primary_not_selected(self):
        """Cannot set primary to non-selected item."""
        selection = Selection()
        obj1 = MockObject("1")
        obj2 = MockObject("2")

        selection.select(obj1, SelectionOperation.SET)

        assert selection.set_primary(obj2) is False
        assert selection.primary == obj1

    def test_selection_change_callback(self):
        """Selection change triggers callback."""
        selection = Selection()
        changes = []
        selection.on_changed = lambda items: changes.append(len(items))

        selection.select(MockObject(), SelectionOperation.SET)
        assert len(changes) == 1
        assert changes[0] == 1

    def test_selection_filter(self):
        """Filter affects selection."""
        selection = Selection()
        filter = SelectionFilter()
        filter.allow_type(MockObject)
        selection.set_filter(filter)

        obj = MockObject()
        other = "not a mock object"

        selection.select(obj, SelectionOperation.ADD)
        assert selection.count == 1

        # String should be rejected
        selection.select(other, SelectionOperation.ADD)
        assert selection.count == 1

    def test_selection_iteration(self):
        """Selection can be iterated."""
        selection = Selection()
        objs = [MockObject(str(i)) for i in range(3)]
        selection.select_all(objs)

        iterated = list(selection)
        assert len(iterated) == 3

    def test_selection_len(self):
        """Selection supports len()."""
        selection = Selection()
        objs = [MockObject(str(i)) for i in range(5)]
        selection.select_all(objs)

        assert len(selection) == 5


class TestSelectionSet:
    """Tests for SelectionSet class."""

    def test_selection_set_creation(self):
        """SelectionSet should be created with name."""
        ss = SelectionSet("my_selection")
        assert ss.name == "my_selection"
        assert ss.count == 0
        assert ss.locked is False

    def test_selection_set_with_items(self):
        """SelectionSet can be created with items."""
        objs = {MockObject("1"), MockObject("2")}
        ss = SelectionSet("my_selection", objs)

        assert ss.count == 2

    def test_selection_set_add_remove(self):
        """Items can be added and removed."""
        ss = SelectionSet("test")
        obj = MockObject()

        assert ss.add(obj) is True
        assert ss.count == 1

        assert ss.remove(obj) is True
        assert ss.count == 0

    def test_selection_set_clear(self):
        """SelectionSet can be cleared."""
        ss = SelectionSet("test")
        ss.add(MockObject())
        ss.add(MockObject())

        assert ss.clear() is True
        assert ss.count == 0

    def test_selection_set_locked(self):
        """Locked set cannot be modified."""
        ss = SelectionSet("test")
        ss.add(MockObject())
        ss.lock()

        assert ss.add(MockObject()) is False
        assert ss.remove(list(ss.items)[0]) is False
        assert ss.clear() is False
        assert ss.count == 1

    def test_selection_set_unlock(self):
        """Locked set can be unlocked."""
        ss = SelectionSet("test")
        ss.lock()
        ss.unlock()

        assert ss.add(MockObject()) is True


class TestMarqueeSelection:
    """Tests for MarqueeSelection class."""

    def test_marquee_creation(self):
        """MarqueeSelection should be created inactive."""
        marquee = MarqueeSelection()
        assert marquee.active is False

    def test_marquee_begin_end(self):
        """Marquee can be started and ended."""
        marquee = MarqueeSelection()

        marquee.begin(100, 100)
        assert marquee.active is True
        assert marquee.start_x == 100
        assert marquee.start_y == 100

        marquee.update(200, 150)
        assert marquee.end_x == 200
        assert marquee.end_y == 150

        result = marquee.end()
        assert marquee.active is False
        assert isinstance(result, set)

    def test_marquee_rect(self):
        """Marquee provides normalized rect."""
        marquee = MarqueeSelection()
        marquee.begin(200, 200)
        marquee.update(100, 150)

        rect = marquee.rect
        min_x, min_y, max_x, max_y = rect

        assert min_x == 100
        assert min_y == 150
        assert max_x == 200
        assert max_y == 200

    def test_marquee_dimensions(self):
        """Marquee provides width and height."""
        marquee = MarqueeSelection()
        marquee.begin(100, 100)
        marquee.update(200, 180)

        assert marquee.width == 100
        assert marquee.height == 80

    def test_marquee_candidates(self):
        """Marquee tracks candidates."""
        marquee = MarqueeSelection()
        marquee.begin(0, 0)

        obj1 = MockObject("1")
        obj2 = MockObject("2")

        marquee.add_candidate(obj1)
        marquee.add_candidate(obj2)

        result = marquee.end()
        assert obj1 in result
        assert obj2 in result

    def test_marquee_cancel(self):
        """Marquee can be cancelled."""
        marquee = MarqueeSelection()
        marquee.begin(0, 0)
        marquee.add_candidate(MockObject())

        marquee.cancel()

        assert marquee.active is False
        assert marquee._candidates == set()

    def test_marquee_contains_point(self):
        """Marquee can check point containment."""
        marquee = MarqueeSelection()
        marquee.begin(100, 100)
        marquee.update(200, 200)

        assert marquee.contains_point(150, 150) is True
        assert marquee.contains_point(50, 50) is False
        assert marquee.contains_point(250, 250) is False

    def test_marquee_operation(self):
        """Marquee stores operation mode."""
        marquee = MarqueeSelection()
        marquee.begin(0, 0, SelectionOperation.ADD)

        assert marquee.operation == SelectionOperation.ADD


class TestSelectionGroup:
    """Tests for SelectionGroup class."""

    def test_group_creation(self):
        """SelectionGroup should be created empty."""
        group = SelectionGroup("grp1", "My Group")
        assert group.id == "grp1"
        assert group.name == "My Group"
        assert group.count == 0

    def test_group_add_remove(self):
        """Objects can be added and removed."""
        group = SelectionGroup("grp1")
        obj = MockObject()

        assert group.add(obj) is True
        assert group.count == 1
        assert group.contains(obj) is True

        assert group.remove(obj) is True
        assert group.count == 0

    def test_group_add_duplicate(self):
        """Duplicate objects are not added."""
        group = SelectionGroup("grp1")
        obj = MockObject()

        group.add(obj)
        assert group.add(obj) is False
        assert group.count == 1

    def test_group_clear(self):
        """Group can be cleared."""
        group = SelectionGroup("grp1")
        group.add(MockObject())
        group.add(MockObject())

        assert group.clear() is True
        assert group.count == 0

    def test_group_locked(self):
        """Locked group cannot be modified."""
        group = SelectionGroup("grp1")
        group.add(MockObject())
        group.lock()

        assert group.add(MockObject()) is False
        assert group.remove(group.members[0]) is False
        assert group.clear() is False

    def test_group_reorder(self):
        """Objects can be reordered in group."""
        group = SelectionGroup("grp1")
        obj1 = MockObject("1")
        obj2 = MockObject("2")
        obj3 = MockObject("3")

        group.add(obj1)
        group.add(obj2)
        group.add(obj3)

        assert group.reorder(obj3, 0) is True
        assert group.members[0] == obj3


class TestSelectionManager:
    """Tests for SelectionManager class."""

    def test_manager_creation(self):
        """SelectionManager should be created with empty selection."""
        manager = SelectionManager()
        assert manager.count == 0
        assert manager.primary is None

    def test_manager_select(self):
        """Manager selection operations work."""
        manager = SelectionManager()
        obj = MockObject()

        assert manager.select(obj) is True
        assert manager.is_selected(obj) is True
        assert manager.count == 1

    def test_manager_add_remove_toggle(self):
        """Manager has convenience methods."""
        manager = SelectionManager()
        obj = MockObject()

        manager.add_to_selection(obj)
        assert manager.is_selected(obj) is True

        manager.remove_from_selection(obj)
        assert manager.is_selected(obj) is False

        manager.toggle_selection(obj)
        assert manager.is_selected(obj) is True

    def test_manager_clear(self):
        """Manager can clear selection."""
        manager = SelectionManager()
        manager.select(MockObject())

        manager.clear_selection()
        assert manager.count == 0

    def test_manager_selection_sets(self):
        """Manager can create and use selection sets."""
        manager = SelectionManager()
        obj1 = MockObject("1")
        obj2 = MockObject("2")

        manager.select_all([obj1, obj2])

        # Create set from current selection
        ss = manager.create_set("my_set")
        assert ss is not None
        assert ss.count == 2

        # Clear and restore
        manager.clear_selection()
        assert manager.restore_set("my_set") is True
        assert manager.count == 2

    def test_manager_delete_set(self):
        """Manager can delete selection sets."""
        manager = SelectionManager()
        manager.select(MockObject())
        manager.create_set("test")

        assert manager.delete_set("test") is True
        assert manager.get_set("test") is None

    def test_manager_add_set_to_selection(self):
        """Manager can add set to current selection."""
        manager = SelectionManager()
        obj1 = MockObject("1")
        obj2 = MockObject("2")
        obj3 = MockObject("3")

        manager.select_all([obj1, obj2])
        manager.create_set("partial")

        manager.clear_selection()
        manager.select(obj3)
        manager.add_set_to_selection("partial")

        assert manager.count == 3

    def test_manager_groups(self):
        """Manager can create and use groups."""
        manager = SelectionManager()
        obj1 = MockObject("1")
        obj2 = MockObject("2")

        manager.select_all([obj1, obj2])

        group = manager.create_group("Test Group")
        assert group.count == 2

        # Find groups containing object
        found = manager.find_groups_containing(obj1)
        assert len(found) == 1
        assert found[0] == group

    def test_manager_delete_group(self):
        """Manager can delete groups."""
        manager = SelectionManager()
        manager.select(MockObject())
        group = manager.create_group()

        assert manager.delete_group(group.id) is True
        assert manager.get_group(group.id) is None

    def test_manager_selection_history(self):
        """Manager tracks selection history."""
        manager = SelectionManager()
        obj1 = MockObject("1")
        obj2 = MockObject("2")

        manager.select(obj1)
        manager.select(obj2)

        # Undo
        assert manager.undo_selection() is True
        assert manager.is_selected(obj1) is True
        assert manager.is_selected(obj2) is False

        # Redo
        assert manager.redo_selection() is True
        assert manager.is_selected(obj2) is True

    def test_manager_selection_history_limit(self):
        """Selection history is limited."""
        manager = SelectionManager(max_history=5)

        for i in range(10):
            manager.select(MockObject(str(i)))

        # Should only be able to undo 5 times
        undo_count = 0
        while manager.undo_selection():
            undo_count += 1

        assert undo_count <= 5

    def test_manager_can_undo_redo_properties(self):
        """Manager reports undo/redo availability."""
        manager = SelectionManager()

        assert manager.can_undo_selection is False
        assert manager.can_redo_selection is False

        manager.select(MockObject())
        manager.select(MockObject())

        assert manager.can_undo_selection is True
        assert manager.can_redo_selection is False

        manager.undo_selection()
        assert manager.can_redo_selection is True

    def test_manager_change_callback(self):
        """Manager triggers change callback."""
        manager = SelectionManager()
        changes = []
        manager.on_selection_changed = lambda items: changes.append(len(items))

        manager.select(MockObject())
        assert len(changes) >= 1

    def test_manager_filter(self):
        """Manager filter affects selection."""
        manager = SelectionManager()
        filter = SelectionFilter()
        filter.allow_type(MockObject)
        manager.set_filter(filter)

        obj = MockObject()
        other = "not a mock"

        manager.add_to_selection(obj)
        manager.add_to_selection(other)

        assert manager.count == 1

    def test_manager_invert_selection(self):
        """Manager can invert selection."""
        manager = SelectionManager()
        all_objs = [MockObject(str(i)) for i in range(5)]

        manager.select_all(all_objs[:2])

        manager.invert_selection(all_objs)

        assert manager.count == 3
        assert all_objs[0] not in manager.selected_items
        assert all_objs[4] in manager.selected_items
