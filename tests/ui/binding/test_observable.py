"""
Comprehensive tests for observable collections.

Tests cover:
- ObservableList operations
- ObservableDict operations
- Change notifications
- Notification suspension
- VirtualizedListView
- Thread safety basics
"""
import threading
import pytest

from engine.ui.binding.observable import (
    CollectionChangeAction,
    CollectionChangeEvent,
    IObservableCollection,
    ObservableDict,
    ObservableList,
    VirtualizedListView,
)


# ========== Fixtures ==========


@pytest.fixture
def empty_list():
    """Empty observable list."""
    return ObservableList()


@pytest.fixture
def populated_list():
    """Observable list with initial data."""
    return ObservableList([1, 2, 3, 4, 5])


@pytest.fixture
def empty_dict():
    """Empty observable dict."""
    return ObservableDict()


@pytest.fixture
def populated_dict():
    """Observable dict with initial data."""
    return ObservableDict({"a": 1, "b": 2, "c": 3})


@pytest.fixture
def event_collector():
    """Collects change events for testing."""
    events = []

    def collector(event):
        events.append(event)

    collector.events = events
    return collector


# ========== CollectionChangeEvent Tests ==========


class TestCollectionChangeEvent:
    """Tests for CollectionChangeEvent class."""

    def test_add_event_factory(self):
        """Test creating add event."""
        event = CollectionChangeEvent.add([1, 2], 0)
        assert event.action == CollectionChangeAction.ADD
        assert event.new_items == [1, 2]
        assert event.new_starting_index == 0

    def test_remove_event_factory(self):
        """Test creating remove event."""
        event = CollectionChangeEvent.remove([1], 0)
        assert event.action == CollectionChangeAction.REMOVE
        assert event.old_items == [1]
        assert event.old_starting_index == 0

    def test_replace_event_factory(self):
        """Test creating replace event."""
        event = CollectionChangeEvent.replace(1, 2, 0)
        assert event.action == CollectionChangeAction.REPLACE
        assert event.old_items == [1]
        assert event.new_items == [2]

    def test_move_event_factory(self):
        """Test creating move event."""
        event = CollectionChangeEvent.move(1, 0, 2)
        assert event.action == CollectionChangeAction.MOVE
        assert event.old_starting_index == 0
        assert event.new_starting_index == 2

    def test_reset_event_factory(self):
        """Test creating reset event."""
        event = CollectionChangeEvent.reset()
        assert event.action == CollectionChangeAction.RESET


# ========== ObservableList Tests ==========


class TestObservableList:
    """Tests for ObservableList class."""

    def test_create_empty(self, empty_list):
        """Test creating empty list."""
        assert len(empty_list) == 0

    def test_create_with_initial(self, populated_list):
        """Test creating list with initial data."""
        assert len(populated_list) == 5
        assert populated_list[0] == 1

    def test_getitem(self, populated_list):
        """Test accessing items by index."""
        assert populated_list[0] == 1
        assert populated_list[-1] == 5

    def test_getitem_slice(self, populated_list):
        """Test slicing list."""
        assert populated_list[1:3] == [2, 3]

    def test_setitem(self, populated_list, event_collector):
        """Test setting item by index."""
        populated_list.add_listener(event_collector)
        populated_list[0] = 10

        assert populated_list[0] == 10
        assert len(event_collector.events) == 1
        assert event_collector.events[0].action == CollectionChangeAction.REPLACE

    def test_setitem_slice(self, populated_list, event_collector):
        """Test setting slice triggers reset."""
        populated_list.add_listener(event_collector)
        populated_list[1:3] = [20, 30]

        assert populated_list[1:3] == [20, 30]
        assert event_collector.events[0].action == CollectionChangeAction.RESET

    def test_delitem(self, populated_list, event_collector):
        """Test deleting item by index."""
        populated_list.add_listener(event_collector)
        del populated_list[0]

        assert len(populated_list) == 4
        assert populated_list[0] == 2
        assert event_collector.events[0].action == CollectionChangeAction.REMOVE

    def test_iter(self, populated_list):
        """Test iterating over list."""
        items = list(populated_list)
        assert items == [1, 2, 3, 4, 5]

    def test_contains(self, populated_list):
        """Test membership testing."""
        assert 3 in populated_list
        assert 10 not in populated_list

    def test_eq_observable_list(self, populated_list):
        """Test equality with another ObservableList."""
        other = ObservableList([1, 2, 3, 4, 5])
        assert populated_list == other

    def test_eq_regular_list(self, populated_list):
        """Test equality with regular list."""
        assert populated_list == [1, 2, 3, 4, 5]

    def test_insert(self, empty_list, event_collector):
        """Test inserting item."""
        empty_list.add_listener(event_collector)
        empty_list.insert(0, 1)

        assert len(empty_list) == 1
        assert empty_list[0] == 1
        assert event_collector.events[0].action == CollectionChangeAction.ADD

    def test_append(self, empty_list, event_collector):
        """Test appending item."""
        empty_list.add_listener(event_collector)
        empty_list.append(1)

        assert len(empty_list) == 1
        assert event_collector.events[0].new_starting_index == 0

    def test_extend(self, empty_list, event_collector):
        """Test extending list."""
        empty_list.add_listener(event_collector)
        empty_list.extend([1, 2, 3])

        assert len(empty_list) == 3
        assert event_collector.events[0].new_items == [1, 2, 3]

    def test_pop_default(self, populated_list, event_collector):
        """Test popping last item."""
        populated_list.add_listener(event_collector)
        item = populated_list.pop()

        assert item == 5
        assert len(populated_list) == 4
        assert event_collector.events[0].action == CollectionChangeAction.REMOVE

    def test_pop_index(self, populated_list, event_collector):
        """Test popping item at index."""
        populated_list.add_listener(event_collector)
        item = populated_list.pop(0)

        assert item == 1
        assert event_collector.events[0].old_starting_index == 0

    def test_remove(self, populated_list, event_collector):
        """Test removing item by value."""
        populated_list.add_listener(event_collector)
        populated_list.remove(3)

        assert 3 not in populated_list
        assert len(populated_list) == 4

    def test_clear(self, populated_list, event_collector):
        """Test clearing list."""
        populated_list.add_listener(event_collector)
        populated_list.clear()

        assert len(populated_list) == 0
        assert event_collector.events[0].action == CollectionChangeAction.RESET

    def test_index(self, populated_list):
        """Test finding index of item."""
        assert populated_list.index(3) == 2
        assert populated_list.index(3, 0, 5) == 2

    def test_count(self):
        """Test counting occurrences."""
        obs_list = ObservableList([1, 2, 2, 3, 2])
        assert obs_list.count(2) == 3

    def test_reverse(self, populated_list, event_collector):
        """Test reversing list."""
        populated_list.add_listener(event_collector)
        populated_list.reverse()

        assert populated_list[0] == 5
        assert event_collector.events[0].action == CollectionChangeAction.RESET

    def test_sort(self, event_collector):
        """Test sorting list."""
        obs_list = ObservableList([3, 1, 4, 1, 5])
        obs_list.add_listener(event_collector)
        obs_list.sort()

        assert obs_list[0] == 1
        assert event_collector.events[0].action == CollectionChangeAction.RESET

    def test_sort_reverse(self):
        """Test sorting in reverse."""
        obs_list = ObservableList([1, 2, 3])
        obs_list.sort(reverse=True)
        assert obs_list[0] == 3

    def test_sort_key(self):
        """Test sorting with key function."""
        obs_list = ObservableList(["bb", "a", "ccc"])
        obs_list.sort(key=len)
        assert obs_list[0] == "a"

    def test_copy(self, populated_list):
        """Test copying list data."""
        copied = populated_list.copy()
        assert copied == [1, 2, 3, 4, 5]
        assert isinstance(copied, list)

    def test_move(self, populated_list, event_collector):
        """Test moving item."""
        populated_list.add_listener(event_collector)
        populated_list.move(0, 2)

        assert populated_list[2] == 1
        assert event_collector.events[0].action == CollectionChangeAction.MOVE

    def test_move_same_index(self, populated_list, event_collector):
        """Test moving to same index does nothing."""
        populated_list.add_listener(event_collector)
        populated_list.move(0, 0)

        assert len(event_collector.events) == 0


# ========== ObservableList Listener Tests ==========


class TestObservableListListeners:
    """Tests for ObservableList listener functionality."""

    def test_add_listener(self, empty_list, event_collector):
        """Test adding a listener."""
        empty_list.add_listener(event_collector)
        empty_list.append(1)

        assert len(event_collector.events) == 1

    def test_remove_listener(self, empty_list, event_collector):
        """Test removing a listener."""
        empty_list.add_listener(event_collector)
        empty_list.remove_listener(event_collector)
        empty_list.append(1)

        assert len(event_collector.events) == 0

    def test_add_listener_duplicate(self, empty_list, event_collector):
        """Test adding same listener twice is ignored."""
        empty_list.add_listener(event_collector)
        empty_list.add_listener(event_collector)
        empty_list.append(1)

        assert len(event_collector.events) == 1

    def test_listener_error_silenced(self, empty_list):
        """Test listener errors are silenced."""

        def bad_listener(event):
            raise Exception("Listener error")

        empty_list.add_listener(bad_listener)
        # Should not raise
        empty_list.append(1)

    def test_suspend_notifications(self, empty_list, event_collector):
        """Test suspending notifications."""
        empty_list.add_listener(event_collector)
        empty_list.suspend_notifications()

        empty_list.append(1)
        empty_list.append(2)

        assert len(event_collector.events) == 0

    def test_resume_notifications(self, empty_list, event_collector):
        """Test resuming notifications sends reset."""
        empty_list.add_listener(event_collector)
        empty_list.suspend_notifications()

        empty_list.append(1)
        empty_list.append(2)

        empty_list.resume_notifications()

        # Should get a reset event
        assert len(event_collector.events) == 1
        assert event_collector.events[0].action == CollectionChangeAction.RESET

    def test_resume_without_changes(self, empty_list, event_collector):
        """Test resuming with no pending changes sends nothing."""
        empty_list.add_listener(event_collector)
        empty_list.suspend_notifications()
        empty_list.resume_notifications()

        assert len(event_collector.events) == 0


# ========== ObservableList Virtualization Tests ==========


class TestObservableListVirtualization:
    """Tests for ObservableList virtualization support."""

    def test_get_range(self, populated_list):
        """Test getting range of items."""
        items = populated_list.get_range(1, 3)
        assert items == [2, 3, 4]

    def test_get_range_at_end(self, populated_list):
        """Test getting range at end of list."""
        items = populated_list.get_range(3, 10)
        assert items == [4, 5]

    def test_total_count(self, populated_list):
        """Test total count property."""
        assert populated_list.total_count == 5


# ========== ObservableDict Tests ==========


class TestObservableDict:
    """Tests for ObservableDict class."""

    def test_create_empty(self, empty_dict):
        """Test creating empty dict."""
        assert len(empty_dict) == 0

    def test_create_with_initial(self, populated_dict):
        """Test creating dict with initial data."""
        assert len(populated_dict) == 3
        assert populated_dict["a"] == 1

    def test_getitem(self, populated_dict):
        """Test getting item by key."""
        assert populated_dict["a"] == 1

    def test_getitem_missing(self, populated_dict):
        """Test getting missing key raises KeyError."""
        with pytest.raises(KeyError):
            _ = populated_dict["missing"]

    def test_setitem_new(self, empty_dict, event_collector):
        """Test setting new key triggers add."""
        empty_dict.add_listener(event_collector)
        empty_dict["a"] = 1

        assert empty_dict["a"] == 1
        assert event_collector.events[0].action == CollectionChangeAction.ADD

    def test_setitem_existing(self, populated_dict, event_collector):
        """Test updating existing key triggers replace."""
        populated_dict.add_listener(event_collector)
        populated_dict["a"] = 10

        assert populated_dict["a"] == 10
        assert event_collector.events[0].action == CollectionChangeAction.REPLACE

    def test_delitem(self, populated_dict, event_collector):
        """Test deleting item by key."""
        populated_dict.add_listener(event_collector)
        del populated_dict["a"]

        assert "a" not in populated_dict
        assert event_collector.events[0].action == CollectionChangeAction.REMOVE

    def test_iter(self, populated_dict):
        """Test iterating over keys."""
        keys = list(populated_dict)
        assert "a" in keys
        assert "b" in keys
        assert "c" in keys

    def test_contains(self, populated_dict):
        """Test membership testing."""
        assert "a" in populated_dict
        assert "z" not in populated_dict

    def test_eq_observable_dict(self, populated_dict):
        """Test equality with another ObservableDict."""
        other = ObservableDict({"a": 1, "b": 2, "c": 3})
        assert populated_dict == other

    def test_eq_regular_dict(self, populated_dict):
        """Test equality with regular dict."""
        assert populated_dict == {"a": 1, "b": 2, "c": 3}

    def test_keys(self, populated_dict):
        """Test getting keys view."""
        keys = populated_dict.keys()
        assert set(keys) == {"a", "b", "c"}

    def test_values(self, populated_dict):
        """Test getting values view."""
        values = populated_dict.values()
        assert set(values) == {1, 2, 3}

    def test_items(self, populated_dict):
        """Test getting items view."""
        items = list(populated_dict.items())
        assert ("a", 1) in items

    def test_get_existing(self, populated_dict):
        """Test getting existing key."""
        assert populated_dict.get("a") == 1

    def test_get_missing_default(self, populated_dict):
        """Test getting missing key with default."""
        assert populated_dict.get("z", 100) == 100

    def test_pop(self, populated_dict, event_collector):
        """Test popping item."""
        populated_dict.add_listener(event_collector)
        value = populated_dict.pop("a")

        assert value == 1
        assert "a" not in populated_dict
        assert event_collector.events[0].action == CollectionChangeAction.REMOVE

    def test_pop_missing_default(self, populated_dict):
        """Test popping missing key with default."""
        value = populated_dict.pop("z", 100)
        assert value == 100

    def test_pop_missing_no_default(self, populated_dict):
        """Test popping missing key without default raises KeyError."""
        with pytest.raises(KeyError):
            populated_dict.pop("z")

    def test_popitem(self, populated_dict, event_collector):
        """Test popping arbitrary item."""
        populated_dict.add_listener(event_collector)
        key, value = populated_dict.popitem()

        assert key not in populated_dict
        assert event_collector.events[0].action == CollectionChangeAction.REMOVE

    def test_clear(self, populated_dict, event_collector):
        """Test clearing dict."""
        populated_dict.add_listener(event_collector)
        populated_dict.clear()

        assert len(populated_dict) == 0
        assert event_collector.events[0].action == CollectionChangeAction.RESET

    def test_update(self, populated_dict, event_collector):
        """Test updating dict."""
        populated_dict.add_listener(event_collector)
        populated_dict.update({"d": 4, "e": 5})

        assert populated_dict["d"] == 4
        assert event_collector.events[0].action == CollectionChangeAction.RESET

    def test_setdefault_new(self, populated_dict, event_collector):
        """Test setdefault with new key."""
        populated_dict.add_listener(event_collector)
        value = populated_dict.setdefault("d", 4)

        assert value == 4
        assert populated_dict["d"] == 4
        assert event_collector.events[0].action == CollectionChangeAction.ADD

    def test_setdefault_existing(self, populated_dict, event_collector):
        """Test setdefault with existing key."""
        populated_dict.add_listener(event_collector)
        value = populated_dict.setdefault("a", 100)

        assert value == 1  # Original value
        assert len(event_collector.events) == 0

    def test_copy(self, populated_dict):
        """Test copying dict data."""
        copied = populated_dict.copy()
        assert copied == {"a": 1, "b": 2, "c": 3}
        assert isinstance(copied, dict)


# ========== ObservableDict Listener Tests ==========


class TestObservableDictListeners:
    """Tests for ObservableDict listener functionality."""

    def test_add_listener(self, empty_dict, event_collector):
        """Test adding a listener."""
        empty_dict.add_listener(event_collector)
        empty_dict["a"] = 1

        assert len(event_collector.events) == 1

    def test_remove_listener(self, empty_dict, event_collector):
        """Test removing a listener."""
        empty_dict.add_listener(event_collector)
        empty_dict.remove_listener(event_collector)
        empty_dict["a"] = 1

        assert len(event_collector.events) == 0

    def test_suspend_resume(self, empty_dict, event_collector):
        """Test suspend and resume notifications."""
        empty_dict.add_listener(event_collector)
        empty_dict.suspend_notifications()

        empty_dict["a"] = 1
        empty_dict["b"] = 2

        assert len(event_collector.events) == 0

        empty_dict.resume_notifications()
        assert event_collector.events[0].action == CollectionChangeAction.RESET


# ========== VirtualizedListView Tests ==========


class TestVirtualizedListView:
    """Tests for VirtualizedListView class."""

    def test_create_view(self, populated_list):
        """Test creating virtualized view."""
        view = VirtualizedListView(populated_list, visible_count=3)
        assert view._visible_count == 3

    def test_scroll_offset(self, populated_list):
        """Test scroll offset property."""
        view = VirtualizedListView(populated_list, visible_count=3)
        view.scroll_offset = 2

        assert view.scroll_offset == 2

    def test_scroll_offset_clamped_low(self, populated_list):
        """Test scroll offset clamped to 0."""
        view = VirtualizedListView(populated_list, visible_count=3)
        view.scroll_offset = -5

        assert view.scroll_offset == 0

    def test_scroll_offset_clamped_high(self, populated_list):
        """Test scroll offset clamped to max."""
        view = VirtualizedListView(populated_list, visible_count=3)
        view.scroll_offset = 100

        assert view.scroll_offset == 2  # 5 items, show 3, max scroll = 2

    def test_visible_range(self, populated_list):
        """Test getting visible range."""
        view = VirtualizedListView(populated_list, visible_count=3)
        view.scroll_offset = 1

        start, end = view.visible_range
        assert start == 1
        assert end == 4

    def test_visible_items(self, populated_list):
        """Test getting visible items."""
        view = VirtualizedListView(populated_list, visible_count=3)
        view.scroll_offset = 1

        items = view.visible_items
        assert items == [2, 3, 4]

    def test_total_height(self, populated_list):
        """Test total scrollable height."""
        view = VirtualizedListView(populated_list, visible_count=3, item_height=20.0)
        assert view.total_height == 100.0  # 5 * 20

    def test_viewport_height(self, populated_list):
        """Test viewport height."""
        view = VirtualizedListView(populated_list, visible_count=3, item_height=20.0)
        assert view.viewport_height == 60.0  # 3 * 20

    def test_scroll_by(self, populated_list):
        """Test scrolling by delta."""
        view = VirtualizedListView(populated_list, visible_count=3)
        view.scroll_by(2)

        assert view.scroll_offset == 2

    def test_scroll_to_item_above(self, populated_list):
        """Test scrolling to item above viewport."""
        view = VirtualizedListView(populated_list, visible_count=3)
        view.scroll_offset = 2
        view.scroll_to_item(0)

        assert view.scroll_offset == 0

    def test_scroll_to_item_below(self, populated_list):
        """Test scrolling to item below viewport."""
        view = VirtualizedListView(populated_list, visible_count=3)
        view.scroll_to_item(4)

        assert view.scroll_offset == 2

    def test_widget_recycling(self, populated_list):
        """Test widget recycling pool."""
        view = VirtualizedListView(populated_list, visible_count=3)

        # Add widget to pool
        view.recycle_widget("widget1")
        widget = view.acquire_widget()

        assert widget == "widget1"

    def test_acquire_widget_empty(self, populated_list):
        """Test acquiring from empty pool."""
        view = VirtualizedListView(populated_list, visible_count=3)
        widget = view.acquire_widget()

        assert widget is None

    def test_bind_unbind_widget(self, populated_list):
        """Test binding and unbinding widgets."""
        view = VirtualizedListView(populated_list, visible_count=3)

        view.bind_widget(0, "widget0")
        assert view.get_bound_widget(0) == "widget0"

        widget = view.unbind_widget(0)
        assert widget == "widget0"
        assert view.get_bound_widget(0) is None

    def test_range_listener(self, populated_list):
        """Test range change listener."""
        view = VirtualizedListView(populated_list, visible_count=3)
        changes = []

        def on_change(start, end):
            changes.append((start, end))

        view.add_range_listener(on_change)
        view.scroll_offset = 1

        assert len(changes) == 1
        assert changes[0] == (1, 4)

    def test_remove_range_listener(self, populated_list):
        """Test removing range change listener."""
        view = VirtualizedListView(populated_list, visible_count=3)
        changes = []

        def on_change(start, end):
            changes.append((start, end))

        view.add_range_listener(on_change)
        view.remove_range_listener(on_change)
        view.scroll_offset = 1

        assert len(changes) == 0

    def test_source_change_adjusts_scroll(self, populated_list):
        """Test source changes adjust scroll offset."""
        view = VirtualizedListView(populated_list, visible_count=3)
        view.scroll_offset = 2

        # Remove items, scroll should adjust
        populated_list.clear()

        assert view.scroll_offset == 0

    def test_dispose(self, populated_list):
        """Test disposing view."""
        view = VirtualizedListView(populated_list, visible_count=3)
        view.recycle_widget("widget")
        view.bind_widget(0, "widget0")

        view.dispose()

        assert len(view._recycled_widgets) == 0
        assert len(view._active_widgets) == 0


# ========== Thread Safety Basic Tests ==========


class TestThreadSafety:
    """Basic thread safety tests."""

    def test_list_concurrent_append(self, empty_list):
        """Test concurrent appends don't crash."""

        def append_items():
            for i in range(100):
                empty_list.append(i)

        threads = [threading.Thread(target=append_items) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All items should be present
        assert len(empty_list) == 300

    def test_dict_concurrent_updates(self, empty_dict):
        """Test concurrent dict updates don't crash."""

        def update_items(prefix):
            for i in range(100):
                empty_dict[f"{prefix}_{i}"] = i

        threads = [
            threading.Thread(target=update_items, args=(f"t{i}",))
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(empty_dict) == 300
