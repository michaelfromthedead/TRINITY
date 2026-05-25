"""
Comprehensive tests for the focus management system.

Tests cover:
- FocusManager singleton
- Tab navigation
- Focus groups
- Focus trapping
- Focus events
- Keyboard navigation

Note: This module tests the focus system defined in engine/ui/framework/focus.py.
These tests serve as specifications for the expected focus implementation.
"""

import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

# Import will fail until focus.py is implemented
try:
    from engine.ui.framework.focus import (
        FocusManager,
        FocusGroup,
        FocusNavigator,
        FocusDirection,
        get_focus_manager,
    )
    from engine.ui.framework.widget import Widget
    FOCUS_AVAILABLE = True
except ImportError:
    FOCUS_AVAILABLE = False
    FocusManager = None
    FocusGroup = None
    FocusNavigator = None
    Widget = None

from engine.ui.framework.events import FocusEvent, EventType


pytestmark = pytest.mark.skipif(
    not FOCUS_AVAILABLE,
    reason="Focus system not yet implemented"
)


@pytest.fixture
def focus_manager():
    """Get or create focus manager."""
    manager = get_focus_manager()
    manager.clear()
    return manager


@pytest.fixture
def focusable_widget():
    """Create a focusable widget."""
    w = Widget()
    w.focusable = True
    return w


@pytest.fixture
def non_focusable_widget():
    """Create a non-focusable widget."""
    w = Widget()
    w.focusable = False
    return w


@pytest.fixture
def focus_group():
    """Create a focus group."""
    return FocusGroup()


class TestFocusManagerSingleton:
    """Tests for FocusManager singleton behavior."""

    def test_get_focus_manager(self):
        """get_focus_manager should return FocusManager instance."""
        manager = get_focus_manager()
        assert manager is not None
        assert isinstance(manager, FocusManager)

    def test_singleton_same_instance(self):
        """get_focus_manager should return same instance."""
        m1 = get_focus_manager()
        m2 = get_focus_manager()
        assert m1 is m2

    def test_focus_manager_clear(self, focus_manager):
        """FocusManager clear should reset state."""
        w = Widget()
        w.focusable = True
        focus_manager.set_focus(w)

        focus_manager.clear()

        assert focus_manager.focused_widget is None

    def test_focus_manager_has_focus(self, focus_manager, focusable_widget):
        """FocusManager should track if anything has focus."""
        assert focus_manager.has_focus is False

        focus_manager.set_focus(focusable_widget)
        assert focus_manager.has_focus is True


class TestFocusSetting:
    """Tests for setting focus."""

    def test_set_focus(self, focus_manager, focusable_widget):
        """set_focus should set focus to widget."""
        focus_manager.set_focus(focusable_widget)
        assert focus_manager.focused_widget is focusable_widget

    def test_set_focus_non_focusable_fails(self, focus_manager, non_focusable_widget):
        """set_focus on non-focusable widget should fail."""
        result = focus_manager.set_focus(non_focusable_widget)
        assert result is False
        assert focus_manager.focused_widget is None

    def test_set_focus_disabled_fails(self, focus_manager, focusable_widget):
        """set_focus on disabled widget should fail."""
        focusable_widget.enabled = False
        result = focus_manager.set_focus(focusable_widget)
        assert result is False

    def test_set_focus_invisible_fails(self, focus_manager, focusable_widget):
        """set_focus on invisible widget should fail."""
        focusable_widget.visible = False
        result = focus_manager.set_focus(focusable_widget)
        assert result is False

    def test_set_focus_clears_previous(self, focus_manager):
        """set_focus should clear previous focus."""
        w1 = Widget()
        w1.focusable = True
        w2 = Widget()
        w2.focusable = True

        focus_manager.set_focus(w1)
        focus_manager.set_focus(w2)

        assert focus_manager.focused_widget is w2
        assert focus_manager.previously_focused is w1

    def test_clear_focus(self, focus_manager, focusable_widget):
        """clear_focus should remove focus."""
        focus_manager.set_focus(focusable_widget)
        focus_manager.clear_focus()

        assert focus_manager.focused_widget is None

    def test_focus_none(self, focus_manager, focusable_widget):
        """set_focus(None) should clear focus."""
        focus_manager.set_focus(focusable_widget)
        focus_manager.set_focus(None)

        assert focus_manager.focused_widget is None


class TestFocusEvents:
    """Tests for focus events."""

    def test_focus_in_event_fired(self, focus_manager, focusable_widget):
        """Setting focus should fire FOCUS_IN event."""
        events = []
        focusable_widget.add_event_listener("focus_in", lambda e: events.append(e))

        focus_manager.set_focus(focusable_widget)

        assert len(events) == 1
        assert events[0].event_type == EventType.FOCUS_IN

    def test_focus_out_event_fired(self, focus_manager, focusable_widget):
        """Clearing focus should fire FOCUS_OUT event."""
        events = []
        focusable_widget.add_event_listener("focus_out", lambda e: events.append(e))

        focus_manager.set_focus(focusable_widget)
        focus_manager.clear_focus()

        assert len(events) == 1
        assert events[0].event_type == EventType.FOCUS_OUT

    def test_focus_change_related_target(self, focus_manager):
        """Focus events should have related_target set."""
        w1 = Widget()
        w1.focusable = True
        w2 = Widget()
        w2.focusable = True

        focus_out_events = []
        focus_in_events = []
        w1.add_event_listener("focus_out", lambda e: focus_out_events.append(e))
        w2.add_event_listener("focus_in", lambda e: focus_in_events.append(e))

        focus_manager.set_focus(w1)
        focus_manager.set_focus(w2)

        assert focus_out_events[0].related_target is w2
        assert focus_in_events[0].related_target is w1

    def test_no_event_if_same_widget(self, focus_manager, focusable_widget):
        """Setting focus to already focused widget should not fire events."""
        events = []
        focusable_widget.add_event_listener("focus_in", lambda e: events.append(e))

        focus_manager.set_focus(focusable_widget)
        focus_manager.set_focus(focusable_widget)  # Same widget

        assert len(events) == 1  # Only one event from first focus


class TestTabNavigation:
    """Tests for tab navigation."""

    def test_focus_next(self, focus_manager):
        """focus_next should move to next focusable widget."""
        w1 = Widget()
        w1.focusable = True
        w1.tab_index = 0
        w2 = Widget()
        w2.focusable = True
        w2.tab_index = 1
        w3 = Widget()
        w3.focusable = True
        w3.tab_index = 2

        focus_manager.register(w1)
        focus_manager.register(w2)
        focus_manager.register(w3)
        focus_manager.set_focus(w1)

        focus_manager.focus_next()
        assert focus_manager.focused_widget is w2

        focus_manager.focus_next()
        assert focus_manager.focused_widget is w3

    def test_focus_previous(self, focus_manager):
        """focus_previous should move to previous focusable widget."""
        w1 = Widget()
        w1.focusable = True
        w1.tab_index = 0
        w2 = Widget()
        w2.focusable = True
        w2.tab_index = 1

        focus_manager.register(w1)
        focus_manager.register(w2)
        focus_manager.set_focus(w2)

        focus_manager.focus_previous()
        assert focus_manager.focused_widget is w1

    def test_focus_next_wraps(self, focus_manager):
        """focus_next should wrap to first widget."""
        w1 = Widget()
        w1.focusable = True
        w1.tab_index = 0
        w2 = Widget()
        w2.focusable = True
        w2.tab_index = 1

        focus_manager.register(w1)
        focus_manager.register(w2)
        focus_manager.set_focus(w2)

        focus_manager.focus_next()
        assert focus_manager.focused_widget is w1

    def test_focus_previous_wraps(self, focus_manager):
        """focus_previous should wrap to last widget."""
        w1 = Widget()
        w1.focusable = True
        w1.tab_index = 0
        w2 = Widget()
        w2.focusable = True
        w2.tab_index = 1

        focus_manager.register(w1)
        focus_manager.register(w2)
        focus_manager.set_focus(w1)

        focus_manager.focus_previous()
        assert focus_manager.focused_widget is w2

    def test_skip_non_focusable(self, focus_manager):
        """Tab navigation should skip non-focusable widgets."""
        w1 = Widget()
        w1.focusable = True
        w1.tab_index = 0
        w2 = Widget()
        w2.focusable = False
        w2.tab_index = 1
        w3 = Widget()
        w3.focusable = True
        w3.tab_index = 2

        focus_manager.register(w1)
        focus_manager.register(w2)
        focus_manager.register(w3)
        focus_manager.set_focus(w1)

        focus_manager.focus_next()
        assert focus_manager.focused_widget is w3  # Skipped w2

    def test_skip_disabled(self, focus_manager):
        """Tab navigation should skip disabled widgets."""
        w1 = Widget()
        w1.focusable = True
        w1.tab_index = 0
        w2 = Widget()
        w2.focusable = True
        w2.tab_index = 1
        w2.enabled = False
        w3 = Widget()
        w3.focusable = True
        w3.tab_index = 2

        focus_manager.register(w1)
        focus_manager.register(w2)
        focus_manager.register(w3)
        focus_manager.set_focus(w1)

        focus_manager.focus_next()
        assert focus_manager.focused_widget is w3

    def test_tab_index_order(self, focus_manager):
        """Widgets should be ordered by tab_index."""
        w1 = Widget()
        w1.focusable = True
        w1.tab_index = 2
        w2 = Widget()
        w2.focusable = True
        w2.tab_index = 0
        w3 = Widget()
        w3.focusable = True
        w3.tab_index = 1

        focus_manager.register(w1)
        focus_manager.register(w2)
        focus_manager.register(w3)

        # Focus first (lowest tab_index)
        focus_manager.focus_first()
        assert focus_manager.focused_widget is w2

        focus_manager.focus_next()
        assert focus_manager.focused_widget is w3

        focus_manager.focus_next()
        assert focus_manager.focused_widget is w1

    def test_negative_tab_index_excluded(self, focus_manager):
        """Widgets with negative tab_index should be excluded from tab order."""
        w1 = Widget()
        w1.focusable = True
        w1.tab_index = 0
        w2 = Widget()
        w2.focusable = True
        w2.tab_index = -1  # Excluded
        w3 = Widget()
        w3.focusable = True
        w3.tab_index = 1

        focus_manager.register(w1)
        focus_manager.register(w2)
        focus_manager.register(w3)
        focus_manager.set_focus(w1)

        focus_manager.focus_next()
        assert focus_manager.focused_widget is w3  # Skipped w2


class TestFocusGroups:
    """Tests for focus groups."""

    def test_create_focus_group(self):
        """FocusGroup should be creatable."""
        group = FocusGroup()
        assert group is not None

    def test_focus_group_with_name(self):
        """FocusGroup can have a name."""
        group = FocusGroup(name="toolbar")
        assert group.name == "toolbar"

    def test_add_widget_to_group(self, focus_group):
        """Widgets can be added to focus group."""
        w = Widget()
        w.focusable = True
        focus_group.add(w)

        assert w in focus_group.widgets
        assert w.focus_group is focus_group

    def test_remove_widget_from_group(self, focus_group):
        """Widgets can be removed from focus group."""
        w = Widget()
        w.focusable = True
        focus_group.add(w)
        focus_group.remove(w)

        assert w not in focus_group.widgets
        assert w.focus_group is None

    def test_focus_group_navigation(self, focus_manager, focus_group):
        """Navigation within group should stay in group."""
        w1 = Widget()
        w1.focusable = True
        w2 = Widget()
        w2.focusable = True
        w3 = Widget()
        w3.focusable = True

        focus_group.add(w1)
        focus_group.add(w2)
        focus_group.add(w3)

        focus_manager.set_focus(w1)
        focus_manager.focus_next()

        assert focus_manager.focused_widget is w2

    def test_focus_group_isolation(self, focus_manager):
        """Groups should be isolated from each other."""
        group1 = FocusGroup(name="group1")
        group2 = FocusGroup(name="group2")

        w1 = Widget()
        w1.focusable = True
        w2 = Widget()
        w2.focusable = True

        group1.add(w1)
        group2.add(w2)

        focus_manager.set_focus(w1)
        # Navigation should stay in group1
        focus_manager.focus_next()

        # Should wrap within group1, not go to group2
        assert focus_manager.focused_widget is w1

    def test_focus_group_enter(self, focus_manager, focus_group):
        """Entering group should focus first widget."""
        w1 = Widget()
        w1.focusable = True
        w2 = Widget()
        w2.focusable = True

        focus_group.add(w1)
        focus_group.add(w2)

        focus_manager.enter_group(focus_group)

        assert focus_manager.focused_widget is w1

    def test_focus_group_exit(self, focus_manager, focus_group):
        """Exiting group should restore previous focus."""
        outer_widget = Widget()
        outer_widget.focusable = True
        inner_widget = Widget()
        inner_widget.focusable = True

        focus_group.add(inner_widget)
        focus_manager.register(outer_widget)

        focus_manager.set_focus(outer_widget)
        focus_manager.enter_group(focus_group)
        focus_manager.exit_group()

        assert focus_manager.focused_widget is outer_widget


class TestFocusTrapping:
    """Tests for focus trapping."""

    def test_trap_focus(self, focus_manager, focus_group):
        """trap_focus should keep focus within group."""
        w1 = Widget()
        w1.focusable = True
        w2 = Widget()
        w2.focusable = True
        outside = Widget()
        outside.focusable = True

        focus_group.add(w1)
        focus_group.add(w2)
        focus_manager.register(outside)

        focus_manager.trap_focus(focus_group)
        focus_manager.set_focus(w1)

        # Attempt to focus outside widget should fail
        result = focus_manager.set_focus(outside)
        assert result is False
        assert focus_manager.focused_widget is w1

    def test_release_trap(self, focus_manager, focus_group):
        """release_trap should allow focus outside."""
        w1 = Widget()
        w1.focusable = True
        outside = Widget()
        outside.focusable = True

        focus_group.add(w1)
        focus_manager.register(outside)

        focus_manager.trap_focus(focus_group)
        focus_manager.set_focus(w1)
        focus_manager.release_trap()

        result = focus_manager.set_focus(outside)
        assert result is True

    def test_trap_wraps_at_boundaries(self, focus_manager, focus_group):
        """Focus should wrap within trapped group."""
        w1 = Widget()
        w1.focusable = True
        w1.tab_index = 0
        w2 = Widget()
        w2.focusable = True
        w2.tab_index = 1

        focus_group.add(w1)
        focus_group.add(w2)

        focus_manager.trap_focus(focus_group)
        focus_manager.set_focus(w2)
        focus_manager.focus_next()

        assert focus_manager.focused_widget is w1  # Wrapped


class TestKeyboardNavigation:
    """Tests for keyboard-based focus navigation."""

    def test_handle_tab_key(self, focus_manager):
        """Tab key should move focus forward."""
        w1 = Widget()
        w1.focusable = True
        w2 = Widget()
        w2.focusable = True

        focus_manager.register(w1)
        focus_manager.register(w2)
        focus_manager.set_focus(w1)

        focus_manager.handle_key("Tab")
        assert focus_manager.focused_widget is w2

    def test_handle_shift_tab_key(self, focus_manager):
        """Shift+Tab should move focus backward."""
        w1 = Widget()
        w1.focusable = True
        w2 = Widget()
        w2.focusable = True

        focus_manager.register(w1)
        focus_manager.register(w2)
        focus_manager.set_focus(w2)

        focus_manager.handle_key("Tab", shift=True)
        assert focus_manager.focused_widget is w1

    def test_handle_escape_clears_focus(self, focus_manager, focusable_widget):
        """Escape key should clear focus (if configured)."""
        focus_manager.escape_clears_focus = True
        focus_manager.set_focus(focusable_widget)

        focus_manager.handle_key("Escape")
        assert focus_manager.focused_widget is None

    def test_arrow_key_navigation(self, focus_manager):
        """Arrow keys should navigate spatially."""
        w1 = Widget(local_x=0.0, local_y=0.0)
        w1.focusable = True
        w2 = Widget(local_x=100.0, local_y=0.0)  # Right of w1
        w2.focusable = True
        w3 = Widget(local_x=0.0, local_y=100.0)  # Below w1
        w3.focusable = True

        focus_manager.register(w1)
        focus_manager.register(w2)
        focus_manager.register(w3)
        focus_manager.set_focus(w1)

        focus_manager.handle_key("ArrowRight")
        assert focus_manager.focused_widget is w2

        focus_manager.set_focus(w1)
        focus_manager.handle_key("ArrowDown")
        assert focus_manager.focused_widget is w3


class TestFocusNavigator:
    """Tests for FocusNavigator utility."""

    def test_find_next_focusable(self):
        """find_next_focusable should find next in list."""
        widgets = [Widget() for _ in range(3)]
        for i, w in enumerate(widgets):
            w.focusable = True
            w.tab_index = i

        navigator = FocusNavigator(widgets)
        next_widget = navigator.find_next(widgets[0])

        assert next_widget is widgets[1]

    def test_find_previous_focusable(self):
        """find_previous_focusable should find previous in list."""
        widgets = [Widget() for _ in range(3)]
        for i, w in enumerate(widgets):
            w.focusable = True
            w.tab_index = i

        navigator = FocusNavigator(widgets)
        prev_widget = navigator.find_previous(widgets[2])

        assert prev_widget is widgets[1]

    def test_find_first_focusable(self):
        """find_first should return first focusable widget."""
        widgets = [Widget() for _ in range(3)]
        widgets[0].focusable = False
        widgets[1].focusable = True
        widgets[2].focusable = True

        navigator = FocusNavigator(widgets)
        first = navigator.find_first()

        assert first is widgets[1]

    def test_find_last_focusable(self):
        """find_last should return last focusable widget."""
        widgets = [Widget() for _ in range(3)]
        widgets[0].focusable = True
        widgets[1].focusable = True
        widgets[2].focusable = False

        navigator = FocusNavigator(widgets)
        last = navigator.find_last()

        assert last is widgets[1]

    def test_find_nearest_direction(self):
        """find_nearest should find widget in direction."""
        widgets = [
            Widget(local_x=50.0, local_y=50.0),
            Widget(local_x=150.0, local_y=50.0),  # Right
            Widget(local_x=50.0, local_y=150.0),  # Down
        ]
        for w in widgets:
            w.focusable = True

        navigator = FocusNavigator(widgets)
        nearest_right = navigator.find_nearest(widgets[0], FocusDirection.RIGHT)

        assert nearest_right is widgets[1]


class TestFocusDirection:
    """Tests for FocusDirection enum."""

    def test_up_direction(self):
        """UP direction should exist."""
        assert FocusDirection.UP is not None

    def test_down_direction(self):
        """DOWN direction should exist."""
        assert FocusDirection.DOWN is not None

    def test_left_direction(self):
        """LEFT direction should exist."""
        assert FocusDirection.LEFT is not None

    def test_right_direction(self):
        """RIGHT direction should exist."""
        assert FocusDirection.RIGHT is not None

    def test_next_direction(self):
        """NEXT direction should exist (tab order)."""
        assert FocusDirection.NEXT is not None

    def test_previous_direction(self):
        """PREVIOUS direction should exist (tab order)."""
        assert FocusDirection.PREVIOUS is not None


class TestWidgetFocusProperties:
    """Tests for widget focus-related properties."""

    def test_widget_focusable_default(self, focusable_widget):
        """Widget focusable should be settable."""
        assert focusable_widget.focusable is True

    def test_widget_tab_index_default(self):
        """Widget tab_index should default to 0."""
        w = Widget()
        assert w.tab_index == 0

    def test_widget_tab_index_settable(self):
        """Widget tab_index should be settable."""
        w = Widget()
        w.tab_index = 5
        assert w.tab_index == 5

    def test_widget_is_focused(self, focus_manager, focusable_widget):
        """Widget is_focused should reflect focus state."""
        assert focusable_widget.is_focused is False

        focus_manager.set_focus(focusable_widget)
        assert focusable_widget.is_focused is True

    def test_widget_focus_method(self, focus_manager, focusable_widget):
        """Widget focus() method should request focus."""
        focusable_widget.focus()
        assert focus_manager.focused_widget is focusable_widget

    def test_widget_blur_method(self, focus_manager, focusable_widget):
        """Widget blur() method should release focus."""
        focus_manager.set_focus(focusable_widget)
        focusable_widget.blur()
        assert focus_manager.focused_widget is None


class TestFocusManagerRegistration:
    """Tests for widget registration with focus manager."""

    def test_register_widget(self, focus_manager, focusable_widget):
        """Widgets can be registered with focus manager."""
        focus_manager.register(focusable_widget)
        assert focusable_widget in focus_manager.registered_widgets

    def test_unregister_widget(self, focus_manager, focusable_widget):
        """Widgets can be unregistered from focus manager."""
        focus_manager.register(focusable_widget)
        focus_manager.unregister(focusable_widget)
        assert focusable_widget not in focus_manager.registered_widgets

    def test_unregister_clears_focus(self, focus_manager, focusable_widget):
        """Unregistering focused widget should clear focus."""
        focus_manager.register(focusable_widget)
        focus_manager.set_focus(focusable_widget)
        focus_manager.unregister(focusable_widget)

        assert focus_manager.focused_widget is None

    def test_auto_register_on_mount(self, focus_manager):
        """Widgets should auto-register when mounted if focusable."""
        parent = Widget()
        child = Widget()
        child.focusable = True

        focus_manager.set_root(parent)
        parent.add_child(child)

        assert child in focus_manager.registered_widgets

    def test_auto_unregister_on_unmount(self, focus_manager):
        """Widgets should auto-unregister when unmounted."""
        parent = Widget()
        child = Widget()
        child.focusable = True

        focus_manager.set_root(parent)
        parent.add_child(child)
        parent.remove_child(child)

        assert child not in focus_manager.registered_widgets
