"""
Comprehensive tests for Keyboard Navigation accessibility support.

Tests cover:
- FocusDirection enum
- NavigationMode enum
- TabStop class
- NavigationGroup class
- SkipLink class
- KeyboardShortcut class
- TabOrder class
- KeyboardNavigator class
- Focus management
- Tab order navigation
- Directional navigation
- Grid navigation
- Skip links
- Keyboard shortcuts
- Focus callbacks
- Spatial navigation
"""

import pytest
import sys
from pathlib import Path

# Add engine to path
engine_path = Path(__file__).parent.parent.parent.parent.parent / "engine"
sys.path.insert(0, str(engine_path.parent))

from engine.ui.accessibility.keyboard_nav import (
    FocusDirection,
    NavigationMode,
    TabStop,
    NavigationGroup,
    SkipLink,
    KeyboardShortcut,
    TabOrder,
    KeyboardNavigator,
)


class TestFocusDirection:
    """Test FocusDirection enum."""

    def test_tab_directions(self):
        """Test tab navigation directions."""
        assert FocusDirection.NEXT is not None
        assert FocusDirection.PREVIOUS is not None

    def test_arrow_directions(self):
        """Test arrow key directions."""
        assert FocusDirection.UP is not None
        assert FocusDirection.DOWN is not None
        assert FocusDirection.LEFT is not None
        assert FocusDirection.RIGHT is not None

    def test_page_directions(self):
        """Test page directions."""
        assert FocusDirection.FIRST is not None
        assert FocusDirection.LAST is not None
        assert FocusDirection.PAGE_UP is not None
        assert FocusDirection.PAGE_DOWN is not None


class TestNavigationMode:
    """Test NavigationMode enum."""

    def test_sequential_mode(self):
        """Test sequential navigation mode."""
        assert NavigationMode.SEQUENTIAL is not None

    def test_directional_mode(self):
        """Test directional navigation mode."""
        assert NavigationMode.DIRECTIONAL is not None

    def test_grid_mode(self):
        """Test grid navigation mode."""
        assert NavigationMode.GRID is not None

    def test_roving_mode(self):
        """Test roving tabindex mode."""
        assert NavigationMode.ROVING is not None


class TestTabStop:
    """Test TabStop class."""

    def test_creation(self):
        """Test creating a tab stop."""
        stop = TabStop(widget_id="btn1")
        assert stop.widget_id == "btn1"
        assert stop.focusable is True

    def test_default_tab_index(self):
        """Test default tab index is 0."""
        stop = TabStop(widget_id="btn1")
        assert stop.tab_index == 0

    def test_custom_tab_index(self):
        """Test custom tab index."""
        stop = TabStop(widget_id="btn1", tab_index=5)
        assert stop.tab_index == 5

    def test_negative_tab_index_skips(self):
        """Test negative tab index is not tabbable."""
        stop = TabStop(widget_id="btn1", tab_index=-1)
        assert stop.is_tabbable() is False

    def test_is_tabbable_focusable(self):
        """Test tabbable requires focusable."""
        stop = TabStop(widget_id="btn1", focusable=False)
        assert stop.is_tabbable() is False

    def test_position_properties(self):
        """Test position properties."""
        stop = TabStop(
            widget_id="btn1",
            x=100.0, y=50.0,
            width=80.0, height=30.0,
        )
        assert stop.x == 100.0
        assert stop.y == 50.0
        assert stop.width == 80.0
        assert stop.height == 30.0

    def test_center_coordinates(self):
        """Test center coordinate calculations."""
        stop = TabStop(
            widget_id="btn1",
            x=100.0, y=50.0,
            width=80.0, height=30.0,
        )
        assert stop.center_x() == 140.0  # 100 + 80/2
        assert stop.center_y() == 65.0   # 50 + 30/2

    def test_contains_point(self):
        """Test point containment check."""
        stop = TabStop(
            widget_id="btn1",
            x=100.0, y=50.0,
            width=80.0, height=30.0,
        )
        assert stop.contains_point(120.0, 60.0) is True
        assert stop.contains_point(50.0, 60.0) is False

    def test_group_assignment(self):
        """Test group assignment."""
        stop = TabStop(widget_id="btn1", group="toolbar")
        assert stop.group == "toolbar"


class TestNavigationGroup:
    """Test NavigationGroup class."""

    def test_creation(self):
        """Test creating a navigation group."""
        group = NavigationGroup(group_id="menu")
        assert group.group_id == "menu"
        assert group.mode == NavigationMode.SEQUENTIAL

    def test_custom_mode(self):
        """Test custom navigation mode."""
        group = NavigationGroup(
            group_id="grid1",
            mode=NavigationMode.GRID,
        )
        assert group.mode == NavigationMode.GRID

    def test_add_member(self):
        """Test adding member to group."""
        group = NavigationGroup(group_id="menu")
        group.add_member("item1")
        assert "item1" in group.members
        assert group.get_member_count() == 1

    def test_add_member_no_duplicates(self):
        """Test adding duplicate member."""
        group = NavigationGroup(group_id="menu")
        group.add_member("item1")
        group.add_member("item1")
        assert group.get_member_count() == 1

    def test_remove_member(self):
        """Test removing member from group."""
        group = NavigationGroup(group_id="menu")
        group.add_member("item1")
        group.remove_member("item1")
        assert "item1" not in group.members

    def test_get_active_member(self):
        """Test getting active member."""
        group = NavigationGroup(group_id="menu")
        group.add_member("item1")
        group.add_member("item2")
        group.active_index = 1
        assert group.get_active_member() == "item2"

    def test_get_active_member_empty(self):
        """Test getting active member from empty group."""
        group = NavigationGroup(group_id="menu")
        assert group.get_active_member() is None

    def test_set_active_by_id(self):
        """Test setting active member by ID."""
        group = NavigationGroup(group_id="menu")
        group.add_member("item1")
        group.add_member("item2")
        result = group.set_active_by_id("item2")
        assert result is True
        assert group.active_index == 1

    def test_set_active_by_id_not_found(self):
        """Test setting active with invalid ID."""
        group = NavigationGroup(group_id="menu")
        result = group.set_active_by_id("nonexistent")
        assert result is False

    def test_wrap_setting(self):
        """Test wrap around setting."""
        group = NavigationGroup(group_id="menu", wrap=True)
        assert group.wrap is True

    def test_trap_focus(self):
        """Test focus trap setting."""
        group = NavigationGroup(group_id="dialog", trap_focus=True)
        assert group.trap_focus is True

    def test_grid_columns(self):
        """Test grid columns setting."""
        group = NavigationGroup(
            group_id="grid1",
            mode=NavigationMode.GRID,
            columns=4,
        )
        assert group.columns == 4


class TestSkipLink:
    """Test SkipLink class."""

    def test_creation(self):
        """Test creating a skip link."""
        link = SkipLink(
            link_id="skip-main",
            label="Skip to main content",
            target_widget_id="main-content",
        )
        assert link.link_id == "skip-main"
        assert link.label == "Skip to main content"
        assert link.target_widget_id == "main-content"

    def test_with_shortcut(self):
        """Test skip link with shortcut."""
        link = SkipLink(
            link_id="skip-nav",
            label="Skip to navigation",
            target_widget_id="nav",
            shortcut="Alt+N",
        )
        assert link.shortcut == "Alt+N"

    def test_visible_on_focus(self):
        """Test visible on focus setting."""
        link = SkipLink(
            link_id="skip-main",
            label="Skip",
            target_widget_id="main",
            visible_on_focus=True,
        )
        assert link.visible_on_focus is True


class TestKeyboardShortcut:
    """Test KeyboardShortcut class."""

    def test_creation(self):
        """Test creating a keyboard shortcut."""
        shortcut = KeyboardShortcut(
            key="S",
            action=lambda: None,
            description="Save",
        )
        assert shortcut.key == "S"
        assert shortcut.description == "Save"

    def test_with_modifiers(self):
        """Test shortcut with modifiers."""
        shortcut = KeyboardShortcut(
            key="S",
            action=lambda: None,
            ctrl=True,
            shift=True,
        )
        assert shortcut.ctrl is True
        assert shortcut.shift is True
        assert shortcut.alt is False

    def test_matches_exact(self):
        """Test exact match."""
        shortcut = KeyboardShortcut(
            key="S",
            action=lambda: None,
            ctrl=True,
        )
        assert shortcut.matches("S", ctrl=True) is True
        assert shortcut.matches("S", ctrl=False) is False

    def test_matches_case_insensitive(self):
        """Test case-insensitive matching."""
        shortcut = KeyboardShortcut(key="s", action=lambda: None)
        assert shortcut.matches("S") is True
        assert shortcut.matches("s") is True

    def test_matches_wrong_modifiers(self):
        """Test non-matching modifiers."""
        shortcut = KeyboardShortcut(
            key="S",
            action=lambda: None,
            ctrl=True,
        )
        assert shortcut.matches("S", alt=True) is False

    def test_disabled_shortcut(self):
        """Test disabled shortcut doesn't match."""
        shortcut = KeyboardShortcut(
            key="S",
            action=lambda: None,
            enabled=False,
        )
        assert shortcut.matches("S") is False

    def test_get_shortcut_string(self):
        """Test getting shortcut string."""
        shortcut = KeyboardShortcut(
            key="S",
            action=lambda: None,
            ctrl=True,
            shift=True,
        )
        string = shortcut.get_shortcut_string()
        assert "Ctrl" in string
        assert "Shift" in string
        assert "S" in string

    def test_global_shortcut(self):
        """Test global shortcut flag."""
        shortcut = KeyboardShortcut(
            key="F1",
            action=lambda: None,
            global_shortcut=True,
        )
        assert shortcut.global_shortcut is True

    def test_scoped_to_group(self):
        """Test shortcut scoped to group."""
        shortcut = KeyboardShortcut(
            key="Enter",
            action=lambda: None,
            group_id="menu",
        )
        assert shortcut.group_id == "menu"


class TestTabOrder:
    """Test TabOrder class."""

    def test_creation(self):
        """Test creating tab order."""
        order = TabOrder()
        assert order.get_sorted() == []

    def test_add_stop(self):
        """Test adding a tab stop."""
        order = TabOrder()
        stop = TabStop(widget_id="btn1")
        order.add(stop)
        assert order.get("btn1") is stop

    def test_remove_stop(self):
        """Test removing a tab stop."""
        order = TabOrder()
        order.add(TabStop(widget_id="btn1"))
        order.remove("btn1")
        assert order.get("btn1") is None

    def test_update_stop(self):
        """Test updating a tab stop."""
        order = TabOrder()
        order.add(TabStop(widget_id="btn1", x=0.0))
        order.update("btn1", x=100.0)
        assert order.get("btn1").x == 100.0

    def test_get_sorted(self):
        """Test getting sorted tab order."""
        order = TabOrder()
        order.add(TabStop(widget_id="btn3", tab_index=3))
        order.add(TabStop(widget_id="btn1", tab_index=1))
        order.add(TabStop(widget_id="btn2", tab_index=2))
        sorted_stops = order.get_sorted()
        assert sorted_stops[0].widget_id == "btn1"
        assert sorted_stops[1].widget_id == "btn2"
        assert sorted_stops[2].widget_id == "btn3"

    def test_get_next(self):
        """Test getting next tab stop."""
        order = TabOrder()
        order.add(TabStop(widget_id="btn1", tab_index=1))
        order.add(TabStop(widget_id="btn2", tab_index=2))
        next_stop = order.get_next("btn1")
        assert next_stop.widget_id == "btn2"

    def test_get_next_wrap(self):
        """Test getting next wraps to first."""
        order = TabOrder()
        order.add(TabStop(widget_id="btn1", tab_index=1))
        order.add(TabStop(widget_id="btn2", tab_index=2))
        next_stop = order.get_next("btn2", wrap=True)
        assert next_stop.widget_id == "btn1"

    def test_get_next_no_wrap(self):
        """Test getting next without wrap returns None."""
        order = TabOrder()
        order.add(TabStop(widget_id="btn1", tab_index=1))
        order.add(TabStop(widget_id="btn2", tab_index=2))
        next_stop = order.get_next("btn2", wrap=False)
        assert next_stop is None

    def test_get_previous(self):
        """Test getting previous tab stop."""
        order = TabOrder()
        order.add(TabStop(widget_id="btn1", tab_index=1))
        order.add(TabStop(widget_id="btn2", tab_index=2))
        prev_stop = order.get_previous("btn2")
        assert prev_stop.widget_id == "btn1"

    def test_get_first(self):
        """Test getting first tab stop."""
        order = TabOrder()
        order.add(TabStop(widget_id="btn1", tab_index=1))
        order.add(TabStop(widget_id="btn2", tab_index=2))
        first = order.get_first()
        assert first.widget_id == "btn1"

    def test_get_last(self):
        """Test getting last tab stop."""
        order = TabOrder()
        order.add(TabStop(widget_id="btn1", tab_index=1))
        order.add(TabStop(widget_id="btn2", tab_index=2))
        last = order.get_last()
        assert last.widget_id == "btn2"

    def test_clear(self):
        """Test clearing tab order."""
        order = TabOrder()
        order.add(TabStop(widget_id="btn1"))
        order.clear()
        assert order.get_sorted() == []


class TestKeyboardNavigator:
    """Test KeyboardNavigator class."""

    def test_creation(self):
        """Test creating keyboard navigator."""
        nav = KeyboardNavigator()
        assert nav.enabled is True
        assert nav.current_focus is None

    def test_disable_enable(self):
        """Test disabling and enabling."""
        nav = KeyboardNavigator()
        nav.enabled = False
        assert nav.enabled is False
        nav.enabled = True
        assert nav.enabled is True


class TestKeyboardNavigatorFocusables:
    """Test KeyboardNavigator focusable management."""

    def test_register_focusable(self):
        """Test registering a focusable widget."""
        nav = KeyboardNavigator()
        stop = nav.register_focusable("btn1")
        assert stop is not None
        assert nav.tab_order.get("btn1") is stop

    def test_register_with_position(self):
        """Test registering with position."""
        nav = KeyboardNavigator()
        stop = nav.register_focusable(
            "btn1",
            x=100.0, y=50.0,
            width=80.0, height=30.0,
        )
        assert stop.x == 100.0

    def test_register_with_group(self):
        """Test registering with group."""
        nav = KeyboardNavigator()
        nav.create_group("menu")
        nav.register_focusable("item1", group="menu")
        group = nav.get_group("menu")
        assert "item1" in group.members

    def test_unregister_focusable(self):
        """Test unregistering a focusable widget."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1")
        nav.unregister_focusable("btn1")
        assert nav.tab_order.get("btn1") is None

    def test_update_focusable(self):
        """Test updating a focusable widget."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1", x=0.0)
        nav.update_focusable("btn1", x=100.0)
        stop = nav.tab_order.get("btn1")
        assert stop.x == 100.0


class TestKeyboardNavigatorGroups:
    """Test KeyboardNavigator group management."""

    def test_create_group(self):
        """Test creating a navigation group."""
        nav = KeyboardNavigator()
        group = nav.create_group("menu")
        assert group is not None
        assert nav.get_group("menu") is group

    def test_create_group_with_options(self):
        """Test creating group with options."""
        nav = KeyboardNavigator()
        group = nav.create_group(
            "grid1",
            mode=NavigationMode.GRID,
            columns=4,
            wrap=True,
        )
        assert group.mode == NavigationMode.GRID
        assert group.columns == 4
        assert group.wrap is True

    def test_remove_group(self):
        """Test removing a group."""
        nav = KeyboardNavigator()
        nav.create_group("menu")
        nav.remove_group("menu")
        assert nav.get_group("menu") is None


class TestKeyboardNavigatorSkipLinks:
    """Test KeyboardNavigator skip link management."""

    def test_add_skip_link(self):
        """Test adding a skip link."""
        nav = KeyboardNavigator()
        link = nav.add_skip_link(
            "skip-main",
            "Skip to main content",
            "main-content",
        )
        assert link is not None
        links = nav.get_skip_links()
        assert len(links) == 1

    def test_remove_skip_link(self):
        """Test removing a skip link."""
        nav = KeyboardNavigator()
        nav.add_skip_link("skip-main", "Skip", "main")
        nav.remove_skip_link("skip-main")
        links = nav.get_skip_links()
        assert len(links) == 0

    def test_activate_skip_link(self):
        """Test activating a skip link."""
        nav = KeyboardNavigator()
        nav.register_focusable("main")
        nav.add_skip_link("skip-main", "Skip", "main")
        result = nav.activate_skip_link("skip-main")
        assert result is True
        assert nav.current_focus == "main"


class TestKeyboardNavigatorShortcuts:
    """Test KeyboardNavigator shortcut management."""

    def test_register_shortcut(self):
        """Test registering a shortcut."""
        nav = KeyboardNavigator()
        executed = []

        def action():
            executed.append(True)

        shortcut = nav.register_shortcut("S", action, ctrl=True)
        assert shortcut is not None

    def test_unregister_shortcut(self):
        """Test unregistering a shortcut."""
        nav = KeyboardNavigator()
        shortcut = nav.register_shortcut("S", lambda: None)
        nav.unregister_shortcut(shortcut)
        shortcuts = nav.get_shortcuts()
        assert shortcut not in shortcuts

    def test_handle_key_triggers_shortcut(self):
        """Test handle_key triggers matching shortcut."""
        nav = KeyboardNavigator()
        executed = []

        def action():
            executed.append(True)

        nav.register_shortcut("S", action, ctrl=True)
        result = nav.handle_key("S", ctrl=True)
        assert result is True
        assert len(executed) == 1

    def test_handle_key_no_match(self):
        """Test handle_key with no matching shortcut."""
        nav = KeyboardNavigator()
        result = nav.handle_key("X")
        assert result is False


class TestKeyboardNavigatorFocus:
    """Test KeyboardNavigator focus management."""

    def test_set_focus(self):
        """Test setting focus."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1")
        result = nav.set_focus("btn1")
        assert result is True
        assert nav.current_focus == "btn1"

    def test_set_focus_invalid(self):
        """Test setting focus to invalid widget."""
        nav = KeyboardNavigator()
        result = nav.set_focus("nonexistent")
        assert result is False

    def test_set_focus_unfocusable(self):
        """Test setting focus to unfocusable widget."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1")
        nav.tab_order.get("btn1").focusable = False
        result = nav.set_focus("btn1")
        assert result is False

    def test_clear_focus(self):
        """Test clearing focus."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1")
        nav.set_focus("btn1")
        nav.clear_focus()
        assert nav.current_focus is None

    def test_focus_callback(self):
        """Test focus change callback."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1")
        nav.register_focusable("btn2")
        changes = []

        def callback(old, new):
            changes.append((old, new))

        nav.add_focus_callback(callback)
        nav.set_focus("btn1")
        nav.set_focus("btn2")
        assert len(changes) == 2
        assert changes[1] == ("btn1", "btn2")


class TestKeyboardNavigatorMoveFocus:
    """Test KeyboardNavigator focus movement."""

    def test_move_focus_next(self):
        """Test moving focus to next."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1", tab_index=1)
        nav.register_focusable("btn2", tab_index=2)
        nav.set_focus("btn1")
        result = nav.move_focus(FocusDirection.NEXT)
        assert result is True
        assert nav.current_focus == "btn2"

    def test_move_focus_previous(self):
        """Test moving focus to previous."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1", tab_index=1)
        nav.register_focusable("btn2", tab_index=2)
        nav.set_focus("btn2")
        result = nav.move_focus(FocusDirection.PREVIOUS)
        assert result is True
        assert nav.current_focus == "btn1"

    def test_move_focus_first(self):
        """Test moving focus to first."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1", tab_index=1)
        nav.register_focusable("btn2", tab_index=2)
        nav.set_focus("btn2")
        result = nav.move_focus(FocusDirection.FIRST)
        assert result is True
        assert nav.current_focus == "btn1"

    def test_move_focus_last(self):
        """Test moving focus to last."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1", tab_index=1)
        nav.register_focusable("btn2", tab_index=2)
        nav.set_focus("btn1")
        result = nav.move_focus(FocusDirection.LAST)
        assert result is True
        assert nav.current_focus == "btn2"

    def test_move_focus_disabled(self):
        """Test move focus when disabled."""
        nav = KeyboardNavigator()
        nav.enabled = False
        result = nav.move_focus(FocusDirection.NEXT)
        assert result is False


class TestKeyboardNavigatorGridNavigation:
    """Test KeyboardNavigator grid navigation."""

    def test_grid_navigation_down(self):
        """Test moving down in grid."""
        nav = KeyboardNavigator()
        group = nav.create_group(
            "grid",
            mode=NavigationMode.GRID,
            columns=3,
        )
        # Create 3x2 grid
        for i in range(6):
            nav.register_focusable(f"cell{i}", group="grid")
        nav.set_focus("cell1")  # Second cell in first row
        result = nav.move_focus(FocusDirection.DOWN)
        assert result is True
        assert nav.current_focus == "cell4"  # Below cell1

    def test_grid_navigation_right(self):
        """Test moving right in grid."""
        nav = KeyboardNavigator()
        nav.create_group("grid", mode=NavigationMode.GRID, columns=3)
        for i in range(6):
            nav.register_focusable(f"cell{i}", group="grid")
        nav.set_focus("cell0")
        result = nav.move_focus(FocusDirection.RIGHT)
        assert result is True
        assert nav.current_focus == "cell1"


class TestKeyboardNavigatorSpatialNavigation:
    """Test KeyboardNavigator spatial navigation."""

    def test_find_nearest(self):
        """Test finding nearest widget in direction."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1", x=100.0, y=100.0, width=50.0, height=30.0)
        nav.register_focusable("btn2", x=200.0, y=100.0, width=50.0, height=30.0)
        nav.register_focusable("btn3", x=100.0, y=200.0, width=50.0, height=30.0)
        nearest = nav.find_nearest("btn1", FocusDirection.RIGHT)
        assert nearest == "btn2"

    def test_find_nearest_down(self):
        """Test finding nearest below."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1", x=100.0, y=100.0, width=50.0, height=30.0)
        nav.register_focusable("btn2", x=100.0, y=200.0, width=50.0, height=30.0)
        nearest = nav.find_nearest("btn1", FocusDirection.DOWN)
        assert nearest == "btn2"


class TestKeyboardNavigatorClear:
    """Test KeyboardNavigator clear method."""

    def test_clear(self):
        """Test clearing all navigation data."""
        nav = KeyboardNavigator()
        nav.register_focusable("btn1")
        nav.create_group("menu")
        nav.add_skip_link("skip", "Skip", "main")
        nav.register_shortcut("S", lambda: None)
        nav.clear()
        assert nav.tab_order.get("btn1") is None
        assert nav.get_group("menu") is None
        assert len(nav.get_skip_links()) == 0
        assert len(nav.get_shortcuts()) == 0
