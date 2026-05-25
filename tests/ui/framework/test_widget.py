"""
Comprehensive tests for the Widget base class.

Tests cover:
- Widget creation and initialization
- Parent/child relationships
- add_child, remove_child, find_child
- Transform properties (position, size, rotation)
- Visibility and enabled states
- Lifecycle methods (on_mount, on_unmount, etc.)
- Hit testing
- Dirty tracking
- get_root traversal

Note: This module tests the Widget class defined in engine/ui/framework/widget.py.
"""

import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.ui.framework.widget import Widget, WidgetStyle, LayoutConstraints
from engine.ui.framework.coordinate import Point, Size, Rect
from engine.ui.framework.events import EventType


@pytest.fixture
def widget():
    """Create a basic widget for testing."""
    return Widget()


@pytest.fixture
def parent_widget():
    """Create a parent widget."""
    return Widget()


@pytest.fixture
def child_widget():
    """Create a child widget."""
    return Widget()


class TestWidgetCreation:
    """Tests for Widget creation and initialization."""

    def test_widget_creation_default(self):
        """Widget should be creatable with defaults."""
        w = Widget()
        assert w is not None

    def test_widget_has_unique_id(self):
        """Each widget should have a unique ID."""
        w1 = Widget()
        w2 = Widget()
        assert w1.id != w2.id

    def test_widget_id_is_immutable(self):
        """Widget ID should not be changeable after creation."""
        w = Widget()
        original_id = w.id
        with pytest.raises((AttributeError, TypeError)):
            w.id = 999
        assert w.id == original_id

    def test_widget_default_position(self):
        """Widget should have default position at origin."""
        w = Widget()
        assert w.local_x == 0.0
        assert w.local_y == 0.0

    def test_widget_default_size(self):
        """Widget should have default size of zero."""
        w = Widget()
        assert w.width == 0.0
        assert w.height == 0.0

    def test_widget_with_position(self):
        """Widget can be created with position."""
        w = Widget(x=10.0, y=20.0)
        assert w.local_x == 10.0
        assert w.local_y == 20.0

    def test_widget_with_size(self):
        """Widget can be created with size."""
        w = Widget(width=100.0, height=50.0)
        assert w.width == 100.0
        assert w.height == 50.0

    def test_widget_default_visible(self):
        """Widget should be visible by default."""
        w = Widget()
        assert w.visible is True

    def test_widget_default_enabled(self):
        """Widget should be enabled by default."""
        w = Widget()
        assert w.enabled is True

    def test_widget_default_parent_is_none(self):
        """Widget should have no parent by default."""
        w = Widget()
        assert w.parent is None

    def test_widget_default_no_children(self):
        """Widget should have no children by default."""
        w = Widget()
        assert len(w.children) == 0


class TestWidgetParentChild:
    """Tests for parent/child relationships."""

    def test_add_child(self, parent_widget, child_widget):
        """add_child should add widget as child."""
        parent_widget.add_child(child_widget)
        assert child_widget in parent_widget.children
        assert child_widget.parent is parent_widget

    def test_add_child_sets_parent(self, parent_widget, child_widget):
        """add_child should set parent reference."""
        parent_widget.add_child(child_widget)
        assert child_widget.parent is parent_widget

    def test_add_multiple_children(self, parent_widget):
        """Multiple children can be added."""
        c1 = Widget()
        c2 = Widget()
        c3 = Widget()
        parent_widget.add_child(c1)
        parent_widget.add_child(c2)
        parent_widget.add_child(c3)
        assert len(parent_widget.children) == 3
        assert c1 in parent_widget.children
        assert c2 in parent_widget.children
        assert c3 in parent_widget.children

    def test_add_child_order_preserved(self, parent_widget):
        """Child order should be preserved."""
        c1 = Widget()
        c2 = Widget()
        c3 = Widget()
        parent_widget.add_child(c1)
        parent_widget.add_child(c2)
        parent_widget.add_child(c3)
        assert parent_widget.children[0] is c1
        assert parent_widget.children[1] is c2
        assert parent_widget.children[2] is c3

    def test_move_child_to_index(self, parent_widget):
        """move_child should move widget to specified index."""
        c1 = Widget()
        c2 = Widget()
        c3 = Widget()
        parent_widget.add_child(c1)
        parent_widget.add_child(c2)
        parent_widget.add_child(c3)
        parent_widget.move_child(c3, 0)
        assert parent_widget.children[0] is c3

    def test_remove_child(self, parent_widget, child_widget):
        """remove_child should remove widget from children."""
        parent_widget.add_child(child_widget)
        parent_widget.remove_child(child_widget)
        assert child_widget not in parent_widget.children

    def test_remove_child_clears_parent(self, parent_widget, child_widget):
        """remove_child should clear parent reference."""
        parent_widget.add_child(child_widget)
        parent_widget.remove_child(child_widget)
        assert child_widget.parent is None

    def test_remove_nonexistent_child(self, parent_widget, child_widget):
        """remove_child on non-child should not raise."""
        # Should not raise
        parent_widget.remove_child(child_widget)

    def test_reparenting_requires_remove_first(self, parent_widget, child_widget):
        """Adding child with existing parent should raise ValueError."""
        new_parent = Widget()
        parent_widget.add_child(child_widget)
        with pytest.raises(ValueError, match="already has a parent"):
            new_parent.add_child(child_widget)

    def test_add_self_as_child_raises(self, widget):
        """Adding widget to itself should raise."""
        with pytest.raises(ValueError):
            widget.add_child(widget)

    def test_add_ancestor_as_child_raises(self, parent_widget, child_widget):
        """Adding ancestor as child should raise (prevent cycles)."""
        parent_widget.add_child(child_widget)
        with pytest.raises(ValueError):
            child_widget.add_child(parent_widget)

    def test_child_count(self, parent_widget):
        """child_count should return number of children."""
        assert parent_widget.child_count == 0
        parent_widget.add_child(Widget())
        assert parent_widget.child_count == 1
        parent_widget.add_child(Widget())
        assert parent_widget.child_count == 2

    def test_has_children(self, parent_widget):
        """has_children should return True if children exist."""
        assert parent_widget.has_children is False
        parent_widget.add_child(Widget())
        assert parent_widget.has_children is True


class TestWidgetFindChild:
    """Tests for find_child functionality."""

    def test_find_child_by_id(self, parent_widget):
        """find_child_by_id should find child by ID."""
        child = Widget()
        parent_widget.add_child(child)
        found = parent_widget.find_child_by_id(child.id)
        assert found is child

    def test_find_child_by_name(self, parent_widget):
        """find_child should find child by name."""
        child = Widget(name="test_child")
        parent_widget.add_child(child)
        found = parent_widget.find_child("test_child")
        assert found is child

    def test_find_child_recursive(self, parent_widget):
        """find_child_by_id should search recursively."""
        child = Widget()
        grandchild = Widget()
        parent_widget.add_child(child)
        child.add_child(grandchild)
        found = parent_widget.find_child_by_id(grandchild.id, recursive=True)
        assert found is grandchild

    def test_find_child_not_recursive(self, parent_widget):
        """find_child_by_id without recursive should not find grandchild."""
        child = Widget()
        grandchild = Widget()
        parent_widget.add_child(child)
        child.add_child(grandchild)
        found = parent_widget.find_child_by_id(grandchild.id, recursive=False)
        assert found is None

    def test_find_child_returns_none_if_not_found(self, parent_widget):
        """find_child_by_id should return None if child not found."""
        found = parent_widget.find_child_by_id(99999)
        assert found is None

    def test_find_children_by_type(self, parent_widget):
        """find_children_by_type should return matching children."""
        c1 = Widget()
        c2 = Widget()
        parent_widget.add_child(c1)
        parent_widget.add_child(c2)
        found = parent_widget.find_children_by_type(Widget)
        assert len(found) == 2

    def test_get_child_at_index(self, parent_widget):
        """get_child_at should return child at index."""
        c1 = Widget()
        c2 = Widget()
        parent_widget.add_child(c1)
        parent_widget.add_child(c2)
        assert parent_widget.get_child_at(0) is c1
        assert parent_widget.get_child_at(1) is c2

    def test_get_child_at_invalid_index(self, parent_widget):
        """get_child_at with invalid index should return None."""
        assert parent_widget.get_child_at(0) is None
        assert parent_widget.get_child_at(-1) is None


class TestWidgetTransform:
    """Tests for transform properties."""

    def test_local_x_settable(self, widget):
        """local_x should be settable."""
        widget.local_x = 50.0
        assert widget.local_x == 50.0

    def test_local_y_settable(self, widget):
        """local_y should be settable."""
        widget.local_y = 75.0
        assert widget.local_y == 75.0

    def test_width_settable(self, widget):
        """width should be settable."""
        widget.width = 200.0
        assert widget.width == 200.0

    def test_height_settable(self, widget):
        """height should be settable."""
        widget.height = 150.0
        assert widget.height == 150.0

    def test_negative_width_allowed(self, widget):
        """Widget accepts negative width (validation may be in layout)."""
        # The actual implementation doesn't validate negative values
        # This test documents actual behavior
        widget.width = -10.0
        assert widget.width == -10.0

    def test_negative_height_allowed(self, widget):
        """Widget accepts negative height (validation may be in layout)."""
        # The actual implementation doesn't validate negative values
        widget.height = -10.0
        assert widget.height == -10.0

    def test_position_property(self, widget):
        """position property should return Point."""
        widget.local_x = 10.0
        widget.local_y = 20.0
        assert widget.position == Point(10.0, 20.0)

    def test_position_setter(self, widget):
        """position setter should set x and y."""
        widget.position = Point(30.0, 40.0)
        assert widget.local_x == 30.0
        assert widget.local_y == 40.0

    def test_size_property(self, widget):
        """size property should return Size."""
        widget.width = 100.0
        widget.height = 50.0
        assert widget.size == Size(100.0, 50.0)

    def test_size_setter(self, widget):
        """size setter should set width and height."""
        widget.size = Size(200.0, 150.0)
        assert widget.width == 200.0
        assert widget.height == 150.0

    def test_local_rect_property(self, widget):
        """local_rect property should return Rect."""
        widget.local_x = 10.0
        widget.local_y = 20.0
        widget.width = 100.0
        widget.height = 50.0
        bounds = widget.local_rect
        assert bounds == Rect(10.0, 20.0, 100.0, 50.0)

    def test_global_x_no_parent(self, widget):
        """global_x with no parent should equal local_x."""
        widget.local_x = 50.0
        assert widget.global_x == 50.0

    def test_global_y_no_parent(self, widget):
        """global_y with no parent should equal local_y."""
        widget.local_y = 75.0
        assert widget.global_y == 75.0

    def test_global_x_with_parent(self, parent_widget, child_widget):
        """global_x should include parent offset."""
        parent_widget.local_x = 100.0
        child_widget.local_x = 50.0
        parent_widget.add_child(child_widget)
        assert child_widget.global_x == 150.0

    def test_global_y_with_parent(self, parent_widget, child_widget):
        """global_y should include parent offset."""
        parent_widget.local_y = 100.0
        child_widget.local_y = 50.0
        parent_widget.add_child(child_widget)
        assert child_widget.global_y == 150.0

    def test_global_position_nested(self):
        """global position should accumulate through hierarchy."""
        root = Widget(x=10.0, y=10.0)
        child = Widget(x=20.0, y=20.0)
        grandchild = Widget(x=30.0, y=30.0)
        root.add_child(child)
        child.add_child(grandchild)
        assert grandchild.global_x == 60.0
        assert grandchild.global_y == 60.0

    def test_rotation_default(self, widget):
        """rotation should default to 0."""
        assert widget.rotation == 0.0

    def test_rotation_settable(self, widget):
        """rotation should be settable."""
        import math
        widget.rotation = math.pi / 4
        assert widget.rotation == math.pi / 4

    def test_scale_default(self, widget):
        """scale should default to (1, 1)."""
        assert widget.scale_x == 1.0
        assert widget.scale_y == 1.0

    def test_scale_settable(self, widget):
        """scale should be settable."""
        widget.scale_x = 2.0
        widget.scale_y = 0.5
        assert widget.scale_x == 2.0
        assert widget.scale_y == 0.5


class TestWidgetVisibility:
    """Tests for visibility and enabled states."""

    def test_visible_settable(self, widget):
        """visible should be settable."""
        widget.visible = False
        assert widget.visible is False

    def test_enabled_settable(self, widget):
        """enabled should be settable."""
        widget.enabled = False
        assert widget.enabled is False

    def test_visible_property(self, parent_widget, child_widget):
        """visible property returns the widget's own visible state."""
        parent_widget.add_child(child_widget)
        parent_widget.visible = False
        # Child's own visible property is still True
        assert child_widget.visible is True

    def test_child_visible_with_visible_parent(self, parent_widget, child_widget):
        """Child should be visible when parent is visible."""
        parent_widget.add_child(child_widget)
        assert child_widget.visible is True

    def test_enabled_property(self, parent_widget, child_widget):
        """enabled property returns the widget's own enabled state."""
        parent_widget.add_child(child_widget)
        parent_widget.enabled = False
        # Child's own enabled property is still True
        assert child_widget.enabled is True

    def test_child_enabled_with_enabled_parent(self, parent_widget, child_widget):
        """Child should be enabled when parent is enabled."""
        parent_widget.add_child(child_widget)
        assert child_widget.enabled is True

    def test_style_opacity_default(self, widget):
        """style opacity should default to 1.0."""
        assert widget.style.opacity == 1.0

    def test_style_opacity_settable(self, widget):
        """style opacity should be settable."""
        widget.style.opacity = 0.5
        assert widget.style.opacity == 0.5

    def test_style_opacity_not_auto_clamped(self, widget):
        """style opacity is not auto-clamped (data class stores raw value)."""
        widget.style.opacity = 1.5
        # WidgetStyle doesn't auto-clamp, just stores the value
        assert widget.style.opacity == 1.5


class TestWidgetLifecycle:
    """Tests for lifecycle methods."""

    def test_mount_called_when_added_to_mounted_parent(self, parent_widget, child_widget):
        """_mount should be called when added to a mounted parent."""
        parent_widget._mount()
        assert parent_widget.is_mounted is True

        parent_widget.add_child(child_widget)
        assert child_widget.is_mounted is True

    def test_unmount_called_when_removed_from_mounted_parent(self, parent_widget, child_widget):
        """_unmount should be called when removed from mounted parent."""
        parent_widget._mount()
        parent_widget.add_child(child_widget)
        assert child_widget.is_mounted is True

        parent_widget.remove_child(child_widget)
        assert child_widget.is_mounted is False

    def test_tracked_property_change(self, widget):
        """TrackedDescriptor should trigger _mark_dirty when value changes."""
        widget._clear_dirty()
        assert not widget.is_dirty()

        widget.width = 100.0
        assert widget.is_dirty("width")

    def test_is_mounted_property(self, parent_widget, child_widget):
        """is_mounted should reflect mount state."""
        assert child_widget.is_mounted is False

        parent_widget._mount()  # Mount parent first
        parent_widget.add_child(child_widget)
        assert child_widget.is_mounted is True

        parent_widget.remove_child(child_widget)
        assert child_widget.is_mounted is False


class TestWidgetHitTesting:
    """Tests for hit testing."""

    def test_contains_point_global(self, widget):
        """contains_point should return True for point inside (global coords)."""
        widget.local_x = 0.0
        widget.local_y = 0.0
        widget.width = 100.0
        widget.height = 100.0
        assert widget.contains_point(Point(50.0, 50.0), local=False) is True

    def test_contains_point_outside(self, widget):
        """contains_point should return False for point outside."""
        widget.local_x = 0.0
        widget.local_y = 0.0
        widget.width = 100.0
        widget.height = 100.0
        assert widget.contains_point(Point(150.0, 50.0), local=False) is False

    def test_contains_point_on_boundary(self, widget):
        """contains_point should return True for point on boundary."""
        widget.local_x = 0.0
        widget.local_y = 0.0
        widget.width = 100.0
        widget.height = 100.0
        assert widget.contains_point(Point(0.0, 0.0), local=False) is True
        assert widget.contains_point(Point(100.0, 100.0), local=False) is True

    def test_contains_point_local(self, widget):
        """contains_point with local=True uses local coordinates."""
        widget.local_x = 50.0  # Position doesn't matter for local coords
        widget.local_y = 50.0
        widget.width = 100.0
        widget.height = 100.0
        # In local coords, bounds are (0,0) to (width, height)
        assert widget.contains_point(Point(50.0, 50.0), local=True) is True
        assert widget.contains_point(Point(150.0, 50.0), local=True) is False

    def test_hit_test_returns_self(self, widget):
        """hit_test should return self if point inside."""
        widget.local_x = 0.0
        widget.local_y = 0.0
        widget.width = 100.0
        widget.height = 100.0
        result = widget.hit_test(Point(50.0, 50.0))
        assert result is widget

    def test_hit_test_returns_none_outside(self, widget):
        """hit_test should return None if point outside."""
        widget.local_x = 0.0
        widget.local_y = 0.0
        widget.width = 100.0
        widget.height = 100.0
        result = widget.hit_test(Point(150.0, 50.0))
        assert result is None

    def test_hit_test_returns_topmost_child(self, parent_widget):
        """hit_test should return topmost child that contains point."""
        parent_widget.local_x = 0.0
        parent_widget.local_y = 0.0
        parent_widget.width = 200.0
        parent_widget.height = 200.0

        child1 = Widget(x=0.0, y=0.0, width=100.0, height=100.0)
        child2 = Widget(x=50.0, y=50.0, width=100.0, height=100.0)
        parent_widget.add_child(child1)
        parent_widget.add_child(child2)

        # Point is in overlap area - should return child2 (last/topmost)
        result = parent_widget.hit_test(Point(75.0, 75.0))
        assert result is child2

    def test_hit_test_invisible_widget(self, widget):
        """hit_test on invisible widget should return None."""
        widget.local_x = 0.0
        widget.local_y = 0.0
        widget.width = 100.0
        widget.height = 100.0
        widget.visible = False
        result = widget.hit_test(Point(50.0, 50.0))
        assert result is None

    def test_hit_test_disabled_widget(self, widget):
        """hit_test on disabled widget returns None (implementation detail)."""
        widget.local_x = 0.0
        widget.local_y = 0.0
        widget.width = 100.0
        widget.height = 100.0
        widget.enabled = False
        result = widget.hit_test(Point(50.0, 50.0))
        # Actual implementation returns None for disabled widgets
        assert result is None

    def test_hit_test_uses_global_coordinates(self, parent_widget, child_widget):
        """hit_test should work with global coordinates."""
        parent_widget.local_x = 100.0
        parent_widget.local_y = 100.0
        parent_widget.width = 200.0
        parent_widget.height = 200.0

        child_widget.local_x = 50.0
        child_widget.local_y = 50.0
        child_widget.width = 50.0
        child_widget.height = 50.0

        parent_widget.add_child(child_widget)

        # Child is at global (150, 150) to (200, 200)
        result = parent_widget.hit_test(Point(175.0, 175.0))
        assert result is child_widget


class TestWidgetDirtyTracking:
    """Tests for dirty tracking."""

    def test_widget_starts_clean_when_defaults_match(self, widget):
        """Widget with default values starts clean (no value changes)."""
        # TrackedDescriptor only marks dirty when value changes from previous
        # Default values set in __init__ that match descriptor defaults don't dirty
        fresh_widget = Widget()
        # Actually clean because defaults match
        assert fresh_widget.is_dirty() is False

    def test_clear_dirty(self, widget):
        """_clear_dirty should clear dirty flags."""
        widget._clear_dirty()
        assert widget.is_dirty() is False

    def test_position_change_marks_dirty(self, widget):
        """Position change should mark dirty."""
        widget._clear_dirty()
        widget.local_x = 100.0
        assert widget.is_dirty() is True
        assert widget.is_dirty("local_x") is True

    def test_size_change_marks_dirty(self, widget):
        """Size change should mark dirty."""
        widget._clear_dirty()
        widget.width = 100.0
        assert widget.is_dirty() is True
        assert widget.is_dirty("width") is True

    def test_visibility_change_marks_dirty(self, widget):
        """Visibility change should mark dirty."""
        widget._clear_dirty()
        widget.visible = False
        assert widget.is_dirty() is True
        assert widget.is_dirty("visible") is True

    def test_dirty_flags_tracking(self, widget):
        """Dirty flags should track which properties changed."""
        widget._clear_dirty()
        widget.local_x = 50.0
        widget.width = 100.0
        assert widget.is_dirty("local_x") is True
        assert widget.is_dirty("width") is True

    def test_clear_dirty_removes_flags(self, widget):
        """_clear_dirty should clear all flags."""
        widget.local_x = 50.0
        widget._clear_dirty()
        assert widget.is_dirty() is False

    def test_clear_specific_dirty_flag(self, widget):
        """_clear_dirty(flag) clears specific flag."""
        widget._clear_dirty()
        widget.local_x = 50.0
        widget.width = 100.0
        widget._clear_dirty("local_x")
        assert widget.is_dirty("local_x") is False
        assert widget.is_dirty("width") is True


class TestWidgetGetRoot:
    """Tests for get_root traversal."""

    def test_get_root_no_parent(self, widget):
        """get_root with no parent should return self."""
        assert widget.get_root() is widget

    def test_get_root_with_parent(self, parent_widget, child_widget):
        """get_root should return root of hierarchy."""
        parent_widget.add_child(child_widget)
        assert child_widget.get_root() is parent_widget

    def test_get_root_deep_hierarchy(self):
        """get_root should work for deep hierarchies."""
        root = Widget()
        child = Widget()
        grandchild = Widget()
        great_grandchild = Widget()

        root.add_child(child)
        child.add_child(grandchild)
        grandchild.add_child(great_grandchild)

        assert great_grandchild.get_root() is root
        assert grandchild.get_root() is root
        assert child.get_root() is root


class TestWidgetIteration:
    """Tests for widget iteration."""

    def test_children_property(self, parent_widget):
        """children property returns list of children."""
        c1 = Widget()
        c2 = Widget()
        parent_widget.add_child(c1)
        parent_widget.add_child(c2)

        children = parent_widget.children
        assert c1 in children
        assert c2 in children

    def test_get_descendants(self, parent_widget):
        """get_descendants should yield all descendants."""
        child = Widget()
        grandchild = Widget()
        parent_widget.add_child(child)
        child.add_child(grandchild)

        descendants = list(parent_widget.get_descendants())
        assert child in descendants
        assert grandchild in descendants

    def test_get_ancestors(self, parent_widget, child_widget):
        """get_ancestors should return all ancestors."""
        grandparent = Widget()
        grandparent.add_child(parent_widget)
        parent_widget.add_child(child_widget)

        ancestors = child_widget.get_ancestors()
        assert parent_widget in ancestors
        assert grandparent in ancestors


class TestWidgetNaming:
    """Tests for widget naming."""

    def test_name_default(self, widget):
        """name should default to empty or auto-generated."""
        # Either empty or auto-generated name is acceptable
        assert widget.name is not None or widget.name == ""

    def test_name_settable(self, widget):
        """name should be settable."""
        widget.name = "my_widget"
        assert widget.name == "my_widget"

    def test_name_at_creation(self):
        """name can be set at creation."""
        w = Widget(name="created_widget")
        assert w.name == "created_widget"


class TestWidgetZOrder:
    """Tests for z-order management."""

    def test_z_index_default(self, widget):
        """z_index should default to 0."""
        assert widget.z_index == 0

    def test_z_index_settable(self, widget):
        """z_index should be settable."""
        widget.z_index = 10
        assert widget.z_index == 10

    def test_bring_to_front(self, parent_widget):
        """bring_to_front should move widget to end of children."""
        c1 = Widget()
        c2 = Widget()
        parent_widget.add_child(c1)
        parent_widget.add_child(c2)

        c1.bring_to_front()
        assert parent_widget.children[-1] is c1

    def test_send_to_back(self, parent_widget):
        """send_to_back should move widget to start of children."""
        c1 = Widget()
        c2 = Widget()
        parent_widget.add_child(c1)
        parent_widget.add_child(c2)

        c2.send_to_back()
        assert parent_widget.children[0] is c2


class TestWidgetStyle:
    """Tests for widget styling."""

    def test_default_style(self, widget):
        """Widget should have default WidgetStyle."""
        assert widget.style is not None
        assert isinstance(widget.style, WidgetStyle)

    def test_style_settable(self, widget):
        """Widget style can be replaced."""
        new_style = WidgetStyle(background_color="#FF0000")
        widget.style = new_style
        assert widget.style.background_color == "#FF0000"

    def test_style_merge(self):
        """WidgetStyle should support merging."""
        base = WidgetStyle(background_color="#000000", border_width=2.0)
        override = WidgetStyle(background_color="#FFFFFF")
        merged = base.merge_with(override)
        assert merged.background_color == "#FFFFFF"
        assert merged.border_width == 2.0


class TestWidgetEventHandlers:
    """Tests for widget event handler storage."""

    def test_add_event_handler(self, widget):
        """add_event_handler should store handler."""
        handler = lambda e: None
        widget.add_event_handler(EventType.CLICK, handler)
        # No direct has_event_listener, but we can check internal state
        assert EventType.CLICK in widget._event_handlers

    def test_remove_event_handler(self, widget):
        """remove_event_handler should remove handler."""
        handler = lambda e: None
        widget.add_event_handler(EventType.CLICK, handler)
        result = widget.remove_event_handler(EventType.CLICK, handler)
        assert result is True
        assert EventType.CLICK not in widget._event_handlers or len(widget._event_handlers[EventType.CLICK]) == 0

    def test_multiple_event_handlers(self, widget):
        """Multiple handlers for same event should all be stored."""
        h1 = lambda e: None
        h2 = lambda e: None
        widget.add_event_handler(EventType.CLICK, h1)
        widget.add_event_handler(EventType.CLICK, h2)
        assert len(widget._event_handlers[EventType.CLICK]) == 2

    def test_capture_event_handler(self, widget):
        """Capture handlers should be stored separately."""
        handler = lambda e: None
        widget.add_event_handler(EventType.CLICK, handler, capture=True)
        assert EventType.CLICK in widget._capture_handlers
