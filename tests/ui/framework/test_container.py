"""
Comprehensive tests for the Container widget class.

Tests cover:
- Container creation
- Child management
- Layout delegation
- Clipping behavior
- Nested containers
- Scrolling and overflow

Note: This module tests the Container class defined in engine/ui/framework/container.py.
These tests serve as specifications for the expected Container implementation.
"""

import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

# Import will fail until container.py is implemented
try:
    from engine.ui.framework.container import Container, Layout, LayoutDirection
    from engine.ui.framework.widget import Widget
    CONTAINER_AVAILABLE = True
except ImportError:
    CONTAINER_AVAILABLE = False
    Container = None
    Layout = None
    Widget = None

from engine.ui.framework.coordinate import Point, Size, Rect, Margins


pytestmark = pytest.mark.skipif(
    not CONTAINER_AVAILABLE,
    reason="Container class not yet implemented"
)


@pytest.fixture
def container():
    """Create a basic container for testing."""
    return Container()


@pytest.fixture
def layout_container():
    """Create a container with layout."""
    return Container(layout=Layout(direction=LayoutDirection.VERTICAL))


@pytest.fixture
def child_widgets():
    """Create a list of child widgets."""
    return [Widget() for _ in range(5)]


class TestContainerCreation:
    """Tests for Container creation."""

    def test_container_creation_default(self):
        """Container should be creatable with defaults."""
        c = Container()
        assert c is not None

    def test_container_is_widget(self):
        """Container should be a Widget subclass."""
        c = Container()
        assert isinstance(c, Widget)

    def test_container_has_layout(self):
        """Container should have a layout property."""
        c = Container()
        assert hasattr(c, "layout")

    def test_container_default_no_layout(self):
        """Container should have no layout by default (freeform)."""
        c = Container()
        assert c.layout is None

    def test_container_with_layout(self):
        """Container can be created with layout."""
        layout = Layout(direction=LayoutDirection.VERTICAL)
        c = Container(layout=layout)
        assert c.layout is layout

    def test_container_default_size(self):
        """Container should have default size."""
        c = Container()
        assert c.width >= 0
        assert c.height >= 0

    def test_container_default_padding(self):
        """Container should have default padding of zero."""
        c = Container()
        assert c.padding == Margins.zero()

    def test_container_with_padding(self):
        """Container can be created with padding."""
        c = Container(padding=Margins.all(10.0))
        assert c.padding == Margins.all(10.0)


class TestContainerChildManagement:
    """Tests for container child management."""

    def test_add_child_to_container(self, container):
        """Container should support add_child."""
        child = Widget()
        container.add_child(child)
        assert child in container.children

    def test_add_multiple_children(self, container, child_widgets):
        """Container should support multiple children."""
        for child in child_widgets:
            container.add_child(child)
        assert len(container.children) == len(child_widgets)

    def test_remove_child_from_container(self, container):
        """Container should support remove_child."""
        child = Widget()
        container.add_child(child)
        container.remove_child(child)
        assert child not in container.children

    def test_clear_children(self, container, child_widgets):
        """clear_children should remove all children."""
        for child in child_widgets:
            container.add_child(child)
        container.clear_children()
        assert len(container.children) == 0

    def test_add_children_batch(self, container, child_widgets):
        """add_children should add multiple children at once."""
        container.add_children(child_widgets)
        assert len(container.children) == len(child_widgets)

    def test_insert_child_at_index(self, container):
        """insert_child should insert at specific index."""
        c1 = Widget()
        c2 = Widget()
        c3 = Widget()
        container.add_child(c1)
        container.add_child(c2)
        container.insert_child(c3, 1)
        assert container.children[1] is c3

    def test_reorder_child(self, container):
        """reorder_child should change child position."""
        c1 = Widget()
        c2 = Widget()
        c3 = Widget()
        container.add_child(c1)
        container.add_child(c2)
        container.add_child(c3)

        container.reorder_child(c3, 0)
        assert container.children[0] is c3


class TestContainerLayout:
    """Tests for container layout functionality."""

    def test_set_layout(self, container):
        """Container layout can be set."""
        layout = Layout(direction=LayoutDirection.HORIZONTAL)
        container.layout = layout
        assert container.layout is layout

    def test_layout_none_freeform(self, container):
        """None layout means freeform positioning."""
        container.layout = None
        assert container.layout is None

    def test_layout_triggers_relayout(self, layout_container):
        """Setting layout should trigger relayout."""
        child = Widget(width=50.0, height=50.0)
        layout_container.add_child(child)
        layout_container.layout = Layout(direction=LayoutDirection.HORIZONTAL)
        assert layout_container.needs_layout is True

    def test_vertical_layout_positions(self, layout_container):
        """Vertical layout should stack children vertically."""
        c1 = Widget(width=50.0, height=30.0)
        c2 = Widget(width=50.0, height=30.0)
        c3 = Widget(width=50.0, height=30.0)

        layout_container.width = 100.0
        layout_container.height = 200.0
        layout_container.add_child(c1)
        layout_container.add_child(c2)
        layout_container.add_child(c3)
        layout_container.do_layout()

        assert c1.local_y < c2.local_y < c3.local_y

    def test_horizontal_layout_positions(self):
        """Horizontal layout should stack children horizontally."""
        c = Container(
            layout=Layout(direction=LayoutDirection.HORIZONTAL),
            width=200.0,
            height=100.0,
        )
        c1 = Widget(width=30.0, height=50.0)
        c2 = Widget(width=30.0, height=50.0)
        c3 = Widget(width=30.0, height=50.0)

        c.add_child(c1)
        c.add_child(c2)
        c.add_child(c3)
        c.do_layout()

        assert c1.local_x < c2.local_x < c3.local_x

    def test_layout_with_gap(self):
        """Layout gap should add space between children."""
        c = Container(
            layout=Layout(direction=LayoutDirection.VERTICAL, gap=10.0),
            width=100.0,
            height=200.0,
        )
        c1 = Widget(width=50.0, height=30.0)
        c2 = Widget(width=50.0, height=30.0)

        c.add_child(c1)
        c.add_child(c2)
        c.do_layout()

        expected_y = c1.local_y + c1.height + 10.0  # gap
        assert c2.local_y == expected_y

    def test_layout_with_padding(self):
        """Layout should respect container padding."""
        c = Container(
            layout=Layout(direction=LayoutDirection.VERTICAL),
            padding=Margins.all(20.0),
            width=100.0,
            height=200.0,
        )
        child = Widget(width=50.0, height=30.0)
        c.add_child(child)
        c.do_layout()

        assert child.local_x >= 20.0
        assert child.local_y >= 20.0

    def test_auto_layout_on_child_add(self):
        """Adding child should trigger auto-layout."""
        c = Container(
            layout=Layout(direction=LayoutDirection.VERTICAL),
            width=100.0,
            height=200.0,
        )
        c.do_layout()
        c.needs_layout = False

        child = Widget(width=50.0, height=30.0)
        c.add_child(child)

        assert c.needs_layout is True

    def test_auto_layout_on_child_size_change(self, layout_container):
        """Child size change should trigger layout."""
        child = Widget(width=50.0, height=30.0)
        layout_container.add_child(child)
        layout_container.do_layout()
        layout_container.needs_layout = False

        child.height = 60.0
        assert layout_container.needs_layout is True


class TestContainerClipping:
    """Tests for container clipping behavior."""

    def test_clip_children_default(self, container):
        """clip_children should default to False."""
        assert container.clip_children is False

    def test_clip_children_settable(self, container):
        """clip_children should be settable."""
        container.clip_children = True
        assert container.clip_children is True

    def test_clip_bounds(self, container):
        """clip_bounds should return content area."""
        container.width = 200.0
        container.height = 100.0
        container.padding = Margins.all(10.0)

        bounds = container.clip_bounds
        assert bounds.x == 10.0
        assert bounds.y == 10.0
        assert bounds.width == 180.0
        assert bounds.height == 80.0

    def test_is_child_visible_inside(self, container):
        """is_child_visible should return True for child inside bounds."""
        container.width = 200.0
        container.height = 200.0
        container.clip_children = True

        child = Widget(local_x=50.0, local_y=50.0, width=50.0, height=50.0)
        container.add_child(child)

        assert container.is_child_visible(child) is True

    def test_is_child_visible_outside(self, container):
        """is_child_visible should return False for child outside when clipping."""
        container.width = 100.0
        container.height = 100.0
        container.clip_children = True

        child = Widget(local_x=200.0, local_y=200.0, width=50.0, height=50.0)
        container.add_child(child)

        assert container.is_child_visible(child) is False

    def test_is_child_visible_partial(self, container):
        """is_child_visible should return True for partially visible child."""
        container.width = 100.0
        container.height = 100.0
        container.clip_children = True

        child = Widget(local_x=80.0, local_y=80.0, width=50.0, height=50.0)
        container.add_child(child)

        assert container.is_child_visible(child) is True

    def test_no_clip_always_visible(self, container):
        """Without clipping, children are always considered visible."""
        container.width = 100.0
        container.height = 100.0
        container.clip_children = False

        child = Widget(local_x=200.0, local_y=200.0, width=50.0, height=50.0)
        container.add_child(child)

        assert container.is_child_visible(child) is True


class TestNestedContainers:
    """Tests for nested containers."""

    def test_nested_container_creation(self):
        """Containers can be nested."""
        outer = Container()
        inner = Container()
        outer.add_child(inner)

        assert inner in outer.children
        assert inner.parent is outer

    def test_nested_layout_independence(self):
        """Nested containers should have independent layouts."""
        outer = Container(
            layout=Layout(direction=LayoutDirection.VERTICAL),
            width=200.0,
            height=200.0,
        )
        inner = Container(
            layout=Layout(direction=LayoutDirection.HORIZONTAL),
            width=150.0,
            height=50.0,
        )
        outer.add_child(inner)

        c1 = Widget(width=30.0, height=30.0)
        c2 = Widget(width=30.0, height=30.0)
        inner.add_child(c1)
        inner.add_child(c2)

        outer.do_layout()
        inner.do_layout()

        # Inner container children should be horizontal
        assert c1.local_x < c2.local_x

    def test_deep_nesting(self):
        """Deep nesting should work correctly."""
        root = Container(width=400.0, height=400.0)
        current = root

        for i in range(5):
            child = Container(width=300.0 - i * 50, height=300.0 - i * 50)
            current.add_child(child)
            current = child

        # Deepest container should have root as eventual ancestor
        assert current.get_root() is root

    def test_nested_clipping(self):
        """Nested clipping should be cumulative."""
        outer = Container(width=200.0, height=200.0, clip_children=True)
        inner = Container(
            local_x=50.0,
            local_y=50.0,
            width=200.0,  # Extends beyond outer
            height=200.0,
            clip_children=True,
        )
        outer.add_child(inner)

        # Inner's effective clip should be constrained by outer
        effective_clip = inner.get_effective_clip_bounds()
        assert effective_clip.right <= 200.0
        assert effective_clip.bottom <= 200.0

    def test_nested_visibility(self):
        """Nested visibility should cascade."""
        outer = Container()
        inner = Container()
        outer.add_child(inner)
        outer.visible = False

        assert inner.is_visible is False


class TestContainerScrolling:
    """Tests for container scrolling and overflow."""

    def test_scroll_offset_default(self, container):
        """scroll_offset should default to (0, 0)."""
        assert container.scroll_offset == Point.zero()

    def test_scroll_offset_settable(self, container):
        """scroll_offset should be settable."""
        container.scroll_offset = Point(50.0, 100.0)
        assert container.scroll_offset == Point(50.0, 100.0)

    def test_scroll_affects_child_position(self, container):
        """Scroll offset should affect child rendering position."""
        container.width = 100.0
        container.height = 100.0

        child = Widget(local_x=0.0, local_y=0.0, width=50.0, height=50.0)
        container.add_child(child)

        container.scroll_offset = Point(20.0, 30.0)

        # Child's visual position should be offset by scroll
        visual_pos = container.get_child_visual_position(child)
        assert visual_pos.x == -20.0
        assert visual_pos.y == -30.0

    def test_content_size(self, container):
        """content_size should return size needed for all children."""
        c1 = Widget(local_x=0.0, local_y=0.0, width=100.0, height=50.0)
        c2 = Widget(local_x=50.0, local_y=100.0, width=100.0, height=50.0)
        container.add_child(c1)
        container.add_child(c2)

        content_size = container.content_size
        assert content_size.width >= 150.0  # c2.x + c2.width
        assert content_size.height >= 150.0  # c2.y + c2.height

    def test_can_scroll_horizontal(self, container):
        """can_scroll_horizontal should return True if content wider."""
        container.width = 100.0
        child = Widget(width=200.0, height=50.0)
        container.add_child(child)

        assert container.can_scroll_horizontal is True

    def test_can_scroll_vertical(self, container):
        """can_scroll_vertical should return True if content taller."""
        container.height = 100.0
        child = Widget(width=50.0, height=200.0)
        container.add_child(child)

        assert container.can_scroll_vertical is True

    def test_scroll_to_child(self, container):
        """scroll_to_child should scroll to make child visible."""
        container.width = 100.0
        container.height = 100.0
        container.clip_children = True

        child = Widget(local_x=200.0, local_y=200.0, width=50.0, height=50.0)
        container.add_child(child)

        container.scroll_to_child(child)

        # Child should now be visible
        assert container.is_child_visible(child)

    def test_scroll_clamped(self, container):
        """Scroll should be clamped to content bounds."""
        container.width = 100.0
        container.height = 100.0

        child = Widget(width=200.0, height=200.0)
        container.add_child(child)

        # Try to scroll beyond content
        container.scroll_offset = Point(500.0, 500.0)

        # Should be clamped
        max_scroll_x = 200.0 - 100.0  # content width - container width
        max_scroll_y = 200.0 - 100.0
        assert container.scroll_offset.x <= max_scroll_x
        assert container.scroll_offset.y <= max_scroll_y


class TestContainerContentArea:
    """Tests for container content area calculations."""

    def test_content_area(self, container):
        """content_area should return bounds minus padding."""
        container.width = 200.0
        container.height = 100.0
        container.padding = Margins(10.0, 20.0, 10.0, 20.0)

        area = container.content_area
        assert area.x == 20.0  # left padding
        assert area.y == 10.0  # top padding
        assert area.width == 160.0  # 200 - 20 - 20
        assert area.height == 80.0  # 100 - 10 - 10

    def test_content_area_no_padding(self, container):
        """content_area with no padding equals bounds."""
        container.width = 200.0
        container.height = 100.0
        container.padding = Margins.zero()

        area = container.content_area
        assert area.x == 0.0
        assert area.y == 0.0
        assert area.width == 200.0
        assert area.height == 100.0


class TestContainerOverflow:
    """Tests for container overflow handling."""

    def test_overflow_default(self, container):
        """overflow should default to visible."""
        assert container.overflow == "visible"

    def test_overflow_hidden(self, container):
        """overflow hidden should clip content."""
        container.overflow = "hidden"
        assert container.overflow == "hidden"
        assert container.clip_children is True

    def test_overflow_scroll(self, container):
        """overflow scroll should enable scrolling."""
        container.overflow = "scroll"
        assert container.overflow == "scroll"

    def test_overflow_auto(self, container):
        """overflow auto should scroll only when needed."""
        container.overflow = "auto"
        assert container.overflow == "auto"


class TestContainerMinMax:
    """Tests for container min/max size constraints."""

    def test_min_width_default(self, container):
        """min_width should default to 0."""
        assert container.min_width == 0.0

    def test_min_height_default(self, container):
        """min_height should default to 0."""
        assert container.min_height == 0.0

    def test_max_width_default(self, container):
        """max_width should default to infinity or None."""
        assert container.max_width is None or container.max_width == float("inf")

    def test_max_height_default(self, container):
        """max_height should default to infinity or None."""
        assert container.max_height is None or container.max_height == float("inf")

    def test_min_width_enforced(self, container):
        """width should not go below min_width."""
        container.min_width = 100.0
        container.width = 50.0
        assert container.width >= 100.0

    def test_max_width_enforced(self, container):
        """width should not exceed max_width."""
        container.max_width = 200.0
        container.width = 300.0
        assert container.width <= 200.0


class TestLayout:
    """Tests for Layout configuration."""

    def test_layout_creation(self):
        """Layout should be creatable."""
        layout = Layout(direction=LayoutDirection.VERTICAL)
        assert layout.direction == LayoutDirection.VERTICAL

    def test_layout_default_gap(self):
        """Layout gap should default to 0."""
        layout = Layout(direction=LayoutDirection.VERTICAL)
        assert layout.gap == 0.0

    def test_layout_with_gap(self):
        """Layout can have custom gap."""
        layout = Layout(direction=LayoutDirection.VERTICAL, gap=10.0)
        assert layout.gap == 10.0

    def test_layout_alignment_default(self):
        """Layout alignment should default to start."""
        layout = Layout(direction=LayoutDirection.VERTICAL)
        assert layout.alignment == "start"

    def test_layout_alignment_center(self):
        """Layout can have center alignment."""
        layout = Layout(direction=LayoutDirection.VERTICAL, alignment="center")
        assert layout.alignment == "center"

    def test_layout_alignment_end(self):
        """Layout can have end alignment."""
        layout = Layout(direction=LayoutDirection.VERTICAL, alignment="end")
        assert layout.alignment == "end"

    def test_layout_justify_default(self):
        """Layout justify should default to start."""
        layout = Layout(direction=LayoutDirection.VERTICAL)
        assert layout.justify == "start"

    def test_layout_justify_space_between(self):
        """Layout can have space-between justify."""
        layout = Layout(direction=LayoutDirection.VERTICAL, justify="space-between")
        assert layout.justify == "space-between"


class TestLayoutDirection:
    """Tests for LayoutDirection enum."""

    def test_vertical_direction(self):
        """VERTICAL direction should exist."""
        assert LayoutDirection.VERTICAL is not None

    def test_horizontal_direction(self):
        """HORIZONTAL direction should exist."""
        assert LayoutDirection.HORIZONTAL is not None

    def test_directions_are_distinct(self):
        """Directions should be distinct."""
        assert LayoutDirection.VERTICAL != LayoutDirection.HORIZONTAL
