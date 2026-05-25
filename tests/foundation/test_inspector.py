"""
Comprehensive unit tests for the Inspector system.
Tests object visualization, navigation, views, and widget mapping.
"""
import pytest
import sys
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.inspector import (
    inspector, Inspector, InspectorPanel, TextUIContext,
    FieldsView, RawView, JSONView, CollectionView, HistoryEntry, View
)


class TestInspector:
    """Tests for the main Inspector class."""

    def test_inspect_returns_panel(self):
        """Inspector.inspect should return an InspectorPanel."""
        obj = {"key": "value"}
        panel = inspector.inspect(obj)
        assert isinstance(panel, InspectorPanel)

    def test_panel_has_target(self):
        """The panel should store a reference to the target object."""
        obj = [1, 2, 3]
        panel = inspector.inspect(obj)
        assert panel.target is obj

    def test_panel_has_views(self):
        """The panel should have at least one available view."""
        obj = {"test": 1}
        panel = inspector.inspect(obj)
        assert len(panel.views) > 0

    def test_default_view_is_set(self):
        """A default view should be selected upon inspection."""
        @dataclass
        class Simple:
            value: int = 42

        panel = inspector.inspect(Simple())
        assert panel.current_view is not None

    def test_inspector_singleton_pattern(self):
        """The module-level inspector should be an Inspector instance."""
        assert isinstance(inspector, Inspector)

    def test_new_inspector_instance(self):
        """Creating a new Inspector should work independently."""
        new_inspector = Inspector()
        panel = new_inspector.inspect({"x": 1})
        assert isinstance(panel, InspectorPanel)


class TestNavigation:
    """Tests for panel navigation functionality."""

    def test_navigate_to(self):
        """navigate_to should change the target object."""
        outer = {"inner": {"value": 1}}
        panel = inspector.inspect(outer)

        panel.navigate_to(outer["inner"])
        assert panel.target == {"value": 1}

    def test_back_navigation(self):
        """back() should return to the previous object."""
        a = {"name": "a"}
        b = {"name": "b"}

        panel = inspector.inspect(a)
        panel.navigate_to(b)

        assert panel.can_go_back is True
        result = panel.back()
        assert result is True
        assert panel.target == a

    def test_forward_navigation(self):
        """forward() should move to the next object in history."""
        a = {"name": "a"}
        b = {"name": "b"}

        panel = inspector.inspect(a)
        panel.navigate_to(b)
        panel.back()

        assert panel.can_go_forward is True
        result = panel.forward()
        assert result is True
        assert panel.target == b

    def test_initial_state_no_back(self):
        """Initially there should be no back navigation available."""
        panel = inspector.inspect({"x": 1})
        assert panel.can_go_back is False

    def test_initial_state_no_forward(self):
        """Initially there should be no forward navigation available."""
        panel = inspector.inspect({"x": 1})
        assert panel.can_go_forward is False

    def test_back_returns_false_when_unavailable(self):
        """back() should return False when no history is available."""
        panel = inspector.inspect({"x": 1})
        result = panel.back()
        assert result is False

    def test_forward_returns_false_when_unavailable(self):
        """forward() should return False when at end of history."""
        panel = inspector.inspect({"x": 1})
        result = panel.forward()
        assert result is False

    def test_navigate_clears_forward_history(self):
        """Navigating to a new object should clear forward history."""
        a = {"name": "a"}
        b = {"name": "b"}
        c = {"name": "c"}

        panel = inspector.inspect(a)
        panel.navigate_to(b)
        panel.back()  # Now at 'a', can go forward to 'b'

        assert panel.can_go_forward is True
        panel.navigate_to(c)  # Should clear forward history
        assert panel.can_go_forward is False

    def test_history_property(self):
        """history property should return list of HistoryEntry objects."""
        a = {"name": "a"}
        b = {"name": "b"}

        panel = inspector.inspect(a)
        panel.navigate_to(b)

        history = panel.history
        assert len(history) == 2
        assert isinstance(history[0], HistoryEntry)
        assert history[0].obj == a
        assert history[1].obj == b


class TestViews:
    """Tests for view management and rendering."""

    def test_set_view(self):
        """set_view should change the current view."""
        panel = inspector.inspect({"x": 1})
        views = panel.views
        if len(views) > 1:
            result = panel.set_view(views[1].name)
            assert result is True
            assert panel.current_view.name == views[1].name

    def test_set_view_nonexistent(self):
        """set_view should return False for nonexistent views."""
        panel = inspector.inspect({"x": 1})
        result = panel.set_view("NonexistentView")
        assert result is False

    def test_render_returns_string(self):
        """render should return a string."""
        panel = inspector.inspect({"value": 42})
        ctx = TextUIContext()
        result = panel.render(ctx)
        assert isinstance(result, str)

    def test_render_with_default_context(self):
        """render should work without providing a context."""
        panel = inspector.inspect({"value": 42})
        result = panel.render()
        assert isinstance(result, str)

    def test_get_views_filters_by_capability(self):
        """get_views should return only views that can render the object."""
        insp = Inspector()
        dict_views = insp.get_views({"x": 1})
        # CollectionView should be able to render dicts
        view_names = [v.name for v in dict_views]
        assert "Collection" in view_names


class TestTextUIContext:
    """Tests for the TextUIContext class."""

    def test_text_output(self):
        """text() should add content to output."""
        ctx = TextUIContext()
        ctx.text("Hello")
        output = ctx.get_output()
        assert "Hello" in output

    def test_label_output(self):
        """label() should add labeled content to output."""
        ctx = TextUIContext()
        ctx.label("Name", "Alice")
        output = ctx.get_output()
        assert "Name" in output
        assert "Alice" in output

    def test_input_output(self):
        """input() should add input field to output."""
        ctx = TextUIContext()
        ctx.input("field", "value", lambda v: None)
        output = ctx.get_output()
        assert "field" in output
        assert "value" in output

    def test_button_output(self):
        """button() should add button to output."""
        ctx = TextUIContext()
        ctx.button("Click Me", lambda: None)
        output = ctx.get_output()
        assert "Click Me" in output

    def test_group_returns_new_context(self):
        """group() should return a new indented context."""
        ctx = TextUIContext()
        sub_ctx = ctx.group("Section")
        assert isinstance(sub_ctx, TextUIContext)
        output = ctx.get_output()
        assert "Section" in output

    def test_set_value_calls_callback(self):
        """set_value should call the registered callback."""
        ctx = TextUIContext()
        received = []
        ctx.input("field", "old", lambda v: received.append(v))
        ctx.set_value("field", "new")
        assert received == ["new"]

    def test_click_button_calls_callback(self):
        """click_button should call the registered callback."""
        ctx = TextUIContext()
        clicked = []
        ctx.button("Test", lambda: clicked.append(True))
        ctx.click_button("Test")
        assert clicked == [True]

    def test_indentation(self):
        """Nested contexts should have proper indentation."""
        ctx = TextUIContext()
        sub_ctx = ctx.group("Section")
        sub_ctx.text("Nested content")
        output = sub_ctx.get_output()
        # Should have 2 spaces of indentation
        assert "  Nested content" in output


class TestWidgetMapping:
    """Tests for widget type mapping."""

    def test_get_widget_for_bool(self):
        """Should return a widget factory for bool type."""
        widget = inspector.get_widget(bool)
        assert widget is not None
        assert callable(widget)

    def test_get_widget_for_int(self):
        """Should return a widget factory for int type."""
        widget = inspector.get_widget(int)
        assert widget is not None
        assert callable(widget)

    def test_get_widget_for_float(self):
        """Should return a widget factory for float type."""
        widget = inspector.get_widget(float)
        assert widget is not None
        assert callable(widget)

    def test_get_widget_for_str(self):
        """Should return a widget factory for str type."""
        widget = inspector.get_widget(str)
        assert widget is not None
        assert callable(widget)

    def test_get_widget_for_list(self):
        """Should return a widget factory for list type."""
        widget = inspector.get_widget(list)
        assert widget is not None
        assert callable(widget)

    def test_get_widget_for_dict(self):
        """Should return a widget factory for dict type."""
        widget = inspector.get_widget(dict)
        assert widget is not None
        assert callable(widget)

    def test_get_widget_for_enum(self):
        """Should return enum widget for Enum subclasses."""
        class Color(Enum):
            RED = 1
            GREEN = 2

        widget = inspector.get_widget(Color)
        assert widget is not None
        assert callable(widget)

    def test_get_widget_for_custom_type(self):
        """Should return object widget for unknown types."""
        class CustomClass:
            pass

        widget = inspector.get_widget(CustomClass)
        assert widget is not None
        assert callable(widget)


class TestBuiltInViews:
    """Tests for built-in view implementations."""

    def test_fields_view_can_render_dataclass(self):
        """FieldsView should be able to render dataclass instances."""
        @dataclass
        class Sample:
            name: str = "test"

        view = FieldsView()
        assert view.can_render(Sample()) is True

    def test_fields_view_cannot_render_primitives(self):
        """FieldsView should not render primitives."""
        view = FieldsView()
        assert view.can_render(42) is False
        assert view.can_render("string") is False
        assert view.can_render(True) is False
        assert view.can_render(None) is False

    def test_raw_view_can_render_objects(self):
        """RawView should render objects with __dict__ or __slots__."""
        class HasDict:
            def __init__(self):
                self.x = 1

        view = RawView()
        assert view.can_render(HasDict()) is True

    def test_json_view_render(self):
        """JSONView should render serializable objects."""
        view = JSONView()
        if view.can_render({}):
            ctx = TextUIContext()
            result = view.render({"key": "value"}, ctx)
            assert isinstance(result, str)

    def test_collection_view_can_render_list(self):
        """CollectionView should render lists."""
        view = CollectionView()
        assert view.can_render([1, 2, 3]) is True

    def test_collection_view_can_render_dict(self):
        """CollectionView should render dicts."""
        view = CollectionView()
        assert view.can_render({"a": 1}) is True

    def test_collection_view_can_render_set(self):
        """CollectionView should render sets."""
        view = CollectionView()
        assert view.can_render({1, 2, 3}) is True

    def test_collection_view_can_render_tuple(self):
        """CollectionView should render tuples."""
        view = CollectionView()
        assert view.can_render((1, 2, 3)) is True

    def test_collection_view_cannot_render_string(self):
        """CollectionView should not render strings (even though iterable)."""
        view = CollectionView()
        assert view.can_render("string") is False


class TestCustomViews:
    """Tests for registering custom views."""

    def test_register_custom_view(self):
        """register_view should add a custom view."""
        class CustomView:
            name = "custom"
            def can_render(self, obj): return isinstance(obj, str)
            def render(self, obj, ctx):
                ctx.text(f"Custom: {obj}")
                return ctx.get_output() if hasattr(ctx, "get_output") else ""

        test_inspector = Inspector()
        original_count = len(test_inspector._views)
        test_inspector.register_view(CustomView())
        assert len(test_inspector._views) == original_count + 1

    def test_custom_view_takes_priority(self):
        """Custom views should be checked before built-in views."""
        class PriorityView:
            name = "priority"
            def can_render(self, obj): return True
            def render(self, obj, ctx):
                ctx.text("Priority view")
                return ctx.get_output() if hasattr(ctx, "get_output") else ""

        test_inspector = Inspector()
        test_inspector.register_view(PriorityView())

        views = test_inspector.get_views({"test": 1})
        assert views[0].name == "priority"

    def test_register_custom_widget(self):
        """register_widget should add a custom widget factory."""
        class CustomType:
            pass

        def custom_widget(name, value, ctx, on_change):
            ctx.text(f"Custom widget for {name}")

        test_inspector = Inspector()
        test_inspector.register_widget(CustomType, custom_widget)

        widget = test_inspector.get_widget(CustomType)
        assert widget is custom_widget


class TestHistoryEntry:
    """Tests for the HistoryEntry dataclass."""

    def test_history_entry_creation(self):
        """HistoryEntry should store object and optional view name."""
        obj = {"test": 1}
        entry = HistoryEntry(obj)
        assert entry.obj is obj
        assert entry.view_name is None

    def test_history_entry_with_view(self):
        """HistoryEntry should store view name when provided."""
        entry = HistoryEntry({"test": 1}, view_name="Fields")
        assert entry.view_name == "Fields"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_inspect_none(self):
        """Should handle None gracefully."""
        panel = inspector.inspect(None)
        assert panel.target is None

    def test_inspect_empty_dict(self):
        """Should handle empty dict."""
        panel = inspector.inspect({})
        result = panel.render()
        assert isinstance(result, str)

    def test_inspect_empty_list(self):
        """Should handle empty list."""
        panel = inspector.inspect([])
        result = panel.render()
        assert isinstance(result, str)

    def test_inspect_nested_structure(self):
        """Should handle deeply nested structures."""
        nested = {"a": {"b": {"c": {"d": [1, 2, 3]}}}}
        panel = inspector.inspect(nested)
        assert panel.target == nested

    def test_multiple_navigation_steps(self):
        """Should handle multiple navigation steps correctly."""
        objs = [{"id": i} for i in range(5)]
        panel = inspector.inspect(objs[0])

        for i in range(1, 5):
            panel.navigate_to(objs[i])

        assert panel.target == objs[4]
        assert panel.can_go_back is True

        # Go back to start
        for i in range(4):
            panel.back()

        assert panel.target == objs[0]
        assert panel.can_go_back is False
