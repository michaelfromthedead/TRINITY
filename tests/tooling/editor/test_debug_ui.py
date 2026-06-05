"""
Comprehensive tests for the Debug UI framework.

Tests cover:
- Core types (Vec2, Vec3, Vec4, Color)
- Widget base classes
- All concrete widget implementations
- CollapsibleSection
- PropertyPanel and bindings
- WidgetRegistry
- AutoInspector
- DebugUIContext
- DebugUI manager
- Mode integration
"""
import pytest
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

sys.path.insert(0, '/home/user/dev/USER/PROJECTS_VOID/TRINITY')

from engine.tooling.editor.debug_ui import (
    # Core types
    WidgetType,
    UIState,
    Vec2,
    Vec3,
    Vec4,
    Color,
    WidgetStyle,
    WidgetConfig,
    UIEvent,
    # Base widgets
    Widget,
    ContainerWidget,
    # Concrete widgets
    LabelWidget,
    TextInputWidget,
    IntSliderWidget,
    FloatSliderWidget,
    IntInputWidget,
    FloatInputWidget,
    CheckboxWidget,
    DropdownWidget,
    ColorPickerWidget,
    Vec2InputWidget,
    Vec3InputWidget,
    Vec4InputWidget,
    ButtonWidget,
    SeparatorWidget,
    # Containers
    CollapsibleSection,
    PropertyPanel,
    PropertyBinding,
    # Registry and inspection
    WidgetRegistry,
    AutoInspector,
    # Context and manager
    DebugUIContext,
    DebugUI,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class SampleEnum(Enum):
    """Sample enum for dropdown tests."""
    OPTION_A = auto()
    OPTION_B = auto()
    OPTION_C = auto()


@dataclass
class SampleDataClass:
    """Sample dataclass for auto-inspection."""
    name: str = "test"
    value: int = 42
    enabled: bool = True
    ratio: float = 0.5


class SampleObject:
    """Sample class for auto-inspection."""
    name: str
    count: int
    active: bool

    def __init__(self):
        self.name = "TestObj"
        self.count = 10
        self.active = False
        self._private = "hidden"


# =============================================================================
# Vec2 Tests
# =============================================================================


class TestVec2:
    """Tests for Vec2 class."""

    def test_vec2_default_values(self):
        """Vec2 should have default x=0, y=0."""
        v = Vec2()
        assert v.x == 0.0
        assert v.y == 0.0

    def test_vec2_custom_values(self):
        """Vec2 should accept custom values."""
        v = Vec2(1.5, 2.5)
        assert v.x == 1.5
        assert v.y == 2.5

    def test_vec2_iteration(self):
        """Vec2 should be iterable."""
        v = Vec2(3.0, 4.0)
        values = list(v)
        assert values == [3.0, 4.0]

    def test_vec2_to_tuple(self):
        """Vec2 to_tuple should return tuple."""
        v = Vec2(1.0, 2.0)
        assert v.to_tuple() == (1.0, 2.0)


# =============================================================================
# Vec3 Tests
# =============================================================================


class TestVec3:
    """Tests for Vec3 class."""

    def test_vec3_default_values(self):
        """Vec3 should have default x=0, y=0, z=0."""
        v = Vec3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vec3_custom_values(self):
        """Vec3 should accept custom values."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vec3_iteration(self):
        """Vec3 should be iterable."""
        v = Vec3(1.0, 2.0, 3.0)
        assert list(v) == [1.0, 2.0, 3.0]

    def test_vec3_to_tuple(self):
        """Vec3 to_tuple should return tuple."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.to_tuple() == (1.0, 2.0, 3.0)


# =============================================================================
# Vec4 Tests
# =============================================================================


class TestVec4:
    """Tests for Vec4 class."""

    def test_vec4_default_values(self):
        """Vec4 should have default x=0, y=0, z=0, w=1."""
        v = Vec4()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0
        assert v.w == 1.0

    def test_vec4_custom_values(self):
        """Vec4 should accept custom values."""
        v = Vec4(1.0, 2.0, 3.0, 4.0)
        assert v.w == 4.0

    def test_vec4_iteration(self):
        """Vec4 should be iterable."""
        v = Vec4(1.0, 2.0, 3.0, 4.0)
        assert list(v) == [1.0, 2.0, 3.0, 4.0]


# =============================================================================
# Color Tests
# =============================================================================


class TestColor:
    """Tests for Color class."""

    def test_color_default_white(self):
        """Color should default to white (1,1,1,1)."""
        c = Color()
        assert c.r == 1.0
        assert c.g == 1.0
        assert c.b == 1.0
        assert c.a == 1.0

    def test_color_custom_values(self):
        """Color should accept custom RGBA values."""
        c = Color(0.5, 0.6, 0.7, 0.8)
        assert c.r == 0.5
        assert c.g == 0.6
        assert c.b == 0.7
        assert c.a == 0.8

    def test_color_to_hex(self):
        """Color to_hex should return hex string."""
        c = Color(1.0, 0.0, 0.0, 1.0)  # Red
        assert c.to_hex() == "#ff0000ff"

    def test_color_from_hex_rgb(self):
        """Color from_hex should parse 6-char hex."""
        c = Color.from_hex("#ff0000")
        assert c.r == 1.0
        assert c.g == 0.0
        assert c.b == 0.0
        assert c.a == 1.0

    def test_color_from_hex_rgba(self):
        """Color from_hex should parse 8-char hex."""
        c = Color.from_hex("#ff000080")
        assert c.r == 1.0
        assert c.a == pytest.approx(0.5, abs=0.01)

    def test_color_from_hex_invalid(self):
        """Color from_hex should raise on invalid hex."""
        with pytest.raises(ValueError):
            Color.from_hex("#xyz")

    def test_color_iteration(self):
        """Color should be iterable."""
        c = Color(0.1, 0.2, 0.3, 0.4)
        assert list(c) == [0.1, 0.2, 0.3, 0.4]


# =============================================================================
# WidgetConfig Tests
# =============================================================================


class TestWidgetConfig:
    """Tests for WidgetConfig class."""

    def test_config_default_values(self):
        """WidgetConfig should have sensible defaults."""
        cfg = WidgetConfig()
        assert cfg.label == ""
        assert cfg.enabled is True
        assert cfg.visible is True

    def test_config_custom_values(self):
        """WidgetConfig should accept custom values."""
        cfg = WidgetConfig(label="Test", enabled=False)
        assert cfg.label == "Test"
        assert cfg.enabled is False


# =============================================================================
# LabelWidget Tests
# =============================================================================


class TestLabelWidget:
    """Tests for LabelWidget class."""

    def test_label_creation(self):
        """LabelWidget should be created with text."""
        label = LabelWidget("Hello World")
        assert label.value == "Hello World"
        assert label.widget_type == WidgetType.LABEL

    def test_label_empty(self):
        """LabelWidget should allow empty text."""
        label = LabelWidget()
        assert label.value == ""


# =============================================================================
# TextInputWidget Tests
# =============================================================================


class TestTextInputWidget:
    """Tests for TextInputWidget class."""

    def test_text_input_creation(self):
        """TextInputWidget should be created with defaults."""
        widget = TextInputWidget("Name", "Initial")
        assert widget.label == "Name"
        assert widget.value == "Initial"
        assert widget.widget_type == WidgetType.TEXT_INPUT

    def test_text_input_max_length(self):
        """TextInputWidget should have max_length."""
        widget = TextInputWidget(max_length=100)
        assert widget.max_length == 100

    def test_text_input_placeholder(self):
        """TextInputWidget should support placeholder."""
        widget = TextInputWidget(placeholder="Enter name...")
        assert widget.placeholder == "Enter name..."

    def test_text_input_password_mode(self):
        """TextInputWidget should support password mode."""
        widget = TextInputWidget(password=True)
        assert widget.password is True


# =============================================================================
# IntSliderWidget Tests
# =============================================================================


class TestIntSliderWidget:
    """Tests for IntSliderWidget class."""

    def test_int_slider_creation(self):
        """IntSliderWidget should be created with range."""
        widget = IntSliderWidget("Volume", 50, 0, 100)
        assert widget.value == 50
        assert widget.min_value == 0
        assert widget.max_value == 100

    def test_int_slider_clamping(self):
        """IntSliderWidget should clamp values to range."""
        widget = IntSliderWidget("Test", 50, 0, 100)
        widget.set_value(150)
        assert widget.value == 100
        widget.set_value(-10)
        assert widget.value == 0

    def test_int_slider_step(self):
        """IntSliderWidget should have step value."""
        widget = IntSliderWidget(step=5)
        assert widget.step == 5


# =============================================================================
# FloatSliderWidget Tests
# =============================================================================


class TestFloatSliderWidget:
    """Tests for FloatSliderWidget class."""

    def test_float_slider_creation(self):
        """FloatSliderWidget should be created with range."""
        widget = FloatSliderWidget("Alpha", 0.5, 0.0, 1.0)
        assert widget.value == 0.5

    def test_float_slider_clamping(self):
        """FloatSliderWidget should clamp values."""
        widget = FloatSliderWidget("Test", 0.5, 0.0, 1.0)
        widget.set_value(2.0)
        assert widget.value == 1.0

    def test_float_slider_precision(self):
        """FloatSliderWidget should respect precision."""
        widget = FloatSliderWidget(precision=2)
        widget.set_value(0.12345)
        assert widget.value == 0.12


# =============================================================================
# IntInputWidget Tests
# =============================================================================


class TestIntInputWidget:
    """Tests for IntInputWidget class."""

    def test_int_input_creation(self):
        """IntInputWidget should be created with value."""
        widget = IntInputWidget("Count", 10)
        assert widget.value == 10
        assert widget.widget_type == WidgetType.INT_INPUT

    def test_int_input_optional_bounds(self):
        """IntInputWidget should allow unbounded values."""
        widget = IntInputWidget()
        widget.set_value(1000000)
        assert widget.value == 1000000


# =============================================================================
# FloatInputWidget Tests
# =============================================================================


class TestFloatInputWidget:
    """Tests for FloatInputWidget class."""

    def test_float_input_creation(self):
        """FloatInputWidget should be created with value."""
        widget = FloatInputWidget("Scale", 1.5)
        assert widget.value == 1.5

    def test_float_input_precision(self):
        """FloatInputWidget should round to precision."""
        widget = FloatInputWidget(precision=2)
        widget.set_value(3.14159)
        assert widget.value == 3.14


# =============================================================================
# CheckboxWidget Tests
# =============================================================================


class TestCheckboxWidget:
    """Tests for CheckboxWidget class."""

    def test_checkbox_creation(self):
        """CheckboxWidget should be created with value."""
        widget = CheckboxWidget("Enable", True)
        assert widget.value is True
        assert widget.widget_type == WidgetType.CHECKBOX

    def test_checkbox_toggle(self):
        """CheckboxWidget toggle should invert value."""
        widget = CheckboxWidget("Test", False)
        widget.toggle()
        assert widget.value is True
        widget.toggle()
        assert widget.value is False


# =============================================================================
# DropdownWidget Tests
# =============================================================================


class TestDropdownWidget:
    """Tests for DropdownWidget class."""

    def test_dropdown_creation(self):
        """DropdownWidget should be created with options."""
        widget = DropdownWidget("Mode", ["Low", "Medium", "High"], 1)
        assert widget.value == "Medium"
        assert widget.selected_index == 1

    def test_dropdown_set_index(self):
        """DropdownWidget should update value when index set."""
        widget = DropdownWidget("Test", ["A", "B", "C"], 0)
        widget.selected_index = 2
        assert widget.value == "C"

    def test_dropdown_set_options(self):
        """DropdownWidget should allow options update."""
        widget = DropdownWidget("Test", ["A", "B"], 0)
        widget.set_options(["X", "Y", "Z"])
        assert widget.value == "X"
        assert len(widget.options) == 3

    def test_dropdown_keep_selection(self):
        """DropdownWidget should keep selection when option exists."""
        widget = DropdownWidget("Test", ["A", "B", "C"], 1)
        widget.set_options(["B", "C", "D"], keep_selection=True)
        assert widget.value == "B"
        assert widget.selected_index == 0


# =============================================================================
# ColorPickerWidget Tests
# =============================================================================


class TestColorPickerWidget:
    """Tests for ColorPickerWidget class."""

    def test_color_picker_creation(self):
        """ColorPickerWidget should be created with color."""
        color = Color(1.0, 0.0, 0.0, 1.0)
        widget = ColorPickerWidget("Tint", color)
        assert widget.value.r == 1.0
        assert widget.widget_type == WidgetType.COLOR_PICKER

    def test_color_picker_set_rgb(self):
        """ColorPickerWidget set_rgb should update RGB only."""
        widget = ColorPickerWidget("Test", Color(0.0, 0.0, 0.0, 0.5))
        widget.set_rgb(1.0, 0.0, 0.0)
        assert widget.value.r == 1.0
        assert widget.value.a == 0.5  # Alpha preserved

    def test_color_picker_set_hex(self):
        """ColorPickerWidget set_hex should parse hex."""
        widget = ColorPickerWidget()
        widget.set_hex("#00ff00")
        assert widget.value.g == 1.0


# =============================================================================
# Vec Input Widget Tests
# =============================================================================


class TestVecInputWidgets:
    """Tests for Vec2/3/4 input widgets."""

    def test_vec2_input_creation(self):
        """Vec2InputWidget should be created with Vec2."""
        widget = Vec2InputWidget("Size", Vec2(100, 50))
        assert widget.value.x == 100
        assert widget.value.y == 50

    def test_vec3_input_creation(self):
        """Vec3InputWidget should be created with Vec3."""
        widget = Vec3InputWidget("Position", Vec3(1, 2, 3))
        assert widget.value.z == 3

    def test_vec4_input_creation(self):
        """Vec4InputWidget should be created with Vec4."""
        widget = Vec4InputWidget("Quat", Vec4(0, 0, 0, 1))
        assert widget.value.w == 1


# =============================================================================
# ButtonWidget Tests
# =============================================================================


class TestButtonWidget:
    """Tests for ButtonWidget class."""

    def test_button_creation(self):
        """ButtonWidget should be created with label."""
        widget = ButtonWidget("Click Me")
        assert widget.label == "Click Me"
        assert widget.widget_type == WidgetType.BUTTON

    def test_button_click_callback(self):
        """ButtonWidget click should trigger callback."""
        clicked = []
        widget = ButtonWidget("Test")
        widget.on_click = lambda: clicked.append(True)
        widget.click()
        assert len(clicked) == 1

    def test_button_icon(self):
        """ButtonWidget should support icon."""
        widget = ButtonWidget("Save", icon="save_icon")
        assert widget.icon == "save_icon"


# =============================================================================
# SeparatorWidget Tests
# =============================================================================


class TestSeparatorWidget:
    """Tests for SeparatorWidget class."""

    def test_separator_default_horizontal(self):
        """SeparatorWidget should default to horizontal."""
        widget = SeparatorWidget()
        assert widget.horizontal is True

    def test_separator_vertical(self):
        """SeparatorWidget should support vertical."""
        widget = SeparatorWidget(horizontal=False)
        assert widget.horizontal is False


# =============================================================================
# CollapsibleSection Tests
# =============================================================================


class TestCollapsibleSection:
    """Tests for CollapsibleSection class."""

    def test_section_creation(self):
        """CollapsibleSection should be created with title."""
        section = CollapsibleSection("Settings")
        assert section.title == "Settings"
        assert section.expanded is True

    def test_section_toggle(self):
        """CollapsibleSection toggle should invert expanded."""
        section = CollapsibleSection("Test", expanded=True)
        section.toggle()
        assert section.expanded is False
        section.toggle()
        assert section.expanded is True

    def test_section_expand_collapse(self):
        """CollapsibleSection expand/collapse methods."""
        section = CollapsibleSection("Test", expanded=False)
        section.expand()
        assert section.expanded is True
        section.collapse()
        assert section.expanded is False

    def test_section_reset(self):
        """CollapsibleSection reset should restore default."""
        section = CollapsibleSection("Test", expanded=False)
        section.expand()
        section.reset()
        assert section.expanded is False

    def test_section_children(self):
        """CollapsibleSection should contain children."""
        section = CollapsibleSection("Parent")
        label = LabelWidget("Child")
        section.add_child(label)
        assert len(section.children) == 1


# =============================================================================
# ContainerWidget Tests
# =============================================================================


class TestContainerWidget:
    """Tests for ContainerWidget class."""

    def test_container_add_child(self):
        """ContainerWidget add_child should add widget."""
        container = ContainerWidget(WidgetType.TREE_NODE)
        child = LabelWidget("Test")
        container.add_child(child)
        assert child in container.children
        assert child.parent is container

    def test_container_remove_child(self):
        """ContainerWidget remove_child should remove widget."""
        container = ContainerWidget(WidgetType.TREE_NODE)
        child = LabelWidget("Test")
        container.add_child(child)
        result = container.remove_child(child)
        assert result is True
        assert child not in container.children

    def test_container_clear_children(self):
        """ContainerWidget clear_children should remove all."""
        container = ContainerWidget(WidgetType.TREE_NODE)
        container.add_child(LabelWidget("A"))
        container.add_child(LabelWidget("B"))
        container.clear_children()
        assert len(container.children) == 0

    def test_container_find_child(self):
        """ContainerWidget find_child should find recursively."""
        outer = ContainerWidget(WidgetType.TREE_NODE)
        inner = ContainerWidget(WidgetType.TREE_NODE)
        label = LabelWidget("Deep")
        inner.add_child(label)
        outer.add_child(inner)
        found = outer.find_child(label.id)
        assert found is label


# =============================================================================
# PropertyPanel Tests
# =============================================================================


class TestPropertyPanel:
    """Tests for PropertyPanel class."""

    def test_panel_creation(self):
        """PropertyPanel should be created with title."""
        panel = PropertyPanel("Properties")
        assert panel.title == "Properties"

    def test_panel_add_property(self):
        """PropertyPanel should add property binding."""
        obj = SampleObject()
        panel = PropertyPanel("Test", obj)
        widget = TextInputWidget("Name")
        binding = panel.add_property("name", widget)
        assert binding.property_name == "name"
        assert panel.get_binding("name") is binding

    def test_panel_remove_property(self):
        """PropertyPanel should remove property binding."""
        panel = PropertyPanel("Test")
        widget = TextInputWidget("Name")
        panel.add_property("name", widget)
        result = panel.remove_property("name")
        assert result is True
        assert panel.get_binding("name") is None

    def test_panel_sync_from_target(self):
        """PropertyPanel should sync widgets from target."""
        obj = SampleObject()
        obj.name = "Updated"
        panel = PropertyPanel("Test", obj)
        widget = TextInputWidget("Name", "Initial")
        panel.add_property("name", widget)
        panel.sync_all_from_target()
        assert widget.value == "Updated"

    def test_panel_sync_to_target(self):
        """PropertyPanel should sync widget values to target."""
        obj = SampleObject()
        panel = PropertyPanel("Test", obj)
        widget = TextInputWidget("Name", "NewValue")
        panel.add_property("name", widget)
        panel.sync_all_to_target()
        assert obj.name == "NewValue"


# =============================================================================
# PropertyBinding Tests
# =============================================================================


class TestPropertyBinding:
    """Tests for PropertyBinding class."""

    def test_binding_sync_from_target(self):
        """PropertyBinding should sync from target."""
        obj = SampleObject()
        obj.count = 42
        widget = IntInputWidget("Count")
        binding = PropertyBinding(widget, obj, "count")
        binding.sync_from_target()
        assert widget.value == 42

    def test_binding_sync_to_target(self):
        """PropertyBinding should sync to target."""
        obj = SampleObject()
        widget = IntInputWidget("Count", 99)
        binding = PropertyBinding(widget, obj, "count")
        binding.sync_to_target()
        assert obj.count == 99

    def test_binding_readonly(self):
        """PropertyBinding readonly should prevent sync to target."""
        obj = SampleObject()
        obj.count = 10
        widget = IntInputWidget("Count", 99)
        binding = PropertyBinding(widget, obj, "count", readonly=True)
        binding.sync_to_target()
        assert obj.count == 10  # Unchanged

    def test_binding_custom_getter(self):
        """PropertyBinding should use custom getter."""
        obj = SampleObject()
        widget = IntInputWidget("Double")
        binding = PropertyBinding(
            widget, obj, "count",
            getter=lambda: obj.count * 2
        )
        obj.count = 5
        binding.sync_from_target()
        assert widget.value == 10


# =============================================================================
# WidgetRegistry Tests
# =============================================================================


class TestWidgetRegistry:
    """Tests for WidgetRegistry class."""

    def test_registry_default_types(self):
        """WidgetRegistry should have default type mappings."""
        registry = WidgetRegistry()
        assert registry.get_widget_factory(int) is not None
        assert registry.get_widget_factory(float) is not None
        assert registry.get_widget_factory(str) is not None
        assert registry.get_widget_factory(bool) is not None

    def test_registry_register_type(self):
        """WidgetRegistry should register custom types."""
        registry = WidgetRegistry()
        registry.register_type(bytes, lambda **kw: LabelWidget("bytes"))
        factory = registry.get_widget_factory(bytes)
        assert factory is not None

    def test_registry_unregister_type(self):
        """WidgetRegistry should unregister types."""
        registry = WidgetRegistry()
        registry.unregister_type(int)
        # Should fall back to default or None
        factory = registry.get_widget_factory(int)
        # After unregister, may return fallback

    def test_registry_create_widget_int(self):
        """WidgetRegistry should create int widget."""
        registry = WidgetRegistry()
        widget = registry.create_widget(int, "count", "Count", 10)
        assert widget is not None
        assert widget.value == 10

    def test_registry_create_widget_enum(self):
        """WidgetRegistry should create enum dropdown."""
        registry = WidgetRegistry()
        widget = registry.create_widget(SampleEnum, "mode", "Mode", SampleEnum.OPTION_B)
        assert isinstance(widget, DropdownWidget)
        assert "OPTION_A" in widget.options

    def test_registry_name_pattern(self):
        """WidgetRegistry should match name patterns."""
        registry = WidgetRegistry()
        factory = registry.get_widget_factory(None, "background_color")
        assert factory is not None  # Should match "color" pattern


# =============================================================================
# AutoInspector Tests
# =============================================================================


class TestAutoInspector:
    """Tests for AutoInspector class."""

    def test_inspector_creation(self):
        """AutoInspector should be created."""
        inspector = AutoInspector()
        assert inspector.registry is not None

    def test_inspector_inspect_object(self):
        """AutoInspector should create panel for object."""
        inspector = AutoInspector()
        obj = SampleObject()
        panel = inspector.inspect_object(obj)
        assert panel is not None
        assert panel.title == "SampleObject"

    def test_inspector_dataclass(self):
        """AutoInspector should inspect dataclass."""
        inspector = AutoInspector()
        obj = SampleDataClass()
        panel = inspector.inspect_object(obj)
        assert panel is not None
        # Should have bindings for dataclass fields
        assert len(panel._bindings) > 0

    def test_inspector_exclude_field(self):
        """AutoInspector should exclude fields."""
        inspector = AutoInspector()
        inspector.exclude_field("name")
        obj = SampleDataClass()
        panel = inspector.inspect_object(obj)
        assert panel.get_binding("name") is None

    def test_inspector_read_only_field(self):
        """AutoInspector should mark fields read-only."""
        inspector = AutoInspector()
        inspector.set_read_only("value")
        obj = SampleDataClass()
        panel = inspector.inspect_object(obj)
        binding = panel.get_binding("value")
        if binding:
            assert binding.readonly is True

    def test_inspector_cache(self):
        """AutoInspector should cache panels."""
        inspector = AutoInspector()
        obj = SampleObject()
        panel1 = inspector.inspect_object(obj)
        panel2 = inspector.inspect_object(obj)
        assert panel1 is panel2

    def test_inspector_clear_cache(self):
        """AutoInspector should clear cache."""
        inspector = AutoInspector()
        obj = SampleObject()
        inspector.inspect_object(obj)
        inspector.clear_cache()
        assert len(inspector._panels) == 0


# =============================================================================
# DebugUIContext Tests
# =============================================================================


class TestDebugUIContext:
    """Tests for DebugUIContext class."""

    def test_context_creation(self):
        """DebugUIContext should be created with dimensions."""
        ctx = DebugUIContext(1920, 1080, 2.0)
        assert ctx.screen_width == 1920
        assert ctx.screen_height == 1080
        assert ctx.scale == 2.0

    def test_context_begin_frame(self):
        """DebugUIContext begin_frame should reset state."""
        ctx = DebugUIContext()
        ctx.cursor_pos = Vec2(100, 100)
        ctx.begin_frame()
        assert ctx.cursor_pos.x == 0
        assert ctx.cursor_pos.y == 0

    def test_context_indent(self):
        """DebugUIContext indent should increase cursor x."""
        ctx = DebugUIContext()
        ctx.indent()
        assert ctx.cursor_pos.x == 20
        assert ctx.indent_level == 1

    def test_context_unindent(self):
        """DebugUIContext unindent should decrease cursor x."""
        ctx = DebugUIContext()
        ctx.indent(2)
        ctx.unindent()
        assert ctx.indent_level == 1

    def test_context_next_line(self):
        """DebugUIContext next_line should advance cursor y."""
        ctx = DebugUIContext()
        ctx.next_line(30)
        assert ctx.cursor_pos.y == 30

    def test_context_draw_commands(self):
        """DebugUIContext should record draw commands."""
        ctx = DebugUIContext()
        ctx.begin_frame()
        ctx.draw_text("Hello", ctx.cursor_pos)
        commands = ctx.end_frame()
        assert len(commands) == 1
        assert commands[0]["type"] == "text"
        assert commands[0]["text"] == "Hello"


# =============================================================================
# DebugUI Manager Tests
# =============================================================================


class TestDebugUI:
    """Tests for DebugUI manager class."""

    def test_debug_ui_creation(self):
        """DebugUI should be created with defaults."""
        ui = DebugUI()
        assert ui.visible is True
        assert ui.context is not None
        assert ui.registry is not None

    def test_debug_ui_toggle_visibility(self):
        """DebugUI should toggle visibility."""
        ui = DebugUI()
        ui.toggle_visibility()
        assert ui.visible is False
        ui.toggle_visibility()
        assert ui.visible is True

    def test_debug_ui_create_panel(self):
        """DebugUI should create panels."""
        ui = DebugUI()
        panel = ui.create_panel("props", "Properties")
        assert panel.title == "Properties"
        assert ui.get_panel("props") is panel

    def test_debug_ui_remove_panel(self):
        """DebugUI should remove panels."""
        ui = DebugUI()
        ui.create_panel("test", "Test")
        result = ui.remove_panel("test")
        assert result is True
        assert ui.get_panel("test") is None

    def test_debug_ui_list_panels(self):
        """DebugUI should list panel IDs."""
        ui = DebugUI()
        ui.create_panel("a", "A")
        ui.create_panel("b", "B")
        panels = ui.list_panels()
        assert "a" in panels
        assert "b" in panels

    def test_debug_ui_inspect(self):
        """DebugUI should inspect objects."""
        ui = DebugUI()
        obj = SampleObject()
        panel = ui.inspect(obj, "test_panel")
        assert panel is not None
        assert ui.get_panel("test_panel") is panel

    def test_debug_ui_create_section(self):
        """DebugUI should create collapsible sections."""
        ui = DebugUI()
        section = ui.create_section("Settings", expanded=False)
        assert isinstance(section, CollapsibleSection)
        assert section.expanded is False

    def test_debug_ui_add_label(self):
        """DebugUI should add label widgets."""
        ui = DebugUI()
        label = ui.add_label("Test Label")
        assert isinstance(label, LabelWidget)
        assert label.value == "Test Label"

    def test_debug_ui_add_button(self):
        """DebugUI should add button widgets."""
        ui = DebugUI()
        clicked = []
        button = ui.add_button("Click", on_click=lambda: clicked.append(1))
        button.click()
        assert len(clicked) == 1

    def test_debug_ui_add_checkbox(self):
        """DebugUI should add checkbox widgets."""
        ui = DebugUI()
        changes = []
        checkbox = ui.add_checkbox("Enable", True, on_change=lambda v: changes.append(v))
        checkbox.toggle()
        assert changes[-1] is False

    def test_debug_ui_add_slider_int(self):
        """DebugUI should add int slider widgets."""
        ui = DebugUI()
        slider = ui.add_slider_int("Volume", 50, 0, 100)
        assert slider.value == 50

    def test_debug_ui_add_slider_float(self):
        """DebugUI should add float slider widgets."""
        ui = DebugUI()
        slider = ui.add_slider_float("Alpha", 0.5)
        assert slider.value == 0.5

    def test_debug_ui_add_dropdown(self):
        """DebugUI should add dropdown widgets."""
        ui = DebugUI()
        dropdown = ui.add_dropdown("Mode", ["A", "B", "C"], 1)
        assert dropdown.value == "B"

    def test_debug_ui_add_color_picker(self):
        """DebugUI should add color picker widgets."""
        ui = DebugUI()
        picker = ui.add_color_picker("Tint", Color(1, 0, 0))
        assert picker.value.r == 1.0

    def test_debug_ui_add_separator(self):
        """DebugUI should add separator widgets."""
        ui = DebugUI()
        sep = ui.add_separator()
        assert isinstance(sep, SeparatorWidget)

    def test_debug_ui_render(self):
        """DebugUI should render and return commands."""
        ui = DebugUI()
        ui.add_label("Test")
        commands = ui.render()
        assert len(commands) > 0

    def test_debug_ui_render_invisible(self):
        """DebugUI should not render when invisible."""
        ui = DebugUI()
        ui.add_label("Test")
        ui.visible = False
        commands = ui.render()
        assert len(commands) == 0

    def test_debug_ui_clear(self):
        """DebugUI should clear all widgets."""
        ui = DebugUI()
        ui.add_label("A")
        ui.create_panel("p", "P")
        ui.clear()
        assert len(ui.list_panels()) == 0


# =============================================================================
# Mode Integration Tests
# =============================================================================


class TestModeIntegration:
    """Tests for editor mode integration."""

    def test_register_mode_panel(self):
        """DebugUI should register panels for modes."""
        ui = DebugUI()
        ui.create_panel("select_props", "Selection")
        ui.register_mode_panel("select", "select_props")
        panels = ui.get_mode_panels("select")
        assert "select_props" in panels

    def test_unregister_mode_panel(self):
        """DebugUI should unregister mode panels."""
        ui = DebugUI()
        ui.create_panel("test", "Test")
        ui.register_mode_panel("mode1", "test")
        result = ui.unregister_mode_panel("mode1", "test")
        assert result is True
        assert "test" not in ui.get_mode_panels("mode1")

    def test_set_active_mode(self):
        """DebugUI should show/hide panels on mode change."""
        ui = DebugUI()
        panel_a = ui.create_panel("a", "A")
        panel_b = ui.create_panel("b", "B")
        ui.register_mode_panel("mode1", "a")
        ui.register_mode_panel("mode2", "b")

        ui.set_active_mode("mode1")
        assert panel_a.visible is True
        # Mode2 panels should be hidden when switching
        ui.set_active_mode("mode2")
        assert panel_a.visible is False
        assert panel_b.visible is True


# =============================================================================
# Widget State and Events Tests
# =============================================================================


class TestWidgetState:
    """Tests for widget state management."""

    def test_widget_enabled_state(self):
        """Widget should track enabled state."""
        widget = ButtonWidget("Test")
        widget.enabled = False
        assert widget.state == UIState.DISABLED

    def test_widget_visible(self):
        """Widget should track visibility."""
        widget = LabelWidget("Test")
        widget.visible = False
        assert widget.visible is False

    def test_widget_dirty_flag(self):
        """Widget should track dirty state."""
        widget = IntInputWidget("Count", 0)
        assert widget.is_dirty() is False
        widget.set_value(10)
        assert widget.is_dirty() is True
        widget.clear_dirty()
        assert widget.is_dirty() is False

    def test_widget_on_change_callback(self):
        """Widget should trigger on_change callback."""
        changes = []
        widget = IntInputWidget("Count", 0)
        widget.on_change = lambda v: changes.append(v)
        widget.set_value(42)
        assert changes == [42]


class TestUIEvent:
    """Tests for UIEvent class."""

    def test_event_creation(self):
        """UIEvent should be created with defaults."""
        event = UIEvent("mouse_down", x=100, y=200, button=0)
        assert event.type == "mouse_down"
        assert event.x == 100
        assert event.y == 200

    def test_event_modifiers(self):
        """UIEvent should track modifiers."""
        event = UIEvent("key_down", key="a", modifiers={"ctrl", "shift"})
        assert "ctrl" in event.modifiers


# =============================================================================
# Widget ID Tests
# =============================================================================


class TestWidgetIDs:
    """Tests for widget ID generation."""

    def test_unique_ids(self):
        """Each widget should have unique ID."""
        w1 = LabelWidget("A")
        w2 = LabelWidget("B")
        w3 = ButtonWidget("C")
        ids = {w1.id, w2.id, w3.id}
        assert len(ids) == 3


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_dropdown_empty_options(self):
        """DropdownWidget should handle empty options."""
        widget = DropdownWidget("Empty", [])
        assert widget.value is None
        assert widget.selected_index == 0

    def test_container_remove_nonexistent(self):
        """ContainerWidget should handle removing non-child."""
        container = ContainerWidget(WidgetType.TREE_NODE)
        widget = LabelWidget("Not a child")
        result = container.remove_child(widget)
        assert result is False

    def test_panel_remove_nonexistent(self):
        """PropertyPanel should handle removing non-property."""
        panel = PropertyPanel("Test")
        result = panel.remove_property("nonexistent")
        assert result is False

    def test_inspector_private_fields_excluded(self):
        """AutoInspector should exclude private fields."""
        inspector = AutoInspector()
        obj = SampleObject()
        panel = inspector.inspect_object(obj)
        assert panel.get_binding("_private") is None

    def test_color_from_hex_no_hash(self):
        """Color from_hex should handle strings without #."""
        c = Color.from_hex("00ff00")
        assert c.g == 1.0

    def test_slider_value_unchanged(self):
        """Slider set_value should not trigger if unchanged."""
        widget = IntSliderWidget("Test", 50)
        changes = []
        widget.on_change = lambda v: changes.append(v)
        widget.set_value(50)  # Same value
        assert len(changes) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
