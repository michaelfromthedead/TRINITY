"""
Comprehensive tests for the @debuggable decorator and auto-inspector.

Tests cover:
- DebugFieldConfig and debug_field() helper
- DebugSectionConfig and debug_section() helper
- DebugConfig class-level configuration
- DebuggableRegistry singleton
- DebuggablePanel generation
- DebuggableInspector integration
- @debuggable decorator
- Widget creation for all types
- Field visibility (show_if)
- Sections and grouping
- Read-only fields
- Custom widgets
- Enum handling
- Choice dropdowns
- Value synchronization
- Dirty tracking
"""
import pytest
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

sys.path.insert(0, '/home/user/dev/USER/PROJECTS_VOID/TRINITY')

from engine.tooling.editor.debuggable import (
    # Core decorator
    debuggable,
    debug_field,
    debug_section,
    # Configuration classes
    DebugConfig,
    DebugFieldConfig,
    DebugSectionConfig,
    WidgetHint,
    # Panel and inspector
    DebuggablePanel,
    DebuggableInspector,
    DebuggableRegistry,
    # Helper functions
    is_debuggable,
    get_debug_panel,
    create_debug_ui_for,
)

from engine.tooling.editor.debug_ui import (
    AutoInspector,
    CheckboxWidget,
    CollapsibleSection,
    Color,
    ColorPickerWidget,
    DebugUI,
    DebugUIContext,
    DropdownWidget,
    FloatInputWidget,
    FloatSliderWidget,
    IntInputWidget,
    IntSliderWidget,
    PropertyPanel,
    TextInputWidget,
    Vec2,
    Vec2InputWidget,
    Vec3,
    Vec3InputWidget,
    Vec4,
    Vec4InputWidget,
    Widget,
    WidgetRegistry,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class SampleEnum(Enum):
    """Sample enum for testing."""
    OPTION_A = auto()
    OPTION_B = auto()
    OPTION_C = auto()


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test."""
    DebuggableRegistry.reset_instance()
    yield
    DebuggableRegistry.reset_instance()


@pytest.fixture
def debug_ui():
    """Create a DebugUI instance for testing."""
    return DebugUI(800, 600)


@pytest.fixture
def debug_context():
    """Create a DebugUIContext for rendering tests."""
    return DebugUIContext(800, 600)


# =============================================================================
# DebugFieldConfig Tests
# =============================================================================


class TestDebugFieldConfig:
    """Tests for DebugFieldConfig."""

    def test_default_values(self):
        """DebugFieldConfig should have sensible defaults."""
        cfg = DebugFieldConfig()
        assert cfg.label is None
        assert cfg.tooltip == ""
        assert cfg.order == 0
        assert cfg.min_value is None
        assert cfg.max_value is None
        assert cfg.step is None
        assert cfg.precision == 3
        assert cfg.widget == WidgetHint.AUTO
        assert cfg.custom_widget is None
        assert cfg.hidden is False
        assert cfg.readonly is False
        assert cfg.show_if is None
        assert cfg.section is None

    def test_custom_values(self):
        """DebugFieldConfig should accept custom values."""
        cfg = DebugFieldConfig(
            label="Health",
            tooltip="Player health",
            order=1,
            min_value=0,
            max_value=100,
            step=5,
            precision=0,
            widget=WidgetHint.SLIDER,
            hidden=False,
            readonly=True,
            section="Stats",
        )
        assert cfg.label == "Health"
        assert cfg.tooltip == "Player health"
        assert cfg.order == 1
        assert cfg.min_value == 0
        assert cfg.max_value == 100
        assert cfg.step == 5
        assert cfg.precision == 0
        assert cfg.widget == WidgetHint.SLIDER
        assert cfg.readonly is True
        assert cfg.section == "Stats"

    def test_show_if_callable(self):
        """DebugFieldConfig should accept show_if callable."""
        cfg = DebugFieldConfig(show_if=lambda obj: obj.enabled)
        assert cfg.show_if is not None
        assert callable(cfg.show_if)

    def test_validator_callable(self):
        """DebugFieldConfig should accept validator callable."""
        cfg = DebugFieldConfig(validator=lambda val: val >= 0)
        assert cfg.validator is not None
        assert cfg.validator(10) is True
        assert cfg.validator(-5) is False

    def test_on_change_callable(self):
        """DebugFieldConfig should accept on_change callable."""
        called = []
        cfg = DebugFieldConfig(on_change=lambda old, new: called.append((old, new)))
        assert cfg.on_change is not None
        cfg.on_change(10, 20)
        assert called == [(10, 20)]

    def test_choices_list(self):
        """DebugFieldConfig should accept choices list."""
        cfg = DebugFieldConfig(choices=["Low", "Medium", "High"])
        assert cfg.choices == ["Low", "Medium", "High"]

    def test_choice_labels_dict(self):
        """DebugFieldConfig should accept choice labels dict."""
        cfg = DebugFieldConfig(
            choices=[1, 2, 3],
            choice_labels={1: "One", 2: "Two", 3: "Three"},
        )
        assert cfg.choice_labels[1] == "One"
        assert cfg.choice_labels[2] == "Two"


# =============================================================================
# debug_field() Helper Tests
# =============================================================================


class TestDebugFieldHelper:
    """Tests for debug_field() helper function."""

    def test_returns_metadata_dict(self):
        """debug_field() should return dict with debug_field key."""
        result = debug_field()
        assert isinstance(result, dict)
        assert "debug_field" in result
        assert isinstance(result["debug_field"], DebugFieldConfig)

    def test_passes_all_parameters(self):
        """debug_field() should pass all parameters to config."""
        result = debug_field(
            label="Test",
            tooltip="Help",
            order=5,
            min_value=0,
            max_value=100,
            step=10,
            precision=2,
            widget=WidgetHint.SLIDER,
            hidden=True,
            readonly=True,
            section="Group",
        )
        cfg = result["debug_field"]
        assert cfg.label == "Test"
        assert cfg.tooltip == "Help"
        assert cfg.order == 5
        assert cfg.min_value == 0
        assert cfg.max_value == 100
        assert cfg.step == 10
        assert cfg.precision == 2
        assert cfg.widget == WidgetHint.SLIDER
        assert cfg.hidden is True
        assert cfg.readonly is True
        assert cfg.section == "Group"

    def test_widget_kwargs(self):
        """debug_field() should accept widget_kwargs."""
        result = debug_field(widget_kwargs={"placeholder": "Enter name"})
        cfg = result["debug_field"]
        assert cfg.widget_kwargs == {"placeholder": "Enter name"}


# =============================================================================
# DebugSectionConfig Tests
# =============================================================================


class TestDebugSectionConfig:
    """Tests for DebugSectionConfig."""

    def test_default_values(self):
        """DebugSectionConfig should have sensible defaults."""
        cfg = DebugSectionConfig(name="test")
        assert cfg.name == "test"
        assert cfg.label is None
        assert cfg.expanded is True
        assert cfg.order == 0
        assert cfg.icon == ""
        assert cfg.collapsible is True
        assert cfg.show_if is None

    def test_custom_values(self):
        """DebugSectionConfig should accept custom values."""
        cfg = DebugSectionConfig(
            name="stats",
            label="Player Stats",
            expanded=False,
            order=1,
            icon="stats-icon",
            collapsible=True,
        )
        assert cfg.name == "stats"
        assert cfg.label == "Player Stats"
        assert cfg.expanded is False
        assert cfg.order == 1
        assert cfg.icon == "stats-icon"


class TestDebugSectionHelper:
    """Tests for debug_section() helper function."""

    def test_creates_section_config(self):
        """debug_section() should create DebugSectionConfig."""
        result = debug_section("test", label="Test Section", expanded=True)
        assert isinstance(result, DebugSectionConfig)
        assert result.name == "test"
        assert result.label == "Test Section"
        assert result.expanded is True


# =============================================================================
# DebugConfig Tests
# =============================================================================


class TestDebugConfig:
    """Tests for DebugConfig class."""

    def test_default_values(self):
        """DebugConfig should have sensible defaults."""
        cfg = DebugConfig()
        assert cfg.title is None
        assert cfg.expanded is True
        assert cfg.auto_sync is True
        assert cfg.exclude_private is True
        assert cfg.exclude_dunder is True
        assert len(cfg.exclude_fields) == 0
        assert cfg.include_fields is None
        assert len(cfg.sections) == 0
        assert cfg.default_section is None
        assert cfg.category == "General"

    def test_add_section(self):
        """DebugConfig should support adding sections."""
        cfg = DebugConfig()
        section = DebugSectionConfig(name="test", label="Test")
        cfg.add_section(section)
        assert len(cfg.sections) == 1
        assert cfg.sections[0].name == "test"

    def test_get_section(self):
        """DebugConfig should support getting sections by name."""
        cfg = DebugConfig()
        section = DebugSectionConfig(name="test", label="Test")
        cfg.add_section(section)
        found = cfg.get_section("test")
        assert found is not None
        assert found.name == "test"

    def test_get_section_not_found(self):
        """DebugConfig get_section should return None for unknown."""
        cfg = DebugConfig()
        assert cfg.get_section("nonexistent") is None


# =============================================================================
# DebuggableRegistry Tests
# =============================================================================


class TestDebuggableRegistry:
    """Tests for DebuggableRegistry singleton."""

    def test_singleton_pattern(self):
        """DebuggableRegistry should be a singleton."""
        r1 = DebuggableRegistry.get_instance()
        r2 = DebuggableRegistry.get_instance()
        assert r1 is r2

    def test_reset_instance(self):
        """DebuggableRegistry.reset_instance should create new instance."""
        r1 = DebuggableRegistry.get_instance()
        DebuggableRegistry.reset_instance()
        r2 = DebuggableRegistry.get_instance()
        assert r1 is not r2

    def test_register_class(self):
        """DebuggableRegistry should register classes."""
        class TestClass:
            pass

        registry = DebuggableRegistry.get_instance()
        config = DebugConfig(title="Test")
        registry.register(TestClass, config, {})
        assert registry.is_debuggable(TestClass)

    def test_unregister_class(self):
        """DebuggableRegistry should unregister classes."""
        class TestClass:
            pass

        registry = DebuggableRegistry.get_instance()
        config = DebugConfig()
        registry.register(TestClass, config, {})
        assert registry.is_debuggable(TestClass)
        result = registry.unregister(TestClass)
        assert result is True
        assert not registry.is_debuggable(TestClass)

    def test_unregister_nonexistent(self):
        """DebuggableRegistry.unregister should return False for unknown."""
        class TestClass:
            pass

        registry = DebuggableRegistry.get_instance()
        assert registry.unregister(TestClass) is False

    def test_is_debuggable_with_instance(self):
        """DebuggableRegistry.is_debuggable should work with instances."""
        class TestClass:
            pass

        registry = DebuggableRegistry.get_instance()
        registry.register(TestClass, DebugConfig(), {})
        obj = TestClass()
        assert registry.is_debuggable(obj)

    def test_get_config(self):
        """DebuggableRegistry should return config for class."""
        class TestClass:
            pass

        registry = DebuggableRegistry.get_instance()
        config = DebugConfig(title="My Test")
        registry.register(TestClass, config, {})
        retrieved = registry.get_config(TestClass)
        assert retrieved is not None
        assert retrieved.title == "My Test"

    def test_get_field_configs(self):
        """DebuggableRegistry should return field configs."""
        class TestClass:
            pass

        registry = DebuggableRegistry.get_instance()
        field_cfgs = {"name": DebugFieldConfig(label="Name")}
        registry.register(TestClass, DebugConfig(), field_cfgs)
        retrieved = registry.get_field_configs(TestClass)
        assert "name" in retrieved
        assert retrieved["name"].label == "Name"

    def test_get_field_config_single(self):
        """DebuggableRegistry should return single field config."""
        class TestClass:
            pass

        registry = DebuggableRegistry.get_instance()
        field_cfgs = {"name": DebugFieldConfig(label="Name")}
        registry.register(TestClass, DebugConfig(), field_cfgs)
        cfg = registry.get_field_config(TestClass, "name")
        assert cfg is not None
        assert cfg.label == "Name"

    def test_list_classes(self):
        """DebuggableRegistry should list all registered classes."""
        class ClassA:
            pass

        class ClassB:
            pass

        registry = DebuggableRegistry.get_instance()
        registry.register(ClassA, DebugConfig(), {})
        registry.register(ClassB, DebugConfig(), {})
        classes = registry.list_classes()
        assert ClassA in classes
        assert ClassB in classes

    def test_list_classes_by_category(self):
        """DebuggableRegistry should list classes by category."""
        class ClassA:
            pass

        class ClassB:
            pass

        registry = DebuggableRegistry.get_instance()
        registry.register(ClassA, DebugConfig(category="Game"), {})
        registry.register(ClassB, DebugConfig(category="UI"), {})
        game_classes = registry.list_classes_by_category("Game")
        assert ClassA in game_classes
        assert ClassB not in game_classes

    def test_get_categories(self):
        """DebuggableRegistry should return all categories."""
        class ClassA:
            pass

        class ClassB:
            pass

        registry = DebuggableRegistry.get_instance()
        registry.register(ClassA, DebugConfig(category="Game"), {})
        registry.register(ClassB, DebugConfig(category="UI"), {})
        categories = registry.get_categories()
        assert "Game" in categories
        assert "UI" in categories


# =============================================================================
# @debuggable Decorator Tests
# =============================================================================


class TestDebuggableDecorator:
    """Tests for @debuggable decorator."""

    def test_basic_decoration(self):
        """@debuggable should mark class as debuggable."""
        @debuggable()
        class TestClass:
            name: str = ""

        assert is_debuggable(TestClass)
        assert hasattr(TestClass, "_debuggable")
        assert TestClass._debuggable is True

    def test_with_title(self):
        """@debuggable should accept title parameter."""
        @debuggable(title="My Component")
        class TestClass:
            pass

        config = DebuggableRegistry.get_instance().get_config(TestClass)
        assert config.title == "My Component"

    def test_with_category(self):
        """@debuggable should accept category parameter."""
        @debuggable(category="Physics")
        class TestClass:
            pass

        config = DebuggableRegistry.get_instance().get_config(TestClass)
        assert config.category == "Physics"

    def test_exclude_private(self):
        """@debuggable should respect exclude_private setting."""
        @debuggable(exclude_private=True)
        class TestClass:
            name: str = ""
            _private: int = 0

        registry = DebuggableRegistry.get_instance()
        fields = registry.get_inspectable_fields(TestClass)
        field_names = [f[0] for f in fields]
        assert "name" in field_names
        assert "_private" not in field_names

    def test_include_private(self):
        """@debuggable should include private when exclude_private=False."""
        @debuggable(exclude_private=False)
        class TestClass:
            name: str = ""
            _private: int = 0

        registry = DebuggableRegistry.get_instance()
        fields = registry.get_inspectable_fields(TestClass)
        field_names = [f[0] for f in fields]
        assert "name" in field_names
        assert "_private" in field_names

    def test_exclude_fields(self):
        """@debuggable should exclude specified fields."""
        @debuggable(exclude_fields={"hidden_field"})
        class TestClass:
            visible: str = ""
            hidden_field: int = 0

        registry = DebuggableRegistry.get_instance()
        fields = registry.get_inspectable_fields(TestClass)
        field_names = [f[0] for f in fields]
        assert "visible" in field_names
        assert "hidden_field" not in field_names

    def test_include_fields(self):
        """@debuggable should only include specified fields."""
        @debuggable(include_fields={"only_this"})
        class TestClass:
            only_this: str = ""
            not_this: int = 0

        registry = DebuggableRegistry.get_instance()
        fields = registry.get_inspectable_fields(TestClass)
        field_names = [f[0] for f in fields]
        assert "only_this" in field_names
        assert "not_this" not in field_names

    def test_with_sections(self):
        """@debuggable should accept sections parameter."""
        @debuggable(sections=[
            debug_section("stats", label="Statistics"),
            debug_section("config", label="Configuration"),
        ])
        class TestClass:
            pass

        config = DebuggableRegistry.get_instance().get_config(TestClass)
        assert len(config.sections) == 2
        assert config.sections[0].name == "stats"
        assert config.sections[1].name == "config"

    def test_with_dataclass(self):
        """@debuggable should work with dataclasses."""
        @debuggable(title="Player")
        @dataclass
        class Player:
            name: str = ""
            health: int = 100

        assert is_debuggable(Player)
        registry = DebuggableRegistry.get_instance()
        fields = registry.get_inspectable_fields(Player)
        field_names = [f[0] for f in fields]
        assert "name" in field_names
        assert "health" in field_names

    def test_with_dataclass_debug_field(self):
        """@debuggable should extract debug_field metadata from dataclass."""
        @debuggable()
        @dataclass
        class Player:
            health: int = field(
                default=100,
                metadata=debug_field(min_value=0, max_value=100, label="HP")
            )

        registry = DebuggableRegistry.get_instance()
        cfg = registry.get_field_config(Player, "health")
        assert cfg is not None
        assert cfg.min_value == 0
        assert cfg.max_value == 100
        assert cfg.label == "HP"


# =============================================================================
# DebuggablePanel Tests
# =============================================================================


class TestDebuggablePanel:
    """Tests for DebuggablePanel."""

    def test_panel_creation(self):
        """DebuggablePanel should be created from config."""
        @debuggable(title="Test Panel")
        @dataclass
        class TestClass:
            name: str = "test"
            value: int = 42

        obj = TestClass()
        registry = DebuggableRegistry.get_instance()
        config = registry.get_config(TestClass)
        field_configs = registry.get_field_configs(TestClass)
        panel = DebuggablePanel(obj, config, field_configs)

        assert panel.title == "Test Panel"
        assert panel.target is obj

    def test_panel_widgets_created(self):
        """DebuggablePanel should create widgets for fields."""
        @debuggable()
        @dataclass
        class TestClass:
            name: str = "test"
            count: int = 10

        obj = TestClass()
        registry = DebuggableRegistry.get_instance()
        panel = DebuggablePanel(
            obj,
            registry.get_config(TestClass),
            registry.get_field_configs(TestClass),
        )

        widgets = panel.widgets
        assert "name" in widgets
        assert "count" in widgets

    def test_panel_get_widget(self):
        """DebuggablePanel.get_widget should return correct widget."""
        @debuggable()
        @dataclass
        class TestClass:
            name: str = "test"

        obj = TestClass()
        registry = DebuggableRegistry.get_instance()
        panel = DebuggablePanel(
            obj,
            registry.get_config(TestClass),
            registry.get_field_configs(TestClass),
        )

        widget = panel.get_widget("name")
        assert widget is not None
        assert isinstance(widget, TextInputWidget)

    def test_panel_sync_from_target(self):
        """DebuggablePanel.sync_from_target should update widgets."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = 10

        obj = TestClass()
        registry = DebuggableRegistry.get_instance()
        panel = DebuggablePanel(
            obj,
            registry.get_config(TestClass),
            registry.get_field_configs(TestClass),
        )

        # Change object value
        obj.value = 50
        panel.sync_from_target()

        widget = panel.get_widget("value")
        assert widget.value == 50

    def test_panel_dirty_tracking(self):
        """DebuggablePanel should track dirty fields."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = 10

        obj = TestClass()
        registry = DebuggableRegistry.get_instance()
        panel = DebuggablePanel(
            obj,
            registry.get_config(TestClass),
            registry.get_field_configs(TestClass),
        )

        assert not panel.is_dirty()
        widget = panel.get_widget("value")
        widget.set_value(20)
        assert panel.is_dirty()
        assert "value" in panel.get_dirty_fields()

    def test_panel_clear_dirty(self):
        """DebuggablePanel.clear_dirty should reset dirty state."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = 10

        obj = TestClass()
        registry = DebuggableRegistry.get_instance()
        panel = DebuggablePanel(
            obj,
            registry.get_config(TestClass),
            registry.get_field_configs(TestClass),
        )

        widget = panel.get_widget("value")
        widget.set_value(20)
        assert panel.is_dirty()
        panel.clear_dirty()
        assert not panel.is_dirty()

    def test_panel_set_field_visible(self):
        """DebuggablePanel.set_field_visible should toggle visibility."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = 10

        obj = TestClass()
        registry = DebuggableRegistry.get_instance()
        panel = DebuggablePanel(
            obj,
            registry.get_config(TestClass),
            registry.get_field_configs(TestClass),
        )

        widget = panel.get_widget("value")
        assert widget.visible is True
        panel.set_field_visible("value", False)
        assert widget.visible is False

    def test_panel_set_field_enabled(self):
        """DebuggablePanel.set_field_enabled should toggle enabled."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = 10

        obj = TestClass()
        registry = DebuggableRegistry.get_instance()
        panel = DebuggablePanel(
            obj,
            registry.get_config(TestClass),
            registry.get_field_configs(TestClass),
        )

        widget = panel.get_widget("value")
        assert widget.enabled is True
        panel.set_field_enabled("value", False)
        assert widget.enabled is False


# =============================================================================
# Widget Creation Tests
# =============================================================================


class TestWidgetCreation:
    """Tests for automatic widget creation."""

    def test_creates_text_input_for_string(self):
        """Should create TextInputWidget for str fields."""
        @debuggable()
        @dataclass
        class TestClass:
            name: str = "test"

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("name")
        assert isinstance(widget, TextInputWidget)
        assert widget.value == "test"

    def test_creates_int_slider_with_range(self):
        """Should create IntSliderWidget for int with min/max."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = field(
                default=50,
                metadata=debug_field(min_value=0, max_value=100)
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("value")
        assert isinstance(widget, IntSliderWidget)
        assert widget.min_value == 0
        assert widget.max_value == 100

    def test_creates_int_input_without_range(self):
        """Should create IntInputWidget for int without min/max."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = 50

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("value")
        assert isinstance(widget, IntInputWidget)

    def test_creates_float_slider_with_range(self):
        """Should create FloatSliderWidget for float with min/max."""
        @debuggable()
        @dataclass
        class TestClass:
            ratio: float = field(
                default=0.5,
                metadata=debug_field(min_value=0.0, max_value=1.0)
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("ratio")
        assert isinstance(widget, FloatSliderWidget)

    def test_creates_checkbox_for_bool(self):
        """Should create CheckboxWidget for bool fields."""
        @debuggable()
        @dataclass
        class TestClass:
            enabled: bool = True

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("enabled")
        assert isinstance(widget, CheckboxWidget)
        assert widget.value is True

    def test_creates_dropdown_for_enum(self):
        """Should create DropdownWidget for Enum fields."""
        @debuggable()
        @dataclass
        class TestClass:
            mode: SampleEnum = SampleEnum.OPTION_A

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("mode")
        assert isinstance(widget, DropdownWidget)
        assert "OPTION_A" in widget.options
        assert "OPTION_B" in widget.options
        assert "OPTION_C" in widget.options

    def test_creates_dropdown_for_choices(self):
        """Should create DropdownWidget for fields with choices."""
        @debuggable()
        @dataclass
        class TestClass:
            quality: str = field(
                default="Medium",
                metadata=debug_field(choices=["Low", "Medium", "High"])
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("quality")
        assert isinstance(widget, DropdownWidget)
        assert widget.options == ["Low", "Medium", "High"]

    def test_creates_color_picker_for_color(self):
        """Should create ColorPickerWidget for Color fields."""
        @debuggable()
        @dataclass
        class TestClass:
            tint: Color = field(default_factory=lambda: Color(1, 0, 0, 1))

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("tint")
        assert isinstance(widget, ColorPickerWidget)

    def test_creates_vec2_input_for_vec2(self):
        """Should create Vec2InputWidget for Vec2 fields."""
        @debuggable()
        @dataclass
        class TestClass:
            position: Vec2 = field(default_factory=Vec2)

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("position")
        assert isinstance(widget, Vec2InputWidget)

    def test_creates_vec3_input_for_vec3(self):
        """Should create Vec3InputWidget for Vec3 fields."""
        @debuggable()
        @dataclass
        class TestClass:
            position: Vec3 = field(default_factory=Vec3)

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("position")
        assert isinstance(widget, Vec3InputWidget)

    def test_creates_vec4_input_for_vec4(self):
        """Should create Vec4InputWidget for Vec4 fields."""
        @debuggable()
        @dataclass
        class TestClass:
            rotation: Vec4 = field(default_factory=Vec4)

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("rotation")
        assert isinstance(widget, Vec4InputWidget)

    def test_widget_hint_forces_slider(self):
        """WidgetHint.SLIDER should force slider widget."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = field(
                default=50,
                metadata=debug_field(
                    min_value=0,
                    max_value=100,
                    widget=WidgetHint.SLIDER
                )
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("value")
        assert isinstance(widget, IntSliderWidget)

    def test_widget_hint_forces_input(self):
        """WidgetHint.INPUT should force input widget."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = field(
                default=50,
                metadata=debug_field(
                    min_value=0,
                    max_value=100,
                    widget=WidgetHint.INPUT
                )
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("value")
        assert isinstance(widget, IntInputWidget)

    def test_widget_hint_color(self):
        """WidgetHint.COLOR should create color picker."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = field(
                default=0,
                metadata=debug_field(widget=WidgetHint.COLOR)
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("value")
        assert isinstance(widget, ColorPickerWidget)


# =============================================================================
# Sections Tests
# =============================================================================


class TestSections:
    """Tests for field grouping into sections."""

    def test_fields_grouped_into_sections(self):
        """Fields should be grouped into specified sections."""
        @debuggable(sections=[
            debug_section("stats", label="Stats"),
            debug_section("config", label="Config"),
        ])
        @dataclass
        class TestClass:
            health: int = field(
                default=100,
                metadata=debug_field(section="stats")
            )
            mana: int = field(
                default=50,
                metadata=debug_field(section="stats")
            )
            difficulty: str = field(
                default="Normal",
                metadata=debug_field(section="config")
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        sections = panel.sections
        assert "stats" in sections
        assert "config" in sections

    def test_default_section(self):
        """Fields without section should go to default_section."""
        @debuggable(
            default_section="misc",
            sections=[debug_section("misc", label="Miscellaneous")]
        )
        @dataclass
        class TestClass:
            value: int = 10  # No section specified

        obj = TestClass()
        panel = get_debug_panel(obj)
        assert "misc" in panel.sections

    def test_expand_all_sections(self):
        """DebuggablePanel.expand_all should expand all sections."""
        @debuggable(sections=[
            debug_section("a", expanded=False),
            debug_section("b", expanded=False),
        ])
        @dataclass
        class TestClass:
            x: int = field(default=1, metadata=debug_field(section="a"))
            y: int = field(default=2, metadata=debug_field(section="b"))

        obj = TestClass()
        panel = get_debug_panel(obj)
        panel.expand_all()
        for section in panel.sections.values():
            assert section.expanded is True

    def test_collapse_all_sections(self):
        """DebuggablePanel.collapse_all should collapse all sections."""
        @debuggable(sections=[
            debug_section("a", expanded=True),
            debug_section("b", expanded=True),
        ])
        @dataclass
        class TestClass:
            x: int = field(default=1, metadata=debug_field(section="a"))
            y: int = field(default=2, metadata=debug_field(section="b"))

        obj = TestClass()
        panel = get_debug_panel(obj)
        panel.collapse_all()
        for section in panel.sections.values():
            assert section.expanded is False

    def test_set_section_expanded(self):
        """DebuggablePanel.set_section_expanded should toggle section."""
        @debuggable(sections=[debug_section("test", expanded=True)])
        @dataclass
        class TestClass:
            x: int = field(default=1, metadata=debug_field(section="test"))

        obj = TestClass()
        panel = get_debug_panel(obj)
        panel.set_section_expanded("test", False)
        assert panel.get_section("test").expanded is False


# =============================================================================
# Visibility Tests (show_if)
# =============================================================================


class TestShowIf:
    """Tests for dynamic field visibility."""

    def test_show_if_hides_field(self):
        """show_if returning False should hide field."""
        @debuggable()
        @dataclass
        class TestClass:
            enabled: bool = False
            value: int = field(
                default=10,
                metadata=debug_field(show_if=lambda obj: obj.enabled)
            )

        obj = TestClass(enabled=False)
        panel = get_debug_panel(obj)
        panel._update_visibility()
        widget = panel.get_widget("value")
        assert widget.visible is False

    def test_show_if_shows_field(self):
        """show_if returning True should show field."""
        @debuggable()
        @dataclass
        class TestClass:
            enabled: bool = True
            value: int = field(
                default=10,
                metadata=debug_field(show_if=lambda obj: obj.enabled)
            )

        obj = TestClass(enabled=True)
        panel = get_debug_panel(obj)
        panel._update_visibility()
        widget = panel.get_widget("value")
        assert widget.visible is True

    def test_show_if_updates_on_sync(self):
        """Visibility should update when syncing from target."""
        @debuggable()
        @dataclass
        class TestClass:
            enabled: bool = False
            value: int = field(
                default=10,
                metadata=debug_field(show_if=lambda obj: obj.enabled)
            )

        obj = TestClass(enabled=False)
        panel = get_debug_panel(obj)
        panel._update_visibility()
        widget = panel.get_widget("value")
        assert widget.visible is False

        # Enable and sync
        obj.enabled = True
        panel.sync_from_target()
        assert widget.visible is True


# =============================================================================
# Read-only Tests
# =============================================================================


class TestReadonly:
    """Tests for read-only fields."""

    def test_readonly_field_binding(self):
        """Read-only fields should have readonly binding."""
        @debuggable()
        @dataclass
        class TestClass:
            id: int = field(
                default=123,
                metadata=debug_field(readonly=True)
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        binding = panel._bindings.get("id")
        assert binding is not None
        assert binding.readonly is True


# =============================================================================
# DebuggableInspector Tests
# =============================================================================


class TestDebuggableInspector:
    """Tests for DebuggableInspector."""

    def test_inspect_debuggable_returns_panel(self):
        """inspect() should return DebuggablePanel for @debuggable."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = 10

        inspector = DebuggableInspector()
        obj = TestClass()
        panel = inspector.inspect(obj)
        assert isinstance(panel, DebuggablePanel)

    def test_inspect_regular_returns_property_panel(self):
        """inspect() should return PropertyPanel for non-debuggable."""
        class RegularClass:
            value: int = 10

        inspector = DebuggableInspector()
        obj = RegularClass()
        panel = inspector.inspect(obj)
        assert isinstance(panel, PropertyPanel)

    def test_inspect_with_title_override(self):
        """inspect() should accept title override."""
        @debuggable(title="Original")
        @dataclass
        class TestClass:
            value: int = 10

        inspector = DebuggableInspector()
        obj = TestClass()
        panel = inspector.inspect(obj, title="Override")
        assert panel.title == "Override"

    def test_panel_caching(self):
        """DebuggableInspector should cache panels."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = 10

        inspector = DebuggableInspector()
        obj = TestClass()
        panel1 = inspector.inspect(obj)
        panel2 = inspector.inspect(obj)
        assert panel1 is panel2

    def test_clear_cache(self):
        """DebuggableInspector.clear_cache should remove cached panels."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = 10

        inspector = DebuggableInspector()
        obj = TestClass()
        panel1 = inspector.inspect(obj)
        inspector.clear_cache()
        panel2 = inspector.inspect(obj)
        assert panel1 is not panel2


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_debuggable_true(self):
        """is_debuggable() should return True for decorated classes."""
        @debuggable()
        class TestClass:
            pass

        assert is_debuggable(TestClass) is True
        assert is_debuggable(TestClass()) is True

    def test_is_debuggable_false(self):
        """is_debuggable() should return False for regular classes."""
        class RegularClass:
            pass

        assert is_debuggable(RegularClass) is False

    def test_get_debug_panel_success(self):
        """get_debug_panel() should return panel for debuggable."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = 10

        obj = TestClass()
        panel = get_debug_panel(obj)
        assert panel is not None
        assert isinstance(panel, DebuggablePanel)

    def test_get_debug_panel_none(self):
        """get_debug_panel() should return None for non-debuggable."""
        class RegularClass:
            pass

        obj = RegularClass()
        panel = get_debug_panel(obj)
        assert panel is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with DebugUI."""

    def test_create_debug_ui_for(self, debug_ui):
        """create_debug_ui_for() should register panel with DebugUI."""
        @debuggable(title="Test")
        @dataclass
        class TestClass:
            value: int = 10

        obj = TestClass()
        panel = create_debug_ui_for(obj, debug_ui, "test_panel")
        assert panel is not None
        assert "test_panel" in debug_ui._panels

    def test_panel_render(self, debug_context):
        """DebuggablePanel should render without errors."""
        @debuggable()
        @dataclass
        class TestClass:
            name: str = "test"
            value: int = 10

        obj = TestClass()
        panel = get_debug_panel(obj)
        panel.render(debug_context)
        # Should not raise


# =============================================================================
# Field Order Tests
# =============================================================================


class TestFieldOrder:
    """Tests for field ordering."""

    def test_order_by_order_field(self):
        """Fields should be ordered by order parameter."""
        @debuggable()
        @dataclass
        class TestClass:
            z_field: int = field(default=3, metadata=debug_field(order=2))
            a_field: int = field(default=1, metadata=debug_field(order=0))
            m_field: int = field(default=2, metadata=debug_field(order=1))

        registry = DebuggableRegistry.get_instance()
        fields = registry.get_inspectable_fields(TestClass)
        field_names = [f[0] for f in fields]
        assert field_names == ["a_field", "m_field", "z_field"]

    def test_order_alphabetically_same_order(self):
        """Fields with same order should be sorted alphabetically."""
        @debuggable()
        @dataclass
        class TestClass:
            zebra: int = 1
            apple: int = 2
            mango: int = 3

        registry = DebuggableRegistry.get_instance()
        fields = registry.get_inspectable_fields(TestClass)
        field_names = [f[0] for f in fields]
        assert field_names == ["apple", "mango", "zebra"]


# =============================================================================
# Label Formatting Tests
# =============================================================================


class TestLabelFormatting:
    """Tests for automatic label formatting."""

    def test_snake_case_to_title(self):
        """Field names should be converted from snake_case to Title Case."""
        @debuggable()
        @dataclass
        class TestClass:
            player_health: int = 100

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("player_health")
        assert widget.label == "Player Health"

    def test_custom_label_override(self):
        """Custom label should override automatic formatting."""
        @debuggable()
        @dataclass
        class TestClass:
            player_health: int = field(
                default=100,
                metadata=debug_field(label="HP")
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("player_health")
        assert widget.label == "HP"


# =============================================================================
# WidgetHint Enum Tests
# =============================================================================


class TestWidgetHintEnum:
    """Tests for WidgetHint enum values."""

    def test_all_hints_exist(self):
        """All expected widget hints should exist."""
        assert WidgetHint.AUTO is not None
        assert WidgetHint.SLIDER is not None
        assert WidgetHint.INPUT is not None
        assert WidgetHint.TEXT is not None
        assert WidgetHint.TEXT_AREA is not None
        assert WidgetHint.CHECKBOX is not None
        assert WidgetHint.DROPDOWN is not None
        assert WidgetHint.COLOR is not None
        assert WidgetHint.VEC2 is not None
        assert WidgetHint.VEC3 is not None
        assert WidgetHint.VEC4 is not None
        assert WidgetHint.BUTTON is not None
        assert WidgetHint.LABEL is not None
        assert WidgetHint.CUSTOM is not None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_class(self):
        """@debuggable should work with classes having no fields."""
        @debuggable()
        class EmptyClass:
            pass

        obj = EmptyClass()
        panel = get_debug_panel(obj)
        assert panel is not None
        assert len(panel.widgets) == 0

    def test_none_initial_value(self):
        """Should handle None initial values."""
        @debuggable()
        @dataclass
        class TestClass:
            value: Optional[int] = None

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("value")
        assert widget is not None

    def test_weak_reference_target(self):
        """Panel should hold weak reference to target."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = 10

        obj = TestClass()
        panel = get_debug_panel(obj)
        assert panel.target is obj

        # Panel should not keep object alive (weak ref)
        # Note: This is hard to test definitively

    def test_show_if_exception_handling(self):
        """show_if exception should not crash panel."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = field(
                default=10,
                metadata=debug_field(show_if=lambda obj: obj.nonexistent)
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        # Should not raise
        panel._update_visibility()

    def test_on_change_exception_handling(self):
        """on_change exception should not crash panel."""
        def bad_callback(old, new):
            raise ValueError("Test error")

        @debuggable()
        @dataclass
        class TestClass:
            value: int = field(
                default=10,
                metadata=debug_field(on_change=bad_callback)
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("value")
        # Should not raise
        widget.set_value(20)


# =============================================================================
# Class-Level __debug_fields__ Tests
# =============================================================================


class TestClassLevelDebugFields:
    """Tests for __debug_fields__ class attribute."""

    def test_debug_fields_dict(self):
        """@debuggable should read __debug_fields__ dict."""
        @debuggable()
        class TestClass:
            value: int = 10
            __debug_fields__ = {
                "value": DebugFieldConfig(label="Custom Label", min_value=0)
            }

        registry = DebuggableRegistry.get_instance()
        cfg = registry.get_field_config(TestClass, "value")
        assert cfg is not None
        assert cfg.label == "Custom Label"
        assert cfg.min_value == 0


# =============================================================================
# Precision Tests
# =============================================================================


class TestPrecision:
    """Tests for float precision settings."""

    def test_custom_precision(self):
        """Custom precision should be passed to widget."""
        @debuggable()
        @dataclass
        class TestClass:
            value: float = field(
                default=0.5,
                metadata=debug_field(
                    min_value=0.0,
                    max_value=1.0,
                    precision=5
                )
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("value")
        assert isinstance(widget, (FloatSliderWidget, FloatInputWidget))
        assert widget.precision == 5


# =============================================================================
# Step Size Tests
# =============================================================================


class TestStepSize:
    """Tests for step size settings."""

    def test_int_step_size(self):
        """Integer step size should be applied."""
        @debuggable()
        @dataclass
        class TestClass:
            value: int = field(
                default=50,
                metadata=debug_field(min_value=0, max_value=100, step=10)
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("value")
        assert isinstance(widget, IntSliderWidget)
        assert widget.step == 10

    def test_float_step_size(self):
        """Float step size should be applied."""
        @debuggable()
        @dataclass
        class TestClass:
            value: float = field(
                default=0.5,
                metadata=debug_field(min_value=0.0, max_value=1.0, step=0.05)
            )

        obj = TestClass()
        panel = get_debug_panel(obj)
        widget = panel.get_widget("value")
        assert isinstance(widget, FloatSliderWidget)
        assert widget.step == 0.05
