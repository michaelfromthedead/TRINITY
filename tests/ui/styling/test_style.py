"""
Comprehensive tests for the Style class and related components.

Tests cover:
- Style creation and validation
- Style property descriptors
- Style inheritance and merging
- Visual states and StateStyles
- Style selectors
- Stylesheets
- StyleBuilder
"""
import pytest

from engine.ui.styling.brush import SolidBrush
from engine.ui.styling.color import Color
from engine.ui.styling.style import (
    SelectorType,
    StateStyles,
    Style,
    StyleBuilder,
    StylePropertyDescriptor,
    StyleRule,
    StyleSelector,
    Stylesheet,
    VisualState,
    style_property,
)


# ========== Fixtures ==========


@pytest.fixture
def red_color():
    """Pure red color."""
    return Color(1.0, 0.0, 0.0)


@pytest.fixture
def blue_color():
    """Pure blue color."""
    return Color(0.0, 0.0, 1.0)


@pytest.fixture
def default_style():
    """Empty default style."""
    return Style()


@pytest.fixture
def styled_style(red_color):
    """Style with some properties set."""
    return Style(
        background_color=red_color,
        border_width=2.0,
        border_radius=8.0,
        opacity=0.9,
        font_size=14.0,
        padding_left=10.0,
        padding_right=10.0,
        padding_top=5.0,
        padding_bottom=5.0,
    )


@pytest.fixture
def state_styles(red_color, blue_color):
    """StateStyles with multiple states."""
    return StateStyles(
        normal=Style(background_color=red_color),
        hovered=Style(background_color=blue_color),
        pressed=Style(opacity=0.8),
        disabled=Style(opacity=0.5),
    )


# ========== Style Creation Tests ==========


class TestStyleCreation:
    """Tests for creating Style instances."""

    def test_create_default_style(self):
        """Test creating style with defaults."""
        style = Style()
        assert style.background is None
        assert style.border_width == 0.0
        assert style.opacity == 1.0

    def test_create_style_with_values(self, red_color):
        """Test creating style with explicit values."""
        style = Style(
            background_color=red_color,
            border_width=1.0,
            border_radius=4.0,
        )
        assert style.background_color == red_color
        assert style.border_width == 1.0
        assert style.border_radius == 4.0

    def test_invalid_opacity_low(self):
        """Test validation rejects opacity below 0."""
        with pytest.raises(ValueError, match="Opacity must be in"):
            Style(opacity=-0.1)

    def test_invalid_opacity_high(self):
        """Test validation rejects opacity above 1."""
        with pytest.raises(ValueError, match="Opacity must be in"):
            Style(opacity=1.5)

    def test_invalid_border_width_negative(self):
        """Test validation rejects negative border width."""
        with pytest.raises(ValueError, match="Border width must be non-negative"):
            Style(border_width=-1.0)

    def test_invalid_border_radius_negative(self):
        """Test validation rejects negative border radius."""
        with pytest.raises(ValueError, match="Border radius must be non-negative"):
            Style(border_radius=-1.0)


# ========== Padding/Margin Shortcut Tests ==========


class TestPaddingMarginShortcuts:
    """Tests for padding and margin shortcut properties."""

    def test_get_padding_tuple(self, styled_style):
        """Test getting padding as tuple."""
        padding = styled_style.padding
        assert padding == (5.0, 10.0, 5.0, 10.0)

    def test_set_padding_single_value(self):
        """Test setting padding with single value."""
        style = Style()
        style.padding = 10.0
        assert style.padding_top == 10.0
        assert style.padding_right == 10.0
        assert style.padding_bottom == 10.0
        assert style.padding_left == 10.0

    def test_set_padding_two_values(self):
        """Test setting padding with two values (vertical, horizontal)."""
        style = Style()
        style.padding = (10.0, 20.0)
        assert style.padding_top == 10.0
        assert style.padding_bottom == 10.0
        assert style.padding_left == 20.0
        assert style.padding_right == 20.0

    def test_set_padding_four_values(self):
        """Test setting padding with four values."""
        style = Style()
        style.padding = (1.0, 2.0, 3.0, 4.0)
        assert style.padding_top == 1.0
        assert style.padding_right == 2.0
        assert style.padding_bottom == 3.0
        assert style.padding_left == 4.0

    def test_set_padding_invalid_count(self):
        """Test setting padding with invalid value count."""
        style = Style()
        with pytest.raises(ValueError, match="Padding must be 1, 2, or 4 values"):
            style.padding = (1.0, 2.0, 3.0)

    def test_get_margin_tuple(self):
        """Test getting margin as tuple."""
        style = Style(margin_top=1.0, margin_right=2.0, margin_bottom=3.0, margin_left=4.0)
        assert style.margin == (1.0, 2.0, 3.0, 4.0)

    def test_set_margin_single_value(self):
        """Test setting margin with single value."""
        style = Style()
        style.margin = 15.0
        assert style.margin_top == 15.0
        assert style.margin_right == 15.0
        assert style.margin_bottom == 15.0
        assert style.margin_left == 15.0


# ========== Border Radius Tests ==========


class TestBorderRadius:
    """Tests for border radius shortcuts."""

    def test_get_border_radii_uniform(self):
        """Test getting uniform border radii."""
        style = Style(border_radius=8.0)
        radii = style.get_border_radii()
        assert radii == (8.0, 8.0, 8.0, 8.0)

    def test_get_border_radii_individual(self):
        """Test getting individual border radii."""
        style = Style(
            border_radius=8.0,
            border_radius_top_left=10.0,
            border_radius_bottom_right=5.0,
        )
        radii = style.get_border_radii()
        assert radii == (10.0, 8.0, 5.0, 8.0)


# ========== Style Inheritance and Merging Tests ==========


class TestStyleInheritance:
    """Tests for style inheritance and merging."""

    def test_merge_styles(self, red_color, blue_color):
        """Test merging two styles."""
        style1 = Style(background_color=red_color, border_width=2.0)
        style2 = Style(background_color=blue_color, font_size=16.0)

        merged = style1.merge(style2)

        # style1 values take precedence
        assert merged.background_color == red_color
        assert merged.border_width == 2.0
        # style2 values fill in missing
        assert merged.font_size == 16.0

    def test_inherit_from_parent(self, red_color, blue_color):
        """Test inherit_from is same as merge."""
        parent = Style(background_color=red_color, font_size=14.0)
        child = Style(background_color=blue_color)

        inherited = child.inherit_from(parent)

        assert inherited.background_color == blue_color  # Child wins
        assert inherited.font_size == 14.0  # Parent fills in

    def test_merge_preserves_none(self, red_color):
        """Test that None values don't override."""
        style1 = Style(background_color=red_color)
        style2 = Style()

        merged = style1.merge(style2)
        assert merged.background_color == red_color


# ========== Style Cloning Tests ==========


class TestStyleCloning:
    """Tests for style cloning."""

    def test_clone_basic(self, styled_style):
        """Test cloning creates independent copy."""
        cloned = styled_style.clone()
        assert cloned is not styled_style
        assert cloned.background_color == styled_style.background_color
        assert cloned.border_width == styled_style.border_width

    def test_clone_with_brush(self, red_color):
        """Test cloning includes brush clone."""
        brush = SolidBrush(red_color)
        style = Style(background=brush)
        cloned = style.clone()

        assert cloned.background is not brush
        assert cloned.background.color == red_color


# ========== VisualState Tests ==========


class TestVisualState:
    """Tests for VisualState enum."""

    def test_all_states_exist(self):
        """Test all visual states are defined."""
        assert VisualState.NORMAL
        assert VisualState.HOVERED
        assert VisualState.PRESSED
        assert VisualState.FOCUSED
        assert VisualState.DISABLED
        assert VisualState.SELECTED


# ========== StateStyles Tests ==========


class TestStateStyles:
    """Tests for StateStyles class."""

    def test_create_default(self):
        """Test creating default StateStyles."""
        styles = StateStyles()
        assert styles.normal is not None
        assert styles.hovered is None

    def test_get_style_normal(self, state_styles):
        """Test getting normal state style."""
        style = state_styles.get_style(VisualState.NORMAL)
        assert style.background_color.r == 1.0  # Red

    def test_get_style_hovered(self, state_styles):
        """Test getting hovered state style (merged with normal)."""
        style = state_styles.get_style(VisualState.HOVERED)
        # Hovered has blue, which takes precedence
        assert style.background_color.b == 1.0

    def test_get_style_undefined_falls_back(self, state_styles):
        """Test undefined state falls back to normal."""
        style = state_styles.get_style(VisualState.SELECTED)
        assert style.background_color.r == 1.0  # Falls back to normal's red

    def test_set_style(self, state_styles, blue_color):
        """Test setting a state style."""
        new_style = Style(background_color=blue_color)
        state_styles.set_style(VisualState.FOCUSED, new_style)
        assert state_styles.focused is not None
        assert state_styles.focused.background_color == blue_color

    def test_get_computed_style_single_state(self, state_styles):
        """Test computed style with single active state."""
        style = state_styles.get_computed_style({VisualState.HOVERED})
        assert style.background_color.b == 1.0

    def test_get_computed_style_multiple_states(self, state_styles):
        """Test computed style with multiple active states."""
        style = state_styles.get_computed_style({VisualState.HOVERED, VisualState.PRESSED})
        # Pressed has higher precedence, adds opacity
        assert style.opacity == 0.8

    def test_get_computed_style_disabled_precedence(self, state_styles):
        """Test disabled has highest precedence."""
        style = state_styles.get_computed_style({VisualState.HOVERED, VisualState.DISABLED})
        assert style.opacity == 0.5

    def test_clone_state_styles(self, state_styles):
        """Test cloning StateStyles."""
        cloned = state_styles.clone()
        assert cloned is not state_styles
        assert cloned.normal is not state_styles.normal
        assert cloned.hovered is not state_styles.hovered


# ========== StyleSelector Tests ==========


class TestStyleSelector:
    """Tests for StyleSelector class."""

    def test_by_type(self):
        """Test creating type selector."""
        selector = StyleSelector.by_type(str)
        assert selector.selector_type == SelectorType.TYPE
        assert selector.value == str

    def test_by_name(self):
        """Test creating name selector."""
        selector = StyleSelector.by_name("myWidget")
        assert selector.selector_type == SelectorType.NAME
        assert selector.value == "myWidget"

    def test_by_class(self):
        """Test creating class selector."""
        selector = StyleSelector.by_class("highlight")
        assert selector.selector_type == SelectorType.CLASS
        assert selector.value == "highlight"

    def test_by_state(self):
        """Test creating state selector."""
        selector = StyleSelector.by_state(VisualState.HOVERED)
        assert selector.selector_type == SelectorType.STATE
        assert selector.value == VisualState.HOVERED

    def test_by_id(self):
        """Test creating ID selector."""
        selector = StyleSelector.by_id("widget-123")
        assert selector.selector_type == SelectorType.ID
        assert selector.value == "widget-123"

    def test_universal(self):
        """Test creating universal selector."""
        selector = StyleSelector.universal()
        assert selector.selector_type == SelectorType.UNIVERSAL

    def test_with_state(self):
        """Test adding pseudo-state conditions."""
        selector = StyleSelector.by_class("button").with_state(VisualState.HOVERED)
        assert len(selector.pseudo_states) == 1
        assert VisualState.HOVERED in selector.pseudo_states

    def test_matches_universal(self):
        """Test universal selector matches everything."""
        selector = StyleSelector.universal()
        assert selector.matches(widget_type=str, widget_name="test")

    def test_matches_type(self):
        """Test type selector matching."""
        selector = StyleSelector.by_type(str)
        assert selector.matches(widget_type=str)
        assert not selector.matches(widget_type=int)

    def test_matches_type_subclass(self):
        """Test type selector matches subclasses."""
        selector = StyleSelector.by_type(Exception)
        assert selector.matches(widget_type=ValueError)

    def test_matches_name(self):
        """Test name selector matching."""
        selector = StyleSelector.by_name("myWidget")
        assert selector.matches(widget_name="myWidget")
        assert not selector.matches(widget_name="other")

    def test_matches_class(self):
        """Test class selector matching."""
        selector = StyleSelector.by_class("highlight")
        assert selector.matches(style_classes={"highlight", "active"})
        assert not selector.matches(style_classes={"active"})

    def test_matches_state(self):
        """Test state selector matching."""
        selector = StyleSelector.by_state(VisualState.HOVERED)
        assert selector.matches(active_states={VisualState.HOVERED})
        assert not selector.matches(active_states={VisualState.PRESSED})

    def test_matches_id(self):
        """Test ID selector matching."""
        selector = StyleSelector.by_id("widget-123")
        assert selector.matches(widget_id="widget-123")
        assert not selector.matches(widget_id="widget-456")

    def test_matches_with_pseudo_state(self):
        """Test selector with pseudo-state conditions."""
        selector = StyleSelector.by_class("button").with_state(VisualState.HOVERED)
        assert selector.matches(
            style_classes={"button"},
            active_states={VisualState.HOVERED}
        )
        assert not selector.matches(
            style_classes={"button"},
            active_states={VisualState.PRESSED}
        )

    def test_specificity_id(self):
        """Test ID selector has highest specificity."""
        selector = StyleSelector.by_id("widget")
        assert selector.specificity == (1, 0, 0)

    def test_specificity_class(self):
        """Test class selector specificity."""
        selector = StyleSelector.by_class("highlight")
        assert selector.specificity == (0, 1, 0)

    def test_specificity_type(self):
        """Test type selector specificity."""
        selector = StyleSelector.by_type(str)
        assert selector.specificity == (0, 0, 1)

    def test_specificity_with_pseudo_states(self):
        """Test pseudo-states increase specificity."""
        selector = StyleSelector.by_class("button").with_state(VisualState.HOVERED, VisualState.FOCUSED)
        # 1 class + 2 pseudo-states = 3 class points
        assert selector.specificity == (0, 3, 0)


# ========== StyleRule Tests ==========


class TestStyleRule:
    """Tests for StyleRule class."""

    def test_create_rule(self, styled_style):
        """Test creating a style rule."""
        selector = StyleSelector.by_class("button")
        rule = StyleRule(selector=selector, style=styled_style)
        assert rule.selector == selector
        assert rule.style == styled_style
        assert rule.priority == 0

    def test_rule_specificity(self, styled_style):
        """Test rule specificity includes priority."""
        selector = StyleSelector.by_class("button")
        rule = StyleRule(selector=selector, style=styled_style, priority=1)
        # (priority, id, class, type)
        assert rule.specificity == (1, 0, 1, 0)


# ========== Stylesheet Tests ==========


class TestStylesheet:
    """Tests for Stylesheet class."""

    def test_add_rule(self, styled_style):
        """Test adding a rule to stylesheet."""
        stylesheet = Stylesheet()
        selector = StyleSelector.by_class("button")
        rule = StyleRule(selector=selector, style=styled_style)

        stylesheet.add_rule(rule)
        assert len(stylesheet) == 1

    def test_add_style(self, styled_style):
        """Test add_style convenience method."""
        stylesheet = Stylesheet()
        selector = StyleSelector.by_class("button")

        stylesheet.add_style(selector, styled_style)
        assert len(stylesheet) == 1

    def test_remove_rules_for_selector(self, styled_style):
        """Test removing rules by selector."""
        stylesheet = Stylesheet()
        selector = StyleSelector.by_class("button")
        stylesheet.add_style(selector, styled_style)
        stylesheet.add_style(selector, styled_style)

        removed = stylesheet.remove_rules_for_selector(selector)
        assert removed == 2
        assert len(stylesheet) == 0

    def test_get_computed_style_no_match(self):
        """Test computed style with no matching rules."""
        stylesheet = Stylesheet()
        style = stylesheet.get_computed_style(widget_type=str)
        assert style.background is None

    def test_get_computed_style_with_match(self, red_color):
        """Test computed style with matching rule."""
        stylesheet = Stylesheet()
        selector = StyleSelector.by_class("highlight")
        rule_style = Style(background_color=red_color)
        stylesheet.add_style(selector, rule_style)

        style = stylesheet.get_computed_style(style_classes={"highlight"})
        assert style.background_color == red_color

    def test_get_computed_style_with_base(self, red_color, blue_color):
        """Test computed style with base style."""
        stylesheet = Stylesheet()
        selector = StyleSelector.by_class("highlight")
        rule_style = Style(background_color=red_color)
        stylesheet.add_style(selector, rule_style)

        base_style = Style(border_width=2.0)
        style = stylesheet.get_computed_style(
            style_classes={"highlight"},
            base_style=base_style
        )
        assert style.background_color == red_color
        assert style.border_width == 2.0

    def test_rules_sorted_by_specificity(self, red_color, blue_color):
        """Test rules are sorted by specificity."""
        stylesheet = Stylesheet()

        # Add rules in reverse specificity order
        type_style = Style(background_color=red_color)
        id_style = Style(background_color=blue_color)

        stylesheet.add_style(StyleSelector.by_type(str), type_style)
        stylesheet.add_style(StyleSelector.by_id("widget"), id_style)

        # ID has higher specificity, should be applied last
        style = stylesheet.get_computed_style(
            widget_type=str,
            widget_id="widget"
        )
        assert style.background_color == blue_color

    def test_clear_stylesheet(self, styled_style):
        """Test clearing all rules."""
        stylesheet = Stylesheet()
        stylesheet.add_style(StyleSelector.by_class("a"), styled_style)
        stylesheet.add_style(StyleSelector.by_class("b"), styled_style)

        stylesheet.clear()
        assert len(stylesheet) == 0

    def test_iterate_rules(self, styled_style):
        """Test iterating over rules."""
        stylesheet = Stylesheet()
        stylesheet.add_style(StyleSelector.by_class("a"), styled_style)
        stylesheet.add_style(StyleSelector.by_class("b"), styled_style)

        rules = list(stylesheet)
        assert len(rules) == 2


# ========== StyleBuilder Tests ==========


class TestStyleBuilder:
    """Tests for StyleBuilder fluent API."""

    def test_builder_chain(self):
        """Test builder method chaining."""
        style = (
            StyleBuilder()
            .background_color("#FF0000")
            .border(width=2.0, color="#000000", radius=4.0)
            .padding(10, 20)
            .build()
        )
        assert style.background_color.r == 1.0
        assert style.border_width == 2.0
        assert style.border_radius == 4.0
        assert style.padding_top == 10.0

    def test_builder_background_with_brush(self, red_color):
        """Test setting background with brush."""
        brush = SolidBrush(red_color)
        style = StyleBuilder().background(brush).build()
        assert style.background is not None

    def test_builder_background_with_color(self, red_color):
        """Test setting background with color."""
        style = StyleBuilder().background(red_color).build()
        assert style.background is not None

    def test_builder_background_with_string(self):
        """Test setting background with string."""
        style = StyleBuilder().background("#FF0000").build()
        assert style.background is not None

    def test_builder_border_radius_uniform(self):
        """Test setting uniform border radius."""
        style = StyleBuilder().border_radius(8.0).build()
        assert style.border_radius == 8.0

    def test_builder_border_radius_individual(self):
        """Test setting individual corner radii."""
        style = StyleBuilder().border_radius((1.0, 2.0, 3.0, 4.0)).build()
        assert style.border_radius_top_left == 1.0
        assert style.border_radius_top_right == 2.0
        assert style.border_radius_bottom_right == 3.0
        assert style.border_radius_bottom_left == 4.0

    def test_builder_foreground(self):
        """Test setting foreground color."""
        style = StyleBuilder().foreground("#FFFFFF").build()
        assert style.foreground_color.r == 1.0
        assert style.foreground_color.g == 1.0
        assert style.foreground_color.b == 1.0

    def test_builder_font(self):
        """Test setting font properties."""
        style = (
            StyleBuilder()
            .font(family="Arial", size=16.0, weight="bold", style="italic")
            .build()
        )
        assert style.font_family == "Arial"
        assert style.font_size == 16.0
        assert style.font_weight == "bold"
        assert style.font_style == "italic"

    def test_builder_text(self):
        """Test setting text properties."""
        style = (
            StyleBuilder()
            .text(align="center", line_height=1.5, letter_spacing=0.5)
            .build()
        )
        assert style.text_align == "center"
        assert style.line_height == 1.5
        assert style.letter_spacing == 0.5

    def test_builder_opacity(self):
        """Test setting opacity."""
        style = StyleBuilder().opacity(0.5).build()
        assert style.opacity == 0.5

    def test_builder_padding_one_value(self):
        """Test setting padding with one value."""
        style = StyleBuilder().padding(10).build()
        assert style.padding == (10.0, 10.0, 10.0, 10.0)

    def test_builder_padding_two_values(self):
        """Test setting padding with two values."""
        style = StyleBuilder().padding(10, 20).build()
        assert style.padding == (10.0, 20.0, 10.0, 20.0)

    def test_builder_padding_four_values(self):
        """Test setting padding with four values."""
        style = StyleBuilder().padding(1, 2, 3, 4).build()
        assert style.padding == (1.0, 2.0, 3.0, 4.0)

    def test_builder_padding_invalid_count(self):
        """Test setting padding with invalid value count."""
        with pytest.raises(ValueError, match="padding accepts 1, 2, or 4"):
            StyleBuilder().padding(1, 2, 3)

    def test_builder_margin(self):
        """Test setting margin."""
        style = StyleBuilder().margin(5).build()
        assert style.margin == (5.0, 5.0, 5.0, 5.0)

    def test_builder_shadow(self):
        """Test setting shadow properties."""
        style = (
            StyleBuilder()
            .shadow(color="#000000", offset_x=2, offset_y=4, blur=8, spread=1)
            .build()
        )
        assert style.shadow_color is not None
        assert style.shadow_offset_x == 2
        assert style.shadow_offset_y == 4
        assert style.shadow_blur == 8
        assert style.shadow_spread == 1

    def test_builder_transform(self):
        """Test setting transform properties."""
        style = (
            StyleBuilder()
            .transform(scale_x=1.5, scale_y=0.8, rotation=45, translate_x=10, translate_y=20)
            .build()
        )
        assert style.scale_x == 1.5
        assert style.scale_y == 0.8
        assert style.rotation == 45
        assert style.translate_x == 10
        assert style.translate_y == 20

    def test_builder_cursor(self):
        """Test setting cursor."""
        style = StyleBuilder().cursor("pointer").build()
        assert style.cursor == "pointer"

    def test_builder_transition(self):
        """Test setting transition properties."""
        style = StyleBuilder().transition(duration=0.3, easing="ease-in").build()
        assert style.transition_duration == 0.3
        assert style.transition_easing == "ease-in"

    def test_builder_with_base_style(self, styled_style):
        """Test builder starting from base style."""
        style = StyleBuilder(styled_style).opacity(0.5).build()
        assert style.opacity == 0.5
        # Other properties preserved
        assert style.border_width == 2.0

    def test_builder_build_returns_clone(self):
        """Test build returns independent style."""
        builder = StyleBuilder().opacity(0.5)
        style1 = builder.build()
        style2 = builder.build()
        assert style1 is not style2


# ========== StylePropertyDescriptor Tests ==========


class TestStylePropertyDescriptor:
    """Tests for StylePropertyDescriptor."""

    def test_descriptor_with_default(self):
        """Test descriptor returns default value."""

        class TestClass:
            prop = style_property(float, default=10.0)

        obj = TestClass()
        assert obj.prop == 10.0

    def test_descriptor_set_value(self):
        """Test setting descriptor value."""

        class TestClass:
            prop = style_property(float, default=0.0)

        obj = TestClass()
        obj.prop = 5.0
        assert obj.prop == 5.0

    def test_descriptor_type_validation(self):
        """Test descriptor validates type."""

        class TestClass:
            prop = style_property(float, default=0.0)

        obj = TestClass()
        with pytest.raises(TypeError, match="expects float"):
            obj.prop = "not a float"

    def test_descriptor_min_value(self):
        """Test descriptor validates minimum value."""

        class TestClass:
            prop = style_property(float, default=0.0, min_value=0.0)

        obj = TestClass()
        with pytest.raises(ValueError, match="must be >="):
            obj.prop = -1.0

    def test_descriptor_max_value(self):
        """Test descriptor validates maximum value."""

        class TestClass:
            prop = style_property(float, default=0.0, max_value=1.0)

        obj = TestClass()
        with pytest.raises(ValueError, match="must be <="):
            obj.prop = 2.0

    def test_descriptor_choices(self):
        """Test descriptor validates choices."""

        class TestClass:
            prop = style_property(str, default="a", choices={"a", "b", "c"})

        obj = TestClass()
        obj.prop = "b"
        assert obj.prop == "b"

        with pytest.raises(ValueError, match="must be one of"):
            obj.prop = "d"

    def test_descriptor_custom_validator(self):
        """Test descriptor with custom validator."""

        class TestClass:
            prop = style_property(str, default="", validator=lambda x: len(x) > 0)

        obj = TestClass()
        with pytest.raises(ValueError, match="failed validation"):
            obj.prop = ""

    def test_descriptor_converter(self):
        """Test descriptor with value converter."""

        class TestClass:
            prop = style_property(float, default=0.0, converter=float)

        obj = TestClass()
        obj.prop = "5.5"
        assert obj.prop == 5.5

    def test_descriptor_set_none(self):
        """Test setting None value."""

        class TestClass:
            prop = style_property(float, default=0.0)

        obj = TestClass()
        obj.prop = 10.0
        obj.prop = None
        assert obj.prop is None

    def test_descriptor_class_access(self):
        """Test accessing descriptor on class returns descriptor."""

        class TestClass:
            prop = style_property(float, default=0.0)

        assert isinstance(TestClass.prop, StylePropertyDescriptor)
