"""
Comprehensive tests for Responsive layout module.

Tests cover:
- Breakpoint enum values and its three members (MOBILE, TABLET, DESKTOP)
- Default breakpoint thresholds (0, 600, 1024)
- BreakpointManager initialization, property access, and error validation
- T-1.11: Breakpoint matching (_calculate_breakpoint) across all thresholds
- BreakpointManager orientation detection and callbacks
- BreakpointManager get_value, get_columns, get_spacing helpers
- T-1.12: ResponsiveValue creation, get() with cascading fallback
- T-1.13: SafeAreaInsets creation, validation, with_* helpers, uniform, symmetric
- ResponsiveRule creation and defaults
- ResponsiveContainer initialization, rule management, visibility rules
- ResponsiveContainer calculate_layout delegation
- Helper functions (responsive_spacing, responsive_font_size, hide_on_mobile,
  show_only_on_mobile, hide_on_desktop)
"""

import pytest
from typing import Any, Optional

from engine.ui.layout.responsive import (
    Breakpoint,
    Orientation,
    Visibility,
    SafeAreaInsets,
    ResponsiveValue,
    ResponsiveRule,
    BreakpointManager,
    ResponsiveContainer,
    DEFAULT_BREAKPOINTS,
    BREAKPOINT_MOBILE_MIN,
    BREAKPOINT_TABLET_MIN,
    BREAKPOINT_DESKTOP_MIN,
    responsive_spacing,
    responsive_font_size,
    hide_on_mobile,
    show_only_on_mobile,
    hide_on_desktop,
)
from engine.ui.layout.canvas import Rect


# ============================================================
# T-1.11: Verify Breakpoint Matching
# ============================================================

class TestBreakpointEnum:
    """Tests for Breakpoint enum."""

    def test_breakpoint_has_three_members(self):
        """Test breakpoint enum has exactly MOBILE, TABLET, DESKTOP."""
        assert len(Breakpoint) == 3

    def test_breakpoint_mobile_value(self):
        """Test MOBILE is the first member."""
        assert Breakpoint.MOBILE.value == 1

    def test_breakpoint_tablet_value(self):
        """Test TABLET is the second member."""
        assert Breakpoint.TABLET.value == 2

    def test_breakpoint_desktop_value(self):
        """Test DESKTOP is the third member."""
        assert Breakpoint.DESKTOP.value == 3


class TestBreakpointDefaults:
    """Tests for default breakpoint thresholds."""

    def test_mobile_min_threshold(self):
        """Test mobile starts at 0px."""
        assert BREAKPOINT_MOBILE_MIN == 0

    def test_tablet_min_threshold(self):
        """Test tablet starts at 600px."""
        assert BREAKPOINT_TABLET_MIN == 600

    def test_desktop_min_threshold(self):
        """Test desktop starts at 1024px."""
        assert BREAKPOINT_DESKTOP_MIN == 1024

    def test_default_breakpoints_dict_contains_all(self):
        """Test DEFAULT_BREAKPOINTS maps each Breakpoint to its threshold."""
        assert Breakpoint.MOBILE in DEFAULT_BREAKPOINTS
        assert Breakpoint.TABLET in DEFAULT_BREAKPOINTS
        assert Breakpoint.DESKTOP in DEFAULT_BREAKPOINTS
        assert DEFAULT_BREAKPOINTS[Breakpoint.MOBILE] == 0
        assert DEFAULT_BREAKPOINTS[Breakpoint.TABLET] == 600
        assert DEFAULT_BREAKPOINTS[Breakpoint.DESKTOP] == 1024


class TestBreakpointManagerInit:
    """Tests for BreakpointManager initialization."""

    def test_default_initialization(self):
        """Test manager with no args defaults to 0x0 and MOBILE."""
        mgr = BreakpointManager()
        assert mgr.width == 0
        assert mgr.height == 0
        assert isinstance(mgr.safe_area, SafeAreaInsets)
        assert mgr.breakpoint == Breakpoint.MOBILE

    def test_custom_dimensions(self):
        """Test manager stores custom width and height."""
        mgr = BreakpointManager(width=800, height=600)
        assert mgr.width == 800
        assert mgr.height == 600

    def test_desktop_width_on_init(self):
        """Test manager at desktop width starts as DESKTOP."""
        mgr = BreakpointManager(width=1200, height=800)
        assert mgr.breakpoint == Breakpoint.DESKTOP

    def test_tablet_width_on_init(self):
        """Test manager at tablet width starts as TABLET."""
        mgr = BreakpointManager(width=768, height=1024)
        assert mgr.breakpoint == Breakpoint.TABLET

    def test_negative_width_raises_value_error(self):
        """Test negative width is rejected."""
        with pytest.raises(ValueError, match="Width cannot be negative"):
            BreakpointManager(width=-100, height=600)

    def test_negative_height_raises_value_error(self):
        """Test negative height is rejected."""
        with pytest.raises(ValueError, match="Height cannot be negative"):
            BreakpointManager(width=800, height=-100)

    def test_custom_safe_area(self):
        """Test manager stores provided safe area."""
        safe = SafeAreaInsets(top=44, bottom=34)
        mgr = BreakpointManager(width=800, height=600, safe_area=safe)
        assert mgr.safe_area.top == 44
        assert mgr.safe_area.bottom == 34

    def test_custom_breakpoints_dict(self):
        """Test custom breakpoint thresholds change matching."""
        custom = {
            Breakpoint.MOBILE: 0,
            Breakpoint.TABLET: 500,
            Breakpoint.DESKTOP: 900,
        }
        mgr = BreakpointManager(width=700, height=600, breakpoints=custom)
        assert mgr.breakpoint == Breakpoint.TABLET

        mgr2 = BreakpointManager(width=1000, height=600, breakpoints=custom)
        assert mgr2.breakpoint == Breakpoint.DESKTOP


class TestBreakpointMatching:
    """T-1.11: Verify breakpoint matching (_calculate_breakpoint)."""

    def test_width_zero_returns_mobile(self):
        """Width 0 (absolute minimum) returns MOBILE."""
        mgr = BreakpointManager(width=0, height=600)
        assert mgr.breakpoint == Breakpoint.MOBILE

    def test_width_599_returns_mobile(self):
        """Width 599 (just below tablet threshold) returns MOBILE."""
        mgr = BreakpointManager(width=599, height=600)
        assert mgr.breakpoint == Breakpoint.MOBILE

    def test_width_600_returns_tablet(self):
        """Width 600 (exact tablet threshold) returns TABLET."""
        mgr = BreakpointManager(width=600, height=600)
        assert mgr.breakpoint == Breakpoint.TABLET

    def test_width_767_returns_tablet(self):
        """Width 767 (mid-tablet) returns TABLET."""
        mgr = BreakpointManager(width=767, height=600)
        assert mgr.breakpoint == Breakpoint.TABLET

    def test_width_1023_returns_tablet(self):
        """Width 1023 (just below desktop threshold) returns TABLET."""
        mgr = BreakpointManager(width=1023, height=600)
        assert mgr.breakpoint == Breakpoint.TABLET

    def test_width_1024_returns_desktop(self):
        """Width 1024 (exact desktop threshold) returns DESKTOP."""
        mgr = BreakpointManager(width=1024, height=600)
        assert mgr.breakpoint == Breakpoint.DESKTOP

    def test_width_1920_returns_desktop(self):
        """Width 1920 (full HD) returns DESKTOP."""
        mgr = BreakpointManager(width=1920, height=600)
        assert mgr.breakpoint == Breakpoint.DESKTOP


class TestBreakpointManagerProperties:
    """Tests for BreakpointManager convenience properties."""

    def test_is_mobile(self):
        """Test is_mobile is True when MOBILE, False otherwise."""
        mgr = BreakpointManager(width=400, height=600)
        assert mgr.is_mobile is True
        assert mgr.is_tablet is False
        assert mgr.is_desktop is False

    def test_is_tablet(self):
        """Test is_tablet is True when TABLET, False otherwise."""
        mgr = BreakpointManager(width=768, height=600)
        assert mgr.is_tablet is True
        assert mgr.is_mobile is False
        assert mgr.is_desktop is False

    def test_is_desktop(self):
        """Test is_desktop is True when DESKTOP, False otherwise."""
        mgr = BreakpointManager(width=1200, height=600)
        assert mgr.is_desktop is True
        assert mgr.is_mobile is False
        assert mgr.is_tablet is False

    def test_safe_width_subtracts_horizontal_insets(self):
        """Test safe_width subtracts left + right from width."""
        safe = SafeAreaInsets(left=10, right=20)
        mgr = BreakpointManager(width=800, height=600, safe_area=safe)
        assert mgr.safe_width == 770

    def test_safe_height_subtracts_vertical_insets(self):
        """Test safe_height subtracts top + bottom from height."""
        safe = SafeAreaInsets(top=44, bottom=34)
        mgr = BreakpointManager(width=800, height=600, safe_area=safe)
        assert mgr.safe_height == 522

    def test_safe_rect_returns_correct_rect(self):
        """Test safe_rect returns a Rect representing the safe content area."""
        safe = SafeAreaInsets(top=44, left=10, right=20, bottom=34)
        mgr = BreakpointManager(width=800, height=600, safe_area=safe)
        rect = mgr.safe_rect
        assert isinstance(rect, Rect)
        assert rect.x == 10
        assert rect.y == 44
        assert rect.width == 770
        assert rect.height == 522

    def test_safe_width_floors_to_zero(self):
        """Test safe_width does not go below 0 when insets exceed width."""
        safe = SafeAreaInsets(left=500, right=500)
        mgr = BreakpointManager(width=800, height=600, safe_area=safe)
        assert mgr.safe_width == 0


class TestBreakpointManagerOrientation:
    """Tests for orientation detection."""

    def test_portrait_when_height_exceeds_width(self):
        """Test height > width gives PORTRAIT."""
        mgr = BreakpointManager(width=400, height=800)
        assert mgr.orientation == Orientation.PORTRAIT
        assert mgr.is_portrait is True
        assert mgr.is_landscape is False

    def test_landscape_when_width_exceeds_height(self):
        """Test width > height gives LANDSCAPE."""
        mgr = BreakpointManager(width=800, height=400)
        assert mgr.orientation == Orientation.LANDSCAPE
        assert mgr.is_landscape is True
        assert mgr.is_portrait is False

    def test_square_is_landscape(self):
        """Test equal width and height gives LANDSCAPE."""
        mgr = BreakpointManager(width=600, height=600)
        assert mgr.orientation == Orientation.LANDSCAPE


class TestBreakpointManagerUpdateSize:
    """Tests for BreakpointManager.update_size()."""

    def test_update_size_changes_breakpoint(self):
        """Test updating width from mobile to tablet changes breakpoint."""
        mgr = BreakpointManager(width=400, height=600)
        assert mgr.breakpoint == Breakpoint.MOBILE

        mgr.update_size(width=800, height=600)
        assert mgr.breakpoint == Breakpoint.TABLET

    def test_update_size_changes_orientation(self):
        """Test updating dimensions from portrait to landscape."""
        mgr = BreakpointManager(width=400, height=800)
        assert mgr.orientation == Orientation.PORTRAIT

        mgr.update_size(width=800, height=400)
        assert mgr.orientation == Orientation.LANDSCAPE

    def test_update_size_preserves_width_when_unchanged(self):
        """Test width unchanged when only height changes."""
        mgr = BreakpointManager(width=400, height=600)
        mgr.update_size(width=400, height=800)
        assert mgr.width == 400
        assert mgr.breakpoint == Breakpoint.MOBILE

    def test_update_size_negative_width_rejected(self):
        """Test update_size with negative width raises ValueError."""
        mgr = BreakpointManager(width=800, height=600)
        with pytest.raises(ValueError, match="Width cannot be negative"):
            mgr.update_size(width=-100, height=600)

    def test_update_size_negative_height_rejected(self):
        """Test update_size with negative height raises ValueError."""
        mgr = BreakpointManager(width=800, height=600)
        with pytest.raises(ValueError, match="Height cannot be negative"):
            mgr.update_size(width=800, height=-100)

    def test_update_size_updates_safe_area(self):
        """Test update_size can replace the safe area."""
        mgr = BreakpointManager(width=800, height=600)
        new_safe = SafeAreaInsets(top=50)
        mgr.update_size(width=800, height=600, safe_area=new_safe)
        assert mgr.safe_area.top == 50

    def test_no_callback_when_breakpoint_unchanged(self):
        """Test breakpoint callback not called when breakpoint stays same."""
        mgr = BreakpointManager(width=400, height=600)
        calls = []
        mgr.set_on_breakpoint_changed(lambda bp: calls.append(bp))

        mgr.update_size(width=500, height=600)
        assert len(calls) == 0

    def test_breakpoint_change_callback_fires(self):
        """Test breakpoint callback fires on transition MOBILE -> TABLET."""
        mgr = BreakpointManager(width=400, height=600)
        calls = []
        mgr.set_on_breakpoint_changed(lambda bp: calls.append(bp))

        mgr.update_size(width=800, height=600)
        assert len(calls) == 1
        assert calls[0] == Breakpoint.TABLET

    def test_breakpoint_change_callback_mobile_to_desktop(self):
        """Test breakpoint callback fires on MOBILE -> DESKTOP."""
        mgr = BreakpointManager(width=400, height=600)
        calls = []
        mgr.set_on_breakpoint_changed(lambda bp: calls.append(bp))

        mgr.update_size(width=1200, height=600)
        assert len(calls) == 1
        assert calls[0] == Breakpoint.DESKTOP

    def test_orientation_change_callback_fires(self):
        """Test orientation callback fires on PORTRAIT -> LANDSCAPE."""
        mgr = BreakpointManager(width=400, height=800)
        calls = []
        mgr.set_on_orientation_changed(lambda o: calls.append(o))

        mgr.update_size(width=800, height=400)
        assert len(calls) == 1
        assert calls[0] == Orientation.LANDSCAPE

    def test_no_orientation_callback_when_unchanged(self):
        """Test orientation callback not called when orientation stays same."""
        mgr = BreakpointManager(width=400, height=800)
        calls = []
        mgr.set_on_orientation_changed(lambda o: calls.append(o))

        mgr.update_size(width=450, height=750)
        assert len(calls) == 0

    def test_breakpoint_and_orientation_callbacks_both_fire(self):
        """Test both callbacks fire when both breakpoint and orientation change."""
        mgr = BreakpointManager(width=400, height=800)
        bp_calls = []
        or_calls = []
        mgr.set_on_breakpoint_changed(lambda bp: bp_calls.append(bp))
        mgr.set_on_orientation_changed(lambda o: or_calls.append(o))

        mgr.update_size(width=1200, height=600)
        assert len(bp_calls) == 1
        assert len(or_calls) == 1


class TestBreakpointManagerGetColumns:
    """Tests for BreakpointManager.get_columns()."""

    def test_get_columns_mobile_returns_one(self):
        """Test get_columns returns 1 on mobile."""
        mgr = BreakpointManager(width=400, height=600)
        assert mgr.get_columns(mobile=1, tablet=2, desktop=3) == 1

    def test_get_columns_tablet_returns_two(self):
        """Test get_columns returns 2 on tablet."""
        mgr = BreakpointManager(width=768, height=600)
        assert mgr.get_columns(mobile=1, tablet=2, desktop=3) == 2

    def test_get_columns_desktop_returns_three(self):
        """Test get_columns returns 3 on desktop."""
        mgr = BreakpointManager(width=1200, height=600)
        assert mgr.get_columns(mobile=1, tablet=2, desktop=3) == 3


class TestBreakpointManagerGetSpacing:
    """Tests for BreakpointManager.get_spacing()."""

    def test_get_spacing_mobile_applies_mobile_scale(self):
        """Test get_spacing multiplies base by mobile_scale."""
        mgr = BreakpointManager(width=400, height=600)
        assert mgr.get_spacing(16, mobile_scale=0.75) == 12.0

    def test_get_spacing_tablet_applies_tablet_scale(self):
        """Test get_spacing multiplies base by tablet_scale."""
        mgr = BreakpointManager(width=768, height=600)
        assert mgr.get_spacing(16, tablet_scale=1.25) == 20.0

    def test_get_spacing_desktop_applies_desktop_scale(self):
        """Test get_spacing multiplies base by desktop_scale."""
        mgr = BreakpointManager(width=1200, height=600)
        assert mgr.get_spacing(16, desktop_scale=1.5) == 24.0

    def test_get_spacing_default_scale_is_one(self):
        """Test get_spacing returns base value when all scales are 1.0."""
        mgr = BreakpointManager(width=768, height=600)
        assert mgr.get_spacing(16) == 16.0


class TestBreakpointManagerGetValue:
    """Tests for BreakpointManager.get_value()."""

    def test_get_value_mobile_returns_mobile(self):
        """Test get_value returns mobile field on MOBILE breakpoint."""
        mgr = BreakpointManager(width=400, height=600)
        rv = ResponsiveValue(mobile=10, tablet=20, desktop=30)
        assert mgr.get_value(rv) == 10

    def test_get_value_tablet_returns_tablet(self):
        """Test get_value returns tablet field on TABLET breakpoint."""
        mgr = BreakpointManager(width=768, height=600)
        rv = ResponsiveValue(mobile=10, tablet=20, desktop=30)
        assert mgr.get_value(rv) == 20

    def test_get_value_desktop_returns_desktop(self):
        """Test get_value returns desktop field on DESKTOP breakpoint."""
        mgr = BreakpointManager(width=1200, height=600)
        rv = ResponsiveValue(mobile=10, tablet=20, desktop=30)
        assert mgr.get_value(rv) == 30

    def test_get_value_updates_after_resize(self):
        """Test get_value reflects new breakpoint after update_size."""
        mgr = BreakpointManager(width=400, height=600)
        rv = ResponsiveValue(mobile=10, tablet=20, desktop=30)
        assert mgr.get_value(rv) == 10

        mgr.update_size(width=1200, height=600)
        assert mgr.get_value(rv) == 30


# ============================================================
# T-1.12: Verify Responsive Value Resolution
# ============================================================

class TestResponsiveValueInit:
    """Tests for ResponsiveValue initialization."""

    def test_all_fields(self):
        """Test ResponsiveValue with all three breakpoint values."""
        rv = ResponsiveValue(mobile=10, tablet=20, desktop=30)
        assert rv.mobile == 10
        assert rv.tablet == 20
        assert rv.desktop == 30

    def test_mobile_only(self):
        """Test ResponsiveValue with only mobile; tablet/desktop default None."""
        rv = ResponsiveValue(mobile=42)
        assert rv.mobile == 42
        assert rv.tablet is None
        assert rv.desktop is None

    def test_works_with_integer_values(self):
        """Test ResponsiveValue works with int type (T-1.12)."""
        rv = ResponsiveValue(mobile=10, tablet=20, desktop=30)
        assert rv.get(Breakpoint.MOBILE) == 10

    def test_works_with_string_values(self):
        """Test ResponsiveValue works with str type (T-1.12)."""
        rv = ResponsiveValue(mobile="small", tablet="medium", desktop="large")
        assert rv.get(Breakpoint.DESKTOP) == "large"

    def test_works_with_dict_values(self):
        """Test ResponsiveValue works with dict type (T-1.12)."""
        rv = ResponsiveValue(
            mobile={"cols": 1},
            tablet={"cols": 2},
            desktop={"cols": 3},
        )
        assert rv.get(Breakpoint.TABLET) == {"cols": 2}

    def test_constant_classmethod(self):
        """Test constant() returns same value for all breakpoints via get()."""
        rv = ResponsiveValue.constant(42)
        assert rv.mobile == 42
        assert rv.tablet is None
        assert rv.desktop is None
        assert rv.get(Breakpoint.MOBILE) == 42
        assert rv.get(Breakpoint.TABLET) == 42
        assert rv.get(Breakpoint.DESKTOP) == 42


class TestResponsiveValueGet:
    """T-1.12: Verify ResponsiveValue.get() with cascading fallback."""

    def test_get_exact_mobile(self):
        """Test get(MOBILE) returns mobile value when set."""
        rv = ResponsiveValue(mobile=1, tablet=2, desktop=3)
        assert rv.get(Breakpoint.MOBILE) == 1

    def test_get_exact_tablet(self):
        """Test get(TABLET) returns tablet value when set."""
        rv = ResponsiveValue(mobile=1, tablet=2, desktop=3)
        assert rv.get(Breakpoint.TABLET) == 2

    def test_get_exact_desktop(self):
        """Test get(DESKTOP) returns desktop value when set."""
        rv = ResponsiveValue(mobile=1, tablet=2, desktop=3)
        assert rv.get(Breakpoint.DESKTOP) == 3

    def test_get_tablet_falls_back_to_desktop(self):
        """Test get(TABLET) falls back to desktop when no tablet value."""
        rv = ResponsiveValue(mobile=1, desktop=3)
        assert rv.get(Breakpoint.TABLET) == 3

    def test_get_tablet_falls_back_to_mobile(self):
        """Test get(TABLET) falls back to mobile when no tablet or desktop."""
        rv = ResponsiveValue(mobile=1)
        assert rv.get(Breakpoint.TABLET) == 1

    def test_get_desktop_falls_back_to_mobile(self):
        """Test get(DESKTOP) falls back to mobile when no desktop value."""
        rv = ResponsiveValue(mobile=1, tablet=2)
        assert rv.get(Breakpoint.DESKTOP) == 1


# ============================================================
# T-1.13: Verify Safe Area Insets
# ============================================================

class TestSafeAreaInsetsInit:
    """Tests for SafeAreaInsets initialization (T-1.13)."""

    def test_default_all_zero(self):
        """Test default SafeAreaInsets has 0 on all sides."""
        insets = SafeAreaInsets()
        assert insets.top == 0.0
        assert insets.right == 0.0
        assert insets.bottom == 0.0
        assert insets.left == 0.0

    def test_custom_values(self):
        """Test SafeAreaInsets with all four custom values (T-1.13)."""
        insets = SafeAreaInsets(top=44, right=20, bottom=34, left=20)
        assert insets.top == 44
        assert insets.right == 20
        assert insets.bottom == 34
        assert insets.left == 20

    def test_negative_top_raises_value_error(self):
        """Test negative top is rejected."""
        with pytest.raises(ValueError, match="top cannot be negative"):
            SafeAreaInsets(top=-1)

    def test_negative_right_raises_value_error(self):
        """Test negative right is rejected."""
        with pytest.raises(ValueError, match="right cannot be negative"):
            SafeAreaInsets(right=-1)

    def test_negative_bottom_raises_value_error(self):
        """Test negative bottom is rejected."""
        with pytest.raises(ValueError, match="bottom cannot be negative"):
            SafeAreaInsets(bottom=-1)

    def test_negative_left_raises_value_error(self):
        """Test negative left is rejected."""
        with pytest.raises(ValueError, match="left cannot be negative"):
            SafeAreaInsets(left=-1)

    def test_zero_insets_accepted(self):
        """Test zero is accepted for all sides."""
        insets = SafeAreaInsets(top=0, right=0, bottom=0, left=0)
        assert insets.top == 0


class TestSafeAreaInsetsProperties:
    """Tests for SafeAreaInsets derived properties (T-1.13)."""

    def test_horizontal(self):
        """Test horizontal = left + right."""
        insets = SafeAreaInsets(left=10, right=20)
        assert insets.horizontal == 30

    def test_vertical(self):
        """Test vertical = top + bottom."""
        insets = SafeAreaInsets(top=44, bottom=34)
        assert insets.vertical == 78

    def test_horizontal_zero_when_no_horizontal_insets(self):
        """Test horizontal is 0 when left and right are 0."""
        insets = SafeAreaInsets()
        assert insets.horizontal == 0

    def test_vertical_zero_when_no_vertical_insets(self):
        """Test vertical is 0 when top and bottom are 0."""
        insets = SafeAreaInsets()
        assert insets.vertical == 0


class TestSafeAreaInsetsHelpers:
    """Tests for SafeAreaInsets helper methods."""

    def test_with_top_returns_new_instance(self):
        """Test with_top creates a new instance; original unchanged."""
        insets = SafeAreaInsets(top=44, bottom=34)
        modified = insets.with_top(20)
        assert modified.top == 20
        assert modified.bottom == 34
        assert insets.top == 44

    def test_with_right_returns_new_instance(self):
        """Test with_right creates a new instance."""
        insets = SafeAreaInsets(right=20)
        modified = insets.with_right(30)
        assert modified.right == 30
        assert insets.right == 20

    def test_with_bottom_returns_new_instance(self):
        """Test with_bottom creates a new instance."""
        insets = SafeAreaInsets(bottom=34)
        modified = insets.with_bottom(50)
        assert modified.bottom == 50

    def test_with_left_returns_new_instance(self):
        """Test with_left creates a new instance."""
        insets = SafeAreaInsets(left=20)
        modified = insets.with_left(15)
        assert modified.left == 15

    def test_uniform(self):
        """Test uniform sets all four sides to the same value."""
        insets = SafeAreaInsets.uniform(10)
        assert insets.top == 10
        assert insets.right == 10
        assert insets.bottom == 10
        assert insets.left == 10

    def test_symmetric(self):
        """Test symmetric sets vertical = top/bottom, horizontal = left/right."""
        insets = SafeAreaInsets.symmetric(horizontal=20, vertical=10)
        assert insets.top == 10
        assert insets.right == 20
        assert insets.bottom == 10
        assert insets.left == 20

    def test_symmetric_defaults_to_zero(self):
        """Test symmetric with no args defaults to 0 on all sides."""
        insets = SafeAreaInsets.symmetric()
        assert insets.top == 0
        assert insets.right == 0
        assert insets.bottom == 0
        assert insets.left == 0


# ============================================================
# ResponsiveRule Tests
# ============================================================

class TestResponsiveRule:
    """Tests for ResponsiveRule dataclass."""

    def test_default_values(self):
        """Test default rule values."""
        rule = ResponsiveRule(breakpoint=Breakpoint.MOBILE)
        assert rule.breakpoint == Breakpoint.MOBILE
        assert rule.visibility == Visibility.VISIBLE
        assert rule.padding_scale == 1.0
        assert rule.margin_scale == 1.0
        assert rule.gap_scale == 1.0
        assert rule.font_scale == 1.0
        assert rule.custom_properties == {}

    def test_custom_values(self):
        """Test rule with explicit custom values."""
        rule = ResponsiveRule(
            breakpoint=Breakpoint.DESKTOP,
            visibility=Visibility.HIDDEN,
            padding_scale=1.5,
            gap_scale=1.2,
            custom_properties={"columns": 4},
        )
        assert rule.breakpoint == Breakpoint.DESKTOP
        assert rule.visibility == Visibility.HIDDEN
        assert rule.custom_properties["columns"] == 4


# ============================================================
# ResponsiveContainer Tests
# ============================================================

class _SentinelLayout:
    """Minimal layout stub for ResponsiveContainer tests."""
    def __init__(self):
        self._children = []
        self.padding_left = 0.0
        self.gap = 0.0
        self._layout_result = {}

    def calculate_layout(self):
        return self._layout_result

    def set_padding(self, all=0.0):
        self.padding_left = all


class _DummyWidget:
    """Minimal widget for ResponsiveContainer visibility tests."""
    def __init__(self, name: str = ""):
        self.name = name


class TestResponsiveContainerInit:
    """Tests for ResponsiveContainer initialization."""

    def test_wraps_layout_and_manager(self):
        """Test container stores layout and breakpoint_manager references."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=800, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        assert container.layout is layout
        assert container.breakpoint_manager is mgr

    def test_current_breakpoint_matches_manager(self):
        """Test current_breakpoint reflects the breakpoint manager's state."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=1200, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        assert container.current_breakpoint == Breakpoint.DESKTOP

    def test_current_breakpoint_is_tablet(self):
        """Test container correctly reports TABLET breakpoint."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=768, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        assert container.current_breakpoint == Breakpoint.TABLET

    def test_current_breakpoint_is_mobile(self):
        """Test container correctly reports MOBILE breakpoint."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        assert container.current_breakpoint == Breakpoint.MOBILE

    def test_accepts_initial_rules(self):
        """Test container accepts rules list on init."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=1200, height=600)
        rule = ResponsiveRule(breakpoint=Breakpoint.DESKTOP)
        container = ResponsiveContainer(
            layout=layout, breakpoint_manager=mgr, rules=[rule]
        )
        assert container.current_rule is not None
        assert container.current_rule.breakpoint == Breakpoint.DESKTOP

    def test_no_current_rule_when_no_match(self):
        """Test current_rule is None when no rule matches active breakpoint."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=768, height=600)
        rule = ResponsiveRule(breakpoint=Breakpoint.DESKTOP)
        container = ResponsiveContainer(
            layout=layout, breakpoint_manager=mgr, rules=[rule]
        )
        assert container.current_rule is None


class TestResponsiveContainerRules:
    """Tests for ResponsiveContainer rule management."""

    def test_add_rule(self):
        """Test add_rule stores rule and applies if breakpoint matches."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=1200, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        rule = ResponsiveRule(breakpoint=Breakpoint.DESKTOP)
        container.add_rule(rule)
        assert container.current_rule is not None
        assert container.current_rule.breakpoint == Breakpoint.DESKTOP

    def test_add_rule_no_match(self):
        """Test add_rule stores rule but current_rule unchanged if no match."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        rule = ResponsiveRule(breakpoint=Breakpoint.DESKTOP)
        container.add_rule(rule)
        assert container.current_rule is None

    def test_remove_existing_rule(self):
        """Test remove_rule returns True and clears match for that breakpoint."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=1200, height=600)
        rule = ResponsiveRule(breakpoint=Breakpoint.DESKTOP)
        container = ResponsiveContainer(
            layout=layout, breakpoint_manager=mgr, rules=[rule]
        )
        assert container.current_rule is not None
        result = container.remove_rule(Breakpoint.DESKTOP)
        assert result is True
        assert container.current_rule is None

    def test_remove_nonexistent_rule(self):
        """Test remove_rule returns False for non-existent breakpoint."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=800, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        result = container.remove_rule(Breakpoint.MOBILE)
        assert result is False


class TestResponsiveContainerVisibility:
    """Tests for ResponsiveContainer visibility rules."""

    def test_widget_visible_by_default(self):
        """Test get_widget_visibility returns VISIBLE for unknown widget."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=800, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")
        assert container.get_widget_visibility(widget) == Visibility.VISIBLE
        assert container.is_widget_visible(widget) is True

    def test_set_visibility_hidden_on_mobile(self):
        """Test widget can be hidden on mobile breakpoint."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")

        container.set_visibility_rule(
            widget,
            mobile=Visibility.HIDDEN,
            tablet=Visibility.VISIBLE,
            desktop=Visibility.VISIBLE,
        )

        assert container.get_widget_visibility(widget) == Visibility.HIDDEN
        assert container.is_widget_visible(widget) is False

    def test_visibility_changes_with_breakpoint(self):
        """Test visibility updates after breakpoint transition."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")

        container.set_visibility_rule(
            widget,
            mobile=Visibility.HIDDEN,
            tablet=Visibility.VISIBLE,
            desktop=Visibility.VISIBLE,
        )

        assert container.is_widget_visible(widget) is False

        mgr.update_size(width=800, height=600)
        assert container.is_widget_visible(widget) is True

    def test_multiple_widgets_different_visibility(self):
        """Test independent visibility rules for multiple widgets."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        w1 = _DummyWidget("always")
        w2 = _DummyWidget("mobile-only")

        container.set_visibility_rule(
            w1,
            mobile=Visibility.VISIBLE,
            tablet=Visibility.VISIBLE,
            desktop=Visibility.VISIBLE,
        )
        container.set_visibility_rule(
            w2,
            mobile=Visibility.VISIBLE,
            tablet=Visibility.HIDDEN,
            desktop=Visibility.HIDDEN,
        )

        assert container.is_widget_visible(w1) is True
        assert container.is_widget_visible(w2) is True

        mgr.update_size(width=1200, height=600)
        assert container.is_widget_visible(w1) is True
        assert container.is_widget_visible(w2) is False

    def test_collapsed_visibility_returned(self):
        """Test get_widget_visibility returns COLLAPSED when set."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")

        container.set_visibility_rule(
            widget,
            mobile=Visibility.COLLAPSED,
            tablet=Visibility.VISIBLE,
            desktop=Visibility.VISIBLE,
        )

        assert container.get_widget_visibility(widget) == Visibility.COLLAPSED


class TestResponsiveContainerCalculateLayout:
    """Tests for ResponsiveContainer.calculate_layout()."""

    def test_delegates_to_underlying_layout(self):
        """Test calculate_layout returns result from wrapped layout."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=800, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)

        expected = {1: Rect(x=0, y=0, width=100, height=50)}
        layout._layout_result = expected

        result = container.calculate_layout()
        assert result == expected

    def test_empty_layout_returns_empty_dict(self):
        """Test calculate_layout with no layout result returns empty dict."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=800, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)

        result = container.calculate_layout()
        assert result == {}


class TestResponsiveContainerCallbacks:
    """Tests for ResponsiveContainer layout-changed callback."""

    def test_layout_changed_callback_on_breakpoint_change(self):
        """Test set_on_layout_changed callback fires on breakpoint transition."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        calls = []
        container.set_on_layout_changed(lambda: calls.append("changed"))

        mgr.update_size(width=800, height=600)
        assert len(calls) == 1


# ============================================================
# Helper Function Tests
# ============================================================

class TestResponsiveSpacing:
    """Tests for responsive_spacing() helper."""

    def test_default_scales(self):
        """Test responsive_spacing applies default scales (0.75, 1.0, 1.25)."""
        rv = responsive_spacing(base=16)
        assert rv.mobile == 12.0
        assert rv.tablet == 16.0
        assert rv.desktop == 20.0

    def test_custom_scales(self):
        """Test responsive_spacing with user-supplied scales."""
        rv = responsive_spacing(
            base=8,
            mobile_scale=0.5,
            tablet_scale=1.0,
            desktop_scale=2.0,
        )
        assert rv.mobile == 4.0
        assert rv.tablet == 8.0
        assert rv.desktop == 16.0


class TestResponsiveFontSize:
    """Tests for responsive_font_size() helper."""

    def test_default_scales(self):
        """Test responsive_font_size applies default scales (0.875, 1.0, 1.125)."""
        rv = responsive_font_size(base=16)
        assert rv.mobile == 14.0
        assert rv.tablet == 16.0
        assert rv.desktop == 18.0

    def test_custom_scales(self):
        """Test responsive_font_size with user-supplied scales."""
        rv = responsive_font_size(
            base=20,
            mobile_scale=0.5,
            tablet_scale=1.0,
            desktop_scale=1.5,
        )
        assert rv.mobile == 10.0
        assert rv.desktop == 30.0


class TestHideOnMobile:
    """Tests for hide_on_mobile() helper."""

    def test_hides_on_mobile(self):
        """Test hide_on_mobile sets HIDDEN for mobile, VISIBLE otherwise."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")

        hide_on_mobile(widget, container)

        assert container.is_widget_visible(widget) is False

    def test_visible_on_tablet(self):
        """Test widget hidden on mobile becomes visible on tablet."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")

        hide_on_mobile(widget, container)

        mgr.update_size(width=768, height=600)
        assert container.is_widget_visible(widget) is True

    def test_visible_on_desktop(self):
        """Test widget hidden on mobile becomes visible on desktop."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")

        hide_on_mobile(widget, container)

        mgr.update_size(width=1200, height=600)
        assert container.is_widget_visible(widget) is True


class TestShowOnlyOnMobile:
    """Tests for show_only_on_mobile() helper."""

    def test_visible_on_mobile(self):
        """Test show_only_on_mobile keeps widget visible on mobile."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")

        show_only_on_mobile(widget, container)

        assert container.is_widget_visible(widget) is True

    def test_hidden_on_tablet(self):
        """Test widget hidden on tablet."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")

        show_only_on_mobile(widget, container)

        mgr.update_size(width=768, height=600)
        assert container.is_widget_visible(widget) is False

    def test_hidden_on_desktop(self):
        """Test widget hidden on desktop."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")

        show_only_on_mobile(widget, container)

        mgr.update_size(width=1200, height=600)
        assert container.is_widget_visible(widget) is False


class TestHideOnDesktop:
    """Tests for hide_on_desktop() helper."""

    def test_visible_on_mobile(self):
        """Test hide_on_desktop keeps widget visible on mobile."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")

        hide_on_desktop(widget, container)

        assert container.is_widget_visible(widget) is True

    def test_visible_on_tablet(self):
        """Test widget visible on tablet."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=768, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")

        hide_on_desktop(widget, container)

        assert container.is_widget_visible(widget) is True

    def test_hidden_on_desktop(self):
        """Test hide_on_desktop hides widget on desktop."""
        layout = _SentinelLayout()
        mgr = BreakpointManager(width=400, height=600)
        container = ResponsiveContainer(layout=layout, breakpoint_manager=mgr)
        widget = _DummyWidget("test")

        hide_on_desktop(widget, container)

        mgr.update_size(width=1200, height=600)
        assert container.is_widget_visible(widget) is False
