"""
Comprehensive tests for the Checkbox widget.

Tests cover:
- Initialization and default values
- Check states (checked, unchecked, indeterminate)
- Visual states (normal, hovered, focused, disabled)
- Mouse and keyboard interactions
- State change events
- Indeterminate state handling
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def checkbox_class():
    """Reset checkbox ID counter before each test."""
    from engine.ui.widgets.input.checkbox import Checkbox
    Checkbox.reset_id_counter()
    return Checkbox


class TestCheckState:
    """Tests for CheckState enumeration."""

    def test_check_state_unchecked(self):
        """Test UNCHECKED state exists."""
        from engine.ui.widgets.input.checkbox import CheckState

        assert CheckState.UNCHECKED is not None

    def test_check_state_checked(self):
        """Test CHECKED state exists."""
        from engine.ui.widgets.input.checkbox import CheckState

        assert CheckState.CHECKED is not None

    def test_check_state_indeterminate(self):
        """Test INDETERMINATE state exists."""
        from engine.ui.widgets.input.checkbox import CheckState

        assert CheckState.INDETERMINATE is not None


class TestCheckboxState:
    """Tests for CheckboxState (visual state) enumeration."""

    def test_checkbox_state_normal(self):
        """Test NORMAL visual state exists."""
        from engine.ui.widgets.input.checkbox import CheckboxState

        assert CheckboxState.NORMAL is not None

    def test_checkbox_state_hovered(self):
        """Test HOVERED visual state exists."""
        from engine.ui.widgets.input.checkbox import CheckboxState

        assert CheckboxState.HOVERED is not None

    def test_checkbox_state_focused(self):
        """Test FOCUSED visual state exists."""
        from engine.ui.widgets.input.checkbox import CheckboxState

        assert CheckboxState.FOCUSED is not None

    def test_checkbox_state_disabled(self):
        """Test DISABLED visual state exists."""
        from engine.ui.widgets.input.checkbox import CheckboxState

        assert CheckboxState.DISABLED is not None


class TestCheckboxStyle:
    """Tests for CheckboxStyle configuration."""

    def test_checkbox_style_defaults(self):
        """Test default CheckboxStyle values."""
        from engine.ui.widgets.input.checkbox import CheckboxStyle

        style = CheckboxStyle()
        assert style.box_size == 20.0
        assert style.box_color == "#FFFFFF"
        assert style.checked_color == "#4A90D9"
        assert style.check_mark_color == "#FFFFFF"
        assert style.border_width == 2.0
        assert style.corner_radius == 3.0
        assert style.label_spacing == 8.0
        assert style.font_size == 14.0


class TestCheckboxInitialization:
    """Tests for Checkbox initialization."""

    def test_checkbox_default_initialization(self, checkbox_class):
        """Test Checkbox initializes with correct defaults."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class()
        assert checkbox.label == ""
        assert checkbox.check_state == CheckState.UNCHECKED
        assert checkbox.checked is False
        assert checkbox.enabled is True
        assert checkbox.visible is True
        assert checkbox.allow_indeterminate is False

    def test_checkbox_with_label(self, checkbox_class):
        """Test Checkbox with label."""
        checkbox = checkbox_class(label="Accept Terms")
        assert checkbox.label == "Accept Terms"

    def test_checkbox_checked_initial(self, checkbox_class):
        """Test Checkbox created in checked state."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class(checked=True)
        assert checkbox.checked is True
        assert checkbox.check_state == CheckState.CHECKED

    def test_checkbox_disabled_initial(self, checkbox_class):
        """Test Checkbox created in disabled state."""
        from engine.ui.widgets.input.checkbox import CheckboxState

        checkbox = checkbox_class(enabled=False)
        assert checkbox.enabled is False
        assert checkbox.visual_state == CheckboxState.DISABLED

    def test_checkbox_allow_indeterminate(self, checkbox_class):
        """Test Checkbox with indeterminate allowed."""
        checkbox = checkbox_class(allow_indeterminate=True)
        assert checkbox.allow_indeterminate is True

    def test_checkbox_unique_ids(self, checkbox_class):
        """Test checkboxes get unique IDs."""
        cb1 = checkbox_class()
        cb2 = checkbox_class()
        cb3 = checkbox_class()

        assert cb1.id != cb2.id
        assert cb2.id != cb3.id


class TestCheckboxProperties:
    """Tests for Checkbox property getters and setters."""

    def test_checkbox_label_setter(self, checkbox_class):
        """Test setting checkbox label."""
        checkbox = checkbox_class()
        checkbox.label = "New Label"
        assert checkbox.label == "New Label"

    def test_checkbox_check_state_setter(self, checkbox_class):
        """Test setting check state."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class()
        checkbox.check_state = CheckState.CHECKED
        assert checkbox.check_state == CheckState.CHECKED
        assert checkbox.checked is True

    def test_checkbox_check_state_invalid_type(self, checkbox_class):
        """Test setting check state with invalid type."""
        checkbox = checkbox_class()
        with pytest.raises(TypeError, match="must be a CheckState"):
            checkbox.check_state = "checked"

    def test_checkbox_check_state_indeterminate_not_allowed(self, checkbox_class):
        """Test setting indeterminate when not allowed."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class(allow_indeterminate=False)
        with pytest.raises(ValueError, match="Indeterminate state not allowed"):
            checkbox.check_state = CheckState.INDETERMINATE

    def test_checkbox_checked_setter_true(self, checkbox_class):
        """Test setting checked to True."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class()
        checkbox.checked = True
        assert checkbox.checked is True
        assert checkbox.check_state == CheckState.CHECKED

    def test_checkbox_checked_setter_false(self, checkbox_class):
        """Test setting checked to False."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class(checked=True)
        checkbox.checked = False
        assert checkbox.checked is False
        assert checkbox.check_state == CheckState.UNCHECKED

    def test_checkbox_is_indeterminate(self, checkbox_class):
        """Test is_indeterminate property."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class(allow_indeterminate=True)
        assert checkbox.is_indeterminate is False

        checkbox.check_state = CheckState.INDETERMINATE
        assert checkbox.is_indeterminate is True

    def test_checkbox_enabled_setter(self, checkbox_class):
        """Test setting enabled state."""
        from engine.ui.widgets.input.checkbox import CheckboxState

        checkbox = checkbox_class()
        checkbox.enabled = False
        assert checkbox.enabled is False
        assert checkbox.visual_state == CheckboxState.DISABLED

    def test_checkbox_width_setter(self, checkbox_class):
        """Test setting width."""
        checkbox = checkbox_class()
        checkbox.width = 150
        assert checkbox.width == 150

    def test_checkbox_width_negative_fails(self, checkbox_class):
        """Test negative width fails."""
        checkbox = checkbox_class()
        with pytest.raises(ValueError, match="width must be >= 0"):
            checkbox.width = -10

    def test_checkbox_bounds(self, checkbox_class):
        """Test getting checkbox bounds."""
        checkbox = checkbox_class(x=10, y=20, width=100, height=30)
        assert checkbox.bounds == (10, 20, 100, 30)

    def test_checkbox_box_bounds(self, checkbox_class):
        """Test getting box bounds."""
        checkbox = checkbox_class(x=10, y=20)
        bounds = checkbox.box_bounds
        assert bounds[0] == 10
        assert bounds[1] == 20
        assert bounds[2] == checkbox.style.box_size
        assert bounds[3] == checkbox.style.box_size


class TestCheckboxContainsPoint:
    """Tests for point containment."""

    def test_checkbox_contains_point_inside(self, checkbox_class):
        """Test point inside checkbox."""
        checkbox = checkbox_class(x=0, y=0, width=100, height=20)
        assert checkbox.contains_point(50, 10) is True

    def test_checkbox_contains_point_outside(self, checkbox_class):
        """Test point outside checkbox."""
        checkbox = checkbox_class(x=0, y=0, width=100, height=20)
        assert checkbox.contains_point(150, 10) is False


class TestCheckboxMouseInteraction:
    """Tests for mouse interaction handling."""

    def test_checkbox_mouse_enter(self, checkbox_class):
        """Test mouse enter sets hovered state."""
        from engine.ui.widgets.input.checkbox import CheckboxState

        checkbox = checkbox_class()
        checkbox.handle_mouse_enter()
        assert checkbox.visual_state == CheckboxState.HOVERED

    def test_checkbox_mouse_enter_disabled(self, checkbox_class):
        """Test mouse enter is ignored when disabled."""
        from engine.ui.widgets.input.checkbox import CheckboxState

        checkbox = checkbox_class(enabled=False)
        checkbox.handle_mouse_enter()
        assert checkbox.visual_state == CheckboxState.DISABLED

    def test_checkbox_mouse_leave(self, checkbox_class):
        """Test mouse leave clears hovered state."""
        from engine.ui.widgets.input.checkbox import CheckboxState

        checkbox = checkbox_class()
        checkbox.handle_mouse_enter()
        checkbox.handle_mouse_leave()
        assert checkbox.visual_state == CheckboxState.NORMAL

    def test_checkbox_mouse_down_consumed(self, checkbox_class):
        """Test mouse down is consumed inside checkbox."""
        checkbox = checkbox_class(x=0, y=0, width=100, height=20)
        result = checkbox.handle_mouse_down(50, 10)
        assert result is True

    def test_checkbox_mouse_down_outside(self, checkbox_class):
        """Test mouse down outside checkbox is not consumed."""
        checkbox = checkbox_class(x=0, y=0, width=100, height=20)
        result = checkbox.handle_mouse_down(150, 10)
        assert result is False

    def test_checkbox_mouse_up_toggles(self, checkbox_class):
        """Test mouse up toggles checkbox."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class(x=0, y=0, width=100, height=20)
        checkbox.handle_mouse_up(50, 10)
        assert checkbox.check_state == CheckState.CHECKED

    def test_checkbox_mouse_up_disabled(self, checkbox_class):
        """Test mouse up is ignored when disabled."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class(x=0, y=0, width=100, height=20, enabled=False)
        checkbox.handle_mouse_up(50, 10)
        assert checkbox.check_state == CheckState.UNCHECKED


class TestCheckboxKeyboardInteraction:
    """Tests for keyboard interaction handling."""

    def test_checkbox_focus_gained(self, checkbox_class):
        """Test focus gained sets focused state."""
        from engine.ui.widgets.input.checkbox import CheckboxState

        checkbox = checkbox_class()
        checkbox.handle_focus_gained()
        assert checkbox.visual_state == CheckboxState.FOCUSED

    def test_checkbox_focus_gained_disabled(self, checkbox_class):
        """Test focus gained is ignored when disabled."""
        from engine.ui.widgets.input.checkbox import CheckboxState

        checkbox = checkbox_class(enabled=False)
        checkbox.handle_focus_gained()
        assert checkbox.visual_state == CheckboxState.DISABLED

    def test_checkbox_focus_lost(self, checkbox_class):
        """Test focus lost clears focused state."""
        from engine.ui.widgets.input.checkbox import CheckboxState

        checkbox = checkbox_class()
        checkbox.handle_focus_gained()
        checkbox.handle_focus_lost()
        assert checkbox.visual_state == CheckboxState.NORMAL

    def test_checkbox_space_key_toggles(self, checkbox_class):
        """Test space key toggles checkbox."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class()
        checkbox.handle_focus_gained()
        result = checkbox.handle_key_down("space")
        assert result is True
        assert checkbox.check_state == CheckState.CHECKED

    def test_checkbox_space_key_not_focused(self, checkbox_class):
        """Test space key is ignored when not focused."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class()
        result = checkbox.handle_key_down("space")
        assert result is False
        assert checkbox.check_state == CheckState.UNCHECKED


class TestCheckboxToggle:
    """Tests for toggle behavior."""

    def test_checkbox_toggle_unchecked_to_checked(self, checkbox_class):
        """Test toggle from unchecked to checked."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class()
        checkbox.toggle()
        assert checkbox.check_state == CheckState.CHECKED

    def test_checkbox_toggle_checked_to_unchecked(self, checkbox_class):
        """Test toggle from checked to unchecked."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class(checked=True)
        checkbox.toggle()
        assert checkbox.check_state == CheckState.UNCHECKED

    def test_checkbox_toggle_three_state(self, checkbox_class):
        """Test toggle with indeterminate allowed cycles through all states."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class(allow_indeterminate=True)

        checkbox.toggle()
        assert checkbox.check_state == CheckState.CHECKED

        checkbox.toggle()
        assert checkbox.check_state == CheckState.INDETERMINATE

        checkbox.toggle()
        assert checkbox.check_state == CheckState.UNCHECKED

    def test_checkbox_toggle_disabled(self, checkbox_class):
        """Test toggle is ignored when disabled."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class(enabled=False)
        checkbox.toggle()
        assert checkbox.check_state == CheckState.UNCHECKED


class TestCheckboxIndeterminate:
    """Tests for indeterminate state handling."""

    def test_checkbox_set_indeterminate(self, checkbox_class):
        """Test set_indeterminate method."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class(allow_indeterminate=True)
        checkbox.set_indeterminate()
        assert checkbox.check_state == CheckState.INDETERMINATE

    def test_checkbox_set_indeterminate_not_allowed(self, checkbox_class):
        """Test set_indeterminate raises when not allowed."""
        checkbox = checkbox_class(allow_indeterminate=False)
        with pytest.raises(ValueError, match="Indeterminate state not allowed"):
            checkbox.set_indeterminate()

    def test_checkbox_disallow_indeterminate_clears_state(self, checkbox_class):
        """Test disallowing indeterminate clears indeterminate state."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class(allow_indeterminate=True)
        checkbox.set_indeterminate()
        assert checkbox.check_state == CheckState.INDETERMINATE

        checkbox.allow_indeterminate = False
        assert checkbox.check_state == CheckState.UNCHECKED


class TestCheckboxEvents:
    """Tests for checkbox events."""

    def test_checkbox_change_event_on_toggle(self, checkbox_class):
        """Test change event is emitted on toggle."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class()
        handler = MagicMock()
        checkbox.on_change(handler)

        checkbox.toggle()

        assert handler.called
        event = handler.call_args[0][0]
        assert event.new_state == CheckState.CHECKED
        assert event.previous_state == CheckState.UNCHECKED
        assert event.is_user_action is True

    def test_checkbox_change_event_programmatic(self, checkbox_class):
        """Test change event on programmatic change marks is_user_action False."""
        from engine.ui.widgets.input.checkbox import CheckState

        checkbox = checkbox_class()
        handler = MagicMock()
        checkbox.on_change(handler)

        checkbox.checked = True

        assert handler.called
        event = handler.call_args[0][0]
        assert event.is_user_action is False

    def test_checkbox_unsubscribe(self, checkbox_class):
        """Test unsubscribing from change events."""
        checkbox = checkbox_class()
        handler = MagicMock()

        unsubscribe = checkbox.on_change(handler)
        unsubscribe()

        checkbox.toggle()
        assert not handler.called

    def test_checkbox_no_event_when_unchanged(self, checkbox_class):
        """Test no event when value doesn't change."""
        checkbox = checkbox_class(checked=True)
        handler = MagicMock()
        checkbox.on_change(handler)

        checkbox.checked = True  # Same value

        assert not handler.called


class TestCheckboxColors:
    """Tests for checkbox color based on state."""

    def test_checkbox_box_color_normal(self, checkbox_class):
        """Test box color in normal state."""
        from engine.ui.widgets.input.checkbox import CheckboxStyle

        style = CheckboxStyle(box_color="#AABBCC")
        checkbox = checkbox_class(style=style)

        assert checkbox.get_current_box_color() == "#AABBCC"

    def test_checkbox_box_color_hovered(self, checkbox_class):
        """Test box color in hovered state."""
        from engine.ui.widgets.input.checkbox import CheckboxStyle

        style = CheckboxStyle(box_hover_color="#DDEEFF")
        checkbox = checkbox_class(style=style)
        checkbox.handle_mouse_enter()

        assert checkbox.get_current_box_color() == "#DDEEFF"

    def test_checkbox_box_color_checked(self, checkbox_class):
        """Test box color when checked."""
        from engine.ui.widgets.input.checkbox import CheckboxStyle

        style = CheckboxStyle(checked_color="#112233")
        checkbox = checkbox_class(checked=True, style=style)

        assert checkbox.get_current_box_color() == "#112233"

    def test_checkbox_box_color_indeterminate(self, checkbox_class):
        """Test box color when indeterminate."""
        from engine.ui.widgets.input.checkbox import CheckboxStyle

        style = CheckboxStyle(checked_color="#112233")
        checkbox = checkbox_class(allow_indeterminate=True, style=style)
        checkbox.set_indeterminate()

        # Indeterminate uses checked color
        assert checkbox.get_current_box_color() == "#112233"

    def test_checkbox_box_color_disabled(self, checkbox_class):
        """Test box color when disabled."""
        from engine.ui.widgets.input.checkbox import CheckboxStyle

        style = CheckboxStyle(disabled_color="#999999")
        checkbox = checkbox_class(enabled=False, style=style)

        assert checkbox.get_current_box_color() == "#999999"


class TestCheckboxDirtyState:
    """Tests for dirty state tracking."""

    def test_checkbox_dirty_after_check_change(self, checkbox_class):
        """Test checkbox is dirty after check state changes."""
        checkbox = checkbox_class()
        checkbox.mark_clean()
        checkbox.toggle()
        assert checkbox.is_dirty

    def test_checkbox_dirty_after_label_change(self, checkbox_class):
        """Test checkbox is dirty after label changes."""
        checkbox = checkbox_class(label="Old")
        checkbox.mark_clean()
        checkbox.label = "New"
        assert checkbox.is_dirty

    def test_checkbox_mark_clean(self, checkbox_class):
        """Test mark_clean clears dirty state."""
        checkbox = checkbox_class()
        checkbox.mark_clean()
        assert checkbox.is_dirty is False
