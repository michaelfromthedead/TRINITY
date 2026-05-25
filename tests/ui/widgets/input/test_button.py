"""
Comprehensive tests for the Button widget.

Tests cover:
- Initialization and default values
- Button states (normal, hovered, pressed, focused, disabled)
- Click event handling
- Toggle mode behavior
- Mouse and keyboard interactions
- Event subscriptions
- Dirty state tracking
"""

import pytest
from unittest.mock import MagicMock, patch
from time import sleep


@pytest.fixture
def button_class():
    """Reset button ID counter before each test."""
    from engine.ui.widgets.input.button import Button
    Button.reset_id_counter()
    return Button


class TestButtonState:
    """Tests for ButtonState enumeration."""

    def test_button_state_normal(self):
        """Test NORMAL state exists."""
        from engine.ui.widgets.input.button import ButtonState

        assert ButtonState.NORMAL is not None

    def test_button_state_hovered(self):
        """Test HOVERED state exists."""
        from engine.ui.widgets.input.button import ButtonState

        assert ButtonState.HOVERED is not None

    def test_button_state_pressed(self):
        """Test PRESSED state exists."""
        from engine.ui.widgets.input.button import ButtonState

        assert ButtonState.PRESSED is not None

    def test_button_state_focused(self):
        """Test FOCUSED state exists."""
        from engine.ui.widgets.input.button import ButtonState

        assert ButtonState.FOCUSED is not None

    def test_button_state_disabled(self):
        """Test DISABLED state exists."""
        from engine.ui.widgets.input.button import ButtonState

        assert ButtonState.DISABLED is not None


class TestButtonStyle:
    """Tests for ButtonStyle configuration."""

    def test_button_style_defaults(self):
        """Test default ButtonStyle values."""
        from engine.ui.widgets.input.button import ButtonStyle

        style = ButtonStyle()
        assert style.background_color == "#4A90D9"
        assert style.hover_color == "#5BA0E9"
        assert style.pressed_color == "#3A80C9"
        assert style.disabled_color == "#CCCCCC"
        assert style.text_color == "#FFFFFF"
        assert style.border_width == 1.0
        assert style.corner_radius == 4.0
        assert style.padding_horizontal == 16.0
        assert style.padding_vertical == 8.0
        assert style.font_size == 14.0

    def test_button_style_custom(self):
        """Test custom ButtonStyle values."""
        from engine.ui.widgets.input.button import ButtonStyle

        style = ButtonStyle(
            background_color="#FF0000",
            hover_color="#FF3333",
            border_width=2.0,
            corner_radius=8.0
        )
        assert style.background_color == "#FF0000"
        assert style.hover_color == "#FF3333"
        assert style.border_width == 2.0
        assert style.corner_radius == 8.0


class TestButtonInitialization:
    """Tests for Button initialization."""

    def test_button_default_initialization(self, button_class):
        """Test Button initializes with correct defaults."""
        button = button_class()
        assert button.text == ""
        assert button.icon is None
        assert button.enabled is True
        assert button.visible is True
        assert button.toggle_mode is False
        assert button.toggled_on is False

    def test_button_with_text(self, button_class):
        """Test Button with text label."""
        button = button_class(text="Click Me")
        assert button.text == "Click Me"

    def test_button_with_icon(self, button_class):
        """Test Button with icon."""
        button = button_class(icon="icons/save.png")
        assert button.icon == "icons/save.png"

    def test_button_with_text_and_icon(self, button_class):
        """Test Button with both text and icon."""
        button = button_class(text="Save", icon="icons/save.png")
        assert button.text == "Save"
        assert button.icon == "icons/save.png"

    def test_button_disabled_initial(self, button_class):
        """Test Button created in disabled state."""
        from engine.ui.widgets.input.button import ButtonState

        button = button_class(enabled=False)
        assert button.enabled is False
        assert button.state == ButtonState.DISABLED

    def test_button_toggle_mode_initial(self, button_class):
        """Test Button created in toggle mode."""
        button = button_class(toggle_mode=True, toggled_on=True)
        assert button.toggle_mode is True
        assert button.toggled_on is True

    def test_button_with_position(self, button_class):
        """Test Button with position."""
        button = button_class(x=100, y=50)
        assert button.x == 100
        assert button.y == 50

    def test_button_with_dimensions(self, button_class):
        """Test Button with dimensions."""
        button = button_class(width=200, height=60)
        assert button.width == 200
        assert button.height == 60

    def test_button_unique_ids(self, button_class):
        """Test buttons get unique IDs."""
        button1 = button_class()
        button2 = button_class()
        button3 = button_class()

        assert button1.id != button2.id
        assert button2.id != button3.id
        assert button1.id != button3.id


class TestButtonProperties:
    """Tests for Button property getters and setters."""

    def test_button_text_setter(self, button_class):
        """Test setting button text."""
        button = button_class()
        button.text = "New Text"
        assert button.text == "New Text"

    def test_button_icon_setter(self, button_class):
        """Test setting button icon."""
        button = button_class()
        button.icon = "icons/new.png"
        assert button.icon == "icons/new.png"

    def test_button_enabled_setter(self, button_class):
        """Test setting enabled state."""
        from engine.ui.widgets.input.button import ButtonState

        button = button_class()
        button.enabled = False
        assert button.enabled is False
        assert button.state == ButtonState.DISABLED

    def test_button_visible_setter(self, button_class):
        """Test setting visibility."""
        button = button_class()
        button.visible = False
        assert button.visible is False

    def test_button_focusable_disabled(self, button_class):
        """Test focusable is False when disabled."""
        button = button_class(enabled=False)
        assert button.focusable is False

    def test_button_focusable_enabled(self, button_class):
        """Test focusable is True when enabled."""
        button = button_class(enabled=True)
        assert button.focusable is True

    def test_button_width_setter(self, button_class):
        """Test setting width."""
        button = button_class()
        button.width = 150
        assert button.width == 150

    def test_button_width_negative_fails(self, button_class):
        """Test negative width fails."""
        button = button_class()
        with pytest.raises(ValueError, match="width must be >= 0"):
            button.width = -10

    def test_button_height_setter(self, button_class):
        """Test setting height."""
        button = button_class()
        button.height = 50
        assert button.height == 50

    def test_button_height_negative_fails(self, button_class):
        """Test negative height fails."""
        button = button_class()
        with pytest.raises(ValueError, match="height must be >= 0"):
            button.height = -10

    def test_button_bounds(self, button_class):
        """Test getting button bounds."""
        button = button_class(x=10, y=20, width=100, height=40)
        assert button.bounds == (10, 20, 100, 40)

    def test_button_style_setter(self, button_class):
        """Test setting button style."""
        from engine.ui.widgets.input.button import ButtonStyle

        button = button_class()
        new_style = ButtonStyle(background_color="#00FF00")
        button.style = new_style
        assert button.style.background_color == "#00FF00"


class TestButtonContainsPoint:
    """Tests for point containment."""

    def test_button_contains_point_inside(self, button_class):
        """Test point inside button."""
        button = button_class(x=0, y=0, width=100, height=40)
        assert button.contains_point(50, 20) is True

    def test_button_contains_point_edge(self, button_class):
        """Test point on button edge."""
        button = button_class(x=0, y=0, width=100, height=40)
        assert button.contains_point(0, 0) is True
        assert button.contains_point(100, 40) is True

    def test_button_contains_point_outside(self, button_class):
        """Test point outside button."""
        button = button_class(x=0, y=0, width=100, height=40)
        assert button.contains_point(-1, 20) is False
        assert button.contains_point(101, 20) is False
        assert button.contains_point(50, -1) is False
        assert button.contains_point(50, 41) is False


class TestButtonMouseInteraction:
    """Tests for mouse interaction handling."""

    def test_button_mouse_enter(self, button_class):
        """Test mouse enter sets hovered state."""
        from engine.ui.widgets.input.button import ButtonState

        button = button_class()
        button.handle_mouse_enter()
        assert button.state == ButtonState.HOVERED

    def test_button_mouse_enter_disabled(self, button_class):
        """Test mouse enter is ignored when disabled."""
        from engine.ui.widgets.input.button import ButtonState

        button = button_class(enabled=False)
        button.handle_mouse_enter()
        assert button.state == ButtonState.DISABLED

    def test_button_mouse_leave(self, button_class):
        """Test mouse leave clears hovered state."""
        from engine.ui.widgets.input.button import ButtonState

        button = button_class()
        button.handle_mouse_enter()
        button.handle_mouse_leave()
        assert button.state == ButtonState.NORMAL

    def test_button_mouse_down(self, button_class):
        """Test mouse down sets pressed state."""
        from engine.ui.widgets.input.button import ButtonState

        button = button_class(x=0, y=0, width=100, height=40)
        result = button.handle_mouse_down(50, 20)
        assert result is True
        assert button.state == ButtonState.PRESSED

    def test_button_mouse_down_outside(self, button_class):
        """Test mouse down outside button is not consumed."""
        button = button_class(x=0, y=0, width=100, height=40)
        result = button.handle_mouse_down(150, 20)
        assert result is False

    def test_button_mouse_down_disabled(self, button_class):
        """Test mouse down is ignored when disabled."""
        button = button_class(x=0, y=0, width=100, height=40, enabled=False)
        result = button.handle_mouse_down(50, 20)
        assert result is False

    def test_button_mouse_up_triggers_click(self, button_class):
        """Test mouse up after mouse down triggers click."""
        button = button_class(x=0, y=0, width=100, height=40)
        click_handler = MagicMock()
        button.on_click(click_handler)

        button.handle_mouse_down(50, 20)
        button.handle_mouse_up(50, 20)

        assert click_handler.called

    def test_button_mouse_up_outside_no_click(self, button_class):
        """Test mouse up outside button does not trigger click."""
        button = button_class(x=0, y=0, width=100, height=40)
        click_handler = MagicMock()
        button.on_click(click_handler)

        button.handle_mouse_down(50, 20)
        button.handle_mouse_up(150, 20)  # Outside

        assert not click_handler.called

    def test_button_mouse_leave_cancels_press(self, button_class):
        """Test mouse leave while pressed cancels the press."""
        from engine.ui.widgets.input.button import ButtonState

        button = button_class(x=0, y=0, width=100, height=40)
        press_handler = MagicMock()
        button.on_press(press_handler)

        button.handle_mouse_down(50, 20)
        button.handle_mouse_leave()

        # Press end event should be emitted
        assert press_handler.call_count == 2  # Press start and end
        assert button.state == ButtonState.NORMAL


class TestButtonKeyboardInteraction:
    """Tests for keyboard interaction handling."""

    def test_button_focus_gained(self, button_class):
        """Test focus gained sets focused state."""
        from engine.ui.widgets.input.button import ButtonState

        button = button_class()
        button.handle_focus_gained()
        assert button.state == ButtonState.FOCUSED

    def test_button_focus_gained_disabled(self, button_class):
        """Test focus gained is ignored when disabled."""
        from engine.ui.widgets.input.button import ButtonState

        button = button_class(enabled=False)
        button.handle_focus_gained()
        assert button.state == ButtonState.DISABLED

    def test_button_focus_lost(self, button_class):
        """Test focus lost clears focused state."""
        from engine.ui.widgets.input.button import ButtonState

        button = button_class()
        button.handle_focus_gained()
        button.handle_focus_lost()
        assert button.state == ButtonState.NORMAL

    def test_button_space_key_down(self, button_class):
        """Test space key activates button."""
        from engine.ui.widgets.input.button import ButtonState

        button = button_class()
        button.handle_focus_gained()
        result = button.handle_key_down("space")
        assert result is True
        assert button.state == ButtonState.PRESSED

    def test_button_enter_key_down(self, button_class):
        """Test enter key activates button."""
        from engine.ui.widgets.input.button import ButtonState

        button = button_class()
        button.handle_focus_gained()
        result = button.handle_key_down("enter")
        assert result is True
        assert button.state == ButtonState.PRESSED

    def test_button_key_down_not_focused(self, button_class):
        """Test key down is ignored when not focused."""
        button = button_class()
        result = button.handle_key_down("space")
        assert result is False

    def test_button_key_up_triggers_click(self, button_class):
        """Test key up after key down triggers click."""
        button = button_class()
        click_handler = MagicMock()
        button.on_click(click_handler)

        button.handle_focus_gained()
        button.handle_key_down("space")
        button.handle_key_up("space")

        assert click_handler.called

    def test_button_key_modifiers(self, button_class):
        """Test key events pass modifier states."""
        button = button_class()
        click_handler = MagicMock()
        button.on_click(click_handler)

        button.handle_focus_gained()
        button.handle_key_down("space", shift=True, ctrl=True)
        button.handle_key_up("space", shift=True, ctrl=True)

        event = click_handler.call_args[0][0]
        assert event.modifier_shift is True
        assert event.modifier_ctrl is True


class TestButtonToggleMode:
    """Tests for toggle button behavior."""

    def test_button_toggle_on_click(self, button_class):
        """Test toggle state changes on click."""
        button = button_class(toggle_mode=True)
        button.handle_mouse_down(50, 20)
        button.handle_mouse_up(50, 20)

        assert button.toggled_on is True

    def test_button_toggle_off_on_second_click(self, button_class):
        """Test toggle state toggles on second click."""
        button = button_class(toggle_mode=True)

        button.handle_mouse_down(50, 20)
        button.handle_mouse_up(50, 20)
        assert button.toggled_on is True

        button.handle_mouse_down(50, 20)
        button.handle_mouse_up(50, 20)
        assert button.toggled_on is False

    def test_button_toggle_event(self, button_class):
        """Test toggle event is emitted."""
        button = button_class(toggle_mode=True)
        toggle_handler = MagicMock()
        button.on_toggle(toggle_handler)

        button.handle_mouse_down(50, 20)
        button.handle_mouse_up(50, 20)

        assert toggle_handler.called
        event = toggle_handler.call_args[0][0]
        assert event.toggled_on is True
        assert event.previous_state is False

    def test_button_toggle_setter_emits_event(self, button_class):
        """Test setting toggled_on programmatically emits event."""
        button = button_class(toggle_mode=True)
        toggle_handler = MagicMock()
        button.on_toggle(toggle_handler)

        button.toggled_on = True

        assert toggle_handler.called
        event = toggle_handler.call_args[0][0]
        assert event.toggled_on is True

    def test_button_toggle_setter_no_event_if_unchanged(self, button_class):
        """Test no event if toggled_on value unchanged."""
        button = button_class(toggle_mode=True, toggled_on=True)
        toggle_handler = MagicMock()
        button.on_toggle(toggle_handler)

        button.toggled_on = True  # Same value

        assert not toggle_handler.called


class TestButtonEvents:
    """Tests for button event handling."""

    def test_button_click_event_properties(self, button_class):
        """Test click event has correct properties."""
        button = button_class(x=10, y=20, width=100, height=40)
        click_handler = MagicMock()
        button.on_click(click_handler)

        button.handle_mouse_down(60, 40, shift=True, ctrl=False, alt=True)
        button.handle_mouse_up(60, 40, shift=True, ctrl=False, alt=True)

        event = click_handler.call_args[0][0]
        assert event.button is button
        assert event.timestamp > 0
        assert event.position == (50, 20)  # Local position
        assert event.modifier_shift is True
        assert event.modifier_ctrl is False
        assert event.modifier_alt is True

    def test_button_press_event(self, button_class):
        """Test press events are emitted."""
        button = button_class(x=0, y=0, width=100, height=40)
        press_handler = MagicMock()
        button.on_press(press_handler)

        button.handle_mouse_down(50, 20)
        assert press_handler.call_count == 1
        assert press_handler.call_args[0][0].pressed is True

        button.handle_mouse_up(50, 20)
        assert press_handler.call_count == 2
        assert press_handler.call_args[0][0].pressed is False

    def test_button_unsubscribe_click(self, button_class):
        """Test unsubscribing from click events."""
        button = button_class()
        handler = MagicMock()

        unsubscribe = button.on_click(handler)
        unsubscribe()

        button.click()
        assert not handler.called

    def test_button_unsubscribe_press(self, button_class):
        """Test unsubscribing from press events."""
        button = button_class(x=0, y=0, width=100, height=40)
        handler = MagicMock()

        unsubscribe = button.on_press(handler)
        unsubscribe()

        button.handle_mouse_down(50, 20)
        assert not handler.called

    def test_button_unsubscribe_toggle(self, button_class):
        """Test unsubscribing from toggle events."""
        button = button_class(toggle_mode=True)
        handler = MagicMock()

        unsubscribe = button.on_toggle(handler)
        unsubscribe()

        button.toggled_on = True
        assert not handler.called

    def test_button_multiple_click_handlers(self, button_class):
        """Test multiple click handlers are all called."""
        button = button_class()
        handler1 = MagicMock()
        handler2 = MagicMock()
        handler3 = MagicMock()

        button.on_click(handler1)
        button.on_click(handler2)
        button.on_click(handler3)

        button.click()

        assert handler1.called
        assert handler2.called
        assert handler3.called


class TestButtonProgrammaticClick:
    """Tests for programmatic button activation."""

    def test_button_click_method(self, button_class):
        """Test click() method triggers click event."""
        button = button_class()
        click_handler = MagicMock()
        button.on_click(click_handler)

        button.click()

        assert click_handler.called

    def test_button_click_disabled(self, button_class):
        """Test click() is ignored when disabled."""
        button = button_class(enabled=False)
        click_handler = MagicMock()
        button.on_click(click_handler)

        button.click()

        assert not click_handler.called

    def test_button_click_toggles_in_toggle_mode(self, button_class):
        """Test click() toggles state in toggle mode."""
        button = button_class(toggle_mode=True)

        button.click()
        assert button.toggled_on is True

        button.click()
        assert button.toggled_on is False


class TestButtonColors:
    """Tests for button color based on state."""

    def test_button_background_color_normal(self, button_class):
        """Test background color in normal state."""
        from engine.ui.widgets.input.button import ButtonStyle

        style = ButtonStyle(background_color="#123456")
        button = button_class(style=style)

        assert button.get_current_background_color() == "#123456"

    def test_button_background_color_hovered(self, button_class):
        """Test background color in hovered state."""
        from engine.ui.widgets.input.button import ButtonStyle

        style = ButtonStyle(hover_color="#ABCDEF")
        button = button_class(style=style)
        button.handle_mouse_enter()

        assert button.get_current_background_color() == "#ABCDEF"

    def test_button_background_color_pressed(self, button_class):
        """Test background color in pressed state."""
        from engine.ui.widgets.input.button import ButtonStyle

        style = ButtonStyle(pressed_color="#FEDCBA")
        button = button_class(x=0, y=0, width=100, height=40, style=style)
        button.handle_mouse_down(50, 20)

        assert button.get_current_background_color() == "#FEDCBA"

    def test_button_background_color_disabled(self, button_class):
        """Test background color in disabled state."""
        from engine.ui.widgets.input.button import ButtonStyle

        style = ButtonStyle(disabled_color="#999999")
        button = button_class(enabled=False, style=style)

        assert button.get_current_background_color() == "#999999"

    def test_button_text_color_normal(self, button_class):
        """Test text color in normal state."""
        from engine.ui.widgets.input.button import ButtonStyle

        style = ButtonStyle(text_color="#FFFFFF")
        button = button_class(style=style)

        assert button.get_current_text_color() == "#FFFFFF"

    def test_button_text_color_disabled(self, button_class):
        """Test text color in disabled state."""
        from engine.ui.widgets.input.button import ButtonStyle

        style = ButtonStyle(disabled_text_color="#666666")
        button = button_class(enabled=False, style=style)

        assert button.get_current_text_color() == "#666666"


class TestButtonDirtyState:
    """Tests for dirty state tracking."""

    def test_button_dirty_after_text_change(self, button_class):
        """Test button is dirty after text changes."""
        button = button_class(text="Hello")
        button.mark_clean()
        button.text = "World"
        assert button.is_dirty

    def test_button_dirty_after_position_change(self, button_class):
        """Test button is dirty after position changes."""
        button = button_class()
        button.mark_clean()
        button.x = 100
        assert button.is_dirty

    def test_button_dirty_after_state_change(self, button_class):
        """Test button is dirty after state changes."""
        button = button_class()
        button.mark_clean()
        button.handle_mouse_enter()
        assert button.is_dirty

    def test_button_mark_clean(self, button_class):
        """Test mark_clean clears dirty state."""
        button = button_class()
        button.mark_clean()
        assert button.is_dirty is False

    def test_button_text_same_value_not_dirty(self, button_class):
        """Test setting same text value does not mark dirty."""
        button = button_class(text="Hello")
        button.mark_clean()
        button.text = "Hello"  # Same value
        assert button.is_dirty is False
