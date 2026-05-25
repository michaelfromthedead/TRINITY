"""
Comprehensive tests for the Slider widget.

Tests cover:
- Initialization and default values
- Value range and stepping
- Orientation (horizontal/vertical)
- Mouse dragging behavior
- Keyboard interaction
- Value change events
- Thumb position calculation
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def slider_class():
    """Reset slider ID counter before each test."""
    from engine.ui.widgets.input.slider import Slider
    Slider.reset_id_counter()
    return Slider


class TestSliderOrientation:
    """Tests for SliderOrientation enumeration."""

    def test_slider_orientation_horizontal(self):
        """Test HORIZONTAL orientation exists."""
        from engine.ui.widgets.input.slider import SliderOrientation

        assert SliderOrientation.HORIZONTAL is not None

    def test_slider_orientation_vertical(self):
        """Test VERTICAL orientation exists."""
        from engine.ui.widgets.input.slider import SliderOrientation

        assert SliderOrientation.VERTICAL is not None


class TestSliderState:
    """Tests for SliderState enumeration."""

    def test_slider_state_normal(self):
        """Test NORMAL state exists."""
        from engine.ui.widgets.input.slider import SliderState

        assert SliderState.NORMAL is not None

    def test_slider_state_hovered(self):
        """Test HOVERED state exists."""
        from engine.ui.widgets.input.slider import SliderState

        assert SliderState.HOVERED is not None

    def test_slider_state_dragging(self):
        """Test DRAGGING state exists."""
        from engine.ui.widgets.input.slider import SliderState

        assert SliderState.DRAGGING is not None

    def test_slider_state_focused(self):
        """Test FOCUSED state exists."""
        from engine.ui.widgets.input.slider import SliderState

        assert SliderState.FOCUSED is not None

    def test_slider_state_disabled(self):
        """Test DISABLED state exists."""
        from engine.ui.widgets.input.slider import SliderState

        assert SliderState.DISABLED is not None


class TestSliderStyle:
    """Tests for SliderStyle configuration."""

    def test_slider_style_defaults(self):
        """Test default SliderStyle values."""
        from engine.ui.widgets.input.slider import SliderStyle

        style = SliderStyle()
        assert style.track_color == "#E0E0E0"
        assert style.track_fill_color == "#4A90D9"
        assert style.thumb_color == "#4A90D9"
        assert style.thumb_size == 20.0
        assert style.track_height == 6.0
        assert style.show_fill is True
        assert style.show_ticks is False


class TestSliderInitialization:
    """Tests for Slider initialization."""

    def test_slider_default_initialization(self, slider_class):
        """Test Slider initializes with correct defaults."""
        from engine.ui.widgets.input.slider import SliderOrientation

        slider = slider_class()
        assert slider.value == 0.0
        assert slider.min_value == 0.0
        assert slider.max_value == 100.0
        assert slider.step == 0.0
        assert slider.orientation == SliderOrientation.HORIZONTAL
        assert slider.enabled is True
        assert slider.visible is True

    def test_slider_with_value(self, slider_class):
        """Test Slider with initial value."""
        slider = slider_class(value=50.0)
        assert slider.value == 50.0

    def test_slider_with_range(self, slider_class):
        """Test Slider with custom range."""
        slider = slider_class(min_value=10.0, max_value=50.0)
        assert slider.min_value == 10.0
        assert slider.max_value == 50.0

    def test_slider_with_step(self, slider_class):
        """Test Slider with step value."""
        slider = slider_class(step=5.0)
        assert slider.step == 5.0

    def test_slider_value_clamped_to_range(self, slider_class):
        """Test value is clamped to range."""
        slider = slider_class(value=150.0, max_value=100.0)
        assert slider.value == 100.0

    def test_slider_value_stepped(self, slider_class):
        """Test value is rounded to step."""
        slider = slider_class(value=33.0, step=10.0)
        assert slider.value == 30.0

    def test_slider_invalid_range(self, slider_class):
        """Test invalid range raises error."""
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            slider_class(min_value=100.0, max_value=0.0)

    def test_slider_negative_step(self, slider_class):
        """Test negative step raises error."""
        with pytest.raises(ValueError, match="step must be >= 0"):
            slider_class(step=-5.0)

    def test_slider_vertical_orientation(self, slider_class):
        """Test Slider with vertical orientation."""
        from engine.ui.widgets.input.slider import SliderOrientation

        slider = slider_class(orientation=SliderOrientation.VERTICAL)
        assert slider.orientation == SliderOrientation.VERTICAL

    def test_slider_unique_ids(self, slider_class):
        """Test sliders get unique IDs."""
        s1 = slider_class()
        s2 = slider_class()
        s3 = slider_class()

        assert s1.id != s2.id
        assert s2.id != s3.id


class TestSliderProperties:
    """Tests for Slider property getters and setters."""

    def test_slider_value_setter(self, slider_class):
        """Test setting slider value."""
        slider = slider_class()
        slider.value = 75.0
        assert slider.value == 75.0

    def test_slider_value_clamped_on_set(self, slider_class):
        """Test value is clamped when set."""
        slider = slider_class(max_value=100.0)
        slider.value = 150.0
        assert slider.value == 100.0

    def test_slider_value_stepped_on_set(self, slider_class):
        """Test value is stepped when set."""
        slider = slider_class(step=10.0)
        slider.value = 47.0
        assert slider.value == 50.0

    def test_slider_min_value_setter(self, slider_class):
        """Test setting min value."""
        slider = slider_class(value=50.0)
        slider.min_value = 25.0
        assert slider.min_value == 25.0

    def test_slider_min_value_clamps_current(self, slider_class):
        """Test changing min value clamps current value."""
        slider = slider_class(value=10.0)
        slider.min_value = 25.0
        assert slider.value == 25.0

    def test_slider_min_value_invalid(self, slider_class):
        """Test setting invalid min value raises error."""
        slider = slider_class(max_value=100.0)
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            slider.min_value = 150.0

    def test_slider_max_value_setter(self, slider_class):
        """Test setting max value."""
        slider = slider_class(value=50.0)
        slider.max_value = 200.0
        assert slider.max_value == 200.0

    def test_slider_max_value_clamps_current(self, slider_class):
        """Test changing max value clamps current value."""
        slider = slider_class(value=90.0, max_value=100.0)
        slider.max_value = 50.0
        assert slider.value == 50.0

    def test_slider_max_value_invalid(self, slider_class):
        """Test setting invalid max value raises error."""
        slider = slider_class(min_value=50.0)
        with pytest.raises(ValueError, match="max_value must be greater than min_value"):
            slider.max_value = 25.0

    def test_slider_step_setter(self, slider_class):
        """Test setting step value."""
        slider = slider_class(value=45.0)
        slider.step = 10.0
        assert slider.step == 10.0
        assert slider.value == 50.0  # Re-stepped

    def test_slider_step_invalid(self, slider_class):
        """Test setting invalid step raises error."""
        slider = slider_class()
        with pytest.raises(ValueError, match="step must be >= 0"):
            slider.step = -5.0

    def test_slider_range_property(self, slider_class):
        """Test range property."""
        slider = slider_class(min_value=20.0, max_value=80.0)
        assert slider.range == 60.0

    def test_slider_normalized_value(self, slider_class):
        """Test normalized_value property."""
        slider = slider_class(min_value=0.0, max_value=100.0, value=25.0)
        assert slider.normalized_value == pytest.approx(0.25)

    def test_slider_percentage(self, slider_class):
        """Test percentage property."""
        slider = slider_class(min_value=0.0, max_value=100.0, value=75.0)
        assert slider.percentage == pytest.approx(75.0)

    def test_slider_enabled_setter(self, slider_class):
        """Test setting enabled state."""
        from engine.ui.widgets.input.slider import SliderState

        slider = slider_class()
        slider.enabled = False
        assert slider.enabled is False
        assert slider.state == SliderState.DISABLED

    def test_slider_enabled_cancels_drag(self, slider_class):
        """Test disabling cancels active drag."""
        slider = slider_class(x=0, y=0, width=200, height=30)
        slider.handle_mouse_down(100, 15)
        assert slider.is_dragging is True

        drag_end_handler = MagicMock()
        slider.on_drag_end(drag_end_handler)

        slider.enabled = False
        assert slider.is_dragging is False
        assert drag_end_handler.called

    def test_slider_width_setter(self, slider_class):
        """Test setting width."""
        slider = slider_class()
        slider.width = 300
        assert slider.width == 300

    def test_slider_width_negative_fails(self, slider_class):
        """Test negative width fails."""
        slider = slider_class()
        with pytest.raises(ValueError, match="width must be >= 0"):
            slider.width = -10

    def test_slider_bounds(self, slider_class):
        """Test getting slider bounds."""
        slider = slider_class(x=10, y=20, width=200, height=30)
        assert slider.bounds == (10, 20, 200, 30)


class TestSliderThumbPosition:
    """Tests for thumb position calculation."""

    def test_slider_thumb_position_min(self, slider_class):
        """Test thumb position at minimum value."""
        slider = slider_class(x=0, y=0, width=200, height=30, value=0)
        thumb_x, thumb_y = slider.get_thumb_position()
        # Thumb should be at left (plus half thumb size)
        assert thumb_x == pytest.approx(10.0)  # half of default thumb_size

    def test_slider_thumb_position_max(self, slider_class):
        """Test thumb position at maximum value."""
        slider = slider_class(x=0, y=0, width=200, height=30, value=100)
        thumb_x, thumb_y = slider.get_thumb_position()
        # Thumb should be at right (minus half thumb size)
        assert thumb_x == pytest.approx(190.0)  # width - half thumb_size

    def test_slider_thumb_position_middle(self, slider_class):
        """Test thumb position at middle value."""
        slider = slider_class(x=0, y=0, width=200, height=30, value=50)
        thumb_x, thumb_y = slider.get_thumb_position()
        assert thumb_x == pytest.approx(100.0)

    def test_slider_thumb_position_vertical(self, slider_class):
        """Test thumb position in vertical orientation."""
        from engine.ui.widgets.input.slider import SliderOrientation

        slider = slider_class(
            x=0, y=0, width=30, height=200,
            value=50,
            orientation=SliderOrientation.VERTICAL
        )
        thumb_x, thumb_y = slider.get_thumb_position()
        assert thumb_y == pytest.approx(100.0)


class TestSliderTrackBounds:
    """Tests for track bounds calculation."""

    def test_slider_track_bounds_horizontal(self, slider_class):
        """Test track bounds in horizontal orientation."""
        slider = slider_class(x=10, y=20, width=200, height=30)
        x, y, w, h = slider.get_track_bounds()
        assert x == 10
        assert w == 200
        assert h == slider.style.track_height

    def test_slider_track_bounds_vertical(self, slider_class):
        """Test track bounds in vertical orientation."""
        from engine.ui.widgets.input.slider import SliderOrientation

        slider = slider_class(
            x=10, y=20, width=30, height=200,
            orientation=SliderOrientation.VERTICAL
        )
        x, y, w, h = slider.get_track_bounds()
        assert y == 20
        assert h == 200
        assert w == slider.style.track_height


class TestSliderMouseInteraction:
    """Tests for mouse interaction handling."""

    def test_slider_mouse_enter(self, slider_class):
        """Test mouse enter sets hovered state."""
        from engine.ui.widgets.input.slider import SliderState

        slider = slider_class()
        slider.handle_mouse_enter()
        assert slider.state == SliderState.HOVERED

    def test_slider_mouse_enter_disabled(self, slider_class):
        """Test mouse enter is ignored when disabled."""
        from engine.ui.widgets.input.slider import SliderState

        slider = slider_class(enabled=False)
        slider.handle_mouse_enter()
        assert slider.state == SliderState.DISABLED

    def test_slider_mouse_leave(self, slider_class):
        """Test mouse leave clears hovered state."""
        from engine.ui.widgets.input.slider import SliderState

        slider = slider_class()
        slider.handle_mouse_enter()
        slider.handle_mouse_leave()
        assert slider.state == SliderState.NORMAL

    def test_slider_mouse_down_starts_drag(self, slider_class):
        """Test mouse down starts dragging."""
        from engine.ui.widgets.input.slider import SliderState

        slider = slider_class(x=0, y=0, width=200, height=30)
        result = slider.handle_mouse_down(100, 15)
        assert result is True
        assert slider.is_dragging is True
        assert slider.state == SliderState.DRAGGING

    def test_slider_mouse_down_outside(self, slider_class):
        """Test mouse down outside slider is not consumed."""
        slider = slider_class(x=0, y=0, width=200, height=30)
        result = slider.handle_mouse_down(250, 15)
        assert result is False
        assert slider.is_dragging is False

    def test_slider_mouse_down_disabled(self, slider_class):
        """Test mouse down is ignored when disabled."""
        slider = slider_class(x=0, y=0, width=200, height=30, enabled=False)
        result = slider.handle_mouse_down(100, 15)
        assert result is False

    def test_slider_mouse_down_on_track_jumps_value(self, slider_class):
        """Test clicking on track (not thumb) jumps to value."""
        slider = slider_class(x=0, y=0, width=200, height=30, value=0)
        # Click near middle of track
        slider.handle_mouse_down(100, 15)
        assert slider.value == pytest.approx(50.0, abs=5.0)

    def test_slider_mouse_move_during_drag(self, slider_class):
        """Test mouse move during drag updates value."""
        slider = slider_class(x=0, y=0, width=200, height=30, value=0)
        slider.handle_mouse_down(10, 15)  # Start drag at left

        slider.handle_mouse_move(100, 15)  # Drag to middle
        assert slider.value == pytest.approx(50.0, abs=5.0)

        slider.handle_mouse_move(190, 15)  # Drag to right
        assert slider.value == pytest.approx(100.0)

    def test_slider_mouse_move_not_dragging(self, slider_class):
        """Test mouse move is ignored when not dragging."""
        slider = slider_class(x=0, y=0, width=200, height=30, value=50)
        result = slider.handle_mouse_move(100, 15)
        assert result is False
        assert slider.value == 50.0

    def test_slider_mouse_up_ends_drag(self, slider_class):
        """Test mouse up ends dragging."""
        from engine.ui.widgets.input.slider import SliderState

        slider = slider_class(x=0, y=0, width=200, height=30)
        slider.handle_mouse_down(100, 15)
        slider.handle_mouse_up(100, 15)
        assert slider.is_dragging is False
        assert slider.state != SliderState.DRAGGING


class TestSliderKeyboardInteraction:
    """Tests for keyboard interaction handling."""

    def test_slider_focus_gained(self, slider_class):
        """Test focus gained sets focused state."""
        from engine.ui.widgets.input.slider import SliderState

        slider = slider_class()
        slider.handle_focus_gained()
        assert slider.state == SliderState.FOCUSED

    def test_slider_focus_lost(self, slider_class):
        """Test focus lost clears focused state."""
        from engine.ui.widgets.input.slider import SliderState

        slider = slider_class()
        slider.handle_focus_gained()
        slider.handle_focus_lost()
        assert slider.state == SliderState.NORMAL

    def test_slider_right_key_increases_value(self, slider_class):
        """Test right arrow increases value."""
        slider = slider_class(value=50)
        slider.handle_focus_gained()
        slider.handle_key_down("right")
        assert slider.value > 50.0

    def test_slider_left_key_decreases_value(self, slider_class):
        """Test left arrow decreases value."""
        slider = slider_class(value=50)
        slider.handle_focus_gained()
        slider.handle_key_down("left")
        assert slider.value < 50.0

    def test_slider_up_key_increases_value(self, slider_class):
        """Test up arrow increases value."""
        slider = slider_class(value=50)
        slider.handle_focus_gained()
        slider.handle_key_down("up")
        assert slider.value > 50.0

    def test_slider_down_key_decreases_value(self, slider_class):
        """Test down arrow decreases value."""
        slider = slider_class(value=50)
        slider.handle_focus_gained()
        slider.handle_key_down("down")
        assert slider.value < 50.0

    def test_slider_home_key_sets_min(self, slider_class):
        """Test home key sets to minimum."""
        slider = slider_class(value=50, min_value=10)
        slider.handle_focus_gained()
        slider.handle_key_down("home")
        assert slider.value == 10.0

    def test_slider_end_key_sets_max(self, slider_class):
        """Test end key sets to maximum."""
        slider = slider_class(value=50, max_value=90)
        slider.handle_focus_gained()
        slider.handle_key_down("end")
        assert slider.value == 90.0

    def test_slider_shift_increases_step(self, slider_class):
        """Test shift key increases step size."""
        slider = slider_class(value=50, step=5)
        slider.handle_focus_gained()

        slider.handle_key_down("right", shift=True)
        # With shift, should step by 10x
        assert slider.value == 100.0

    def test_slider_key_not_focused(self, slider_class):
        """Test keyboard is ignored when not focused."""
        slider = slider_class(value=50)
        result = slider.handle_key_down("right")
        assert result is False
        assert slider.value == 50.0


class TestSliderEvents:
    """Tests for slider events."""

    def test_slider_value_change_event(self, slider_class):
        """Test value change event is emitted."""
        slider = slider_class()
        handler = MagicMock()
        slider.on_value_change(handler)

        slider.value = 50.0

        assert handler.called
        event = handler.call_args[0][0]
        assert event.new_value == 50.0
        assert event.previous_value == 0.0
        assert event.is_user_action is False

    def test_slider_value_change_event_user_action(self, slider_class):
        """Test value change marks user action during drag."""
        slider = slider_class(x=0, y=0, width=200, height=30)
        handler = MagicMock()
        slider.on_value_change(handler)

        slider.handle_mouse_down(100, 15)

        assert handler.called
        event = handler.call_args[0][0]
        assert event.is_user_action is True
        assert event.is_dragging is True

    def test_slider_drag_start_event(self, slider_class):
        """Test drag start event is emitted."""
        slider = slider_class(x=0, y=0, width=200, height=30)
        handler = MagicMock()
        slider.on_drag_start(handler)

        slider.handle_mouse_down(100, 15)

        assert handler.called
        assert handler.call_args[0][0] is slider

    def test_slider_drag_end_event(self, slider_class):
        """Test drag end event is emitted."""
        slider = slider_class(x=0, y=0, width=200, height=30)
        handler = MagicMock()
        slider.on_drag_end(handler)

        slider.handle_mouse_down(100, 15)
        slider.handle_mouse_up(100, 15)

        assert handler.called
        assert handler.call_args[0][0] is slider

    def test_slider_unsubscribe_value_change(self, slider_class):
        """Test unsubscribing from value change events."""
        slider = slider_class()
        handler = MagicMock()

        unsubscribe = slider.on_value_change(handler)
        unsubscribe()

        slider.value = 50.0
        assert not handler.called

    def test_slider_no_event_when_unchanged(self, slider_class):
        """Test no event when value doesn't change."""
        slider = slider_class(value=50)
        handler = MagicMock()
        slider.on_value_change(handler)

        slider.value = 50.0  # Same value

        assert not handler.called


class TestSliderColors:
    """Tests for slider color based on state."""

    def test_slider_thumb_color_normal(self, slider_class):
        """Test thumb color in normal state."""
        from engine.ui.widgets.input.slider import SliderStyle

        style = SliderStyle(thumb_color="#123456")
        slider = slider_class(style=style)

        assert slider.get_current_thumb_color() == "#123456"

    def test_slider_thumb_color_hovered(self, slider_class):
        """Test thumb color in hovered state."""
        from engine.ui.widgets.input.slider import SliderStyle

        style = SliderStyle(thumb_hover_color="#ABCDEF")
        slider = slider_class(style=style)
        slider.handle_mouse_enter()

        assert slider.get_current_thumb_color() == "#ABCDEF"

    def test_slider_thumb_color_dragging(self, slider_class):
        """Test thumb color when dragging."""
        from engine.ui.widgets.input.slider import SliderStyle

        style = SliderStyle(thumb_active_color="#FEDCBA")
        slider = slider_class(x=0, y=0, width=200, height=30, style=style)
        slider.handle_mouse_down(100, 15)

        assert slider.get_current_thumb_color() == "#FEDCBA"

    def test_slider_thumb_color_disabled(self, slider_class):
        """Test thumb color when disabled."""
        from engine.ui.widgets.input.slider import SliderStyle

        style = SliderStyle(disabled_color="#999999")
        slider = slider_class(enabled=False, style=style)

        assert slider.get_current_thumb_color() == "#999999"


class TestSliderTicks:
    """Tests for tick marks."""

    def test_slider_tick_values_hidden(self, slider_class):
        """Test no tick values when ticks disabled."""
        from engine.ui.widgets.input.slider import SliderStyle

        style = SliderStyle(show_ticks=False)
        slider = slider_class(style=style)

        assert slider.get_tick_values() == []

    def test_slider_tick_values_shown(self, slider_class):
        """Test tick values when ticks enabled."""
        from engine.ui.widgets.input.slider import SliderStyle

        style = SliderStyle(show_ticks=True, tick_count=5)
        slider = slider_class(min_value=0, max_value=100, style=style)

        ticks = slider.get_tick_values()
        assert len(ticks) == 5
        assert ticks[0] == 0.0
        assert ticks[2] == 50.0
        assert ticks[4] == 100.0

    def test_slider_tick_values_min_count(self, slider_class):
        """Test tick values with minimum count."""
        from engine.ui.widgets.input.slider import SliderStyle

        style = SliderStyle(show_ticks=True, tick_count=1)
        slider = slider_class(style=style)

        # Less than 2 ticks returns empty
        assert slider.get_tick_values() == []


class TestSliderDirtyState:
    """Tests for dirty state tracking."""

    def test_slider_dirty_after_value_change(self, slider_class):
        """Test slider is dirty after value changes."""
        slider = slider_class()
        slider.mark_clean()
        slider.value = 50.0
        assert slider.is_dirty

    def test_slider_dirty_after_drag(self, slider_class):
        """Test slider is dirty during drag."""
        slider = slider_class(x=0, y=0, width=200, height=30)
        slider.mark_clean()
        slider.handle_mouse_down(100, 15)
        assert slider.is_dirty

    def test_slider_mark_clean(self, slider_class):
        """Test mark_clean clears dirty state."""
        slider = slider_class()
        slider.mark_clean()
        assert slider.is_dirty is False
