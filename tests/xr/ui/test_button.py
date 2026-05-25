"""
Tests for XR Button (button.py).

Tests the XRButton component and related classes:
    XRButton, XRButtonState, HapticFeedback, @xr_button decorator

Each test verifies:
1. Button creation and configuration
2. State transitions (hover, press, release)
3. Haptic feedback generation
4. Decorator application
"""

import sys
from pathlib import Path

import pytest

# Add engine to path for direct imports
engine_path = Path(__file__).parents[3]
if str(engine_path) not in sys.path:
    sys.path.insert(0, str(engine_path))

from engine.xr.ui.button import (
    XRButton,
    XRButtonState,
    XRButtonStyle,
    HapticFeedback,
    xr_button,
    XRButtonGroup,
)


# =============================================================================
# HapticFeedback
# =============================================================================


class TestHapticFeedback:
    def test_default_values(self):
        feedback = HapticFeedback()
        assert feedback.amplitude == 0.5
        assert feedback.duration_ms == 50
        assert feedback.frequency == 200.0

    def test_custom_values(self):
        feedback = HapticFeedback(
            amplitude=0.8,
            duration_ms=100,
            frequency=300.0
        )
        assert feedback.amplitude == 0.8
        assert feedback.duration_ms == 100
        assert feedback.frequency == 300.0

    def test_amplitude_clamping(self):
        feedback = HapticFeedback(amplitude=1.5)
        assert feedback.amplitude == 1.0

        feedback = HapticFeedback(amplitude=-0.5)
        assert feedback.amplitude == 0.0

    def test_duration_clamping(self):
        feedback = HapticFeedback(duration_ms=-10)
        assert feedback.duration_ms == 0

    def test_frequency_clamping(self):
        feedback = HapticFeedback(frequency=-100)
        assert feedback.frequency == 0.0


# =============================================================================
# XRButtonStyle
# =============================================================================


class TestXRButtonStyle:
    def test_default_colors(self):
        style = XRButtonStyle()
        assert len(style.normal_color) == 4
        assert len(style.hover_color) == 4
        assert len(style.pressed_color) == 4
        assert len(style.disabled_color) == 4

    def test_custom_colors(self):
        style = XRButtonStyle(
            normal_color=(1.0, 0.0, 0.0, 1.0),
            hover_color=(0.0, 1.0, 0.0, 1.0),
        )
        assert style.normal_color == (1.0, 0.0, 0.0, 1.0)
        assert style.hover_color == (0.0, 1.0, 0.0, 1.0)


# =============================================================================
# XRButton
# =============================================================================


class TestXRButton:
    def test_default_creation(self):
        button = XRButton()
        assert button.label == ""
        assert button.state == XRButtonState.NORMAL
        assert button.is_hovered is False
        assert button.is_pressed is False
        assert button.haptic_on_press is True

    def test_with_label(self):
        button = XRButton(label="Start")
        assert button.label == "Start"

    def test_dimensions(self):
        button = XRButton(width=0.2, height=0.1)
        assert button.width == 0.2
        assert button.height == 0.1

    def test_is_enabled(self):
        button = XRButton()
        assert button.is_enabled is True

        button.disable()
        assert button.is_enabled is False

        button.enable()
        assert button.is_enabled is True

    def test_disable_clears_state(self):
        button = XRButton()
        button.is_hovered = True
        button.is_pressed = True
        button.press_depth = 0.01

        button.disable()

        assert button.is_hovered is False
        assert button.is_pressed is False
        assert button.press_depth == 0.0
        assert button.state == XRButtonState.DISABLED

    def test_current_color_normal(self):
        button = XRButton()
        assert button.current_color == button.style.normal_color

    def test_current_color_hovered(self):
        button = XRButton()
        button.is_hovered = True
        assert button.current_color == button.style.hover_color

    def test_current_color_pressed(self):
        button = XRButton()
        button.is_pressed = True
        assert button.current_color == button.style.pressed_color

    def test_current_color_disabled(self):
        button = XRButton()
        button.disable()
        assert button.current_color == button.style.disabled_color

    def test_visual_depth(self):
        button = XRButton(max_press_depth=0.02)
        assert button.visual_depth == 0.0

        button.is_pressed = True
        button.press_depth = 0.01
        assert button.visual_depth == -0.01

    def test_visual_depth_clamped(self):
        button = XRButton(max_press_depth=0.02)
        button.is_pressed = True
        button.press_depth = 0.05  # Exceeds max
        assert button.visual_depth == -0.02

    def test_set_label(self):
        button = XRButton()
        button.set_label("New Label")
        assert button.label == "New Label"

    def test_hover_begin(self):
        button = XRButton()
        haptic = button.hover_begin(interactor_id=1)

        assert button.is_hovered is True
        assert button._interactor_id == 1
        assert button.state == XRButtonState.HOVERED
        assert haptic is not None  # Light haptic feedback

    def test_hover_begin_when_disabled(self):
        button = XRButton()
        button.disable()
        haptic = button.hover_begin(interactor_id=1)

        assert button.is_hovered is False
        assert haptic is None

    def test_hover_end(self):
        button = XRButton()
        button.hover_begin(interactor_id=1)
        button.hover_end()

        assert button.is_hovered is False
        assert button.is_pressed is False
        assert button._interactor_id is None
        assert button.state == XRButtonState.NORMAL

    def test_press_begin(self):
        button = XRButton()
        haptic = button.press_begin(interactor_id=1)

        assert button.is_pressed is True
        assert button.state == XRButtonState.PRESSED
        assert haptic is not None

    def test_press_begin_no_haptic(self):
        button = XRButton(haptic_on_press=False)
        haptic = button.press_begin(interactor_id=1)

        assert button.is_pressed is True
        assert haptic is None

    def test_press_update(self):
        button = XRButton(press_threshold=0.015)

        # Below threshold
        haptic = button.press_update(0.01)
        assert button.is_pressed is False
        assert haptic is None

        # Above threshold
        haptic = button.press_update(0.02)
        assert button.is_pressed is True
        assert haptic is not None

    def test_press_end_triggers_click(self):
        button = XRButton()
        clicked = False

        def on_click():
            nonlocal clicked
            clicked = True

        button.on_click(on_click)
        button.press_begin(interactor_id=1)
        was_clicked, haptic = button.press_end()

        assert was_clicked is True
        assert clicked is True

    def test_press_end_no_click_when_disabled(self):
        button = XRButton()
        clicked = False

        def on_click():
            nonlocal clicked
            clicked = True

        button.on_click(on_click)
        button.press_begin(interactor_id=1)
        button.disable()
        was_clicked, haptic = button.press_end()

        assert was_clicked is False
        assert clicked is False

    def test_hover_callbacks(self):
        button = XRButton()
        entered = False
        exited = False

        def on_enter():
            nonlocal entered
            entered = True

        def on_exit():
            nonlocal exited
            exited = True

        button.on_hover_enter(on_enter)
        button.on_hover_exit(on_exit)

        button.hover_begin(1)
        assert entered is True

        button.hover_end()
        assert exited is True

    def test_hit_test(self):
        button = XRButton(
            position=(0.5, 0.5),
            width=0.1,
            height=0.05
        )

        # Center of button
        assert button.hit_test(0.5, 0.5) is True

        # Edge of button
        assert button.hit_test(0.45, 0.5) is True

        # Outside button
        assert button.hit_test(0.0, 0.0) is False


# =============================================================================
# @xr_button decorator
# =============================================================================


class TestXRButtonDecorator:
    def test_basic_application(self):
        @xr_button()
        class TestButton:
            pass

        assert TestButton._xr_button is True

    def test_with_label(self):
        @xr_button(label="Play")
        class TestButton:
            pass

        assert TestButton._button_label == "Play"

    def test_with_dimensions(self):
        @xr_button(width=0.2, height=0.1)
        class TestButton:
            pass

        assert TestButton._button_width == 0.2
        assert TestButton._button_height == 0.1

    def test_haptic_setting(self):
        @xr_button(haptic=False)
        class TestButton:
            pass

        assert TestButton._button_haptic is False

    def test_press_depth(self):
        @xr_button(press_depth=0.03)
        class TestButton:
            pass

        assert TestButton._button_press_depth == 0.03

    def test_invalid_width(self):
        with pytest.raises(ValueError, match="Width must be positive"):
            @xr_button(width=0)
            class TestButton:
                pass

    def test_invalid_height(self):
        with pytest.raises(ValueError, match="Height must be positive"):
            @xr_button(height=-0.1)
            class TestButton:
                pass

    def test_invalid_press_depth(self):
        with pytest.raises(ValueError, match="Press depth must be non-negative"):
            @xr_button(press_depth=-0.01)
            class TestButton:
                pass

    def test_tags(self):
        @xr_button(label="Test", haptic=True)
        class TestButton:
            pass

        assert TestButton._tags["xr_button"] is True
        assert TestButton._tags["button_label"] == "Test"
        assert TestButton._tags["button_haptic"] is True

    def test_applied_decorators(self):
        @xr_button()
        class TestButton:
            pass

        assert "xr_button" in TestButton._applied_decorators

    def test_registries(self):
        @xr_button()
        class TestButton:
            pass

        assert "xr" in TestButton._registries


# =============================================================================
# XRButtonGroup
# =============================================================================


class TestXRButtonGroup:
    def test_creation(self):
        group = XRButtonGroup()
        assert len(group.buttons) == 0

    def test_add_button(self):
        group = XRButtonGroup()
        button = XRButton(label="Test")

        group.add(button)
        assert button in group.buttons

    def test_remove_button(self):
        group = XRButtonGroup()
        button = XRButton(label="Test")

        group.add(button)
        group.remove(button)
        assert button not in group.buttons

    def test_single_selection(self):
        group = XRButtonGroup(selection_mode="single")
        button1 = XRButton(label="One")
        button2 = XRButton(label="Two")

        group.add(button1)
        group.add(button2)
        group.select(0)

        assert group.selected == button1

        group.select(1)
        assert group.selected == button2

    def test_enable_disable_all(self):
        group = XRButtonGroup()
        button1 = XRButton(label="One")
        button2 = XRButton(label="Two")

        group.add(button1)
        group.add(button2)

        group.disable_all()
        assert button1.is_enabled is False
        assert button2.is_enabled is False

        group.enable_all()
        assert button1.is_enabled is True
        assert button2.is_enabled is True
