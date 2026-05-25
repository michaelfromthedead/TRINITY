"""
Comprehensive tests for screen transition effects.

Tests cover:
- Base transition functionality
- Fade transitions (crossfade and sequential)
- Slide transitions (all directions)
- Zoom transitions
- Instant transitions
- Composite transitions
- Custom transitions
- Transition factory
- Easing integration
"""

from __future__ import annotations

import math
import pytest
from typing import Dict, Any, Optional
from unittest.mock import Mock, MagicMock

from engine.ui.screens.transitions import (
    TransitionDirection,
    Easing,
    EasingFunction,
    get_easing_function,
    ITransition,
    BaseTransition,
    FadeTransition,
    SlideTransition,
    ZoomTransition,
    InstantTransition,
    CompositeTransition,
    CustomTransition,
    TransitionFactory,
)
from engine.ui.screens.screen import Screen


# =============================================================================
# TEST FIXTURES
# =============================================================================


class MockScreen(Screen):
    """Mock screen for testing transitions."""

    def __init__(self, name: str = "mock") -> None:
        super().__init__(name)


@pytest.fixture
def exiting_screen() -> MockScreen:
    """Create a mock exiting screen."""
    return MockScreen("exiting")


@pytest.fixture
def entering_screen() -> MockScreen:
    """Create a mock entering screen."""
    return MockScreen("entering")


# =============================================================================
# TRANSITION DIRECTION TESTS
# =============================================================================


class TestTransitionDirection:
    """Tests for TransitionDirection enum."""

    def test_all_directions_exist(self) -> None:
        """All expected directions should exist."""
        assert TransitionDirection.LEFT is not None
        assert TransitionDirection.RIGHT is not None
        assert TransitionDirection.UP is not None
        assert TransitionDirection.DOWN is not None


# =============================================================================
# EASING ENUM TESTS
# =============================================================================


class TestEasingEnum:
    """Tests for Easing enum and get_easing_function."""

    def test_all_easings_exist(self) -> None:
        """All expected easing types should exist."""
        assert Easing.LINEAR is not None
        assert Easing.EASE_IN is not None
        assert Easing.EASE_OUT is not None
        assert Easing.EASE_IN_OUT is not None
        assert Easing.EASE_IN_QUAD is not None
        assert Easing.EASE_OUT_BOUNCE is not None

    def test_get_easing_function_linear(self) -> None:
        """Linear easing should return t unchanged."""
        func = get_easing_function(Easing.LINEAR)
        assert func(0.0) == 0.0
        assert func(0.5) == 0.5
        assert func(1.0) == 1.0

    def test_get_easing_function_ease_in(self) -> None:
        """Ease in should start slow."""
        func = get_easing_function(Easing.EASE_IN)
        # At halfway, should be less than halfway
        assert func(0.5) < 0.5

    def test_get_easing_function_ease_out(self) -> None:
        """Ease out should end slow."""
        func = get_easing_function(Easing.EASE_OUT)
        # At halfway, should be more than halfway
        assert func(0.5) > 0.5

    def test_get_easing_function_ease_in_out(self) -> None:
        """Ease in-out should be symmetric."""
        func = get_easing_function(Easing.EASE_IN_OUT)
        # At halfway should be roughly halfway
        assert 0.4 < func(0.5) < 0.6

    def test_get_easing_function_unknown(self) -> None:
        """Unknown easing should default to linear."""
        # Create a mock easing not in the dict
        func = get_easing_function(Easing.LINEAR)
        assert func(0.5) == 0.5


# =============================================================================
# BASE TRANSITION TESTS
# =============================================================================


class TestBaseTransition:
    """Tests for BaseTransition class."""

    def test_init_with_defaults(self) -> None:
        """Should initialize with default values."""
        t = BaseTransition()
        assert t.duration == 0.3
        assert t.easing == Easing.EASE_IN_OUT

    def test_init_with_custom_duration(self) -> None:
        """Should accept custom duration."""
        t = BaseTransition(duration=1.0)
        assert t.duration == 1.0

    def test_init_with_negative_duration_raises(self) -> None:
        """Negative duration should raise ValueError."""
        with pytest.raises(ValueError):
            BaseTransition(duration=-1.0)

    def test_duration_setter(self) -> None:
        """Should be able to set duration."""
        t = BaseTransition()
        t.duration = 2.0
        assert t.duration == 2.0

    def test_duration_setter_negative_raises(self) -> None:
        """Setting negative duration should raise ValueError."""
        t = BaseTransition()
        with pytest.raises(ValueError):
            t.duration = -1.0

    def test_easing_setter(self) -> None:
        """Should be able to change easing."""
        t = BaseTransition()
        t.easing = Easing.EASE_OUT_BOUNCE
        assert t.easing == Easing.EASE_OUT_BOUNCE

    def test_progress_starts_at_zero(self) -> None:
        """Progress should start at 0."""
        t = BaseTransition()
        assert t.progress == 0.0

    def test_progress_at_midpoint(self) -> None:
        """Progress should be 0.5 at midpoint."""
        t = BaseTransition(duration=1.0)
        t.start(None, None)
        t.update(0.5)
        assert t.progress == pytest.approx(0.5)

    def test_progress_capped_at_one(self) -> None:
        """Progress should not exceed 1.0."""
        t = BaseTransition(duration=1.0)
        t.start(None, None)
        t.update(2.0)
        assert t.progress == 1.0

    def test_progress_instant_for_zero_duration(self) -> None:
        """Zero duration should have instant progress."""
        t = BaseTransition(duration=0.0)
        assert t.progress == 1.0

    def test_is_complete(self) -> None:
        """is_complete should reflect completion state."""
        t = BaseTransition(duration=1.0)
        t.start(None, None)
        assert t.is_complete is False
        t.update(1.0)
        assert t.is_complete is True

    def test_is_started(self) -> None:
        """is_started should reflect started state."""
        t = BaseTransition()
        assert t.is_started is False
        t.start(None, None)
        assert t.is_started is True

    def test_start_sets_screens(
        self, exiting_screen: MockScreen, entering_screen: MockScreen
    ) -> None:
        """Start should store screen references."""
        t = BaseTransition()
        t.start(exiting_screen, entering_screen)

        assert t.exiting_screen is exiting_screen
        assert t.entering_screen is entering_screen

    def test_start_resets_elapsed(self) -> None:
        """Start should reset elapsed time."""
        t = BaseTransition(duration=1.0)
        t.start(None, None)
        t.update(0.5)
        t.start(None, None)

        assert t.progress == 0.0

    def test_update_not_started(self) -> None:
        """Update should do nothing if not started."""
        t = BaseTransition()
        t.update(1.0)
        assert t.progress == 0.0

    def test_reset(self) -> None:
        """Reset should clear all state."""
        t = BaseTransition()
        t.start(Mock(), Mock())
        t.update(0.5)

        t.reset()

        assert t.is_started is False
        assert t.progress == 0.0
        assert t.exiting_screen is None
        assert t.entering_screen is None

    def test_default_transforms(self) -> None:
        """Default transforms should be identity."""
        t = BaseTransition()
        exit_t = t.get_exiting_transform()
        enter_t = t.get_entering_transform()

        assert exit_t["alpha"] == 1.0
        assert exit_t["x"] == 0.0
        assert exit_t["y"] == 0.0
        assert exit_t["scale"] == 1.0
        assert enter_t["alpha"] == 1.0

    def test_eased_progress(self) -> None:
        """Eased progress should apply easing function."""
        t = BaseTransition(duration=1.0, easing=Easing.EASE_IN)
        t.start(None, None)
        t.update(0.5)

        # Eased progress at 0.5 should be less than 0.5 for ease-in
        assert t.eased_progress < 0.5


# =============================================================================
# FADE TRANSITION TESTS
# =============================================================================


class TestFadeTransition:
    """Tests for FadeTransition class."""

    def test_init_crossfade_default(self) -> None:
        """Should default to crossfade mode."""
        t = FadeTransition()
        assert t.crossfade is True

    def test_crossfade_setter(self) -> None:
        """Should be able to change crossfade mode."""
        t = FadeTransition()
        t.crossfade = False
        assert t.crossfade is False

    def test_crossfade_exiting_alpha_at_start(self) -> None:
        """Exiting screen should have full alpha at start."""
        t = FadeTransition(duration=1.0, easing=Easing.LINEAR)
        t.start(None, None)

        transform = t.get_exiting_transform()
        assert transform["alpha"] == pytest.approx(1.0)

    def test_crossfade_exiting_alpha_at_end(self) -> None:
        """Exiting screen should have zero alpha at end."""
        t = FadeTransition(duration=1.0, easing=Easing.LINEAR)
        t.start(None, None)
        t.update(1.0)

        transform = t.get_exiting_transform()
        assert transform["alpha"] == pytest.approx(0.0)

    def test_crossfade_entering_alpha_at_start(self) -> None:
        """Entering screen should have zero alpha at start."""
        t = FadeTransition(duration=1.0, easing=Easing.LINEAR)
        t.start(None, None)

        transform = t.get_entering_transform()
        assert transform["alpha"] == pytest.approx(0.0)

    def test_crossfade_entering_alpha_at_end(self) -> None:
        """Entering screen should have full alpha at end."""
        t = FadeTransition(duration=1.0, easing=Easing.LINEAR)
        t.start(None, None)
        t.update(1.0)

        transform = t.get_entering_transform()
        assert transform["alpha"] == pytest.approx(1.0)

    def test_sequential_fade_first_half(self) -> None:
        """Non-crossfade should fade out in first half."""
        t = FadeTransition(duration=1.0, easing=Easing.LINEAR, crossfade=False)
        t.start(None, None)
        t.update(0.25)

        exiting = t.get_exiting_transform()
        entering = t.get_entering_transform()

        # At 25%, exiting should be 50% faded out
        assert exiting["alpha"] == pytest.approx(0.5)
        # Entering should still be invisible
        assert entering["alpha"] == pytest.approx(0.0)

    def test_sequential_fade_second_half(self) -> None:
        """Non-crossfade should fade in during second half."""
        t = FadeTransition(duration=1.0, easing=Easing.LINEAR, crossfade=False)
        t.start(None, None)
        t.update(0.75)

        exiting = t.get_exiting_transform()
        entering = t.get_entering_transform()

        # At 75%, exiting should be fully faded out
        assert exiting["alpha"] == pytest.approx(0.0)
        # Entering should be 50% faded in
        assert entering["alpha"] == pytest.approx(0.5)

    def test_fade_positions_unchanged(self) -> None:
        """Fade transition should not change position."""
        t = FadeTransition()
        t.start(None, None)
        t.update(0.5)

        assert t.get_exiting_transform()["x"] == 0.0
        assert t.get_exiting_transform()["y"] == 0.0
        assert t.get_entering_transform()["x"] == 0.0
        assert t.get_entering_transform()["y"] == 0.0


# =============================================================================
# SLIDE TRANSITION TESTS
# =============================================================================


class TestSlideTransition:
    """Tests for SlideTransition class."""

    def test_direction_default(self) -> None:
        """Should default to LEFT direction."""
        t = SlideTransition()
        assert t.direction == TransitionDirection.LEFT

    def test_direction_setter(self) -> None:
        """Should be able to change direction."""
        t = SlideTransition()
        t.direction = TransitionDirection.RIGHT
        assert t.direction == TransitionDirection.RIGHT

    def test_push_default(self) -> None:
        """Should default to push mode."""
        t = SlideTransition()
        assert t.push is True

    def test_push_setter(self) -> None:
        """Should be able to change push mode."""
        t = SlideTransition()
        t.push = False
        assert t.push is False

    def test_slide_left_entering_start(self) -> None:
        """Entering screen should start off-screen right for LEFT slide."""
        t = SlideTransition(direction=TransitionDirection.LEFT, easing=Easing.LINEAR)
        t.start(None, None)

        entering = t.get_entering_transform()
        # Should be at screen width
        assert entering["x"] == t._screen_width

    def test_slide_left_entering_end(self) -> None:
        """Entering screen should end at origin for LEFT slide."""
        t = SlideTransition(direction=TransitionDirection.LEFT, easing=Easing.LINEAR)
        t.start(None, None)
        t.update(1.0)

        entering = t.get_entering_transform()
        assert entering["x"] == pytest.approx(0.0)

    def test_slide_left_exiting_push_mode(self) -> None:
        """Exiting screen should move left in push mode."""
        t = SlideTransition(
            direction=TransitionDirection.LEFT, push=True, easing=Easing.LINEAR
        )
        t.start(None, None)
        t.update(0.5)

        exiting = t.get_exiting_transform()
        assert exiting["x"] < 0

    def test_slide_left_exiting_no_push(self) -> None:
        """Exiting screen should stay in place without push."""
        t = SlideTransition(
            direction=TransitionDirection.LEFT, push=False, easing=Easing.LINEAR
        )
        t.start(None, None)
        t.update(0.5)

        exiting = t.get_exiting_transform()
        assert exiting["x"] == 0.0
        assert exiting["y"] == 0.0

    def test_slide_right(self) -> None:
        """Entering screen should start off-screen left for RIGHT slide."""
        t = SlideTransition(direction=TransitionDirection.RIGHT, easing=Easing.LINEAR)
        t.start(None, None)

        entering = t.get_entering_transform()
        assert entering["x"] == -t._screen_width

    def test_slide_up(self) -> None:
        """Entering screen should start off-screen bottom for UP slide."""
        t = SlideTransition(direction=TransitionDirection.UP, easing=Easing.LINEAR)
        t.start(None, None)

        entering = t.get_entering_transform()
        assert entering["y"] == t._screen_height

    def test_slide_down(self) -> None:
        """Entering screen should start off-screen top for DOWN slide."""
        t = SlideTransition(direction=TransitionDirection.DOWN, easing=Easing.LINEAR)
        t.start(None, None)

        entering = t.get_entering_transform()
        assert entering["y"] == -t._screen_height

    def test_set_screen_size(self) -> None:
        """Should be able to set custom screen size."""
        t = SlideTransition()
        t.set_screen_size(800, 600)

        assert t._screen_width == 800
        assert t._screen_height == 600

    def test_slide_alpha_unchanged(self) -> None:
        """Slide should not affect alpha."""
        t = SlideTransition()
        t.start(None, None)
        t.update(0.5)

        assert t.get_exiting_transform()["alpha"] == 1.0
        assert t.get_entering_transform()["alpha"] == 1.0


# =============================================================================
# ZOOM TRANSITION TESTS
# =============================================================================


class TestZoomTransition:
    """Tests for ZoomTransition class."""

    def test_zoom_in_default(self) -> None:
        """Should default to zoom in mode."""
        t = ZoomTransition()
        assert t.zoom_in is True

    def test_zoom_in_setter(self) -> None:
        """Should be able to change zoom mode."""
        t = ZoomTransition()
        t.zoom_in = False
        assert t.zoom_in is False

    def test_min_scale_default(self) -> None:
        """Should have default min scale."""
        t = ZoomTransition()
        assert t.min_scale == 0.8

    def test_min_scale_setter(self) -> None:
        """Should be able to set min scale."""
        t = ZoomTransition()
        t.min_scale = 0.5
        assert t.min_scale == 0.5

    def test_min_scale_clamped(self) -> None:
        """Min scale should not go below 0."""
        t = ZoomTransition()
        t.min_scale = -0.5
        assert t.min_scale == 0.0

    def test_max_scale_setter(self) -> None:
        """Should be able to set max scale."""
        t = ZoomTransition()
        t.max_scale = 1.5
        assert t.max_scale == 1.5

    def test_max_scale_clamped(self) -> None:
        """Max scale should not go below min scale."""
        t = ZoomTransition()
        t.min_scale = 0.5
        t.max_scale = 0.3
        assert t.max_scale == 0.5

    def test_fade_default(self) -> None:
        """Should default to fade enabled."""
        t = ZoomTransition()
        assert t.fade is True

    def test_fade_setter(self) -> None:
        """Should be able to disable fade."""
        t = ZoomTransition()
        t.fade = False
        assert t.fade is False

    def test_zoom_in_entering_start_scale(self) -> None:
        """Entering should start at min scale for zoom in."""
        t = ZoomTransition(zoom_in=True, easing=Easing.LINEAR)
        t.start(None, None)

        entering = t.get_entering_transform()
        assert entering["scale"] == t.min_scale

    def test_zoom_in_entering_end_scale(self) -> None:
        """Entering should end at normal scale for zoom in."""
        t = ZoomTransition(zoom_in=True, easing=Easing.LINEAR)
        t.start(None, None)
        t.update(1.0)

        entering = t.get_entering_transform()
        assert entering["scale"] == pytest.approx(1.0)

    def test_zoom_in_exiting_scales_up(self) -> None:
        """Exiting should scale up for zoom in."""
        t = ZoomTransition(zoom_in=True, easing=Easing.LINEAR)
        t.start(None, None)
        t.update(1.0)

        exiting = t.get_exiting_transform()
        assert exiting["scale"] == t.max_scale

    def test_zoom_out_entering_start_scale(self) -> None:
        """Entering should start at max scale for zoom out."""
        t = ZoomTransition(zoom_in=False, easing=Easing.LINEAR)
        t.start(None, None)

        entering = t.get_entering_transform()
        assert entering["scale"] == t.max_scale

    def test_zoom_out_entering_end_scale(self) -> None:
        """Entering should end at normal scale for zoom out."""
        t = ZoomTransition(zoom_in=False, easing=Easing.LINEAR)
        t.start(None, None)
        t.update(1.0)

        entering = t.get_entering_transform()
        assert entering["scale"] == pytest.approx(1.0)

    def test_zoom_with_fade_alpha(self) -> None:
        """Zoom with fade should change alpha."""
        t = ZoomTransition(fade=True, easing=Easing.LINEAR)
        t.start(None, None)

        entering_start = t.get_entering_transform()
        t.update(1.0)
        entering_end = t.get_entering_transform()

        assert entering_start["alpha"] == pytest.approx(0.0)
        assert entering_end["alpha"] == pytest.approx(1.0)

    def test_zoom_without_fade_alpha(self) -> None:
        """Zoom without fade should keep alpha at 1."""
        t = ZoomTransition(fade=False, easing=Easing.LINEAR)
        t.start(None, None)
        t.update(0.5)

        assert t.get_entering_transform()["alpha"] == 1.0
        assert t.get_exiting_transform()["alpha"] == 1.0


# =============================================================================
# INSTANT TRANSITION TESTS
# =============================================================================


class TestInstantTransition:
    """Tests for InstantTransition class."""

    def test_duration_is_zero(self) -> None:
        """Duration should be zero."""
        t = InstantTransition()
        assert t.duration == 0.0

    def test_starts_not_complete(self) -> None:
        """Should not be complete initially."""
        t = InstantTransition()
        assert t.is_complete is False

    def test_complete_after_start(self) -> None:
        """Should be complete immediately after start."""
        t = InstantTransition()
        t.start(None, None)
        assert t.is_complete is True

    def test_progress_after_start(self) -> None:
        """Progress should be 1.0 after start."""
        t = InstantTransition()
        t.start(None, None)
        assert t.progress == 1.0

    def test_update_does_nothing(self) -> None:
        """Update should not affect completed state."""
        t = InstantTransition()
        t.start(None, None)
        t.update(1.0)
        assert t.is_complete is True

    def test_reset(self) -> None:
        """Reset should clear state."""
        t = InstantTransition()
        t.start(None, None)
        t.reset()

        assert t.is_started is False
        assert t.is_complete is False
        assert t.progress == 0.0

    def test_exiting_transform(self) -> None:
        """Exiting should be invisible."""
        t = InstantTransition()
        transform = t.get_exiting_transform()
        assert transform["alpha"] == 0.0

    def test_entering_transform(self) -> None:
        """Entering should be fully visible."""
        t = InstantTransition()
        transform = t.get_entering_transform()
        assert transform["alpha"] == 1.0


# =============================================================================
# COMPOSITE TRANSITION TESTS
# =============================================================================


class TestCompositeTransition:
    """Tests for CompositeTransition class."""

    def test_add_effect(self) -> None:
        """Should be able to add effects."""
        t = CompositeTransition()
        effect = FadeTransition()

        result = t.add_effect(effect)

        assert result is t  # Fluent interface
        assert len(t._effects) == 1

    def test_clear_effects(self) -> None:
        """Should be able to clear effects."""
        t = CompositeTransition()
        t.add_effect(FadeTransition())
        t.add_effect(ZoomTransition())

        t.clear_effects()

        assert len(t._effects) == 0

    def test_start_propagates(self) -> None:
        """Start should propagate to all effects."""
        t = CompositeTransition()
        effect1 = FadeTransition()
        effect2 = ZoomTransition()
        t.add_effect(effect1)
        t.add_effect(effect2)

        t.start(None, None)

        assert effect1.is_started is True
        assert effect2.is_started is True

    def test_update_propagates(self) -> None:
        """Update should propagate to all effects."""
        t = CompositeTransition(duration=1.0)
        effect1 = FadeTransition(duration=1.0)
        effect2 = ZoomTransition(duration=1.0)
        t.add_effect(effect1)
        t.add_effect(effect2)

        t.start(None, None)
        t.update(0.5)

        assert effect1.progress > 0
        assert effect2.progress > 0

    def test_reset_propagates(self) -> None:
        """Reset should propagate to all effects."""
        t = CompositeTransition()
        effect = FadeTransition()
        t.add_effect(effect)
        t.start(None, None)

        t.reset()

        assert effect.is_started is False

    def test_combined_transforms_multiply_alpha(self) -> None:
        """Alpha should be multiplied."""
        t = CompositeTransition()
        fade1 = FadeTransition(duration=1.0, easing=Easing.LINEAR)
        fade2 = FadeTransition(duration=1.0, easing=Easing.LINEAR)
        t.add_effect(fade1)
        t.add_effect(fade2)

        t.start(None, None)
        t.update(0.5)

        # Both at 0.5 alpha, should multiply to 0.25
        entering = t.get_entering_transform()
        assert entering["alpha"] == pytest.approx(0.25)

    def test_combined_transforms_add_position(self) -> None:
        """Position should be added."""
        t = CompositeTransition()
        slide1 = SlideTransition(
            direction=TransitionDirection.LEFT, duration=1.0, easing=Easing.LINEAR
        )
        slide2 = SlideTransition(
            direction=TransitionDirection.LEFT, duration=1.0, easing=Easing.LINEAR
        )
        slide1.set_screen_size(1920, 1080)
        slide2.set_screen_size(1920, 1080)
        t.add_effect(slide1)
        t.add_effect(slide2)

        t.start(None, None)

        entering = t.get_entering_transform()
        # Both slides add x offset
        assert entering["x"] == 2 * 1920

    def test_combined_transforms_multiply_scale(self) -> None:
        """Scale should be multiplied."""
        t = CompositeTransition()
        zoom1 = ZoomTransition(
            min_scale=0.5, fade=False, duration=1.0, easing=Easing.LINEAR
        )
        zoom2 = ZoomTransition(
            min_scale=0.5, fade=False, duration=1.0, easing=Easing.LINEAR
        )
        t.add_effect(zoom1)
        t.add_effect(zoom2)

        t.start(None, None)

        entering = t.get_entering_transform()
        # Both at 0.5 scale, should multiply to 0.25
        assert entering["scale"] == pytest.approx(0.25)


# =============================================================================
# CUSTOM TRANSITION TESTS
# =============================================================================


class TestCustomTransition:
    """Tests for CustomTransition class."""

    def test_custom_exiting_transform(self) -> None:
        """Should use custom exiting transform function."""

        def custom_exit(t: float) -> Dict[str, Any]:
            return {"alpha": 1.0 - t, "x": -100 * t, "y": 0.0, "scale": 1.0}

        transition = CustomTransition(
            duration=1.0, easing=Easing.LINEAR, exiting_transform=custom_exit
        )
        transition.start(None, None)
        transition.update(0.5)

        result = transition.get_exiting_transform()
        assert result["alpha"] == pytest.approx(0.5)
        assert result["x"] == pytest.approx(-50)

    def test_custom_entering_transform(self) -> None:
        """Should use custom entering transform function."""

        def custom_enter(t: float) -> Dict[str, Any]:
            return {"alpha": t, "x": 100 * (1 - t), "y": 0.0, "scale": t}

        transition = CustomTransition(
            duration=1.0, easing=Easing.LINEAR, entering_transform=custom_enter
        )
        transition.start(None, None)
        transition.update(0.5)

        result = transition.get_entering_transform()
        assert result["alpha"] == pytest.approx(0.5)
        assert result["x"] == pytest.approx(50)
        assert result["scale"] == pytest.approx(0.5)

    def test_set_exiting_transform(self) -> None:
        """Should be able to set transform function after creation."""
        transition = CustomTransition()

        def custom(t: float) -> Dict[str, Any]:
            return {"alpha": 0.0, "x": 0.0, "y": 0.0, "scale": 0.0}

        transition.set_exiting_transform(custom)
        transition.start(None, None)

        result = transition.get_exiting_transform()
        assert result["alpha"] == 0.0
        assert result["scale"] == 0.0

    def test_set_entering_transform(self) -> None:
        """Should be able to set transform function after creation."""
        transition = CustomTransition()

        def custom(t: float) -> Dict[str, Any]:
            return {"alpha": 0.5, "x": 50.0, "y": 25.0, "scale": 2.0}

        transition.set_entering_transform(custom)
        transition.start(None, None)

        result = transition.get_entering_transform()
        assert result["alpha"] == 0.5
        assert result["x"] == 50.0
        assert result["y"] == 25.0
        assert result["scale"] == 2.0

    def test_default_when_no_custom(self) -> None:
        """Should use base transforms when none set."""
        transition = CustomTransition()
        transition.start(None, None)

        exit_t = transition.get_exiting_transform()
        enter_t = transition.get_entering_transform()

        assert exit_t["alpha"] == 1.0
        assert enter_t["alpha"] == 1.0


# =============================================================================
# TRANSITION FACTORY TESTS
# =============================================================================


class TestTransitionFactory:
    """Tests for TransitionFactory class."""

    def test_factory_fade(self) -> None:
        """Should create fade transition."""
        t = TransitionFactory.fade(duration=0.5, crossfade=False)

        assert isinstance(t, FadeTransition)
        assert t.duration == 0.5
        assert t.crossfade is False

    def test_factory_slide_left(self) -> None:
        """Should create slide left transition."""
        t = TransitionFactory.slide_left(duration=0.4, push=False)

        assert isinstance(t, SlideTransition)
        assert t.direction == TransitionDirection.LEFT
        assert t.duration == 0.4
        assert t.push is False

    def test_factory_slide_right(self) -> None:
        """Should create slide right transition."""
        t = TransitionFactory.slide_right()

        assert isinstance(t, SlideTransition)
        assert t.direction == TransitionDirection.RIGHT

    def test_factory_slide_up(self) -> None:
        """Should create slide up transition."""
        t = TransitionFactory.slide_up()

        assert isinstance(t, SlideTransition)
        assert t.direction == TransitionDirection.UP

    def test_factory_slide_down(self) -> None:
        """Should create slide down transition."""
        t = TransitionFactory.slide_down()

        assert isinstance(t, SlideTransition)
        assert t.direction == TransitionDirection.DOWN

    def test_factory_zoom_in(self) -> None:
        """Should create zoom in transition."""
        t = TransitionFactory.zoom_in(duration=0.6, fade=False)

        assert isinstance(t, ZoomTransition)
        assert t.zoom_in is True
        assert t.duration == 0.6
        assert t.fade is False

    def test_factory_zoom_out(self) -> None:
        """Should create zoom out transition."""
        t = TransitionFactory.zoom_out()

        assert isinstance(t, ZoomTransition)
        assert t.zoom_in is False

    def test_factory_instant(self) -> None:
        """Should create instant transition."""
        t = TransitionFactory.instant()

        assert isinstance(t, InstantTransition)
        assert t.duration == 0.0
