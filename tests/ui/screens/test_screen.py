"""
Comprehensive tests for the Screen base class and lifecycle management.

Tests cover:
- Screen initialization and configuration
- Lifecycle state transitions
- Widget management
- Callback registration and invocation
- Screen parameters and results
- Edge cases and error handling
"""

from __future__ import annotations

import pytest
from typing import List, Tuple, Optional, Any
from unittest.mock import Mock, MagicMock

from engine.ui.screens.screen import (
    Screen,
    ScreenState,
    ScreenParams,
    ScreenResult,
    LifecycleCallback,
    StateChangeCallback,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


class ConcreteScreen(Screen):
    """Concrete implementation of Screen for testing."""

    def __init__(self, name: str = "test_screen") -> None:
        super().__init__(name)
        self.lifecycle_calls: List[str] = []
        self.update_deltas: List[float] = []

    def on_create(self) -> None:
        self.lifecycle_calls.append("on_create")

    def on_destroy(self) -> None:
        self.lifecycle_calls.append("on_destroy")

    def on_enter(self) -> None:
        self.lifecycle_calls.append("on_enter")

    def on_enter_complete(self) -> None:
        self.lifecycle_calls.append("on_enter_complete")

    def on_exit(self) -> None:
        self.lifecycle_calls.append("on_exit")

    def on_exit_complete(self) -> None:
        self.lifecycle_calls.append("on_exit_complete")

    def on_pause(self) -> None:
        self.lifecycle_calls.append("on_pause")

    def on_resume(self) -> None:
        self.lifecycle_calls.append("on_resume")

    def update(self, delta_time: float) -> None:
        self.update_deltas.append(delta_time)


@pytest.fixture
def screen() -> ConcreteScreen:
    """Create a fresh screen for each test."""
    return ConcreteScreen("test_screen")


@pytest.fixture
def params() -> ScreenParams:
    """Create screen parameters for testing."""
    return ScreenParams(data={"key": "value", "count": 42})


# =============================================================================
# SCREEN INITIALIZATION TESTS
# =============================================================================


class TestScreenInitialization:
    """Tests for screen initialization."""

    def test_screen_name_is_set(self, screen: ConcreteScreen) -> None:
        """Screen should have the correct name."""
        assert screen.name == "test_screen"

    def test_screen_initial_state_is_entering(self, screen: ConcreteScreen) -> None:
        """Screen should start in ENTERING state."""
        assert screen.state == ScreenState.ENTERING

    def test_screen_has_empty_params(self, screen: ConcreteScreen) -> None:
        """Screen should have empty params initially."""
        assert screen.params.data == {}

    def test_screen_has_no_result(self, screen: ConcreteScreen) -> None:
        """Screen should have no result initially."""
        assert screen.result is None

    def test_screen_has_no_stack(self, screen: ConcreteScreen) -> None:
        """Screen should not be attached to a stack initially."""
        assert screen.stack is None

    def test_screen_has_no_root_widget(self, screen: ConcreteScreen) -> None:
        """Screen should have no root widget initially."""
        assert screen.root_widget is None

    def test_screen_default_configuration(self, screen: ConcreteScreen) -> None:
        """Screen should have expected default configuration."""
        assert screen.is_modal is False
        assert screen.blocks_input is True
        assert screen.is_overlay is False
        assert screen.can_go_back is True
        assert screen.pause_below is True


class TestScreenConfiguration:
    """Tests for screen configuration properties."""

    def test_set_is_modal(self, screen: ConcreteScreen) -> None:
        """Should be able to set modal flag."""
        screen.is_modal = True
        assert screen.is_modal is True

    def test_set_blocks_input(self, screen: ConcreteScreen) -> None:
        """Should be able to set blocks_input flag."""
        screen.blocks_input = False
        assert screen.blocks_input is False

    def test_set_is_overlay(self, screen: ConcreteScreen) -> None:
        """Should be able to set overlay flag."""
        screen.is_overlay = True
        assert screen.is_overlay is True

    def test_set_can_go_back(self, screen: ConcreteScreen) -> None:
        """Should be able to set can_go_back flag."""
        screen.can_go_back = False
        assert screen.can_go_back is False

    def test_set_pause_below(self, screen: ConcreteScreen) -> None:
        """Should be able to set pause_below flag."""
        screen.pause_below = False
        assert screen.pause_below is False


# =============================================================================
# SCREEN STATE PROPERTY TESTS
# =============================================================================


class TestScreenStateProperties:
    """Tests for screen state convenience properties."""

    def test_is_active_when_active(self, screen: ConcreteScreen) -> None:
        """is_active should be True when state is ACTIVE."""
        screen._state = ScreenState.ACTIVE
        assert screen.is_active is True

    def test_is_active_when_not_active(self, screen: ConcreteScreen) -> None:
        """is_active should be False when state is not ACTIVE."""
        screen._state = ScreenState.PAUSED
        assert screen.is_active is False

    def test_is_entering_when_entering(self, screen: ConcreteScreen) -> None:
        """is_entering should be True when state is ENTERING."""
        screen._state = ScreenState.ENTERING
        assert screen.is_entering is True

    def test_is_exiting_when_exiting(self, screen: ConcreteScreen) -> None:
        """is_exiting should be True when state is EXITING."""
        screen._state = ScreenState.EXITING
        assert screen.is_exiting is True

    def test_is_paused_when_paused(self, screen: ConcreteScreen) -> None:
        """is_paused should be True when state is PAUSED."""
        screen._state = ScreenState.PAUSED
        assert screen.is_paused is True

    def test_is_destroyed_when_destroyed(self, screen: ConcreteScreen) -> None:
        """is_destroyed should be True when state is DESTROYED."""
        screen._state = ScreenState.DESTROYED
        assert screen.is_destroyed is True


# =============================================================================
# LIFECYCLE TRANSITION TESTS
# =============================================================================


class TestLifecycleTransitions:
    """Tests for screen lifecycle state transitions."""

    def test_enter_triggers_on_create_once(self, screen: ConcreteScreen) -> None:
        """_enter should call on_create only once."""
        screen._enter()
        screen._enter()
        assert screen.lifecycle_calls.count("on_create") == 1

    def test_enter_triggers_on_enter(self, screen: ConcreteScreen) -> None:
        """_enter should call on_enter."""
        screen._enter()
        assert "on_enter" in screen.lifecycle_calls

    def test_enter_sets_state_to_entering(self, screen: ConcreteScreen) -> None:
        """_enter should set state to ENTERING."""
        screen._enter()
        assert screen.state == ScreenState.ENTERING

    def test_enter_with_params(self, screen: ConcreteScreen, params: ScreenParams) -> None:
        """_enter should set screen params."""
        screen._enter(params)
        assert screen.params.get("key") == "value"
        assert screen.params.get("count") == 42

    def test_enter_complete_sets_state_to_active(self, screen: ConcreteScreen) -> None:
        """_enter_complete should set state to ACTIVE."""
        screen._enter()
        screen._enter_complete()
        assert screen.state == ScreenState.ACTIVE

    def test_enter_complete_triggers_callback(self, screen: ConcreteScreen) -> None:
        """_enter_complete should call on_enter_complete."""
        screen._enter()
        screen._enter_complete()
        assert "on_enter_complete" in screen.lifecycle_calls

    def test_exit_sets_state_to_exiting(self, screen: ConcreteScreen) -> None:
        """_exit should set state to EXITING."""
        screen._exit()
        assert screen.state == ScreenState.EXITING

    def test_exit_triggers_on_exit(self, screen: ConcreteScreen) -> None:
        """_exit should call on_exit."""
        screen._exit()
        assert "on_exit" in screen.lifecycle_calls

    def test_exit_with_result(self, screen: ConcreteScreen) -> None:
        """_exit should set the result."""
        result = ScreenResult(success=True, data={"answer": 42})
        screen._exit(result)
        assert screen.result is not None
        assert screen.result.get("answer") == 42

    def test_exit_complete_sets_state_to_destroyed(self, screen: ConcreteScreen) -> None:
        """_exit_complete should set state to DESTROYED."""
        screen._exit()
        screen._exit_complete()
        assert screen.state == ScreenState.DESTROYED

    def test_exit_complete_triggers_on_destroy(self, screen: ConcreteScreen) -> None:
        """_exit_complete should call on_destroy."""
        screen._exit()
        screen._exit_complete()
        assert "on_destroy" in screen.lifecycle_calls
        assert "on_exit_complete" in screen.lifecycle_calls

    def test_pause_sets_state_to_paused(self, screen: ConcreteScreen) -> None:
        """_pause should set state to PAUSED."""
        screen._state = ScreenState.ACTIVE
        screen._pause()
        assert screen.state == ScreenState.PAUSED

    def test_pause_triggers_on_pause(self, screen: ConcreteScreen) -> None:
        """_pause should call on_pause."""
        screen._state = ScreenState.ACTIVE
        screen._pause()
        assert "on_pause" in screen.lifecycle_calls

    def test_pause_ignored_when_not_active(self, screen: ConcreteScreen) -> None:
        """_pause should be ignored when not ACTIVE or ENTERING."""
        screen._state = ScreenState.DESTROYED
        screen._pause()
        assert "on_pause" not in screen.lifecycle_calls

    def test_resume_sets_state_to_active(self, screen: ConcreteScreen) -> None:
        """_resume should set state to ACTIVE."""
        screen._state = ScreenState.PAUSED
        screen._resume()
        assert screen.state == ScreenState.ACTIVE

    def test_resume_triggers_on_resume(self, screen: ConcreteScreen) -> None:
        """_resume should call on_resume."""
        screen._state = ScreenState.PAUSED
        screen._resume()
        assert "on_resume" in screen.lifecycle_calls

    def test_resume_ignored_when_not_paused(self, screen: ConcreteScreen) -> None:
        """_resume should be ignored when not PAUSED."""
        screen._state = ScreenState.ACTIVE
        screen._resume()
        assert "on_resume" not in screen.lifecycle_calls


# =============================================================================
# CALLBACK REGISTRATION TESTS
# =============================================================================


class TestCallbackRegistration:
    """Tests for callback registration and invocation."""

    def test_add_on_enter_callback(self, screen: ConcreteScreen) -> None:
        """Should invoke registered on_enter callbacks."""
        callback = Mock()
        screen.add_on_enter(callback)
        screen._enter()
        callback.assert_called_once_with(screen)

    def test_add_on_exit_callback(self, screen: ConcreteScreen) -> None:
        """Should invoke registered on_exit callbacks."""
        callback = Mock()
        screen.add_on_exit(callback)
        screen._exit()
        callback.assert_called_once_with(screen)

    def test_add_on_pause_callback(self, screen: ConcreteScreen) -> None:
        """Should invoke registered on_pause callbacks."""
        callback = Mock()
        screen._state = ScreenState.ACTIVE
        screen.add_on_pause(callback)
        screen._pause()
        callback.assert_called_once_with(screen)

    def test_add_on_resume_callback(self, screen: ConcreteScreen) -> None:
        """Should invoke registered on_resume callbacks."""
        callback = Mock()
        screen._state = ScreenState.PAUSED
        screen.add_on_resume(callback)
        screen._resume()
        callback.assert_called_once_with(screen)

    def test_add_on_state_change_callback(self, screen: ConcreteScreen) -> None:
        """Should invoke registered state change callbacks."""
        callback = Mock()
        screen.add_on_state_change(callback)
        screen._enter()
        callback.assert_called()
        # Check it was called with screen, old_state, new_state
        args = callback.call_args[0]
        assert args[0] is screen
        assert isinstance(args[1], ScreenState)
        assert isinstance(args[2], ScreenState)

    def test_remove_on_enter_callback(self, screen: ConcreteScreen) -> None:
        """Should be able to remove on_enter callbacks."""
        callback = Mock()
        screen.add_on_enter(callback)
        assert screen.remove_on_enter(callback) is True
        screen._enter()
        callback.assert_not_called()

    def test_remove_on_exit_callback(self, screen: ConcreteScreen) -> None:
        """Should be able to remove on_exit callbacks."""
        callback = Mock()
        screen.add_on_exit(callback)
        assert screen.remove_on_exit(callback) is True
        screen._exit()
        callback.assert_not_called()

    def test_remove_on_pause_callback(self, screen: ConcreteScreen) -> None:
        """Should be able to remove on_pause callbacks."""
        callback = Mock()
        screen.add_on_pause(callback)
        assert screen.remove_on_pause(callback) is True
        screen._state = ScreenState.ACTIVE
        screen._pause()
        callback.assert_not_called()

    def test_remove_on_resume_callback(self, screen: ConcreteScreen) -> None:
        """Should be able to remove on_resume callbacks."""
        callback = Mock()
        screen.add_on_resume(callback)
        assert screen.remove_on_resume(callback) is True
        screen._state = ScreenState.PAUSED
        screen._resume()
        callback.assert_not_called()

    def test_remove_nonexistent_callback_returns_false(self, screen: ConcreteScreen) -> None:
        """Removing nonexistent callback should return False."""
        callback = Mock()
        assert screen.remove_on_enter(callback) is False
        assert screen.remove_on_exit(callback) is False
        assert screen.remove_on_pause(callback) is False
        assert screen.remove_on_resume(callback) is False


# =============================================================================
# WIDGET MANAGEMENT TESTS
# =============================================================================


class TestWidgetManagement:
    """Tests for widget management."""

    def test_add_widget(self, screen: ConcreteScreen) -> None:
        """Should be able to add widgets."""
        widget = Mock()
        screen.add_widget("button1", widget)
        assert screen.get_widget("button1") is widget

    def test_get_widget_returns_none_if_not_found(self, screen: ConcreteScreen) -> None:
        """get_widget should return None for unknown IDs."""
        assert screen.get_widget("unknown") is None

    def test_remove_widget(self, screen: ConcreteScreen) -> None:
        """Should be able to remove widgets."""
        widget = Mock()
        screen.add_widget("button1", widget)
        removed = screen.remove_widget("button1")
        assert removed is widget
        assert screen.get_widget("button1") is None

    def test_remove_nonexistent_widget_returns_none(self, screen: ConcreteScreen) -> None:
        """Removing nonexistent widget should return None."""
        assert screen.remove_widget("unknown") is None

    def test_has_widget(self, screen: ConcreteScreen) -> None:
        """has_widget should correctly identify existing widgets."""
        screen.add_widget("button1", Mock())
        assert screen.has_widget("button1") is True
        assert screen.has_widget("unknown") is False

    def test_get_all_widgets(self, screen: ConcreteScreen) -> None:
        """get_all_widgets should return copy of all widgets."""
        widget1 = Mock()
        widget2 = Mock()
        screen.add_widget("w1", widget1)
        screen.add_widget("w2", widget2)

        widgets = screen.get_all_widgets()
        assert widgets == {"w1": widget1, "w2": widget2}
        # Should be a copy
        widgets["w3"] = Mock()
        assert "w3" not in screen.get_all_widgets()

    def test_clear_widgets(self, screen: ConcreteScreen) -> None:
        """clear_widgets should remove all widgets and root widget."""
        screen.add_widget("w1", Mock())
        screen.add_widget("w2", Mock())
        screen.root_widget = Mock()

        screen.clear_widgets()

        assert screen.get_all_widgets() == {}
        assert screen.root_widget is None

    def test_set_root_widget(self, screen: ConcreteScreen) -> None:
        """Should be able to set root widget."""
        widget = Mock()
        screen.root_widget = widget
        assert screen.root_widget is widget


# =============================================================================
# SCREEN PARAMS TESTS
# =============================================================================


class TestScreenParams:
    """Tests for ScreenParams dataclass."""

    def test_params_get(self, params: ScreenParams) -> None:
        """Should get parameter values."""
        assert params.get("key") == "value"
        assert params.get("count") == 42

    def test_params_get_with_default(self, params: ScreenParams) -> None:
        """Should return default for missing keys."""
        assert params.get("missing") is None
        assert params.get("missing", "default") == "default"

    def test_params_set(self, params: ScreenParams) -> None:
        """Should set parameter values."""
        params.set("new_key", "new_value")
        assert params.get("new_key") == "new_value"

    def test_params_has(self, params: ScreenParams) -> None:
        """Should check for parameter existence."""
        assert params.has("key") is True
        assert params.has("missing") is False

    def test_params_clear(self, params: ScreenParams) -> None:
        """Should clear all parameters."""
        params.source_screen = "source"
        params.transition_override = "fade"
        params.clear()

        assert params.data == {}
        assert params.source_screen is None
        assert params.transition_override is None

    def test_params_copy(self, params: ScreenParams) -> None:
        """Should create a copy of parameters."""
        params.source_screen = "source"
        params.transition_override = "fade"

        copy = params.copy()

        assert copy.data == params.data
        assert copy.source_screen == params.source_screen
        assert copy.transition_override == params.transition_override
        # Should be independent
        copy.set("new", "value")
        assert params.has("new") is False


# =============================================================================
# SCREEN RESULT TESTS
# =============================================================================


class TestScreenResult:
    """Tests for ScreenResult dataclass."""

    def test_result_default_success(self) -> None:
        """Result should default to success=True."""
        result = ScreenResult()
        assert result.success is True

    def test_result_get(self) -> None:
        """Should get result values."""
        result = ScreenResult(data={"key": "value"})
        assert result.get("key") == "value"

    def test_result_get_with_default(self) -> None:
        """Should return default for missing keys."""
        result = ScreenResult()
        assert result.get("missing") is None
        assert result.get("missing", "default") == "default"

    def test_result_set(self) -> None:
        """Should set result values."""
        result = ScreenResult()
        result.set("key", "value")
        assert result.get("key") == "value"


# =============================================================================
# RESULT HANDLING TESTS
# =============================================================================


class TestResultHandling:
    """Tests for screen result handling."""

    def test_set_result(self, screen: ConcreteScreen) -> None:
        """Should set result with success flag and data."""
        screen.set_result(True, answer=42, name="test")

        assert screen.result is not None
        assert screen.result.success is True
        assert screen.result.get("answer") == 42
        assert screen.result.get("name") == "test"

    def test_set_result_failure(self, screen: ConcreteScreen) -> None:
        """Should set failure result."""
        screen.set_result(False, error="Something went wrong")

        assert screen.result is not None
        assert screen.result.success is False
        assert screen.result.get("error") == "Something went wrong"


# =============================================================================
# UPDATE TESTS
# =============================================================================


class TestScreenUpdate:
    """Tests for screen update method."""

    def test_update_receives_delta_time(self, screen: ConcreteScreen) -> None:
        """update should receive delta time."""
        screen.update(0.016)
        screen.update(0.033)

        assert screen.update_deltas == [0.016, 0.033]


# =============================================================================
# STRING REPRESENTATION TESTS
# =============================================================================


class TestStringRepresentation:
    """Tests for string representation."""

    def test_repr(self, screen: ConcreteScreen) -> None:
        """__repr__ should include name and state."""
        repr_str = repr(screen)
        assert "test_screen" in repr_str
        assert "ENTERING" in repr_str

    def test_str(self, screen: ConcreteScreen) -> None:
        """__str__ should be human-readable."""
        str_str = str(screen)
        assert "test_screen" in str_str


# =============================================================================
# BACK NAVIGATION TESTS
# =============================================================================


class TestBackNavigation:
    """Tests for back navigation handling."""

    def test_on_back_default_returns_false(self, screen: ConcreteScreen) -> None:
        """Default on_back should return False."""
        assert screen.on_back() is False

    def test_custom_on_back(self) -> None:
        """Custom on_back can return True to consume event."""

        class CustomScreen(Screen):
            def __init__(self):
                super().__init__("custom")
                self.back_called = False

            def on_back(self) -> bool:
                self.back_called = True
                return True

        screen = CustomScreen()
        assert screen.on_back() is True
        assert screen.back_called is True
