"""
Screen Base Class.

Implements the base Screen component with lifecycle management, state handling,
content widget tree, and screen parameters for data passing.

References:
- UI_CONTEXT.md Section: Screen Stack Pattern
- ARCHITECTURE_UI.md Section: Screen Management
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .screen_stack import ScreenStack


# =============================================================================
# SCREEN STATE ENUM
# =============================================================================


class ScreenState(Enum):
    """State of a screen in its lifecycle."""
    ENTERING = auto()      # Screen is transitioning in
    ACTIVE = auto()        # Screen is fully active and interactive
    PAUSED = auto()        # Screen is paused (e.g., another screen on top)
    EXITING = auto()       # Screen is transitioning out
    DESTROYED = auto()     # Screen has been removed and cleaned up


# =============================================================================
# SCREEN PARAMETERS
# =============================================================================


@dataclass
class ScreenParams:
    """
    Parameters passed to a screen during navigation.

    Allows data passing between screens during push/replace operations.
    """
    data: Dict[str, Any] = field(default_factory=dict)
    source_screen: Optional[str] = None
    transition_override: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a parameter value."""
        self.data[key] = value

    def has(self, key: str) -> bool:
        """Check if a parameter exists."""
        return key in self.data

    def clear(self) -> None:
        """Clear all parameters."""
        self.data.clear()
        self.source_screen = None
        self.transition_override = None

    def copy(self) -> "ScreenParams":
        """Create a copy of the parameters."""
        return ScreenParams(
            data=dict(self.data),
            source_screen=self.source_screen,
            transition_override=self.transition_override,
        )


# =============================================================================
# SCREEN RESULT
# =============================================================================


@dataclass
class ScreenResult:
    """
    Result returned from a screen when it exits.

    Allows screens to return data to the screen that pushed them.
    """
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a result value."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a result value."""
        self.data[key] = value


# =============================================================================
# SCREEN LIFECYCLE CALLBACKS
# =============================================================================


LifecycleCallback = Callable[["Screen"], None]
StateChangeCallback = Callable[["Screen", ScreenState, ScreenState], None]


# =============================================================================
# SCREEN BASE CLASS
# =============================================================================


class Screen(ABC):
    """
    Base class for all screens in the UI system.

    Provides:
    - Lifecycle management (on_enter, on_exit, on_pause, on_resume)
    - State tracking (entering, active, paused, exiting, destroyed)
    - Content widget tree management
    - Screen parameters for data passing
    - Result handling for returning data to parent screens

    Usage:
        class MainMenuScreen(Screen):
            def __init__(self):
                super().__init__("main_menu")

            def on_create(self) -> None:
                # Build UI here
                pass

            def on_destroy(self) -> None:
                # Cleanup here
                pass
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a screen.

        Args:
            name: Unique name/identifier for the screen
        """
        self._name = name
        self._state = ScreenState.ENTERING
        self._params = ScreenParams()
        self._result: Optional[ScreenResult] = None
        self._stack: Optional["ScreenStack"] = None

        # Widget tree (root widget for this screen's content)
        self._root_widget: Optional[Any] = None
        self._widgets: Dict[str, Any] = {}

        # Lifecycle callbacks
        self._on_enter_callbacks: List[LifecycleCallback] = []
        self._on_exit_callbacks: List[LifecycleCallback] = []
        self._on_pause_callbacks: List[LifecycleCallback] = []
        self._on_resume_callbacks: List[LifecycleCallback] = []
        self._on_state_change_callbacks: List[StateChangeCallback] = []

        # Configuration
        self._is_modal: bool = False
        self._blocks_input: bool = True
        self._is_overlay: bool = False
        self._can_go_back: bool = True
        self._pause_below: bool = True

        # Cached state
        self._is_created: bool = False
        self._is_destroyed: bool = False
        self._enter_time: float = 0.0
        self._exit_time: float = 0.0

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def name(self) -> str:
        """Get the screen name."""
        return self._name

    @property
    def state(self) -> ScreenState:
        """Get the current screen state."""
        return self._state

    @property
    def params(self) -> ScreenParams:
        """Get the screen parameters."""
        return self._params

    @property
    def result(self) -> Optional[ScreenResult]:
        """Get the screen result (set when exiting)."""
        return self._result

    @property
    def stack(self) -> Optional["ScreenStack"]:
        """Get the screen stack this screen belongs to."""
        return self._stack

    @property
    def root_widget(self) -> Optional[Any]:
        """Get the root widget of this screen."""
        return self._root_widget

    @root_widget.setter
    def root_widget(self, widget: Any) -> None:
        """Set the root widget of this screen."""
        self._root_widget = widget

    @property
    def is_active(self) -> bool:
        """Check if the screen is currently active."""
        return self._state == ScreenState.ACTIVE

    @property
    def is_entering(self) -> bool:
        """Check if the screen is entering."""
        return self._state == ScreenState.ENTERING

    @property
    def is_exiting(self) -> bool:
        """Check if the screen is exiting."""
        return self._state == ScreenState.EXITING

    @property
    def is_paused(self) -> bool:
        """Check if the screen is paused."""
        return self._state == ScreenState.PAUSED

    @property
    def is_destroyed(self) -> bool:
        """Check if the screen has been destroyed."""
        return self._state == ScreenState.DESTROYED

    @property
    def is_modal(self) -> bool:
        """Check if this is a modal screen."""
        return self._is_modal

    @is_modal.setter
    def is_modal(self, value: bool) -> None:
        """Set whether this is a modal screen."""
        self._is_modal = value

    @property
    def blocks_input(self) -> bool:
        """Check if this screen blocks input to screens below."""
        return self._blocks_input

    @blocks_input.setter
    def blocks_input(self, value: bool) -> None:
        """Set whether this screen blocks input to screens below."""
        self._blocks_input = value

    @property
    def is_overlay(self) -> bool:
        """Check if this screen is an overlay (doesn't hide screens below)."""
        return self._is_overlay

    @is_overlay.setter
    def is_overlay(self, value: bool) -> None:
        """Set whether this is an overlay screen."""
        self._is_overlay = value

    @property
    def can_go_back(self) -> bool:
        """Check if back navigation is allowed from this screen."""
        return self._can_go_back

    @can_go_back.setter
    def can_go_back(self, value: bool) -> None:
        """Set whether back navigation is allowed."""
        self._can_go_back = value

    @property
    def pause_below(self) -> bool:
        """Check if screens below should be paused when this screen is shown."""
        return self._pause_below

    @pause_below.setter
    def pause_below(self, value: bool) -> None:
        """Set whether screens below should be paused."""
        self._pause_below = value

    # =========================================================================
    # WIDGET MANAGEMENT
    # =========================================================================

    def add_widget(self, widget_id: str, widget: Any) -> None:
        """Add a widget to the screen's widget registry."""
        self._widgets[widget_id] = widget

    def get_widget(self, widget_id: str) -> Optional[Any]:
        """Get a widget by its ID."""
        return self._widgets.get(widget_id)

    def remove_widget(self, widget_id: str) -> Optional[Any]:
        """Remove and return a widget by its ID."""
        return self._widgets.pop(widget_id, None)

    def has_widget(self, widget_id: str) -> bool:
        """Check if a widget exists."""
        return widget_id in self._widgets

    def get_all_widgets(self) -> Dict[str, Any]:
        """Get all widgets."""
        return dict(self._widgets)

    def clear_widgets(self) -> None:
        """Remove all widgets."""
        self._widgets.clear()
        self._root_widget = None

    # =========================================================================
    # LIFECYCLE CALLBACKS REGISTRATION
    # =========================================================================

    def add_on_enter(self, callback: LifecycleCallback) -> None:
        """Add a callback for when the screen enters."""
        self._on_enter_callbacks.append(callback)

    def add_on_exit(self, callback: LifecycleCallback) -> None:
        """Add a callback for when the screen exits."""
        self._on_exit_callbacks.append(callback)

    def add_on_pause(self, callback: LifecycleCallback) -> None:
        """Add a callback for when the screen is paused."""
        self._on_pause_callbacks.append(callback)

    def add_on_resume(self, callback: LifecycleCallback) -> None:
        """Add a callback for when the screen is resumed."""
        self._on_resume_callbacks.append(callback)

    def add_on_state_change(self, callback: StateChangeCallback) -> None:
        """Add a callback for when the screen state changes."""
        self._on_state_change_callbacks.append(callback)

    def remove_on_enter(self, callback: LifecycleCallback) -> bool:
        """Remove an enter callback. Returns True if found."""
        try:
            self._on_enter_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def remove_on_exit(self, callback: LifecycleCallback) -> bool:
        """Remove an exit callback. Returns True if found."""
        try:
            self._on_exit_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def remove_on_pause(self, callback: LifecycleCallback) -> bool:
        """Remove a pause callback. Returns True if found."""
        try:
            self._on_pause_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def remove_on_resume(self, callback: LifecycleCallback) -> bool:
        """Remove a resume callback. Returns True if found."""
        try:
            self._on_resume_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # =========================================================================
    # STATE TRANSITIONS
    # =========================================================================

    def _set_state(self, new_state: ScreenState) -> None:
        """Internal method to set the screen state."""
        old_state = self._state
        self._state = new_state

        # Notify state change callbacks
        for callback in self._on_state_change_callbacks:
            callback(self, old_state, new_state)

    def _enter(self, params: Optional[ScreenParams] = None) -> None:
        """Called when the screen is about to enter."""
        if params:
            self._params = params

        self._set_state(ScreenState.ENTERING)

        # Create if not created
        if not self._is_created:
            self.on_create()
            self._is_created = True

        # Call lifecycle method
        self.on_enter()

        # Call registered callbacks
        for callback in self._on_enter_callbacks:
            callback(self)

    def _enter_complete(self) -> None:
        """Called when the enter transition is complete."""
        self._set_state(ScreenState.ACTIVE)
        self.on_enter_complete()

    def _exit(self, result: Optional[ScreenResult] = None) -> None:
        """Called when the screen is about to exit."""
        self._result = result or ScreenResult()
        self._set_state(ScreenState.EXITING)

        # Call lifecycle method
        self.on_exit()

        # Call registered callbacks
        for callback in self._on_exit_callbacks:
            callback(self)

    def _exit_complete(self) -> None:
        """Called when the exit transition is complete."""
        self._set_state(ScreenState.DESTROYED)
        self.on_exit_complete()

        # Cleanup - call user override first
        self.on_destroy()
        self._is_destroyed = True

        # Clear internal references to prevent memory leaks
        self._cleanup_internal()

    def _pause(self) -> None:
        """Called when the screen is paused."""
        # Only allow pausing from ACTIVE state to avoid inconsistent states
        # Pausing during ENTERING could cause the screen to never reach ACTIVE
        if self._state != ScreenState.ACTIVE:
            return

        self._set_state(ScreenState.PAUSED)

        # Call lifecycle method
        self.on_pause()

        # Call registered callbacks
        for callback in self._on_pause_callbacks:
            callback(self)

    def _resume(self) -> None:
        """Called when the screen is resumed."""
        if self._state != ScreenState.PAUSED:
            return

        self._set_state(ScreenState.ACTIVE)

        # Call lifecycle method
        self.on_resume()

        # Call registered callbacks
        for callback in self._on_resume_callbacks:
            callback(self)

    def _cleanup_internal(self) -> None:
        """
        Clean up internal references to prevent memory leaks.

        Called automatically after on_destroy(). Subclasses should not
        override this; use on_destroy() for custom cleanup instead.
        """
        # Clear widgets
        self.clear_widgets()

        # Clear callbacks to break potential circular references
        self._on_enter_callbacks.clear()
        self._on_exit_callbacks.clear()
        self._on_pause_callbacks.clear()
        self._on_resume_callbacks.clear()
        self._on_state_change_callbacks.clear()

        # Clear params (but preserve result for callers to read after pop)
        self._params.clear()

    # =========================================================================
    # LIFECYCLE METHODS (OVERRIDE THESE)
    # =========================================================================

    def on_create(self) -> None:
        """
        Called once when the screen is first created.

        Override to build the screen's widget tree.
        """
        pass

    def on_destroy(self) -> None:
        """
        Called once when the screen is destroyed.

        Override to cleanup resources.
        """
        pass

    def on_enter(self) -> None:
        """
        Called when the screen starts entering.

        Override to perform actions when showing the screen.
        """
        pass

    def on_enter_complete(self) -> None:
        """
        Called when the screen has fully entered.

        Override to perform actions after the enter transition completes.
        """
        pass

    def on_exit(self) -> None:
        """
        Called when the screen starts exiting.

        Override to perform actions when hiding the screen.
        """
        pass

    def on_exit_complete(self) -> None:
        """
        Called when the screen has fully exited.

        Override to perform actions after the exit transition completes.
        """
        pass

    def on_pause(self) -> None:
        """
        Called when the screen is paused (another screen pushed on top).

        Override to pause animations, timers, etc.
        """
        pass

    def on_resume(self) -> None:
        """
        Called when the screen is resumed (screen on top was removed).

        Override to resume animations, timers, etc.
        """
        pass

    def on_back(self) -> bool:
        """
        Called when back navigation is requested.

        Override to handle back navigation. Return True to consume
        the event, False to allow default behavior (pop screen).
        """
        return False

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(self, delta_time: float) -> None:
        """
        Called every frame to update the screen.

        Override to implement per-frame logic.

        Args:
            delta_time: Time since last update in seconds
        """
        pass

    # =========================================================================
    # RESULT HANDLING
    # =========================================================================

    def set_result(self, success: bool = True, **data: Any) -> None:
        """
        Set the result that will be returned when this screen exits.

        Args:
            success: Whether the screen completed successfully
            **data: Key-value pairs to include in the result
        """
        self._result = ScreenResult(success=success, data=data)

    def finish(
        self,
        result: Optional[ScreenResult] = None,
        **data: Any,
    ) -> None:
        """
        Request to close this screen and optionally return a result.

        Args:
            result: Result to return, or None to use default
            **data: If result is None, key-value pairs for the result
        """
        if result is None and data:
            result = ScreenResult(success=True, data=data)

        if self._stack:
            self._stack.pop(result)

    # =========================================================================
    # STRING REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        return f"Screen(name={self._name!r}, state={self._state.name})"

    def __str__(self) -> str:
        return f"Screen[{self._name}]"


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    # Enums
    "ScreenState",
    # Data classes
    "ScreenParams",
    "ScreenResult",
    # Types
    "LifecycleCallback",
    "StateChangeCallback",
    # Classes
    "Screen",
]
