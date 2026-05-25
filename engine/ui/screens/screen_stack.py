"""
Screen Stack Manager.

Implements the screen stack with push/pop/replace/clear operations,
history tracking, modal screen support, and screen caching.

References:
- UI_CONTEXT.md Section: Screen Stack Pattern
- ARCHITECTURE_UI.md Section: Screen Management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

from .screen import Screen, ScreenParams, ScreenResult, ScreenState

if TYPE_CHECKING:
    from .transitions import ITransition


# =============================================================================
# STACK OPERATION ENUM
# =============================================================================


class StackOperation(Enum):
    """Type of stack operation."""
    PUSH = auto()      # Add a new screen on top
    POP = auto()       # Remove the top screen
    REPLACE = auto()   # Replace the top screen
    CLEAR = auto()     # Remove all screens
    SWAP = auto()      # Swap top two screens


# =============================================================================
# HISTORY ENTRY
# =============================================================================


@dataclass
class HistoryEntry:
    """Entry in the navigation history."""
    screen_name: str
    params: Optional[ScreenParams] = None
    timestamp: float = 0.0
    operation: StackOperation = StackOperation.PUSH


# =============================================================================
# SCREEN FACTORY
# =============================================================================


ScreenFactory = Callable[[str, Optional[ScreenParams]], Screen]


# =============================================================================
# SCREEN STACK EVENTS
# =============================================================================


StackEventCallback = Callable[["ScreenStack", StackOperation, Optional[Screen], Optional[Screen]], None]


# =============================================================================
# SCREEN CACHE
# =============================================================================


class ScreenCache:
    """
    Cache for screen instances.

    Allows reusing screen instances instead of creating new ones,
    which can improve performance for frequently-used screens.
    """

    def __init__(self, max_size: int = 10) -> None:
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of screens to cache
        """
        self._cache: Dict[str, Screen] = {}
        self._access_order: List[str] = []
        self._max_size = max_size
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable caching."""
        self._enabled = value
        if not value:
            self.clear()

    @property
    def max_size(self) -> int:
        """Get the maximum cache size."""
        return self._max_size

    @max_size.setter
    def max_size(self, value: int) -> None:
        """Set the maximum cache size."""
        self._max_size = max(1, value)
        self._evict_if_needed()

    @property
    def size(self) -> int:
        """Get the current cache size."""
        return len(self._cache)

    def get(self, name: str) -> Optional[Screen]:
        """
        Get a screen from the cache.

        Args:
            name: Screen name

        Returns:
            Cached screen instance or None
        """
        if not self._enabled or name not in self._cache:
            return None

        # Update access order (LRU)
        self._access_order.remove(name)
        self._access_order.append(name)

        return self._cache[name]

    def put(self, screen: Screen) -> None:
        """
        Add a screen to the cache.

        Args:
            screen: Screen instance to cache
        """
        if not self._enabled:
            return

        name = screen.name

        # Remove old entry if exists
        if name in self._cache:
            self._access_order.remove(name)

        # Add new entry
        self._cache[name] = screen
        self._access_order.append(name)

        # Evict if necessary
        self._evict_if_needed()

    def remove(self, name: str) -> Optional[Screen]:
        """
        Remove a screen from the cache.

        Args:
            name: Screen name

        Returns:
            Removed screen or None
        """
        if name in self._cache:
            screen = self._cache.pop(name)
            self._access_order.remove(name)
            return screen
        return None

    def contains(self, name: str) -> bool:
        """Check if a screen is in the cache."""
        return name in self._cache

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()

    def _evict_if_needed(self) -> None:
        """Evict least recently used screens if cache is full."""
        while len(self._cache) > self._max_size:
            # Remove least recently used
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)


# =============================================================================
# SCREEN STACK
# =============================================================================


class ScreenStack:
    """
    Manages a stack of screens with navigation operations.

    Provides:
    - Push/pop/replace/clear operations
    - History tracking
    - Modal screen support
    - Screen caching
    - Transition management

    Usage:
        stack = ScreenStack()
        stack.register_factory("main_menu", lambda n, p: MainMenuScreen())
        stack.push("main_menu")
        stack.push("settings", ScreenParams(data={"tab": "audio"}))
        stack.pop()
    """

    def __init__(self) -> None:
        """Initialize the screen stack."""
        self._stack: List[Screen] = []
        self._factories: Dict[str, ScreenFactory] = {}
        self._history: List[HistoryEntry] = []
        self._history_limit = 100
        self._cache = ScreenCache()

        # Transition
        self._current_transition: Optional["ITransition"] = None
        self._default_transition: Optional["ITransition"] = None

        # Event callbacks
        self._event_callbacks: List[StackEventCallback] = []

        # Configuration
        self._allow_duplicate_screens = False
        self._auto_pause_below = True
        self._history_enabled = True

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def stack(self) -> List[Screen]:
        """Get a copy of the screen stack."""
        return list(self._stack)

    @property
    def history(self) -> List[HistoryEntry]:
        """Get a copy of the navigation history."""
        return list(self._history)

    @property
    def cache(self) -> ScreenCache:
        """Get the screen cache."""
        return self._cache

    @property
    def top(self) -> Optional[Screen]:
        """Get the top screen."""
        return self._stack[-1] if self._stack else None

    @property
    def bottom(self) -> Optional[Screen]:
        """Get the bottom screen."""
        return self._stack[0] if self._stack else None

    @property
    def count(self) -> int:
        """Get the number of screens in the stack."""
        return len(self._stack)

    @property
    def is_empty(self) -> bool:
        """Check if the stack is empty."""
        return len(self._stack) == 0

    @property
    def is_transitioning(self) -> bool:
        """Check if a transition is in progress."""
        return self._current_transition is not None

    @property
    def default_transition(self) -> Optional["ITransition"]:
        """Get the default transition."""
        return self._default_transition

    @default_transition.setter
    def default_transition(self, transition: Optional["ITransition"]) -> None:
        """Set the default transition."""
        self._default_transition = transition

    @property
    def history_limit(self) -> int:
        """Get the history limit."""
        return self._history_limit

    @history_limit.setter
    def history_limit(self, value: int) -> None:
        """Set the history limit."""
        self._history_limit = max(0, value)
        self._trim_history()

    @property
    def allow_duplicate_screens(self) -> bool:
        """Check if duplicate screens are allowed."""
        return self._allow_duplicate_screens

    @allow_duplicate_screens.setter
    def allow_duplicate_screens(self, value: bool) -> None:
        """Set whether duplicate screens are allowed."""
        self._allow_duplicate_screens = value

    @property
    def history_enabled(self) -> bool:
        """Check if history tracking is enabled."""
        return self._history_enabled

    @history_enabled.setter
    def history_enabled(self, value: bool) -> None:
        """Enable or disable history tracking."""
        self._history_enabled = value

    # =========================================================================
    # FACTORY REGISTRATION
    # =========================================================================

    def register_factory(self, name: str, factory: ScreenFactory) -> None:
        """
        Register a factory for creating screens.

        Args:
            name: Screen name
            factory: Function that creates screen instances
        """
        self._factories[name] = factory

    def unregister_factory(self, name: str) -> bool:
        """
        Unregister a screen factory.

        Args:
            name: Screen name

        Returns:
            True if factory was removed
        """
        if name in self._factories:
            del self._factories[name]
            return True
        return False

    def has_factory(self, name: str) -> bool:
        """Check if a factory is registered for a screen name."""
        return name in self._factories

    def get_registered_names(self) -> List[str]:
        """Get list of registered screen names."""
        return list(self._factories.keys())

    # =========================================================================
    # EVENT CALLBACKS
    # =========================================================================

    def add_event_callback(self, callback: StackEventCallback) -> None:
        """Add a callback for stack events."""
        self._event_callbacks.append(callback)

    def remove_event_callback(self, callback: StackEventCallback) -> bool:
        """Remove an event callback. Returns True if found."""
        try:
            self._event_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _notify_event(
        self,
        operation: StackOperation,
        entering: Optional[Screen],
        exiting: Optional[Screen],
    ) -> None:
        """Notify event callbacks."""
        for callback in self._event_callbacks:
            callback(self, operation, entering, exiting)

    # =========================================================================
    # STACK OPERATIONS
    # =========================================================================

    def push(
        self,
        name_or_screen: str | Screen,
        params: Optional[ScreenParams] = None,
        transition: Optional["ITransition"] = None,
        use_cache: bool = True,
    ) -> Optional[Screen]:
        """
        Push a new screen onto the stack.

        Args:
            name_or_screen: Screen name or screen instance
            params: Parameters to pass to the screen
            transition: Transition to use (or None for default)
            use_cache: Whether to use cached screen if available

        Returns:
            The pushed screen, or None if push failed
        """
        # Get or create screen
        if isinstance(name_or_screen, str):
            screen = self._get_or_create_screen(name_or_screen, params, use_cache)
            if screen is None:
                return None
        else:
            screen = name_or_screen

        # Check for duplicates
        if not self._allow_duplicate_screens:
            for s in self._stack:
                if s.name == screen.name:
                    return None

        # Get the current top screen
        old_top = self.top

        # Pause the current top screen
        if old_top and self._auto_pause_below and screen.pause_below:
            old_top._pause()

        # Add to stack
        self._stack.append(screen)
        screen._stack = self

        # Record history
        self._record_history(screen.name, params, StackOperation.PUSH)

        # Start enter transition
        screen._enter(params)

        # Use provided transition, transition from params, or default
        actual_transition = transition
        if actual_transition is None and params and params.transition_override:
            # Could look up transition by name here
            pass
        if actual_transition is None:
            actual_transition = self._default_transition

        self._current_transition = actual_transition

        # If no transition, complete immediately
        if actual_transition is None:
            screen._enter_complete()

        # Notify listeners
        self._notify_event(StackOperation.PUSH, screen, old_top)

        return screen

    def pop(
        self,
        result: Optional[ScreenResult] = None,
        transition: Optional["ITransition"] = None,
    ) -> Optional[Screen]:
        """
        Pop the top screen from the stack.

        Args:
            result: Result to return to the screen below
            transition: Transition to use (or None for default)

        Returns:
            The popped screen, or None if stack was empty
        """
        if self.is_empty:
            return None

        screen = self._stack.pop()
        old_top = self.top  # New top after pop

        # Start exit transition
        screen._exit(result)

        # Use provided transition or default
        actual_transition = transition or self._default_transition
        self._current_transition = actual_transition

        # If no transition, complete immediately
        if actual_transition is None:
            screen._exit_complete()

        # Resume the screen below
        if old_top and old_top.is_paused:
            old_top._resume()

        # Record history
        self._record_history(screen.name, None, StackOperation.POP)

        # Clear the screen's stack reference to prevent memory leaks
        screen._stack = None

        # Cache the screen if caching is enabled
        self._cache.put(screen)

        # Notify listeners
        self._notify_event(StackOperation.POP, old_top, screen)

        return screen

    def replace(
        self,
        name_or_screen: str | Screen,
        params: Optional[ScreenParams] = None,
        transition: Optional["ITransition"] = None,
        use_cache: bool = True,
    ) -> Optional[Screen]:
        """
        Replace the top screen with a new screen.

        Args:
            name_or_screen: Screen name or screen instance
            params: Parameters to pass to the new screen
            transition: Transition to use (or None for default)
            use_cache: Whether to use cached screen if available

        Returns:
            The new screen, or None if replace failed
        """
        if self.is_empty:
            # If stack is empty, just push
            return self.push(name_or_screen, params, transition, use_cache)

        # Get or create the new screen
        if isinstance(name_or_screen, str):
            new_screen = self._get_or_create_screen(name_or_screen, params, use_cache)
            if new_screen is None:
                return None
        else:
            new_screen = name_or_screen

        # Get the old screen
        old_screen = self._stack[-1]

        # Replace in stack
        self._stack[-1] = new_screen
        new_screen._stack = self

        # Start transitions
        old_screen._exit()
        new_screen._enter(params)

        # Use provided transition or default
        actual_transition = transition or self._default_transition
        self._current_transition = actual_transition

        # If no transition, complete immediately
        if actual_transition is None:
            old_screen._exit_complete()
            new_screen._enter_complete()

        # Record history
        self._record_history(new_screen.name, params, StackOperation.REPLACE)

        # Clear old screen's stack reference to prevent memory leaks
        old_screen._stack = None

        # Cache the old screen
        self._cache.put(old_screen)

        # Notify listeners
        self._notify_event(StackOperation.REPLACE, new_screen, old_screen)

        return new_screen

    def clear(
        self,
        transition: Optional["ITransition"] = None,
    ) -> List[Screen]:
        """
        Clear all screens from the stack.

        Args:
            transition: Transition for the top screen exit

        Returns:
            List of cleared screens
        """
        if self.is_empty:
            return []

        cleared = list(self._stack)
        top = self._stack[-1]

        # Exit all screens
        for screen in reversed(self._stack):
            if screen is top:
                screen._exit()
            else:
                screen._exit()
                screen._exit_complete()

        self._stack.clear()

        # Handle transition for top screen
        actual_transition = transition or self._default_transition
        self._current_transition = actual_transition

        if actual_transition is None:
            top._exit_complete()

        # Record history
        self._record_history("", None, StackOperation.CLEAR)

        # Clear stack references and cache all screens
        for screen in cleared:
            screen._stack = None
            self._cache.put(screen)

        # Notify listeners
        self._notify_event(StackOperation.CLEAR, None, top)

        return cleared

    def pop_to(
        self,
        name: str,
        result: Optional[ScreenResult] = None,
        transition: Optional["ITransition"] = None,
    ) -> List[Screen]:
        """
        Pop screens until a screen with the given name is on top.

        Args:
            name: Name of the screen to pop to
            result: Result to pass when popping
            transition: Transition for the final pop

        Returns:
            List of popped screens
        """
        popped: List[Screen] = []

        # Find the target screen
        target_index = -1
        for i, screen in enumerate(self._stack):
            if screen.name == name:
                target_index = i
                break

        if target_index < 0:
            return popped  # Target not found

        # Pop screens above the target
        while len(self._stack) > target_index + 1:
            screen = self._stack.pop()
            screen._exit()
            screen._exit_complete()
            screen._stack = None  # Clear reference to prevent memory leaks
            popped.append(screen)
            self._cache.put(screen)

        # Resume the target screen
        if self.top and self.top.is_paused:
            self.top._resume()

        return popped

    def pop_to_root(
        self,
        result: Optional[ScreenResult] = None,
        transition: Optional["ITransition"] = None,
    ) -> List[Screen]:
        """
        Pop all screens except the bottom one.

        Args:
            result: Result to pass when popping
            transition: Transition for the final pop

        Returns:
            List of popped screens
        """
        if len(self._stack) <= 1:
            return []

        root_name = self._stack[0].name
        return self.pop_to(root_name, result, transition)

    def swap(
        self,
        transition: Optional["ITransition"] = None,
    ) -> bool:
        """
        Swap the top two screens.

        Args:
            transition: Transition to use

        Returns:
            True if swap succeeded
        """
        if len(self._stack) < 2:
            return False

        # Swap top two
        self._stack[-1], self._stack[-2] = self._stack[-2], self._stack[-1]

        old_top = self._stack[-2]
        new_top = self._stack[-1]

        # Update states
        old_top._pause()
        if new_top.is_paused:
            new_top._resume()

        # Record history
        self._record_history(new_top.name, None, StackOperation.SWAP)

        # Notify listeners
        self._notify_event(StackOperation.SWAP, new_top, old_top)

        return True

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    def get(self, index: int) -> Optional[Screen]:
        """Get a screen by index (0 = bottom)."""
        if 0 <= index < len(self._stack):
            return self._stack[index]
        return None

    def get_by_name(self, name: str) -> Optional[Screen]:
        """Get a screen by name (first match from top)."""
        for screen in reversed(self._stack):
            if screen.name == name:
                return screen
        return None

    def contains(self, name: str) -> bool:
        """Check if a screen with the given name is in the stack."""
        return any(s.name == name for s in self._stack)

    def index_of(self, name: str) -> int:
        """Get the index of a screen by name. Returns -1 if not found."""
        for i, screen in enumerate(self._stack):
            if screen.name == name:
                return i
        return -1

    def get_screens_above(self, name: str) -> List[Screen]:
        """Get all screens above the named screen."""
        index = self.index_of(name)
        if index < 0:
            return []
        return list(self._stack[index + 1:])

    def get_screens_below(self, name: str) -> List[Screen]:
        """Get all screens below the named screen."""
        index = self.index_of(name)
        if index <= 0:
            return []
        return list(self._stack[:index])

    # =========================================================================
    # MODAL SUPPORT
    # =========================================================================

    def push_modal(
        self,
        name_or_screen: str | Screen,
        params: Optional[ScreenParams] = None,
        transition: Optional["ITransition"] = None,
    ) -> Optional[Screen]:
        """
        Push a modal screen (blocks input, doesn't pause screens below).

        Args:
            name_or_screen: Screen name or screen instance
            params: Parameters to pass to the screen
            transition: Transition to use

        Returns:
            The pushed modal screen
        """
        # Get or create screen
        if isinstance(name_or_screen, str):
            screen = self._get_or_create_screen(name_or_screen, params, True)
            if screen is None:
                return None
        else:
            screen = name_or_screen

        # Configure as modal
        screen.is_modal = True
        screen.blocks_input = True
        screen.is_overlay = True
        screen.pause_below = False

        return self.push(screen, params, transition, use_cache=False)

    def push_overlay(
        self,
        name_or_screen: str | Screen,
        params: Optional[ScreenParams] = None,
        transition: Optional["ITransition"] = None,
    ) -> Optional[Screen]:
        """
        Push an overlay screen (doesn't block input or pause screens below).

        Args:
            name_or_screen: Screen name or screen instance
            params: Parameters to pass to the screen
            transition: Transition to use

        Returns:
            The pushed overlay screen
        """
        # Get or create screen
        if isinstance(name_or_screen, str):
            screen = self._get_or_create_screen(name_or_screen, params, True)
            if screen is None:
                return None
        else:
            screen = name_or_screen

        # Configure as overlay
        screen.is_modal = False
        screen.blocks_input = False
        screen.is_overlay = True
        screen.pause_below = False

        return self.push(screen, params, transition, use_cache=False)

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(self, delta_time: float) -> None:
        """
        Update all screens and transitions.

        Args:
            delta_time: Time since last update in seconds
        """
        # Update transition
        if self._current_transition:
            self._current_transition.update(delta_time)
            if self._current_transition.is_complete:
                self._current_transition = None
                # Complete any pending transitions on screens
                if self.top and self.top.is_entering:
                    self.top._enter_complete()

        # Update active screens
        for screen in self._stack:
            if screen.is_active or (screen.is_overlay and not screen.is_paused):
                screen.update(delta_time)

    # =========================================================================
    # BACK NAVIGATION
    # =========================================================================

    def back(self, result: Optional[ScreenResult] = None) -> bool:
        """
        Handle back navigation.

        First gives the top screen a chance to handle it,
        then pops if allowed.

        Args:
            result: Result to return when popping

        Returns:
            True if back was handled
        """
        if self.is_empty:
            return False

        top = self.top

        # Let the screen handle it first
        if top.on_back():
            return True

        # Check if back navigation is allowed
        if not top.can_go_back:
            return False

        # Pop the screen
        self.pop(result)
        return True

    # =========================================================================
    # HISTORY
    # =========================================================================

    def _record_history(
        self,
        screen_name: str,
        params: Optional[ScreenParams],
        operation: StackOperation,
    ) -> None:
        """Record a navigation event in history."""
        if not self._history_enabled:
            return

        import time
        entry = HistoryEntry(
            screen_name=screen_name,
            params=params.copy() if params else None,
            timestamp=time.time(),
            operation=operation,
        )
        self._history.append(entry)
        self._trim_history()

    def _trim_history(self) -> None:
        """Trim history to the limit."""
        if self._history_limit > 0:
            while len(self._history) > self._history_limit:
                self._history.pop(0)

    def clear_history(self) -> None:
        """Clear the navigation history."""
        self._history.clear()

    def get_history_entry(self, index: int) -> Optional[HistoryEntry]:
        """Get a history entry by index."""
        if 0 <= index < len(self._history):
            return self._history[index]
        return None

    def get_last_history_entry(self) -> Optional[HistoryEntry]:
        """Get the most recent history entry."""
        return self._history[-1] if self._history else None

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _get_or_create_screen(
        self,
        name: str,
        params: Optional[ScreenParams],
        use_cache: bool,
    ) -> Optional[Screen]:
        """Get a screen from cache or create a new one."""
        # Try cache first
        if use_cache:
            cached = self._cache.get(name)
            if cached:
                return cached

        # Create new screen
        factory = self._factories.get(name)
        if factory is None:
            return None

        return factory(name, params)

    # =========================================================================
    # STRING REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        names = [s.name for s in self._stack]
        return f"ScreenStack({names})"

    def __str__(self) -> str:
        if self.is_empty:
            return "ScreenStack[empty]"
        return f"ScreenStack[{' > '.join(s.name for s in self._stack)}]"


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    # Enums
    "StackOperation",
    # Data classes
    "HistoryEntry",
    # Types
    "ScreenFactory",
    "StackEventCallback",
    # Classes
    "ScreenCache",
    "ScreenStack",
]
