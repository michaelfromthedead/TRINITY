"""
Cell state machine for world partition streaming.

Manages the lifecycle of streaming cells through state transitions:
    UNLOADED -> LOADING -> LOADED -> ACTIVATED

Provides validation, callbacks, and error handling for state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
import time


class CellState(Enum):
    """
    States in the cell lifecycle.

    State transitions:
        UNLOADED -> LOADING (begin_load)
        LOADING -> LOADED (complete_load)
        LOADING -> UNLOADED (cancel_load)
        LOADED -> ACTIVATED (activate)
        LOADED -> UNLOADING (begin_unload)
        ACTIVATED -> LOADED (deactivate)
        ACTIVATED -> UNLOADING (begin_unload)
        UNLOADING -> UNLOADED (complete_unload)
    """

    UNLOADED = auto()    # No data in memory
    LOADING = auto()     # Async load in progress
    LOADED = auto()      # Data loaded but not active
    ACTIVATED = auto()   # Fully active and ticking
    UNLOADING = auto()   # Async unload in progress

    @property
    def is_in_memory(self) -> bool:
        """Check if this state has data in memory."""
        return self in (CellState.LOADED, CellState.ACTIVATED)

    @property
    def is_transitioning(self) -> bool:
        """Check if this state is a transitional state."""
        return self in (CellState.LOADING, CellState.UNLOADING)

    @property
    def can_tick(self) -> bool:
        """Check if this state allows tick updates."""
        return self == CellState.ACTIVATED


class CellStateError(Exception):
    """Exception raised for invalid state transitions."""

    def __init__(
        self,
        message: str,
        current_state: CellState,
        attempted_state: Optional[CellState] = None,
        cell_id: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(message)
        self.current_state = current_state
        self.attempted_state = attempted_state
        self.cell_id = cell_id


# Type alias for state transition callbacks
StateTransitionCallback = Callable[
    ["CellStateMachine", CellState, CellState], None
]


@dataclass
class StateTransition:
    """Represents a valid state transition."""

    from_state: CellState
    to_state: CellState
    action_name: str
    requires_data: bool = False

    def __hash__(self) -> int:
        return hash((self.from_state, self.to_state))


# Define all valid state transitions
VALID_TRANSITIONS: Dict[Tuple[CellState, CellState], StateTransition] = {
    (CellState.UNLOADED, CellState.LOADING): StateTransition(
        CellState.UNLOADED, CellState.LOADING, "begin_load"
    ),
    (CellState.LOADING, CellState.LOADED): StateTransition(
        CellState.LOADING, CellState.LOADED, "complete_load", requires_data=True
    ),
    (CellState.LOADING, CellState.UNLOADED): StateTransition(
        CellState.LOADING, CellState.UNLOADED, "cancel_load"
    ),
    (CellState.LOADED, CellState.ACTIVATED): StateTransition(
        CellState.LOADED, CellState.ACTIVATED, "activate"
    ),
    (CellState.LOADED, CellState.UNLOADING): StateTransition(
        CellState.LOADED, CellState.UNLOADING, "begin_unload"
    ),
    (CellState.ACTIVATED, CellState.LOADED): StateTransition(
        CellState.ACTIVATED, CellState.LOADED, "deactivate"
    ),
    (CellState.ACTIVATED, CellState.UNLOADING): StateTransition(
        CellState.ACTIVATED, CellState.UNLOADING, "begin_unload"
    ),
    (CellState.UNLOADING, CellState.UNLOADED): StateTransition(
        CellState.UNLOADING, CellState.UNLOADED, "complete_unload"
    ),
}


def get_valid_transitions_from(state: CellState) -> List[CellState]:
    """Get all valid target states from a given state."""
    return [
        to_state
        for (from_state, to_state) in VALID_TRANSITIONS.keys()
        if from_state == state
    ]


def is_valid_transition(from_state: CellState, to_state: CellState) -> bool:
    """Check if a state transition is valid."""
    return (from_state, to_state) in VALID_TRANSITIONS


@dataclass
class CellStateMachine:
    """
    State machine for managing cell lifecycle.

    Handles state transitions, validation, callbacks, and timing for
    streaming cell states.
    """

    # Cell identification
    cell_x: int = 0
    cell_y: int = 0

    # Current state
    _state: CellState = field(default=CellState.UNLOADED)

    # Timing
    state_enter_time: float = 0.0
    load_start_time: float = 0.0
    load_complete_time: float = 0.0
    activate_time: float = 0.0
    last_access_time: float = 0.0

    # Progress tracking
    load_progress: float = 0.0

    # Callbacks
    _on_state_enter: Dict[CellState, List[StateTransitionCallback]] = field(
        default_factory=dict
    )
    _on_state_exit: Dict[CellState, List[StateTransitionCallback]] = field(
        default_factory=dict
    )
    _on_transition: List[StateTransitionCallback] = field(default_factory=list)

    # Error handling
    _last_error: Optional[CellStateError] = field(default=None, repr=False)
    _transition_history: List[Tuple[CellState, CellState, float]] = field(
        default_factory=list, repr=False
    )

    # Configuration
    max_history_length: int = 100
    validate_transitions: bool = True

    def __post_init__(self) -> None:
        """Initialize callback dictionaries."""
        if not self._on_state_enter:
            self._on_state_enter = {state: [] for state in CellState}
        if not self._on_state_exit:
            self._on_state_exit = {state: [] for state in CellState}
        if not self._transition_history:
            self._transition_history = []
        self.state_enter_time = time.time()

    @property
    def state(self) -> CellState:
        """Get the current state."""
        return self._state

    @property
    def cell_id(self) -> Tuple[int, int]:
        """Get the cell ID as a tuple."""
        return (self.cell_x, self.cell_y)

    @property
    def is_loaded(self) -> bool:
        """Check if cell data is in memory."""
        return self._state.is_in_memory

    @property
    def is_active(self) -> bool:
        """Check if cell is activated."""
        return self._state == CellState.ACTIVATED

    @property
    def is_transitioning(self) -> bool:
        """Check if cell is in a transitional state."""
        return self._state.is_transitioning

    @property
    def can_tick(self) -> bool:
        """Check if cell can receive tick updates."""
        return self._state.can_tick

    @property
    def time_in_state(self) -> float:
        """Get time spent in current state."""
        return time.time() - self.state_enter_time

    @property
    def age_since_load(self) -> float:
        """Get time since load completed."""
        if self.load_complete_time <= 0:
            return 0.0
        return time.time() - self.load_complete_time

    @property
    def age_since_activate(self) -> float:
        """Get time since activation."""
        if self.activate_time <= 0:
            return 0.0
        return time.time() - self.activate_time

    def _transition_to(
        self,
        new_state: CellState,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Internal method to perform state transition.

        Args:
            new_state: Target state
            timestamp: Optional timestamp for the transition

        Returns:
            True if transition succeeded

        Raises:
            CellStateError: If transition is invalid and validate_transitions is True
        """
        old_state = self._state

        if self.validate_transitions and not is_valid_transition(old_state, new_state):
            error = CellStateError(
                f"Invalid state transition from {old_state.name} to {new_state.name}",
                current_state=old_state,
                attempted_state=new_state,
                cell_id=self.cell_id,
            )
            self._last_error = error
            raise error

        if timestamp is None:
            timestamp = time.time()

        # Fire exit callbacks
        for callback in self._on_state_exit.get(old_state, []):
            callback(self, old_state, new_state)

        # Update state
        self._state = new_state
        self.state_enter_time = timestamp

        # Record history
        self._transition_history.append((old_state, new_state, timestamp))
        if len(self._transition_history) > self.max_history_length:
            self._transition_history = self._transition_history[-self.max_history_length:]

        # Fire enter callbacks
        for callback in self._on_state_enter.get(new_state, []):
            callback(self, old_state, new_state)

        # Fire general transition callbacks
        for callback in self._on_transition:
            callback(self, old_state, new_state)

        return True

    def begin_load(self, timestamp: Optional[float] = None) -> bool:
        """
        Begin loading the cell.

        Transitions: UNLOADED -> LOADING

        Args:
            timestamp: Optional timestamp for the transition

        Returns:
            True if transition succeeded
        """
        if self._state != CellState.UNLOADED:
            if self.validate_transitions:
                raise CellStateError(
                    f"Cannot begin load from state {self._state.name}",
                    current_state=self._state,
                    attempted_state=CellState.LOADING,
                    cell_id=self.cell_id,
                )
            return False

        if timestamp is None:
            timestamp = time.time()

        self.load_start_time = timestamp
        self.load_progress = 0.0
        return self._transition_to(CellState.LOADING, timestamp)

    def update_load_progress(self, progress: float) -> None:
        """
        Update the loading progress.

        Args:
            progress: Progress value from 0.0 to 1.0
        """
        self.load_progress = max(0.0, min(1.0, progress))
        self.last_access_time = time.time()

    def complete_load(self, timestamp: Optional[float] = None) -> bool:
        """
        Complete the cell loading.

        Transitions: LOADING -> LOADED

        Args:
            timestamp: Optional timestamp for the transition

        Returns:
            True if transition succeeded
        """
        if self._state != CellState.LOADING:
            if self.validate_transitions:
                raise CellStateError(
                    f"Cannot complete load from state {self._state.name}",
                    current_state=self._state,
                    attempted_state=CellState.LOADED,
                    cell_id=self.cell_id,
                )
            return False

        if timestamp is None:
            timestamp = time.time()

        self.load_complete_time = timestamp
        self.load_progress = 1.0
        return self._transition_to(CellState.LOADED, timestamp)

    def cancel_load(self, timestamp: Optional[float] = None) -> bool:
        """
        Cancel an in-progress load.

        Transitions: LOADING -> UNLOADED

        Args:
            timestamp: Optional timestamp for the transition

        Returns:
            True if transition succeeded
        """
        if self._state != CellState.LOADING:
            if self.validate_transitions:
                raise CellStateError(
                    f"Cannot cancel load from state {self._state.name}",
                    current_state=self._state,
                    attempted_state=CellState.UNLOADED,
                    cell_id=self.cell_id,
                )
            return False

        self.load_progress = 0.0
        self.load_start_time = 0.0
        return self._transition_to(CellState.UNLOADED, timestamp)

    def activate(self, timestamp: Optional[float] = None) -> bool:
        """
        Activate the cell for gameplay.

        Transitions: LOADED -> ACTIVATED

        Args:
            timestamp: Optional timestamp for the transition

        Returns:
            True if transition succeeded
        """
        if self._state != CellState.LOADED:
            if self.validate_transitions:
                raise CellStateError(
                    f"Cannot activate from state {self._state.name}",
                    current_state=self._state,
                    attempted_state=CellState.ACTIVATED,
                    cell_id=self.cell_id,
                )
            return False

        if timestamp is None:
            timestamp = time.time()

        self.activate_time = timestamp
        return self._transition_to(CellState.ACTIVATED, timestamp)

    def deactivate(self, timestamp: Optional[float] = None) -> bool:
        """
        Deactivate the cell but keep data loaded.

        Transitions: ACTIVATED -> LOADED

        Args:
            timestamp: Optional timestamp for the transition

        Returns:
            True if transition succeeded
        """
        if self._state != CellState.ACTIVATED:
            if self.validate_transitions:
                raise CellStateError(
                    f"Cannot deactivate from state {self._state.name}",
                    current_state=self._state,
                    attempted_state=CellState.LOADED,
                    cell_id=self.cell_id,
                )
            return False

        return self._transition_to(CellState.LOADED, timestamp)

    def begin_unload(self, timestamp: Optional[float] = None) -> bool:
        """
        Begin unloading the cell.

        Transitions: LOADED -> UNLOADING or ACTIVATED -> UNLOADING

        Args:
            timestamp: Optional timestamp for the transition

        Returns:
            True if transition succeeded
        """
        if self._state not in (CellState.LOADED, CellState.ACTIVATED):
            if self.validate_transitions:
                raise CellStateError(
                    f"Cannot begin unload from state {self._state.name}",
                    current_state=self._state,
                    attempted_state=CellState.UNLOADING,
                    cell_id=self.cell_id,
                )
            return False

        # If activated, deactivate first internally
        if self._state == CellState.ACTIVATED:
            self._state = CellState.LOADED

        return self._transition_to(CellState.UNLOADING, timestamp)

    def complete_unload(self, timestamp: Optional[float] = None) -> bool:
        """
        Complete the cell unloading.

        Transitions: UNLOADING -> UNLOADED

        Args:
            timestamp: Optional timestamp for the transition

        Returns:
            True if transition succeeded
        """
        if self._state != CellState.UNLOADING:
            if self.validate_transitions:
                raise CellStateError(
                    f"Cannot complete unload from state {self._state.name}",
                    current_state=self._state,
                    attempted_state=CellState.UNLOADED,
                    cell_id=self.cell_id,
                )
            return False

        # Reset timing
        self.load_complete_time = 0.0
        self.activate_time = 0.0
        self.load_progress = 0.0

        return self._transition_to(CellState.UNLOADED, timestamp)

    def force_state(self, state: CellState, timestamp: Optional[float] = None) -> None:
        """
        Force the state machine to a specific state (bypasses validation).

        Use with caution - this can leave the system in an inconsistent state.

        Args:
            state: Target state
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = time.time()

        old_state = self._state
        self._state = state
        self.state_enter_time = timestamp

        self._transition_history.append((old_state, state, timestamp))
        if len(self._transition_history) > self.max_history_length:
            self._transition_history = self._transition_history[-self.max_history_length:]

    def on_state_enter(
        self,
        state: CellState,
        callback: StateTransitionCallback,
    ) -> None:
        """
        Register a callback for entering a state.

        Args:
            state: State to listen for
            callback: Callback function (machine, old_state, new_state)
        """
        if state not in self._on_state_enter:
            self._on_state_enter[state] = []
        self._on_state_enter[state].append(callback)

    def on_state_exit(
        self,
        state: CellState,
        callback: StateTransitionCallback,
    ) -> None:
        """
        Register a callback for exiting a state.

        Args:
            state: State to listen for
            callback: Callback function (machine, old_state, new_state)
        """
        if state not in self._on_state_exit:
            self._on_state_exit[state] = []
        self._on_state_exit[state].append(callback)

    def on_transition(self, callback: StateTransitionCallback) -> None:
        """
        Register a callback for any state transition.

        Args:
            callback: Callback function (machine, old_state, new_state)
        """
        self._on_transition.append(callback)

    def remove_callback(self, callback: StateTransitionCallback) -> bool:
        """
        Remove a registered callback.

        Args:
            callback: Callback to remove

        Returns:
            True if callback was found and removed
        """
        removed = False

        for callbacks in self._on_state_enter.values():
            if callback in callbacks:
                callbacks.remove(callback)
                removed = True

        for callbacks in self._on_state_exit.values():
            if callback in callbacks:
                callbacks.remove(callback)
                removed = True

        if callback in self._on_transition:
            self._on_transition.remove(callback)
            removed = True

        return removed

    def clear_callbacks(self) -> None:
        """Remove all registered callbacks."""
        for callbacks in self._on_state_enter.values():
            callbacks.clear()
        for callbacks in self._on_state_exit.values():
            callbacks.clear()
        self._on_transition.clear()

    def get_transition_history(
        self,
        limit: Optional[int] = None,
    ) -> List[Tuple[CellState, CellState, float]]:
        """
        Get the transition history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of (old_state, new_state, timestamp) tuples
        """
        if limit is None:
            return list(self._transition_history)
        return self._transition_history[-limit:]

    def get_valid_transitions(self) -> List[CellState]:
        """Get all valid target states from current state."""
        return get_valid_transitions_from(self._state)

    def can_transition_to(self, state: CellState) -> bool:
        """Check if transition to a state is valid."""
        return is_valid_transition(self._state, state)

    def reset(self) -> None:
        """Reset the state machine to initial state."""
        self._state = CellState.UNLOADED
        self.state_enter_time = time.time()
        self.load_start_time = 0.0
        self.load_complete_time = 0.0
        self.activate_time = 0.0
        self.last_access_time = 0.0
        self.load_progress = 0.0
        self._last_error = None
        self._transition_history.clear()

    def __repr__(self) -> str:
        return (
            f"CellStateMachine(cell_id={self.cell_id}, "
            f"state={self._state.name}, "
            f"time_in_state={self.time_in_state:.2f}s)"
        )


__all__ = [
    "CellState",
    "CellStateError",
    "CellStateMachine",
    "StateTransition",
    "StateTransitionCallback",
    "VALID_TRANSITIONS",
    "get_valid_transitions_from",
    "is_valid_transition",
]
