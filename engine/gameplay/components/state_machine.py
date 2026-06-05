"""
State Machine Component - Runtime state machine with transitions and callbacks.

Provides state management for entities including state transitions,
enter/exit callbacks, hierarchical states, and history tracking.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from engine.gameplay.components.constants import StateMachineConstants


class StateMachineError(Exception):
    """Error raised by state machine operations."""
    pass


class StateMachine:
    """
    A runtime state machine implementation.

    Provides state management with:
    - State transitions with validation
    - Enter/exit callbacks for state changes
    - Hierarchical sub-state machines
    - State history tracking
    - State data storage

    Attributes:
        current_state: Current active state name
        previous_state: Previous state name (if any)
        states: Set of all valid states
        history: List of state history
    """

    __slots__ = (
        "__weakref__",
        "_states",
        "_transitions",
        "_current_state",
        "_previous_state",
        "_state_history",
        "_history_limit",
        "_on_enter",
        "_on_exit",
        "_on_state_changed",
        "_state_data",
        "_sub_machines",
        "_parent_machine",
    )

    def __init__(
        self,
        states: set,
        initial: str,
        transitions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Initialize the state machine.

        Args:
            states: Set of valid state names
            initial: Initial state name (must be in states)
            transitions: Optional dict mapping state -> list of allowed target states.
                         If a state maps to an empty list, no transitions are allowed.
                         If a state is not in the dict, all transitions are allowed.

        Raises:
            ValueError: If initial state is not in states
        """
        if initial not in states:
            raise ValueError(f"Initial state '{initial}' not in states {states}")

        self._states = frozenset(states)
        self._transitions = transitions or {}
        self._current_state = initial
        self._previous_state: Optional[str] = None
        self._state_history: List[str] = [initial]
        self._history_limit = StateMachineConstants.DEFAULT_HISTORY_LIMIT

        # Callbacks: state -> list of callbacks
        self._on_enter: Dict[str, List[Callable]] = {s: [] for s in states}
        self._on_exit: Dict[str, List[Callable]] = {s: [] for s in states}
        self._on_state_changed: List[Callable[[str, str], None]] = []

        # State data storage
        self._state_data: Dict[str, Any] = {}

        # Hierarchical state machine support
        self._sub_machines: Dict[str, StateMachine] = {}
        self._parent_machine: Optional[StateMachine] = None

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def current_state(self) -> str:
        """Get current state."""
        return self._current_state

    @property
    def previous_state(self) -> Optional[str]:
        """Get previous state."""
        return self._previous_state

    @property
    def states(self) -> frozenset:
        """Get all states."""
        return self._states

    @property
    def history(self) -> List[str]:
        """Get state history."""
        return list(self._state_history)

    # =========================================================================
    # TRANSITION METHODS
    # =========================================================================

    def can_transition_to(self, target: str) -> bool:
        """
        Check if transition to target state is allowed.

        Args:
            target: Target state to check

        Returns:
            True if transition is allowed
        """
        if target not in self._states:
            return False
        if not self._transitions:
            return True  # No restrictions defined at all

        # If current state is not in transitions dict, allow all
        if self._current_state not in self._transitions:
            return True

        # If current state is in transitions dict, only allow listed targets
        # An empty list means NO transitions are allowed
        allowed = self._transitions[self._current_state]
        return target in allowed

    def transition_to(self, target: str, data: Optional[Dict] = None) -> bool:
        """
        Transition to a target state.

        Args:
            target: Target state name
            data: Optional data to pass to state

        Returns:
            True if transition successful

        Raises:
            StateMachineError: If target state is unknown
        """
        if target not in self._states:
            raise StateMachineError(f"Unknown state: {target}")

        if not self.can_transition_to(target):
            return False

        if target == self._current_state:
            return True  # Already in state

        old_state = self._current_state

        # Call exit callbacks
        for callback in self._on_exit[old_state]:
            callback(old_state, target)

        # Update state
        self._previous_state = old_state
        self._current_state = target

        # Update history
        self._state_history.append(target)
        if len(self._state_history) > self._history_limit:
            self._state_history = self._state_history[-self._history_limit:]

        # Store state data
        if data:
            self._state_data[target] = data

        # Call enter callbacks
        for callback in self._on_enter[target]:
            callback(old_state, target)

        # Notify listeners
        for callback in self._on_state_changed:
            callback(old_state, target)

        return True

    def force_state(self, state: str) -> None:
        """
        Force state without checking transitions or calling callbacks.

        Args:
            state: State to force

        Raises:
            StateMachineError: If state is unknown
        """
        if state not in self._states:
            raise StateMachineError(f"Unknown state: {state}")

        old_state = self._current_state
        self._previous_state = old_state
        self._current_state = state
        self._state_history.append(state)

    def revert_to_previous(self) -> bool:
        """
        Revert to previous state.

        Returns:
            True if reverted successfully, False if no previous state
        """
        if self._previous_state is None:
            return False
        return self.transition_to(self._previous_state)

    # =========================================================================
    # CALLBACK REGISTRATION
    # =========================================================================

    def add_on_enter(self, state: str, callback: Callable) -> None:
        """
        Add callback for entering a state.

        Args:
            state: State name
            callback: Callback function(old_state, new_state)

        Raises:
            StateMachineError: If state is unknown
        """
        if state not in self._states:
            raise StateMachineError(f"Unknown state: {state}")
        self._on_enter[state].append(callback)

    def add_on_exit(self, state: str, callback: Callable) -> None:
        """
        Add callback for exiting a state.

        Args:
            state: State name
            callback: Callback function(old_state, new_state)

        Raises:
            StateMachineError: If state is unknown
        """
        if state not in self._states:
            raise StateMachineError(f"Unknown state: {state}")
        self._on_exit[state].append(callback)

    def on_state_changed(self, callback: Callable[[str, str], None]) -> None:
        """
        Add callback for any state change.

        Args:
            callback: Callback function(old_state, new_state)
        """
        self._on_state_changed.append(callback)

    # =========================================================================
    # STATE DATA
    # =========================================================================

    def get_state_data(self, state: Optional[str] = None) -> Any:
        """
        Get data for a state.

        Args:
            state: State name (default: current state)

        Returns:
            State data or None
        """
        state = state or self._current_state
        return self._state_data.get(state)

    def set_state_data(self, data: Any, state: Optional[str] = None) -> None:
        """
        Set data for a state.

        Args:
            data: Data to store
            state: State name (default: current state)
        """
        state = state or self._current_state
        self._state_data[state] = data

    # =========================================================================
    # HISTORY
    # =========================================================================

    def clear_history(self) -> None:
        """Clear state history, keeping only current state."""
        self._state_history = [self._current_state]

    # =========================================================================
    # HIERARCHICAL STATE MACHINES
    # =========================================================================

    def add_sub_machine(self, state: str, sub_machine: StateMachine) -> None:
        """
        Add a sub-state machine for hierarchical states.

        Args:
            state: Parent state name
            sub_machine: Sub-state machine

        Raises:
            StateMachineError: If state is unknown
        """
        if state not in self._states:
            raise StateMachineError(f"Unknown state: {state}")
        self._sub_machines[state] = sub_machine
        sub_machine._parent_machine = self

    def get_sub_machine(self, state: Optional[str] = None) -> Optional[StateMachine]:
        """
        Get sub-state machine for a state.

        Args:
            state: State name (default: current state)

        Returns:
            Sub-state machine or None
        """
        state = state or self._current_state
        return self._sub_machines.get(state)

    def get_full_state_path(self) -> List[str]:
        """
        Get full state path including sub-states.

        Returns:
            List of state names from root to deepest sub-state
        """
        path = [self._current_state]
        sub = self.get_sub_machine()
        while sub:
            path.append(sub._current_state)
            sub = sub.get_sub_machine()
        return path


__all__ = [
    "StateMachine",
    "StateMachineError",
]
