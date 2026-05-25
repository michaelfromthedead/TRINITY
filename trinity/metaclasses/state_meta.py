"""
StateMeta - Metaclass for state machine states.

Handles state registration and transition validation.
States are used in unified state machines that feed both simulation and presentation.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, ClassVar, Optional

from trinity.constants import DEFAULT_STATE_HISTORY_SIZE
from trinity.decorators.ops import Op, Step
from trinity.metaclasses.engine_meta import EngineMeta

logger = logging.getLogger(__name__)


class StateMeta(EngineMeta):
    """
    Metaclass for state machine states.

    Created classes will:
    - Be registered as valid states
    - Have transitions validated
    - Support enter/exit hooks
    - Be associated with a state machine

    Optional class attributes (set by decorators or class definition):
    - _state_transitions: set[str] (valid transition target state names)
    - _state_on_enter: Callable (enter hook)
    - _state_on_exit: Callable (exit hook)
    - _state_machine_cls: type (parent state machine class)

    Attached attributes:
    - _state_id: int (unique identifier)
    - _state_name: str (state name, typically class name)
    - _state_qualified_name: str (module.Class name)
    """

    _registry: ClassVar[dict[type, dict[str, type]]] = {}  # machine -> {name -> state}
    _global_registry: ClassVar[dict[int, type]] = {}  # id -> state
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _state_history: ClassVar[dict[type, list[str]]] = {}  # machine -> [state_names]

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> StateMeta:
        """Create a new state type."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base State class
        if name == "State":
            return cls

        with mcs._lock:
            # === 1. GENERATE UNIQUE ID ===
            cls._state_id = mcs._next_id
            mcs._next_id += 1
            cls._state_name = name
            cls._state_qualified_name = f"{cls.__module__}.{name}"

            # 3.6.2: TAG steps for state_id and state_name
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "state_id", "value": cls._state_id})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "state_name", "value": cls._state_name})
            )

            # === 2. SET DEFAULTS ===
            if not hasattr(cls, "_state_transitions"):
                cls._state_transitions = set()  # Empty = can transition to any state
            if not hasattr(cls, "_state_on_enter"):
                cls._state_on_enter = None
            if not hasattr(cls, "_state_on_exit"):
                cls._state_on_exit = None
            if not hasattr(cls, "_state_machine_cls"):
                cls._state_machine_cls = None
            if not hasattr(cls, "_state_parent"):
                cls._state_parent = None
            if not hasattr(cls, "_state_children"):
                cls._state_children = set()

            # 3.6.3: TAG for state_transitions
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "state_transitions", "value": cls._state_transitions})
            )

            # 3.6.4: HOOK for on_enter if set
            if cls._state_on_enter is not None:
                cls._metaclass_steps.append(
                    Step(Op.HOOK, {"event": "on_enter", "callback": cls._state_on_enter})
                )

            # 3.6.5: HOOK for on_exit if set
            if cls._state_on_exit is not None:
                cls._metaclass_steps.append(
                    Step(Op.HOOK, {"event": "on_exit", "callback": cls._state_on_exit})
                )

            # === 3. REGISTER GLOBALLY ===
            mcs._global_registry[cls._state_id] = cls

            # 3.6.6: REGISTER step for global registry
            cls._metaclass_steps.append(
                Step(Op.REGISTER, {"registry": "state_global"})
            )

            # === 4. REGISTER WITH STATE MACHINE IF SPECIFIED ===
            machine_cls = cls._state_machine_cls
            if machine_cls is not None:
                if machine_cls not in mcs._registry:
                    mcs._registry[machine_cls] = {}

                if name in mcs._registry[machine_cls]:
                    raise TypeError(
                        f"State '{name}' is already registered with state machine "
                        f"'{machine_cls.__name__}'"
                    )

                mcs._registry[machine_cls][name] = cls

                # 3.6.7: REGISTER step for state machine
                cls._metaclass_steps.append(
                    Step(Op.REGISTER, {"registry": f"state_machine:{machine_cls.__name__}"})
                )

        return cls

    # =========================================================================
    # REGISTRY ACCESS CLASS METHODS
    # =========================================================================

    @classmethod
    def get_by_id(mcs, state_id: int) -> Optional[type]:
        """Get state class by ID."""
        return mcs._global_registry.get(state_id)

    @classmethod
    def get_by_name(mcs, machine_cls: type, name: str) -> Optional[type]:
        """
        Get state class by name within a state machine.

        Args:
            machine_cls: The state machine class.
            name: The state name.

        Returns:
            The state class, or None if not found.
        """
        machine_states = mcs._registry.get(machine_cls, {})
        return machine_states.get(name)

    @classmethod
    def all_states(mcs) -> list[type]:
        """Get all registered state classes."""
        return list(mcs._global_registry.values())

    @classmethod
    def get_machine_states(mcs, machine_cls: type) -> dict[str, type]:
        """
        Get all states registered with a state machine.

        Args:
            machine_cls: The state machine class.

        Returns:
            Dict mapping state name to state class.
        """
        return dict(mcs._registry.get(machine_cls, {}))

    @classmethod
    def can_transition(mcs, from_state: type, to_state: type) -> bool:
        """
        Check if a transition is valid.

        Args:
            from_state: The current state class.
            to_state: The target state class.

        Returns:
            True if the transition is allowed.
        """
        transitions = getattr(from_state, "_state_transitions", set())

        # Empty transitions set means any transition is allowed
        if not transitions:
            return True

        # Check if target state name is in allowed transitions
        return to_state._state_name in transitions

    @classmethod
    def validate_transitions(mcs, machine_cls: type) -> list[str]:
        """
        Validate all transition declarations for a state machine.

        Args:
            machine_cls: The state machine class to validate.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []
        states = mcs.get_machine_states(machine_cls)
        state_names = set(states.keys())

        for state_name, state_cls in states.items():
            transitions = getattr(state_cls, "_state_transitions", set())

            for target_name in transitions:
                if target_name not in state_names:
                    errors.append(
                        f"State '{state_name}' declares transition to unknown state '{target_name}'"
                    )

        return errors

    @classmethod
    def register_with_machine(mcs, state_cls: type, machine_cls: type) -> None:
        """
        Register a state with a state machine after creation.

        Args:
            state_cls: The state class to register.
            machine_cls: The state machine class.
        """
        with mcs._lock:
            state_cls._state_machine_cls = machine_cls

            if machine_cls not in mcs._registry:
                mcs._registry[machine_cls] = {}

            name = state_cls._state_name
            if name in mcs._registry[machine_cls]:
                raise TypeError(
                    f"State '{name}' is already registered with state machine "
                    f"'{machine_cls.__name__}'"
                )

            mcs._registry[machine_cls][name] = state_cls

    @classmethod
    def get_enter_hook(mcs, state_cls: type) -> Optional[Callable]:
        """Get the enter hook for a state."""
        return getattr(state_cls, "_state_on_enter", None)

    @classmethod
    def get_exit_hook(mcs, state_cls: type) -> Optional[Callable]:
        """Get the exit hook for a state."""
        return getattr(state_cls, "_state_on_exit", None)

    @classmethod
    def register_substate(mcs, parent_state: type, child_state: type) -> None:
        """
        Register a child state under a parent state.

        Args:
            parent_state: The parent state class.
            child_state: The child state class to register.

        Raises:
            ValueError: If parent_state and child_state are the same (self-registration).
            ValueError: If registering would create a cycle in the hierarchy.
        """
        # Prevent self-registration
        if parent_state is child_state:
            raise ValueError(
                f"Cannot register state '{parent_state._state_name}' as its own substate"
            )

        # Prevent cycles: check if parent is already a descendant of child
        with mcs._lock:
            if mcs._would_create_cycle(parent_state, child_state):
                raise ValueError(
                    f"Cannot register '{child_state._state_name}' as substate of "
                    f"'{parent_state._state_name}': would create a cycle"
                )

            if not hasattr(parent_state, "_state_children"):
                parent_state._state_children = set()

            parent_state._state_children.add(child_state._state_name)
            child_state._state_parent = parent_state._state_name

    @classmethod
    def _would_create_cycle(mcs, parent_state: type, child_state: type) -> bool:
        """
        Check if registering child under parent would create a cycle.

        Args:
            parent_state: The proposed parent.
            child_state: The proposed child.

        Returns:
            True if this would create a cycle.
        """
        # Check if parent is already a descendant of child
        visited = set()
        current = parent_state

        while current is not None:
            current_name = getattr(current, "_state_name", None)
            if current_name is None:
                break

            # If we find child in parent's ancestry, we'd create a cycle
            if current is child_state:
                return True

            # Prevent infinite loops in already-broken hierarchies
            if current_name in visited:
                break
            visited.add(current_name)

            # Move up to parent
            parent_name = getattr(current, "_state_parent", None)
            if parent_name is None:
                break

            # Try to find parent state class
            machine_cls = getattr(current, "_state_machine_cls", None)
            if machine_cls is None:
                break

            current = mcs.get_by_name(machine_cls, parent_name)

        return False

    @classmethod
    def get_substates(mcs, state_cls: type) -> set[str]:
        """
        Get all child state names for a state.

        Args:
            state_cls: The parent state class.

        Returns:
            Set of child state names.
        """
        return set(getattr(state_cls, "_state_children", set()))

    @classmethod
    def get_parent_state(mcs, state_cls: type) -> Optional[str]:
        """
        Get the parent state name for a state.

        Args:
            state_cls: The state class.

        Returns:
            Parent state name, or None if this is a top-level state.
        """
        return getattr(state_cls, "_state_parent", None)

    @classmethod
    def is_active_in_hierarchy(
        mcs, state_cls: type, current_state: str, machine_cls: Optional[type] = None
    ) -> bool:
        """
        Check if a state is active in the hierarchy.

        Args:
            state_cls: The state class to check.
            current_state: The name of the current active state.
            machine_cls: Optional state machine class for name resolution.

        Returns:
            True if current_state is the state or any of its descendants.
            False if machine_cls is None and recursive checks are needed.
            False if any child state is not found in the registry.
        """
        state_name = state_cls._state_name

        # Direct match
        if current_state == state_name:
            return True

        # Check if current_state is a descendant
        children = mcs.get_substates(state_cls)
        if current_state in children:
            return True

        # Recursively check descendants (requires machine_cls for resolution)
        if machine_cls is not None:
            for child_name in children:
                child_cls = mcs.get_by_name(machine_cls, child_name)
                if child_cls is None:
                    # Child not found in registry - log warning and skip
                    logger.warning(
                        f"Child state '{child_name}' of '{state_name}' not found "
                        f"in machine {machine_cls.__name__}"
                    )
                    continue

                if mcs.is_active_in_hierarchy(child_cls, current_state, machine_cls):
                    return True

        return False

    @classmethod
    def record_transition(
        mcs,
        machine_cls: type,
        from_state: str,
        to_state: str,
        max_history: int = DEFAULT_STATE_HISTORY_SIZE,
    ) -> None:
        """
        Record a state transition in the history.

        Args:
            machine_cls: The state machine class.
            from_state: The state being transitioned from.
            to_state: The state being transitioned to.
            max_history: Maximum number of transitions to keep (default from constants).

        Raises:
            ValueError: If machine_cls has no registered states.
        """
        with mcs._lock:
            # Validate machine is registered
            if machine_cls not in mcs._registry or not mcs._registry[machine_cls]:
                raise ValueError(
                    f"Cannot record transition for unregistered state machine "
                    f"'{machine_cls.__name__}'. Register states with this machine first."
                )

            # Initialize history if needed
            if machine_cls not in mcs._state_history:
                mcs._state_history[machine_cls] = []

            # Append new state - must modify list in place to maintain thread safety
            history = mcs._state_history[machine_cls]
            history.append(to_state)

            # Trim history to max length - reassign entire list for thread safety
            if len(history) > max_history:
                # Create new list to avoid race conditions
                mcs._state_history[machine_cls] = list(history[-max_history:])

    @classmethod
    def get_previous_state(mcs, machine_cls: type) -> Optional[str]:
        """
        Get the previous state before the current one.

        Args:
            machine_cls: The state machine class.

        Returns:
            The previous state name, or None if no history exists.
        """
        history = mcs._state_history.get(machine_cls, [])
        return history[-2] if len(history) >= 2 else None

    @classmethod
    def get_history(mcs, machine_cls: type, limit: int = DEFAULT_STATE_HISTORY_SIZE) -> list[str]:
        """
        Get the state transition history.

        Args:
            machine_cls: The state machine class.
            limit: Maximum number of historical states to return (default from constants).

        Returns:
            List of state names in chronological order (oldest first).
            Empty list if no history exists for the machine.
        """
        with mcs._lock:
            history = mcs._state_history.get(machine_cls, [])
            # Return a copy to prevent external modification
            return list(history[-limit:]) if history else []

    @classmethod
    def clear_registry(mcs) -> None:
        """Clear the state registry. Useful for testing."""
        with mcs._lock:
            mcs._registry.clear()
            mcs._global_registry.clear()
            mcs._next_id = 1
            mcs._state_history.clear()
        super().clear_registry()
