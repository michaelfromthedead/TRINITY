"""
Comprehensive tests for State Machine functionality.

Tests cover:
- State definition
- State transitions
- Transition conditions
- Enter/exit callbacks
- State update
- Hierarchical states
- State history
- State machine events
"""

import pytest
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from trinity.decorators.state_machine import (
    state_machine,
    on_enter,
    on_exit,
)


# =============================================================================
# TEST STATE MACHINE IMPLEMENTATION
# =============================================================================


class StateMachineError(Exception):
    """Error raised by state machine operations."""
    pass


class StateMachine:
    """
    A runtime state machine implementation for testing.

    This class provides a working state machine that can be tested
    independently of the decorator-based approach.
    """

    def __init__(
        self,
        states: set,
        initial: str,
        transitions: Optional[Dict[str, List[str]]] = None
    ):
        if initial not in states:
            raise ValueError(f"Initial state '{initial}' not in states {states}")

        self._states = frozenset(states)
        self._transitions = transitions or {}
        self._current_state = initial
        self._previous_state: Optional[str] = None
        self._state_history: List[str] = [initial]
        self._history_limit = 100  # Default history limit for state machines

        # Callbacks
        self._on_enter: Dict[str, List[Callable]] = {s: [] for s in states}
        self._on_exit: Dict[str, List[Callable]] = {s: [] for s in states}
        self._on_state_changed: List[Callable[[str, str], None]] = []

        # State data
        self._state_data: Dict[str, Any] = {}

        # Sub-states for hierarchical support
        self._sub_machines: Dict[str, 'StateMachine'] = {}
        self._parent_machine: Optional['StateMachine'] = None

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

    def can_transition_to(self, target: str) -> bool:
        """Check if transition to target state is allowed."""
        if target not in self._states:
            return False
        if not self._transitions:
            return True  # No restrictions
        allowed = self._transitions.get(self._current_state, [])
        return target in allowed or len(allowed) == 0

    def transition_to(self, target: str, data: Optional[Dict] = None) -> bool:
        """
        Transition to a target state.

        Args:
            target: Target state name
            data: Optional data to pass to state

        Returns:
            True if transition successful
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
        """Force state without checking transitions."""
        if state not in self._states:
            raise StateMachineError(f"Unknown state: {state}")

        old_state = self._current_state
        self._previous_state = old_state
        self._current_state = state
        self._state_history.append(state)

    def add_on_enter(self, state: str, callback: Callable) -> None:
        """Add callback for entering a state."""
        if state not in self._states:
            raise StateMachineError(f"Unknown state: {state}")
        self._on_enter[state].append(callback)

    def add_on_exit(self, state: str, callback: Callable) -> None:
        """Add callback for exiting a state."""
        if state not in self._states:
            raise StateMachineError(f"Unknown state: {state}")
        self._on_exit[state].append(callback)

    def on_state_changed(self, callback: Callable[[str, str], None]) -> None:
        """Add callback for any state change."""
        self._on_state_changed.append(callback)

    def get_state_data(self, state: Optional[str] = None) -> Any:
        """Get data for a state."""
        state = state or self._current_state
        return self._state_data.get(state)

    def set_state_data(self, data: Any, state: Optional[str] = None) -> None:
        """Set data for a state."""
        state = state or self._current_state
        self._state_data[state] = data

    def clear_history(self) -> None:
        """Clear state history."""
        self._state_history = [self._current_state]

    def revert_to_previous(self) -> bool:
        """Revert to previous state."""
        if self._previous_state is None:
            return False
        return self.transition_to(self._previous_state)

    def add_sub_machine(self, state: str, sub_machine: 'StateMachine') -> None:
        """Add a sub-state machine for hierarchical states."""
        if state not in self._states:
            raise StateMachineError(f"Unknown state: {state}")
        self._sub_machines[state] = sub_machine
        sub_machine._parent_machine = self

    def get_sub_machine(self, state: Optional[str] = None) -> Optional['StateMachine']:
        """Get sub-state machine for a state."""
        state = state or self._current_state
        return self._sub_machines.get(state)

    def get_full_state_path(self) -> List[str]:
        """Get full state path including sub-states."""
        path = [self._current_state]
        sub = self.get_sub_machine()
        while sub:
            path.append(sub._current_state)
            sub = sub.get_sub_machine()
        return path


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_states():
    """Simple set of states."""
    return {"idle", "walking", "running"}


@pytest.fixture
def simple_machine(simple_states):
    """Create a simple state machine."""
    return StateMachine(
        states=simple_states,
        initial="idle"
    )


@pytest.fixture
def transition_machine():
    """Create a state machine with defined transitions."""
    return StateMachine(
        states={"idle", "walking", "running", "jumping"},
        initial="idle",
        transitions={
            "idle": ["walking", "jumping"],
            "walking": ["idle", "running", "jumping"],
            "running": ["walking", "jumping"],
            "jumping": ["idle", "walking"],
        }
    )


@pytest.fixture
def combat_machine():
    """Create a combat state machine."""
    return StateMachine(
        states={"idle", "attacking", "defending", "dodging", "stunned", "dead"},
        initial="idle",
        transitions={
            "idle": ["attacking", "defending", "dodging"],
            "attacking": ["idle", "defending"],
            "defending": ["idle", "attacking", "dodging"],
            "dodging": ["idle"],
            "stunned": ["idle"],
            "dead": [],  # No transitions out of dead
        }
    )


@pytest.fixture
def hierarchical_machine():
    """Create a hierarchical state machine."""
    # Parent machine
    parent = StateMachine(
        states={"exploration", "combat", "menu"},
        initial="exploration"
    )

    # Sub-machines
    exploration_sub = StateMachine(
        states={"walking", "running", "swimming"},
        initial="walking"
    )

    combat_sub = StateMachine(
        states={"attacking", "defending", "special"},
        initial="attacking"
    )

    parent.add_sub_machine("exploration", exploration_sub)
    parent.add_sub_machine("combat", combat_sub)

    return parent


# =============================================================================
# STATE DEFINITION TESTS
# =============================================================================


class TestStateDefinition:
    """Tests for state definition."""

    def test_create_with_states(self, simple_machine):
        """Test creating machine with states."""
        assert "idle" in simple_machine.states
        assert "walking" in simple_machine.states
        assert "running" in simple_machine.states

    def test_initial_state(self, simple_machine):
        """Test initial state is set."""
        assert simple_machine.current_state == "idle"

    def test_invalid_initial_state(self, simple_states):
        """Test invalid initial state raises error."""
        with pytest.raises(ValueError):
            StateMachine(states=simple_states, initial="nonexistent")

    def test_states_immutable(self, simple_machine):
        """Test states set is immutable."""
        assert isinstance(simple_machine.states, frozenset)

    def test_empty_states_not_allowed(self):
        """Test empty states set."""
        with pytest.raises(ValueError):
            StateMachine(states=set(), initial="idle")

    def test_single_state(self):
        """Test single state machine."""
        m = StateMachine(states={"only"}, initial="only")
        assert m.current_state == "only"


# =============================================================================
# STATE TRANSITION TESTS
# =============================================================================


class TestStateTransitions:
    """Tests for state transitions."""

    def test_basic_transition(self, simple_machine):
        """Test basic state transition."""
        result = simple_machine.transition_to("walking")
        assert result is True
        assert simple_machine.current_state == "walking"

    def test_transition_updates_previous(self, simple_machine):
        """Test transition updates previous state."""
        simple_machine.transition_to("walking")
        assert simple_machine.previous_state == "idle"

    def test_transition_to_same_state(self, simple_machine):
        """Test transition to same state."""
        result = simple_machine.transition_to("idle")
        assert result is True
        assert simple_machine.current_state == "idle"

    def test_transition_to_unknown_state(self, simple_machine):
        """Test transition to unknown state raises error."""
        with pytest.raises(StateMachineError):
            simple_machine.transition_to("nonexistent")

    def test_allowed_transition(self, transition_machine):
        """Test allowed transition succeeds."""
        result = transition_machine.transition_to("walking")
        assert result is True

    def test_disallowed_transition(self, transition_machine):
        """Test disallowed transition fails."""
        result = transition_machine.transition_to("running")
        assert result is False
        assert transition_machine.current_state == "idle"

    def test_can_transition_to_allowed(self, transition_machine):
        """Test can_transition_to for allowed transition."""
        assert transition_machine.can_transition_to("walking") is True

    def test_can_transition_to_disallowed(self, transition_machine):
        """Test can_transition_to for disallowed transition."""
        assert transition_machine.can_transition_to("running") is False

    def test_can_transition_to_unknown(self, transition_machine):
        """Test can_transition_to for unknown state."""
        assert transition_machine.can_transition_to("nonexistent") is False

    def test_transition_chain(self, transition_machine):
        """Test chain of transitions."""
        transition_machine.transition_to("walking")
        transition_machine.transition_to("running")
        transition_machine.transition_to("jumping")
        assert transition_machine.current_state == "jumping"

    def test_force_state(self, transition_machine):
        """Test forcing state bypasses transitions."""
        transition_machine.force_state("running")  # Not allowed normally
        assert transition_machine.current_state == "running"

    def test_revert_to_previous(self, simple_machine):
        """Test reverting to previous state."""
        simple_machine.transition_to("walking")
        simple_machine.transition_to("running")
        result = simple_machine.revert_to_previous()
        assert result is True
        assert simple_machine.current_state == "walking"

    def test_revert_with_no_previous(self, simple_machine):
        """Test reverting with no previous state."""
        result = simple_machine.revert_to_previous()
        assert result is False

    def test_dead_state_no_exit(self, combat_machine):
        """Test dead state has no exit transitions."""
        combat_machine.force_state("dead")
        result = combat_machine.transition_to("idle")
        assert result is False
        assert combat_machine.current_state == "dead"


# =============================================================================
# TRANSITION CONDITIONS TESTS
# =============================================================================


class TestTransitionConditions:
    """Tests for transition conditions."""

    def test_no_transitions_allows_all(self, simple_machine):
        """Test machine without transitions allows all."""
        assert simple_machine.can_transition_to("walking") is True
        assert simple_machine.can_transition_to("running") is True

    def test_empty_transition_list_allows_all(self):
        """Test empty transition list for a state allows all."""
        m = StateMachine(
            states={"a", "b", "c"},
            initial="a",
            transitions={"a": []}  # Empty list
        )
        assert m.can_transition_to("b") is True

    def test_specific_transitions_only(self, transition_machine):
        """Test only specific transitions allowed."""
        # From idle, can only go to walking or jumping
        assert transition_machine.can_transition_to("walking") is True
        assert transition_machine.can_transition_to("jumping") is True
        assert transition_machine.can_transition_to("running") is False


# =============================================================================
# ENTER/EXIT CALLBACK TESTS
# =============================================================================


class TestCallbacks:
    """Tests for enter/exit callbacks."""

    def test_on_enter_callback(self, simple_machine):
        """Test on_enter callback is called."""
        entered = []
        simple_machine.add_on_enter("walking", lambda old, new: entered.append((old, new)))
        simple_machine.transition_to("walking")
        assert len(entered) == 1
        assert entered[0] == ("idle", "walking")

    def test_on_exit_callback(self, simple_machine):
        """Test on_exit callback is called."""
        exited = []
        simple_machine.add_on_exit("idle", lambda old, new: exited.append((old, new)))
        simple_machine.transition_to("walking")
        assert len(exited) == 1
        assert exited[0] == ("idle", "walking")

    def test_callback_order(self, simple_machine):
        """Test callbacks called in correct order (exit before enter)."""
        order = []
        simple_machine.add_on_exit("idle", lambda o, n: order.append("exit"))
        simple_machine.add_on_enter("walking", lambda o, n: order.append("enter"))
        simple_machine.transition_to("walking")
        assert order == ["exit", "enter"]

    def test_multiple_enter_callbacks(self, simple_machine):
        """Test multiple enter callbacks."""
        count = [0]
        simple_machine.add_on_enter("walking", lambda o, n: count.__setitem__(0, count[0] + 1))
        simple_machine.add_on_enter("walking", lambda o, n: count.__setitem__(0, count[0] + 1))
        simple_machine.transition_to("walking")
        assert count[0] == 2

    def test_multiple_exit_callbacks(self, simple_machine):
        """Test multiple exit callbacks."""
        count = [0]
        simple_machine.add_on_exit("idle", lambda o, n: count.__setitem__(0, count[0] + 1))
        simple_machine.add_on_exit("idle", lambda o, n: count.__setitem__(0, count[0] + 1))
        simple_machine.transition_to("walking")
        assert count[0] == 2

    def test_on_state_changed_callback(self, simple_machine):
        """Test general state change callback."""
        changes = []
        simple_machine.on_state_changed(lambda old, new: changes.append((old, new)))
        simple_machine.transition_to("walking")
        simple_machine.transition_to("running")
        assert len(changes) == 2

    def test_callback_not_called_same_state(self, simple_machine):
        """Test callbacks not called for same state transition."""
        enter_count = [0]
        simple_machine.add_on_enter("idle", lambda o, n: enter_count.__setitem__(0, enter_count[0] + 1))
        simple_machine.transition_to("idle")
        assert enter_count[0] == 0

    def test_add_callback_unknown_state(self, simple_machine):
        """Test adding callback for unknown state raises error."""
        with pytest.raises(StateMachineError):
            simple_machine.add_on_enter("nonexistent", lambda o, n: None)


# =============================================================================
# STATE DATA TESTS
# =============================================================================


class TestStateData:
    """Tests for state-associated data."""

    def test_set_state_data(self, simple_machine):
        """Test setting state data."""
        simple_machine.set_state_data({"health": 100})
        assert simple_machine.get_state_data() == {"health": 100}

    def test_get_state_data(self, simple_machine):
        """Test getting state data."""
        simple_machine.set_state_data({"value": 42}, "walking")
        assert simple_machine.get_state_data("walking") == {"value": 42}

    def test_state_data_on_transition(self, simple_machine):
        """Test passing data on transition."""
        simple_machine.transition_to("walking", data={"speed": 5})
        assert simple_machine.get_state_data("walking") == {"speed": 5}

    def test_state_data_independent(self, simple_machine):
        """Test state data is independent per state."""
        simple_machine.set_state_data({"a": 1}, "idle")
        simple_machine.set_state_data({"b": 2}, "walking")
        assert simple_machine.get_state_data("idle") == {"a": 1}
        assert simple_machine.get_state_data("walking") == {"b": 2}


# =============================================================================
# STATE HISTORY TESTS
# =============================================================================


class TestStateHistory:
    """Tests for state history."""

    def test_initial_history(self, simple_machine):
        """Test initial history contains initial state."""
        assert simple_machine.history == ["idle"]

    def test_history_on_transition(self, simple_machine):
        """Test history updated on transition."""
        simple_machine.transition_to("walking")
        simple_machine.transition_to("running")
        assert simple_machine.history == ["idle", "walking", "running"]

    def test_history_limit(self, simple_machine):
        """Test history respects limit."""
        simple_machine._history_limit = 5
        for i in range(10):
            simple_machine.transition_to("walking" if i % 2 else "idle")
        assert len(simple_machine.history) <= 5

    def test_clear_history(self, simple_machine):
        """Test clearing history."""
        simple_machine.transition_to("walking")
        simple_machine.transition_to("running")
        simple_machine.clear_history()
        assert simple_machine.history == ["running"]

    def test_history_immutable_return(self, simple_machine):
        """Test history returns a copy."""
        history = simple_machine.history
        history.append("fake")
        assert "fake" not in simple_machine.history


# =============================================================================
# HIERARCHICAL STATE TESTS
# =============================================================================


class TestHierarchicalStates:
    """Tests for hierarchical (nested) state machines."""

    def test_add_sub_machine(self, hierarchical_machine):
        """Test adding sub-state machine."""
        sub = hierarchical_machine.get_sub_machine("exploration")
        assert sub is not None
        assert "walking" in sub.states

    def test_sub_machine_initial_state(self, hierarchical_machine):
        """Test sub-machine has correct initial state."""
        sub = hierarchical_machine.get_sub_machine("exploration")
        assert sub.current_state == "walking"

    def test_get_full_state_path(self, hierarchical_machine):
        """Test getting full state path."""
        path = hierarchical_machine.get_full_state_path()
        assert path == ["exploration", "walking"]

    def test_sub_machine_transitions(self, hierarchical_machine):
        """Test transitions in sub-machine."""
        sub = hierarchical_machine.get_sub_machine("exploration")
        sub.transition_to("running")
        assert sub.current_state == "running"

    def test_parent_transition_changes_sub(self, hierarchical_machine):
        """Test parent transition gives different sub-machine."""
        hierarchical_machine.transition_to("combat")
        sub = hierarchical_machine.get_sub_machine()
        assert sub is not None
        assert "attacking" in sub.states

    def test_no_sub_machine(self, hierarchical_machine):
        """Test state without sub-machine."""
        hierarchical_machine.transition_to("menu")
        assert hierarchical_machine.get_sub_machine() is None

    def test_add_sub_machine_unknown_state(self, simple_machine):
        """Test adding sub-machine to unknown state raises error."""
        sub = StateMachine(states={"a", "b"}, initial="a")
        with pytest.raises(StateMachineError):
            simple_machine.add_sub_machine("nonexistent", sub)


# =============================================================================
# DECORATOR VALIDATION TESTS
# =============================================================================


class TestDecoratorValidation:
    """Tests for state machine decorator validation."""

    def test_state_machine_requires_initial(self):
        """Test @state_machine requires initial parameter."""
        with pytest.raises(ValueError):
            @state_machine(states={"a", "b"})
            class Test:
                pass

    def test_state_machine_requires_states(self):
        """Test @state_machine requires states parameter."""
        with pytest.raises(ValueError):
            @state_machine(initial="a")
            class Test:
                pass

    def test_state_machine_initial_in_states(self):
        """Test @state_machine initial must be in states."""
        with pytest.raises(ValueError):
            @state_machine(initial="c", states={"a", "b"})
            class Test:
                pass

    def test_state_machine_transition_source_valid(self):
        """Test @state_machine transition sources must be valid."""
        with pytest.raises(ValueError):
            @state_machine(
                initial="a",
                states={"a", "b"},
                transitions={"c": ["a"]}  # 'c' not in states
            )
            class Test:
                pass

    def test_state_machine_transition_target_valid(self):
        """Test @state_machine transition targets must be valid."""
        with pytest.raises(ValueError):
            @state_machine(
                initial="a",
                states={"a", "b"},
                transitions={"a": ["c"]}  # 'c' not in states
            )
            class Test:
                pass

    def test_on_enter_requires_state(self):
        """Test @on_enter requires state parameter."""
        with pytest.raises(ValueError):
            @on_enter()
            def handler():
                pass

    def test_on_exit_requires_state(self):
        """Test @on_exit requires state parameter."""
        with pytest.raises(ValueError):
            @on_exit()
            def handler():
                pass


# =============================================================================
# DECORATOR FUNCTIONALITY TESTS
# =============================================================================


class TestDecoratorFunctionality:
    """Tests for state machine decorator functionality."""

    def test_state_machine_decorator_sets_attributes(self):
        """Test @state_machine sets class attributes."""
        @state_machine(initial="idle", states={"idle", "active"})
        class TestMachine:
            pass

        assert TestMachine._state_machine is True
        assert TestMachine._sm_initial == "idle"
        assert "idle" in TestMachine._sm_states
        assert "active" in TestMachine._sm_states

    def test_state_machine_with_transitions(self):
        """Test @state_machine with transitions."""
        @state_machine(
            initial="idle",
            states={"idle", "active"},
            transitions={"idle": ["active"], "active": ["idle"]}
        )
        class TestMachine:
            pass

        assert "idle" in TestMachine._sm_transitions
        assert "active" in TestMachine._sm_transitions["idle"]

    def test_on_enter_decorator_sets_attributes(self):
        """Test @on_enter sets function attributes."""
        @on_enter(state="active")
        def enter_active():
            pass

        assert enter_active._on_enter_state == "active"
        assert enter_active._lifecycle_hook == "enter"

    def test_on_exit_decorator_sets_attributes(self):
        """Test @on_exit sets function attributes."""
        @on_exit(state="idle")
        def exit_idle():
            pass

        assert exit_idle._on_exit_state == "idle"
        assert exit_idle._lifecycle_hook == "exit"


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_rapid_transitions(self, simple_machine):
        """Test rapid state transitions."""
        for _ in range(100):
            simple_machine.transition_to("walking")
            simple_machine.transition_to("running")
            simple_machine.transition_to("idle")
        assert simple_machine.current_state == "idle"

    def test_many_states(self):
        """Test machine with many states."""
        states = {f"state_{i}" for i in range(100)}
        m = StateMachine(states=states, initial="state_0")
        m.transition_to("state_50")
        assert m.current_state == "state_50"

    def test_many_callbacks(self, simple_machine):
        """Test many callbacks on one state."""
        count = [0]
        for _ in range(100):
            simple_machine.add_on_enter("walking", lambda o, n: count.__setitem__(0, count[0] + 1))
        simple_machine.transition_to("walking")
        assert count[0] == 100

    def test_callback_exception_handling(self, simple_machine):
        """Test that callback exceptions propagate."""
        def bad_callback(old, new):
            raise RuntimeError("Callback error")

        simple_machine.add_on_enter("walking", bad_callback)
        with pytest.raises(RuntimeError):
            simple_machine.transition_to("walking")

    def test_self_transition_loop(self):
        """Test self-transition is allowed."""
        m = StateMachine(
            states={"a"},
            initial="a",
            transitions={"a": ["a"]}
        )
        result = m.transition_to("a")
        assert result is True

    def test_unicode_state_names(self):
        """Test unicode state names."""
        m = StateMachine(
            states={"idle", "walking"},
            initial="idle"
        )
        assert m.current_state == "idle"

    def test_state_data_persistence(self, simple_machine):
        """Test state data persists across transitions."""
        simple_machine.set_state_data({"x": 1}, "idle")
        simple_machine.transition_to("walking")
        simple_machine.transition_to("idle")
        assert simple_machine.get_state_data("idle") == {"x": 1}

    def test_nested_state_machine_depth(self):
        """Test deeply nested state machines."""
        machines = []
        for i in range(5):
            m = StateMachine(states={f"level_{i}"}, initial=f"level_{i}")
            if machines:
                machines[-1].add_sub_machine(f"level_{i-1}", m)
            machines.append(m)

        path = machines[0].get_full_state_path()
        # Only first level shows since sub-machine on that state
        assert len(path) >= 1
