"""
Comprehensive tests for StateMeta - Metaclass for state machine states.

Tests cover:
- State ID assignment
- Machine-scoped registration
- Duplicate state in same machine raises TypeError
- can_transition (empty transitions = allow all)
- can_transition with explicit transition set
- validate_transitions (detects unknown targets)
- register_with_machine (post-creation)
- get_enter_hook / get_exit_hook
- get_machine_states
- Registry clearing
"""
import pytest

from trinity.metaclasses import StateMeta


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    StateMeta.clear_registry()
    yield
    StateMeta.clear_registry()


def test_state_id_assignment():
    """Test that state IDs are assigned sequentially."""

    class State1(metaclass=StateMeta):
        pass

    class State2(metaclass=StateMeta):
        pass

    class State3(metaclass=StateMeta):
        pass

    assert State1._state_id == 1
    assert State2._state_id == 2
    assert State3._state_id == 3


def test_state_name_assignment():
    """Test that _state_name is set to class name."""

    class IdleState(metaclass=StateMeta):
        pass

    assert IdleState._state_name == "IdleState"


def test_state_qualified_name():
    """Test that state qualified name includes module."""

    class TestState(metaclass=StateMeta):
        pass

    assert "." in TestState._state_qualified_name
    assert TestState._state_qualified_name.endswith(".TestState")


def test_transitions_default():
    """Test that _state_transitions defaults to empty set."""

    class TestState(metaclass=StateMeta):
        pass

    assert TestState._state_transitions == set()


def test_transitions_custom():
    """Test that _state_transitions can be set."""

    class IdleState(metaclass=StateMeta):
        _state_transitions = {"Running", "Jumping"}

    assert "Running" in IdleState._state_transitions
    assert "Jumping" in IdleState._state_transitions


def test_on_enter_hook_default():
    """Test that _state_on_enter defaults to None."""

    class TestState(metaclass=StateMeta):
        pass

    assert TestState._state_on_enter is None


def test_on_enter_hook_custom():
    """Test that _state_on_enter can be set."""

    def enter_handler():
        pass

    class TestState(metaclass=StateMeta):
        _state_on_enter = enter_handler

    assert TestState._state_on_enter is enter_handler


def test_on_exit_hook_default():
    """Test that _state_on_exit defaults to None."""

    class TestState(metaclass=StateMeta):
        pass

    assert TestState._state_on_exit is None


def test_on_exit_hook_custom():
    """Test that _state_on_exit can be set."""

    def exit_handler():
        pass

    class TestState(metaclass=StateMeta):
        _state_on_exit = exit_handler

    assert TestState._state_on_exit is exit_handler


def test_machine_cls_default():
    """Test that _state_machine_cls defaults to None."""

    class TestState(metaclass=StateMeta):
        pass

    assert TestState._state_machine_cls is None


def test_machine_scoped_registration():
    """Test that states are registered per state machine."""

    class Machine1:
        pass

    class Machine2:
        pass

    class StateA(metaclass=StateMeta):
        _state_machine_cls = Machine1

    class StateB(metaclass=StateMeta):
        _state_machine_cls = Machine2

    machine1_states = StateMeta.get_machine_states(Machine1)
    machine2_states = StateMeta.get_machine_states(Machine2)

    assert "StateA" in machine1_states
    assert "StateB" not in machine1_states

    assert "StateB" in machine2_states
    assert "StateA" not in machine2_states


def test_duplicate_state_in_machine_raises():
    """Test that duplicate state name in same machine raises TypeError."""

    class TestMachine:
        pass

    class IdleState(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    with pytest.raises(TypeError, match="already registered"):

        class IdleState(metaclass=StateMeta):
            _state_machine_cls = TestMachine


def test_duplicate_state_different_machine_allowed():
    """Test that same state name in different machines is allowed."""

    class Machine1:
        pass

    class Machine2:
        pass

    # Create first IdleState for Machine1
    IdleState1 = StateMeta("IdleState", (), {"_state_machine_cls": Machine1})

    # Same name but different machine - should not raise
    IdleState2 = StateMeta("IdleState", (), {"_state_machine_cls": Machine2})

    # Both should be registered to their respective machines
    assert "IdleState" in StateMeta.get_machine_states(Machine1)
    assert "IdleState" in StateMeta.get_machine_states(Machine2)


def test_can_transition_empty_set():
    """Test that empty transition set allows all transitions."""

    class IdleState(metaclass=StateMeta):
        _state_transitions = set()

    class RunningState(metaclass=StateMeta):
        pass

    # Empty set means any transition is allowed
    assert StateMeta.can_transition(IdleState, RunningState) is True


def test_can_transition_explicit_set():
    """Test can_transition with explicit allowed transitions."""

    class IdleState(metaclass=StateMeta):
        _state_transitions = {"RunningState", "JumpingState"}

    class RunningState(metaclass=StateMeta):
        pass

    class JumpingState(metaclass=StateMeta):
        pass

    class FallingState(metaclass=StateMeta):
        pass

    assert StateMeta.can_transition(IdleState, RunningState) is True
    assert StateMeta.can_transition(IdleState, JumpingState) is True
    assert StateMeta.can_transition(IdleState, FallingState) is False


def test_can_transition_no_transitions_attr():
    """Test can_transition when state has no _state_transitions attribute."""

    class TestState(metaclass=StateMeta):
        pass

    class OtherState(metaclass=StateMeta):
        pass

    # Default empty set means allow all
    assert StateMeta.can_transition(TestState, OtherState) is True


def test_validate_transitions_valid():
    """Test validate_transitions with all valid transitions."""

    class TestMachine:
        pass

    class IdleState(metaclass=StateMeta):
        _state_machine_cls = TestMachine
        _state_transitions = {"RunningState"}

    class RunningState(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    errors = StateMeta.validate_transitions(TestMachine)

    assert errors == []


def test_validate_transitions_invalid():
    """Test validate_transitions detects unknown target states."""

    class TestMachine:
        pass

    class IdleState(metaclass=StateMeta):
        _state_machine_cls = TestMachine
        _state_transitions = {"RunningState", "NonExistentState"}

    class RunningState(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    errors = StateMeta.validate_transitions(TestMachine)

    # Should have one error for "NonExistentState"
    assert len(errors) == 1
    assert "NonExistentState" in errors[0]


def test_validate_transitions_multiple_errors():
    """Test validate_transitions with multiple invalid transitions."""

    class TestMachine:
        pass

    class State1(metaclass=StateMeta):
        _state_machine_cls = TestMachine
        _state_transitions = {"Unknown1", "Unknown2"}

    errors = StateMeta.validate_transitions(TestMachine)

    assert len(errors) == 2


def test_validate_transitions_empty_machine():
    """Test validate_transitions on machine with no states."""

    class EmptyMachine:
        pass

    errors = StateMeta.validate_transitions(EmptyMachine)

    assert errors == []


def test_register_with_machine():
    """Test registering a state with a machine after creation."""

    class TestMachine:
        pass

    class TestState(metaclass=StateMeta):
        pass

    # Initially not registered with machine
    assert TestState._state_machine_cls is None

    # Register with machine
    StateMeta.register_with_machine(TestState, TestMachine)

    assert TestState._state_machine_cls is TestMachine
    assert "TestState" in StateMeta.get_machine_states(TestMachine)


def test_register_with_machine_duplicate_raises():
    """Test that registering duplicate state to machine raises TypeError."""

    class TestMachine:
        pass

    class TestState1(metaclass=StateMeta):
        pass

    class TestState2(metaclass=StateMeta):
        pass

    # Register first state
    StateMeta.register_with_machine(TestState1, TestMachine)

    # Try to register second state with same name - rename it first
    TestState2._state_name = "TestState1"

    with pytest.raises(TypeError, match="already registered"):
        StateMeta.register_with_machine(TestState2, TestMachine)


def test_get_enter_hook():
    """Test get_enter_hook retrieves the enter hook."""

    def enter_fn():
        pass

    class TestState(metaclass=StateMeta):
        _state_on_enter = enter_fn

    hook = StateMeta.get_enter_hook(TestState)
    assert hook is enter_fn


def test_get_enter_hook_none():
    """Test get_enter_hook returns None when no hook is set."""

    class TestState(metaclass=StateMeta):
        pass

    hook = StateMeta.get_enter_hook(TestState)
    assert hook is None


def test_get_exit_hook():
    """Test get_exit_hook retrieves the exit hook."""

    def exit_fn():
        pass

    class TestState(metaclass=StateMeta):
        _state_on_exit = exit_fn

    hook = StateMeta.get_exit_hook(TestState)
    assert hook is exit_fn


def test_get_exit_hook_none():
    """Test get_exit_hook returns None when no hook is set."""

    class TestState(metaclass=StateMeta):
        pass

    hook = StateMeta.get_exit_hook(TestState)
    assert hook is None


def test_get_machine_states():
    """Test get_machine_states returns all states for a machine."""

    class TestMachine:
        pass

    class State1(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    class State2(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    class State3(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    states = StateMeta.get_machine_states(TestMachine)

    assert len(states) == 3
    assert "State1" in states
    assert "State2" in states
    assert "State3" in states


def test_get_machine_states_empty():
    """Test get_machine_states with machine that has no states."""

    class EmptyMachine:
        pass

    states = StateMeta.get_machine_states(EmptyMachine)

    assert states == {}


def test_get_machine_states_returns_copy():
    """Test that get_machine_states returns a copy."""

    class TestMachine:
        pass

    class TestState(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    states1 = StateMeta.get_machine_states(TestMachine)
    states2 = StateMeta.get_machine_states(TestMachine)

    # Should be different dict instances
    assert states1 is not states2
    # But with same content
    assert states1 == states2


def test_get_by_id():
    """Test retrieving state by ID."""

    class TestState(metaclass=StateMeta):
        pass

    retrieved = StateMeta.get_by_id(TestState._state_id)
    assert retrieved is TestState


def test_get_by_name():
    """Test retrieving state by name within a machine."""

    class TestMachine:
        pass

    class TestState(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    retrieved = StateMeta.get_by_name(TestMachine, "TestState")
    assert retrieved is TestState


def test_get_by_name_not_found():
    """Test get_by_name returns None for non-existent state."""

    class TestMachine:
        pass

    result = StateMeta.get_by_name(TestMachine, "NonExistent")
    assert result is None


def test_all_states():
    """Test all_states returns all registered states globally."""

    class Machine1:
        pass

    class Machine2:
        pass

    class State1(metaclass=StateMeta):
        _state_machine_cls = Machine1

    class State2(metaclass=StateMeta):
        _state_machine_cls = Machine2

    class State3(metaclass=StateMeta):
        pass  # No machine

    all_states = StateMeta.all_states()

    assert len(all_states) == 3
    assert State1 in all_states
    assert State2 in all_states
    assert State3 in all_states


def test_clear_registry():
    """Test that clear_registry removes all states."""

    class Machine1:
        pass

    class State1(metaclass=StateMeta):
        _state_machine_cls = Machine1

    class State2(metaclass=StateMeta):
        pass

    assert len(StateMeta.all_states()) == 2

    StateMeta.clear_registry()

    assert len(StateMeta.all_states()) == 0
    assert StateMeta.get_machine_states(Machine1) == {}


def test_clear_registry_resets_id():
    """Test that clear_registry resets ID counter."""

    class State1(metaclass=StateMeta):
        pass

    assert State1._state_id == 1

    StateMeta.clear_registry()

    class State2(metaclass=StateMeta):
        pass

    assert State2._state_id == 1


def test_base_state_class_skipped():
    """Test that base State class is not registered."""

    class State(metaclass=StateMeta):
        pass

    assert len(StateMeta.all_states()) == 0


def test_global_and_machine_registry():
    """Test that states are in both global and machine registries."""

    class TestMachine:
        pass

    class TestState(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    # Should be in global registry
    assert TestState in StateMeta.all_states()

    # Should be in machine registry
    assert "TestState" in StateMeta.get_machine_states(TestMachine)


def test_state_without_machine():
    """Test that states can exist without being assigned to a machine."""

    class TestState(metaclass=StateMeta):
        pass

    # Should be in global registry
    assert TestState in StateMeta.all_states()

    # But not in any machine registry
    assert TestState._state_machine_cls is None


def test_transitions_set_type():
    """Test that _state_transitions is a set."""

    class TestState(metaclass=StateMeta):
        _state_transitions = {"State1", "State2"}

    assert isinstance(TestState._state_transitions, set)


def test_can_transition_uses_state_name():
    """Test that can_transition checks target state's _state_name."""

    class FromState(metaclass=StateMeta):
        _state_transitions = {"ToState"}

    class ToState(metaclass=StateMeta):
        pass

    # Should check ToState._state_name (which is "ToState")
    assert StateMeta.can_transition(FromState, ToState) is True


def test_validate_transitions_checks_state_names():
    """Test that validate_transitions uses state names for lookup."""

    class TestMachine:
        pass

    class State1(metaclass=StateMeta):
        _state_machine_cls = TestMachine
        _state_transitions = {"State2"}  # Valid

    class State2(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    errors = StateMeta.validate_transitions(TestMachine)

    # Should be valid (State2 exists)
    assert errors == []


def test_register_substate_self_registration_raises():
    """Test that registering state as its own substate raises ValueError."""

    class TestState(metaclass=StateMeta):
        pass

    with pytest.raises(ValueError, match="as its own substate"):
        StateMeta.register_substate(TestState, TestState)


def test_register_substate_cycle_detection():
    """Test that register_substate detects and prevents cycles."""

    class TestMachine:
        pass

    class StateA(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    class StateB(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    class StateC(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    # Create chain: A -> B -> C
    StateMeta.register_substate(StateA, StateB)
    StateMeta.register_substate(StateB, StateC)

    # Try to create cycle: C -> A (would make A -> B -> C -> A)
    with pytest.raises(ValueError, match="would create a cycle"):
        StateMeta.register_substate(StateC, StateA)


def test_is_active_in_hierarchy_unregistered_child():
    """Test is_active_in_hierarchy with unregistered child state in deep hierarchy."""

    class TestMachine:
        pass

    class ParentState(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    class ChildState(metaclass=StateMeta):
        _state_machine_cls = TestMachine
        # Manually add a fake grandchild that doesn't exist in registry
        _state_children = {"NonExistentGrandchild"}

    StateMeta.register_substate(ParentState, ChildState)

    # When checking for deep descendant that's declared but not registered,
    # it will match as direct child of ChildState (line 287 check)
    result = StateMeta.is_active_in_hierarchy(
        ChildState, "NonExistentGrandchild", TestMachine
    )
    assert result is True  # Found in direct children

    # But when recursively checking from parent, it will try to resolve and log warning
    # This tests that unregistered children don't crash the recursive check
    result2 = StateMeta.is_active_in_hierarchy(
        ParentState, "SomeOtherState", TestMachine
    )
    assert result2 is False  # Not found anywhere


def test_is_active_in_hierarchy_without_machine_cls():
    """Test is_active_in_hierarchy without machine_cls for recursive checks."""

    class ParentState(metaclass=StateMeta):
        pass

    class ChildState(metaclass=StateMeta):
        pass

    StateMeta.register_substate(ParentState, ChildState)

    # Without machine_cls, can't resolve children recursively
    # Should only check direct children
    result = StateMeta.is_active_in_hierarchy(ParentState, "ChildState", machine_cls=None)

    # Direct child match should still work
    assert result is True


def test_record_transition_unregistered_machine_raises():
    """Test record_transition with unregistered machine raises ValueError."""

    class UnregisteredMachine:
        pass

    # Try to record transition for machine with no states
    with pytest.raises(ValueError, match="unregistered state machine"):
        StateMeta.record_transition(UnregisteredMachine, "StateA", "StateB")


def test_record_transition_thread_safety():
    """Test record_transition handles concurrent calls safely."""
    import threading

    class TestMachine:
        pass

    class StateA(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    class StateB(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    # Record multiple transitions concurrently
    def record_many():
        for i in range(100):
            StateMeta.record_transition(TestMachine, "StateA", f"State{i}")

    threads = [threading.Thread(target=record_many) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # History should be consistent (no corruption)
    history = StateMeta.get_history(TestMachine, limit=1000)

    # Should have some transitions (exact count depends on timing)
    assert len(history) > 0
    # All entries should be strings starting with "State"
    assert all(isinstance(s, str) and s.startswith("State") for s in history)


def test_get_history_empty():
    """Test get_history with empty history returns empty list."""

    class TestMachine:
        pass

    class TestState(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    history = StateMeta.get_history(TestMachine)

    assert history == []


def test_get_history_respects_limit():
    """Test get_history respects limit parameter."""

    class TestMachine:
        pass

    class StateA(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    class StateB(metaclass=StateMeta):
        _state_machine_cls = TestMachine

    # Record 20 transitions
    for i in range(20):
        StateMeta.record_transition(TestMachine, "StateA", f"State{i}", max_history=30)

    # Get last 5
    history = StateMeta.get_history(TestMachine, limit=5)

    assert len(history) == 5
    assert history == ["State15", "State16", "State17", "State18", "State19"]
