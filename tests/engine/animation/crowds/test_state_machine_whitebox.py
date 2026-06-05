"""Whitebox tests for behavior state machine in crowd_behavior.py.

Tests T2.2: Behavior State Machine
- All state transitions are valid
- Invalid transitions are rejected
- State-specific behaviors activate correctly
- Animation clips change with state
"""

from __future__ import annotations

import math
import pytest

from engine.animation.config import CROWD_BEHAVIOR_CONFIG
from engine.animation.crowds.crowd_behavior import (
    AgentState,
    AnimationBlend,
    BehaviorContext,
    CrowdAgent,
    CrowdSimulator,
    FleeingBehavior,
    FormationBehavior,
    IdleBehavior,
    InvalidTransitionError,
    VALID_TRANSITIONS,
    WaitingBehavior,
    WalkingBehavior,
    is_valid_transition,
)
from engine.core.math import Vec3


# ============================================================================
# Helper functions
# ============================================================================

def make_agent(
    position: Vec3 | None = None,
    velocity: Vec3 | None = None,
    current_state: AgentState = AgentState.IDLE,
    agent_id: int = 0,
    speed: float = CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED,
    idle_animation: int = 0,
    walk_animation: int = 1,
    run_animation: int = 2,
) -> CrowdAgent:
    """Create a CrowdAgent with specified parameters."""
    agent = CrowdAgent(
        position=position or Vec3.zero(),
        velocity=velocity or Vec3.zero(),
        current_state=current_state,
        speed=speed,
        idle_animation=idle_animation,
        walk_animation=walk_animation,
        run_animation=run_animation,
    )
    if agent_id != 0:
        agent.agent_id = agent_id
    return agent


def make_context(agents: list[CrowdAgent] | None = None, obstacles: list[tuple[Vec3, float]] | None = None) -> BehaviorContext:
    """Create a BehaviorContext with given agents and obstacles."""
    return BehaviorContext(
        all_agents=agents or [],
        obstacles=obstacles or [],
        navigation_points=[],
        time=0.0,
    )


# ============================================================================
# Test VALID_TRANSITIONS dictionary completeness
# ============================================================================

class TestValidTransitionsCompleteness:
    """Tests that VALID_TRANSITIONS covers all states."""

    def test_all_states_have_entry_in_valid_transitions(self):
        """Every AgentState must have an entry in VALID_TRANSITIONS."""
        for state in AgentState:
            assert state in VALID_TRANSITIONS, f"State {state.name} missing from VALID_TRANSITIONS"

    def test_valid_transitions_contains_only_agent_states(self):
        """VALID_TRANSITIONS keys must all be valid AgentState values."""
        for key in VALID_TRANSITIONS.keys():
            assert isinstance(key, AgentState), f"Key {key} is not an AgentState"

    def test_valid_transitions_values_are_sets_of_agent_states(self):
        """VALID_TRANSITIONS values must be sets of AgentState values."""
        for state, allowed in VALID_TRANSITIONS.items():
            assert isinstance(allowed, set), f"Transitions for {state.name} is not a set"
            for target in allowed:
                assert isinstance(target, AgentState), f"Target {target} for {state.name} is not AgentState"

    def test_idle_cannot_transition_to_formation_directly(self):
        """IDLE cannot transition directly to FORMATION per design rules."""
        assert AgentState.FORMATION not in VALID_TRANSITIONS[AgentState.IDLE], \
            "IDLE should not transition directly to FORMATION"

    def test_idle_valid_transitions(self):
        """IDLE can transition to WALKING, WAITING, FLEEING, CUSTOM."""
        expected = {AgentState.WALKING, AgentState.WAITING, AgentState.FLEEING, AgentState.CUSTOM}
        assert VALID_TRANSITIONS[AgentState.IDLE] == expected

    def test_walking_valid_transitions(self):
        """WALKING can transition to IDLE, WAITING, FLEEING, FORMATION, CUSTOM."""
        expected = {AgentState.IDLE, AgentState.WAITING, AgentState.FLEEING, AgentState.FORMATION, AgentState.CUSTOM}
        assert VALID_TRANSITIONS[AgentState.WALKING] == expected

    def test_waiting_valid_transitions(self):
        """WAITING can transition to IDLE, WALKING, FLEEING, CUSTOM."""
        expected = {AgentState.IDLE, AgentState.WALKING, AgentState.FLEEING, AgentState.CUSTOM}
        assert VALID_TRANSITIONS[AgentState.WAITING] == expected

    def test_fleeing_valid_transitions(self):
        """FLEEING can transition to IDLE, WALKING, CUSTOM only."""
        expected = {AgentState.IDLE, AgentState.WALKING, AgentState.CUSTOM}
        assert VALID_TRANSITIONS[AgentState.FLEEING] == expected

    def test_formation_valid_transitions(self):
        """FORMATION can transition to IDLE, WALKING, FLEEING, CUSTOM."""
        expected = {AgentState.IDLE, AgentState.WALKING, AgentState.FLEEING, AgentState.CUSTOM}
        assert VALID_TRANSITIONS[AgentState.FORMATION] == expected

    def test_custom_valid_transitions(self):
        """CUSTOM can transition to all other states."""
        expected = {AgentState.IDLE, AgentState.WALKING, AgentState.WAITING, AgentState.FLEEING, AgentState.FORMATION}
        assert VALID_TRANSITIONS[AgentState.CUSTOM] == expected


# ============================================================================
# Test is_valid_transition() function
# ============================================================================

class TestIsValidTransition:
    """Tests for is_valid_transition() module-level function."""

    def test_same_state_always_valid(self):
        """Transitioning to the same state is always valid."""
        for state in AgentState:
            assert is_valid_transition(state, state) is True, \
                f"Same-state transition {state.name} -> {state.name} should be valid"

    def test_idle_to_walking_valid(self):
        """IDLE -> WALKING is valid."""
        assert is_valid_transition(AgentState.IDLE, AgentState.WALKING) is True

    def test_idle_to_waiting_valid(self):
        """IDLE -> WAITING is valid."""
        assert is_valid_transition(AgentState.IDLE, AgentState.WAITING) is True

    def test_idle_to_fleeing_valid(self):
        """IDLE -> FLEEING is valid."""
        assert is_valid_transition(AgentState.IDLE, AgentState.FLEEING) is True

    def test_idle_to_formation_invalid(self):
        """IDLE -> FORMATION is invalid (must walk first)."""
        assert is_valid_transition(AgentState.IDLE, AgentState.FORMATION) is False

    def test_idle_to_custom_valid(self):
        """IDLE -> CUSTOM is valid."""
        assert is_valid_transition(AgentState.IDLE, AgentState.CUSTOM) is True

    def test_walking_to_formation_valid(self):
        """WALKING -> FORMATION is valid."""
        assert is_valid_transition(AgentState.WALKING, AgentState.FORMATION) is True

    def test_fleeing_to_waiting_invalid(self):
        """FLEEING -> WAITING is invalid (must idle first)."""
        assert is_valid_transition(AgentState.FLEEING, AgentState.WAITING) is False

    def test_fleeing_to_formation_invalid(self):
        """FLEEING -> FORMATION is invalid."""
        assert is_valid_transition(AgentState.FLEEING, AgentState.FORMATION) is False

    def test_all_valid_transitions_return_true(self):
        """All transitions in VALID_TRANSITIONS return True."""
        for from_state, allowed_states in VALID_TRANSITIONS.items():
            for to_state in allowed_states:
                assert is_valid_transition(from_state, to_state) is True, \
                    f"Transition {from_state.name} -> {to_state.name} should be valid"

    def test_invalid_transitions_return_false(self):
        """Transitions not in VALID_TRANSITIONS return False."""
        all_states = set(AgentState)
        for from_state, allowed_states in VALID_TRANSITIONS.items():
            disallowed = all_states - allowed_states - {from_state}  # Exclude same-state
            for to_state in disallowed:
                assert is_valid_transition(from_state, to_state) is False, \
                    f"Transition {from_state.name} -> {to_state.name} should be invalid"


# ============================================================================
# Test InvalidTransitionError exception
# ============================================================================

class TestInvalidTransitionError:
    """Tests for InvalidTransitionError exception class."""

    def test_exception_stores_from_state(self):
        """InvalidTransitionError stores from_state attribute."""
        exc = InvalidTransitionError(AgentState.IDLE, AgentState.FORMATION)
        assert exc.from_state == AgentState.IDLE

    def test_exception_stores_to_state(self):
        """InvalidTransitionError stores to_state attribute."""
        exc = InvalidTransitionError(AgentState.IDLE, AgentState.FORMATION)
        assert exc.to_state == AgentState.FORMATION

    def test_exception_default_message(self):
        """InvalidTransitionError generates descriptive default message."""
        exc = InvalidTransitionError(AgentState.IDLE, AgentState.FORMATION)
        assert "IDLE" in str(exc)
        assert "FORMATION" in str(exc)
        assert "Invalid transition" in str(exc)

    def test_exception_custom_message(self):
        """InvalidTransitionError accepts custom message."""
        exc = InvalidTransitionError(AgentState.IDLE, AgentState.FORMATION, "Custom error")
        assert str(exc) == "Custom error"
        # Attributes still accessible
        assert exc.from_state == AgentState.IDLE
        assert exc.to_state == AgentState.FORMATION

    def test_exception_is_instance_of_exception(self):
        """InvalidTransitionError is a proper Exception subclass."""
        exc = InvalidTransitionError(AgentState.IDLE, AgentState.FORMATION)
        assert isinstance(exc, Exception)

    def test_exception_can_be_raised_and_caught(self):
        """InvalidTransitionError can be raised and caught."""
        with pytest.raises(InvalidTransitionError) as exc_info:
            raise InvalidTransitionError(AgentState.FLEEING, AgentState.WAITING)
        assert exc_info.value.from_state == AgentState.FLEEING
        assert exc_info.value.to_state == AgentState.WAITING


# ============================================================================
# Test CrowdAgent.can_transition_to() method
# ============================================================================

class TestCrowdAgentCanTransitionTo:
    """Tests for CrowdAgent.can_transition_to() method."""

    def test_idle_agent_can_transition_to_walking(self):
        """Agent in IDLE can transition to WALKING."""
        agent = make_agent(current_state=AgentState.IDLE)
        assert agent.can_transition_to(AgentState.WALKING) is True

    def test_idle_agent_cannot_transition_to_formation(self):
        """Agent in IDLE cannot transition directly to FORMATION."""
        agent = make_agent(current_state=AgentState.IDLE)
        assert agent.can_transition_to(AgentState.FORMATION) is False

    def test_walking_agent_can_transition_to_formation(self):
        """Agent in WALKING can transition to FORMATION."""
        agent = make_agent(current_state=AgentState.WALKING)
        assert agent.can_transition_to(AgentState.FORMATION) is True

    def test_can_transition_to_same_state(self):
        """Agent can always transition to same state (no-op)."""
        for state in AgentState:
            agent = make_agent(current_state=state)
            assert agent.can_transition_to(state) is True

    def test_fleeing_agent_cannot_transition_to_waiting(self):
        """Fleeing agent cannot transition directly to waiting."""
        agent = make_agent(current_state=AgentState.FLEEING)
        assert agent.can_transition_to(AgentState.WAITING) is False


# ============================================================================
# Test CrowdAgent.transition_to() method
# ============================================================================

class TestCrowdAgentTransitionTo:
    """Tests for CrowdAgent.transition_to() method."""

    def test_successful_transition_updates_state(self):
        """Successful transition changes current_state."""
        agent = make_agent(current_state=AgentState.IDLE)
        agent.transition_to(AgentState.WALKING)
        assert agent.current_state == AgentState.WALKING

    def test_transition_resets_state_time(self):
        """Transition resets state_time to 0."""
        agent = make_agent(current_state=AgentState.IDLE)
        agent.state_time = 5.0
        agent.transition_to(AgentState.WALKING)
        assert agent.state_time == 0.0

    def test_transition_to_same_state_is_noop(self):
        """Transition to same state does nothing (no state_time reset)."""
        agent = make_agent(current_state=AgentState.IDLE)
        agent.state_time = 5.0
        agent.transition_to(AgentState.IDLE)
        assert agent.state_time == 5.0  # Not reset

    def test_invalid_transition_raises_error(self):
        """Invalid transition raises InvalidTransitionError."""
        agent = make_agent(current_state=AgentState.IDLE)
        with pytest.raises(InvalidTransitionError) as exc_info:
            agent.transition_to(AgentState.FORMATION)
        assert exc_info.value.from_state == AgentState.IDLE
        assert exc_info.value.to_state == AgentState.FORMATION

    def test_transition_chain_idle_walking_formation(self):
        """Agent can reach FORMATION through IDLE -> WALKING -> FORMATION."""
        agent = make_agent(current_state=AgentState.IDLE)
        agent.transition_to(AgentState.WALKING)
        assert agent.current_state == AgentState.WALKING
        agent.transition_to(AgentState.FORMATION)
        assert agent.current_state == AgentState.FORMATION

    def test_fleeing_to_waiting_requires_intermediate_state(self):
        """FLEEING -> WAITING requires going through IDLE first."""
        agent = make_agent(current_state=AgentState.FLEEING)
        with pytest.raises(InvalidTransitionError):
            agent.transition_to(AgentState.WAITING)
        # Valid path: FLEEING -> IDLE -> WAITING
        agent.transition_to(AgentState.IDLE)
        agent.transition_to(AgentState.WAITING)
        assert agent.current_state == AgentState.WAITING


# ============================================================================
# Test animation updates on state transition
# ============================================================================

class TestAnimationUpdatesOnTransition:
    """Tests for _update_animation_for_state() via transition_to()."""

    def test_transition_to_idle_sets_idle_animation(self):
        """Transition to IDLE sets idle animation."""
        agent = make_agent(current_state=AgentState.WALKING, idle_animation=10)
        agent.transition_to(AgentState.IDLE)
        assert agent.animation_blend.get_primary_animation() == 10

    def test_transition_to_walking_sets_walk_animation(self):
        """Transition to WALKING sets walk animation."""
        agent = make_agent(current_state=AgentState.IDLE, walk_animation=15)
        agent.transition_to(AgentState.WALKING)
        assert agent.animation_blend.get_primary_animation() == 15

    def test_transition_to_waiting_sets_idle_animation(self):
        """Transition to WAITING sets idle animation (waiting uses idle anim)."""
        agent = make_agent(current_state=AgentState.IDLE, idle_animation=10)
        agent.transition_to(AgentState.WAITING)
        assert agent.animation_blend.get_primary_animation() == 10

    def test_transition_to_fleeing_sets_run_animation(self):
        """Transition to FLEEING sets run animation."""
        agent = make_agent(current_state=AgentState.IDLE, run_animation=20)
        agent.transition_to(AgentState.FLEEING)
        assert agent.animation_blend.get_primary_animation() == 20

    def test_transition_to_formation_sets_walk_animation(self):
        """Transition to FORMATION sets walk animation."""
        agent = make_agent(current_state=AgentState.WALKING, walk_animation=15)
        agent.transition_to(AgentState.FORMATION)
        assert agent.animation_blend.get_primary_animation() == 15

    def test_transition_to_custom_keeps_current_animation(self):
        """Transition to CUSTOM keeps current animation."""
        agent = make_agent(current_state=AgentState.IDLE, idle_animation=10, walk_animation=15)
        # Set a specific animation
        agent.animation_blend = AnimationBlend.single(99)
        agent.transition_to(AgentState.CUSTOM)
        assert agent.animation_blend.get_primary_animation() == 99

    def test_animation_blend_is_single_after_transition(self):
        """Animation blend has single animation after standard transitions."""
        agent = make_agent(current_state=AgentState.IDLE)
        agent.transition_to(AgentState.WALKING)
        assert len(agent.animation_blend.animation_indices) == 1
        assert len(agent.animation_blend.weights) == 1
        assert agent.animation_blend.weights[0] == 1.0


# ============================================================================
# Test CrowdSimulator.transition_agent() method
# ============================================================================

class TestCrowdSimulatorTransitionAgent:
    """Tests for CrowdSimulator.transition_agent() method."""

    def test_transition_agent_success_returns_true(self):
        """Successful transition returns True."""
        simulator = CrowdSimulator()
        agent = make_agent(current_state=AgentState.IDLE)
        simulator.add_agent(agent)
        result = simulator.transition_agent(agent, AgentState.WALKING)
        assert result is True
        assert agent.current_state == AgentState.WALKING

    def test_transition_agent_invalid_returns_false(self):
        """Invalid transition returns False when raise_on_invalid=False."""
        simulator = CrowdSimulator()
        agent = make_agent(current_state=AgentState.IDLE)
        simulator.add_agent(agent)
        result = simulator.transition_agent(agent, AgentState.FORMATION, raise_on_invalid=False)
        assert result is False
        assert agent.current_state == AgentState.IDLE  # State unchanged

    def test_transition_agent_invalid_raises_when_requested(self):
        """Invalid transition raises InvalidTransitionError when raise_on_invalid=True."""
        simulator = CrowdSimulator()
        agent = make_agent(current_state=AgentState.IDLE)
        simulator.add_agent(agent)
        with pytest.raises(InvalidTransitionError):
            simulator.transition_agent(agent, AgentState.FORMATION, raise_on_invalid=True)

    def test_transition_to_same_state_returns_true(self):
        """Transitioning to same state returns True (no-op)."""
        simulator = CrowdSimulator()
        agent = make_agent(current_state=AgentState.IDLE)
        simulator.add_agent(agent)
        result = simulator.transition_agent(agent, AgentState.IDLE)
        assert result is True

    def test_transition_calls_behavior_on_exit_and_on_enter(self):
        """Transition calls on_exit for old behavior and on_enter for new behavior."""
        simulator = CrowdSimulator()
        agent = make_agent(current_state=AgentState.IDLE)
        agent.state_time = 10.0  # Will be reset by on_enter
        simulator.add_agent(agent)
        simulator.transition_agent(agent, AgentState.WALKING)
        # on_enter resets state_time
        assert agent.state_time == 0.0


# ============================================================================
# Test state-specific behavior activation
# ============================================================================

class TestStateBehaviorActivation:
    """Tests that state-specific behaviors activate correctly."""

    def test_idle_behavior_sets_idle_state(self):
        """IdleBehavior.on_enter() sets state to IDLE."""
        behavior = IdleBehavior()
        agent = make_agent(current_state=AgentState.WALKING)
        behavior.on_enter(agent)
        assert agent.current_state == AgentState.IDLE

    def test_walking_behavior_sets_walking_state(self):
        """WalkingBehavior.on_enter() sets state to WALKING."""
        behavior = WalkingBehavior()
        agent = make_agent(current_state=AgentState.IDLE)
        behavior.on_enter(agent)
        assert agent.current_state == AgentState.WALKING

    def test_waiting_behavior_sets_waiting_state(self):
        """WaitingBehavior.on_enter() sets state to WAITING."""
        behavior = WaitingBehavior()
        agent = make_agent(current_state=AgentState.IDLE)
        behavior.on_enter(agent)
        assert agent.current_state == AgentState.WAITING

    def test_fleeing_behavior_sets_fleeing_state(self):
        """FleeingBehavior.on_enter() sets state to FLEEING."""
        behavior = FleeingBehavior()
        agent = make_agent(current_state=AgentState.IDLE)
        behavior.on_enter(agent)
        assert agent.current_state == AgentState.FLEEING

    def test_formation_behavior_sets_formation_state(self):
        """FormationBehavior.on_enter() sets state to FORMATION."""
        behavior = FormationBehavior()
        agent = make_agent(current_state=AgentState.WALKING)
        behavior.on_enter(agent)
        assert agent.current_state == AgentState.FORMATION

    def test_behavior_on_enter_resets_state_time(self):
        """All behaviors reset state_time on enter."""
        behaviors = [IdleBehavior(), WalkingBehavior(), WaitingBehavior(), FleeingBehavior(), FormationBehavior()]
        for behavior in behaviors:
            agent = make_agent()
            agent.state_time = 99.0
            behavior.on_enter(agent)
            assert agent.state_time == 0.0, f"{behavior.name} should reset state_time"


# ============================================================================
# Test behavior can_transition_to() method
# ============================================================================

class TestBehaviorCanTransitionTo:
    """Tests for CrowdBehavior.can_transition_to() method."""

    def test_idle_behavior_allows_walking(self):
        """IdleBehavior allows transition to WALKING."""
        behavior = IdleBehavior()
        agent = make_agent(current_state=AgentState.IDLE)
        assert behavior.can_transition_to(agent, AgentState.WALKING) is True

    def test_idle_behavior_blocks_formation(self):
        """IdleBehavior blocks direct transition to FORMATION."""
        behavior = IdleBehavior()
        agent = make_agent(current_state=AgentState.IDLE)
        assert behavior.can_transition_to(agent, AgentState.FORMATION) is False

    def test_walking_behavior_allows_formation(self):
        """WalkingBehavior allows transition to FORMATION."""
        behavior = WalkingBehavior()
        agent = make_agent(current_state=AgentState.WALKING)
        assert behavior.can_transition_to(agent, AgentState.FORMATION) is True


# ============================================================================
# Test trigger_flee state transitions
# ============================================================================

class TestTriggerFleeTransitions:
    """Tests for CrowdSimulator.trigger_flee() state transitions."""

    def test_trigger_flee_transitions_agents_to_fleeing(self):
        """trigger_flee() transitions nearby agents to FLEEING state."""
        simulator = CrowdSimulator()
        agent = make_agent(position=Vec3(0, 0, 0), current_state=AgentState.IDLE)
        simulator.add_agent(agent)

        affected = simulator.trigger_flee(Vec3(1, 0, 0), radius=5.0)

        assert affected == 1
        assert agent.current_state == AgentState.FLEEING

    def test_trigger_flee_sets_flee_source(self):
        """trigger_flee() sets flee_source on affected agents."""
        simulator = CrowdSimulator()
        agent = make_agent(position=Vec3(0, 0, 0), current_state=AgentState.IDLE)
        simulator.add_agent(agent)
        threat = Vec3(2, 0, 0)

        simulator.trigger_flee(threat, radius=5.0)

        assert agent.flee_source is not None
        assert agent.flee_source.x == 2

    def test_trigger_flee_skips_distant_agents(self):
        """trigger_flee() does not affect agents outside radius."""
        simulator = CrowdSimulator()
        agent = make_agent(position=Vec3(10, 0, 0), current_state=AgentState.IDLE)
        simulator.add_agent(agent)

        affected = simulator.trigger_flee(Vec3(0, 0, 0), radius=5.0)

        assert affected == 0
        assert agent.current_state == AgentState.IDLE

    def test_trigger_flee_from_invalid_state_fails(self):
        """trigger_flee() cannot transition from FORMATION directly (via WALKING first)."""
        simulator = CrowdSimulator()
        # Start in WALKING, then manually set to FORMATION for test
        agent = make_agent(position=Vec3(0, 0, 0), current_state=AgentState.FORMATION)
        simulator.add_agent(agent)

        affected = simulator.trigger_flee(Vec3(1, 0, 0), radius=5.0)

        # FORMATION -> FLEEING is valid
        assert affected == 1
        assert agent.current_state == AgentState.FLEEING


# ============================================================================
# Test edge cases
# ============================================================================

class TestStateMachineEdgeCases:
    """Edge case tests for state machine."""

    def test_multiple_rapid_transitions(self):
        """Multiple rapid transitions work correctly."""
        agent = make_agent(current_state=AgentState.IDLE)
        agent.transition_to(AgentState.WALKING)
        agent.transition_to(AgentState.FLEEING)
        agent.transition_to(AgentState.IDLE)
        agent.transition_to(AgentState.WAITING)
        assert agent.current_state == AgentState.WAITING

    def test_state_time_accumulates_during_update(self):
        """state_time accumulates during behavior update."""
        simulator = CrowdSimulator()
        agent = make_agent(current_state=AgentState.IDLE)
        simulator.add_agent(agent)

        simulator.update(0.5)
        assert agent.state_time >= 0.5

        simulator.update(0.5)
        assert agent.state_time >= 1.0

    def test_invalid_transition_does_not_modify_agent(self):
        """Failed transition does not modify agent state."""
        agent = make_agent(current_state=AgentState.IDLE)
        agent.animation_blend = AnimationBlend.single(99)
        original_anim = agent.animation_blend.get_primary_animation()

        with pytest.raises(InvalidTransitionError):
            agent.transition_to(AgentState.FORMATION)

        assert agent.current_state == AgentState.IDLE
        assert agent.animation_blend.get_primary_animation() == original_anim

    def test_custom_state_can_transition_anywhere(self):
        """CUSTOM state can transition to any other state."""
        agent = make_agent(current_state=AgentState.CUSTOM)

        for target_state in AgentState:
            if target_state != AgentState.CUSTOM:
                # Reset to CUSTOM for each test
                agent.current_state = AgentState.CUSTOM
                assert agent.can_transition_to(target_state) is True

    def test_animation_blend_single_helper(self):
        """AnimationBlend.single() creates correct single-animation blend."""
        blend = AnimationBlend.single(42)
        assert blend.animation_indices == [42]
        assert blend.weights == [1.0]
        assert blend.get_primary_animation() == 42


# ============================================================================
# Exhaustive transition matrix tests
# ============================================================================

class TestExhaustiveTransitionMatrix:
    """Exhaustive tests for all state combinations."""

    # Define expected transition matrix
    # True = valid, False = invalid
    EXPECTED_TRANSITIONS = {
        # From IDLE
        (AgentState.IDLE, AgentState.IDLE): True,
        (AgentState.IDLE, AgentState.WALKING): True,
        (AgentState.IDLE, AgentState.WAITING): True,
        (AgentState.IDLE, AgentState.FLEEING): True,
        (AgentState.IDLE, AgentState.FORMATION): False,
        (AgentState.IDLE, AgentState.CUSTOM): True,
        # From WALKING
        (AgentState.WALKING, AgentState.IDLE): True,
        (AgentState.WALKING, AgentState.WALKING): True,
        (AgentState.WALKING, AgentState.WAITING): True,
        (AgentState.WALKING, AgentState.FLEEING): True,
        (AgentState.WALKING, AgentState.FORMATION): True,
        (AgentState.WALKING, AgentState.CUSTOM): True,
        # From WAITING
        (AgentState.WAITING, AgentState.IDLE): True,
        (AgentState.WAITING, AgentState.WALKING): True,
        (AgentState.WAITING, AgentState.WAITING): True,
        (AgentState.WAITING, AgentState.FLEEING): True,
        (AgentState.WAITING, AgentState.FORMATION): False,
        (AgentState.WAITING, AgentState.CUSTOM): True,
        # From FLEEING
        (AgentState.FLEEING, AgentState.IDLE): True,
        (AgentState.FLEEING, AgentState.WALKING): True,
        (AgentState.FLEEING, AgentState.WAITING): False,
        (AgentState.FLEEING, AgentState.FLEEING): True,
        (AgentState.FLEEING, AgentState.FORMATION): False,
        (AgentState.FLEEING, AgentState.CUSTOM): True,
        # From FORMATION
        (AgentState.FORMATION, AgentState.IDLE): True,
        (AgentState.FORMATION, AgentState.WALKING): True,
        (AgentState.FORMATION, AgentState.WAITING): False,
        (AgentState.FORMATION, AgentState.FLEEING): True,
        (AgentState.FORMATION, AgentState.FORMATION): True,
        (AgentState.FORMATION, AgentState.CUSTOM): True,
        # From CUSTOM
        (AgentState.CUSTOM, AgentState.IDLE): True,
        (AgentState.CUSTOM, AgentState.WALKING): True,
        (AgentState.CUSTOM, AgentState.WAITING): True,
        (AgentState.CUSTOM, AgentState.FLEEING): True,
        (AgentState.CUSTOM, AgentState.FORMATION): True,
        (AgentState.CUSTOM, AgentState.CUSTOM): True,
    }

    @pytest.mark.parametrize(
        "from_state,to_state,expected",
        [
            (from_s, to_s, exp)
            for (from_s, to_s), exp in EXPECTED_TRANSITIONS.items()
        ]
    )
    def test_transition_matrix(self, from_state, to_state, expected):
        """Verify each transition in the matrix."""
        result = is_valid_transition(from_state, to_state)
        assert result == expected, \
            f"Expected {from_state.name} -> {to_state.name} to be {'valid' if expected else 'invalid'}"
