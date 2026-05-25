"""
Tests for state machine decorators (state_machine.py).

Tests the 3 state machine decorators built on Ops:
    @state_machine, @on_enter, @on_exit
"""

import pytest

from trinity.decorators.ops import Op, decompose
from trinity.decorators.registry import Tier, registry
from trinity.decorators.state_machine import on_enter, on_exit, state_machine


# =============================================================================
# @state_machine
# =============================================================================


class TestStateMachine:
    def test_basic_application(self):
        @state_machine(initial="idle", states={"idle", "running"}, transitions={"idle": ["running"]})
        class Enemy:
            pass

        assert Enemy._state_machine is True

    def test_initial_state(self):
        @state_machine(initial="idle", states={"idle", "running"})
        class E:
            pass

        assert E._sm_initial == "idle"

    def test_states_stored(self):
        @state_machine(initial="a", states={"a", "b", "c"})
        class E:
            pass

        assert E._sm_states == frozenset({"a", "b", "c"})

    def test_transitions_stored(self):
        t = {"idle": ["running"], "running": ["idle"]}

        @state_machine(initial="idle", states={"idle", "running"}, transitions=t)
        class E:
            pass

        assert E._sm_transitions == t

    def test_current_state_equals_initial(self):
        @state_machine(initial="start", states={"start", "end"})
        class E:
            pass

        assert E._sm_current_state == "start"

    def test_applied_decorators(self):
        @state_machine(initial="a", states={"a"})
        class E:
            pass

        assert "state_machine" in E._applied_decorators

    def test_steps_recorded(self):
        @state_machine(initial="a", states={"a"})
        class E:
            pass

        assert len(E._applied_steps) > 0

    def test_tags_set(self):
        @state_machine(initial="x", states={"x", "y"}, transitions={"x": ["y"]})
        class E:
            pass

        assert E._tags["state_machine"] is True
        assert E._tags["sm_initial"] == "x"
        assert E._tags["sm_states"] == frozenset({"x", "y"})

    def test_registered_in_state_machine_registry(self):
        @state_machine(initial="a", states={"a"})
        class E:
            pass

        assert "state_machine" in E._registries

    def test_no_transitions(self):
        @state_machine(initial="idle", states={"idle", "done"})
        class E:
            pass

        assert E._sm_transitions == {}

    # --- Validation ---

    def test_missing_initial(self):
        with pytest.raises(ValueError, match="'initial' parameter is required"):

            @state_machine(states={"a", "b"})
            class Bad:
                pass

    def test_missing_states(self):
        with pytest.raises(ValueError, match="'states' parameter is required"):

            @state_machine(initial="a")
            class Bad:
                pass

    def test_empty_states(self):
        with pytest.raises(ValueError, match="'states' parameter is required"):

            @state_machine(initial="a", states=set())
            class Bad:
                pass

    def test_initial_not_in_states(self):
        with pytest.raises(ValueError, match="initial state 'x' is not in states"):

            @state_machine(initial="x", states={"a", "b"})
            class Bad:
                pass

    def test_transition_source_not_in_states(self):
        with pytest.raises(ValueError, match="transition source 'z' is not in states"):

            @state_machine(initial="a", states={"a", "b"}, transitions={"z": ["a"]})
            class Bad:
                pass

    def test_transition_target_not_in_states(self):
        with pytest.raises(ValueError, match="transition target 'z' is not in states"):

            @state_machine(initial="a", states={"a", "b"}, transitions={"a": ["z"]})
            class Bad:
                pass

    def test_transitions_not_dict(self):
        with pytest.raises(ValueError, match="'transitions' must be a dict"):

            @state_machine(initial="a", states={"a"}, transitions="bad")
            class Bad:
                pass

    # --- Introspection ---

    def test_decompose_returns_steps(self):
        steps = decompose(state_machine)
        assert isinstance(steps, list)

    def test_multiple_states_complex(self):
        states = {"idle", "walking", "running", "jumping", "falling"}
        transitions = {
            "idle": ["walking", "jumping"],
            "walking": ["idle", "running", "jumping"],
            "running": ["walking", "jumping"],
            "jumping": ["falling"],
            "falling": ["idle"],
        }

        @state_machine(initial="idle", states=states, transitions=transitions)
        class Player:
            pass

        assert Player._sm_states == frozenset(states)
        assert Player._sm_transitions == transitions
        assert Player._sm_current_state == "idle"


# =============================================================================
# @on_enter
# =============================================================================


class TestOnEnter:
    def test_basic_application(self):
        @on_enter(state="running")
        def start_animation():
            pass

        assert start_animation._on_enter_state == "running"

    def test_lifecycle_hook_set(self):
        @on_enter(state="idle")
        def reset():
            pass

        assert reset._lifecycle_hook == "enter"

    def test_tags(self):
        @on_enter(state="combat")
        def enter_combat():
            pass

        assert enter_combat._tags["on_enter_state"] == "combat"
        assert enter_combat._tags["lifecycle_hook"] == "enter"

    def test_applied_decorators(self):
        @on_enter(state="x")
        def f():
            pass

        assert "on_enter" in f._applied_decorators

    def test_steps_recorded(self):
        @on_enter(state="x")
        def f():
            pass

        assert len(f._applied_steps) > 0

    def test_registered(self):
        @on_enter(state="x")
        def f():
            pass

        assert "state_machine" in f._registries

    def test_missing_state(self):
        with pytest.raises(ValueError, match="'state' parameter is required"):

            @on_enter()
            def f():
                pass

    def test_empty_state(self):
        with pytest.raises(ValueError, match="'state' parameter is required"):

            @on_enter(state="")
            def f():
                pass

    def test_decompose(self):
        steps = decompose(on_enter)
        assert isinstance(steps, list)


# =============================================================================
# @on_exit
# =============================================================================


class TestOnExit:
    def test_basic_application(self):
        @on_exit(state="running")
        def stop_animation():
            pass

        assert stop_animation._on_exit_state == "running"

    def test_lifecycle_hook_set(self):
        @on_exit(state="idle")
        def cleanup():
            pass

        assert cleanup._lifecycle_hook == "exit"

    def test_tags(self):
        @on_exit(state="combat")
        def exit_combat():
            pass

        assert exit_combat._tags["on_exit_state"] == "combat"
        assert exit_combat._tags["lifecycle_hook"] == "exit"

    def test_applied_decorators(self):
        @on_exit(state="x")
        def f():
            pass

        assert "on_exit" in f._applied_decorators

    def test_steps_recorded(self):
        @on_exit(state="x")
        def f():
            pass

        assert len(f._applied_steps) > 0

    def test_registered(self):
        @on_exit(state="x")
        def f():
            pass

        assert "state_machine" in f._registries

    def test_missing_state(self):
        with pytest.raises(ValueError, match="'state' parameter is required"):

            @on_exit()
            def f():
                pass

    def test_empty_state(self):
        with pytest.raises(ValueError, match="'state' parameter is required"):

            @on_exit(state="")
            def f():
                pass

    def test_decompose(self):
        steps = decompose(on_exit)
        assert isinstance(steps, list)


# =============================================================================
# Registry
# =============================================================================


class TestStateMachineRegistry:
    def test_state_machine_registered(self):
        spec = registry.get("state_machine")
        assert spec is not None
        assert spec.tier == Tier.STATE_MACHINE

    def test_on_enter_registered(self):
        spec = registry.get("on_enter")
        assert spec is not None
        assert spec.tier == Tier.STATE_MACHINE

    def test_on_exit_registered(self):
        spec = registry.get("on_exit")
        assert spec is not None
        assert spec.tier == Tier.STATE_MACHINE

    def test_state_machine_target_class(self):
        spec = registry.get("state_machine")
        assert "class" in spec.target_types

    def test_on_enter_target_function(self):
        spec = registry.get("on_enter")
        assert "function" in spec.target_types

    def test_on_exit_target_function(self):
        spec = registry.get("on_exit")
        assert "function" in spec.target_types

    def test_tier_has_all_three(self):
        tier_specs = registry.by_tier(Tier.STATE_MACHINE)
        names = {s.name for s in tier_specs}
        assert "state_machine" in names
        assert "on_enter" in names
        assert "on_exit" in names


# =============================================================================
# Stacking
# =============================================================================


class TestStateMachineStacking:
    def test_on_enter_and_on_exit_stack(self):
        @on_exit(state="idle")
        @on_enter(state="idle")
        def handle_idle():
            pass

        assert handle_idle._on_enter_state == "idle"
        assert handle_idle._on_exit_state == "idle"
        assert "on_enter" in handle_idle._applied_decorators
        assert "on_exit" in handle_idle._applied_decorators
