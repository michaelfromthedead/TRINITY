"""
State machine decorators — built from Ops.

These decorators define state machines with transitions, entry/exit hooks.

Decorators:
    @state_machine  - Mark class as a state machine
    @on_enter       - Hook called when entering a state
    @on_exit        - Hook called when exiting a state
"""

from __future__ import annotations

from typing import Any

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_state_machine(
    initial: str = "", states: Any = None, transitions: Any = None, **_: Any
) -> None:
    if not initial:
        raise ValueError("@state_machine: 'initial' parameter is required")
    if states is None or not states:
        raise ValueError("@state_machine: 'states' parameter is required and must be non-empty")
    if not isinstance(states, (set, frozenset, list, tuple)):
        raise ValueError("@state_machine: 'states' must be a set of strings")
    states_set = set(states)
    if initial not in states_set:
        raise ValueError(
            f"@state_machine: initial state '{initial}' is not in states {sorted(states_set)}"
        )
    if transitions is not None:
        if not isinstance(transitions, dict):
            raise ValueError("@state_machine: 'transitions' must be a dict")
        for src, targets in transitions.items():
            if src not in states_set:
                raise ValueError(
                    f"@state_machine: transition source '{src}' is not in states {sorted(states_set)}"
                )
            for tgt in targets:
                if tgt not in states_set:
                    raise ValueError(
                        f"@state_machine: transition target '{tgt}' is not in states {sorted(states_set)}"
                    )


def _validate_on_enter(state: str = "", **_: Any) -> None:
    if not state:
        raise ValueError("@on_enter: 'state' parameter is required")


def _validate_on_exit(state: str = "", **_: Any) -> None:
    if not state:
        raise ValueError("@on_exit: 'state' parameter is required")


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _state_machine_steps(params: dict[str, Any]) -> list[Step]:
    initial = params.get("initial", "")
    states = set(params.get("states", set()))
    transitions = params.get("transitions", {})
    return [
        Step(Op.TAG, {"key": "state_machine", "value": True}),
        Step(Op.TAG, {"key": "sm_initial", "value": initial}),
        Step(Op.TAG, {"key": "sm_states", "value": frozenset(states)}),
        Step(Op.TAG, {"key": "sm_transitions", "value": dict(transitions)}),
        Step(Op.REGISTER, {"registry": "state_machine"}),
    ]


def _on_enter_steps(params: dict[str, Any]) -> list[Step]:
    state = params.get("state", "")
    return [
        Step(Op.TAG, {"key": "on_enter_state", "value": state}),
        Step(Op.TAG, {"key": "lifecycle_hook", "value": "enter"}),
        Step(Op.REGISTER, {"registry": "state_machine"}),
    ]


def _on_exit_steps(params: dict[str, Any]) -> list[Step]:
    state = params.get("state", "")
    return [
        Step(Op.TAG, {"key": "on_exit_state", "value": state}),
        Step(Op.TAG, {"key": "lifecycle_hook", "value": "exit"}),
        Step(Op.REGISTER, {"registry": "state_machine"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_state_machine(target: Any, params: dict[str, Any]) -> Any:
    initial = params.get("initial", "")
    states = set(params.get("states", set()))
    transitions = params.get("transitions", {})
    target._state_machine = True
    target._sm_initial = initial
    target._sm_states = frozenset(states)
    target._sm_transitions = dict(transitions)
    target._sm_current_state = initial
    return None


def _after_on_enter(target: Any, params: dict[str, Any]) -> Any:
    state = params.get("state", "")
    target._on_enter_state = state
    target._lifecycle_hook = "enter"
    return None


def _after_on_exit(target: Any, params: dict[str, Any]) -> Any:
    state = params.get("state", "")
    target._on_exit_state = state
    target._lifecycle_hook = "exit"
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


state_machine = make_decorator(
    name="state_machine",
    steps=_state_machine_steps,
    doc="Mark class as a state machine with states and transitions.",
    validate=_validate_state_machine,
    after_steps=_after_state_machine,
)

on_enter = make_decorator(
    name="on_enter",
    steps=_on_enter_steps,
    doc="Hook called when entering a state.",
    validate=_validate_on_enter,
    after_steps=_after_on_enter,
)

on_exit = make_decorator(
    name="on_exit",
    steps=_on_exit_steps,
    doc="Hook called when exiting a state.",
    validate=_validate_on_exit,
    after_steps=_after_on_exit,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("state_machine", state_machine, ("class",)),
    ("on_enter", on_enter, ("function",)),
    ("on_exit", on_exit, ("function",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.STATE_MACHINE,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.STATE_MACHINE].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "state_machine",
    "on_enter",
    "on_exit",
]
