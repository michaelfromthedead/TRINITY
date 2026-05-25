"""
Input decorators — built from Ops.

These decorators register input actions and axes for the input system.

Decorators:
    @input_action  - Register an input action with key bindings
    @input_axis    - Register an input axis with positive/negative bindings
"""

from __future__ import annotations

from typing import Any

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_input_action(
    name: str = "", default_bindings: Any = None, **_: Any
) -> None:
    if not name:
        raise ValueError("@input_action: 'name' parameter is required and must be non-empty")
    if default_bindings is None or len(default_bindings) == 0:
        raise ValueError(
            "@input_action: 'default_bindings' parameter is required and must be non-empty"
        )


def _validate_input_axis(
    name: str = "", positive: Any = None, negative: Any = None, **_: Any
) -> None:
    if not name:
        raise ValueError("@input_axis: 'name' parameter is required and must be non-empty")
    if positive is None or len(positive) == 0:
        raise ValueError(
            "@input_axis: 'positive' parameter is required and must be non-empty"
        )
    if negative is None or len(negative) == 0:
        raise ValueError(
            "@input_axis: 'negative' parameter is required and must be non-empty"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _input_action_steps(params: dict[str, Any]) -> list[Step]:
    name = params.get("name", "")
    bindings = list(params.get("default_bindings", []))
    return [
        Step(Op.TAG, {"key": "input_action", "value": True}),
        Step(Op.TAG, {"key": "action_name", "value": name}),
        Step(Op.TAG, {"key": "action_bindings", "value": bindings}),
        Step(Op.REGISTER, {"registry": "input"}),
    ]


def _input_axis_steps(params: dict[str, Any]) -> list[Step]:
    name = params.get("name", "")
    positive = list(params.get("positive", []))
    negative = list(params.get("negative", []))
    return [
        Step(Op.TAG, {"key": "input_axis", "value": True}),
        Step(Op.TAG, {"key": "axis_name", "value": name}),
        Step(Op.TAG, {"key": "axis_positive", "value": positive}),
        Step(Op.TAG, {"key": "axis_negative", "value": negative}),
        Step(Op.REGISTER, {"registry": "input"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_input_action(target: Any, params: dict[str, Any]) -> Any:
    name = params.get("name", "")
    bindings = list(params.get("default_bindings", []))
    target._input_action = True
    target._action_name = name
    target._action_bindings = bindings
    return None


def _after_input_axis(target: Any, params: dict[str, Any]) -> Any:
    name = params.get("name", "")
    positive = list(params.get("positive", []))
    negative = list(params.get("negative", []))
    target._input_axis = True
    target._axis_name = name
    target._axis_positive = positive
    target._axis_negative = negative
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


input_action = make_decorator(
    name="input_action",
    steps=_input_action_steps,
    doc="Register an input action with default key bindings.",
    validate=_validate_input_action,
    after_steps=_after_input_action,
)

input_axis = make_decorator(
    name="input_axis",
    steps=_input_axis_steps,
    doc="Register an input axis with positive and negative bindings.",
    validate=_validate_input_axis,
    after_steps=_after_input_axis,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("input_action", input_action, ("function",)),
    ("input_axis", input_axis, ("function",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.INPUT,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.INPUT].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "input_action",
    "input_axis",
]
