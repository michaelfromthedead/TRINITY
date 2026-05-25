"""
Composition decorators — built from Ops.

These decorators enable decorator composition and aliasing.

Decorators:
    @composite  - Compose multiple decorators into one
    @alias      - Create a named alias for a decorated class/function
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry

T = TypeVar("T")


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_composite(decorators: Any = (), **_: Any) -> None:
    if not decorators:
        raise ValueError(
            "@composite: 'decorators' parameter is required and must be non-empty"
        )
    for i, d in enumerate(decorators):
        if not callable(d):
            raise ValueError(
                f"@composite: decorator at index {i} is not callable: {d!r}"
            )


def _validate_alias(name: str = "", **_: Any) -> None:
    if not name:
        raise ValueError("@alias: 'name' parameter is required and must be non-empty")


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _composite_steps(params: dict[str, Any]) -> list[Step]:
    decorators = params.get("decorators", ())
    return [
        Step(Op.TAG, {"key": "composite", "value": True}),
        Step(Op.TAG, {"key": "composite_decorators", "value": list(decorators)}),
        Step(Op.REGISTER, {"registry": "composition"}),
    ]


def _alias_steps(params: dict[str, Any]) -> list[Step]:
    name = params.get("name", "")
    return [
        Step(Op.TAG, {"key": "alias", "value": True}),
        Step(Op.TAG, {"key": "alias_name", "value": name}),
        Step(Op.REGISTER, {"registry": "composition"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_composite(target: Any, params: dict[str, Any]) -> Any:
    target._composite = True
    target._composite_decorators = list(params.get("decorators", ()))
    return None


def _after_alias(target: Any, params: dict[str, Any]) -> Any:
    target._alias = True
    target._alias_name = params.get("name", "")
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


composite = make_decorator(
    name="composite",
    steps=_composite_steps,
    doc="Compose multiple decorators into one.",
    validate=_validate_composite,
    after_steps=_after_composite,
)

alias = make_decorator(
    name="alias",
    steps=_alias_steps,
    doc="Create a named alias for a decorated class/function.",
    validate=_validate_alias,
    after_steps=_after_alias,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("composite", composite, ("class", "function")),
    ("alias", alias, ("class", "function")),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.COMPOSITION,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.COMPOSITION].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "composite",
    "alias",
]
