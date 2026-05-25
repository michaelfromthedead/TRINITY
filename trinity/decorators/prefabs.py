"""
Prefab decorators — built from Ops.

These decorators mark classes as prefab templates and inheritance chains.

Decorators:
    @prefab   - Mark class as a prefab template
    @extends  - Mark class as extending a parent prefab
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry

T = TypeVar("T")


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_prefab(name: str = "", **_: Any) -> None:
    if not name:
        raise ValueError("@prefab: 'name' parameter is required and must be non-empty")


def _validate_extends(parent: str = "", **_: Any) -> None:
    if not parent:
        raise ValueError(
            "@extends: 'parent' parameter is required and must be non-empty"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _prefab_steps(params: dict[str, Any]) -> list[Step]:
    name = params.get("name", "")
    return [
        Step(Op.TAG, {"key": "prefab", "value": True}),
        Step(Op.TAG, {"key": "prefab_name", "value": name}),
        Step(Op.REGISTER, {"registry": "prefabs"}),
    ]


def _extends_steps(params: dict[str, Any]) -> list[Step]:
    parent = params.get("parent", "")
    return [
        Step(Op.TAG, {"key": "extends", "value": True}),
        Step(Op.TAG, {"key": "extends_parent", "value": parent}),
        Step(Op.REGISTER, {"registry": "prefabs"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_prefab(target: Any, params: dict[str, Any]) -> Any:
    target._prefab = True
    target._prefab_name = params.get("name", "")
    return None


def _after_extends(target: Any, params: dict[str, Any]) -> Any:
    target._extends = True
    target._extends_parent = params.get("parent", "")
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


prefab = make_decorator(
    name="prefab",
    steps=_prefab_steps,
    doc="Mark class as a prefab template.",
    validate=_validate_prefab,
    after_steps=_after_prefab,
)

extends = make_decorator(
    name="extends",
    steps=_extends_steps,
    doc="Mark class as extending a parent prefab.",
    validate=_validate_extends,
    after_steps=_after_extends,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("prefab", prefab, ("class",)),
    ("extends", extends, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.PREFABS,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.PREFABS].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "prefab",
    "extends",
]
