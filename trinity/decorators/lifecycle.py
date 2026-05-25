"""Tier 7: Lifecycle decorators for entity and component events.

This module provides decorators for hooking into entity and component lifecycle events:
- @on_add: Called when a component is added to an entity
- @on_remove: Called when a component is removed from an entity
- @on_change: Called when a component's state changes
- @on_spawn: Called when an entity is spawned
- @on_despawn: Called when an entity is despawned
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

__all__ = ["on_add", "on_remove", "on_change", "on_spawn", "on_despawn"]

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_component_required(component: Optional[Any] = None, **_: Any) -> None:
    """Validate that component parameter is provided."""
    if component is None:
        raise ValueError("component parameter is required")


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _on_add_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @on_add decorator."""
    component = params.get("component")
    return [
        Step(Op.TAG, {"key": "on_add", "value": True}),
        Step(Op.TAG, {"key": "on_add_component", "value": component}),
        Step(Op.HOOK, {"event": "on_add"}),
        Step(Op.REGISTER, {"registry": "lifecycle"}),
    ]


def _on_add_after(func: F, params: dict[str, Any]) -> F:
    """Post-processing for @on_add decorator."""
    component = params.get("component")
    func._on_add_component = component  # type: ignore
    func._lifecycle_hook = "add"  # type: ignore
    return func


def _on_remove_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @on_remove decorator."""
    component = params.get("component")
    return [
        Step(Op.TAG, {"key": "on_remove", "value": True}),
        Step(Op.TAG, {"key": "on_remove_component", "value": component}),
        Step(Op.HOOK, {"event": "on_remove"}),
        Step(Op.REGISTER, {"registry": "lifecycle"}),
    ]


def _on_remove_after(func: F, params: dict[str, Any]) -> F:
    """Post-processing for @on_remove decorator."""
    component = params.get("component")
    func._on_remove_component = component  # type: ignore
    func._lifecycle_hook = "remove"  # type: ignore
    return func


def _on_change_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @on_change decorator."""
    component = params.get("component")
    return [
        Step(Op.TAG, {"key": "on_change", "value": True}),
        Step(Op.TAG, {"key": "on_change_component", "value": component}),
        Step(Op.HOOK, {"event": "on_change"}),
        Step(Op.REGISTER, {"registry": "lifecycle"}),
    ]


def _on_change_after(func: F, params: dict[str, Any]) -> F:
    """Post-processing for @on_change decorator."""
    component = params.get("component")
    func._on_change_component = component  # type: ignore
    func._lifecycle_hook = "change"  # type: ignore
    return func


def _on_spawn_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @on_spawn decorator."""
    return [
        Step(Op.TAG, {"key": "on_spawn", "value": True}),
        Step(Op.HOOK, {"event": "on_spawn"}),
        Step(Op.REGISTER, {"registry": "lifecycle"}),
    ]


def _on_spawn_after(func: F, params: dict[str, Any]) -> F:
    """Post-processing for @on_spawn decorator."""
    func._on_spawn = True  # type: ignore
    func._lifecycle_hook = "spawn"  # type: ignore
    return func


def _on_despawn_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @on_despawn decorator."""
    return [
        Step(Op.TAG, {"key": "on_despawn", "value": True}),
        Step(Op.HOOK, {"event": "on_despawn"}),
        Step(Op.REGISTER, {"registry": "lifecycle"}),
    ]


def _on_despawn_after(func: F, params: dict[str, Any]) -> F:
    """Post-processing for @on_despawn decorator."""
    func._on_despawn = True  # type: ignore
    func._lifecycle_hook = "despawn"  # type: ignore
    return func


# Create decorators
on_add = make_decorator(
    name="on_add",
    steps=_on_add_steps,
    doc="Hook called when a component is added to an entity",
    validate=_validate_component_required,
    after_steps=_on_add_after,
)

on_remove = make_decorator(
    name="on_remove",
    steps=_on_remove_steps,
    doc="Hook called when a component is removed from an entity",
    validate=_validate_component_required,
    after_steps=_on_remove_after,
)

on_change = make_decorator(
    name="on_change",
    steps=_on_change_steps,
    doc="Hook called when a component's state changes",
    validate=_validate_component_required,
    after_steps=_on_change_after,
)

on_spawn = make_decorator(
    name="on_spawn",
    steps=_on_spawn_steps,
    after_steps=_on_spawn_after,
    doc="Hook called when an entity is spawned",
)

on_despawn = make_decorator(
    name="on_despawn",
    steps=_on_despawn_steps,
    after_steps=_on_despawn_after,
    doc="Hook called when an entity is despawned",
)


# Register all lifecycle decorators
_REGISTRY_ENTRIES = [
    ("on_add", on_add, ("function",)),
    ("on_remove", on_remove, ("function",)),
    ("on_change", on_change, ("function",)),
    ("on_spawn", on_spawn, ("function",)),
    ("on_despawn", on_despawn, ("function",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.LIFECYCLE,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.LIFECYCLE].append(_spec)
