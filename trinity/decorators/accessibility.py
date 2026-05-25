"""
Accessibility decorators — built from Ops.

These decorators mark code for accessibility support: screen reader
annotations and ARIA-like role assignments for game UI elements.

Decorators:
    @accessible  - Screen reader and role support
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.base import validate_target_type
from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# VALID VALUES
# =============================================================================

VALID_ROLES = frozenset({"button", "slider", "text", "image", "list", "listitem"})


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_accessible(
    screen_reader: Optional[str] = None,
    role: Optional[str] = None,
    **_: Any,
) -> None:
    if role is not None and role not in VALID_ROLES:
        raise ValueError(
            f"@accessible: invalid role '{role}'. "
            f"Valid roles: {sorted(VALID_ROLES)}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _accessible_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "accessible", "value": True}),
        Step(Op.TAG, {"key": "accessible_screen_reader", "value": params.get("screen_reader")}),
        Step(Op.TAG, {"key": "accessible_role", "value": params.get("role")}),
        Step(Op.REGISTER, {"registry": "accessibility"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_accessible(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "accessible", ("class",))
    target._accessible = True
    target._accessible_screen_reader = params.get("screen_reader")
    target._accessible_role = params.get("role")
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


accessible = make_decorator(
    name="accessible",
    steps=_accessible_steps,
    doc="Mark class with screen reader support and accessibility role.",
    validate=_validate_accessible,
    after_steps=_after_accessible,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("accessible", accessible, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.ACCESSIBILITY,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.ACCESSIBILITY].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "accessible",
    "VALID_ROLES",
]
