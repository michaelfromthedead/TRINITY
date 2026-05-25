"""
Transaction decorators — built from Ops.

Decorators for transactional operations and undo support.

Decorators:
    @transactional  - Mark function as transactional
    @undoable       - Mark operation as undoable
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from trinity.decorators.base import validate_target_type
from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# =============================================================================
# VALID VALUES
# =============================================================================

VALID_ISOLATION_LEVELS = frozenset(
    {"read_uncommitted", "read_committed", "repeatable_read", "serializable"}
)

# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_transactional(isolation: str = "serializable", **_: Any) -> None:
    if isolation not in VALID_ISOLATION_LEVELS:
        raise ValueError(
            f"@transactional: invalid isolation level '{isolation}'. "
            f"Valid levels: {sorted(VALID_ISOLATION_LEVELS)}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _transactional_steps(params: dict[str, Any]) -> list[Step]:
    isolation = params.get("isolation", "serializable")
    return [
        Step(Op.TAG, {"key": "transactional", "value": True}),
        Step(Op.TAG, {"key": "tx_isolation", "value": isolation}),
        Step(Op.REGISTER, {"registry": "transactions"}),
    ]


def _undoable_steps(params: dict[str, Any]) -> list[Step]:
    group = params.get("group", None)
    return [
        Step(Op.TAG, {"key": "undoable", "value": True}),
        Step(Op.TAG, {"key": "undo_group", "value": group}),
        Step(Op.REGISTER, {"registry": "transactions"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_transactional(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "transactional", ("function",))
    target._transactional = True
    target._tx_isolation = params.get("isolation", "serializable")
    return None


def _after_undoable(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "undoable", ("function",))
    target._undoable = True
    target._undo_group = params.get("group", None)
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

transactional = make_decorator(
    name="transactional",
    steps=_transactional_steps,
    doc="Mark function as transactional with configurable isolation level.",
    validate=_validate_transactional,
    after_steps=_after_transactional,
)

undoable = make_decorator(
    name="undoable",
    steps=_undoable_steps,
    doc="Mark operation as undoable with optional undo group.",
    after_steps=_after_undoable,
)

# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("transactional", transactional, ("function",)),
    ("undoable", undoable, ("function",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.TRANSACTIONS,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.TRANSACTIONS].append(_spec)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "transactional",
    "undoable",
    "VALID_ISOLATION_LEVELS",
]
