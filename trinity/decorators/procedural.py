"""
Trinity Pattern - Tier 37: PROCEDURAL Decorators

Procedural generation, seeding, and constraint decorators.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_SEED_SOURCES = frozenset({"world", "chunk", "entity", "explicit"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _seeded_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @seeded decorator."""
    seed_source = params.get("seed_source", "world")

    return [
        Step(Op.TAG, {"key": "seeded", "value": True}),
        Step(Op.TAG, {"key": "seed_source", "value": seed_source}),
        Step(Op.REGISTER, {"registry": "procedural"}),
        Step(Op.DESCRIBE, {}),
    ]


def _procedural_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @procedural decorator."""
    cache = params.get("cache", True)
    validate_fn = params.get("validate")

    return [
        Step(Op.TAG, {"key": "procedural", "value": True}),
        Step(Op.TAG, {"key": "procedural_cache", "value": cache}),
        Step(Op.TAG, {"key": "procedural_validate", "value": validate_fn}),
        Step(Op.REGISTER, {"registry": "procedural"}),
        Step(Op.DESCRIBE, {}),
    ]


def _constraint_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @constraint decorator."""
    rules = params.get("rules", [])

    return [
        Step(Op.TAG, {"key": "constraint", "value": True}),
        Step(Op.TAG, {"key": "constraint_rules", "value": list(rules)}),
        Step(Op.REGISTER, {"registry": "procedural"}),
        Step(Op.DESCRIBE, {}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_seeded_params(**kwargs: Any) -> None:
    """Validate @seeded parameters."""
    seed_source = kwargs.get("seed_source", "world")
    if seed_source not in VALID_SEED_SOURCES:
        raise ValueError(
            f"Invalid seed_source '{seed_source}'. "
            f"Must be one of {sorted(VALID_SEED_SOURCES)}"
        )


def _validate_constraint_params(**kwargs: Any) -> None:
    """Validate @constraint parameters."""
    rules = kwargs.get("rules")
    if not rules:
        raise ValueError("rules must be a non-empty list")
    if not isinstance(rules, list):
        raise TypeError("rules must be a list")


# ============================================================================
# After-apply functions
# ============================================================================


def _seeded_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @seeded is applied."""
    seed_source = params.get("seed_source", "world")

    obj._seeded = True
    obj._seed_source = seed_source


def _procedural_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @procedural is applied."""
    cache = params.get("cache", True)
    validate_fn = params.get("validate")

    obj._procedural = True
    obj._procedural_cache = cache
    obj._procedural_validate = validate_fn


def _constraint_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @constraint is applied."""
    rules = params.get("rules", [])

    obj._constraint = True
    obj._constraint_rules = list(rules)


# ============================================================================
# Decorator creation
# ============================================================================

seeded = make_decorator(
    name="seeded",
    steps=_seeded_steps,
    validate=_validate_seeded_params,
    after_steps=_seeded_after_apply,
)

procedural = make_decorator(
    name="procedural",
    steps=_procedural_steps,
    after_steps=_procedural_after_apply,
)

constraint = make_decorator(
    name="constraint",
    steps=_constraint_steps,
    validate=_validate_constraint_params,
    after_steps=_constraint_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("seeded", seeded, ("class",)),
    ("procedural", procedural, ("class",)),
    ("constraint", constraint, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.PROCEDURAL,
            func=_func,
            unique=_name in ("seeded", "procedural"),
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.PROCEDURAL].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "seeded",
    "procedural",
    "constraint",
    "VALID_SEED_SOURCES",
]
