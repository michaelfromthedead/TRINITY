"""
Achievement decorators — built from Ops.

Decorators for cross-platform achievements, progress tracking,
and player statistics.

Decorators:
    @achievement - Cross-platform achievement definition
    @progress    - Progress tracking toward a goal
    @stat        - Player statistics tracking
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from trinity.decorators.base import validate_target_type
from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# =============================================================================
# VALID VALUES
# =============================================================================

VALID_AGGREGATIONS = frozenset({"sum", "max", "min", "latest", "average"})

# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_achievement(
    id: str = "", platform_ids: Any = None, secret: bool = False, **_: Any
) -> None:
    if not id:
        raise ValueError("@achievement: 'id' parameter is required and must be non-empty")


def _validate_progress(
    id: str = "", target: Any = 0, persistent: bool = True, **_: Any
) -> None:
    if not id:
        raise ValueError("@progress: 'id' parameter is required and must be non-empty")
    if not isinstance(target, (int, float)) or target <= 0:
        raise ValueError(
            f"@progress: 'target' must be a positive number, got {target!r}"
        )


def _validate_stat(
    id: str = "", aggregation: str = "sum", **_: Any
) -> None:
    if not id:
        raise ValueError("@stat: 'id' parameter is required and must be non-empty")
    if aggregation not in VALID_AGGREGATIONS:
        raise ValueError(
            f"@stat: invalid aggregation '{aggregation}'. "
            f"Valid aggregations: {sorted(VALID_AGGREGATIONS)}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _achievement_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "achievement", "value": True}),
        Step(Op.TAG, {"key": "achievement_id", "value": params.get("id", "")}),
        Step(
            Op.TAG,
            {
                "key": "achievement_platform_ids",
                "value": dict(params.get("platform_ids") or {}),
            },
        ),
        Step(Op.TAG, {"key": "achievement_secret", "value": params.get("secret", False)}),
        Step(Op.REGISTER, {"registry": "achievements"}),
    ]


def _progress_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "progress", "value": True}),
        Step(Op.TAG, {"key": "progress_id", "value": params.get("id", "")}),
        Step(Op.TAG, {"key": "progress_target", "value": params.get("target", 0)}),
        Step(Op.TAG, {"key": "progress_persistent", "value": params.get("persistent", True)}),
        Step(Op.REGISTER, {"registry": "achievements"}),
    ]


def _stat_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "stat", "value": True}),
        Step(Op.TAG, {"key": "stat_id", "value": params.get("id", "")}),
        Step(Op.TAG, {"key": "stat_aggregation", "value": params.get("aggregation", "sum")}),
        Step(Op.REGISTER, {"registry": "achievements"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_achievement(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "achievement", ("function", "class"))
    target._achievement = True
    target._achievement_id = params.get("id", "")
    target._achievement_platform_ids = dict(params.get("platform_ids") or {})
    target._achievement_secret = params.get("secret", False)
    return None


def _after_progress(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "progress", ("class",))
    target._progress = True
    target._progress_id = params.get("id", "")
    target._progress_target = params.get("target", 0)
    target._progress_persistent = params.get("persistent", True)
    return None


def _after_stat(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "stat", ("class", "function"))
    target._stat = True
    target._stat_id = params.get("id", "")
    target._stat_aggregation = params.get("aggregation", "sum")
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

achievement = make_decorator(
    name="achievement",
    steps=_achievement_steps,
    doc="Cross-platform achievement definition.",
    validate=_validate_achievement,
    after_steps=_after_achievement,
)

progress = make_decorator(
    name="progress",
    steps=_progress_steps,
    doc="Progress tracking toward a goal.",
    validate=_validate_progress,
    after_steps=_after_progress,
)

stat = make_decorator(
    name="stat",
    steps=_stat_steps,
    doc="Player statistics tracking.",
    validate=_validate_stat,
    after_steps=_after_stat,
)

# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("achievement", achievement, ("function", "class")),
    ("progress", progress, ("class",)),
    ("stat", stat, ("class", "function")),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.ACHIEVEMENTS,
            func=_func,
            unique=True,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.ACHIEVEMENTS].append(_spec)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "achievement",
    "progress",
    "stat",
    "VALID_AGGREGATIONS",
]
