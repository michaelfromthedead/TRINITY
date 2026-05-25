"""
Trinity Pattern - Tier 39: SOCIAL Decorators

Social system decorators for platforms, leaderboards, and sharing.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_LEADERBOARD_SORT = frozenset({"ascending", "descending"})
VALID_LEADERBOARD_UPDATE = frozenset({"immediate", "daily", "weekly"})
VALID_PRESENCE_DETAIL = frozenset({"minimal", "detailed", "rich"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _social_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @social decorator."""
    platform = params.get("platform", "")

    return [
        Step(Op.TAG, {"key": "social", "value": True}),
        Step(Op.TAG, {"key": "social_platform", "value": platform}),
        Step(Op.REGISTER, {"registry": "social"}),
    ]


def _leaderboard_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @leaderboard decorator."""
    id_value = params.get("id", "")
    sort = params.get("sort", "descending")
    update_frequency = params.get("update_frequency", "immediate")

    return [
        Step(Op.TAG, {"key": "leaderboard", "value": True}),
        Step(Op.TAG, {"key": "leaderboard_id", "value": id_value}),
        Step(Op.TAG, {"key": "leaderboard_sort", "value": sort}),
        Step(Op.TAG, {"key": "leaderboard_update_frequency", "value": update_frequency}),
        Step(Op.REGISTER, {"registry": "social"}),
    ]


def _shareable_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @shareable decorator."""
    platforms = params.get("platforms", {"twitter", "facebook", "clipboard"})

    return [
        Step(Op.TAG, {"key": "shareable", "value": True}),
        Step(Op.TAG, {"key": "shareable_platforms", "value": frozenset(platforms)}),
        Step(Op.REGISTER, {"registry": "social"}),
    ]


def _presence_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @presence decorator."""
    detail_level = params.get("detail_level", "detailed")

    return [
        Step(Op.TAG, {"key": "presence", "value": True}),
        Step(Op.TAG, {"key": "presence_detail_level", "value": detail_level}),
        Step(Op.REGISTER, {"registry": "social"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_social(**kwargs: Any) -> None:
    """Validate @social parameters."""
    platform = kwargs.get("platform", "")
    if not platform:
        raise ValueError("@social: 'platform' parameter is required and must be non-empty")


def _validate_leaderboard(**kwargs: Any) -> None:
    """Validate @leaderboard parameters."""
    id_value = kwargs.get("id", "")
    if not id_value:
        raise ValueError("@leaderboard: 'id' parameter is required and must be non-empty")

    sort = kwargs.get("sort", "descending")
    if sort not in VALID_LEADERBOARD_SORT:
        raise ValueError(
            f"@leaderboard: invalid sort '{sort}'. Must be one of {VALID_LEADERBOARD_SORT}"
        )

    update_frequency = kwargs.get("update_frequency", "immediate")
    if update_frequency not in VALID_LEADERBOARD_UPDATE:
        raise ValueError(
            f"@leaderboard: invalid update_frequency '{update_frequency}'. "
            f"Must be one of {VALID_LEADERBOARD_UPDATE}"
        )


def _validate_shareable(**kwargs: Any) -> None:
    """Validate @shareable parameters."""
    platforms = kwargs.get("platforms", {"twitter", "facebook", "clipboard"})
    if not platforms:
        raise ValueError("@shareable: 'platforms' must be a non-empty set")


def _validate_presence(**kwargs: Any) -> None:
    """Validate @presence parameters."""
    detail_level = kwargs.get("detail_level", "detailed")
    if detail_level not in VALID_PRESENCE_DETAIL:
        raise ValueError(
            f"@presence: invalid detail_level '{detail_level}'. "
            f"Must be one of {VALID_PRESENCE_DETAIL}"
        )


# ============================================================================
# After-apply functions
# ============================================================================


def _social_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @social is applied."""
    obj._social = True
    obj._social_platform = params.get("platform", "")


def _leaderboard_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @leaderboard is applied."""
    obj._leaderboard = True
    obj._leaderboard_id = params.get("id", "")
    obj._leaderboard_sort = params.get("sort", "descending")
    obj._leaderboard_update_frequency = params.get("update_frequency", "immediate")


def _shareable_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @shareable is applied."""
    obj._shareable = True
    platforms = params.get("platforms", {"twitter", "facebook", "clipboard"})
    obj._shareable_platforms = frozenset(platforms)


def _presence_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @presence is applied."""
    obj._presence = True
    obj._presence_detail_level = params.get("detail_level", "detailed")


# ============================================================================
# Decorator creation
# ============================================================================

social = make_decorator(
    name="social",
    steps=_social_steps,
    doc="Social platform integration.",
    validate=_validate_social,
    after_steps=_social_after_apply,
)

leaderboard = make_decorator(
    name="leaderboard",
    steps=_leaderboard_steps,
    doc="Leaderboard definition with sort and update frequency.",
    validate=_validate_leaderboard,
    after_steps=_leaderboard_after_apply,
)

shareable = make_decorator(
    name="shareable",
    steps=_shareable_steps,
    doc="Content sharing across social platforms.",
    validate=_validate_shareable,
    after_steps=_shareable_after_apply,
)

presence = make_decorator(
    name="presence",
    steps=_presence_steps,
    doc="Online presence indicator with detail level.",
    validate=_validate_presence,
    after_steps=_presence_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("social", social, ("class",)),
    ("leaderboard", leaderboard, ("class",)),
    ("shareable", shareable, ("class", "function")),
    ("presence", presence, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.SOCIAL,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.SOCIAL].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "social",
    "leaderboard",
    "shareable",
    "presence",
    "VALID_LEADERBOARD_SORT",
    "VALID_LEADERBOARD_UPDATE",
    "VALID_PRESENCE_DETAIL",
]
