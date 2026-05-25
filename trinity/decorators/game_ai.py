"""
Trinity Pattern - Tier 36: GAME_AI Decorators

Game AI behavior, decision-making, and perception decorators.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_SENSES = frozenset({"sight", "hearing", "damage", "squad"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _behavior_tree_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @behavior_tree decorator."""
    id_ = params.get("id", "")
    debug_name = params.get("debug_name")

    return [
        Step(Op.TAG, {"key": "behavior_tree", "value": True}),
        Step(Op.TAG, {"key": "bt_id", "value": id_}),
        Step(Op.TAG, {"key": "bt_debug_name", "value": debug_name}),
        Step(Op.REGISTER, {"registry": "game_ai"}),
        Step(Op.DESCRIBE, {}),
    ]


def _utility_ai_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @utility_ai decorator."""
    id_ = params.get("id", "")
    update_rate = params.get("update_rate", 0.5)

    return [
        Step(Op.TAG, {"key": "utility_ai", "value": True}),
        Step(Op.TAG, {"key": "utility_id", "value": id_}),
        Step(Op.TAG, {"key": "utility_update_rate", "value": update_rate}),
        Step(Op.REGISTER, {"registry": "game_ai"}),
        Step(Op.DESCRIBE, {}),
    ]


def _blackboard_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @blackboard decorator (marker, no params)."""
    return [
        Step(Op.TAG, {"key": "blackboard", "value": True}),
        Step(Op.REGISTER, {"registry": "game_ai"}),
        Step(Op.DESCRIBE, {}),
    ]


def _ai_debug_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @ai_debug decorator (marker, no params)."""
    return [
        Step(Op.TAG, {"key": "ai_debug", "value": True}),
        Step(Op.REGISTER, {"registry": "game_ai"}),
    ]


def _perception_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @perception decorator."""
    sense = params.get("sense", "sight")
    range_val = params.get("range", 10.0)
    fov = params.get("fov")

    return [
        Step(Op.TAG, {"key": "perception", "value": True}),
        Step(Op.TAG, {"key": "perception_sense", "value": sense}),
        Step(Op.TAG, {"key": "perception_range", "value": range_val}),
        Step(Op.TAG, {"key": "perception_fov", "value": fov}),
        Step(Op.REGISTER, {"registry": "game_ai"}),
        Step(Op.DESCRIBE, {}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_behavior_tree_params(**kwargs: Any) -> None:
    """Validate @behavior_tree parameters."""
    id_ = kwargs.get("id", "")
    if not id_:
        raise ValueError("id must be non-empty")


def _validate_utility_ai_params(**kwargs: Any) -> None:
    """Validate @utility_ai parameters."""
    id_ = kwargs.get("id", "")
    if not id_:
        raise ValueError("id must be non-empty")

    update_rate = kwargs.get("update_rate", 0.5)
    if update_rate <= 0:
        raise ValueError(f"update_rate must be > 0, got {update_rate}")


def _validate_perception_params(**kwargs: Any) -> None:
    """Validate @perception parameters."""
    sense = kwargs.get("sense", "sight")
    if sense not in VALID_SENSES:
        raise ValueError(
            f"Invalid sense '{sense}'. Must be one of {sorted(VALID_SENSES)}"
        )

    range_val = kwargs.get("range", 10.0)
    if range_val <= 0:
        raise ValueError(f"range must be > 0, got {range_val}")


# ============================================================================
# After-apply functions
# ============================================================================


def _behavior_tree_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @behavior_tree is applied."""
    id_ = params.get("id", "")
    debug_name = params.get("debug_name")

    obj._behavior_tree = True
    obj._bt_id = id_
    obj._bt_debug_name = debug_name


def _utility_ai_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @utility_ai is applied."""
    id_ = params.get("id", "")
    update_rate = params.get("update_rate", 0.5)

    obj._utility_ai = True
    obj._utility_id = id_
    obj._utility_update_rate = update_rate


def _blackboard_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @blackboard is applied."""
    obj._blackboard = True


def _ai_debug_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @ai_debug is applied."""
    obj._ai_debug = True


def _perception_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @perception is applied."""
    sense = params.get("sense", "sight")
    range_val = params.get("range", 10.0)
    fov = params.get("fov")

    obj._perception = True
    obj._perception_sense = sense
    obj._perception_range = range_val
    obj._perception_fov = fov


# ============================================================================
# Decorator creation
# ============================================================================

behavior_tree = make_decorator(
    name="behavior_tree",
    steps=_behavior_tree_steps,
    validate=_validate_behavior_tree_params,
    after_steps=_behavior_tree_after_apply,
)

utility_ai = make_decorator(
    name="utility_ai",
    steps=_utility_ai_steps,
    validate=_validate_utility_ai_params,
    after_steps=_utility_ai_after_apply,
)

blackboard = make_decorator(
    name="blackboard",
    steps=_blackboard_steps,
    after_steps=_blackboard_after_apply,
)

ai_debug = make_decorator(
    name="ai_debug",
    steps=_ai_debug_steps,
    after_steps=_ai_debug_after_apply,
)

perception = make_decorator(
    name="perception",
    steps=_perception_steps,
    validate=_validate_perception_params,
    after_steps=_perception_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("behavior_tree", behavior_tree, ("class",)),
    ("utility_ai", utility_ai, ("class",)),
    ("blackboard", blackboard, ("class",)),
    ("ai_debug", ai_debug, ("class",)),
    ("perception", perception, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.GAME_AI,
            func=_func,
            unique=_name in ("behavior_tree", "utility_ai", "blackboard", "ai_debug"),
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.GAME_AI].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "behavior_tree",
    "utility_ai",
    "blackboard",
    "ai_debug",
    "perception",
    "VALID_SENSES",
]
