"""
Animation decorators — built from Ops.

Decorators for tween animation and blend tree configuration.

Decorators:
    @tween      - Configure tween animation
    @blend_tree - Configure animation blend tree
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# VALID VALUES
# =============================================================================

VALID_EASING_FUNCTIONS = frozenset(
    {"linear", "ease_in", "ease_out", "ease_in_out", "bounce"}
)


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_tween(
    property: str = "", duration: float = 0.0, easing: str = "linear", **_: Any
) -> None:
    if not property:
        raise ValueError("@tween: 'property' parameter is required")
    if duration <= 0:
        raise ValueError(
            f"@tween: duration must be > 0, got {duration}"
        )
    if easing not in VALID_EASING_FUNCTIONS:
        raise ValueError(
            f"@tween: invalid easing '{easing}'. "
            f"Valid easings: {sorted(VALID_EASING_FUNCTIONS)}"
        )


def _validate_blend_tree(
    parameter: str = "", clips: Any = None, **_: Any
) -> None:
    if not parameter:
        raise ValueError("@blend_tree: 'parameter' parameter is required")
    if not clips:
        raise ValueError("@blend_tree: 'clips' parameter is required and must be non-empty")


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _tween_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "tween", "value": True}),
        Step(Op.TAG, {"key": "tween_property", "value": params.get("property", "")}),
        Step(Op.TAG, {"key": "tween_duration", "value": params.get("duration", 0.0)}),
        Step(Op.TAG, {"key": "tween_easing", "value": params.get("easing", "linear")}),
        Step(Op.REGISTER, {"registry": "animation"}),
    ]


def _blend_tree_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "blend_tree", "value": True}),
        Step(Op.TAG, {"key": "blend_parameter", "value": params.get("parameter", "")}),
        Step(Op.TAG, {"key": "blend_clips", "value": list(params.get("clips", []))}),
        Step(Op.REGISTER, {"registry": "animation"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_tween(target: Any, params: dict[str, Any]) -> Any:
    target._tween = True
    target._tween_property = params.get("property", "")
    target._tween_duration = params.get("duration", 0.0)
    target._tween_easing = params.get("easing", "linear")
    return None


def _after_blend_tree(target: Any, params: dict[str, Any]) -> Any:
    target._blend_tree = True
    target._blend_parameter = params.get("parameter", "")
    target._blend_clips = list(params.get("clips", []))
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

tween = make_decorator(
    name="tween",
    steps=_tween_steps,
    doc="Configure tween animation for a property.",
    validate=_validate_tween,
    after_steps=_after_tween,
)

blend_tree = make_decorator(
    name="blend_tree",
    steps=_blend_tree_steps,
    doc="Configure animation blend tree with parameter and clips.",
    validate=_validate_blend_tree,
    after_steps=_after_blend_tree,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("tween", tween, ("class",)),
    ("blend_tree", blend_tree, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.ANIMATION,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.ANIMATION].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "tween",
    "blend_tree",
    "VALID_EASING_FUNCTIONS",
]
