"""
Debug/Cheat decorators — built from Ops.

Decorators for debug and cheat systems: debug commands, visual debugging,
and inspector integration.

Decorators:
    @cheat       - Register a debug/cheat command
    @debug_draw  - Visual debugging overlay
    @inspector   - Inspector panel integration
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

T = TypeVar("T")

# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_cheat(
    name: str = "",
    category: str = "general",
    requires_confirmation: bool = False,
    **_: Any,
) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("@cheat: 'name' parameter is required and must be a non-empty string")


def _validate_debug_draw(
    color: Any = None,
    duration: float = 0.0,
    depth_test: bool = True,
    **_: Any,
) -> None:
    if not isinstance(duration, (int, float)) or duration < 0:
        raise ValueError(
            f"@debug_draw: duration must be a non-negative number, got {duration!r}"
        )


def _validate_inspector(
    category: str = "default",
    readonly: bool = False,
    range: Optional[tuple] = None,
    **_: Any,
) -> None:
    if range is not None:
        if (
            not isinstance(range, tuple)
            or len(range) != 2
            or not all(isinstance(v, (int, float)) for v in range)
        ):
            raise ValueError(
                f"@inspector: range must be a tuple of 2 numbers (min, max), got {range!r}"
            )
        if range[0] > range[1]:
            raise ValueError(
                f"@inspector: range min ({range[0]}) must be <= max ({range[1]})"
            )

# =============================================================================
# STEP BUILDERS
# =============================================================================


def _cheat_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "cheat", "value": True}),
        Step(Op.TAG, {"key": "cheat_name", "value": params.get("name", "")}),
        Step(Op.TAG, {"key": "cheat_category", "value": params.get("category", "general")}),
        Step(Op.TAG, {"key": "cheat_requires_confirmation", "value": params.get("requires_confirmation", False)}),
        Step(Op.REGISTER, {"registry": "debug_cheat"}),
    ]


def _debug_draw_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "debug_draw", "value": True}),
        Step(Op.TAG, {"key": "debug_draw_color", "value": params.get("color")}),
        Step(Op.TAG, {"key": "debug_draw_duration", "value": params.get("duration", 0.0)}),
        Step(Op.TAG, {"key": "debug_draw_depth_test", "value": params.get("depth_test", True)}),
        Step(Op.REGISTER, {"registry": "debug_cheat"}),
    ]


def _inspector_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "inspector", "value": True}),
        Step(Op.TAG, {"key": "inspector_category", "value": params.get("category", "default")}),
        Step(Op.TAG, {"key": "inspector_readonly", "value": params.get("readonly", False)}),
        Step(Op.TAG, {"key": "inspector_range", "value": params.get("range")}),
        Step(Op.REGISTER, {"registry": "debug_cheat"}),
    ]

# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_cheat(target: Any, params: dict[str, Any]) -> Any:
    target._cheat = True
    target._cheat_name = params.get("name", "")
    target._cheat_category = params.get("category", "general")
    target._cheat_requires_confirmation = params.get("requires_confirmation", False)
    return None


def _after_debug_draw(target: Any, params: dict[str, Any]) -> Any:
    target._debug_draw = True
    target._debug_draw_color = params.get("color")
    target._debug_draw_duration = params.get("duration", 0.0)
    target._debug_draw_depth_test = params.get("depth_test", True)
    return None


def _after_inspector(target: Any, params: dict[str, Any]) -> Any:
    target._inspector = True
    target._inspector_category = params.get("category", "default")
    target._inspector_readonly = params.get("readonly", False)
    target._inspector_range = params.get("range")
    return None

# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

cheat = make_decorator(
    name="cheat",
    steps=_cheat_steps,
    doc="Register a debug/cheat command.",
    validate=_validate_cheat,
    after_steps=_after_cheat,
)

debug_draw = make_decorator(
    name="debug_draw",
    steps=_debug_draw_steps,
    doc="Enable visual debugging overlay.",
    validate=_validate_debug_draw,
    after_steps=_after_debug_draw,
)

inspector = make_decorator(
    name="inspector",
    steps=_inspector_steps,
    doc="Integrate with inspector panel.",
    validate=_validate_inspector,
    after_steps=_after_inspector,
)

# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("cheat", cheat, ("function",)),
    ("debug_draw", debug_draw, ("class", "function")),
    ("inspector", inspector, ("class", "function")),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.DEBUG_CHEAT,
            func=_func,
            unique=True,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.DEBUG_CHEAT].append(_spec)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "cheat",
    "debug_draw",
    "inspector",
]
