"""
UI decorators — built from Ops.

Decorators for UI system configuration: widgets and layouts.

Decorators:
    @widget  - Mark class as UI widget
    @layout  - Configure layout direction, gap, and padding
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

VALID_LAYOUT_DIRECTIONS = frozenset({"vertical", "horizontal", "grid"})


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_widget(style: Optional[dict] = None, **_: Any) -> None:
    if style is not None and not isinstance(style, dict):
        raise ValueError("@widget: 'style' must be a dict or None")


def _validate_layout(
    direction: str = "vertical",
    gap: Any = 0,
    padding: Any = 0,
    **_: Any,
) -> None:
    if direction not in VALID_LAYOUT_DIRECTIONS:
        raise ValueError(
            f"@layout: invalid direction '{direction}'. "
            f"Valid directions: {sorted(VALID_LAYOUT_DIRECTIONS)}"
        )
    if not isinstance(gap, (int, float)):
        raise ValueError("@layout: 'gap' must be a number")
    if gap < 0:
        raise ValueError(f"@layout: 'gap' must be >= 0, got {gap}")
    if not isinstance(padding, (int, float)):
        raise ValueError("@layout: 'padding' must be a number")
    if padding < 0:
        raise ValueError(f"@layout: 'padding' must be >= 0, got {padding}")


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _widget_steps(params: dict[str, Any]) -> list[Step]:
    style = dict(params.get("style") or {})
    return [
        Step(Op.TAG, {"key": "widget", "value": True}),
        Step(Op.TAG, {"key": "widget_style", "value": style}),
        Step(Op.REGISTER, {"registry": "ui"}),
    ]


def _layout_steps(params: dict[str, Any]) -> list[Step]:
    direction = params.get("direction", "vertical")
    gap = params.get("gap", 0)
    padding = params.get("padding", 0)
    return [
        Step(Op.TAG, {"key": "layout", "value": True}),
        Step(Op.TAG, {"key": "layout_direction", "value": direction}),
        Step(Op.TAG, {"key": "layout_gap", "value": gap}),
        Step(Op.TAG, {"key": "layout_padding", "value": padding}),
        Step(Op.REGISTER, {"registry": "ui"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_widget(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "widget", ("class",))
    target._widget = True
    target._widget_style = dict(params.get("style") or {})
    return None


def _after_layout(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "layout", ("class",))
    target._layout = True
    target._layout_direction = params.get("direction", "vertical")
    target._layout_gap = params.get("gap", 0)
    target._layout_padding = params.get("padding", 0)
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


widget = make_decorator(
    name="widget",
    steps=_widget_steps,
    doc="Mark class as UI widget with optional style properties.",
    validate=_validate_widget,
    after_steps=_after_widget,
)

layout = make_decorator(
    name="layout",
    steps=_layout_steps,
    doc="Configure layout direction, gap, and padding for UI containers.",
    validate=_validate_layout,
    after_steps=_after_layout,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("widget", widget, ("class",)),
    ("layout", layout, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.UI,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.UI].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "widget",
    "layout",
    "VALID_LAYOUT_DIRECTIONS",
]
