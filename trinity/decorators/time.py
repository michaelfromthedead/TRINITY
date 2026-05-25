"""
Time decorators — built from Ops.

Decorators for time scaling, pausing, rewinding, and determinism.

Decorators:
    @time_scale    - Per-system time scaling
    @pausable      - What pauses when game pauses
    @rewindable    - Component state rewind
    @deterministic - Mark system as deterministic
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.base import validate_target_type
from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry

T = TypeVar("T")

# =============================================================================
# VALID VALUES
# =============================================================================

VALID_INTERPOLATIONS = frozenset({"linear", "hermite", "none"})
DEFAULT_PAUSE_LAYERS = frozenset({"gameplay"})

# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_time_scale(
    layer: str = "gameplay",
    min_scale: float = 0.0,
    max_scale: float = 10.0,
    **_: Any,
) -> None:
    if not layer or not isinstance(layer, str):
        raise ValueError("@time_scale: 'layer' must be a non-empty string")
    if min_scale < 0:
        raise ValueError("@time_scale: 'min_scale' must be >= 0")
    if max_scale <= 0:
        raise ValueError("@time_scale: 'max_scale' must be > 0")
    if min_scale > max_scale:
        raise ValueError(
            f"@time_scale: 'min_scale' ({min_scale}) must be <= 'max_scale' ({max_scale})"
        )


def _validate_pausable(
    pause_layers: Any = None,
    **_: Any,
) -> None:
    pass  # No validation needed; None is valid (defaults to {"gameplay"})


def _validate_rewindable(
    history_seconds: float = 5.0,
    interpolation: str = "linear",
    **_: Any,
) -> None:
    if history_seconds <= 0:
        raise ValueError("@rewindable: 'history_seconds' must be > 0")
    if interpolation not in VALID_INTERPOLATIONS:
        raise ValueError(
            f"@rewindable: invalid interpolation '{interpolation}'. "
            f"Valid: {sorted(VALID_INTERPOLATIONS)}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _time_scale_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "time_scale", "value": True}),
        Step(Op.TAG, {"key": "time_layer", "value": params.get("layer", "gameplay")}),
        Step(Op.TAG, {"key": "time_min_scale", "value": params.get("min_scale", 0.0)}),
        Step(Op.TAG, {"key": "time_max_scale", "value": params.get("max_scale", 10.0)}),
        Step(Op.REGISTER, {"registry": "time"}),
    ]


def _pausable_steps(params: dict[str, Any]) -> list[Step]:
    pause_layers = params.get("pause_layers")
    resolved = set(pause_layers or DEFAULT_PAUSE_LAYERS)
    return [
        Step(Op.TAG, {"key": "pausable", "value": True}),
        Step(Op.TAG, {"key": "pause_layers", "value": resolved}),
        Step(Op.REGISTER, {"registry": "time"}),
    ]


def _rewindable_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "rewindable", "value": True}),
        Step(Op.TAG, {"key": "rewind_history", "value": params.get("history_seconds", 5.0)}),
        Step(Op.TAG, {"key": "rewind_interpolation", "value": params.get("interpolation", "linear")}),
        Step(Op.REGISTER, {"registry": "time"}),
    ]


def _deterministic_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "deterministic", "value": True}),
        Step(Op.REGISTER, {"registry": "time"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_time_scale(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "time_scale", ("function",))
    target._time_scale = True
    target._time_layer = params.get("layer", "gameplay")
    target._time_min_scale = params.get("min_scale", 0.0)
    target._time_max_scale = params.get("max_scale", 10.0)
    return None


def _after_pausable(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "pausable", ("function",))
    pause_layers = params.get("pause_layers")
    target._pausable = True
    target._pause_layers = set(pause_layers or DEFAULT_PAUSE_LAYERS)
    return None


def _after_rewindable(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "rewindable", ("class",))
    target._rewindable = True
    target._rewind_history = params.get("history_seconds", 5.0)
    target._rewind_interpolation = params.get("interpolation", "linear")
    return None


def _after_deterministic(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "deterministic", ("function",))
    target._deterministic = True
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

time_scale = make_decorator(
    name="time_scale",
    steps=_time_scale_steps,
    doc="Per-system time scaling with layer support.",
    validate=_validate_time_scale,
    after_steps=_after_time_scale,
)

pausable = make_decorator(
    name="pausable",
    steps=_pausable_steps,
    doc="Mark function as pausable with specified layers.",
    validate=_validate_pausable,
    after_steps=_after_pausable,
)

rewindable = make_decorator(
    name="rewindable",
    steps=_rewindable_steps,
    doc="Enable state rewind with configurable history and interpolation.",
    validate=_validate_rewindable,
    after_steps=_after_rewindable,
)

deterministic = make_decorator(
    name="deterministic",
    steps=_deterministic_steps,
    doc="Mark system as deterministic for replay consistency.",
    after_steps=_after_deterministic,
)

# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("time_scale", time_scale, ("function",)),
    ("pausable", pausable, ("function",)),
    ("rewindable", rewindable, ("class",)),
    ("deterministic", deterministic, ("function",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.TIME,
            func=_func,
            unique=True,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.TIME].append(_spec)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "time_scale",
    "pausable",
    "rewindable",
    "deterministic",
    "VALID_INTERPOLATIONS",
    "DEFAULT_PAUSE_LAYERS",
]
