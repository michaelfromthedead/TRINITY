"""
Trinity Pattern - Tier 32: PLATFORM Decorators

Platform-specific optimization and configuration decorators.
All decorators use the ops-based system via make_decorator().

Note: @platform decorator exists in tier 0 (compilation.py).
This tier contains only @battery_aware for mobile power management.
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_BATTERY_MODES = frozenset({"performance", "balanced", "battery_saver"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _battery_aware_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @battery_aware decorator."""
    mode = params.get("mode", "balanced")

    return [
        Step(Op.TAG, {"key": "battery_aware", "value": True}),
        Step(Op.TAG, {"key": "battery_mode", "value": mode}),
        Step(Op.REGISTER, {"registry": "platform"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_battery_aware_params(**kwargs: Any) -> None:
    """Validate @battery_aware parameters."""
    mode = kwargs.get("mode", "balanced")
    if mode not in VALID_BATTERY_MODES:
        raise ValueError(
            f"Invalid battery mode '{mode}'. Must be one of {VALID_BATTERY_MODES}"
        )


# ============================================================================
# After-apply functions
# ============================================================================


def _battery_aware_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @battery_aware is applied."""
    mode = params.get("mode", "balanced")

    obj._battery_aware = True
    obj._battery_mode = mode


# ============================================================================
# Decorator creation
# ============================================================================

battery_aware = make_decorator(
    name="battery_aware",
    steps=_battery_aware_steps,
    doc="Mobile power management with performance/battery tradeoff modes.",
    validate=_validate_battery_aware_params,
    after_steps=_battery_aware_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("battery_aware", battery_aware, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.PLATFORM,
            func=_func,
            unique=True,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.PLATFORM].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "battery_aware",
    "VALID_BATTERY_MODES",
]
