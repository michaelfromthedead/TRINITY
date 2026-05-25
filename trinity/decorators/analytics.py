"""
Analytics decorators — built from Ops.

Decorators for telemetry events, conversion funnel tracking,
and spatial heatmap analytics.

Decorators:
    @telemetry - Analytics event tracking
    @funnel    - Conversion funnel step
    @heatmap   - Spatial analytics collection
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

VALID_CONSENT_LEVELS = frozenset({"none", "analytics", "full"})

# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_telemetry(
    event_name: str = "",
    pii: bool = False,
    required_consent: str = "analytics",
    **_: Any,
) -> None:
    if not event_name:
        raise ValueError(
            "@telemetry: 'event_name' parameter is required and must be non-empty"
        )
    if required_consent not in VALID_CONSENT_LEVELS:
        raise ValueError(
            f"@telemetry: invalid consent level '{required_consent}'. "
            f"Valid levels: {sorted(VALID_CONSENT_LEVELS)}"
        )


def _validate_funnel(
    name: str = "", step: Any = 0, **_: Any
) -> None:
    if not name:
        raise ValueError("@funnel: 'name' parameter is required and must be non-empty")
    if not isinstance(step, int) or step <= 0:
        raise ValueError(
            f"@funnel: 'step' must be a positive integer, got {step!r}"
        )


def _validate_heatmap(resolution: Any = 1.0, **_: Any) -> None:
    if not isinstance(resolution, (int, float)) or resolution <= 0:
        raise ValueError(
            f"@heatmap: 'resolution' must be a positive number, got {resolution!r}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _telemetry_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "telemetry", "value": True}),
        Step(Op.TAG, {"key": "telemetry_event", "value": params.get("event_name", "")}),
        Step(Op.TAG, {"key": "telemetry_pii", "value": params.get("pii", False)}),
        Step(
            Op.TAG,
            {"key": "telemetry_consent", "value": params.get("required_consent", "analytics")},
        ),
        Step(Op.REGISTER, {"registry": "analytics"}),
    ]


def _funnel_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "funnel", "value": True}),
        Step(Op.TAG, {"key": "funnel_name", "value": params.get("name", "")}),
        Step(Op.TAG, {"key": "funnel_step", "value": params.get("step", 0)}),
        Step(Op.REGISTER, {"registry": "analytics"}),
    ]


def _heatmap_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "heatmap", "value": True}),
        Step(Op.TAG, {"key": "heatmap_resolution", "value": params.get("resolution", 1.0)}),
        Step(Op.REGISTER, {"registry": "analytics"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_telemetry(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "telemetry", ("function",))
    target._telemetry = True
    target._telemetry_event = params.get("event_name", "")
    target._telemetry_pii = params.get("pii", False)
    target._telemetry_consent = params.get("required_consent", "analytics")
    return None


def _after_funnel(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "funnel", ("function",))
    target._funnel = True
    target._funnel_name = params.get("name", "")
    target._funnel_step = params.get("step", 0)
    return None


def _after_heatmap(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "heatmap", ("function",))
    target._heatmap = True
    target._heatmap_resolution = params.get("resolution", 1.0)
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

telemetry = make_decorator(
    name="telemetry",
    steps=_telemetry_steps,
    doc="Analytics event tracking with consent management.",
    validate=_validate_telemetry,
    after_steps=_after_telemetry,
)

funnel = make_decorator(
    name="funnel",
    steps=_funnel_steps,
    doc="Conversion funnel step tracking.",
    validate=_validate_funnel,
    after_steps=_after_funnel,
)

heatmap = make_decorator(
    name="heatmap",
    steps=_heatmap_steps,
    doc="Spatial analytics collection.",
    validate=_validate_heatmap,
    after_steps=_after_heatmap,
)

# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("telemetry", telemetry, ("function",)),
    ("funnel", funnel, ("function",)),
    ("heatmap", heatmap, ("function",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.ANALYTICS,
            func=_func,
            unique=True,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.ANALYTICS].append(_spec)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "telemetry",
    "funnel",
    "heatmap",
    "VALID_CONSENT_LEVELS",
]
