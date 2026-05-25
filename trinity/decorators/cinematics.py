"""
Trinity Pattern - Tier 35: CINEMATICS Decorators

Cutscene and camera control decorators.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Validators
# ============================================================================


def _validate_cutscene_params(**kwargs: Any) -> None:
    """Validate @cutscene parameters."""
    id_param = kwargs.get("id", "")
    if not id_param or not isinstance(id_param, str):
        raise ValueError("id must be a non-empty string")


def _validate_camera_track_params(**kwargs: Any) -> None:
    """Validate @camera_track parameters."""
    blend_in = kwargs.get("blend_in", 0.5)
    if not isinstance(blend_in, (int, float)) or blend_in < 0:
        raise ValueError("blend_in must be >= 0")

    blend_out = kwargs.get("blend_out", 0.5)
    if not isinstance(blend_out, (int, float)) or blend_out < 0:
        raise ValueError("blend_out must be >= 0")


# ============================================================================
# Step builders
# ============================================================================


def _cutscene_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @cutscene decorator."""
    id_param = params.get("id", "")
    skippable = params.get("skippable", True)
    pause_gameplay = params.get("pause_gameplay", True)

    return [
        Step(Op.TAG, {"key": "cutscene", "value": True}),
        Step(Op.TAG, {"key": "cutscene_id", "value": id_param}),
        Step(Op.TAG, {"key": "cutscene_skippable", "value": skippable}),
        Step(Op.TAG, {"key": "cutscene_pause_gameplay", "value": pause_gameplay}),
        Step(Op.REGISTER, {"registry": "cinematics"}),
        Step(Op.DESCRIBE, {}),
    ]


def _camera_track_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @camera_track decorator."""
    blend_in = params.get("blend_in", 0.5)
    blend_out = params.get("blend_out", 0.5)

    return [
        Step(Op.TAG, {"key": "camera_track", "value": True}),
        Step(Op.TAG, {"key": "camera_track_blend_in", "value": blend_in}),
        Step(Op.TAG, {"key": "camera_track_blend_out", "value": blend_out}),
        Step(Op.REGISTER, {"registry": "cinematics"}),
    ]


# ============================================================================
# After-apply functions
# ============================================================================


def _cutscene_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @cutscene is applied."""
    id_param = params.get("id", "")
    skippable = params.get("skippable", True)
    pause_gameplay = params.get("pause_gameplay", True)

    obj._cutscene = True
    obj._cutscene_id = id_param
    obj._cutscene_skippable = skippable
    obj._cutscene_pause_gameplay = pause_gameplay


def _camera_track_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @camera_track is applied."""
    blend_in = params.get("blend_in", 0.5)
    blend_out = params.get("blend_out", 0.5)

    obj._camera_track = True
    obj._camera_track_blend_in = blend_in
    obj._camera_track_blend_out = blend_out


# ============================================================================
# Decorator creation
# ============================================================================

cutscene = make_decorator(
    name="cutscene",
    steps=_cutscene_steps,
    doc="Cutscene definition with id, skippable flag, and pause gameplay flag.",
    validate=_validate_cutscene_params,
    after_steps=_cutscene_after_apply,
)

camera_track = make_decorator(
    name="camera_track",
    steps=_camera_track_steps,
    doc="Cutscene camera with blend in/out times.",
    validate=_validate_camera_track_params,
    after_steps=_camera_track_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("cutscene", cutscene, ("class",)),
    ("camera_track", camera_track, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.CINEMATICS,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.CINEMATICS].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "cutscene",
    "camera_track",
]
