"""
Audio decorators — built from Ops.

Decorators for audio system configuration: sound banks, audio buses,
and spatial audio settings.

Decorators:
    @sound        - Mark class as sound component with bank
    @audio_bus    - Configure audio bus routing
    @spatial_audio - Configure spatial audio falloff
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

VALID_FALLOFF_TYPES = frozenset({"inverse", "linear", "exponential"})


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_sound(bank: str = "", preload: bool = False, **_: Any) -> None:
    if not bank:
        raise ValueError("@sound: 'bank' parameter is required and must be non-empty")
    if not isinstance(bank, str):
        raise ValueError("@sound: 'bank' must be a string")


def _validate_audio_bus(
    name: str = "",
    volume: float = 1.0,
    effects: Optional[list[str]] = None,
    **_: Any,
) -> None:
    if not name:
        raise ValueError(
            "@audio_bus: 'name' parameter is required and must be non-empty"
        )
    if not isinstance(volume, (int, float)):
        raise ValueError("@audio_bus: 'volume' must be a number")
    if not (0.0 <= volume <= 1.0):
        raise ValueError(
            f"@audio_bus: 'volume' must be between 0.0 and 1.0, got {volume}"
        )
    if effects is not None and not isinstance(effects, (list, tuple)):
        raise ValueError("@audio_bus: 'effects' must be a list of strings or None")


def _validate_spatial_audio(
    falloff: str = "inverse",
    max_distance: float = 100.0,
    **_: Any,
) -> None:
    if falloff not in VALID_FALLOFF_TYPES:
        raise ValueError(
            f"@spatial_audio: invalid falloff '{falloff}'. "
            f"Valid types: {sorted(VALID_FALLOFF_TYPES)}"
        )
    if not isinstance(max_distance, (int, float)):
        raise ValueError("@spatial_audio: 'max_distance' must be a number")
    if max_distance <= 0:
        raise ValueError(
            f"@spatial_audio: 'max_distance' must be > 0, got {max_distance}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _sound_steps(params: dict[str, Any]) -> list[Step]:
    bank = params.get("bank", "")
    preload = params.get("preload", False)
    return [
        Step(Op.TAG, {"key": "sound", "value": True}),
        Step(Op.TAG, {"key": "sound_bank", "value": bank}),
        Step(Op.TAG, {"key": "sound_preload", "value": preload}),
        Step(Op.REGISTER, {"registry": "audio"}),
    ]


def _audio_bus_steps(params: dict[str, Any]) -> list[Step]:
    name = params.get("name", "")
    volume = params.get("volume", 1.0)
    effects = list(params.get("effects") or [])
    return [
        Step(Op.TAG, {"key": "audio_bus", "value": True}),
        Step(Op.TAG, {"key": "bus_name", "value": name}),
        Step(Op.TAG, {"key": "bus_volume", "value": volume}),
        Step(Op.TAG, {"key": "bus_effects", "value": effects}),
        Step(Op.REGISTER, {"registry": "audio"}),
    ]


def _spatial_audio_steps(params: dict[str, Any]) -> list[Step]:
    falloff = params.get("falloff", "inverse")
    max_distance = params.get("max_distance", 100.0)
    return [
        Step(Op.TAG, {"key": "spatial_audio", "value": True}),
        Step(Op.TAG, {"key": "audio_falloff", "value": falloff}),
        Step(Op.TAG, {"key": "audio_max_distance", "value": max_distance}),
        Step(Op.REGISTER, {"registry": "audio"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_sound(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "sound", ("class",))
    target._sound = True
    target._sound_bank = params.get("bank", "")
    target._sound_preload = params.get("preload", False)
    return None


def _after_audio_bus(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "audio_bus", ("class",))
    target._audio_bus = True
    target._bus_name = params.get("name", "")
    target._bus_volume = params.get("volume", 1.0)
    target._bus_effects = list(params.get("effects") or [])
    return None


def _after_spatial_audio(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "spatial_audio", ("class",))
    target._spatial_audio = True
    target._audio_falloff = params.get("falloff", "inverse")
    target._audio_max_distance = params.get("max_distance", 100.0)
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


sound = make_decorator(
    name="sound",
    steps=_sound_steps,
    doc="Mark class as sound component with a sound bank.",
    validate=_validate_sound,
    after_steps=_after_sound,
)

audio_bus = make_decorator(
    name="audio_bus",
    steps=_audio_bus_steps,
    doc="Configure audio bus with name, volume, and effects chain.",
    validate=_validate_audio_bus,
    after_steps=_after_audio_bus,
)

spatial_audio = make_decorator(
    name="spatial_audio",
    steps=_spatial_audio_steps,
    doc="Configure spatial audio with falloff and max distance.",
    validate=_validate_spatial_audio,
    after_steps=_after_spatial_audio,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("sound", sound, ("class",)),
    ("audio_bus", audio_bus, ("class",)),
    ("spatial_audio", spatial_audio, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.AUDIO,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.AUDIO].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "sound",
    "audio_bus",
    "spatial_audio",
    "VALID_FALLOFF_TYPES",
]
