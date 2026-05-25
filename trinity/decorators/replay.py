"""
Replay decorators — built from Ops.

Decorators for replay system: recording component state, controlling
replay authority, and keyframe-based seeking.

Decorators:
    @recorded          - Mark component for replay recording
    @replay_authority  - Control replay behavior source
    @keyframe          - Full state snapshots for seeking
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

T = TypeVar("T")

# =============================================================================
# VALID VALUES
# =============================================================================

VALID_FREQUENCIES = frozenset({"every_frame", "fixed_tick", "on_change"})
VALID_SOURCES = frozenset({"recording", "simulation", "hybrid"})

# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_recorded(frequency: str = "fixed_tick", **_: Any) -> None:
    if frequency not in VALID_FREQUENCIES:
        raise ValueError(
            f"@recorded: invalid frequency '{frequency}'. "
            f"Valid frequencies: {sorted(VALID_FREQUENCIES)}"
        )


def _validate_replay_authority(source: str = "recording", **_: Any) -> None:
    if source not in VALID_SOURCES:
        raise ValueError(
            f"@replay_authority: invalid source '{source}'. "
            f"Valid sources: {sorted(VALID_SOURCES)}"
        )


def _validate_keyframe(interval: float = 1.0, **_: Any) -> None:
    if not isinstance(interval, (int, float)) or interval <= 0:
        raise ValueError(
            f"@keyframe: interval must be a positive number, got {interval!r}"
        )

# =============================================================================
# STEP BUILDERS
# =============================================================================


def _recorded_steps(params: dict[str, Any]) -> list[Step]:
    frequency = params.get("frequency", "fixed_tick")
    return [
        Step(Op.TAG, {"key": "recorded", "value": True}),
        Step(Op.TAG, {"key": "record_frequency", "value": frequency}),
        Step(Op.REGISTER, {"registry": "replay"}),
    ]


def _replay_authority_steps(params: dict[str, Any]) -> list[Step]:
    source = params.get("source", "recording")
    return [
        Step(Op.TAG, {"key": "replay_authority", "value": True}),
        Step(Op.TAG, {"key": "replay_source", "value": source}),
        Step(Op.REGISTER, {"registry": "replay"}),
    ]


def _keyframe_steps(params: dict[str, Any]) -> list[Step]:
    interval = params.get("interval", 1.0)
    return [
        Step(Op.TAG, {"key": "keyframe", "value": True}),
        Step(Op.TAG, {"key": "keyframe_interval", "value": interval}),
        Step(Op.REGISTER, {"registry": "replay"}),
    ]

# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_recorded(target: Any, params: dict[str, Any]) -> Any:
    target._recorded = True
    target._record_frequency = params.get("frequency", "fixed_tick")
    return None


def _after_replay_authority(target: Any, params: dict[str, Any]) -> Any:
    target._replay_authority = True
    target._replay_source = params.get("source", "recording")
    return None


def _after_keyframe(target: Any, params: dict[str, Any]) -> Any:
    target._keyframe = True
    target._keyframe_interval = params.get("interval", 1.0)
    return None

# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

recorded = make_decorator(
    name="recorded",
    steps=_recorded_steps,
    doc="Mark component for replay recording at specified frequency.",
    validate=_validate_recorded,
    after_steps=_after_recorded,
)

replay_authority = make_decorator(
    name="replay_authority",
    steps=_replay_authority_steps,
    doc="Control replay behavior source (recording, simulation, or hybrid).",
    validate=_validate_replay_authority,
    after_steps=_after_replay_authority,
)

keyframe = make_decorator(
    name="keyframe",
    steps=_keyframe_steps,
    doc="Enable full state snapshots at intervals for seeking.",
    validate=_validate_keyframe,
    after_steps=_after_keyframe,
)

# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("recorded", recorded, ("class",)),
    ("replay_authority", replay_authority, ("class",)),
    ("keyframe", keyframe, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.REPLAY,
            func=_func,
            unique=True,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.REPLAY].append(_spec)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "recorded",
    "replay_authority",
    "keyframe",
    "VALID_FREQUENCIES",
    "VALID_SOURCES",
]
