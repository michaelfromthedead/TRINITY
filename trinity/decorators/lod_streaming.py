"""
LOD & Streaming decorators — built from Ops.

Decorators for level-of-detail management, asset streaming,
and world chunking.

Decorators:
    @lod              - Level of detail management
    @streamable       - Asset streaming configuration
    @chunk            - World chunk definition
    @loading_priority - Smart load ordering by weights
    @unloadable       - Unload policy with age and state saving
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

VALID_STREAM_PRIORITIES = frozenset({"critical", "high", "normal", "low"})

# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_lod(
    levels: int = 4,
    distances: Optional[list[float]] = None,
    bias: float = 0.0,
    **_: Any,
) -> None:
    if not isinstance(levels, int) or levels <= 0:
        raise ValueError("@lod: 'levels' must be an int > 0")
    if distances is not None:
        if len(distances) != levels:
            raise ValueError(
                f"@lod: 'distances' length ({len(distances)}) must equal "
                f"'levels' ({levels})"
            )
        for d in distances:
            if d <= 0:
                raise ValueError("@lod: all distances must be > 0")
        for i in range(len(distances) - 1):
            if distances[i + 1] <= distances[i]:
                raise ValueError("@lod: distances must be strictly ascending")


def _validate_streamable(
    priority: str = "normal",
    keep_loaded: bool = False,
    **_: Any,
) -> None:
    if priority not in VALID_STREAM_PRIORITIES:
        raise ValueError(
            f"@streamable: invalid priority '{priority}'. "
            f"Valid priorities: {sorted(VALID_STREAM_PRIORITIES)}"
        )


def _validate_loading_priority(
    visibility_weight: float = 1.0,
    player_velocity_weight: float = 1.0,
    **_: Any,
) -> None:
    if visibility_weight < 0:
        raise ValueError(
            f"@loading_priority: 'visibility_weight' must be >= 0, got {visibility_weight}"
        )
    if player_velocity_weight < 0:
        raise ValueError(
            f"@loading_priority: 'player_velocity_weight' must be >= 0, got {player_velocity_weight}"
        )


def _validate_unloadable(
    min_age: float = 60.0,
    save_state: bool = True,
    **_: Any,
) -> None:
    if min_age <= 0:
        raise ValueError(
            f"@unloadable: 'min_age' must be > 0, got {min_age}"
        )


def _validate_chunk(
    size: Any = None,
    overlap: float = 0.0,
    **_: Any,
) -> None:
    if size is None:
        raise ValueError("@chunk: 'size' parameter is required")
    if not (isinstance(size, (tuple, list)) and len(size) == 3):
        raise ValueError("@chunk: 'size' must be a tuple of 3 floats")
    for v in size:
        if v <= 0:
            raise ValueError("@chunk: all size values must be > 0")
    if overlap < 0:
        raise ValueError("@chunk: 'overlap' must be >= 0")


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _lod_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "lod", "value": True}),
        Step(Op.TAG, {"key": "lod_levels", "value": params.get("levels", 4)}),
        Step(Op.TAG, {"key": "lod_distances", "value": params.get("distances")}),
        Step(Op.TAG, {"key": "lod_bias", "value": params.get("bias", 0.0)}),
        Step(Op.REGISTER, {"registry": "lod_streaming"}),
    ]


def _streamable_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "streamable", "value": True}),
        Step(Op.TAG, {"key": "stream_priority", "value": params.get("priority", "normal")}),
        Step(Op.TAG, {"key": "stream_keep_loaded", "value": params.get("keep_loaded", False)}),
        Step(Op.REGISTER, {"registry": "lod_streaming"}),
    ]


def _loading_priority_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "loading_priority", "value": True}),
        Step(Op.TAG, {"key": "loading_priority_visibility_weight", "value": params.get("visibility_weight", 1.0)}),
        Step(Op.TAG, {"key": "loading_priority_player_velocity_weight", "value": params.get("player_velocity_weight", 1.0)}),
        Step(Op.REGISTER, {"registry": "lod_streaming"}),
    ]


def _unloadable_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "unloadable", "value": True}),
        Step(Op.TAG, {"key": "unloadable_min_age", "value": params.get("min_age", 60.0)}),
        Step(Op.TAG, {"key": "unloadable_save_state", "value": params.get("save_state", True)}),
        Step(Op.REGISTER, {"registry": "lod_streaming"}),
    ]


def _chunk_steps(params: dict[str, Any]) -> list[Step]:
    size = params.get("size")
    return [
        Step(Op.TAG, {"key": "chunk", "value": True}),
        Step(Op.TAG, {"key": "chunk_size", "value": tuple(size) if size else None}),
        Step(Op.TAG, {"key": "chunk_overlap", "value": params.get("overlap", 0.0)}),
        Step(Op.REGISTER, {"registry": "lod_streaming"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_lod(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "lod", ("class",))
    target._lod = True
    target._lod_levels = params.get("levels", 4)
    target._lod_distances = params.get("distances")
    target._lod_bias = params.get("bias", 0.0)
    return None


def _after_streamable(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "streamable", ("class",))
    target._streamable = True
    target._stream_priority = params.get("priority", "normal")
    target._stream_keep_loaded = params.get("keep_loaded", False)
    return None


def _after_loading_priority(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "loading_priority", ("class",))
    target._loading_priority = True
    target._loading_priority_visibility_weight = params.get("visibility_weight", 1.0)
    target._loading_priority_player_velocity_weight = params.get("player_velocity_weight", 1.0)
    return None


def _after_unloadable(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "unloadable", ("class",))
    target._unloadable = True
    target._unloadable_min_age = params.get("min_age", 60.0)
    target._unloadable_save_state = params.get("save_state", True)
    return None


def _after_chunk(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "chunk", ("class",))
    size = params.get("size")
    target._chunk = True
    target._chunk_size = tuple(size) if size else None
    target._chunk_overlap = params.get("overlap", 0.0)
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

lod = make_decorator(
    name="lod",
    steps=_lod_steps,
    doc="Level of detail management for classes.",
    validate=_validate_lod,
    after_steps=_after_lod,
)

streamable = make_decorator(
    name="streamable",
    steps=_streamable_steps,
    doc="Mark class for asset streaming with priority control.",
    validate=_validate_streamable,
    after_steps=_after_streamable,
)

chunk = make_decorator(
    name="chunk",
    steps=_chunk_steps,
    doc="Define world chunk dimensions and overlap.",
    validate=_validate_chunk,
    after_steps=_after_chunk,
)

loading_priority = make_decorator(
    name="loading_priority",
    steps=_loading_priority_steps,
    validate=_validate_loading_priority,
    after_steps=_after_loading_priority,
)

unloadable = make_decorator(
    name="unloadable",
    steps=_unloadable_steps,
    validate=_validate_unloadable,
    after_steps=_after_unloadable,
)

# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("lod", lod, ("class",)),
    ("streamable", streamable, ("class",)),
    ("chunk", chunk, ("class",)),
    ("loading_priority", loading_priority, ("class",)),
    ("unloadable", unloadable, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.LOD_STREAMING,
            func=_func,
            unique=True,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.LOD_STREAMING].append(_spec)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "lod",
    "streamable",
    "chunk",
    "loading_priority",
    "unloadable",
    "VALID_STREAM_PRIORITIES",
]
