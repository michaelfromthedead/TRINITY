"""
Trinity Pattern - Tier 50: NETWORK_EXTENDED Decorators

Network interest management, bandwidth allocation, interpolation, and reconciliation.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Configuration dataclasses
# ============================================================================


@dataclass(frozen=True)
class InterestConfig:
    """Interest management configuration."""

    type: str
    radius: Optional[float] = None
    always_relevant_to_owner: bool = True


@dataclass(frozen=True)
class BandwidthPriorityConfig:
    """Bandwidth priority configuration."""

    priority: int
    max_bps: Optional[int] = None


@dataclass(frozen=True)
class SnapshotInterpolationConfig:
    """Snapshot interpolation configuration."""

    buffer_size_ms: float
    interp_delay_ms: float


@dataclass(frozen=True)
class ServerReconcileConfig:
    """Server reconciliation configuration."""

    max_reconcile_frames: int
    snap_threshold: float


# ============================================================================
# Step builders
# ============================================================================


def _interest_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @interest decorator."""
    interest_type = params.get("type", "radius")
    radius = params.get("radius")
    always_relevant_to_owner = params.get("always_relevant_to_owner", True)

    return [
        Step(Op.TAG, {"key": "interest", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "interest_config",
                "value": InterestConfig(
                    type=interest_type,
                    radius=radius,
                    always_relevant_to_owner=always_relevant_to_owner,
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "network_extended"}),
    ]


def _bandwidth_priority_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @bandwidth_priority decorator."""
    priority = params.get("priority", 0)
    max_bps = params.get("max_bps")

    return [
        Step(Op.TAG, {"key": "bandwidth_priority", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "bandwidth_priority_config",
                "value": BandwidthPriorityConfig(priority=priority, max_bps=max_bps),
            },
        ),
        Step(Op.REGISTER, {"registry": "network_extended"}),
    ]


def _snapshot_interpolation_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @snapshot_interpolation decorator."""
    buffer_size_ms = params.get("buffer_size_ms", 100.0)
    interp_delay_ms = params.get("interp_delay_ms", 100.0)

    return [
        Step(Op.TAG, {"key": "snapshot_interpolation", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "snapshot_interpolation_config",
                "value": SnapshotInterpolationConfig(
                    buffer_size_ms=buffer_size_ms, interp_delay_ms=interp_delay_ms
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "network_extended"}),
    ]


def _server_reconcile_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @server_reconcile decorator."""
    max_reconcile_frames = params.get("max_reconcile_frames", 10)
    snap_threshold = params.get("snap_threshold", 0.5)

    return [
        Step(Op.TAG, {"key": "server_reconcile", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "server_reconcile_config",
                "value": ServerReconcileConfig(
                    max_reconcile_frames=max_reconcile_frames,
                    snap_threshold=snap_threshold,
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "network_extended"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_interest_params(**kwargs: Any) -> None:
    """Validate @interest parameters."""
    interest_type = kwargs.get("type")
    if interest_type not in ("radius", "grid", "custom"):
        raise ValueError(
            f"Invalid interest type '{interest_type}'. Must be 'radius', 'grid', or 'custom'"
        )

    if interest_type == "radius":
        radius = kwargs.get("radius")
        if radius is None:
            raise ValueError("radius must be provided when type='radius'")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")


def _validate_bandwidth_priority_params(**kwargs: Any) -> None:
    """Validate @bandwidth_priority parameters."""
    max_bps = kwargs.get("max_bps")
    if max_bps is not None:
        if not isinstance(max_bps, int) or max_bps <= 0:
            raise ValueError(f"max_bps must be > 0, got {max_bps}")


def _validate_snapshot_interpolation_params(**kwargs: Any) -> None:
    """Validate @snapshot_interpolation parameters."""
    buffer_size_ms = kwargs.get("buffer_size_ms", 100.0)
    if not isinstance(buffer_size_ms, (int, float)) or buffer_size_ms <= 0:
        raise ValueError(f"buffer_size_ms must be > 0, got {buffer_size_ms}")

    interp_delay_ms = kwargs.get("interp_delay_ms", 100.0)
    if not isinstance(interp_delay_ms, (int, float)) or interp_delay_ms <= 0:
        raise ValueError(f"interp_delay_ms must be > 0, got {interp_delay_ms}")


def _validate_server_reconcile_params(**kwargs: Any) -> None:
    """Validate @server_reconcile parameters."""
    max_reconcile_frames = kwargs.get("max_reconcile_frames", 10)
    if not isinstance(max_reconcile_frames, int) or max_reconcile_frames <= 0:
        raise ValueError(
            f"max_reconcile_frames must be > 0, got {max_reconcile_frames}"
        )

    snap_threshold = kwargs.get("snap_threshold", 0.5)
    if not isinstance(snap_threshold, (int, float)) or snap_threshold <= 0:
        raise ValueError(f"snap_threshold must be > 0, got {snap_threshold}")


# ============================================================================
# After-apply functions
# ============================================================================


def _interest_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @interest is applied."""
    interest_type = params.get("type", "radius")
    radius = params.get("radius")
    always_relevant_to_owner = params.get("always_relevant_to_owner", True)

    obj._interest = True
    obj._interest_type = interest_type
    obj._interest_radius = radius
    obj._interest_always_relevant_to_owner = always_relevant_to_owner
    obj._interest_config = InterestConfig(
        type=interest_type,
        radius=radius,
        always_relevant_to_owner=always_relevant_to_owner,
    )


def _bandwidth_priority_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @bandwidth_priority is applied."""
    priority = params.get("priority", 0)
    max_bps = params.get("max_bps")

    obj._bandwidth_priority = True
    obj._bandwidth_priority_value = priority
    obj._bandwidth_max_bps = max_bps
    obj._bandwidth_priority_config = BandwidthPriorityConfig(
        priority=priority, max_bps=max_bps
    )


def _snapshot_interpolation_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @snapshot_interpolation is applied."""
    buffer_size_ms = params.get("buffer_size_ms", 100.0)
    interp_delay_ms = params.get("interp_delay_ms", 100.0)

    obj._snapshot_interpolation = True
    obj._snapshot_buffer_size_ms = buffer_size_ms
    obj._snapshot_interp_delay_ms = interp_delay_ms
    obj._snapshot_interpolation_config = SnapshotInterpolationConfig(
        buffer_size_ms=buffer_size_ms, interp_delay_ms=interp_delay_ms
    )


def _server_reconcile_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @server_reconcile is applied."""
    max_reconcile_frames = params.get("max_reconcile_frames", 10)
    snap_threshold = params.get("snap_threshold", 0.5)

    obj._server_reconcile = True
    obj._server_reconcile_max_frames = max_reconcile_frames
    obj._server_reconcile_snap_threshold = snap_threshold
    obj._server_reconcile_config = ServerReconcileConfig(
        max_reconcile_frames=max_reconcile_frames, snap_threshold=snap_threshold
    )


# ============================================================================
# Decorator creation
# ============================================================================

interest = make_decorator(
    name="interest",
    steps=_interest_steps,
    validate=_validate_interest_params,
    after_steps=_interest_after_apply,
)

bandwidth_priority = make_decorator(
    name="bandwidth_priority",
    steps=_bandwidth_priority_steps,
    validate=_validate_bandwidth_priority_params,
    after_steps=_bandwidth_priority_after_apply,
)

snapshot_interpolation = make_decorator(
    name="snapshot_interpolation",
    steps=_snapshot_interpolation_steps,
    validate=_validate_snapshot_interpolation_params,
    after_steps=_snapshot_interpolation_after_apply,
)

server_reconcile = make_decorator(
    name="server_reconcile",
    steps=_server_reconcile_steps,
    validate=_validate_server_reconcile_params,
    after_steps=_server_reconcile_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("interest", interest, ("class",)),
    ("bandwidth_priority", bandwidth_priority, ("class",)),
    ("snapshot_interpolation", snapshot_interpolation, ("class",)),
    ("server_reconcile", server_reconcile, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.NETWORK_EXTENDED,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.NETWORK_EXTENDED].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "interest",
    "bandwidth_priority",
    "snapshot_interpolation",
    "server_reconcile",
    "InterestConfig",
    "BandwidthPriorityConfig",
    "SnapshotInterpolationConfig",
    "ServerReconcileConfig",
]
