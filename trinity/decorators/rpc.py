"""
RPC decorators — built from Ops.

Decorators for remote procedure call marking.

Decorators:
    @rpc  - Mark function as remote procedure call
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from trinity.decorators.base import validate_target_type
from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import Tier, registry

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# =============================================================================
# VALID VALUES
# =============================================================================

VALID_AUTHORITIES = frozenset({"server", "client", "owner"})

# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_rpc(
    authority: str = "server", reliable: bool = True, **_: Any
) -> None:
    if authority not in VALID_AUTHORITIES:
        raise ValueError(
            f"@rpc: invalid authority '{authority}'. "
            f"Valid authorities: {sorted(VALID_AUTHORITIES)}"
        )
    if not isinstance(reliable, bool):
        raise ValueError(
            f"@rpc: 'reliable' must be a bool, got {type(reliable).__name__}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _rpc_steps(params: dict[str, Any]) -> list[Step]:
    authority = params.get("authority", "server")
    reliable = params.get("reliable", True)
    return [
        Step(Op.TAG, {"key": "rpc", "value": True}),
        Step(Op.TAG, {"key": "rpc_authority", "value": authority}),
        Step(Op.TAG, {"key": "rpc_reliable", "value": reliable}),
        Step(Op.REGISTER, {"registry": "rpc"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_rpc(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "rpc", ("function",))
    target._rpc = True
    target._rpc_authority = params.get("authority", "server")
    target._rpc_reliable = params.get("reliable", True)
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

rpc = make_decorator(
    name="rpc",
    steps=_rpc_steps,
    doc="Mark function as remote procedure call with authority and reliability settings.",
    validate=_validate_rpc,
    after_steps=_after_rpc,
)

# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("rpc", rpc, ("function",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.NETWORK_RPC,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.NETWORK_RPC].append(_spec)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "rpc",
    "VALID_AUTHORITIES",
]
