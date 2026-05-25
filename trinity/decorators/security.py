"""
Security decorators — built from Ops.

These decorators mark code for security enforcement: server authority,
input validation, rate limiting, and encryption.

Decorators:
    @server_authoritative - Must be server-validated
    @validated           - Input validation rules
    @rate_limited        - Rate limiting
    @encrypted           - Sensitive data protection
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

T = TypeVar("T")

# =============================================================================
# VALID VALUES
# =============================================================================

VALID_RATE_SCOPES = frozenset({"player", "global"})

# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_validated(rules: Any = None, **_: Any) -> None:
    if rules is not None:
        for i, r in enumerate(rules):
            if not callable(r):
                raise ValueError(
                    f"@validated: rule at index {i} must be callable, got {type(r).__name__}"
                )


def _validate_rate_limited(
    max_per_second: float = 0, per: str = "player", **_: Any
) -> None:
    if not isinstance(max_per_second, (int, float)) or max_per_second <= 0:
        raise ValueError(
            f"@rate_limited: 'max_per_second' must be > 0, got {max_per_second!r}"
        )
    if per not in VALID_RATE_SCOPES:
        raise ValueError(
            f"@rate_limited: invalid scope '{per}'. "
            f"Valid scopes: {sorted(VALID_RATE_SCOPES)}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _server_authoritative_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "server_authoritative", "value": True}),
        Step(Op.REGISTER, {"registry": "security"}),
    ]


def _validated_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "validated", "value": True}),
        Step(
            Op.TAG,
            {"key": "validation_rules", "value": list(params.get("rules") or [])},
        ),
        Step(Op.REGISTER, {"registry": "security"}),
    ]


def _rate_limited_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "rate_limited", "value": True}),
        Step(
            Op.TAG,
            {"key": "rate_limit_max", "value": params.get("max_per_second", 0)},
        ),
        Step(Op.TAG, {"key": "rate_limit_per", "value": params.get("per", "player")}),
        Step(Op.REGISTER, {"registry": "security"}),
    ]


def _encrypted_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "encrypted", "value": True}),
        Step(Op.REGISTER, {"registry": "security"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_server_authoritative(target: Any, params: dict[str, Any]) -> Any:
    target._server_authoritative = True
    return None


def _after_validated(target: Any, params: dict[str, Any]) -> Any:
    target._validated = True
    target._validation_rules = list(params.get("rules") or [])
    return None


def _after_rate_limited(target: Any, params: dict[str, Any]) -> Any:
    target._rate_limited = True
    target._rate_limit_max = params.get("max_per_second", 0)
    target._rate_limit_per = params.get("per", "player")
    return None


def _after_encrypted(target: Any, params: dict[str, Any]) -> Any:
    target._encrypted = True
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

server_authoritative = make_decorator(
    name="server_authoritative",
    steps=_server_authoritative_steps,
    doc="Mark as requiring server-side validation.",
    after_steps=_after_server_authoritative,
)

validated = make_decorator(
    name="validated",
    steps=_validated_steps,
    doc="Attach input validation rules.",
    validate=_validate_validated,
    after_steps=_after_validated,
)

rate_limited = make_decorator(
    name="rate_limited",
    steps=_rate_limited_steps,
    doc="Apply rate limiting.",
    validate=_validate_rate_limited,
    after_steps=_after_rate_limited,
)

encrypted = make_decorator(
    name="encrypted",
    steps=_encrypted_steps,
    doc="Mark as containing sensitive encrypted data.",
    after_steps=_after_encrypted,
)

# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("server_authoritative", server_authoritative, ("function",)),
    ("validated", validated, ("class",)),
    ("rate_limited", rate_limited, ("function",)),
    ("encrypted", encrypted, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.SECURITY,
            func=_func,
            unique=True,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.SECURITY].append(_spec)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "server_authoritative",
    "validated",
    "rate_limited",
    "encrypted",
    "VALID_RATE_SCOPES",
]
