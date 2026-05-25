"""
Trinity Pattern - Tier 40: ERROR_HANDLING Decorators

Error handling, crash recovery, and debugging decorators.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_RECOVERY_STRATEGIES = frozenset({"retry", "skip", "fallback", "crash"})
VALID_ERROR_SCOPES = frozenset({"system", "entity", "global"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _crash_safe_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @crash_safe decorator."""
    recovery = params.get("recovery", "retry")

    return [
        Step(Op.TAG, {"key": "crash_safe", "value": True}),
        Step(Op.TAG, {"key": "crash_recovery", "value": recovery}),
        Step(Op.REGISTER, {"registry": "error_handling"}),
    ]


def _recoverable_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @recoverable decorator."""
    checkpoint = params.get("checkpoint", True)

    return [
        Step(Op.TAG, {"key": "recoverable", "value": True}),
        Step(Op.TAG, {"key": "recoverable_checkpoint", "value": checkpoint}),
        Step(Op.REGISTER, {"registry": "error_handling"}),
        Step(Op.TRACK, {}),
    ]


def _error_boundary_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @error_boundary decorator."""
    scope = params.get("scope", "system")

    return [
        Step(Op.TAG, {"key": "error_boundary", "value": True}),
        Step(Op.TAG, {"key": "error_scope", "value": scope}),
        Step(Op.REGISTER, {"registry": "error_handling"}),
    ]


def _bug_report_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @bug_report decorator."""
    include = params.get("include", {"screenshot", "logs", "save", "replay"})

    return [
        Step(Op.TAG, {"key": "bug_report", "value": True}),
        Step(Op.TAG, {"key": "bug_report_include", "value": frozenset(include)}),
        Step(Op.REGISTER, {"registry": "error_handling"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_crash_safe_params(**kwargs: Any) -> None:
    """Validate @crash_safe parameters."""
    recovery = kwargs.get("recovery")
    if recovery not in VALID_RECOVERY_STRATEGIES:
        raise ValueError(
            f"Invalid recovery strategy '{recovery}'. "
            f"Must be one of {sorted(VALID_RECOVERY_STRATEGIES)}"
        )


def _validate_error_boundary_params(**kwargs: Any) -> None:
    """Validate @error_boundary parameters."""
    scope = kwargs.get("scope", "system")
    if scope not in VALID_ERROR_SCOPES:
        raise ValueError(
            f"Invalid scope '{scope}'. Must be one of {sorted(VALID_ERROR_SCOPES)}"
        )


def _validate_bug_report_params(**kwargs: Any) -> None:
    """Validate @bug_report parameters."""
    include = kwargs.get("include", {"screenshot", "logs", "save", "replay"})
    if not include:
        raise ValueError("include must be a non-empty set")
    if not isinstance(include, (set, frozenset)):
        raise TypeError("include must be a set")


# ============================================================================
# After-apply functions
# ============================================================================


def _crash_safe_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @crash_safe is applied."""
    recovery = params.get("recovery", "retry")

    obj._crash_safe = True
    obj._crash_recovery = recovery


def _recoverable_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @recoverable is applied."""
    checkpoint = params.get("checkpoint", True)

    obj._recoverable = True
    obj._recoverable_checkpoint = checkpoint


def _error_boundary_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @error_boundary is applied."""
    scope = params.get("scope", "system")

    obj._error_boundary = True
    obj._error_scope = scope


def _bug_report_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @bug_report is applied."""
    include = params.get("include", {"screenshot", "logs", "save", "replay"})

    obj._bug_report = True
    obj._bug_report_include = frozenset(include)


# ============================================================================
# Decorator creation
# ============================================================================

crash_safe = make_decorator(
    name="crash_safe",
    steps=_crash_safe_steps,
    doc="Mark code with crash handling strategy (retry, skip, fallback, crash).",
    validate=_validate_crash_safe_params,
    after_steps=_crash_safe_after_apply,
)

recoverable = make_decorator(
    name="recoverable",
    steps=_recoverable_steps,
    doc="Enable state recovery with optional checkpoint support.",
    after_steps=_recoverable_after_apply,
)

error_boundary = make_decorator(
    name="error_boundary",
    steps=_error_boundary_steps,
    doc="Define error isolation boundary (system, entity, global).",
    validate=_validate_error_boundary_params,
    after_steps=_error_boundary_after_apply,
)

bug_report = make_decorator(
    name="bug_report",
    steps=_bug_report_steps,
    doc="Auto-generate bug report data (screenshot, logs, save, replay).",
    validate=_validate_bug_report_params,
    after_steps=_bug_report_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("crash_safe", crash_safe, ("class", "function")),
    ("recoverable", recoverable, ("class",)),
    ("error_boundary", error_boundary, ("class", "function")),
    ("bug_report", bug_report, ("class", "function")),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.ERROR_HANDLING,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.ERROR_HANDLING].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "crash_safe",
    "recoverable",
    "error_boundary",
    "bug_report",
    "VALID_RECOVERY_STRATEGIES",
    "VALID_ERROR_SCOPES",
]
