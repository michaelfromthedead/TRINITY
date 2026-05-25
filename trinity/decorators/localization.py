"""
Localization decorators — built from Ops.

These decorators mark code for internationalization and localization:
text translation, pluralization, RTL layout support, and text overflow handling.

Decorators:
    @localized      - Mark text for translation
    @plural         - Pluralization rules
    @rtl_aware      - RTL layout compatibility marker
    @text_overflow  - Text overflow handling strategy
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

VALID_OVERFLOW_STRATEGIES = frozenset({"truncate", "shrink", "scroll", "wrap"})


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_localized(
    key: Optional[str] = None,
    context: str = "",
    max_length: Optional[int] = None,
    **_: Any,
) -> None:
    if max_length is not None and (not isinstance(max_length, int) or max_length <= 0):
        raise ValueError(
            "@localized: 'max_length' must be a positive integer or None"
        )


def _validate_plural(
    one: str = "",
    other: str = "",
    zero: Optional[str] = None,
    few: Optional[str] = None,
    many: Optional[str] = None,
    **_: Any,
) -> None:
    if not one or not isinstance(one, str):
        raise ValueError("@plural: 'one' is required and must be a non-empty string")
    if not other or not isinstance(other, str):
        raise ValueError("@plural: 'other' is required and must be a non-empty string")


def _validate_text_overflow(strategy: str = "truncate", **_: Any) -> None:
    if strategy not in VALID_OVERFLOW_STRATEGIES:
        raise ValueError(
            f"@text_overflow: invalid strategy '{strategy}'. "
            f"Valid strategies: {sorted(VALID_OVERFLOW_STRATEGIES)}"
        )


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _localized_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "localized", "value": True}),
        Step(Op.TAG, {"key": "localized_key", "value": params.get("key")}),
        Step(Op.TAG, {"key": "localized_context", "value": params.get("context", "")}),
        Step(Op.TAG, {"key": "localized_max_length", "value": params.get("max_length")}),
        Step(Op.REGISTER, {"registry": "localization"}),
    ]


def _plural_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "plural", "value": True}),
        Step(Op.TAG, {"key": "plural_one", "value": params.get("one", "")}),
        Step(Op.TAG, {"key": "plural_other", "value": params.get("other", "")}),
        Step(Op.TAG, {"key": "plural_zero", "value": params.get("zero")}),
        Step(Op.TAG, {"key": "plural_few", "value": params.get("few")}),
        Step(Op.TAG, {"key": "plural_many", "value": params.get("many")}),
        Step(Op.REGISTER, {"registry": "localization"}),
    ]


def _rtl_aware_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "rtl_aware", "value": True}),
        Step(Op.REGISTER, {"registry": "localization"}),
    ]


def _text_overflow_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "text_overflow", "value": True}),
        Step(Op.TAG, {"key": "text_overflow_strategy", "value": params.get("strategy", "truncate")}),
        Step(Op.REGISTER, {"registry": "localization"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_localized(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "localized", ("function", "class"))
    target._localized = True
    target._localized_key = params.get("key")
    target._localized_context = params.get("context", "")
    target._localized_max_length = params.get("max_length")
    return None


def _after_plural(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "plural", ("function",))
    target._plural = True
    target._plural_one = params.get("one", "")
    target._plural_other = params.get("other", "")
    target._plural_zero = params.get("zero")
    target._plural_few = params.get("few")
    target._plural_many = params.get("many")
    return None


def _after_rtl_aware(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "rtl_aware", ("class",))
    target._rtl_aware = True
    return None


def _after_text_overflow(target: Any, params: dict[str, Any]) -> Any:
    validate_target_type(target, "text_overflow", ("class",))
    target._text_overflow = True
    target._text_overflow_strategy = params.get("strategy", "truncate")
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


localized = make_decorator(
    name="localized",
    steps=_localized_steps,
    doc="Mark text for translation with optional key, context, and max length.",
    validate=_validate_localized,
    after_steps=_after_localized,
)

plural = make_decorator(
    name="plural",
    steps=_plural_steps,
    doc="Define pluralization forms for translatable text.",
    validate=_validate_plural,
    after_steps=_after_plural,
)

rtl_aware = make_decorator(
    name="rtl_aware",
    steps=_rtl_aware_steps,
    doc="Mark widget as right-to-left layout compatible.",
    after_steps=_after_rtl_aware,
)

text_overflow = make_decorator(
    name="text_overflow",
    steps=_text_overflow_steps,
    doc="Specify text overflow handling strategy.",
    validate=_validate_text_overflow,
    after_steps=_after_text_overflow,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

from trinity.decorators.registry import DecoratorSpec

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("localized", localized, ("function", "class")),
    ("plural", plural, ("function",)),
    ("rtl_aware", rtl_aware, ("class",)),
    ("text_overflow", text_overflow, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.LOCALIZATION,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.LOCALIZATION].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "localized",
    "plural",
    "rtl_aware",
    "text_overflow",
    "VALID_OVERFLOW_STRATEGIES",
]
