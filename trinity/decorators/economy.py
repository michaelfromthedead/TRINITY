"""
Trinity Pattern - Tier 38: ECONOMY Decorators

Economy system decorators for currency, transactions, and monetization.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _currency_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @currency decorator."""
    id_value = params.get("id", "")
    premium = params.get("premium", False)
    max_value = params.get("max_value")

    steps = [
        Step(Op.TAG, {"key": "currency", "value": True}),
        Step(Op.TAG, {"key": "currency_id", "value": id_value}),
        Step(Op.TAG, {"key": "currency_premium", "value": premium}),
        Step(Op.REGISTER, {"registry": "economy"}),
    ]

    if max_value is not None:
        steps.append(Step(Op.TAG, {"key": "currency_max_value", "value": max_value}))

    return steps


def _transaction_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @transaction decorator."""
    atomic = params.get("atomic", True)
    log = params.get("log", True)

    return [
        Step(Op.TAG, {"key": "transaction", "value": True}),
        Step(Op.TAG, {"key": "transaction_atomic", "value": atomic}),
        Step(Op.TAG, {"key": "transaction_log", "value": log}),
        Step(Op.REGISTER, {"registry": "economy"}),
    ]


def _mtx_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @mtx decorator."""
    product_id = params.get("product_id", "")
    platforms = params.get("platforms", {})

    return [
        Step(Op.TAG, {"key": "mtx", "value": True}),
        Step(Op.TAG, {"key": "mtx_product_id", "value": product_id}),
        Step(Op.TAG, {"key": "mtx_platforms", "value": dict(platforms)}),
        Step(Op.REGISTER, {"registry": "economy"}),
    ]


def _daily_reward_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @daily_reward decorator."""
    reset_hour_utc = params.get("reset_hour_utc", 0)

    return [
        Step(Op.TAG, {"key": "daily_reward", "value": True}),
        Step(Op.TAG, {"key": "daily_reward_reset_hour", "value": reset_hour_utc}),
        Step(Op.REGISTER, {"registry": "economy"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_currency(**kwargs: Any) -> None:
    """Validate @currency parameters."""
    id_value = kwargs.get("id", "")
    if not id_value:
        raise ValueError("@currency: 'id' parameter is required and must be non-empty")

    max_value = kwargs.get("max_value")
    if max_value is not None and max_value <= 0:
        raise ValueError(f"@currency: 'max_value' must be > 0, got {max_value}")


def _validate_mtx(**kwargs: Any) -> None:
    """Validate @mtx parameters."""
    product_id = kwargs.get("product_id", "")
    if not product_id:
        raise ValueError("@mtx: 'product_id' parameter is required and must be non-empty")

    platforms = kwargs.get("platforms", {})
    if not platforms:
        raise ValueError("@mtx: 'platforms' must be a non-empty dict")
    if not isinstance(platforms, dict):
        raise TypeError("@mtx: 'platforms' must be a dict")


def _validate_daily_reward(**kwargs: Any) -> None:
    """Validate @daily_reward parameters."""
    reset_hour_utc = kwargs.get("reset_hour_utc", 0)
    if not isinstance(reset_hour_utc, int):
        raise TypeError(
            f"@daily_reward: 'reset_hour_utc' must be an int, got {type(reset_hour_utc).__name__}"
        )
    if not 0 <= reset_hour_utc <= 23:
        raise ValueError(
            f"@daily_reward: 'reset_hour_utc' must be 0-23, got {reset_hour_utc}"
        )


# ============================================================================
# After-apply functions
# ============================================================================


def _currency_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @currency is applied."""
    obj._currency = True
    obj._currency_id = params.get("id", "")
    obj._currency_premium = params.get("premium", False)
    max_value = params.get("max_value")
    if max_value is not None:
        obj._currency_max_value = max_value


def _transaction_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @transaction is applied."""
    obj._transaction = True
    obj._transaction_atomic = params.get("atomic", True)
    obj._transaction_log = params.get("log", True)


def _mtx_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @mtx is applied."""
    obj._mtx = True
    obj._mtx_product_id = params.get("product_id", "")
    obj._mtx_platforms = dict(params.get("platforms", {}))


def _daily_reward_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @daily_reward is applied."""
    obj._daily_reward = True
    obj._daily_reward_reset_hour = params.get("reset_hour_utc", 0)


# ============================================================================
# Decorator creation
# ============================================================================

currency = make_decorator(
    name="currency",
    steps=_currency_steps,
    doc="Currency definition for in-game economy.",
    validate=_validate_currency,
    after_steps=_currency_after_apply,
)

transaction = make_decorator(
    name="transaction",
    steps=_transaction_steps,
    doc="Economy transaction with atomic and logging options.",
    after_steps=_transaction_after_apply,
)

mtx = make_decorator(
    name="mtx",
    steps=_mtx_steps,
    doc="Microtransaction product definition with platform mappings.",
    validate=_validate_mtx,
    after_steps=_mtx_after_apply,
)

daily_reward = make_decorator(
    name="daily_reward",
    steps=_daily_reward_steps,
    doc="Daily reward system with configurable reset time.",
    validate=_validate_daily_reward,
    after_steps=_daily_reward_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("currency", currency, ("class",)),
    ("transaction", transaction, ("function", "class")),
    ("mtx", mtx, ("class",)),
    ("daily_reward", daily_reward, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.ECONOMY,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.ECONOMY].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "currency",
    "transaction",
    "mtx",
    "daily_reward",
]
