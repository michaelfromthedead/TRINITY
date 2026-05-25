"""
Trinity Pattern - Tier 53: BRIDGES_CACHING Decorators

Cross-cutting bridge decorators for caching, lazy loading, batching,
async loading, diffing, priority, retry, network throttling, and observability.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from typing import Any, Callable, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Valid constants
VALID_CACHE_SCOPES = frozenset({"global", "entity", "frame"})
VALID_LAZY_INIT_MODES = frozenset({"first_access", "first_frame", "explicit"})
VALID_FLUSH_MODES = frozenset({"full", "frame_end", "explicit", "timeout"})
VALID_DIFF_STRATEGIES = frozenset({"shallow", "deep", "structural", "custom"})
VALID_BACKOFF_STRATEGIES = frozenset({"none", "linear", "exponential"})
VALID_THROTTLE_SCOPES = frozenset({"entity", "component", "global"})
VALID_NOTIFY_MODES = frozenset({"sync", "deferred", "batched"})

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Step builders
# ============================================================================


def _cached_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @cached decorator."""
    ttl = params.get("ttl")
    max_size = params.get("max_size")
    scope = params.get("scope", "global")

    return [
        Step(Op.TAG, {"key": "cached", "value": True}),
        Step(Op.TAG, {"key": "cached_ttl", "value": ttl}),
        Step(Op.TAG, {"key": "cached_max_size", "value": max_size}),
        Step(Op.TAG, {"key": "cached_scope", "value": scope}),
        Step(Op.REGISTER, {"registry": "bridges_caching"}),
    ]


def _lazy_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @lazy decorator."""
    init_on = params.get("init_on", "first_access")
    thread_safe = params.get("thread_safe", True)
    fallback = params.get("fallback")

    return [
        Step(Op.TAG, {"key": "lazy", "value": True}),
        Step(Op.TAG, {"key": "lazy_init_on", "value": init_on}),
        Step(Op.TAG, {"key": "lazy_thread_safe", "value": thread_safe}),
        Step(Op.TAG, {"key": "lazy_fallback", "value": fallback}),
        Step(Op.REGISTER, {"registry": "bridges_caching"}),
    ]


def _batch_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @batch decorator."""
    max_size = params.get("max_size", 64)
    flush_on = params.get("flush_on", "frame_end")
    timeout_ms = params.get("timeout_ms")
    coalesce = params.get("coalesce", False)

    return [
        Step(Op.TAG, {"key": "batch", "value": True}),
        Step(Op.TAG, {"key": "batch_max_size", "value": max_size}),
        Step(Op.TAG, {"key": "batch_flush_on", "value": flush_on}),
        Step(Op.TAG, {"key": "batch_timeout_ms", "value": timeout_ms}),
        Step(Op.TAG, {"key": "batch_coalesce", "value": coalesce}),
        Step(Op.REGISTER, {"registry": "bridges_caching"}),
    ]


def _async_load_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @async_load decorator."""
    priority = params.get("priority", 0)
    timeout_ms = params.get("timeout_ms")
    fallback = params.get("fallback")

    return [
        Step(Op.TAG, {"key": "async_load", "value": True}),
        Step(Op.TAG, {"key": "async_load_priority", "value": priority}),
        Step(Op.TAG, {"key": "async_load_timeout_ms", "value": timeout_ms}),
        Step(Op.TAG, {"key": "async_load_fallback", "value": fallback}),
        Step(Op.REGISTER, {"registry": "bridges_caching"}),
    ]


def _diff_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @diff decorator."""
    strategy = params.get("strategy", "shallow")
    include_fields = params.get("include_fields")
    exclude_fields = params.get("exclude_fields")

    return [
        Step(Op.TAG, {"key": "diff", "value": True}),
        Step(Op.TAG, {"key": "diff_strategy", "value": strategy}),
        Step(Op.TAG, {"key": "diff_include_fields", "value": include_fields}),
        Step(Op.TAG, {"key": "diff_exclude_fields", "value": exclude_fields}),
        Step(Op.REGISTER, {"registry": "bridges_caching"}),
    ]


def _priority_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @priority decorator."""
    value = params.get("value", 0)
    queue = params.get("queue", "default")
    higher_first = params.get("higher_first", True)

    return [
        Step(Op.TAG, {"key": "priority", "value": True}),
        Step(Op.TAG, {"key": "priority_value", "value": value}),
        Step(Op.TAG, {"key": "priority_queue", "value": queue}),
        Step(Op.TAG, {"key": "priority_higher_first", "value": higher_first}),
        Step(Op.REGISTER, {"registry": "bridges_caching"}),
    ]


def _retry_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @retry decorator."""
    max_attempts = params.get("max_attempts", 3)
    backoff = params.get("backoff", "exponential")
    base_delay_ms = params.get("base_delay_ms", 100)
    max_delay_ms = params.get("max_delay_ms", 10000)

    return [
        Step(Op.TAG, {"key": "retry", "value": True}),
        Step(Op.TAG, {"key": "retry_max_attempts", "value": max_attempts}),
        Step(Op.TAG, {"key": "retry_backoff", "value": backoff}),
        Step(Op.TAG, {"key": "retry_base_delay_ms", "value": base_delay_ms}),
        Step(Op.TAG, {"key": "retry_max_delay_ms", "value": max_delay_ms}),
        Step(Op.REGISTER, {"registry": "bridges_caching"}),
    ]


def _throttle_network_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @throttle_network decorator."""
    max_updates_per_second = params.get("max_updates_per_second", 20.0)
    priority_decay = params.get("priority_decay", 0.9)
    burst_allowance = params.get("burst_allowance", 3)
    per = params.get("per", "entity")

    return [
        Step(Op.TAG, {"key": "throttle_network", "value": True}),
        Step(Op.TAG, {"key": "throttle_network_max_ups", "value": max_updates_per_second}),
        Step(Op.TAG, {"key": "throttle_network_priority_decay", "value": priority_decay}),
        Step(Op.TAG, {"key": "throttle_network_burst_allowance", "value": burst_allowance}),
        Step(Op.TAG, {"key": "throttle_network_per", "value": per}),
        Step(Op.REGISTER, {"registry": "bridges_caching"}),
    ]


def _observable_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @observable decorator."""
    notify = params.get("notify", "sync")
    batch_delay_ms = params.get("batch_delay_ms", 16.0)

    return [
        Step(Op.TAG, {"key": "observable", "value": True}),
        Step(Op.TAG, {"key": "observable_notify", "value": notify}),
        Step(Op.TAG, {"key": "observable_batch_delay_ms", "value": batch_delay_ms}),
        Step(Op.REGISTER, {"registry": "bridges_caching"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_cached_params(**kwargs: Any) -> None:
    """Validate @cached parameters."""
    scope = kwargs.get("scope", "global")
    if scope not in VALID_CACHE_SCOPES:
        raise ValueError(
            f"Invalid scope '{scope}'. Must be one of {sorted(VALID_CACHE_SCOPES)}"
        )

    ttl = kwargs.get("ttl")
    if ttl is not None and ttl <= 0:
        raise ValueError(f"ttl must be > 0, got {ttl}")

    max_size = kwargs.get("max_size")
    if max_size is not None and max_size <= 0:
        raise ValueError(f"max_size must be > 0, got {max_size}")


def _validate_lazy_params(**kwargs: Any) -> None:
    """Validate @lazy parameters."""
    init_on = kwargs.get("init_on", "first_access")
    if init_on not in VALID_LAZY_INIT_MODES:
        raise ValueError(
            f"Invalid init_on '{init_on}'. Must be one of {sorted(VALID_LAZY_INIT_MODES)}"
        )


def _validate_batch_params(**kwargs: Any) -> None:
    """Validate @batch parameters."""
    max_size = kwargs.get("max_size", 64)
    if not isinstance(max_size, int) or max_size <= 0:
        raise ValueError(f"max_size must be > 0, got {max_size}")

    flush_on = kwargs.get("flush_on", "frame_end")
    if flush_on not in VALID_FLUSH_MODES:
        raise ValueError(
            f"Invalid flush_on '{flush_on}'. Must be one of {sorted(VALID_FLUSH_MODES)}"
        )

    timeout_ms = kwargs.get("timeout_ms")
    if timeout_ms is not None and timeout_ms <= 0:
        raise ValueError(f"timeout_ms must be > 0, got {timeout_ms}")


def _validate_async_load_params(**kwargs: Any) -> None:
    """Validate @async_load parameters."""
    timeout_ms = kwargs.get("timeout_ms")
    if timeout_ms is not None and timeout_ms <= 0:
        raise ValueError(f"timeout_ms must be > 0, got {timeout_ms}")


def _validate_diff_params(**kwargs: Any) -> None:
    """Validate @diff parameters."""
    strategy = kwargs.get("strategy", "shallow")
    if strategy not in VALID_DIFF_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Must be one of {sorted(VALID_DIFF_STRATEGIES)}"
        )

    include_fields = kwargs.get("include_fields")
    exclude_fields = kwargs.get("exclude_fields")
    if include_fields is not None and exclude_fields is not None:
        raise ValueError("Cannot specify both include_fields and exclude_fields")

    custom_differ = kwargs.get("custom_differ")
    if strategy == "custom" and custom_differ is None:
        raise ValueError("custom_differ is required when strategy is 'custom'")


def _validate_priority_params(**kwargs: Any) -> None:
    """Validate @priority parameters."""
    queue = kwargs.get("queue", "default")
    if not queue:
        raise ValueError("queue must be a non-empty string")


def _validate_retry_params(**kwargs: Any) -> None:
    """Validate @retry parameters."""
    max_attempts = kwargs.get("max_attempts", 3)
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError(f"max_attempts must be > 0, got {max_attempts}")

    backoff = kwargs.get("backoff", "exponential")
    if backoff not in VALID_BACKOFF_STRATEGIES:
        raise ValueError(
            f"Invalid backoff '{backoff}'. Must be one of {sorted(VALID_BACKOFF_STRATEGIES)}"
        )

    base_delay_ms = kwargs.get("base_delay_ms", 100)
    if base_delay_ms <= 0:
        raise ValueError(f"base_delay_ms must be > 0, got {base_delay_ms}")

    max_delay_ms = kwargs.get("max_delay_ms", 10000)
    if max_delay_ms <= 0:
        raise ValueError(f"max_delay_ms must be > 0, got {max_delay_ms}")

    if base_delay_ms > max_delay_ms:
        raise ValueError(
            f"base_delay_ms ({base_delay_ms}) must be <= max_delay_ms ({max_delay_ms})"
        )


def _validate_throttle_network_params(**kwargs: Any) -> None:
    """Validate @throttle_network parameters."""
    max_updates_per_second = kwargs.get("max_updates_per_second", 20.0)
    if max_updates_per_second <= 0:
        raise ValueError(
            f"max_updates_per_second must be > 0, got {max_updates_per_second}"
        )

    priority_decay = kwargs.get("priority_decay", 0.9)
    if not 0 <= priority_decay <= 1:
        raise ValueError(
            f"priority_decay must be between 0 and 1, got {priority_decay}"
        )

    burst_allowance = kwargs.get("burst_allowance", 3)
    if not isinstance(burst_allowance, int) or burst_allowance < 0:
        raise ValueError(f"burst_allowance must be >= 0, got {burst_allowance}")

    per = kwargs.get("per", "entity")
    if per not in VALID_THROTTLE_SCOPES:
        raise ValueError(
            f"Invalid per '{per}'. Must be one of {sorted(VALID_THROTTLE_SCOPES)}"
        )


def _validate_observable_params(**kwargs: Any) -> None:
    """Validate @observable parameters."""
    notify = kwargs.get("notify", "sync")
    if notify not in VALID_NOTIFY_MODES:
        raise ValueError(
            f"Invalid notify '{notify}'. Must be one of {sorted(VALID_NOTIFY_MODES)}"
        )

    batch_delay_ms = kwargs.get("batch_delay_ms", 16.0)
    if batch_delay_ms <= 0:
        raise ValueError(f"batch_delay_ms must be > 0, got {batch_delay_ms}")


# ============================================================================
# After-apply functions
# ============================================================================


def _cached_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @cached is applied."""
    obj._cached = True
    obj._cached_ttl = params.get("ttl")
    obj._cached_max_size = params.get("max_size")
    obj._cached_scope = params.get("scope", "global")


def _lazy_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @lazy is applied."""
    obj._lazy = True
    obj._lazy_init_on = params.get("init_on", "first_access")
    obj._lazy_thread_safe = params.get("thread_safe", True)
    obj._lazy_fallback = params.get("fallback")


def _batch_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @batch is applied."""
    obj._batch = True
    obj._batch_max_size = params.get("max_size", 64)
    obj._batch_flush_on = params.get("flush_on", "frame_end")
    obj._batch_timeout_ms = params.get("timeout_ms")
    obj._batch_coalesce = params.get("coalesce", False)


def _async_load_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @async_load is applied."""
    obj._async_load = True
    obj._async_load_priority = params.get("priority", 0)
    obj._async_load_timeout_ms = params.get("timeout_ms")
    obj._async_load_fallback = params.get("fallback")


def _diff_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @diff is applied."""
    obj._diff = True
    obj._diff_strategy = params.get("strategy", "shallow")
    obj._diff_include_fields = params.get("include_fields")
    obj._diff_exclude_fields = params.get("exclude_fields")


def _priority_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @priority is applied."""
    obj._priority = True
    obj._priority_value = params.get("value", 0)
    obj._priority_queue = params.get("queue", "default")
    obj._priority_higher_first = params.get("higher_first", True)


def _retry_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @retry is applied."""
    obj._retry = True
    obj._retry_max_attempts = params.get("max_attempts", 3)
    obj._retry_backoff = params.get("backoff", "exponential")
    obj._retry_base_delay_ms = params.get("base_delay_ms", 100)
    obj._retry_max_delay_ms = params.get("max_delay_ms", 10000)


def _throttle_network_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @throttle_network is applied."""
    obj._throttle_network = True
    obj._throttle_network_max_ups = params.get("max_updates_per_second", 20.0)
    obj._throttle_network_priority_decay = params.get("priority_decay", 0.9)
    obj._throttle_network_burst_allowance = params.get("burst_allowance", 3)
    obj._throttle_network_per = params.get("per", "entity")


def _observable_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @observable is applied."""
    obj._observable = True
    obj._observable_notify = params.get("notify", "sync")
    obj._observable_batch_delay_ms = params.get("batch_delay_ms", 16.0)


# ============================================================================
# Decorator creation
# ============================================================================

cached = make_decorator(
    name="cached",
    steps=_cached_steps,
    validate=_validate_cached_params,
    after_steps=_cached_after_apply,
)

lazy = make_decorator(
    name="lazy",
    steps=_lazy_steps,
    validate=_validate_lazy_params,
    after_steps=_lazy_after_apply,
)

batch = make_decorator(
    name="batch",
    steps=_batch_steps,
    validate=_validate_batch_params,
    after_steps=_batch_after_apply,
)

async_load = make_decorator(
    name="async_load",
    steps=_async_load_steps,
    validate=_validate_async_load_params,
    after_steps=_async_load_after_apply,
)

diff = make_decorator(
    name="diff",
    steps=_diff_steps,
    validate=_validate_diff_params,
    after_steps=_diff_after_apply,
)

priority = make_decorator(
    name="priority",
    steps=_priority_steps,
    validate=_validate_priority_params,
    after_steps=_priority_after_apply,
)

retry = make_decorator(
    name="retry",
    steps=_retry_steps,
    validate=_validate_retry_params,
    after_steps=_retry_after_apply,
)

throttle_network = make_decorator(
    name="throttle_network",
    steps=_throttle_network_steps,
    validate=_validate_throttle_network_params,
    after_steps=_throttle_network_after_apply,
)

observable = make_decorator(
    name="observable",
    steps=_observable_steps,
    validate=_validate_observable_params,
    after_steps=_observable_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("cached", cached, ("class", "function")),
    ("lazy", lazy, ("class", "function")),
    ("batch", batch, ("class", "function")),
    ("async_load", async_load, ("class", "function")),
    ("diff", diff, ("class",)),
    ("priority", priority, ("class",)),
    ("retry", retry, ("class", "function")),
    ("throttle_network", throttle_network, ("class",)),
    ("observable", observable, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.BRIDGES_CACHING,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.BRIDGES_CACHING].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "cached",
    "lazy",
    "batch",
    "async_load",
    "diff",
    "priority",
    "retry",
    "throttle_network",
    "observable",
    "VALID_CACHE_SCOPES",
    "VALID_LAZY_INIT_MODES",
    "VALID_FLUSH_MODES",
    "VALID_DIFF_STRATEGIES",
    "VALID_BACKOFF_STRATEGIES",
    "VALID_THROTTLE_SCOPES",
    "VALID_NOTIFY_MODES",
]
