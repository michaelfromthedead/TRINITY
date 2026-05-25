"""
Trinity Pattern - Tier 51: DEBUG_EXTENDED Decorators

Network debugging and automated testing decorators.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Configuration dataclasses
# ============================================================================


@dataclass(frozen=True)
class NetworkDebugConfig:
    """Network debug configuration."""

    log_packets: bool
    simulate_latency: float
    simulate_loss: float


@dataclass(frozen=True)
class AutomationTestConfig:
    """Automation test configuration."""

    category: str
    timeout_seconds: float
    required_features: frozenset[str]


# ============================================================================
# Step builders
# ============================================================================


def _network_debug_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @network_debug decorator."""
    log_packets = params.get("log_packets", False)
    simulate_latency = params.get("simulate_latency", 0.0)
    simulate_loss = params.get("simulate_loss", 0.0)

    return [
        Step(Op.TAG, {"key": "network_debug", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "network_debug_config",
                "value": NetworkDebugConfig(
                    log_packets=log_packets,
                    simulate_latency=simulate_latency,
                    simulate_loss=simulate_loss,
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "debug_extended"}),
    ]


def _automation_test_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @automation_test decorator."""
    category = params.get("category", "")
    timeout_seconds = params.get("timeout_seconds", 30.0)
    required_features = params.get("required_features", set())

    return [
        Step(Op.TAG, {"key": "automation_test", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "automation_test_config",
                "value": AutomationTestConfig(
                    category=category,
                    timeout_seconds=timeout_seconds,
                    required_features=frozenset(required_features),
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "debug_extended"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_network_debug_params(**kwargs: Any) -> None:
    """Validate @network_debug parameters."""
    simulate_latency = kwargs.get("simulate_latency", 0.0)
    if not isinstance(simulate_latency, (int, float)) or simulate_latency < 0:
        raise ValueError(f"simulate_latency must be >= 0, got {simulate_latency}")

    simulate_loss = kwargs.get("simulate_loss", 0.0)
    if not isinstance(simulate_loss, (int, float)) or not (0 <= simulate_loss <= 1):
        raise ValueError(f"simulate_loss must be between 0 and 1, got {simulate_loss}")


def _validate_automation_test_params(**kwargs: Any) -> None:
    """Validate @automation_test parameters."""
    category = kwargs.get("category", "")
    if not category:
        raise ValueError("category must be a non-empty string")

    timeout_seconds = kwargs.get("timeout_seconds", 30.0)
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")


# ============================================================================
# After-apply functions
# ============================================================================


def _network_debug_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @network_debug is applied."""
    log_packets = params.get("log_packets", False)
    simulate_latency = params.get("simulate_latency", 0.0)
    simulate_loss = params.get("simulate_loss", 0.0)

    obj._network_debug = True
    obj._network_debug_log_packets = log_packets
    obj._network_debug_simulate_latency = simulate_latency
    obj._network_debug_simulate_loss = simulate_loss
    obj._network_debug_config = NetworkDebugConfig(
        log_packets=log_packets,
        simulate_latency=simulate_latency,
        simulate_loss=simulate_loss,
    )


def _automation_test_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @automation_test is applied."""
    category = params.get("category", "")
    timeout_seconds = params.get("timeout_seconds", 30.0)
    required_features = params.get("required_features", set())

    obj._automation_test = True
    obj._automation_test_category = category
    obj._automation_test_timeout_seconds = timeout_seconds
    obj._automation_test_required_features = frozenset(required_features)
    obj._automation_test_config = AutomationTestConfig(
        category=category,
        timeout_seconds=timeout_seconds,
        required_features=frozenset(required_features),
    )


# ============================================================================
# Decorator creation
# ============================================================================

network_debug = make_decorator(
    name="network_debug",
    steps=_network_debug_steps,
    validate=_validate_network_debug_params,
    after_steps=_network_debug_after_apply,
)

automation_test = make_decorator(
    name="automation_test",
    steps=_automation_test_steps,
    validate=_validate_automation_test_params,
    after_steps=_automation_test_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("network_debug", network_debug, ("class",)),
    ("automation_test", automation_test, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.DEBUG_EXTENDED,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.DEBUG_EXTENDED].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "network_debug",
    "automation_test",
    "NetworkDebugConfig",
    "AutomationTestConfig",
]
