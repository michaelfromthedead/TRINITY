"""DEV (Tier 6) decorators for development tools.

Provides decorators for profiling, tracing, debugging, testing, and deprecation.
All decorators use the Ops-based system via make_decorator().
"""

from __future__ import annotations

import functools
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

__all__ = [
    "profile",
    "gpu_profile",
    "trace",
    "reloadable",
    "editor",
    "test",
    "bench",
    "invariant",
    "deprecated",
    "ProfileConfig",
    "ReloadableConfig",
]

# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class ProfileConfig:
    """Configuration for @profile decorator."""

    name: Optional[str] = None
    gpu: bool = False
    warn_ms: Optional[float] = None
    track_allocations: bool = False


@dataclass
class ReloadableConfig:
    """Configuration for @reloadable decorator."""

    enabled: bool = True
    preserve: list = field(default_factory=list)
    reinitialize: list = field(default_factory=list)
    validate: Optional[Callable] = None


# ============================================================================
# @profile - Performance profiling
# ============================================================================


def _profile_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @profile decorator."""
    return [
        Step(
            Op.TAG,
            {
                "key": "profiled",
                "value": {
                    "name": params.get("name"),
                    "warn_ms": params.get("warn_ms"),
                    "track_allocations": params.get("track_allocations", False),
                },
            },
        ),
        Step(Op.HOOK, {"event": "before_call"}),
        Step(Op.HOOK, {"event": "after_call"}),
        Step(Op.REGISTER, {"registry": "dev"}),
    ]


def _after_profile(target: Any, params: dict[str, Any]) -> Any:
    """After-steps handler for @profile decorator."""
    # For classes, just set attributes
    if isinstance(target, type):
        target._profiled = True
        target._profile_name = params.get("name") or getattr(
            target, "__name__", "unknown"
        )
        return None

    # For functions, wrap with timing logic
    if not callable(target):
        target._profiled = True
        target._profile_name = params.get("name") or getattr(
            target, "__name__", "unknown"
        )
        return None

    name = params.get("name") or getattr(target, "__name__", "unknown")
    stats = {
        "call_count": 0,
        "total_ms": 0.0,
        "min_ms": float("inf"),
        "max_ms": 0.0,
    }

    @functools.wraps(target)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = target(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000

        stats["call_count"] += 1
        stats["total_ms"] += elapsed
        if elapsed < stats["min_ms"]:
            stats["min_ms"] = elapsed
        if elapsed > stats["max_ms"]:
            stats["max_ms"] = elapsed

        warn_ms = params.get("warn_ms")
        if warn_ms is not None and elapsed > warn_ms:
            warnings.warn(f"{name} took {elapsed:.2f}ms (threshold: {warn_ms}ms)")

        return result

    def profile_stats() -> dict[str, Any]:
        """Return profiling statistics."""
        cc = stats["call_count"]
        return {
            "name": name,
            "call_count": cc,
            "total_ms": stats["total_ms"],
            "min_ms": stats["min_ms"] if cc > 0 else 0.0,
            "max_ms": stats["max_ms"],
            "avg_ms": stats["total_ms"] / cc if cc > 0 else 0.0,
        }

    def profile_reset() -> None:
        """Reset profiling statistics."""
        stats["call_count"] = 0
        stats["total_ms"] = 0.0
        stats["min_ms"] = float("inf")
        stats["max_ms"] = 0.0

    # Copy decorator metadata from target
    for attr in ("_tags", "_applied_steps", "_applied_decorators", "_registries"):
        val = getattr(target, attr, None)
        if val is not None:
            setattr(wrapper, attr, val)

    wrapper._profiled = True
    wrapper._profile_name = name
    wrapper.profile_stats = profile_stats
    wrapper.profile_reset = profile_reset

    return wrapper


profile = make_decorator(
    name="profile",
    steps=_profile_steps,
    doc="Mark function/class for performance profiling.",
    after_steps=_after_profile,
)


# ============================================================================
# @gpu_profile - GPU performance profiling
# ============================================================================


def _gpu_profile_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @gpu_profile decorator."""
    return [
        Step(
            Op.TAG,
            {
                "key": "gpu_profiled",
                "value": {
                    "category": params.get("category"),
                    "include_memory": params.get("include_memory", False),
                },
            },
        ),
        Step(Op.HOOK, {"event": "before_call"}),
        Step(Op.HOOK, {"event": "after_call"}),
        Step(Op.REGISTER, {"registry": "dev"}),
    ]


def _validate_gpu_profile(**kwargs) -> None:
    """Validate @gpu_profile parameters."""
    category = kwargs.get("category")
    if not category or not isinstance(category, str):
        raise ValueError("category must be a non-empty string")


def _after_gpu_profile(target: Any, params: dict[str, Any]) -> Any:
    """After-steps handler for @gpu_profile decorator."""
    target._gpu_profiled = True
    target._gpu_profile_category = params.get("category")
    target._gpu_profile_include_memory = params.get("include_memory", False)

    # Add stub method for GPU stats
    def gpu_stats() -> dict[str, Any]:
        """Return GPU profiling statistics (stub)."""
        return {
            "category": params.get("category"),
            "include_memory": params.get("include_memory", False),
        }

    target.gpu_stats = gpu_stats
    return None


gpu_profile = make_decorator(
    name="gpu_profile",
    steps=_gpu_profile_steps,
    doc="Mark function/class for GPU performance profiling.",
    validate=_validate_gpu_profile,
    after_steps=_after_gpu_profile,
)


# ============================================================================
# @trace - Execution tracing
# ============================================================================


def _trace_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @trace decorator."""
    return [
        Step(Op.TAG, {"key": "traced", "value": {"level": params.get("level", "debug")}}),
        Step(Op.HOOK, {"event": "before_call"}),
        Step(Op.HOOK, {"event": "after_call"}),
        Step(Op.REGISTER, {"registry": "dev"}),
    ]


def _validate_trace(**kwargs) -> None:
    """Validate @trace parameters."""
    level = kwargs.get("level", "debug")
    valid_levels = {"debug", "info", "warn"}
    if level not in valid_levels:
        raise ValueError(f"level must be one of {valid_levels}, got {level}")


def _after_trace(target: Any, params: dict[str, Any]) -> Any:
    """After-steps handler for @trace decorator."""
    if not callable(target) or isinstance(target, type):
        target._traced = True
        target._trace_level = params.get("level", "debug")
        return None

    level = params.get("level", "debug")
    logger = logging.getLogger("trinity.trace")

    # Map level strings to logging levels
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
    }
    log_level = level_map.get(level, logging.DEBUG)

    @functools.wraps(target)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.log(log_level, f"ENTER {target.__name__}")
        try:
            result = target(*args, **kwargs)
            logger.log(log_level, f"EXIT {target.__name__}")
            return result
        except Exception as e:
            logger.log(log_level, f"ERROR {target.__name__}: {e}")
            raise

    # Copy decorator metadata from target
    for attr in ("_tags", "_applied_steps", "_applied_decorators", "_registries"):
        val = getattr(target, attr, None)
        if val is not None:
            setattr(wrapper, attr, val)

    wrapper._traced = True
    wrapper._trace_level = level

    return wrapper


trace = make_decorator(
    name="trace",
    steps=_trace_steps,
    doc="Enable execution tracing for function/class.",
    validate=_validate_trace,
    after_steps=_after_trace,
)


# ============================================================================
# @reloadable - Hot reload support
# ============================================================================


def _reloadable_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @reloadable decorator."""
    return [
        Step(
            Op.TAG,
            {
                "key": "reloadable",
                "value": {
                    "enabled": params.get("enabled", True),
                    "preserve": params.get("preserve", []),
                    "reinitialize": params.get("reinitialize", []),
                },
            },
        ),
        Step(Op.REGISTER, {"registry": "dev"}),
    ]


def _after_reloadable(target: Any, params: dict[str, Any]) -> Any:
    """After-steps handler for @reloadable decorator."""
    target._reloadable = True
    target._reload_preserve = list(params.get("preserve", []))
    target._reload_reinitialize = list(params.get("reinitialize", []))
    target._reload_validate = params.get("validate")
    return None


reloadable = make_decorator(
    name="reloadable",
    steps=_reloadable_steps,
    doc="Enable hot reload support for class/function.",
    after_steps=_after_reloadable,
)


# ============================================================================
# @editor - Editor integration
# ============================================================================


def _editor_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @editor decorator."""
    return [
        Step(
            Op.TAG,
            {
                "key": "editor",
                "value": {
                    "category": params.get("category", "General"),
                    "hidden": params.get("hidden", False),
                },
            },
        ),
        Step(Op.REGISTER, {"registry": "dev"}),
    ]


def _after_editor(target: Any, params: dict[str, Any]) -> Any:
    """After-steps handler for @editor decorator."""
    target._editor = True
    target._editor_category = params.get("category", "General")
    target._editor_hidden = params.get("hidden", False)
    return None


editor = make_decorator(
    name="editor",
    steps=_editor_steps,
    doc="Mark for editor integration.",
    after_steps=_after_editor,
)


# ============================================================================
# @test - Test configuration
# ============================================================================


def _test_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @test decorator."""
    return [
        Step(
            Op.TAG,
            {
                "key": "test",
                "value": {
                    "cases": params.get("cases", []),
                    "fuzz": params.get("fuzz", False),
                    "property_based": params.get("property_based", False),
                },
            },
        ),
        Step(Op.REGISTER, {"registry": "dev"}),
    ]


def _after_test(target: Any, params: dict[str, Any]) -> Any:
    """After-steps handler for @test decorator."""
    target._test = True
    target._test_cases = list(params.get("cases", []))
    target._test_fuzz = params.get("fuzz", False)
    target._test_property_based = params.get("property_based", False)
    return None


test = make_decorator(
    name="test",
    steps=_test_steps,
    doc="Configure test cases for function/class.",
    after_steps=_after_test,
)


# ============================================================================
# @bench - Benchmarking configuration
# ============================================================================


def _bench_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @bench decorator."""
    return [
        Step(
            Op.TAG,
            {
                "key": "bench",
                "value": {
                    "iterations": params.get("iterations", 1000),
                    "warmup": params.get("warmup", 100),
                },
            },
        ),
        Step(Op.REGISTER, {"registry": "dev"}),
    ]


def _validate_bench(**kwargs) -> None:
    """Validate @bench parameters."""
    iterations = kwargs.get("iterations", 1000)
    warmup = kwargs.get("warmup", 100)

    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")


def _after_bench(target: Any, params: dict[str, Any]) -> Any:
    """After-steps handler for @bench decorator."""
    target._bench = True
    target._bench_iterations = params.get("iterations", 1000)
    target._bench_warmup = params.get("warmup", 100)
    return None


bench = make_decorator(
    name="bench",
    steps=_bench_steps,
    doc="Configure benchmarking for function/class.",
    validate=_validate_bench,
    after_steps=_after_bench,
)


# ============================================================================
# @invariant - Runtime invariants
# ============================================================================


def _invariant_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @invariant decorator."""
    return [
        Step(
            Op.TAG,
            {
                "key": "invariant",
                "value": {
                    "check": params.get("check"),
                    "when": params.get("when", "debug"),
                },
            },
        ),
        Step(Op.VALIDATE, {"constraint": "invariant_check"}),
        Step(Op.REGISTER, {"registry": "dev"}),
    ]


def _validate_invariant(**kwargs) -> None:
    """Validate @invariant parameters."""
    check = kwargs.get("check")
    if not callable(check):
        raise ValueError("check must be a callable")

    when = kwargs.get("when", "debug")
    valid_when = {"debug", "always"}
    if when not in valid_when:
        raise ValueError(f"when must be one of {valid_when}, got {when}")


def _after_invariant(target: Any, params: dict[str, Any]) -> Any:
    """After-steps handler for @invariant decorator."""
    # Accumulate invariants in a list
    if not hasattr(target, "_invariants"):
        target._invariants = []

    target._invariants.append(
        {
            "check": params.get("check"),
            "when": params.get("when", "debug"),
        }
    )
    return None


invariant = make_decorator(
    name="invariant",
    steps=_invariant_steps,
    doc="Add runtime invariant checks to class/function.",
    validate=_validate_invariant,
    after_steps=_after_invariant,
)


# ============================================================================
# @deprecated - Deprecation warnings
# ============================================================================


def _deprecated_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @deprecated decorator."""
    return [
        Step(
            Op.TAG,
            {
                "key": "deprecated",
                "value": {
                    "since": params.get("since"),
                    "replacement": params.get("replacement"),
                    "remove_in": params.get("remove_in"),
                },
            },
        ),
        Step(Op.HOOK, {"event": "before_call"}),
        Step(Op.REGISTER, {"registry": "dev"}),
    ]


def _validate_deprecated(**kwargs) -> None:
    """Validate @deprecated parameters."""
    since = kwargs.get("since")
    if not since or not isinstance(since, str):
        raise ValueError("since must be a non-empty string")


def _after_deprecated(target: Any, params: dict[str, Any]) -> Any:
    """After-steps handler for @deprecated decorator."""
    # For classes, just set attributes
    if isinstance(target, type):
        target._deprecated = True
        target._deprecated_since = params.get("since")
        target._deprecated_replacement = params.get("replacement")
        target._deprecated_remove_in = params.get("remove_in")
        return None

    # For functions, wrap to emit warning
    if not callable(target):
        target._deprecated = True
        target._deprecated_since = params.get("since")
        target._deprecated_replacement = params.get("replacement")
        target._deprecated_remove_in = params.get("remove_in")
        return None

    since = params.get("since")
    replacement = params.get("replacement")
    remove_in = params.get("remove_in")

    # Build deprecation message
    msg = f"{target.__name__} is deprecated since {since}."
    if replacement:
        msg += f" Use {replacement} instead."
    if remove_in:
        msg += f" Will be removed in {remove_in}."

    @functools.wraps(target)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return target(*args, **kwargs)

    # Copy decorator metadata from target
    for attr in ("_tags", "_applied_steps", "_applied_decorators", "_registries"):
        val = getattr(target, attr, None)
        if val is not None:
            setattr(wrapper, attr, val)

    wrapper._deprecated = True
    wrapper._deprecated_since = since
    wrapper._deprecated_replacement = replacement
    wrapper._deprecated_remove_in = remove_in

    return wrapper


deprecated = make_decorator(
    name="deprecated",
    steps=_deprecated_steps,
    doc="Mark function/class as deprecated.",
    validate=_validate_deprecated,
    after_steps=_after_deprecated,
)


# ============================================================================
# Registry Registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("profile", profile, ("function", "class")),
    ("gpu_profile", gpu_profile, ("function", "class")),
    ("trace", trace, ("function", "class")),
    ("reloadable", reloadable, ("class", "function")),
    ("editor", editor, ("any",)),
    ("test", test, ("function", "class")),
    ("bench", bench, ("function", "class")),
    ("invariant", invariant, ("class", "function")),
    ("deprecated", deprecated, ("function", "class")),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.DEV,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.DEV].append(_spec)
