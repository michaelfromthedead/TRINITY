"""
Scheduling decorators — built from Ops.

Control when and how systems execute: timing, ordering,
parallelization, and conditional execution.

Every decorator here is a named list of Steps, created by make_decorator.

Decorators:
    @phase          - Define execution phases
    @parallel       - Multi-threaded execution
    @exclusive      - Exclusive world access
    @after          - Run after specified systems
    @before         - Run before specified systems
    @run_if         - Conditional execution
    @fixed          - Fixed timestep execution
    @job            - Job system task properties
    @async_system   - Async systems
    @throttle       - Spread work across frames
    @deferred       - Accumulate commands, execute later
    @chain          - Explicit pipeline
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Literal, Optional, TypeVar

from trinity.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MIN_BATCH,
    DEFAULT_PHYSICS_HZ,
    DEFAULT_STACK_SIZE,
)
from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _phase_steps(params: dict[str, Any]) -> list[Step]:
    name = params.get("name", "")
    return [
        Step(Op.TAG, {"key": "phase", "value": True}),
        Step(Op.TAG, {"key": "phase_name", "value": name}),
        Step(Op.TAG, {"key": "phase_after", "value": params.get("after", ())}),
        Step(Op.TAG, {"key": "phase_before", "value": params.get("before", ())}),
        Step(Op.REGISTER, {"registry": "scheduling"}),
    ]


def _parallel_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "parallel", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "parallel_chunk_size",
                "value": params.get("chunk_size", DEFAULT_CHUNK_SIZE),
            },
        ),
        Step(
            Op.TAG,
            {
                "key": "parallel_min_batch",
                "value": params.get("min_batch", DEFAULT_MIN_BATCH),
            },
        ),
        Step(Op.REGISTER, {"registry": "scheduling"}),
    ]


def _exclusive_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "exclusive", "value": True}),
        Step(Op.REGISTER, {"registry": "scheduling"}),
    ]


def _after_steps_builder(params: dict[str, Any]) -> list[Step]:
    systems = params.get("systems", ())
    names = tuple(s.__name__ if hasattr(s, "__name__") else str(s) for s in systems)
    return [
        Step(Op.TAG, {"key": "after", "value": systems}),
        Step(Op.TAG, {"key": "after_names", "value": names}),
        Step(Op.REGISTER, {"registry": "scheduling"}),
    ]


def _before_steps_builder(params: dict[str, Any]) -> list[Step]:
    systems = params.get("systems", ())
    names = tuple(s.__name__ if hasattr(s, "__name__") else str(s) for s in systems)
    return [
        Step(Op.TAG, {"key": "before", "value": systems}),
        Step(Op.TAG, {"key": "before_names", "value": names}),
        Step(Op.REGISTER, {"registry": "scheduling"}),
    ]


def _run_if_steps(params: dict[str, Any]) -> list[Step]:
    condition = params.get("condition")
    cond_name = getattr(condition, "__name__", str(condition)) if condition else ""
    return [
        Step(Op.TAG, {"key": "run_if", "value": condition}),
        Step(Op.TAG, {"key": "run_if_name", "value": cond_name}),
        Step(Op.REGISTER, {"registry": "scheduling"}),
    ]


def _fixed_steps(params: dict[str, Any]) -> list[Step]:
    hz = params.get("hz", DEFAULT_PHYSICS_HZ)
    return [
        Step(Op.TAG, {"key": "fixed", "value": True}),
        Step(Op.TAG, {"key": "fixed_hz", "value": hz}),
        Step(Op.TAG, {"key": "fixed_delta", "value": 1.0 / hz if hz else 0}),
        Step(Op.REGISTER, {"registry": "scheduling"}),
    ]


def _job_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "job", "value": True}),
        Step(Op.TAG, {"key": "job_priority", "value": params.get("priority", 0)}),
        Step(Op.TAG, {"key": "job_affinity", "value": params.get("affinity", "any")}),
        Step(
            Op.TAG,
            {
                "key": "job_stack_size",
                "value": params.get("stack_size", DEFAULT_STACK_SIZE),
            },
        ),
        Step(Op.REGISTER, {"registry": "scheduling"}),
    ]


def _async_system_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "async_system", "value": True}),
        Step(Op.REGISTER, {"registry": "scheduling"}),
    ]


def _throttle_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "throttle", "value": True}),
        Step(Op.TAG, {"key": "throttle_max_hz", "value": params.get("max_hz")}),
        Step(Op.TAG, {"key": "throttle_max_ms", "value": params.get("max_ms")}),
        Step(Op.REGISTER, {"registry": "scheduling"}),
    ]


def _deferred_steps(params: dict[str, Any]) -> list[Step]:
    return [
        Step(Op.TAG, {"key": "deferred", "value": True}),
        Step(Op.REGISTER, {"registry": "scheduling"}),
    ]


def _chain_steps(params: dict[str, Any]) -> list[Step]:
    systems = params.get("systems", ())
    names = tuple(s.__name__ if hasattr(s, "__name__") else str(s) for s in systems)
    return [
        Step(Op.TAG, {"key": "chain", "value": True}),
        Step(Op.TAG, {"key": "chain_systems", "value": systems}),
        Step(Op.TAG, {"key": "chain_names", "value": names}),
        Step(Op.REGISTER, {"registry": "scheduling"}),
    ]


# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_throttle(
    max_hz: Optional[float] = None,
    max_ms: Optional[float] = None,
    **_: Any,
) -> None:
    if max_hz is None and max_ms is None:
        raise ValueError("@throttle requires at least one of max_hz or max_ms")


# =============================================================================
# AFTER-STEPS (domain behavior)
# =============================================================================


def _after_phase(target: Any, params: dict[str, Any]) -> Any:
    target._phase = True
    target._phase_name = params.get("name", "")
    target._phase_after = params.get("after", ())
    target._phase_before = params.get("before", ())
    return None


def _after_parallel(target: Any, params: dict[str, Any]) -> Any:
    target._parallel = True
    target._parallel_chunk_size = params.get("chunk_size", DEFAULT_CHUNK_SIZE)
    target._parallel_min_batch = params.get("min_batch", DEFAULT_MIN_BATCH)
    return None


def _after_exclusive(target: Any, params: dict[str, Any]) -> Any:
    target._exclusive = True
    return None


def _after_after(target: Any, params: dict[str, Any]) -> Any:
    systems = params.get("systems", ())
    existing_after = getattr(target, "_after", ())
    existing_names = getattr(target, "_after_names", ())
    target._after = existing_after + systems
    target._after_names = existing_names + tuple(
        s.__name__ if hasattr(s, "__name__") else str(s) for s in systems
    )
    return None


def _after_before(target: Any, params: dict[str, Any]) -> Any:
    systems = params.get("systems", ())
    existing_before = getattr(target, "_before", ())
    existing_names = getattr(target, "_before_names", ())
    target._before = existing_before + systems
    target._before_names = existing_names + tuple(
        s.__name__ if hasattr(s, "__name__") else str(s) for s in systems
    )
    return None


def _after_run_if(target: Any, params: dict[str, Any]) -> Any:
    condition = params.get("condition")
    existing = getattr(target, "_run_if_conditions", [])
    existing.append(condition)
    target._run_if = condition
    target._run_if_conditions = existing
    target._run_if_name = getattr(condition, "__name__", str(condition))
    return None


def _after_fixed(target: Any, params: dict[str, Any]) -> Any:
    hz = params.get("hz", DEFAULT_PHYSICS_HZ)
    target._fixed = True
    target._fixed_hz = hz
    target._fixed_delta = 1.0 / hz
    return None


def _after_job(target: Any, params: dict[str, Any]) -> Any:
    target._job = True
    target._job_priority = params.get("priority", 0)
    target._job_affinity = params.get("affinity", "any")
    target._job_stack_size = params.get("stack_size", DEFAULT_STACK_SIZE)
    return None


def _after_async_system(target: Any, params: dict[str, Any]) -> Any:
    target._async_system = True
    target._is_coroutine = asyncio.iscoroutinefunction(target)
    return None


def _after_throttle(target: Any, params: dict[str, Any]) -> Any:
    target._throttle = True
    target._throttle_max_hz = params.get("max_hz")
    target._throttle_max_ms = params.get("max_ms")
    target._throttle_last_run = 0.0
    target._throttle_accumulated_work = None
    return None


def _after_deferred(target: Any, params: dict[str, Any]) -> Any:
    target._deferred = True
    return None


def _after_chain(target: Any, params: dict[str, Any]) -> Any:
    systems = params.get("systems", ())
    target._chain = True
    target._chain_systems = systems
    target._chain_names = tuple(
        s.__name__ if hasattr(s, "__name__") else str(s) for s in systems
    )
    # Mark each system in the chain
    for i, sys in enumerate(systems):
        sys._chain_member = True
        sys._chain_index = i
        sys._chain_class = target
        if i > 0:
            prev = systems[i - 1]
            existing = getattr(sys, "_after", ())
            if prev not in existing:
                sys._after = existing + (prev,)
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================


phase = make_decorator(
    name="phase",
    steps=_phase_steps,
    doc="Define an execution phase for system scheduling.",
    after_steps=_after_phase,
)

parallel = make_decorator(
    name="parallel",
    steps=_parallel_steps,
    doc="Enable multi-threaded parallel execution for a system.",
    after_steps=_after_parallel,
)

exclusive = make_decorator(
    name="exclusive",
    steps=_exclusive_steps,
    doc="System requires exclusive World access, cannot run in parallel.",
    after_steps=_after_exclusive,
)

after = make_decorator(
    name="after",
    steps=_after_steps_builder,
    doc="Specify systems that must run before this one.",
    after_steps=_after_after,
)

before = make_decorator(
    name="before",
    steps=_before_steps_builder,
    doc="Specify systems that must run after this one.",
    after_steps=_after_before,
)

run_if = make_decorator(
    name="run_if",
    steps=_run_if_steps,
    doc="Conditionally execute system based on runtime predicate.",
    after_steps=_after_run_if,
)

fixed = make_decorator(
    name="fixed",
    steps=_fixed_steps,
    doc="Execute system at a fixed timestep rate.",
    after_steps=_after_fixed,
)

job = make_decorator(
    name="job",
    steps=_job_steps,
    doc="Configure job system task properties for parallel work.",
    after_steps=_after_job,
)

async_system = make_decorator(
    name="async_system",
    steps=_async_system_steps,
    doc="Mark system as async (can await I/O operations).",
    after_steps=_after_async_system,
)

throttle = make_decorator(
    name="throttle",
    steps=_throttle_steps,
    doc="Throttle system execution by rate or time budget.",
    validate=_validate_throttle,
    after_steps=_after_throttle,
)

deferred = make_decorator(
    name="deferred",
    steps=_deferred_steps,
    doc="Defer structural changes until system completes.",
    after_steps=_after_deferred,
)

chain = make_decorator(
    name="chain",
    steps=_chain_steps,
    doc="Create an explicit system pipeline.",
    after_steps=_after_chain,
)


# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, dict[str, Any]]] = [
    ("phase", phase, {"foundation": False}),
    ("parallel", parallel, {"unique": True, "excludes": ("exclusive",)}),
    ("exclusive", exclusive, {"unique": True, "excludes": ("parallel",)}),
    ("after", after, {}),
    ("before", before, {}),
    ("run_if", run_if, {}),
    ("fixed", fixed, {"unique": True, "excludes": ("throttle",)}),
    ("job", job, {"unique": True}),
    ("async_system", async_system, {"unique": True}),
    ("throttle", throttle, {"unique": True, "excludes": ("fixed",)}),
    ("deferred", deferred, {"unique": True}),
    ("chain", chain, {"unique": True}),
]

for _name, _func, _extra in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.SCHEDULING,
            func=_func,
            unique=_extra.get("unique", False),
            foundation=_extra.get("foundation", False),
            excludes=_extra.get("excludes", ()),
            doc=getattr(_func, "__doc__", ""),
            target_types=("function", "class"),
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.SCHEDULING].append(_spec)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "phase",
    "parallel",
    "exclusive",
    "after",
    "before",
    "run_if",
    "fixed",
    "job",
    "async_system",
    "throttle",
    "deferred",
    "chain",
]
