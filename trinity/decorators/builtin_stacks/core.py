"""Core built-in stacks: production_component, safe_system, saveable_data."""
from __future__ import annotations

from trinity.constants import DEFAULT_POOL_SIZE
from trinity.decorators.stacks import Stack, parameterized_stack, stack, _noop
from trinity.decorators.ecs_core import component
from trinity.decorators.memory import packed, pooled, budget
from trinity.decorators.debug_safety import track_changes
from trinity.decorators.ecs_core import system
from trinity.decorators.debug_safety import reads, writes
from trinity.decorators.data_flow import serializable, versioned

__all__ = ["production_component", "safe_system", "saveable_data"]


@parameterized_stack
def production_component(
    pool_size: int = DEFAULT_POOL_SIZE,
    layout: str = "soa",
    category: str = "gameplay",
) -> Stack:
    """Production-ready ECS component with pooling, packing, and budgeting."""
    return stack(
        track_changes,
        budget(category=category),
        pooled(initial_size=pool_size),
        packed(layout=layout),
        component,
    )


@parameterized_stack
def safe_system(
    phase: str = "update",
    read: tuple = (),
    write: tuple = (),
) -> Stack:
    """System with explicit access declarations."""
    return stack(
        system(phase=phase),
        reads(*read) if read else _noop,
        writes(*write) if write else _noop,
    )


@parameterized_stack
def saveable_data(
    version: int = 1,
    format: str = "binary",
    migrations: dict = None,
) -> Stack:
    """Persistent data with versioning."""
    return stack(
        track_changes,
        versioned(version=version, migrations=migrations or {}),
        serializable(format=format),
    )
