"""Persistence built-in stacks: versioned_saveable, replay_ready, deterministic_data."""
from __future__ import annotations

from trinity.decorators.stacks import Stack, parameterized_stack, stack
from trinity.decorators.data_flow import serializable, versioned, snapshot
from trinity.decorators.debug_safety import track_changes
from trinity.decorators.replay import recorded, replay_authority, keyframe
from trinity.decorators.bridges_caching import diff
from trinity.decorators.ecs_core import component
from trinity.decorators.time import deterministic

__all__ = [
    "versioned_saveable",
    "replay_ready",
    "deterministic_data",
]


@parameterized_stack
def versioned_saveable(version: int = 1, migrations: dict = None) -> Stack:
    """Saveable data with version migration."""
    return stack(
        serializable(format="binary"),
        versioned(version=version, migrations=migrations or {}),
        track_changes,
    )


@parameterized_stack
def replay_ready(history_frames: int = 600, keyframe_interval: float = 5.0) -> Stack:
    """Component ready for replay recording."""
    return stack(
        recorded(frequency="fixed_tick"),
        replay_authority(source="recording"),
        serializable(format="binary"),
        track_changes,
        snapshot(history_frames=history_frames),
        keyframe(interval=keyframe_interval),
        diff(strategy="structural"),
    )


@parameterized_stack
def deterministic_data() -> Stack:
    """Data for deterministic simulation."""
    return stack(
        component,
        deterministic,
        serializable(format="binary"),
        track_changes,
    )
