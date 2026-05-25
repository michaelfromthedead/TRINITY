"""Network built-in stacks: networked_entity, bandwidth_efficient, predicted_entity, secure_multiplayer."""
from __future__ import annotations

from trinity.decorators.stacks import Stack, parameterized_stack, stack

from trinity.decorators.ecs_core import component
from trinity.decorators.memory import packed, pooled
from trinity.decorators.data_flow import networked, serializable, snapshot
from trinity.decorators.debug_safety import track_changes
from trinity.decorators.bridges_caching import diff, batch, throttle_network
from trinity.decorators.network_extended import interest, bandwidth_priority, server_reconcile
from trinity.decorators.security import server_authoritative, validated, rate_limited

__all__ = [
    "networked_entity",
    "bandwidth_efficient",
    "predicted_entity",
    "secure_multiplayer",
]


@parameterized_stack
def networked_entity(
    authority: str = "server",
    relevance: str = "spatial",
    priority: int = 10,
    pool_size: int = 64,
) -> Stack:
    """Basic network replication."""
    return stack(
        component,
        packed(layout="soa"),
        pooled(initial_size=pool_size),
        networked(authority=authority, relevance=relevance, priority=priority),
        serializable(format="binary"),
        track_changes,
    )


@parameterized_stack
def bandwidth_efficient(
    radius: int = 5000,
    max_updates_per_second: float = 20.0,
    priority: int = 50,
) -> Stack:
    """Bandwidth-optimized networking."""
    return stack(
        networked(relevance="spatial", delta=True),
        diff(strategy="structural"),
        interest(type="radius", radius=radius),
        bandwidth_priority(priority=priority),
        throttle_network(max_updates_per_second=max_updates_per_second),
        batch(flush_on="frame_end", coalesce=True),
    )


@parameterized_stack
def predicted_entity(
    history_frames: int = 30,
    max_reconcile_frames: int = 10,
    snap_threshold: float = 0.5,
) -> Stack:
    """Client-side prediction with reconciliation."""
    return stack(
        networked(authority="server", predicted=True, interpolated="hermite"),
        snapshot(history_frames=history_frames),
        server_reconcile(
            max_reconcile_frames=max_reconcile_frames,
            snap_threshold=snap_threshold,
        ),
        diff(strategy="shallow"),
    )


@parameterized_stack
def secure_multiplayer(
    rate_limit: int = 10,
) -> Stack:
    """Anti-cheat hardened."""
    return stack(
        server_authoritative,
        validated(rules=[]),
        rate_limited(max_per_second=rate_limit, per="player"),
    )
