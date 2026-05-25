"""Composite stacks that combine Phase B built-in stacks together."""

from __future__ import annotations

from typing import Tuple

from trinity.decorators.stacks import Stack, parameterized_stack, stack

from trinity.decorators.builtin_stacks.core import production_component
from trinity.decorators.builtin_stacks.network import (
    bandwidth_efficient,
    predicted_entity,
    secure_multiplayer,
)
from trinity.decorators.builtin_stacks.persistence import (
    deterministic_data,
    replay_ready,
    versioned_saveable,
)
from trinity.decorators.builtin_stacks.streaming import lod_scalable, streaming_chunk

__all__ = [
    "multiplayer_character",
    "competitive_entity",
    "open_world_entity",
    "mmo_entity",
    "moddable_content",
    "resilient_cloud_service",
    "reactive_ui_component",
    "smart_query_cache",
    "streaming_asset_loader",
    "optimized_network_sync",
    "saveable_game_state",
    "observable_game_event",
]


@parameterized_stack
def multiplayer_character(
    pool_size: int = 64, history_frames: int = 30, version: int = 1
) -> Stack:
    """Full multiplayer character with prediction, saving, and security."""
    return (
        production_component(pool_size=pool_size)
        + predicted_entity(history_frames=history_frames)
        + versioned_saveable(version=version)
        + secure_multiplayer()
    )


@parameterized_stack
def competitive_entity(
    pool_size: int = 128, history_frames: int = 600
) -> Stack:
    """Competitive game entity with determinism, replay, and anti-cheat."""
    return (
        production_component(pool_size=pool_size)
        + deterministic_data()
        + replay_ready(history_frames=history_frames)
        + predicted_entity(history_frames=history_frames)
        + secure_multiplayer()
    )


@parameterized_stack
def open_world_entity(
    pool_size: int = 10000,
    chunk_size: Tuple[int, int, int] = (100, 100, 100),
) -> Stack:
    """Open world entity with streaming, LOD, and persistence."""
    return (
        production_component(pool_size=pool_size)
        + streaming_chunk(chunk_size=chunk_size)
        + lod_scalable()
        + versioned_saveable()
    )


@parameterized_stack
def mmo_entity(
    pool_size: int = 5000, relevance_radius: int = 10000
) -> Stack:
    """MMO entity with bandwidth optimization and security."""
    return (
        production_component(pool_size=pool_size)
        + bandwidth_efficient(radius=relevance_radius)
        + secure_multiplayer()
        + versioned_saveable()
    )


@parameterized_stack
def moddable_content(namespace: str, version: int = 1) -> Stack:
    """Moddable content entity with serialization and change tracking."""
    from trinity.decorators.bridges_caching import observable
    from trinity.decorators.data_flow import serializable, versioned
    from trinity.decorators.debug_safety import track_changes
    from trinity.decorators.ecs_core import component
    from trinity.decorators.modding import moddable

    return stack(
        component,
        moddable(namespace=namespace),
        serializable(format="json"),
        versioned(version=version),
        track_changes,
        observable(),
    )


@parameterized_stack
def resilient_cloud_service(
    max_attempts: int = 5,
    cache_ttl: float = 300.0,
    timeout_ms: int = 10000,
    pool_size: int = 32,
) -> Stack:
    """Cloud service with retry, caching, and async loading."""
    from trinity.decorators.builtin_stacks.core import production_component
    from trinity.decorators.bridges_caching import retry, async_load, cached

    return production_component(pool_size=pool_size) + stack(
        retry(max_attempts=max_attempts, backoff="exponential"),
        async_load(timeout_ms=timeout_ms),
        cached(ttl=cache_ttl, scope="global"),
    )


@parameterized_stack
def reactive_ui_component(
    pool_size: int = 256,
    batch_delay_ms: float = 16.0,
) -> Stack:
    """Reactive UI data with observable changes and lazy init."""
    from trinity.decorators.builtin_stacks.core import production_component
    from trinity.decorators.bridges_caching import observable, lazy, diff
    from trinity.decorators.data_flow import serializable

    return production_component(pool_size=pool_size) + stack(
        observable(notify="batched", batch_delay_ms=batch_delay_ms),
        diff(strategy="shallow"),
        lazy(init_on="first_access"),
        serializable(format="json"),
    )


@parameterized_stack
def smart_query_cache(
    ttl: float = 2.0,
    max_size: int = 10000,
    batch_size: int = 64,
) -> Stack:
    """Cached query with batching and priority."""
    from trinity.decorators.bridges_caching import cached, batch, priority
    from trinity.decorators.builtin_stacks.development import profiled_dev

    return stack(
        cached(ttl=ttl, max_size=max_size, scope="global"),
        batch(max_size=batch_size, flush_on="frame_end", coalesce=True),
        priority(queue="queries"),
    ) + profiled_dev(name="query_cache")


@parameterized_stack
def streaming_asset_loader(
    fallback: str = None,
    timeout_ms: int = 5000,
    cache_size: int = 500,
) -> Stack:
    """Asset loader with streaming, async loading, and caching."""
    from trinity.decorators.lod_streaming import streamable, loading_priority, unloadable
    from trinity.decorators.bridges_caching import async_load, lazy, priority, batch, cached

    return stack(
        streamable(priority="normal"),
        loading_priority(visibility_weight=2.0, player_velocity_weight=1.0),
        unloadable(min_age=30.0, save_state=False),
        async_load(timeout_ms=timeout_ms, fallback=fallback),
        lazy(init_on="first_frame"),
        priority(queue="assets"),
        batch(max_size=32, flush_on="frame_end"),
        cached(max_size=cache_size, scope="global"),
    )


@parameterized_stack
def optimized_network_sync(
    max_updates_per_second: float = 30.0,
    batch_size: int = 128,
) -> Stack:
    """Network sync with throttling, diffing, and batched updates."""
    from trinity.decorators.bridges_caching import throttle_network, diff, batch

    return stack(
        throttle_network(max_updates_per_second=max_updates_per_second),
        diff(strategy="shallow"),
        batch(max_size=batch_size, flush_on="frame_end"),
    )


@parameterized_stack
def saveable_game_state(
    version: int = 1,
    max_retry: int = 3,
) -> Stack:
    """Persistent game state with encryption, retry, and change tracking."""
    from trinity.decorators.security import encrypted
    from trinity.decorators.bridges_caching import retry, diff
    from trinity.decorators.data_flow import serializable, versioned

    return stack(
        serializable(format="binary"),
        versioned(version=version),
        encrypted(),
        retry(max_attempts=max_retry, backoff="exponential"),
        diff(strategy="deep"),
    )


@parameterized_stack
def observable_game_event(
    priority_value: int = 10,
    batch_size: int = 256,
) -> Stack:
    """Observable game event with priority and batching."""
    from trinity.decorators.ecs_core import event
    from trinity.decorators.bridges_caching import observable, priority, batch

    return stack(
        event(),
        observable(notify="batched"),
        priority(value=priority_value, queue="events"),
        batch(max_size=batch_size, flush_on="frame_end"),
    )
