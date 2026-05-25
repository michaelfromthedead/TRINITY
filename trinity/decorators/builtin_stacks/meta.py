"""Meta-compositions that compose composite stacks into full game profiles."""

from __future__ import annotations

from trinity.decorators.stacks import Stack, parameterized_stack, stack

__all__ = [
    "production_multiplayer_game",
    "open_world_mmo",
    "competitive_esports",
    "moddable_singleplayer",
    "mobile_optimized",
]


@parameterized_stack
def production_multiplayer_game(
    pool_size: int = 64,
    history_frames: int = 30,
    max_retry: int = 5,
    cache_ttl: float = 300.0,
) -> Stack:
    """Full production multiplayer game stack."""
    from trinity.decorators.builtin_stacks.composite import (
        multiplayer_character, optimized_network_sync,
        saveable_game_state, resilient_cloud_service,
    )
    return (
        multiplayer_character(pool_size=pool_size, history_frames=history_frames)
        + optimized_network_sync()
        + saveable_game_state()
        + resilient_cloud_service(max_attempts=max_retry, cache_ttl=cache_ttl)
    )


@parameterized_stack
def open_world_mmo(
    pool_size: int = 10000,
    chunk_size: tuple = (100, 100, 100),
    cache_size: int = 500,
) -> Stack:
    """Open world MMO with streaming, networking, and persistence."""
    from trinity.decorators.builtin_stacks.composite import (
        open_world_entity, streaming_asset_loader,
        optimized_network_sync, smart_query_cache,
        saveable_game_state,
    )
    return (
        open_world_entity(pool_size=pool_size, chunk_size=chunk_size)
        + streaming_asset_loader(cache_size=cache_size)
        + optimized_network_sync()
        + smart_query_cache()
        + saveable_game_state()
    )


@parameterized_stack
def competitive_esports(
    pool_size: int = 128,
    history_frames: int = 600,
) -> Stack:
    """Competitive esports with determinism, replay, and events."""
    from trinity.decorators.builtin_stacks.composite import (
        competitive_entity, multiplayer_character,
        observable_game_event, smart_query_cache,
    )
    return (
        competitive_entity(pool_size=pool_size, history_frames=history_frames)
        + multiplayer_character()
        + observable_game_event()
        + smart_query_cache()
    )


@parameterized_stack
def moddable_singleplayer(
    pool_size: int = 10000,
    chunk_size: tuple = (100, 100, 100),
    namespace: str = "default",
) -> Stack:
    """Moddable singleplayer with streaming, saving, and UI."""
    from trinity.decorators.builtin_stacks.composite import (
        open_world_entity, streaming_asset_loader,
        saveable_game_state, moddable_content,
        reactive_ui_component,
    )
    return (
        open_world_entity(pool_size=pool_size, chunk_size=chunk_size)
        + streaming_asset_loader()
        + saveable_game_state()
        + moddable_content(namespace=namespace)
        + reactive_ui_component()
    )


@parameterized_stack
def mobile_optimized(
    pool_size: int = 512,
    cache_ttl: float = 60.0,
    batch_delay_ms: float = 33.0,
) -> Stack:
    """Mobile-optimized with strict budgets, streaming, and cloud."""
    from trinity.decorators.builtin_stacks.composite import (
        streaming_asset_loader, resilient_cloud_service,
        reactive_ui_component,
    )
    from trinity.decorators.builtin_stacks.core import production_component
    from trinity.decorators.platform_specifics import battery_aware
    return (
        production_component(pool_size=pool_size)
        + streaming_asset_loader()
        + resilient_cloud_service(cache_ttl=cache_ttl)
        + reactive_ui_component(batch_delay_ms=batch_delay_ms)
        + stack(battery_aware)
    )
