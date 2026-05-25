"""Streaming built-in stacks: streaming_chunk, lod_scalable."""
from __future__ import annotations

from trinity.decorators.stacks import Stack, parameterized_stack, stack


@parameterized_stack
def streaming_chunk(
    chunk_size: tuple = (100, 100, 100),
    overlap: int = 10,
    min_age: float = 60.0,
) -> Stack:
    """Open world chunk streaming."""
    from trinity.decorators.lod_streaming import chunk, streamable, loading_priority, unloadable
    from trinity.decorators.data_flow import serializable
    from trinity.decorators.debug_safety import track_changes
    from trinity.decorators.bridges_caching import async_load, lazy

    return stack(
        chunk(size=chunk_size, overlap=overlap),
        streamable(priority="normal"),
        loading_priority(visibility_weight=3.0, player_velocity_weight=1.5),
        unloadable(min_age=min_age, save_state=True),
        serializable(format="binary"),
        track_changes,
        async_load(priority=0, fallback=None),
        lazy(init_on="first_access"),
    )


@parameterized_stack
def lod_scalable(
    levels: int = 4,
    distances: list = None,
) -> Stack:
    """Scalable quality rendering."""
    from trinity.decorators.lod_streaming import lod, streamable
    from trinity.decorators.assets import residency

    distances = distances or [10, 50, 200, 1000]

    return stack(
        lod(levels=levels, distances=distances),
        streamable(priority="normal"),
        residency(priority="normal", min_mip=2),
    )


__all__ = [
    "streaming_chunk",
    "lod_scalable",
]
