"""Platform domain composite stacks."""
from __future__ import annotations
from trinity.decorators.stacks import Stack, parameterized_stack, stack


@parameterized_stack
def platform_adaptive(
    lod_levels: int = 3,
) -> Stack:
    """Platform-aware rendering with battery optimization and LOD."""
    from trinity.decorators.lod_streaming import lod, streamable
    from trinity.decorators.platform_specifics import battery_aware
    return stack(
        battery_aware(),
        lod(levels=lod_levels),
        streamable(priority="normal"),
    )


__all__ = ["platform_adaptive"]
