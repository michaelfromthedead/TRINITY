"""Modding domain composite stacks."""
from __future__ import annotations
from trinity.decorators.stacks import Stack, parameterized_stack, stack


@parameterized_stack
def mod_friendly(
    namespace: str = "default",
) -> Stack:
    """Mod-friendly entity with observable changes and JSON serialization."""
    from trinity.decorators.bridges_caching import observable
    from trinity.decorators.data_flow import serializable
    from trinity.decorators.modding import moddable
    return stack(
        moddable(namespace=namespace),
        observable(notify="sync"),
        serializable(format="json"),
    )


__all__ = ["mod_friendly"]
