"""Development built-in stacks: profiled_dev."""
from __future__ import annotations

from trinity.decorators.stacks import Stack, parameterized_stack, stack
from trinity.decorators.dev import profile, trace
from trinity.decorators.build_deploy import build_only

__all__ = ["profiled_dev"]


@parameterized_stack
def profiled_dev(
    name: str,
    warn_ms: float = 2.0,
) -> Stack:
    """Development-time instrumentation (stripped in release)."""
    return stack(
        profile(name=name, warn_ms=warn_ms),
        trace(level="debug"),
        build_only(configurations={"debug", "development"}),
    )
