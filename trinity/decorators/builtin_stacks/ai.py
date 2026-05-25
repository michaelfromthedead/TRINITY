"""AI built-in stacks: complete_ai."""
from __future__ import annotations

from trinity.decorators.stacks import Stack, parameterized_stack, stack
from trinity.decorators.game_ai import behavior_tree, blackboard, perception, ai_debug
from trinity.decorators.state_machine import state_machine
from trinity.decorators.dev import profile
from trinity.decorators.bridges_caching import cached
from trinity.decorators.data_flow import serializable
from trinity.decorators.debug_safety import track_changes

__all__ = ["complete_ai"]


@parameterized_stack
def complete_ai(
    behavior_tree_id: str,
    sense: str = "sight",
    sense_range: float = 50,
    sense_fov: float = 120,
    states: set = None,
    initial_state: str = "idle",
) -> Stack:
    """Full AI entity setup with behavior tree, perception, state machine, and utilities."""
    states = states or {"idle", "alert", "combat"}
    return stack(
        behavior_tree(id=behavior_tree_id),
        blackboard,
        perception(sense=sense, range=sense_range, fov=sense_fov),
        state_machine(initial=initial_state, states=states),
        ai_debug,
        profile(warn_ms=1.0),
        cached(ttl=0.5, scope="entity"),
        serializable(format="binary"),
        track_changes,
    )
