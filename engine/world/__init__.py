"""
World Layer module.

Provides level architecture, world partition, terrain, foliage, environment systems,
and world queries for managing game world content.
"""

from engine.world.level import (
    Actor,
    Level,
    LevelBounds,
    LevelInstance,
    LevelLayer,
    LevelState,
    LevelType,
    StreamingLevel,
    SubLevel,
    WorldComposition,
)

# Re-export partition module
from engine.world import partition

__all__ = [
    # Level module
    "Actor",
    "Level",
    "LevelBounds",
    "LevelInstance",
    "LevelLayer",
    "LevelState",
    "LevelType",
    "StreamingLevel",
    "SubLevel",
    "WorldComposition",
    # Partition submodule
    "partition",
]
