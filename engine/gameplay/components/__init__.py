"""
Gameplay Components Package.

This package provides core gameplay components for the game engine.
"""

from engine.gameplay.components.constants import (
    HealthConstants,
    MovementConstants,
    TransformConstants,
    TeamConstants,
    StatsConstants,
    StateMachineConstants,
)
from engine.gameplay.components.health import (
    HealthComponent,
    DamageType,
    HealthState,
    DamageEvent,
    HealEvent,
)
from engine.gameplay.components.movement import (
    MovementComponent,
    MovementMode,
    MovementState,
    MovementSettings,
    MovementSnapshot,
)
from engine.gameplay.components.transform import (
    TransformComponent,
    TransformSnapshot,
    TransformSpace,
)
from engine.gameplay.components.team import (
    TeamComponent,
    TeamRelation,
    IFFResponse,
    Team,
    Faction,
    TeamRegistry,
    get_team_registry,
    set_team_registry,
)
from engine.gameplay.components.stats import (
    StatsComponent,
    Stat,
    StatModifier,
    ModifierType,
    ModifierSource,
)


__all__ = [
    # Constants
    "HealthConstants",
    "MovementConstants",
    "TransformConstants",
    "TeamConstants",
    "StatsConstants",
    "StateMachineConstants",
    # Health
    "HealthComponent",
    "DamageType",
    "HealthState",
    "DamageEvent",
    "HealEvent",
    # Movement
    "MovementComponent",
    "MovementMode",
    "MovementState",
    "MovementSettings",
    "MovementSnapshot",
    # Transform
    "TransformComponent",
    "TransformSnapshot",
    "TransformSpace",
    # Team
    "TeamComponent",
    "TeamRelation",
    "IFFResponse",
    "Team",
    "Faction",
    "TeamRegistry",
    "get_team_registry",
    "set_team_registry",
    # Stats
    "StatsComponent",
    "Stat",
    "StatModifier",
    "ModifierType",
    "ModifierSource",
]
