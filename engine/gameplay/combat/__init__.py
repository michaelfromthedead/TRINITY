"""
Combat system module for game modes, matches, and spawning.

This module provides:
- GameMode: Base class for game mode rules, win conditions, and scoring
- Match: Match lifecycle management (Lobby -> Countdown -> Play -> End -> Results)
- SpawnManager: Spawn point management and team-based spawning
- Game Modes: Deathmatch, Team Deathmatch, CTF, KOTH, Battle Royale
"""

from engine.gameplay.combat.game_mode import (
    GameMode,
    GameModeConfig,
    GameModeRules,
    WinCondition,
    WinConditionType,
    ScoringEvent,
    ScoringEventType,
)
from engine.gameplay.combat.match import (
    Match,
    MatchState,
    MatchConfig,
    MatchResult,
    MatchEvents,
)
from engine.gameplay.combat.spawn_manager import (
    SpawnManager,
    SpawnPoint,
    SpawnRule,
    SpawnRuleType,
    TeamSpawnConfig,
)

__all__ = [
    # Game Mode
    "GameMode",
    "GameModeConfig",
    "GameModeRules",
    "WinCondition",
    "WinConditionType",
    "ScoringEvent",
    "ScoringEventType",
    # Match
    "Match",
    "MatchState",
    "MatchConfig",
    "MatchResult",
    "MatchEvents",
    # Spawn Manager
    "SpawnManager",
    "SpawnPoint",
    "SpawnRule",
    "SpawnRuleType",
    "TeamSpawnConfig",
]
