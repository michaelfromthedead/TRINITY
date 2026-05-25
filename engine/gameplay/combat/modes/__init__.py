"""
Game mode implementations.

Provides concrete implementations of various game modes:
- Deathmatch: Free-for-all, first to kill limit or highest score at time
- Team Deathmatch: Team-based kill competition
- Capture the Flag: Capture enemy flag and return to base
- King of the Hill: Control objective zones
- Battle Royale: Last player/team standing with shrinking zone
"""

from engine.gameplay.combat.modes.deathmatch import Deathmatch, DeathmatchConfig

try:
    from engine.gameplay.combat.modes.team_deathmatch import TeamDeathmatch, TeamDeathmatchConfig
except ImportError:
    TeamDeathmatch = None  # not yet implemented
    TeamDeathmatchConfig = None

try:
    from engine.gameplay.combat.modes.ctf import CaptureTheFlag, CTFConfig, FlagState
except ImportError:
    CaptureTheFlag = None
    CTFConfig = None
    FlagState = None

try:
    from engine.gameplay.combat.modes.koth import KingOfTheHill, KOTHConfig, HillZone
except ImportError:
    KingOfTheHill = None
    KOTHConfig = None
    HillZone = None

try:
    from engine.gameplay.combat.modes.battle_royale import (
        BattleRoyale,
        BattleRoyaleConfig,
        ShrinkingZone,
        ZonePhase,
    )
except ImportError:
    BattleRoyale = None
    BattleRoyaleConfig = None
    ShrinkingZone = None
    ZonePhase = None

__all__ = [
    # Deathmatch
    "Deathmatch",
    "DeathmatchConfig",
    # Team Deathmatch
    "TeamDeathmatch",
    "TeamDeathmatchConfig",
    # Capture the Flag
    "CaptureTheFlag",
    "CTFConfig",
    "FlagState",
    # King of the Hill
    "KingOfTheHill",
    "KOTHConfig",
    "HillZone",
    # Battle Royale
    "BattleRoyale",
    "BattleRoyaleConfig",
    "ShrinkingZone",
    "ZonePhase",
]
