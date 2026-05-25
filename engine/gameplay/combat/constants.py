"""
Combat System Constants

Defines base values for damage, resistances, regeneration rates, hitbox multipliers,
team configurations, and scoring parameters used throughout the combat system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Dict, FrozenSet


# =============================================================================
# DAMAGE TYPES
# =============================================================================


class DamageType(Enum):
    """Enumeration of all damage types in the combat system."""

    PHYSICAL = "physical"
    FIRE = "fire"
    ICE = "ice"
    LIGHTNING = "lightning"
    POISON = "poison"
    ARCANE = "arcane"
    HOLY = "holy"
    SHADOW = "shadow"
    NATURE = "nature"
    BLEED = "bleed"
    TRUE = "true"  # Ignores all resistances


# Damage types that can be resisted
RESISTABLE_DAMAGE_TYPES: FrozenSet[DamageType] = frozenset(
    dt for dt in DamageType if dt != DamageType.TRUE
)

# Elemental damage types
ELEMENTAL_DAMAGE_TYPES: FrozenSet[DamageType] = frozenset({
    DamageType.FIRE,
    DamageType.ICE,
    DamageType.LIGHTNING,
    DamageType.ARCANE,
    DamageType.NATURE,
})

# Physical damage types (affected by armor)
PHYSICAL_DAMAGE_TYPES: FrozenSet[DamageType] = frozenset({
    DamageType.PHYSICAL,
    DamageType.BLEED,
})

# Damage over time types
DOT_DAMAGE_TYPES: FrozenSet[DamageType] = frozenset({
    DamageType.POISON,
    DamageType.BLEED,
    DamageType.FIRE,  # Burning
})


# =============================================================================
# BASE DAMAGE VALUES
# =============================================================================


# Default base damage for different weapon categories
BASE_DAMAGE_UNARMED: float = 5.0
BASE_DAMAGE_LIGHT_WEAPON: float = 10.0
BASE_DAMAGE_MEDIUM_WEAPON: float = 20.0
BASE_DAMAGE_HEAVY_WEAPON: float = 35.0
BASE_DAMAGE_RANGED: float = 15.0
BASE_DAMAGE_MAGIC: float = 25.0

# Minimum damage after all reductions (prevents healing from negative damage)
MINIMUM_DAMAGE: float = 1.0

# Maximum damage cap (prevents one-shot kills in most cases)
MAXIMUM_DAMAGE: float = 99999.0


# =============================================================================
# ARMOR AND RESISTANCE
# =============================================================================


# Armor reduction formula constants
# damage_after_armor = damage * (100 / (100 + armor))
ARMOR_CONSTANT: float = 100.0

# Maximum armor effectiveness (cap at 90% reduction)
MAX_ARMOR_REDUCTION: float = 0.90

# Maximum resistance per damage type (cap at 75%)
MAX_RESISTANCE: float = 0.75

# Minimum resistance (can go negative for vulnerability)
MIN_RESISTANCE: float = -0.50

# Default resistance values by damage type
DEFAULT_RESISTANCES: Dict[DamageType, float] = {
    DamageType.PHYSICAL: 0.0,
    DamageType.FIRE: 0.0,
    DamageType.ICE: 0.0,
    DamageType.LIGHTNING: 0.0,
    DamageType.POISON: 0.0,
    DamageType.ARCANE: 0.0,
    DamageType.HOLY: 0.0,
    DamageType.SHADOW: 0.0,
    DamageType.NATURE: 0.0,
    DamageType.BLEED: 0.0,
    DamageType.TRUE: 0.0,
}


# =============================================================================
# HEALTH AND REGENERATION
# =============================================================================


# Default health values
DEFAULT_MAX_HEALTH: float = 100.0
DEFAULT_CURRENT_HEALTH: float = 100.0
MINIMUM_MAX_HEALTH: float = 1.0

# Health regeneration rates (per second)
DEFAULT_HEALTH_REGEN_RATE: float = 0.0
BASE_HEALTH_REGEN_RATE: float = 1.0  # 1 HP per second
MAX_HEALTH_REGEN_RATE: float = 100.0

# Regeneration delay after taking damage (seconds)
REGEN_DELAY_AFTER_DAMAGE: float = 5.0

# Out of combat regeneration multiplier
OUT_OF_COMBAT_REGEN_MULTIPLIER: float = 3.0

# Time without damage to be considered "out of combat" (seconds)
OUT_OF_COMBAT_THRESHOLD: float = 10.0


# =============================================================================
# INVULNERABILITY
# =============================================================================


# Default invulnerability duration after respawn (seconds)
RESPAWN_INVULNERABILITY_DURATION: float = 3.0

# Invulnerability flash interval for visual feedback (seconds)
INVULNERABILITY_FLASH_INTERVAL: float = 0.1


# =============================================================================
# DEATH AND RESPAWN
# =============================================================================


class DeathState(IntEnum):
    """Entity death state progression."""

    ALIVE = 0
    DYING = auto()  # Playing death animation
    DEAD = auto()  # Fully dead, awaiting cleanup
    RESPAWNING = auto()  # In respawn process


# Time to remain in DYING state (seconds)
DYING_DURATION: float = 1.0

# Default respawn time (seconds)
DEFAULT_RESPAWN_TIME: float = 5.0

# Minimum respawn time
MIN_RESPAWN_TIME: float = 0.0

# Maximum respawn time
MAX_RESPAWN_TIME: float = 300.0  # 5 minutes

# Respawn health percentage (1.0 = full health)
RESPAWN_HEALTH_PERCENTAGE: float = 1.0


# =============================================================================
# HITBOX ZONES AND MULTIPLIERS
# =============================================================================


class HitboxZone(Enum):
    """Hitbox zones with associated damage multipliers."""

    HEAD = "head"
    NECK = "neck"
    TORSO = "torso"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    BACK = "back"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    GENERIC = "generic"  # Default zone


# Damage multipliers by hitbox zone
HITBOX_DAMAGE_MULTIPLIERS: Dict[HitboxZone, float] = {
    HitboxZone.HEAD: 2.0,       # Headshot - double damage
    HitboxZone.NECK: 1.75,      # Critical area
    HitboxZone.TORSO: 1.0,      # Base damage
    HitboxZone.CHEST: 1.0,      # Base damage
    HitboxZone.ABDOMEN: 0.9,    # Slightly less critical
    HitboxZone.BACK: 1.25,      # Backstab bonus
    HitboxZone.LEFT_ARM: 0.75,  # Limb - reduced damage
    HitboxZone.RIGHT_ARM: 0.75,
    HitboxZone.LEFT_HAND: 0.5,  # Extremity - heavily reduced
    HitboxZone.RIGHT_HAND: 0.5,
    HitboxZone.LEFT_LEG: 0.75,  # Limb - reduced damage
    HitboxZone.RIGHT_LEG: 0.75,
    HitboxZone.LEFT_FOOT: 0.5,  # Extremity - heavily reduced
    HitboxZone.RIGHT_FOOT: 0.5,
    HitboxZone.GENERIC: 1.0,    # Default multiplier
}

# Critical hit zones (can trigger critical effects)
CRITICAL_HIT_ZONES: FrozenSet[HitboxZone] = frozenset({
    HitboxZone.HEAD,
    HitboxZone.NECK,
})

# Zones that can cause limb damage/disabling
LIMB_ZONES: FrozenSet[HitboxZone] = frozenset({
    HitboxZone.LEFT_ARM,
    HitboxZone.RIGHT_ARM,
    HitboxZone.LEFT_HAND,
    HitboxZone.RIGHT_HAND,
    HitboxZone.LEFT_LEG,
    HitboxZone.RIGHT_LEG,
    HitboxZone.LEFT_FOOT,
    HitboxZone.RIGHT_FOOT,
})


# =============================================================================
# TEAMS AND FACTIONS
# =============================================================================


class TeamRelation(IntEnum):
    """Relationship between teams."""

    HOSTILE = 0     # Can attack each other
    NEUTRAL = 1     # Cannot attack by default
    FRIENDLY = 2    # Same team / allied


# Default team ID for entities without a team
DEFAULT_TEAM_ID: int = 0

# Neutral/world team ID (environmental hazards, etc.)
NEUTRAL_TEAM_ID: int = -1

# Maximum number of teams
MAX_TEAMS: int = 64

# Friendly fire settings
FRIENDLY_FIRE_FULL: float = 1.0      # 100% friendly fire damage
FRIENDLY_FIRE_REDUCED: float = 0.5    # 50% friendly fire damage
FRIENDLY_FIRE_NONE: float = 0.0       # No friendly fire


# =============================================================================
# SCORING
# =============================================================================


# Points awarded for various actions
POINTS_PER_KILL: int = 100
POINTS_PER_ASSIST: int = 50
POINTS_PER_DEATH: int = 0  # Deaths typically don't award points (but track K/D)
POINTS_PER_OBJECTIVE: int = 200
POINTS_PER_HEADSHOT_BONUS: int = 25
POINTS_PER_FIRST_BLOOD: int = 50
POINTS_PER_REVENGE_KILL: int = 25
POINTS_PER_KILLSTREAK_BONUS: int = 10  # Per kill in streak

# Assist tracking
ASSIST_DAMAGE_THRESHOLD: float = 0.1  # Must deal 10% of max HP for assist
ASSIST_TIME_WINDOW: float = 10.0  # Seconds after damage to qualify for assist

# Killstreak thresholds
KILLSTREAK_THRESHOLDS: Dict[int, str] = {
    3: "killing_spree",
    5: "rampage",
    7: "dominating",
    10: "unstoppable",
    15: "godlike",
    20: "legendary",
}

# Multi-kill time windows (seconds between kills)
MULTI_KILL_WINDOW: float = 4.0

MULTI_KILL_NAMES: Dict[int, str] = {
    2: "double_kill",
    3: "triple_kill",
    4: "quad_kill",
    5: "penta_kill",
    6: "mega_kill",
}


# =============================================================================
# COMBAT EVENTS
# =============================================================================


class CombatEventType(Enum):
    """Types of combat events that can be emitted."""

    DAMAGE_DEALT = "damage_dealt"
    DAMAGE_RECEIVED = "damage_received"
    HEALTH_CHANGED = "health_changed"
    DEATH = "death"
    RESPAWN = "respawn"
    KILL = "kill"
    ASSIST = "assist"
    HEADSHOT = "headshot"
    CRITICAL_HIT = "critical_hit"
    KILLSTREAK = "killstreak"
    KILLSTREAK_ENDED = "killstreak_ended"
    MULTI_KILL = "multi_kill"
    FIRST_BLOOD = "first_blood"
    REVENGE = "revenge"
    INVULNERABILITY_START = "invulnerability_start"
    INVULNERABILITY_END = "invulnerability_end"
    TEAM_CHANGED = "team_changed"


# =============================================================================
# DAMAGE SOURCE CATEGORIES
# =============================================================================


class DamageSource(Enum):
    """Categories of damage sources for tracking and filtering."""

    PLAYER = "player"
    NPC = "npc"
    ENVIRONMENT = "environment"
    SELF = "self"  # Self-inflicted damage
    WORLD = "world"  # Fall damage, drowning, etc.
    DOT = "dot"  # Damage over time effects
    REFLECT = "reflect"  # Reflected damage
    UNKNOWN = "unknown"


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class DamageConfig:
    """Configuration for damage calculation."""

    armor_constant: float = ARMOR_CONSTANT
    max_armor_reduction: float = MAX_ARMOR_REDUCTION
    max_resistance: float = MAX_RESISTANCE
    min_resistance: float = MIN_RESISTANCE
    minimum_damage: float = MINIMUM_DAMAGE
    maximum_damage: float = MAXIMUM_DAMAGE


@dataclass(frozen=True)
class HealthConfig:
    """Configuration for health system."""

    default_max_health: float = DEFAULT_MAX_HEALTH
    minimum_max_health: float = MINIMUM_MAX_HEALTH
    default_regen_rate: float = DEFAULT_HEALTH_REGEN_RATE
    max_regen_rate: float = MAX_HEALTH_REGEN_RATE
    regen_delay_after_damage: float = REGEN_DELAY_AFTER_DAMAGE
    out_of_combat_threshold: float = OUT_OF_COMBAT_THRESHOLD
    out_of_combat_regen_multiplier: float = OUT_OF_COMBAT_REGEN_MULTIPLIER


@dataclass(frozen=True)
class DeathConfig:
    """Configuration for death and respawn."""

    dying_duration: float = DYING_DURATION
    default_respawn_time: float = DEFAULT_RESPAWN_TIME
    min_respawn_time: float = MIN_RESPAWN_TIME
    max_respawn_time: float = MAX_RESPAWN_TIME
    respawn_health_percentage: float = RESPAWN_HEALTH_PERCENTAGE
    respawn_invulnerability_duration: float = RESPAWN_INVULNERABILITY_DURATION


@dataclass(frozen=True)
class ScoringConfig:
    """Configuration for scoring system."""

    points_per_kill: int = POINTS_PER_KILL
    points_per_assist: int = POINTS_PER_ASSIST
    points_per_death: int = POINTS_PER_DEATH
    points_per_objective: int = POINTS_PER_OBJECTIVE
    points_per_headshot_bonus: int = POINTS_PER_HEADSHOT_BONUS
    points_per_first_blood: int = POINTS_PER_FIRST_BLOOD
    points_per_revenge_kill: int = POINTS_PER_REVENGE_KILL
    points_per_killstreak_bonus: int = POINTS_PER_KILLSTREAK_BONUS
    assist_damage_threshold: float = ASSIST_DAMAGE_THRESHOLD
    assist_time_window: float = ASSIST_TIME_WINDOW
    multi_kill_window: float = MULTI_KILL_WINDOW


@dataclass(frozen=True)
class TeamConfig:
    """Configuration for team system."""

    max_teams: int = MAX_TEAMS
    default_friendly_fire: float = FRIENDLY_FIRE_NONE
    allow_team_changes: bool = True
    allow_team_damage: bool = False


# Default configurations
DEFAULT_DAMAGE_CONFIG = DamageConfig()
DEFAULT_HEALTH_CONFIG = HealthConfig()
DEFAULT_DEATH_CONFIG = DeathConfig()
DEFAULT_SCORING_CONFIG = ScoringConfig()
DEFAULT_TEAM_CONFIG = TeamConfig()


# =============================================================================
# HITBOX COMBAT MODIFIERS
# =============================================================================


# Counter-hit damage bonus multiplier
COUNTER_HIT_DAMAGE_MULTIPLIER: float = 1.25


# =============================================================================
# SPAWN SCORING WEIGHTS
# =============================================================================


# Bonus points for spawning at team-owned spawn point
SPAWN_TEAM_BONUS_SCORE: int = 20

# Maximum bonus points for spawn time freshness
SPAWN_TIME_FRESHNESS_MAX_BONUS: float = 10.0

# Time divisor for freshness calculation (60 seconds = full bonus)
SPAWN_TIME_FRESHNESS_DIVISOR: float = 6.0

# Maximum bonus points for distance from enemies
SPAWN_DISTANCE_MAX_BONUS: float = 30.0

# Distance divisor for spawn point scoring
SPAWN_DISTANCE_DIVISOR: float = 10.0


# =============================================================================
# MATCH DEFAULTS
# =============================================================================


# Default maximum spectators for a match
DEFAULT_MAX_SPECTATORS: int = 10


# =============================================================================
# HISTORY LIMITS
# =============================================================================


# Maximum damage history entries to keep
MAX_DAMAGE_HISTORY_SIZE: int = 1000

# Maximum scoring event history entries to keep
MAX_SCORING_HISTORY_SIZE: int = 10000


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "DamageType",
    "DeathState",
    "HitboxZone",
    "TeamRelation",
    "CombatEventType",
    "DamageSource",
    # Damage type sets
    "RESISTABLE_DAMAGE_TYPES",
    "ELEMENTAL_DAMAGE_TYPES",
    "PHYSICAL_DAMAGE_TYPES",
    "DOT_DAMAGE_TYPES",
    # Base damage values
    "BASE_DAMAGE_UNARMED",
    "BASE_DAMAGE_LIGHT_WEAPON",
    "BASE_DAMAGE_MEDIUM_WEAPON",
    "BASE_DAMAGE_HEAVY_WEAPON",
    "BASE_DAMAGE_RANGED",
    "BASE_DAMAGE_MAGIC",
    "MINIMUM_DAMAGE",
    "MAXIMUM_DAMAGE",
    # Armor and resistance
    "ARMOR_CONSTANT",
    "MAX_ARMOR_REDUCTION",
    "MAX_RESISTANCE",
    "MIN_RESISTANCE",
    "DEFAULT_RESISTANCES",
    # Health and regeneration
    "DEFAULT_MAX_HEALTH",
    "DEFAULT_CURRENT_HEALTH",
    "MINIMUM_MAX_HEALTH",
    "DEFAULT_HEALTH_REGEN_RATE",
    "BASE_HEALTH_REGEN_RATE",
    "MAX_HEALTH_REGEN_RATE",
    "REGEN_DELAY_AFTER_DAMAGE",
    "OUT_OF_COMBAT_REGEN_MULTIPLIER",
    "OUT_OF_COMBAT_THRESHOLD",
    # Invulnerability
    "RESPAWN_INVULNERABILITY_DURATION",
    "INVULNERABILITY_FLASH_INTERVAL",
    # Death and respawn
    "DYING_DURATION",
    "DEFAULT_RESPAWN_TIME",
    "MIN_RESPAWN_TIME",
    "MAX_RESPAWN_TIME",
    "RESPAWN_HEALTH_PERCENTAGE",
    # Hitbox
    "HITBOX_DAMAGE_MULTIPLIERS",
    "CRITICAL_HIT_ZONES",
    "LIMB_ZONES",
    # Teams
    "DEFAULT_TEAM_ID",
    "NEUTRAL_TEAM_ID",
    "MAX_TEAMS",
    "FRIENDLY_FIRE_FULL",
    "FRIENDLY_FIRE_REDUCED",
    "FRIENDLY_FIRE_NONE",
    # Scoring
    "POINTS_PER_KILL",
    "POINTS_PER_ASSIST",
    "POINTS_PER_DEATH",
    "POINTS_PER_OBJECTIVE",
    "POINTS_PER_HEADSHOT_BONUS",
    "POINTS_PER_FIRST_BLOOD",
    "POINTS_PER_REVENGE_KILL",
    "POINTS_PER_KILLSTREAK_BONUS",
    "ASSIST_DAMAGE_THRESHOLD",
    "ASSIST_TIME_WINDOW",
    "KILLSTREAK_THRESHOLDS",
    "MULTI_KILL_WINDOW",
    "MULTI_KILL_NAMES",
    # Config dataclasses
    "DamageConfig",
    "HealthConfig",
    "DeathConfig",
    "ScoringConfig",
    "TeamConfig",
    "DEFAULT_DAMAGE_CONFIG",
    "DEFAULT_HEALTH_CONFIG",
    "DEFAULT_DEATH_CONFIG",
    "DEFAULT_SCORING_CONFIG",
    "DEFAULT_TEAM_CONFIG",
    # Hitbox combat modifiers
    "COUNTER_HIT_DAMAGE_MULTIPLIER",
    # Spawn scoring weights
    "SPAWN_TEAM_BONUS_SCORE",
    "SPAWN_TIME_FRESHNESS_MAX_BONUS",
    "SPAWN_TIME_FRESHNESS_DIVISOR",
    "SPAWN_DISTANCE_MAX_BONUS",
    "SPAWN_DISTANCE_DIVISOR",
    # Match defaults
    "DEFAULT_MAX_SPECTATORS",
    # History limits
    "MAX_DAMAGE_HISTORY_SIZE",
    "MAX_SCORING_HISTORY_SIZE",
]
