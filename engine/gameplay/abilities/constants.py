"""
Ability System Constants.

Default values for cooldowns, attribute ranges, effect durations, and other
configuration constants used throughout the ability system.
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import Final


# =============================================================================
# ABILITY PHASES
# =============================================================================


class AbilityPhase(IntEnum):
    """Phases of ability execution flow."""

    NONE = 0
    ACTIVATE = auto()  # Check CanActivate, start cooldown, grant tags
    COMMIT = auto()    # Pay costs, consume resources
    EXECUTE = auto()   # Run logic, apply effects, spawn actors
    END = auto()       # Cleanup, end cooldown, remove tags


class AbilityEndReason(IntEnum):
    """Reasons an ability may end."""

    COMPLETED = 0      # Normal completion
    CANCELLED = auto() # Manually cancelled
    INTERRUPTED = auto()  # Interrupted by another ability/effect
    EXPIRED = auto()   # Duration expired
    KILLED = auto()    # Owner died


# =============================================================================
# EFFECT TYPES
# =============================================================================


class EffectType(IntEnum):
    """Types of gameplay effects."""

    INSTANT = 0        # One-time application
    DURATION = auto()  # Time-limited effect
    INFINITE = auto()  # Until explicitly removed
    PERIODIC = auto()  # Repeating tick effect


class ModifierOperation(IntEnum):
    """Operations for attribute modifiers."""

    ADD = 0            # Flat addition: base + modifier
    MULTIPLY = auto()  # Multiplicative: base * (1 + modifier)
    OVERRIDE = auto()  # Replace value entirely
    STACKING = auto()  # Special stacking behavior


# =============================================================================
# BUFF/DEBUFF STACKING
# =============================================================================


class StackingMode(IntEnum):
    """How buffs/debuffs stack when reapplied."""

    NONE = 0           # No stacking, refresh only
    DURATION = auto()  # Extend duration, no intensity increase
    INTENSITY = auto() # Increase intensity, no duration extension
    INDEPENDENT = auto()  # Create separate stack instances


# =============================================================================
# TARGETING
# =============================================================================


class TargetingMode(IntEnum):
    """Modes for ability targeting."""

    SELF = 0           # Target self only
    ACTOR = auto()     # Target another actor
    POINT = auto()     # Target a world position
    AREA = auto()      # Area of effect
    CONFIRMATION = auto()  # Requires explicit confirmation


class AreaShape(IntEnum):
    """Shapes for area targeting."""

    CIRCLE = 0
    RECTANGLE = auto()
    CONE = auto()
    LINE = auto()
    CAPSULE = auto()


# =============================================================================
# DEFAULT VALUES - COOLDOWNS
# =============================================================================

# Minimum cooldown in seconds
DEFAULT_MIN_COOLDOWN: Final[float] = 0.0

# Maximum cooldown in seconds (10 minutes)
DEFAULT_MAX_COOLDOWN: Final[float] = 600.0

# Global cooldown (shared between abilities)
DEFAULT_GLOBAL_COOLDOWN: Final[float] = 1.0

# Cooldown reduction cap (percentage, 0.0 to 1.0)
DEFAULT_MAX_COOLDOWN_REDUCTION: Final[float] = 0.75


# =============================================================================
# DEFAULT VALUES - ATTRIBUTES
# =============================================================================

# Standard attribute ranges
DEFAULT_ATTRIBUTE_MIN: Final[float] = 0.0
DEFAULT_ATTRIBUTE_MAX: Final[float] = 999999.0

# Common attribute caps
DEFAULT_HEALTH_MIN: Final[float] = 0.0
DEFAULT_HEALTH_MAX: Final[float] = 10000.0

DEFAULT_MANA_MIN: Final[float] = 0.0
DEFAULT_MANA_MAX: Final[float] = 5000.0

DEFAULT_STAMINA_MIN: Final[float] = 0.0
DEFAULT_STAMINA_MAX: Final[float] = 1000.0

# Regeneration defaults (per second)
DEFAULT_HEALTH_REGEN: Final[float] = 0.0
DEFAULT_MANA_REGEN: Final[float] = 1.0
DEFAULT_STAMINA_REGEN: Final[float] = 10.0

# Movement speed defaults
DEFAULT_SPEED_MIN: Final[float] = 0.0
DEFAULT_SPEED_MAX: Final[float] = 2000.0
DEFAULT_SPEED_BASE: Final[float] = 400.0

# Damage/armor defaults
DEFAULT_DAMAGE_MIN: Final[float] = 0.0
DEFAULT_DAMAGE_MAX: Final[float] = 100000.0
DEFAULT_ARMOR_MIN: Final[float] = 0.0
DEFAULT_ARMOR_MAX: Final[float] = 10000.0


# =============================================================================
# DEFAULT VALUES - EFFECTS
# =============================================================================

# Effect duration limits
DEFAULT_MIN_DURATION: Final[float] = 0.0
DEFAULT_MAX_DURATION: Final[float] = 3600.0  # 1 hour

# Periodic effect tick rates
DEFAULT_MIN_TICK_RATE: Final[float] = 0.1  # 10 ticks per second max
DEFAULT_MAX_TICK_RATE: Final[float] = 60.0  # Once per minute min
DEFAULT_TICK_RATE: Final[float] = 1.0

# Buff stacking limits
DEFAULT_MAX_STACKS: Final[int] = 99
DEFAULT_MIN_STACKS: Final[int] = 1

# Effect magnitude limits
DEFAULT_MIN_MAGNITUDE: Final[float] = -999999.0
DEFAULT_MAX_MAGNITUDE: Final[float] = 999999.0


# =============================================================================
# DEFAULT VALUES - TARGETING
# =============================================================================

# Range defaults
DEFAULT_MIN_RANGE: Final[float] = 0.0
DEFAULT_MAX_RANGE: Final[float] = 10000.0
DEFAULT_MELEE_RANGE: Final[float] = 2.5
DEFAULT_RANGED_RANGE: Final[float] = 30.0

# Area of effect defaults
DEFAULT_AOE_RADIUS: Final[float] = 5.0
DEFAULT_AOE_MIN_RADIUS: Final[float] = 0.1
DEFAULT_AOE_MAX_RADIUS: Final[float] = 100.0

# Cone defaults
DEFAULT_CONE_ANGLE: Final[float] = 45.0  # degrees
DEFAULT_CONE_MIN_ANGLE: Final[float] = 1.0
DEFAULT_CONE_MAX_ANGLE: Final[float] = 180.0

# Line defaults
DEFAULT_LINE_WIDTH: Final[float] = 1.0
DEFAULT_LINE_MIN_WIDTH: Final[float] = 0.1
DEFAULT_LINE_MAX_WIDTH: Final[float] = 20.0


# =============================================================================
# GAMEPLAY TAG CONSTANTS
# =============================================================================

# Tag separator for hierarchy
TAG_SEPARATOR: Final[str] = "."

# Maximum tag depth
MAX_TAG_DEPTH: Final[int] = 10

# Wildcard character for tag matching
TAG_WILDCARD: Final[str] = "*"

# Tag registry cache size
TAG_REGISTRY_CACHE_SIZE: Final[int] = 1024


# =============================================================================
# TIMING CONSTANTS
# =============================================================================

# Epsilon for float comparisons
EPSILON: Final[float] = 1e-6

# Frame time constants (60 FPS reference)
FRAME_TIME_60FPS: Final[float] = 1.0 / 60.0
FRAME_TIME_30FPS: Final[float] = 1.0 / 30.0


# =============================================================================
# MODIFIER ORDER OF OPERATIONS
# =============================================================================

# Order in which modifiers are applied to attributes
# Lower values are applied first
MODIFIER_ORDER_ADD_BASE: Final[int] = 0
MODIFIER_ORDER_MULTIPLY_BASE: Final[int] = 100
MODIFIER_ORDER_ADD_BONUS: Final[int] = 200
MODIFIER_ORDER_MULTIPLY_BONUS: Final[int] = 300
MODIFIER_ORDER_OVERRIDE: Final[int] = 400
MODIFIER_ORDER_CLAMP: Final[int] = 500


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "AbilityPhase",
    "AbilityEndReason",
    "EffectType",
    "ModifierOperation",
    "StackingMode",
    "TargetingMode",
    "AreaShape",
    # Cooldown defaults
    "DEFAULT_MIN_COOLDOWN",
    "DEFAULT_MAX_COOLDOWN",
    "DEFAULT_GLOBAL_COOLDOWN",
    "DEFAULT_MAX_COOLDOWN_REDUCTION",
    # Attribute defaults
    "DEFAULT_ATTRIBUTE_MIN",
    "DEFAULT_ATTRIBUTE_MAX",
    "DEFAULT_HEALTH_MIN",
    "DEFAULT_HEALTH_MAX",
    "DEFAULT_MANA_MIN",
    "DEFAULT_MANA_MAX",
    "DEFAULT_STAMINA_MIN",
    "DEFAULT_STAMINA_MAX",
    "DEFAULT_HEALTH_REGEN",
    "DEFAULT_MANA_REGEN",
    "DEFAULT_STAMINA_REGEN",
    "DEFAULT_SPEED_MIN",
    "DEFAULT_SPEED_MAX",
    "DEFAULT_SPEED_BASE",
    "DEFAULT_DAMAGE_MIN",
    "DEFAULT_DAMAGE_MAX",
    "DEFAULT_ARMOR_MIN",
    "DEFAULT_ARMOR_MAX",
    # Effect defaults
    "DEFAULT_MIN_DURATION",
    "DEFAULT_MAX_DURATION",
    "DEFAULT_MIN_TICK_RATE",
    "DEFAULT_MAX_TICK_RATE",
    "DEFAULT_TICK_RATE",
    "DEFAULT_MAX_STACKS",
    "DEFAULT_MIN_STACKS",
    "DEFAULT_MIN_MAGNITUDE",
    "DEFAULT_MAX_MAGNITUDE",
    # Targeting defaults
    "DEFAULT_MIN_RANGE",
    "DEFAULT_MAX_RANGE",
    "DEFAULT_MELEE_RANGE",
    "DEFAULT_RANGED_RANGE",
    "DEFAULT_AOE_RADIUS",
    "DEFAULT_AOE_MIN_RADIUS",
    "DEFAULT_AOE_MAX_RADIUS",
    "DEFAULT_CONE_ANGLE",
    "DEFAULT_CONE_MIN_ANGLE",
    "DEFAULT_CONE_MAX_ANGLE",
    "DEFAULT_LINE_WIDTH",
    "DEFAULT_LINE_MIN_WIDTH",
    "DEFAULT_LINE_MAX_WIDTH",
    # Tag constants
    "TAG_SEPARATOR",
    "MAX_TAG_DEPTH",
    "TAG_WILDCARD",
    "TAG_REGISTRY_CACHE_SIZE",
    # Timing constants
    "EPSILON",
    "FRAME_TIME_60FPS",
    "FRAME_TIME_30FPS",
    # Modifier order
    "MODIFIER_ORDER_ADD_BASE",
    "MODIFIER_ORDER_MULTIPLY_BASE",
    "MODIFIER_ORDER_ADD_BONUS",
    "MODIFIER_ORDER_MULTIPLY_BONUS",
    "MODIFIER_ORDER_OVERRIDE",
    "MODIFIER_ORDER_CLAMP",
]
