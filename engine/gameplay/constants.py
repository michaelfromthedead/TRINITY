"""Gameplay layer constants. All gameplay-level magic numbers live here."""

from enum import IntEnum, auto
from typing import Final

# === Entity Lifecycle ===
ENTITY_STATE_CREATING: int = 0
ENTITY_STATE_INITIALIZING: int = 1
ENTITY_STATE_ACTIVE: int = 2
ENTITY_STATE_DEACTIVATING: int = 3
ENTITY_STATE_DESTROYING: int = 4


class EntityState(IntEnum):
    """Entity lifecycle states."""
    CREATING = 0
    INITIALIZING = 1
    ACTIVE = 2
    DEACTIVATING = 3
    DESTROYING = 4


# === Actor/Pawn Types ===
ACTOR_TYPE_STATIC: int = 0
ACTOR_TYPE_DYNAMIC: int = 1
ACTOR_TYPE_PAWN: int = 2
ACTOR_TYPE_CHARACTER: int = 3


class ActorType(IntEnum):
    """Actor type classification."""
    STATIC = 0
    DYNAMIC = 1
    PAWN = 2
    CHARACTER = 3


# === AI Constants ===
AI_UPDATE_RATE_DEFAULT: float = 0.1  # 10 Hz
AI_UPDATE_RATE_FAST: float = 0.033  # 30 Hz
AI_UPDATE_RATE_SLOW: float = 0.5  # 2 Hz

# Behavior Tree
BT_NODE_STATUS_RUNNING: int = 0
BT_NODE_STATUS_SUCCESS: int = 1
BT_NODE_STATUS_FAILURE: int = 2


class BTNodeStatus(IntEnum):
    """Behavior tree node execution status."""
    RUNNING = 0
    SUCCESS = 1
    FAILURE = 2


# Utility AI
UTILITY_SCORE_MIN: float = 0.0
UTILITY_SCORE_MAX: float = 1.0
UTILITY_CURVE_LINEAR: int = 0
UTILITY_CURVE_QUADRATIC: int = 1
UTILITY_CURVE_LOGISTIC: int = 2
UTILITY_CURVE_EXPONENTIAL: int = 3
UTILITY_LOGISTIC_CENTER: float = 0.5
UTILITY_LOGISTIC_STEEPNESS: float = 10.0


class UtilityCurveType(IntEnum):
    """Utility AI response curve types."""
    LINEAR = 0
    QUADRATIC = 1
    LOGISTIC = 2
    EXPONENTIAL = 3


# GOAP
GOAP_MAX_PLAN_DEPTH: int = 10
GOAP_MAX_ITERATIONS: int = 1000

# Perception
PERCEPTION_SENSE_SIGHT: int = 0
PERCEPTION_SENSE_HEARING: int = 1
PERCEPTION_SENSE_DAMAGE: int = 2
PERCEPTION_SENSE_SQUAD: int = 3


class PerceptionSense(IntEnum):
    """AI perception sense types."""
    SIGHT = 0
    HEARING = 1
    DAMAGE = 2
    SQUAD = 3


PERCEPTION_DEFAULT_SIGHT_RANGE: float = 50.0
PERCEPTION_DEFAULT_HEARING_RANGE: float = 30.0
PERCEPTION_DEFAULT_FOV: float = 90.0  # degrees

# === Navigation Constants ===
NAV_AGENT_RADIUS_DEFAULT: float = 0.5
NAV_AGENT_HEIGHT_DEFAULT: float = 2.0
NAV_STEP_HEIGHT_DEFAULT: float = 0.3
NAV_MAX_SLOPE_DEFAULT: float = 45.0  # degrees

NAV_LINK_TYPE_JUMP: int = 0
NAV_LINK_TYPE_DROP: int = 1
NAV_LINK_TYPE_CLIMB: int = 2
NAV_LINK_TYPE_TELEPORT: int = 3


class NavLinkType(IntEnum):
    """Navigation link types."""
    JUMP = 0
    DROP = 1
    CLIMB = 2
    TELEPORT = 3


# Pathfinding
PATHFIND_ALGORITHM_ASTAR: int = 0
PATHFIND_ALGORITHM_JPS: int = 1
PATHFIND_ALGORITHM_THETA_STAR: int = 2
PATHFIND_ALGORITHM_HPA_STAR: int = 3


class PathfindAlgorithm(IntEnum):
    """Pathfinding algorithm selection."""
    ASTAR = 0
    JPS = 1
    THETA_STAR = 2
    HPA_STAR = 3


PATH_MAX_NODES: int = 10000
PATH_SMOOTH_ITERATIONS: int = 3

# Steering
STEERING_ARRIVE_SLOW_RADIUS: float = 5.0
STEERING_ARRIVE_TARGET_RADIUS: float = 0.5
STEERING_SEPARATION_RADIUS: float = 2.0
STEERING_COHESION_RADIUS: float = 10.0
STEERING_ALIGNMENT_RADIUS: float = 5.0

# Avoidance
AVOIDANCE_RVO: int = 0
AVOIDANCE_ORCA: int = 1
AVOIDANCE_FORCE: int = 2


class AvoidanceType(IntEnum):
    """Collision avoidance algorithm type."""
    RVO = 0
    ORCA = 1
    FORCE = 2


AVOIDANCE_TIME_HORIZON: float = 2.0
AVOIDANCE_MAX_NEIGHBORS: int = 10

# === Input Constants ===
INPUT_DEVICE_KEYBOARD: int = 0
INPUT_DEVICE_MOUSE: int = 1
INPUT_DEVICE_GAMEPAD: int = 2
INPUT_DEVICE_TOUCH: int = 3
INPUT_DEVICE_MOTION: int = 4
INPUT_DEVICE_XR: int = 5


class InputDeviceType(IntEnum):
    """Input device types."""
    KEYBOARD = 0
    MOUSE = 1
    GAMEPAD = 2
    TOUCH = 3
    MOTION = 4
    XR = 5


INPUT_TRIGGER_PRESSED: int = 0
INPUT_TRIGGER_RELEASED: int = 1
INPUT_TRIGGER_HOLD: int = 2
INPUT_TRIGGER_TAP: int = 3
INPUT_TRIGGER_COMBO: int = 4


class InputTrigger(IntEnum):
    """Input action trigger types."""
    PRESSED = 0
    RELEASED = 1
    HOLD = 2
    TAP = 3
    COMBO = 4


INPUT_DEADZONE_DEFAULT: float = 0.15
INPUT_HOLD_TIME_DEFAULT: float = 0.5
INPUT_TAP_TIME_MAX: float = 0.2
INPUT_COMBO_TIMEOUT: float = 0.5

# Context priorities
INPUT_CONTEXT_PRIORITY_MENU: int = 100
INPUT_CONTEXT_PRIORITY_DIALOGUE: int = 90
INPUT_CONTEXT_PRIORITY_VEHICLE: int = 50
INPUT_CONTEXT_PRIORITY_GAMEPLAY: int = 0

# === Camera Constants ===
CAMERA_TYPE_FIRST_PERSON: int = 0
CAMERA_TYPE_THIRD_PERSON: int = 1
CAMERA_TYPE_TOP_DOWN: int = 2
CAMERA_TYPE_ISOMETRIC: int = 3
CAMERA_TYPE_FREE: int = 4
CAMERA_TYPE_CINEMATIC: int = 5


class CameraType(IntEnum):
    """Camera mode types."""
    FIRST_PERSON = 0
    THIRD_PERSON = 1
    TOP_DOWN = 2
    ISOMETRIC = 3
    FREE = 4
    CINEMATIC = 5


CAMERA_DEFAULT_FOV: float = 75.0
CAMERA_DEFAULT_NEAR_CLIP: float = 0.1
CAMERA_DEFAULT_FAR_CLIP: float = 1000.0

# Third person defaults
CAMERA_ORBIT_DISTANCE_DEFAULT: float = 5.0
CAMERA_ORBIT_HEIGHT_DEFAULT: float = 2.0
CAMERA_LAG_SPEED_DEFAULT: float = 10.0

# Collision
CAMERA_COLLISION_RADIUS: float = 0.2
CAMERA_COLLISION_PUSH_SPEED: float = 8.0
CAMERA_COLLISION_PULL_SPEED: float = 4.0

# Effects
CAMERA_SHAKE_DECAY_RATE: float = 0.8
CAMERA_SHAKE_MAX_OFFSET: float = 0.5
CAMERA_SHAKE_MAX_ROTATION: float = 5.0  # degrees

# === Ability System Constants ===
ABILITY_STATE_INACTIVE: int = 0
ABILITY_STATE_ACTIVATING: int = 1
ABILITY_STATE_ACTIVE: int = 2
ABILITY_STATE_ENDING: int = 3
ABILITY_STATE_COOLDOWN: int = 4


class AbilityState(IntEnum):
    """Ability execution states."""
    INACTIVE = 0
    ACTIVATING = 1
    ACTIVE = 2
    ENDING = 3
    COOLDOWN = 4


# Effect types
EFFECT_TYPE_INSTANT: int = 0
EFFECT_TYPE_DURATION: int = 1
EFFECT_TYPE_INFINITE: int = 2
EFFECT_TYPE_PERIODIC: int = 3


class EffectType(IntEnum):
    """Gameplay effect duration types."""
    INSTANT = 0
    DURATION = 1
    INFINITE = 2
    PERIODIC = 3


# Modifier operations
MODIFIER_OP_ADD: int = 0
MODIFIER_OP_MULTIPLY: int = 1
MODIFIER_OP_OVERRIDE: int = 2


class ModifierOp(IntEnum):
    """Attribute modifier operations."""
    ADD = 0
    MULTIPLY = 1
    OVERRIDE = 2


# Stacking modes
STACKING_NONE: int = 0
STACKING_DURATION: int = 1
STACKING_INTENSITY: int = 2
STACKING_INDEPENDENT: int = 3


class StackingMode(IntEnum):
    """Buff stacking modes."""
    NONE = 0
    DURATION = 1
    INTENSITY = 2
    INDEPENDENT = 3


# Targeting
TARGETING_SELF: int = 0
TARGETING_SINGLE: int = 1
TARGETING_AOE: int = 2
TARGETING_CONE: int = 3
TARGETING_LINE: int = 4
TARGETING_PROJECTILE: int = 5


class TargetingType(IntEnum):
    """Ability targeting types."""
    SELF = 0
    SINGLE = 1
    AOE = 2
    CONE = 3
    LINE = 4
    PROJECTILE = 5


# === Economy Constants ===
# Item types
ITEM_TYPE_EQUIPMENT: int = 0
ITEM_TYPE_CONSUMABLE: int = 1
ITEM_TYPE_MATERIAL: int = 2
ITEM_TYPE_KEY_ITEM: int = 3
ITEM_TYPE_CURRENCY: int = 4


class ItemType(IntEnum):
    """Item type classification."""
    EQUIPMENT = 0
    CONSUMABLE = 1
    MATERIAL = 2
    KEY_ITEM = 3
    CURRENCY = 4


# Rarity
RARITY_COMMON: int = 0
RARITY_UNCOMMON: int = 1
RARITY_RARE: int = 2
RARITY_EPIC: int = 3
RARITY_LEGENDARY: int = 4


class ItemRarity(IntEnum):
    """Item rarity tiers."""
    COMMON = 0
    UNCOMMON = 1
    RARE = 2
    EPIC = 3
    LEGENDARY = 4


# Equipment slots
SLOT_HEAD: int = 0
SLOT_CHEST: int = 1
SLOT_HANDS: int = 2
SLOT_LEGS: int = 3
SLOT_FEET: int = 4
SLOT_WEAPON: int = 5
SLOT_OFFHAND: int = 6


class EquipmentSlot(IntEnum):
    """Equipment slot positions."""
    HEAD = 0
    CHEST = 1
    HANDS = 2
    LEGS = 3
    FEET = 4
    WEAPON = 5
    OFFHAND = 6


INVENTORY_DEFAULT_CAPACITY: int = 50
INVENTORY_STACK_MAX_DEFAULT: int = 99

# Loot
LOOT_ROLL_PRECISION: int = 10000  # For weighted selection

# === Quest Constants ===
QUEST_STATE_UNAVAILABLE: int = 0
QUEST_STATE_AVAILABLE: int = 1
QUEST_STATE_ACTIVE: int = 2
QUEST_STATE_COMPLETE: int = 3
QUEST_STATE_TURNED_IN: int = 4
QUEST_STATE_FAILED: int = 5


class QuestState(IntEnum):
    """Quest progression states."""
    UNAVAILABLE = 0
    AVAILABLE = 1
    ACTIVE = 2
    COMPLETE = 3
    TURNED_IN = 4
    FAILED = 5


# Objective types
OBJECTIVE_TYPE_KILL: int = 0
OBJECTIVE_TYPE_COLLECT: int = 1
OBJECTIVE_TYPE_TALK: int = 2
OBJECTIVE_TYPE_REACH: int = 3
OBJECTIVE_TYPE_ESCORT: int = 4
OBJECTIVE_TYPE_INTERACT: int = 5


class ObjectiveType(IntEnum):
    """Quest objective types."""
    KILL = 0
    COLLECT = 1
    TALK = 2
    REACH = 3
    ESCORT = 4
    INTERACT = 5


# Objective flow
OBJECTIVE_FLOW_SEQUENTIAL: int = 0
OBJECTIVE_FLOW_PARALLEL: int = 1
OBJECTIVE_FLOW_BRANCHING: int = 2
OBJECTIVE_FLOW_OPTIONAL: int = 3


class ObjectiveFlow(IntEnum):
    """Objective progression flow types."""
    SEQUENTIAL = 0
    PARALLEL = 1
    BRANCHING = 2
    OPTIONAL = 3


# Dialogue node types
DIALOGUE_NODE_TEXT: int = 0
DIALOGUE_NODE_CHOICE: int = 1
DIALOGUE_NODE_BRANCH: int = 2
DIALOGUE_NODE_EVENT: int = 3
DIALOGUE_NODE_RANDOM: int = 4


class DialogueNodeType(IntEnum):
    """Dialogue node types."""
    TEXT = 0
    CHOICE = 1
    BRANCH = 2
    EVENT = 3
    RANDOM = 4


# === Component Constants ===
# Health
HEALTH_MIN: float = 0.0
HEALTH_DEFAULT_MAX: float = 100.0
HEALTH_REGEN_TICK_RATE: float = 1.0  # seconds

# Movement
MOVEMENT_DEFAULT_WALK_SPEED: float = 3.0
MOVEMENT_DEFAULT_RUN_SPEED: float = 6.0
MOVEMENT_DEFAULT_JUMP_FORCE: float = 10.0
MOVEMENT_DEFAULT_GRAVITY: float = -20.0

# State machine
FSM_MAX_STATE_HISTORY: int = 10
FSM_MAX_SUBSTATES: int = 8

# === Combat Constants ===
# Damage types
DAMAGE_TYPE_PHYSICAL: int = 0
DAMAGE_TYPE_MAGICAL: int = 1
DAMAGE_TYPE_FIRE: int = 2
DAMAGE_TYPE_ICE: int = 3
DAMAGE_TYPE_LIGHTNING: int = 4
DAMAGE_TYPE_POISON: int = 5
DAMAGE_TYPE_TRUE: int = 6  # Ignores resistances


class DamageType(IntEnum):
    """Damage type classification."""
    PHYSICAL = 0
    MAGICAL = 1
    FIRE = 2
    ICE = 3
    LIGHTNING = 4
    POISON = 5
    TRUE = 6


# Critical hit
CRIT_CHANCE_DEFAULT: float = 0.05  # 5%
CRIT_MULTIPLIER_DEFAULT: float = 2.0

# Death
DEATH_FADE_TIME: float = 3.0
RESPAWN_DELAY_DEFAULT: float = 5.0

# Teams
TEAM_NEUTRAL: int = 0
TEAM_PLAYER: int = 1
TEAM_ENEMY: int = 2
TEAM_ALLY: int = 3


class Team(IntEnum):
    """Team/faction identifiers."""
    NEUTRAL = 0
    PLAYER = 1
    ENEMY = 2
    ALLY = 3


# Scoring
SCORE_KILL: int = 100
SCORE_DEATH: int = -50
SCORE_ASSIST: int = 50
SCORE_OBJECTIVE: int = 200

# Game modes
GAME_MODE_DEATHMATCH: int = 0
GAME_MODE_TEAM_DEATHMATCH: int = 1
GAME_MODE_CTF: int = 2
GAME_MODE_KOTH: int = 3
GAME_MODE_BATTLE_ROYALE: int = 4


class GameMode(IntEnum):
    """Game mode types."""
    DEATHMATCH = 0
    TEAM_DEATHMATCH = 1
    CTF = 2
    KOTH = 3
    BATTLE_ROYALE = 4


# Match settings
MATCH_TIME_LIMIT_DEFAULT: float = 600.0  # 10 minutes
MATCH_SCORE_LIMIT_DEFAULT: int = 50

# === System Execution Order ===
# These define the order in which gameplay systems run each frame
SYSTEM_ORDER_INPUT: int = 100
SYSTEM_ORDER_AI: int = 200
SYSTEM_ORDER_ABILITY: int = 300
SYSTEM_ORDER_EFFECT: int = 400
SYSTEM_ORDER_MOVEMENT: int = 500
SYSTEM_ORDER_STATE_MACHINE: int = 600
SYSTEM_ORDER_DAMAGE: int = 700
SYSTEM_ORDER_DEATH: int = 800
SYSTEM_ORDER_CLEANUP: int = 900
SYSTEM_ORDER_TRIGGER: int = 1000

# === Event Channels ===
EVENT_CHANNEL_GAMEPLAY: Final[str] = "gameplay"
EVENT_CHANNEL_COMBAT: Final[str] = "combat"
EVENT_CHANNEL_INPUT: Final[str] = "input"
EVENT_CHANNEL_AI: Final[str] = "ai"
EVENT_CHANNEL_QUEST: Final[str] = "quest"
EVENT_CHANNEL_INVENTORY: Final[str] = "inventory"
EVENT_CHANNEL_ABILITY: Final[str] = "ability"
