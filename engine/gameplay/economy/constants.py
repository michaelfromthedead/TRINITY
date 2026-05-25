"""
Constants for the Economy/Inventory System.

Defines limits, rarity levels, item types, and other system-wide constants.
"""

from enum import Enum, IntEnum, auto
from typing import Dict, FrozenSet


# =============================================================================
# Rarity System
# =============================================================================


class Rarity(IntEnum):
    """Item rarity levels with drop weight multipliers."""
    COMMON = 0
    UNCOMMON = 1
    RARE = 2
    EPIC = 3
    LEGENDARY = 4
    MYTHIC = 5


# Rarity display names
RARITY_NAMES: Dict[Rarity, str] = {
    Rarity.COMMON: "Common",
    Rarity.UNCOMMON: "Uncommon",
    Rarity.RARE: "Rare",
    Rarity.EPIC: "Epic",
    Rarity.LEGENDARY: "Legendary",
    Rarity.MYTHIC: "Mythic",
}

# Rarity color codes (hex)
RARITY_COLORS: Dict[Rarity, str] = {
    Rarity.COMMON: "#FFFFFF",    # White
    Rarity.UNCOMMON: "#1EFF00",  # Green
    Rarity.RARE: "#0070DD",      # Blue
    Rarity.EPIC: "#A335EE",      # Purple
    Rarity.LEGENDARY: "#FF8000", # Orange
    Rarity.MYTHIC: "#E6CC80",    # Gold
}

# Base drop weights (inverse - lower = rarer)
RARITY_DROP_WEIGHTS: Dict[Rarity, float] = {
    Rarity.COMMON: 100.0,
    Rarity.UNCOMMON: 50.0,
    Rarity.RARE: 15.0,
    Rarity.EPIC: 5.0,
    Rarity.LEGENDARY: 1.0,
    Rarity.MYTHIC: 0.1,
}

# Minimum guaranteed drops after N misses (pity system)
RARITY_PITY_THRESHOLDS: Dict[Rarity, int] = {
    Rarity.COMMON: 0,       # Always available
    Rarity.UNCOMMON: 5,     # After 5 misses
    Rarity.RARE: 20,        # After 20 misses
    Rarity.EPIC: 50,        # After 50 misses
    Rarity.LEGENDARY: 100,  # After 100 misses
    Rarity.MYTHIC: 200,     # After 200 misses
}


# =============================================================================
# Item Types
# =============================================================================


class ItemType(Enum):
    """Categories of items."""
    EQUIPMENT = auto()
    CONSUMABLE = auto()
    MATERIAL = auto()
    KEY_ITEM = auto()
    CURRENCY = auto()
    QUEST = auto()
    JUNK = auto()


# Which item types can stack
STACKABLE_TYPES: FrozenSet[ItemType] = frozenset({
    ItemType.CONSUMABLE,
    ItemType.MATERIAL,
    ItemType.CURRENCY,
    ItemType.JUNK,
})

# Which item types can be sold
SELLABLE_TYPES: FrozenSet[ItemType] = frozenset({
    ItemType.EQUIPMENT,
    ItemType.CONSUMABLE,
    ItemType.MATERIAL,
    ItemType.JUNK,
})


# =============================================================================
# Equipment Slots
# =============================================================================


class EquipmentSlot(Enum):
    """Body slots for equippable items."""
    HEAD = auto()
    CHEST = auto()
    HANDS = auto()
    LEGS = auto()
    FEET = auto()
    MAIN_HAND = auto()
    OFF_HAND = auto()
    TWO_HAND = auto()  # Uses both weapon slots
    NECK = auto()
    RING_1 = auto()
    RING_2 = auto()
    BACK = auto()
    BELT = auto()
    TRINKET_1 = auto()
    TRINKET_2 = auto()


# Slots that are mutually exclusive
EXCLUSIVE_SLOTS: Dict[EquipmentSlot, FrozenSet[EquipmentSlot]] = {
    EquipmentSlot.TWO_HAND: frozenset({EquipmentSlot.MAIN_HAND, EquipmentSlot.OFF_HAND}),
}


# =============================================================================
# Container Types
# =============================================================================


class ContainerType(Enum):
    """Types of inventory containers."""
    PLAYER_INVENTORY = auto()
    PLAYER_EQUIPMENT = auto()
    CHEST = auto()
    SHOP = auto()
    STASH = auto()
    LOOT = auto()
    CRAFTING_OUTPUT = auto()
    TRADE_OFFER = auto()
    MAIL = auto()


# =============================================================================
# Stack and Weight Limits
# =============================================================================


# Default maximum stack sizes by item type
DEFAULT_STACK_LIMITS: Dict[ItemType, int] = {
    ItemType.EQUIPMENT: 1,
    ItemType.CONSUMABLE: 99,
    ItemType.MATERIAL: 999,
    ItemType.KEY_ITEM: 1,
    ItemType.CURRENCY: 999999,
    ItemType.QUEST: 1,
    ItemType.JUNK: 99,
}

# Maximum stack size override
MAX_STACK_SIZE: int = 999999

# Default container slot counts
DEFAULT_CONTAINER_SLOTS: Dict[ContainerType, int] = {
    ContainerType.PLAYER_INVENTORY: 30,
    ContainerType.PLAYER_EQUIPMENT: 15,
    ContainerType.CHEST: 50,
    ContainerType.SHOP: 100,
    ContainerType.STASH: 200,
    ContainerType.LOOT: 20,
    ContainerType.CRAFTING_OUTPUT: 5,
    ContainerType.TRADE_OFFER: 20,
    ContainerType.MAIL: 10,
}

# Weight limits (0.0 means unlimited)
DEFAULT_WEIGHT_LIMITS: Dict[ContainerType, float] = {
    ContainerType.PLAYER_INVENTORY: 100.0,
    ContainerType.PLAYER_EQUIPMENT: 0.0,  # Unlimited
    ContainerType.CHEST: 500.0,
    ContainerType.SHOP: 0.0,              # Unlimited
    ContainerType.STASH: 0.0,             # Unlimited
    ContainerType.LOOT: 0.0,              # Unlimited
    ContainerType.CRAFTING_OUTPUT: 0.0,   # Unlimited
    ContainerType.TRADE_OFFER: 50.0,
    ContainerType.MAIL: 25.0,
}

# Base weight unit for calculations
WEIGHT_UNIT: float = 0.1


# =============================================================================
# Currency Limits
# =============================================================================


# Maximum currency values
MAX_GOLD: int = 999_999_999
MAX_PREMIUM_CURRENCY: int = 99_999_999

# Currency exchange rates (to base currency, typically copper)
CURRENCY_DENOMINATIONS = {
    "copper": 1,
    "silver": 100,
    "gold": 10000,
    "platinum": 1000000,
}


# =============================================================================
# Trading Constants
# =============================================================================


class TradeState(Enum):
    """States of a trade transaction."""
    PENDING = auto()
    ACCEPTED = auto()
    DECLINED = auto()
    CANCELLED = auto()
    COMPLETED = auto()
    EXPIRED = auto()


# Trade timeout in seconds
TRADE_TIMEOUT: float = 300.0  # 5 minutes

# Maximum distance for in-person trades
TRADE_MAX_DISTANCE: float = 10.0


# =============================================================================
# Crafting Constants
# =============================================================================


class CraftingQuality(IntEnum):
    """Quality levels for crafted items."""
    POOR = 0
    NORMAL = 1
    GOOD = 2
    EXCELLENT = 3
    MASTERWORK = 4


# Quality bonus multipliers
QUALITY_STAT_MULTIPLIERS: Dict[CraftingQuality, float] = {
    CraftingQuality.POOR: 0.8,
    CraftingQuality.NORMAL: 1.0,
    CraftingQuality.GOOD: 1.1,
    CraftingQuality.EXCELLENT: 1.25,
    CraftingQuality.MASTERWORK: 1.5,
}

# Base chance to achieve each quality (modified by skill)
QUALITY_BASE_CHANCES: Dict[CraftingQuality, float] = {
    CraftingQuality.POOR: 0.05,
    CraftingQuality.NORMAL: 0.70,
    CraftingQuality.GOOD: 0.20,
    CraftingQuality.EXCELLENT: 0.04,
    CraftingQuality.MASTERWORK: 0.01,
}


# =============================================================================
# Loot System Constants
# =============================================================================


# Luck bonus per point
LUCK_BONUS_PER_POINT: float = 0.01  # 1% per luck point

# Maximum luck bonus
MAX_LUCK_BONUS: float = 2.0  # 200% maximum

# Pity system increment (how much pity counter increases per miss)
PITY_INCREMENT: int = 1

# Pity system reset on successful drop
PITY_RESET_ON_SUCCESS: bool = True


# =============================================================================
# Attribute Constants
# =============================================================================


class AttributeType(Enum):
    """Character attributes that equipment can modify."""
    STRENGTH = auto()
    DEXTERITY = auto()
    CONSTITUTION = auto()
    INTELLIGENCE = auto()
    WISDOM = auto()
    CHARISMA = auto()
    LUCK = auto()


class ResistanceType(Enum):
    """Damage resistance types."""
    PHYSICAL = auto()
    FIRE = auto()
    ICE = auto()
    LIGHTNING = auto()
    POISON = auto()
    ARCANE = auto()
    HOLY = auto()
    SHADOW = auto()


# Maximum resistance percentage
MAX_RESISTANCE_PERCENT: float = 0.75  # 75% max damage reduction


# =============================================================================
# Equipment Upgrade Constants
# =============================================================================


# Stat bonus per upgrade level (5% per level)
UPGRADE_BONUS_PER_LEVEL: float = 0.05

# Default maximum durability for equipment
DEFAULT_MAX_DURABILITY: float = 100.0


# =============================================================================
# Level and Value Limits
# =============================================================================


# Default minimum level for conditions/requirements
DEFAULT_MIN_LEVEL: int = 1

# Default maximum level for conditions/requirements
DEFAULT_MAX_LEVEL: int = 999

# Default maximum attribute/value cap
DEFAULT_MAX_VALUE: int = 999

# Default maximum drops per loot roll
DEFAULT_MAX_DROPS: int = 999


# =============================================================================
# Pity System Constants
# =============================================================================


# Pity weight boost multiplier when pity triggers
PITY_WEIGHT_BOOST: int = 100


# =============================================================================
# Crafting Skill Constants
# =============================================================================


# Quality bonus per skill level over requirement (2% per level)
SKILL_QUALITY_BONUS_PER_LEVEL: float = 0.02


# =============================================================================
# Event Names
# =============================================================================


class EconomyEvent(Enum):
    """Events emitted by the economy system."""
    ITEM_ADDED = auto()
    ITEM_REMOVED = auto()
    ITEM_MOVED = auto()
    ITEM_SPLIT = auto()
    ITEM_MERGED = auto()
    ITEM_EQUIPPED = auto()
    ITEM_UNEQUIPPED = auto()
    ITEM_USED = auto()
    ITEM_DESTROYED = auto()
    LOOT_ROLLED = auto()
    LOOT_COLLECTED = auto()
    CURRENCY_GAINED = auto()
    CURRENCY_SPENT = auto()
    TRADE_STARTED = auto()
    TRADE_UPDATED = auto()
    TRADE_COMPLETED = auto()
    TRADE_CANCELLED = auto()
    CRAFT_STARTED = auto()
    CRAFT_COMPLETED = auto()
    CRAFT_FAILED = auto()
