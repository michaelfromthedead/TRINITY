"""
Economy/Inventory System.

Provides inventory containers, items, crafting, loot, and trading systems.
"""

from __future__ import annotations

# =============================================================================
# Constants
# =============================================================================

from .constants import (
    # Enums
    Rarity,
    ItemType,
    EquipmentSlot,
    ContainerType,
    CraftingQuality,
    TradeState,
    AttributeType,
    ResistanceType,
    EconomyEvent,
    # Rarity data
    RARITY_NAMES,
    RARITY_COLORS,
    RARITY_DROP_WEIGHTS,
    RARITY_PITY_THRESHOLDS,
    # Stack and weight limits
    STACKABLE_TYPES,
    SELLABLE_TYPES,
    DEFAULT_STACK_LIMITS,
    MAX_STACK_SIZE,
    DEFAULT_CONTAINER_SLOTS,
    DEFAULT_WEIGHT_LIMITS,
    WEIGHT_UNIT,
    EXCLUSIVE_SLOTS,
    # Currency
    MAX_GOLD,
    MAX_PREMIUM_CURRENCY,
    CURRENCY_DENOMINATIONS,
    # Trading
    TRADE_TIMEOUT,
    TRADE_MAX_DISTANCE,
    # Crafting
    QUALITY_STAT_MULTIPLIERS,
    QUALITY_BASE_CHANCES,
    SKILL_QUALITY_BONUS_PER_LEVEL,
    # Loot
    LUCK_BONUS_PER_POINT,
    MAX_LUCK_BONUS,
    PITY_INCREMENT,
    PITY_RESET_ON_SUCCESS,
    PITY_WEIGHT_BOOST,
    # Attributes
    MAX_RESISTANCE_PERCENT,
    UPGRADE_BONUS_PER_LEVEL,
    DEFAULT_MAX_DURABILITY,
    # Limits
    DEFAULT_MIN_LEVEL,
    DEFAULT_MAX_LEVEL,
    DEFAULT_MAX_VALUE,
    DEFAULT_MAX_DROPS,
)

# =============================================================================
# Inventory
# =============================================================================

from .inventory import (
    # Item definitions
    ItemDefinition,
    ItemInstance,
    # Containers
    InventorySlot,
    InventoryContainer,
    InventoryEvent,
    # Registry
    ItemRegistry,
    # Serialization
    Serializer,
    ECONOMY_SCHEMA_VERSION,
)

# =============================================================================
# Crafting
# =============================================================================

from .crafting import (
    # Stations
    CraftingStation,
    # Ingredients
    Ingredient,
    IngredientCategory,
    IngredientRequirement,
    # Recipes
    RecipeOutput,
    SkillRequirement,
    Recipe,
    # Results
    CraftingResultType,
    CraftingResult,
    CraftingContext,
    CraftingQueueEntry,
    # System
    CraftingSystem,
    CraftingCallback,
    # Builder
    RecipeBuilder,
    # Registry
    CraftingRegistry,
    # Factory
    RecipeFactory,
    # Decorators
    recipe,
    crafting_station,
    ingredient,
    economy,
    crafting,
    # Query functions
    get_registered_recipes,
    get_registered_stations,
    get_economy_classes,
    get_recipes_for_station_from_registry,
    get_recipes_by_skill_from_registry,
    get_craftable_items,
    clear_registered,
    # Serialization helpers
    ingredient_from_dict,
    recipe_output_from_dict,
    skill_requirement_from_dict,
    recipe_from_dict,
    crafting_station_from_dict,
)

# =============================================================================
# Loot
# =============================================================================

from .loot import (
    # Random sources
    RandomSource,
    DefaultRandomSource,
    SeededRandomSource,
    # Conditions
    LootCondition,
    LevelCondition,
    QuestCondition,
    FlagCondition,
    AttributeCondition,
    RandomChanceCondition,
    # Entries
    LootEntry,
    NestedTableEntry,
    CurrencyEntry,
    LootTableEntry,
    # Results
    LootDrop,
    CurrencyDrop,
    LootResult,
    # Pity
    PityTracker,
    # Tables
    LootTable,
    LootTableBuilder,
    LootTableRegistry,
    # Roller
    LootRoller,
    # Serialization
    loot_entry_from_dict,
    loot_table_from_dict,
)

# =============================================================================
# Equipment
# =============================================================================

from .equipment import (
    # Modifiers
    StatModifier,
    ResistanceModifier,
    SpecialEffect,
    # Stats
    EquipmentStats,
    # Definition and Instance
    EquipmentDefinition,
    EquipmentInstance,
    # Sets
    SetBonus,
    EquipmentSet,
    # Container
    EquipmentContainer,
    EquipChangeCallback,
)

__all__ = [
    # Constants - Enums
    "Rarity",
    "ItemType",
    "EquipmentSlot",
    "ContainerType",
    "CraftingQuality",
    "TradeState",
    "AttributeType",
    "ResistanceType",
    "EconomyEvent",
    # Constants - Rarity
    "RARITY_NAMES",
    "RARITY_COLORS",
    "RARITY_DROP_WEIGHTS",
    "RARITY_PITY_THRESHOLDS",
    # Constants - Stacking
    "STACKABLE_TYPES",
    "SELLABLE_TYPES",
    "DEFAULT_STACK_LIMITS",
    "MAX_STACK_SIZE",
    "DEFAULT_CONTAINER_SLOTS",
    "DEFAULT_WEIGHT_LIMITS",
    "WEIGHT_UNIT",
    "EXCLUSIVE_SLOTS",
    # Constants - Currency
    "MAX_GOLD",
    "MAX_PREMIUM_CURRENCY",
    "CURRENCY_DENOMINATIONS",
    # Constants - Trading
    "TRADE_TIMEOUT",
    "TRADE_MAX_DISTANCE",
    # Constants - Crafting
    "QUALITY_STAT_MULTIPLIERS",
    "QUALITY_BASE_CHANCES",
    "SKILL_QUALITY_BONUS_PER_LEVEL",
    # Constants - Loot
    "LUCK_BONUS_PER_POINT",
    "MAX_LUCK_BONUS",
    "PITY_INCREMENT",
    "PITY_RESET_ON_SUCCESS",
    "PITY_WEIGHT_BOOST",
    # Constants - Attributes
    "MAX_RESISTANCE_PERCENT",
    "UPGRADE_BONUS_PER_LEVEL",
    "DEFAULT_MAX_DURABILITY",
    # Constants - Limits
    "DEFAULT_MIN_LEVEL",
    "DEFAULT_MAX_LEVEL",
    "DEFAULT_MAX_VALUE",
    "DEFAULT_MAX_DROPS",
    # Inventory
    "ItemDefinition",
    "ItemInstance",
    "InventorySlot",
    "InventoryContainer",
    "InventoryEvent",
    "ItemRegistry",
    "Serializer",
    "ECONOMY_SCHEMA_VERSION",
    # Crafting
    "CraftingStation",
    "Ingredient",
    "IngredientCategory",
    "IngredientRequirement",
    "RecipeOutput",
    "SkillRequirement",
    "Recipe",
    "CraftingResultType",
    "CraftingResult",
    "CraftingContext",
    "CraftingQueueEntry",
    "CraftingSystem",
    "CraftingCallback",
    "RecipeBuilder",
    "CraftingRegistry",
    "RecipeFactory",
    "recipe",
    "crafting_station",
    "ingredient",
    "economy",
    "crafting",
    "get_registered_recipes",
    "get_registered_stations",
    "get_economy_classes",
    "get_recipes_for_station_from_registry",
    "get_recipes_by_skill_from_registry",
    "get_craftable_items",
    "clear_registered",
    "ingredient_from_dict",
    "recipe_output_from_dict",
    "skill_requirement_from_dict",
    "recipe_from_dict",
    "crafting_station_from_dict",
    # Loot
    "RandomSource",
    "DefaultRandomSource",
    "SeededRandomSource",
    "LootCondition",
    "LevelCondition",
    "QuestCondition",
    "FlagCondition",
    "AttributeCondition",
    "RandomChanceCondition",
    "LootEntry",
    "NestedTableEntry",
    "CurrencyEntry",
    "LootTableEntry",
    "LootDrop",
    "CurrencyDrop",
    "LootResult",
    "PityTracker",
    "LootTable",
    "LootTableBuilder",
    "LootTableRegistry",
    "LootRoller",
    "loot_entry_from_dict",
    "loot_table_from_dict",
    # Equipment
    "StatModifier",
    "ResistanceModifier",
    "SpecialEffect",
    "EquipmentStats",
    "EquipmentDefinition",
    "EquipmentInstance",
    "SetBonus",
    "EquipmentSet",
    "EquipmentContainer",
    "EquipChangeCallback",
]
