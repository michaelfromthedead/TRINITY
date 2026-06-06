"""
Crafting System.

Provides recipes, ingredient requirements, crafting stations,
crafting process with quality variance, and skill requirements.

Foundation Integration (T-GP-8.13):
- @recipe decorator registers with Foundation Registry
- @crafting_station decorator registers stations
- @ingredient and @economy decorators for metadata
- Runtime discovery via Registry.query(tag="recipe")
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4

from foundation import register_type, Registry, registry

from .constants import (
    CraftingQuality,
    EconomyEvent,
    ItemType,
    QUALITY_BASE_CHANCES,
    QUALITY_STAT_MULTIPLIERS,
    Rarity,
    SKILL_QUALITY_BONUS_PER_LEVEL,
)
from .inventory import ItemDefinition, ItemInstance, InventoryContainer, ECONOMY_SCHEMA_VERSION


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T", bound=type)


# =============================================================================
# Serialization Decorator
# =============================================================================


def serializable(
    name: Optional[str] = None,
    version: int = ECONOMY_SCHEMA_VERSION,
    exclude_fields: Optional[Set[str]] = None,
) -> Callable[[Type], Type]:
    """Decorator to mark a class as serializable."""
    def decorator(cls: Type) -> Type:
        type_name = name or f"{cls.__module__}.{cls.__name__}"
        register_type(cls, type_name)
        cls._serializable = True
        cls._serializable_version = version
        cls._serializable_exclude = exclude_fields or set()
        return cls
    return decorator


# =============================================================================
# Crafting Station
# =============================================================================


@serializable(version=1)
@dataclass
class CraftingStation:
    """A station where crafting can be performed."""
    station_id: str
    name: str
    categories: Tuple[str, ...] = ()
    level: int = 1
    efficiency_bonus: float = 0.0  # Reduces crafting time
    quality_bonus: float = 0.0    # Improves quality chances

    def __hash__(self) -> int:
        return hash(self.station_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "station_id": self.station_id,
            "name": self.name,
            "categories": list(self.categories),
            "level": self.level,
            "efficiency_bonus": self.efficiency_bonus,
            "quality_bonus": self.quality_bonus,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CraftingStation":
        """Deserialize from dictionary."""
        return cls(
            station_id=data["station_id"],
            name=data["name"],
            categories=tuple(data.get("categories", [])),
            level=data.get("level", 1),
            efficiency_bonus=data.get("efficiency_bonus", 0.0),
            quality_bonus=data.get("quality_bonus", 0.0),
        )


# =============================================================================
# Ingredients
# =============================================================================


@serializable(version=1)
@dataclass(frozen=True)
class Ingredient:
    """A required ingredient for a recipe."""
    item_id: str
    quantity: int = 1
    consumed: bool = True  # If False, item is returned after crafting
    quality_min: Optional[CraftingQuality] = None

    def __post_init__(self):
        if self.quantity < 1:
            raise ValueError("Ingredient quantity must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "Ingredient",
            "item_id": self.item_id,
            "quantity": self.quantity,
            "consumed": self.consumed,
            "quality_min": self.quality_min.name if self.quality_min else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Ingredient":
        """Deserialize from dictionary."""
        quality_min = None
        if data.get("quality_min"):
            quality_min = CraftingQuality[data["quality_min"]]
        return cls(
            item_id=data["item_id"],
            quantity=data.get("quantity", 1),
            consumed=data.get("consumed", True),
            quality_min=quality_min,
        )


@serializable(version=1)
@dataclass(frozen=True)
class IngredientCategory:
    """An ingredient requirement by category (any item in category works)."""
    category: str
    quantity: int = 1
    consumed: bool = True

    def __post_init__(self):
        if self.quantity < 1:
            raise ValueError("Ingredient quantity must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "IngredientCategory",
            "category": self.category,
            "quantity": self.quantity,
            "consumed": self.consumed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngredientCategory":
        """Deserialize from dictionary."""
        return cls(
            category=data["category"],
            quantity=data.get("quantity", 1),
            consumed=data.get("consumed", True),
        )


IngredientRequirement = Union[Ingredient, IngredientCategory]


def ingredient_from_dict(data: Dict[str, Any]) -> IngredientRequirement:
    """Deserialize an ingredient requirement from dictionary."""
    if data.get("type") == "IngredientCategory":
        return IngredientCategory.from_dict(data)
    return Ingredient.from_dict(data)


# =============================================================================
# Recipe Output
# =============================================================================


@serializable(version=1)
@dataclass
class RecipeOutput:
    """Output of a crafting recipe."""
    item_id: str
    base_quantity: int = 1
    bonus_quantity_chance: float = 0.0  # Chance for bonus items
    max_bonus_quantity: int = 0
    quality_variance: bool = True  # Whether quality can vary

    def __post_init__(self):
        if self.base_quantity < 1:
            raise ValueError("Output quantity must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "item_id": self.item_id,
            "base_quantity": self.base_quantity,
            "bonus_quantity_chance": self.bonus_quantity_chance,
            "max_bonus_quantity": self.max_bonus_quantity,
            "quality_variance": self.quality_variance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecipeOutput":
        """Deserialize from dictionary."""
        return cls(
            item_id=data["item_id"],
            base_quantity=data.get("base_quantity", 1),
            bonus_quantity_chance=data.get("bonus_quantity_chance", 0.0),
            max_bonus_quantity=data.get("max_bonus_quantity", 0),
            quality_variance=data.get("quality_variance", True),
        )


# =============================================================================
# Skill Requirement
# =============================================================================


@serializable(version=1)
@dataclass(frozen=True)
class SkillRequirement:
    """Skill required to craft a recipe."""
    skill_id: str
    level: int = 1
    grants_xp: int = 0  # XP granted on successful craft

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "skill_id": self.skill_id,
            "level": self.level,
            "grants_xp": self.grants_xp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillRequirement":
        """Deserialize from dictionary."""
        return cls(
            skill_id=data["skill_id"],
            level=data.get("level", 1),
            grants_xp=data.get("grants_xp", 0),
        )


# =============================================================================
# Recipe
# =============================================================================


@serializable(version=1)
@dataclass
class Recipe:
    """
    A crafting recipe definition.

    Specifies ingredients, outputs, requirements, and crafting parameters.
    """
    recipe_id: str
    name: str
    category: str = "misc"
    ingredients: Tuple[IngredientRequirement, ...] = ()
    outputs: Tuple[RecipeOutput, ...] = ()
    station_required: Optional[str] = None
    station_level: int = 1
    skill_requirements: Tuple[SkillRequirement, ...] = ()
    crafting_time: float = 1.0  # Seconds
    unlock_condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""
    is_discoverable: bool = True
    discovered_by_default: bool = True

    def __hash__(self) -> int:
        return hash(self.recipe_id)

    def check_unlock(self, context: Dict[str, Any]) -> bool:
        """Check if recipe is unlocked."""
        if self.unlock_condition is None:
            return True
        return self.unlock_condition(context)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Note: unlock_condition is not serialized (transient).
        """
        return {
            "__version__": ECONOMY_SCHEMA_VERSION,
            "recipe_id": self.recipe_id,
            "name": self.name,
            "category": self.category,
            "ingredients": [i.to_dict() for i in self.ingredients],
            "outputs": [o.to_dict() for o in self.outputs],
            "station_required": self.station_required,
            "station_level": self.station_level,
            "skill_requirements": [s.to_dict() for s in self.skill_requirements],
            "crafting_time": self.crafting_time,
            "description": self.description,
            "is_discoverable": self.is_discoverable,
            "discovered_by_default": self.discovered_by_default,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Recipe":
        """Deserialize from dictionary."""
        return cls(
            recipe_id=data["recipe_id"],
            name=data["name"],
            category=data.get("category", "misc"),
            ingredients=tuple(
                ingredient_from_dict(i) for i in data.get("ingredients", [])
            ),
            outputs=tuple(
                RecipeOutput.from_dict(o) for o in data.get("outputs", [])
            ),
            station_required=data.get("station_required"),
            station_level=data.get("station_level", 1),
            skill_requirements=tuple(
                SkillRequirement.from_dict(s) for s in data.get("skill_requirements", [])
            ),
            crafting_time=data.get("crafting_time", 1.0),
            unlock_condition=None,  # Cannot restore functions
            description=data.get("description", ""),
            is_discoverable=data.get("is_discoverable", True),
            discovered_by_default=data.get("discovered_by_default", True),
        )


# =============================================================================
# Crafting Result
# =============================================================================


class CraftingResultType(Enum):
    """Type of crafting result."""
    SUCCESS = auto()
    FAILURE = auto()
    CRITICAL_SUCCESS = auto()
    PARTIAL = auto()


@serializable(version=1)
@dataclass
class CraftingResult:
    """Result of a crafting attempt."""
    result_type: CraftingResultType
    outputs: List[ItemInstance] = field(default_factory=list)
    quality: CraftingQuality = CraftingQuality.NORMAL
    consumed_ingredients: List[Tuple[str, int]] = field(default_factory=list)
    returned_ingredients: List[ItemInstance] = field(default_factory=list)
    skill_xp_gained: Dict[str, int] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "result_type": self.result_type.name,
            "outputs": [o.to_dict() for o in self.outputs],
            "quality": self.quality.name,
            "consumed_ingredients": self.consumed_ingredients,
            "returned_ingredients": [r.to_dict() for r in self.returned_ingredients],
            "skill_xp_gained": self.skill_xp_gained,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        definition_registry: Optional[Dict[str, ItemDefinition]] = None,
    ) -> "CraftingResult":
        """Deserialize from dictionary."""
        return cls(
            result_type=CraftingResultType[data["result_type"]],
            outputs=[
                ItemInstance.from_dict(o, definition_registry)
                for o in data.get("outputs", [])
            ],
            quality=CraftingQuality[data.get("quality", "NORMAL")],
            consumed_ingredients=data.get("consumed_ingredients", []),
            returned_ingredients=[
                ItemInstance.from_dict(r, definition_registry)
                for r in data.get("returned_ingredients", [])
            ],
            skill_xp_gained=data.get("skill_xp_gained", {}),
            error_message=data.get("error_message"),
        )


# =============================================================================
# Crafting Context
# =============================================================================


@dataclass
class CraftingContext:
    """Context for a crafting operation."""
    crafter_id: str
    inventory: InventoryContainer
    skills: Dict[str, int] = field(default_factory=dict)
    station: Optional[CraftingStation] = None
    luck: float = 0.0
    quality_bonus: float = 0.0
    speed_bonus: float = 0.0


# =============================================================================
# Crafting Queue Entry
# =============================================================================


@dataclass
class CraftingQueueEntry:
    """An entry in the crafting queue."""
    entry_id: UUID = field(default_factory=uuid4)
    recipe_id: str = ""
    started_at: float = 0.0
    duration: float = 1.0
    quantity: int = 1
    completed: int = 0
    context: Optional[CraftingContext] = None

    @property
    def is_complete(self) -> bool:
        """Check if crafting is complete."""
        return self.completed >= self.quantity


# =============================================================================
# Crafting System
# =============================================================================


CraftingCallback = Callable[[CraftingResult], None]


class CraftingSystem:
    """
    Main crafting system.

    Handles recipe registration, requirement checking, crafting execution,
    and quality variance calculations.
    """

    def __init__(
        self,
        item_registry: Optional[Dict[str, ItemDefinition]] = None,
        item_categories: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        Initialize crafting system.

        Args:
            item_registry: Registry of item definitions
            item_categories: Mapping of category to item IDs
        """
        self._recipes: Dict[str, Recipe] = {}
        self._stations: Dict[str, CraftingStation] = {}
        self._item_registry = item_registry or {}
        self._item_categories = item_categories or {}
        self._discovered_recipes: Dict[str, Set[str]] = {}  # crafter_id -> recipe_ids
        self._crafting_queue: Dict[str, List[CraftingQueueEntry]] = {}
        self._completion_callbacks: List[CraftingCallback] = []
        self._rng = random.Random()

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register_recipe(self, recipe: Recipe) -> None:
        """Register a recipe."""
        if recipe.recipe_id in self._recipes:
            raise ValueError(f"Recipe '{recipe.recipe_id}' already registered")
        self._recipes[recipe.recipe_id] = recipe

    def register_station(self, station: CraftingStation) -> None:
        """Register a crafting station."""
        self._stations[station.station_id] = station

    def register_item_category(self, category: str, item_ids: Set[str]) -> None:
        """Register items in a category."""
        if category not in self._item_categories:
            self._item_categories[category] = set()
        self._item_categories[category].update(item_ids)

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        """Get recipe by ID."""
        return self._recipes.get(recipe_id)

    def get_station(self, station_id: str) -> Optional[CraftingStation]:
        """Get station by ID."""
        return self._stations.get(station_id)

    def get_recipes_by_category(self, category: str) -> List[Recipe]:
        """Get all recipes in a category."""
        return [r for r in self._recipes.values() if r.category == category]

    def get_recipes_for_station(self, station_id: str) -> List[Recipe]:
        """Get all recipes that can be crafted at a station."""
        return [r for r in self._recipes.values() if r.station_required == station_id]

    def get_discovered_recipes(self, crafter_id: str) -> Set[str]:
        """Get recipes discovered by a crafter."""
        return self._discovered_recipes.get(crafter_id, set())

    def discover_recipe(self, crafter_id: str, recipe_id: str) -> bool:
        """Mark a recipe as discovered."""
        if recipe_id not in self._recipes:
            return False
        if crafter_id not in self._discovered_recipes:
            self._discovered_recipes[crafter_id] = set()
        self._discovered_recipes[crafter_id].add(recipe_id)
        return True

    def is_recipe_discovered(self, crafter_id: str, recipe_id: str) -> bool:
        """Check if recipe is discovered."""
        recipe = self._recipes.get(recipe_id)
        if not recipe:
            return False
        if recipe.discovered_by_default:
            return True
        return recipe_id in self._discovered_recipes.get(crafter_id, set())

    # -------------------------------------------------------------------------
    # Requirement Checking
    # -------------------------------------------------------------------------

    def check_requirements(
        self,
        recipe: Recipe,
        context: CraftingContext,
    ) -> Tuple[bool, str]:
        """
        Check if all requirements are met to craft a recipe.

        Args:
            recipe: Recipe to check
            context: Crafting context

        Returns:
            Tuple of (can_craft, error_message)
        """
        # Check unlock condition
        unlock_context = {
            "skills": context.skills,
            "crafter_id": context.crafter_id,
        }
        if not recipe.check_unlock(unlock_context):
            return (False, "Recipe is locked")

        # Check station requirement
        if recipe.station_required:
            if not context.station:
                return (False, f"Requires {recipe.station_required} station")
            if context.station.station_id != recipe.station_required:
                return (False, f"Requires {recipe.station_required} station")
            if context.station.level < recipe.station_level:
                return (False, f"Station level too low (need {recipe.station_level})")

        # Check skill requirements
        for skill_req in recipe.skill_requirements:
            current = context.skills.get(skill_req.skill_id, 0)
            if current < skill_req.level:
                return (False, f"Requires {skill_req.skill_id} level {skill_req.level}")

        # Check ingredients
        missing = self._check_ingredients(recipe, context.inventory)
        if missing:
            missing_str = ", ".join(f"{qty}x {item}" for item, qty in missing)
            return (False, f"Missing: {missing_str}")

        return (True, "")

    def _check_ingredients(
        self,
        recipe: Recipe,
        inventory: InventoryContainer,
    ) -> List[Tuple[str, int]]:
        """Check for missing ingredients."""
        missing = []

        for ingredient in recipe.ingredients:
            if isinstance(ingredient, Ingredient):
                have = inventory.count_item(ingredient.item_id)
                if have < ingredient.quantity:
                    missing.append((ingredient.item_id, ingredient.quantity - have))
            elif isinstance(ingredient, IngredientCategory):
                category_items = self._item_categories.get(ingredient.category, set())
                total = sum(inventory.count_item(item_id) for item_id in category_items)
                if total < ingredient.quantity:
                    missing.append((f"[{ingredient.category}]", ingredient.quantity - total))

        return missing

    def get_craftable_count(
        self,
        recipe: Recipe,
        inventory: InventoryContainer,
    ) -> int:
        """Get how many times a recipe can be crafted."""
        min_count = float('inf')

        for ingredient in recipe.ingredients:
            if isinstance(ingredient, Ingredient):
                have = inventory.count_item(ingredient.item_id)
                can_make = have // ingredient.quantity
            elif isinstance(ingredient, IngredientCategory):
                category_items = self._item_categories.get(ingredient.category, set())
                total = sum(inventory.count_item(item_id) for item_id in category_items)
                can_make = total // ingredient.quantity
            else:
                continue

            min_count = min(min_count, can_make)

        return int(min_count) if min_count != float('inf') else 0

    # -------------------------------------------------------------------------
    # Crafting Execution
    # -------------------------------------------------------------------------

    def craft(
        self,
        recipe_id: str,
        context: CraftingContext,
        quantity: int = 1,
    ) -> CraftingResult:
        """
        Craft a recipe immediately.

        Args:
            recipe_id: Recipe to craft
            context: Crafting context
            quantity: Number of times to craft

        Returns:
            CraftingResult with outputs
        """
        recipe = self._recipes.get(recipe_id)
        if not recipe:
            return CraftingResult(
                result_type=CraftingResultType.FAILURE,
                error_message=f"Unknown recipe: {recipe_id}",
            )

        # Check requirements
        can_craft, error = self.check_requirements(recipe, context)
        if not can_craft:
            return CraftingResult(
                result_type=CraftingResultType.FAILURE,
                error_message=error,
            )

        # Check can craft quantity
        available = self.get_craftable_count(recipe, context.inventory)
        actual_quantity = min(quantity, available)
        if actual_quantity <= 0:
            return CraftingResult(
                result_type=CraftingResultType.FAILURE,
                error_message="Insufficient ingredients",
            )

        # Consume ingredients
        consumed = self._consume_ingredients(recipe, context.inventory, actual_quantity)
        returned = self._get_non_consumed(recipe, context.inventory, actual_quantity)

        # Calculate quality
        quality = self._roll_quality(recipe, context)

        # Generate outputs
        outputs = self._generate_outputs(recipe, actual_quantity, quality)

        # Calculate skill XP
        skill_xp = {}
        for skill_req in recipe.skill_requirements:
            if skill_req.grants_xp > 0:
                skill_xp[skill_req.skill_id] = skill_req.grants_xp * actual_quantity

        # Determine result type
        if quality >= CraftingQuality.EXCELLENT:
            result_type = CraftingResultType.CRITICAL_SUCCESS
        elif quality <= CraftingQuality.POOR:
            result_type = CraftingResultType.PARTIAL
        else:
            result_type = CraftingResultType.SUCCESS

        result = CraftingResult(
            result_type=result_type,
            outputs=outputs,
            quality=quality,
            consumed_ingredients=consumed,
            returned_ingredients=returned,
            skill_xp_gained=skill_xp,
        )

        # Notify callbacks
        for callback in self._completion_callbacks:
            callback(result)

        return result

    def _consume_ingredients(
        self,
        recipe: Recipe,
        inventory: InventoryContainer,
        quantity: int,
    ) -> List[Tuple[str, int]]:
        """Consume ingredients from inventory."""
        consumed = []

        for ingredient in recipe.ingredients:
            if not isinstance(ingredient, Ingredient) or not ingredient.consumed:
                continue

            needed = ingredient.quantity * quantity
            removed = inventory.remove_item(ingredient.item_id, needed)
            if removed > 0:
                consumed.append((ingredient.item_id, removed))

        # Handle category ingredients
        for ingredient in recipe.ingredients:
            if not isinstance(ingredient, IngredientCategory) or not ingredient.consumed:
                continue

            needed = ingredient.quantity * quantity
            category_items = self._item_categories.get(ingredient.category, set())

            for item_id in category_items:
                if needed <= 0:
                    break
                removed = inventory.remove_item(item_id, needed)
                if removed > 0:
                    consumed.append((item_id, removed))
                    needed -= removed

        return consumed

    def _get_non_consumed(
        self,
        recipe: Recipe,
        inventory: InventoryContainer,
        quantity: int,
    ) -> List[ItemInstance]:
        """
        Get non-consumed ingredient instances.

        Non-consumed ingredients (consumed=False) remain in the inventory
        after crafting. This method returns information about which items
        were used but not consumed (e.g., tools, catalysts).

        Args:
            recipe: The recipe being crafted
            inventory: The crafter's inventory
            quantity: Number of times the recipe was crafted

        Returns:
            List of ItemInstance objects representing non-consumed ingredients.
            Currently returns empty list as non-consumed items stay in inventory.
        """
        # Non-consumed ingredients remain in the inventory (they were never removed).
        # This method exists for potential future use cases where we might want to
        # track or report which non-consumed items were "used" during crafting
        # (e.g., for tool durability, catalyst tracking, or recipe feedback).
        returned: List[ItemInstance] = []

        # Collect info about non-consumed ingredients for reporting purposes
        for ingredient in recipe.ingredients:
            if isinstance(ingredient, Ingredient) and not ingredient.consumed:
                # The item stays in inventory, but we could track it was used
                # For now, we just acknowledge these exist but don't return them
                # since they weren't actually removed from inventory
                pass

        return returned

    def _roll_quality(self, recipe: Recipe, context: CraftingContext) -> CraftingQuality:
        """Roll for crafting quality."""
        # Calculate quality bonuses
        total_bonus = context.quality_bonus
        if context.station:
            total_bonus += context.station.quality_bonus

        # Skill bonus
        for skill_req in recipe.skill_requirements:
            current = context.skills.get(skill_req.skill_id, 0)
            skill_excess = current - skill_req.level
            total_bonus += skill_excess * SKILL_QUALITY_BONUS_PER_LEVEL

        # Roll quality
        roll = self._rng.random()

        cumulative = 0.0
        for quality in reversed(list(CraftingQuality)):
            base_chance = QUALITY_BASE_CHANCES.get(quality, 0.0)
            adjusted_chance = base_chance * (1.0 + total_bonus)
            cumulative += adjusted_chance
            if roll < cumulative:
                return quality

        return CraftingQuality.NORMAL

    def _generate_outputs(
        self,
        recipe: Recipe,
        quantity: int,
        quality: CraftingQuality,
    ) -> List[ItemInstance]:
        """Generate output items."""
        outputs = []
        quality_mult = QUALITY_STAT_MULTIPLIERS.get(quality, 1.0)

        for output in recipe.outputs:
            item_def = self._item_registry.get(output.item_id)
            if not item_def:
                continue

            for _ in range(quantity):
                # Base quantity
                total_qty = output.base_quantity

                # Bonus quantity
                if output.bonus_quantity_chance > 0 and output.max_bonus_quantity > 0:
                    if self._rng.random() < output.bonus_quantity_chance:
                        bonus = self._rng.randint(1, output.max_bonus_quantity)
                        total_qty += bonus

                # Create item instance
                item = ItemInstance(
                    definition=item_def,
                    quantity=total_qty,
                    custom_data={"quality": quality.value, "quality_mult": quality_mult},
                )
                outputs.append(item)

        return outputs

    # -------------------------------------------------------------------------
    # Crafting Queue
    # -------------------------------------------------------------------------

    def queue_craft(
        self,
        recipe_id: str,
        context: CraftingContext,
        quantity: int = 1,
        current_time: float = 0.0,
    ) -> Optional[CraftingQueueEntry]:
        """
        Add recipe to crafting queue.

        Args:
            recipe_id: Recipe to queue
            context: Crafting context
            quantity: Number to craft
            current_time: Current game time

        Returns:
            Queue entry or None if requirements not met
        """
        recipe = self._recipes.get(recipe_id)
        if not recipe:
            return None

        can_craft, _ = self.check_requirements(recipe, context)
        if not can_craft:
            return None

        # Calculate duration
        duration = recipe.crafting_time
        if context.station:
            duration *= (1.0 - context.station.efficiency_bonus)
        duration *= (1.0 - context.speed_bonus)

        entry = CraftingQueueEntry(
            recipe_id=recipe_id,
            started_at=current_time,
            duration=duration,
            quantity=quantity,
            context=context,
        )

        crafter_id = context.crafter_id
        if crafter_id not in self._crafting_queue:
            self._crafting_queue[crafter_id] = []
        self._crafting_queue[crafter_id].append(entry)

        return entry

    def update_queue(
        self,
        crafter_id: str,
        current_time: float,
    ) -> List[CraftingResult]:
        """
        Update crafting queue and collect completed items.

        Args:
            crafter_id: Crafter's ID
            current_time: Current game time

        Returns:
            List of completed craft results
        """
        queue = self._crafting_queue.get(crafter_id, [])
        results = []

        for entry in list(queue):
            elapsed = current_time - entry.started_at
            craftable = int(elapsed / entry.duration)
            new_completed = min(craftable, entry.quantity) - entry.completed

            if new_completed > 0:
                # Craft completed items
                if entry.context:
                    recipe = self._recipes.get(entry.recipe_id)
                    if recipe:
                        # Generate results for completed items
                        quality = self._roll_quality(recipe, entry.context)
                        outputs = self._generate_outputs(recipe, new_completed, quality)
                        result = CraftingResult(
                            result_type=CraftingResultType.SUCCESS,
                            outputs=outputs,
                            quality=quality,
                        )
                        results.append(result)

                entry.completed += new_completed

            if entry.is_complete:
                queue.remove(entry)

        return results

    def get_queue(self, crafter_id: str) -> List[CraftingQueueEntry]:
        """Get crafting queue for a crafter."""
        return self._crafting_queue.get(crafter_id, [])

    def cancel_queue_entry(self, crafter_id: str, entry_id: UUID) -> bool:
        """Cancel a queued craft."""
        queue = self._crafting_queue.get(crafter_id, [])
        for entry in queue:
            if entry.entry_id == entry_id:
                queue.remove(entry)
                return True
        return False

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    def add_completion_callback(self, callback: CraftingCallback) -> None:
        """Add crafting completion callback."""
        self._completion_callbacks.append(callback)

    def remove_completion_callback(self, callback: CraftingCallback) -> None:
        """Remove crafting completion callback."""
        if callback in self._completion_callbacks:
            self._completion_callbacks.remove(callback)


# =============================================================================
# Recipe Builder (Fluent API)
# =============================================================================


class RecipeBuilder:
    """Fluent builder for recipes."""

    def __init__(self, recipe_id: str, name: str):
        self._recipe_id = recipe_id
        self._name = name
        self._category = "misc"
        self._ingredients: List[IngredientRequirement] = []
        self._outputs: List[RecipeOutput] = []
        self._station_required: Optional[str] = None
        self._station_level = 1
        self._skill_requirements: List[SkillRequirement] = []
        self._crafting_time = 1.0
        self._unlock_condition: Optional[Callable] = None
        self._description = ""
        self._is_discoverable = True
        self._discovered_by_default = True

    def category(self, category: str) -> RecipeBuilder:
        """Set recipe category."""
        self._category = category
        return self

    def ingredient(
        self,
        item_id: str,
        quantity: int = 1,
        consumed: bool = True,
    ) -> RecipeBuilder:
        """Add ingredient requirement."""
        self._ingredients.append(Ingredient(
            item_id=item_id,
            quantity=quantity,
            consumed=consumed,
        ))
        return self

    def ingredient_category(
        self,
        category: str,
        quantity: int = 1,
        consumed: bool = True,
    ) -> RecipeBuilder:
        """Add category ingredient requirement."""
        self._ingredients.append(IngredientCategory(
            category=category,
            quantity=quantity,
            consumed=consumed,
        ))
        return self

    def output(
        self,
        item_id: str,
        quantity: int = 1,
        bonus_chance: float = 0.0,
        max_bonus: int = 0,
    ) -> RecipeBuilder:
        """Add output item."""
        self._outputs.append(RecipeOutput(
            item_id=item_id,
            base_quantity=quantity,
            bonus_quantity_chance=bonus_chance,
            max_bonus_quantity=max_bonus,
        ))
        return self

    def station(self, station_id: str, level: int = 1) -> RecipeBuilder:
        """Require crafting station."""
        self._station_required = station_id
        self._station_level = level
        return self

    def skill(
        self,
        skill_id: str,
        level: int = 1,
        xp: int = 0,
    ) -> RecipeBuilder:
        """Add skill requirement."""
        self._skill_requirements.append(SkillRequirement(
            skill_id=skill_id,
            level=level,
            grants_xp=xp,
        ))
        return self

    def time(self, seconds: float) -> RecipeBuilder:
        """Set crafting time."""
        self._crafting_time = seconds
        return self

    def unlock_condition(self, condition: Callable[[Dict[str, Any]], bool]) -> RecipeBuilder:
        """Set unlock condition."""
        self._unlock_condition = condition
        return self

    def description(self, text: str) -> RecipeBuilder:
        """Set description."""
        self._description = text
        return self

    def discoverable(self, is_discoverable: bool, discovered_by_default: bool = True) -> RecipeBuilder:
        """Set discoverability."""
        self._is_discoverable = is_discoverable
        self._discovered_by_default = discovered_by_default
        return self

    def build(self) -> Recipe:
        """Build the recipe."""
        return Recipe(
            recipe_id=self._recipe_id,
            name=self._name,
            category=self._category,
            ingredients=tuple(self._ingredients),
            outputs=tuple(self._outputs),
            station_required=self._station_required,
            station_level=self._station_level,
            skill_requirements=tuple(self._skill_requirements),
            crafting_time=self._crafting_time,
            unlock_condition=self._unlock_condition,
            description=self._description,
            is_discoverable=self._is_discoverable,
            discovered_by_default=self._discovered_by_default,
        )


# =============================================================================
# Crafting Registry
# =============================================================================


class CraftingRegistry:
    """Global registry for recipes and stations."""

    _instance: Optional[CraftingRegistry] = None

    def __init__(self):
        self._recipes: Dict[str, Recipe] = {}
        self._stations: Dict[str, CraftingStation] = {}

    @classmethod
    def instance(cls) -> CraftingRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = CraftingRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset registry."""
        cls._instance = None

    def register_recipe(self, recipe: Recipe) -> None:
        """Register recipe."""
        if recipe.recipe_id in self._recipes:
            raise ValueError(f"Recipe '{recipe.recipe_id}' already registered")
        self._recipes[recipe.recipe_id] = recipe

    def register_station(self, station: CraftingStation) -> None:
        """Register station."""
        if station.station_id in self._stations:
            raise ValueError(f"Station '{station.station_id}' already registered")
        self._stations[station.station_id] = station

    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        """Get recipe."""
        return self._recipes.get(recipe_id)

    def get_station(self, station_id: str) -> Optional[CraftingStation]:
        """Get station."""
        return self._stations.get(station_id)

    def all_recipes(self) -> List[Recipe]:
        """Get all recipes."""
        return list(self._recipes.values())

    def all_stations(self) -> List[CraftingStation]:
        """Get all stations."""
        return list(self._stations.values())

    def clear(self) -> None:
        """Clear all registrations."""
        self._recipes.clear()
        self._stations.clear()


# =============================================================================
# Foundation Registry Decorators (T-GP-8.13)
# =============================================================================


def recipe(
    name: str,
    station: Optional[str] = None,
    skill_req: Optional[Dict[str, int]] = None,
    category: str = "misc",
    crafting_time: float = 1.0,
    description: str = "",
    outputs: Optional[List[Dict[str, Any]]] = None,
    ingredients: Optional[List[Dict[str, Any]]] = None,
) -> Callable[[T], T]:
    """
    Decorator for defining recipes with Foundation Registry integration.

    Registers the class with both CraftingRegistry and Foundation Registry,
    enabling runtime discovery via Registry.query(tag="recipe").

    Usage:
        @recipe(
            name="Iron Sword",
            station="forge",
            skill_req={"smithing": 5},
            category="weapons",
            ingredients=[{"item_id": "iron_ingot", "quantity": 3}],
            outputs=[{"item_id": "iron_sword", "quantity": 1}]
        )
        class IronSwordRecipe:
            pass

    Args:
        name: Display name of the recipe
        station: Required crafting station ID (None if no station required)
        skill_req: Dict of skill_id -> required level
        category: Recipe category for grouping
        crafting_time: Time in seconds to craft
        description: Recipe description
        outputs: List of output item dicts
        ingredients: List of ingredient requirement dicts

    Returns:
        Decorated class with recipe metadata
    """
    if not name:
        raise ValueError("Recipe name must be non-empty")

    def decorator(cls: T) -> T:
        # Generate recipe ID from class name if not provided
        recipe_id = getattr(cls, "recipe_id", None) or cls.__name__.lower()

        # Build skill requirements
        skill_requirements: Tuple[SkillRequirement, ...] = ()
        if skill_req:
            skill_requirements = tuple(
                SkillRequirement(skill_id=sid, level=lvl)
                for sid, lvl in skill_req.items()
            )

        # Build ingredient requirements
        ingredient_reqs: List[IngredientRequirement] = []
        if ingredients:
            for ing in ingredients:
                if "category" in ing:
                    ingredient_reqs.append(IngredientCategory(
                        category=ing["category"],
                        quantity=ing.get("quantity", 1),
                        consumed=ing.get("consumed", True),
                    ))
                else:
                    quality_min = None
                    if ing.get("quality_min"):
                        quality_min = CraftingQuality[ing["quality_min"]]
                    ingredient_reqs.append(Ingredient(
                        item_id=ing["item_id"],
                        quantity=ing.get("quantity", 1),
                        consumed=ing.get("consumed", True),
                        quality_min=quality_min,
                    ))

        # Build outputs
        output_list: List[RecipeOutput] = []
        if outputs:
            for out in outputs:
                output_list.append(RecipeOutput(
                    item_id=out["item_id"],
                    base_quantity=out.get("quantity", out.get("base_quantity", 1)),
                    bonus_quantity_chance=out.get("bonus_quantity_chance", 0.0),
                    max_bonus_quantity=out.get("max_bonus_quantity", 0),
                    quality_variance=out.get("quality_variance", True),
                ))

        # Create the Recipe object
        recipe_def = Recipe(
            recipe_id=recipe_id,
            name=name,
            category=category,
            ingredients=tuple(ingredient_reqs),
            outputs=tuple(output_list),
            station_required=station,
            skill_requirements=skill_requirements,
            crafting_time=crafting_time,
            description=description,
        )

        # Attach metadata to class
        cls._recipe = True  # type: ignore[attr-defined]
        cls._recipe_id = recipe_id  # type: ignore[attr-defined]
        cls._recipe_name = name  # type: ignore[attr-defined]
        cls._recipe_station = station  # type: ignore[attr-defined]
        cls._recipe_skill_req = skill_req or {}  # type: ignore[attr-defined]
        cls._recipe_category = category  # type: ignore[attr-defined]
        cls._recipe_definition = recipe_def  # type: ignore[attr-defined]

        # Add standard metadata
        if not hasattr(cls, "_tags"):
            cls._tags = {}  # type: ignore[attr-defined]
        cls._tags["recipe"] = True  # type: ignore[attr-defined]
        cls._tags["recipe_id"] = recipe_id  # type: ignore[attr-defined]
        if station:
            cls._tags["station"] = station  # type: ignore[attr-defined]

        if not hasattr(cls, "_registries"):
            cls._registries = set()  # type: ignore[attr-defined]
        cls._registries.add("economy")  # type: ignore[attr-defined]

        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()  # type: ignore[attr-defined]
        cls._applied_decorators.add("recipe")  # type: ignore[attr-defined]

        # Register with CraftingRegistry
        try:
            CraftingRegistry.instance().register_recipe(recipe_def)
        except ValueError:
            # Already registered, skip
            pass

        # Register with Foundation Registry
        registry_name = f"recipe.{recipe_id}"
        try:
            registry.register(cls, name=registry_name, track_instances=False)
            registry.add_tag(cls, "recipe")
            registry.set_metadata(cls, "recipe", True)
            registry.set_metadata(cls, "recipe_id", recipe_id)
            registry.set_metadata(cls, "name", name)
            registry.set_metadata(cls, "category", category)
            if station:
                registry.add_tag(cls, f"station:{station}")
                registry.set_metadata(cls, "station", station)
            if skill_req:
                for sid, lvl in skill_req.items():
                    registry.set_metadata(cls, f"skill_{sid}", lvl)
                    registry.add_tag(cls, f"skill:{sid}")
            registry.set_metadata(cls, "recipe_definition", recipe_def)
        except ValueError:
            # Already registered, skip
            pass

        return cls

    return decorator


def crafting_station(
    name: str,
    recipes: Optional[List[str]] = None,
    categories: Optional[Tuple[str, ...]] = None,
    level: int = 1,
    efficiency_bonus: float = 0.0,
    quality_bonus: float = 0.0,
) -> Callable[[T], T]:
    """
    Decorator for defining crafting stations with Foundation Registry integration.

    Registers the class with both CraftingRegistry and Foundation Registry,
    enabling runtime discovery via Registry.query(tag="crafting_station").

    Usage:
        @crafting_station(
            name="Blacksmith Forge",
            recipes=["iron_sword", "steel_armor"],
            categories=("weapons", "armor"),
            level=2,
            quality_bonus=0.1
        )
        class BlacksmithForge:
            pass

    Args:
        name: Display name of the station
        recipes: List of recipe IDs available at this station
        categories: Tuple of recipe categories this station can craft
        level: Station level (affects which recipes are available)
        efficiency_bonus: Crafting time reduction (0.0-1.0)
        quality_bonus: Quality improvement bonus (0.0-1.0)

    Returns:
        Decorated class with crafting station metadata
    """
    if not name:
        raise ValueError("Station name must be non-empty")

    def decorator(cls: T) -> T:
        # Generate station ID from class name or name
        station_id = getattr(cls, "station_id", None) or name.lower().replace(" ", "_")

        # Create the CraftingStation object
        station_def = CraftingStation(
            station_id=station_id,
            name=name,
            categories=categories or (),
            level=level,
            efficiency_bonus=efficiency_bonus,
            quality_bonus=quality_bonus,
        )

        # Attach metadata to class
        cls._crafting_station = True  # type: ignore[attr-defined]
        cls._station_id = station_id  # type: ignore[attr-defined]
        cls._station_name = name  # type: ignore[attr-defined]
        cls._station_recipes = recipes or []  # type: ignore[attr-defined]
        cls._station_categories = categories or ()  # type: ignore[attr-defined]
        cls._station_level = level  # type: ignore[attr-defined]
        cls._station_definition = station_def  # type: ignore[attr-defined]

        # Add standard metadata
        if not hasattr(cls, "_tags"):
            cls._tags = {}  # type: ignore[attr-defined]
        cls._tags["crafting_station"] = True  # type: ignore[attr-defined]
        cls._tags["station_id"] = station_id  # type: ignore[attr-defined]

        if not hasattr(cls, "_registries"):
            cls._registries = set()  # type: ignore[attr-defined]
        cls._registries.add("economy")  # type: ignore[attr-defined]

        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()  # type: ignore[attr-defined]
        cls._applied_decorators.add("crafting_station")  # type: ignore[attr-defined]

        # Register with CraftingRegistry
        try:
            CraftingRegistry.instance().register_station(station_def)
        except ValueError:
            # Already registered, skip
            pass

        # Register with Foundation Registry
        registry_name = f"crafting_station.{station_id}"
        try:
            registry.register(cls, name=registry_name, track_instances=False)
            registry.add_tag(cls, "crafting_station")
            registry.set_metadata(cls, "crafting_station", True)
            registry.set_metadata(cls, "station_id", station_id)
            registry.set_metadata(cls, "name", name)
            registry.set_metadata(cls, "level", level)
            registry.set_metadata(cls, "recipes", recipes or [])
            registry.set_metadata(cls, "categories", categories or ())
            registry.set_metadata(cls, "efficiency_bonus", efficiency_bonus)
            registry.set_metadata(cls, "quality_bonus", quality_bonus)
            registry.set_metadata(cls, "station_definition", station_def)
        except ValueError:
            # Already registered, skip
            pass

        return cls

    return decorator


def ingredient(
    item_type: str,
    quantity: int = 1,
    consumed: bool = True,
    quality_min: Optional[str] = None,
) -> Callable[[T], T]:
    """
    Decorator for marking ingredient requirements on a class.

    This decorator adds ingredient metadata to a class, typically used
    in conjunction with @recipe to define complex ingredient relationships.

    Usage:
        @ingredient(item_type="iron_ore", quantity=2)
        @ingredient(item_type="coal", quantity=1)
        @recipe(name="Iron Ingot", station="smelter")
        class IronIngotRecipe:
            pass

    Args:
        item_type: Item type/ID of the ingredient
        quantity: Required quantity
        consumed: Whether ingredient is consumed during crafting
        quality_min: Minimum quality required (e.g., "GOOD", "EXCELLENT")

    Returns:
        Decorated class with ingredient metadata
    """
    if not item_type:
        raise ValueError("Ingredient item_type must be non-empty")
    if quantity < 1:
        raise ValueError("Ingredient quantity must be at least 1")

    def decorator(cls: T) -> T:
        # Initialize ingredients list if not present
        if not hasattr(cls, "_ingredients"):
            cls._ingredients = []  # type: ignore[attr-defined]

        # Create ingredient dict
        ing_data: Dict[str, Any] = {
            "item_type": item_type,
            "quantity": quantity,
            "consumed": consumed,
        }
        if quality_min:
            ing_data["quality_min"] = quality_min

        # Append to ingredients list (prepend for decorator order)
        cls._ingredients.insert(0, ing_data)  # type: ignore[attr-defined]

        # Add standard metadata
        if not hasattr(cls, "_tags"):
            cls._tags = {}  # type: ignore[attr-defined]
        cls._tags["has_ingredients"] = True  # type: ignore[attr-defined]

        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()  # type: ignore[attr-defined]
        cls._applied_decorators.add("ingredient")  # type: ignore[attr-defined]

        return cls

    return decorator


def economy(
    economy_type: str,
    currency_id: Optional[str] = None,
    base_value: float = 0.0,
    tradeable: bool = True,
) -> Callable[[T], T]:
    """
    Decorator for marking economy-related classes (currency, trade).

    Registers the class with Foundation Registry for runtime discovery
    via Registry.query(tag="economy").

    Usage:
        @economy(economy_type="currency", currency_id="gold", base_value=1.0)
        class GoldCurrency:
            pass

        @economy(economy_type="trade", tradeable=True)
        class PlayerTrade:
            pass

    Args:
        economy_type: Type of economy element ("currency", "trade", "market", etc.)
        currency_id: Currency identifier (for currency types)
        base_value: Base value in the economy
        tradeable: Whether this can be traded

    Returns:
        Decorated class with economy metadata
    """
    if not economy_type:
        raise ValueError("Economy type must be non-empty")

    def decorator(cls: T) -> T:
        # Generate economy ID
        economy_id = getattr(cls, "economy_id", None) or cls.__name__.lower()

        # Attach metadata to class
        cls._economy = True  # type: ignore[attr-defined]
        cls._economy_type = economy_type  # type: ignore[attr-defined]
        cls._economy_id = economy_id  # type: ignore[attr-defined]
        cls._currency_id = currency_id  # type: ignore[attr-defined]
        cls._base_value = base_value  # type: ignore[attr-defined]
        cls._tradeable = tradeable  # type: ignore[attr-defined]

        # Add standard metadata
        if not hasattr(cls, "_tags"):
            cls._tags = {}  # type: ignore[attr-defined]
        cls._tags["economy"] = True  # type: ignore[attr-defined]
        cls._tags["economy_type"] = economy_type  # type: ignore[attr-defined]

        if not hasattr(cls, "_registries"):
            cls._registries = set()  # type: ignore[attr-defined]
        cls._registries.add("economy")  # type: ignore[attr-defined]

        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()  # type: ignore[attr-defined]
        cls._applied_decorators.add("economy")  # type: ignore[attr-defined]

        # Register with Foundation Registry
        registry_name = f"economy.{economy_type}.{economy_id}"
        try:
            registry.register(cls, name=registry_name, track_instances=False)
            registry.add_tag(cls, "economy")
            registry.add_tag(cls, f"economy:{economy_type}")
            registry.set_metadata(cls, "economy", True)
            registry.set_metadata(cls, "economy_type", economy_type)
            registry.set_metadata(cls, "economy_id", economy_id)
            if currency_id:
                registry.set_metadata(cls, "currency_id", currency_id)
            registry.set_metadata(cls, "base_value", base_value)
            registry.set_metadata(cls, "tradeable", tradeable)
        except ValueError:
            # Already registered, skip
            pass

        return cls

    return decorator


def crafting(
    quality_curve: Optional[str] = None,
    base_quality: str = "NORMAL",
    craftable_by: Optional[List[str]] = None,
    required_tools: Optional[List[str]] = None,
) -> Callable[[T], T]:
    """
    Decorator for marking craftable item classes.

    Registers the class with Foundation Registry for runtime discovery
    via Registry.query(tag="crafting").

    Usage:
        @crafting(
            quality_curve="linear",
            base_quality="GOOD",
            craftable_by=["blacksmith", "weaponsmith"],
            required_tools=["hammer", "anvil"]
        )
        class CraftableIronSword:
            pass

    Args:
        quality_curve: Quality calculation curve ("linear", "exponential", "step")
        base_quality: Base quality level for the item
        craftable_by: List of profession/skill IDs that can craft this
        required_tools: List of tool item IDs required for crafting

    Returns:
        Decorated class with crafting metadata
    """
    def decorator(cls: T) -> T:
        # Generate crafting ID
        crafting_id = getattr(cls, "crafting_id", None) or cls.__name__.lower()

        # Validate base_quality
        try:
            quality = CraftingQuality[base_quality]
        except KeyError:
            raise ValueError(f"Invalid base_quality: {base_quality}. "
                           f"Must be one of {[q.name for q in CraftingQuality]}")

        # Attach metadata to class
        cls._crafting = True  # type: ignore[attr-defined]
        cls._crafting_id = crafting_id  # type: ignore[attr-defined]
        cls._quality_curve = quality_curve  # type: ignore[attr-defined]
        cls._base_quality = base_quality  # type: ignore[attr-defined]
        cls._base_quality_value = quality  # type: ignore[attr-defined]
        cls._craftable_by = craftable_by or []  # type: ignore[attr-defined]
        cls._required_tools = required_tools or []  # type: ignore[attr-defined]

        # Add standard metadata
        if not hasattr(cls, "_tags"):
            cls._tags = {}  # type: ignore[attr-defined]
        cls._tags["crafting"] = True  # type: ignore[attr-defined]
        cls._tags["craftable"] = True  # type: ignore[attr-defined]

        if not hasattr(cls, "_registries"):
            cls._registries = set()  # type: ignore[attr-defined]
        cls._registries.add("economy")  # type: ignore[attr-defined]

        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()  # type: ignore[attr-defined]
        cls._applied_decorators.add("crafting")  # type: ignore[attr-defined]

        # Register with Foundation Registry
        registry_name = f"crafting.{crafting_id}"
        try:
            registry.register(cls, name=registry_name, track_instances=False)
            registry.add_tag(cls, "crafting")
            registry.add_tag(cls, "craftable")
            registry.set_metadata(cls, "crafting", True)
            registry.set_metadata(cls, "crafting_id", crafting_id)
            if quality_curve:
                registry.set_metadata(cls, "quality_curve", quality_curve)
                registry.add_tag(cls, f"quality_curve:{quality_curve}")
            registry.set_metadata(cls, "base_quality", base_quality)
            registry.set_metadata(cls, "craftable_by", craftable_by or [])
            registry.set_metadata(cls, "required_tools", required_tools or [])
        except ValueError:
            # Already registered, skip
            pass

        return cls

    return decorator


# =============================================================================
# Recipe Factory Methods
# =============================================================================


class RecipeFactory:
    """Factory for creating Recipe instances from registry."""

    @staticmethod
    def from_registry(name: str) -> Optional[Recipe]:
        """
        Create a Recipe instance from the registry by name.

        Args:
            name: Recipe ID or name to look up

        Returns:
            Recipe instance if found, None otherwise
        """
        # First try CraftingRegistry
        crafting_reg = CraftingRegistry.instance()
        recipe_def = crafting_reg.get_recipe(name)
        if recipe_def:
            return recipe_def

        # Try Foundation Registry query
        results = registry.query(tag="recipe", recipe_id=name)
        if results:
            cls = results[0]
            return getattr(cls, "_recipe_definition", None)

        # Try by name metadata
        results = registry.query(tag="recipe", name=name)
        if results:
            cls = results[0]
            return getattr(cls, "_recipe_definition", None)

        return None

    @staticmethod
    def from_class(cls: type) -> Optional[Recipe]:
        """
        Get Recipe instance from a decorated class.

        Args:
            cls: Class decorated with @recipe

        Returns:
            Recipe instance if class has recipe metadata, None otherwise
        """
        return getattr(cls, "_recipe_definition", None)


# Add convenience method to Recipe class
Recipe.from_registry = staticmethod(RecipeFactory.from_registry)  # type: ignore[attr-defined]


# =============================================================================
# Helper Functions for Registry Queries
# =============================================================================


def get_registered_recipes() -> List[type]:
    """
    Query Foundation Registry for all recipe-decorated classes.

    Returns:
        List of classes decorated with @recipe
    """
    return registry.types_with_decorator("recipe")


def get_registered_stations() -> List[type]:
    """
    Query Foundation Registry for all crafting_station-decorated classes.

    Returns:
        List of classes decorated with @crafting_station
    """
    return registry.types_with_decorator("crafting_station")


def get_recipes_for_station_from_registry(station_id: str) -> List[type]:
    """
    Query Foundation Registry for recipes available at a specific station.

    Args:
        station_id: Station identifier

    Returns:
        List of recipe classes for the station
    """
    return registry.query(tag="recipe", station=station_id)


def get_recipes_by_skill_from_registry(skill_id: str) -> List[type]:
    """
    Query Foundation Registry for recipes requiring a specific skill.

    Args:
        skill_id: Skill identifier

    Returns:
        List of recipe classes requiring the skill
    """
    return registry.query(tag=f"skill:{skill_id}")


def get_craftable_items() -> List[type]:
    """
    Query Foundation Registry for all craftable item classes.

    Returns:
        List of classes decorated with @crafting
    """
    return registry.types_with_decorator("crafting")


def get_economy_classes(economy_type: Optional[str] = None) -> List[type]:
    """
    Query Foundation Registry for economy-related classes.

    Args:
        economy_type: Optional filter by economy type

    Returns:
        List of classes decorated with @economy
    """
    if economy_type:
        return registry.query(tag=f"economy:{economy_type}")
    return registry.types_with_decorator("economy")
