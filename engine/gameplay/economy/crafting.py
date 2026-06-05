"""
Crafting System.

Provides recipes, ingredient requirements, crafting stations,
crafting process with quality variance, and skill requirements.
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
    Union,
)
from uuid import UUID, uuid4

from .constants import (
    CraftingQuality,
    EconomyEvent,
    ItemType,
    QUALITY_BASE_CHANCES,
    QUALITY_STAT_MULTIPLIERS,
    Rarity,
    SKILL_QUALITY_BONUS_PER_LEVEL,
)
from .inventory import ItemDefinition, ItemInstance, InventoryContainer


# =============================================================================
# Crafting Station
# =============================================================================


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


# =============================================================================
# Recipe Output
# =============================================================================


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
        """Serialize to dictionary."""
        from .inventory import ECONOMY_SCHEMA_VERSION

        ingredients = []
        for ing in self.ingredients:
            d = ing.to_dict()
            d["type"] = "category" if isinstance(ing, IngredientCategory) else "item"
            ingredients.append(d)

        return {
            "__version__": ECONOMY_SCHEMA_VERSION,
            "recipe_id": self.recipe_id,
            "name": self.name,
            "category": self.category,
            "ingredients": ingredients,
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
        ingredients = []
        for ing_data in data.get("ingredients", []):
            if ing_data.get("type") == "category":
                ingredients.append(IngredientCategory.from_dict(ing_data))
            else:
                ingredients.append(Ingredient.from_dict(ing_data))

        return cls(
            recipe_id=data["recipe_id"],
            name=data["name"],
            category=data.get("category", "misc"),
            ingredients=tuple(ingredients),
            outputs=tuple(RecipeOutput.from_dict(o) for o in data.get("outputs", [])),
            station_required=data.get("station_required"),
            station_level=data.get("station_level", 1),
            skill_requirements=tuple(
                SkillRequirement.from_dict(s) for s in data.get("skill_requirements", [])
            ),
            crafting_time=data.get("crafting_time", 1.0),
            description=data.get("description", ""),
            is_discoverable=data.get("is_discoverable", True),
            discovered_by_default=data.get("discovered_by_default", True),
        )

    @classmethod
    def from_registry(cls, id_or_name: str) -> Optional["Recipe"]:
        """
        Get a recipe from the CraftingRegistry by ID or name.

        Args:
            id_or_name: Recipe ID or display name

        Returns:
            Recipe if found, None otherwise
        """
        crafting_reg = CraftingRegistry.instance()
        # Try by ID first
        recipe = crafting_reg.get_recipe(id_or_name)
        if recipe:
            return recipe
        # Try by name (search all recipes)
        for r in crafting_reg._recipes.values():
            if r.name == id_or_name:
                return r
        return None


# =============================================================================
# Crafting Result
# =============================================================================


class CraftingResultType(Enum):
    """Type of crafting result."""
    SUCCESS = auto()
    FAILURE = auto()
    CRITICAL_SUCCESS = auto()
    PARTIAL = auto()


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
            "consumed_ingredients": list(self.consumed_ingredients),
            "returned_ingredients": [i.to_dict() for i in self.returned_ingredients],
            "skill_xp_gained": dict(self.skill_xp_gained),
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        item_registry: Optional[Any] = None,
    ) -> "CraftingResult":
        """
        Deserialize from dictionary.

        Args:
            data: Serialized data
            item_registry: Optional ItemRegistry for resolving item definitions
        """
        outputs = []
        for o in data.get("outputs", []):
            outputs.append(ItemInstance.from_dict(o, item_registry))

        returned = []
        for i in data.get("returned_ingredients", []):
            returned.append(ItemInstance.from_dict(i, item_registry))

        return cls(
            result_type=CraftingResultType[data["result_type"]],
            outputs=outputs,
            quality=CraftingQuality[data.get("quality", "NORMAL")],
            consumed_ingredients=[tuple(i) for i in data.get("consumed_ingredients", [])],
            returned_ingredients=returned,
            skill_xp_gained=dict(data.get("skill_xp_gained", {})),
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

        # If no ingredients required, allow unlimited crafting (bounded by a large number)
        # This handles "free" recipes that don't consume materials
        if min_count == float('inf'):
            return 999999

        return int(min_count)

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
# Recipe Factory
# =============================================================================


class RecipeFactory:
    """
    Factory for creating and registering recipes.

    Provides convenience methods for building and registering recipes
    with the global CraftingRegistry.
    """

    @staticmethod
    def create(
        recipe_id: str,
        name: str,
        ingredients: List[IngredientRequirement],
        outputs: List[RecipeOutput],
        **kwargs: Any,
    ) -> Recipe:
        """
        Create a recipe.

        Args:
            recipe_id: Unique recipe identifier
            name: Display name
            ingredients: List of ingredient requirements
            outputs: List of outputs
            **kwargs: Additional recipe parameters

        Returns:
            New Recipe instance
        """
        return Recipe(
            recipe_id=recipe_id,
            name=name,
            ingredients=tuple(ingredients),
            outputs=tuple(outputs),
            **kwargs,
        )

    @staticmethod
    def create_and_register(
        recipe_id: str,
        name: str,
        ingredients: List[IngredientRequirement],
        outputs: List[RecipeOutput],
        **kwargs: Any,
    ) -> Recipe:
        """
        Create and register a recipe.

        Args:
            recipe_id: Unique recipe identifier
            name: Display name
            ingredients: List of ingredient requirements
            outputs: List of outputs
            **kwargs: Additional recipe parameters

        Returns:
            Registered Recipe instance
        """
        r = RecipeFactory.create(recipe_id, name, ingredients, outputs, **kwargs)
        CraftingRegistry.instance().register_recipe(r)
        return r

    @staticmethod
    def from_registry(id_or_name: str) -> Optional[Recipe]:
        """
        Get a recipe from the registry by ID or name.

        Args:
            id_or_name: Recipe ID or display name

        Returns:
            Recipe if found, None otherwise
        """
        crafting_reg = CraftingRegistry.instance()
        # Try by ID first
        recipe = crafting_reg.get_recipe(id_or_name)
        if recipe:
            return recipe
        # Try by name (search all recipes)
        for r in crafting_reg._recipes.values():
            if r.name == id_or_name:
                return r
        return None

    @staticmethod
    def from_class(cls: type) -> Optional[Recipe]:
        """
        Get a Recipe from a class decorated with @recipe.

        Args:
            cls: Class that was decorated with @recipe

        Returns:
            Recipe if class has one, None otherwise
        """
        return getattr(cls, "_recipe_definition", None)


# =============================================================================
# Decorator-based Registration
# =============================================================================


# Storage for decorator-registered items
_registered_recipes: List[Recipe] = []
_registered_recipe_classes: List[type] = []
_registered_stations: List[CraftingStation] = []
_registered_station_classes: List[type] = []
_economy_classes: List[type] = []
_craftable_classes: List[type] = []


# Import Foundation Registry (lazy to avoid circular imports)
def _get_registry():
    """Get Foundation Registry, importing lazily."""
    try:
        from foundation import registry
        return registry
    except ImportError:
        return None


def _parse_ingredients(raw_ingredients: List[Any]) -> Tuple[IngredientRequirement, ...]:
    """Parse raw ingredient data into Ingredient/IngredientCategory objects."""
    result = []
    for item in raw_ingredients:
        if isinstance(item, (Ingredient, IngredientCategory)):
            result.append(item)
        elif isinstance(item, dict):
            if "category" in item:
                result.append(IngredientCategory(
                    category=item["category"],
                    quantity=item.get("quantity", 1),
                    consumed=item.get("consumed", True),
                ))
            else:
                quality_min = item.get("quality_min")
                if isinstance(quality_min, str):
                    quality_min = CraftingQuality[quality_min]
                result.append(Ingredient(
                    item_id=item.get("item_id", ""),
                    quantity=item.get("quantity", 1),
                    consumed=item.get("consumed", True),
                    quality_min=quality_min,
                ))
    return tuple(result)


def _parse_outputs(raw_outputs: List[Any]) -> Tuple[RecipeOutput, ...]:
    """Parse raw output data into RecipeOutput objects."""
    result = []
    for item in raw_outputs:
        if isinstance(item, RecipeOutput):
            result.append(item)
        elif isinstance(item, dict):
            result.append(RecipeOutput(
                item_id=item.get("item_id", ""),
                base_quantity=item.get("quantity", 1),
                bonus_quantity_chance=item.get("bonus_quantity_chance", 0.0),
                max_bonus_quantity=item.get("max_bonus_quantity", 0),
                quality_variance=item.get("quality_variance", True),
            ))
    return tuple(result)


def _parse_skill_requirements(skill_req: Optional[Dict[str, int]]) -> Tuple[SkillRequirement, ...]:
    """Parse skill requirements dict into SkillRequirement tuple."""
    if not skill_req:
        return ()
    return tuple(
        SkillRequirement(skill_id=skill_id, level=level)
        for skill_id, level in skill_req.items()
    )


def recipe(
    recipe_id: Optional[str] = None,
    name: Optional[str] = None,
    station: Optional[str] = None,
    skill_req: Optional[Dict[str, int]] = None,
    category: str = "misc",
    crafting_time: float = 1.0,
    description: str = "",
    ingredients: Optional[List[Any]] = None,
    outputs: Optional[List[Any]] = None,
    **kwargs: Any,
) -> Callable[[type], type]:
    """
    Decorator to register a class as a recipe definition.

    Integrates with Foundation Registry with 'recipe' tag and CraftingRegistry.

    Args:
        recipe_id: Unique recipe identifier (defaults to lowercase classname)
        name: Human-readable recipe name (required)
        station: Station required for crafting (e.g., "forge")
        skill_req: Dict of skill_id -> required_level
        category: Recipe category (e.g., "weapons", "armor")
        crafting_time: Time to craft in seconds
        description: Recipe description
        ingredients: List of ingredient dicts or Ingredient objects
        outputs: List of output dicts or RecipeOutput objects

    Usage:
        @recipe(name="Iron Sword", station="forge", skill_req={"smithing": 5})
        class IronSwordRecipe:
            pass
    """
    def decorator(cls: type) -> type:
        nonlocal name

        # Validate name
        if name == "":
            raise ValueError("Recipe name must be non-empty")

        # Generate IDs
        rid = recipe_id or cls.__name__.lower()
        recipe_name = name or getattr(cls, "name", rid)

        # Get class-level attributes (can be overridden by decorator args)
        cls_ingredients = getattr(cls, "ingredients", [])
        cls_outputs = getattr(cls, "outputs", [])
        cls_category = getattr(cls, "category", category)
        cls_station = getattr(cls, "station_required", station)
        cls_skill_req = getattr(cls, "skill_requirements", [])
        cls_crafting_time = getattr(cls, "crafting_time", crafting_time)
        cls_description = getattr(cls, "description", description)

        # Use decorator args if provided, else class attributes
        final_ingredients = ingredients if ingredients is not None else cls_ingredients
        final_outputs = outputs if outputs is not None else cls_outputs
        final_skill_req = skill_req if skill_req is not None else None

        # Parse ingredients and outputs
        parsed_ingredients = _parse_ingredients(final_ingredients)
        parsed_outputs = _parse_outputs(final_outputs)

        # Build skill requirements
        if final_skill_req:
            skill_requirements = _parse_skill_requirements(final_skill_req)
        elif cls_skill_req:
            skill_requirements = tuple(cls_skill_req)
        else:
            skill_requirements = ()

        # Create Recipe object
        r = Recipe(
            recipe_id=rid,
            name=recipe_name,
            ingredients=parsed_ingredients,
            outputs=parsed_outputs,
            category=cls_category,
            station_required=cls_station,
            station_level=getattr(cls, "station_level", 1),
            skill_requirements=skill_requirements,
            crafting_time=cls_crafting_time,
            description=cls_description,
            is_discoverable=getattr(cls, "is_discoverable", True),
            discovered_by_default=getattr(cls, "discovered_by_default", True),
        )

        # Set class metadata
        cls._recipe = True
        cls._recipe_id = rid
        cls._recipe_name = recipe_name
        cls._recipe_station = cls_station
        cls._recipe_skill_req = skill_req
        cls._recipe_category = cls_category
        cls._recipe_definition = r

        # Track applied decorators
        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()
        cls._applied_decorators.add("recipe")

        # Register in module storage
        _registered_recipes.append(r)
        _registered_recipe_classes.append(cls)

        # Register with CraftingRegistry (handle duplicates by using unique ID)
        crafting_reg = CraftingRegistry.instance()
        final_rid = rid
        counter = 1
        while final_rid in crafting_reg._recipes:
            final_rid = f"{rid}_{counter}"
            counter += 1
        if final_rid != rid:
            # Update recipe with unique ID
            r = Recipe(
                recipe_id=final_rid,
                name=r.name,
                ingredients=r.ingredients,
                outputs=r.outputs,
                category=r.category,
                station_required=r.station_required,
                station_level=r.station_level,
                skill_requirements=r.skill_requirements,
                crafting_time=r.crafting_time,
                description=r.description,
                is_discoverable=r.is_discoverable,
                discovered_by_default=r.discovered_by_default,
            )
            cls._recipe_id = final_rid
            cls._recipe_definition = r
            _registered_recipes[-1] = r  # Update stored recipe
        crafting_reg.register_recipe(r)

        # Register with Foundation Registry
        reg = _get_registry()
        if reg is not None:
            # Handle duplicate class names by using unique name
            try:
                reg.register(cls)
            except ValueError:
                # Already registered - generate unique name
                unique_name = f"{cls.__module__}.{cls.__name__}_{final_rid}"
                try:
                    reg.register(cls, name=unique_name)
                except ValueError:
                    # Already registered with this unique name too, skip
                    pass
            if reg.is_registered(cls):
                reg.add_tag(cls, "recipe")
                reg.set_metadata(cls, "recipe_id", final_rid)
                reg.set_metadata(cls, "name", recipe_name)
                reg.set_metadata(cls, "category", cls_category)
                if cls_station:
                    reg.add_tag(cls, f"station:{cls_station}")
                    reg.set_metadata(cls, "station", cls_station)
                if skill_req:
                    for skill_id in skill_req:
                        reg.add_tag(cls, f"skill:{skill_id}")
                    reg.set_metadata(cls, "skill_req", skill_req)

        return cls

    return decorator


def crafting_station(
    station_id: Optional[str] = None,
    name: Optional[str] = None,
    recipes: Optional[List[str]] = None,
    categories: Optional[Tuple[str, ...]] = None,
    level: int = 1,
    efficiency_bonus: float = 0.0,
    quality_bonus: float = 0.0,
    **kwargs: Any,
) -> Callable[[type], type]:
    """
    Decorator to register a class as a crafting station.

    Integrates with Foundation Registry with 'crafting_station' tag.

    Args:
        station_id: Unique station ID (defaults from name)
        name: Human-readable station name
        recipes: List of recipe IDs this station can craft
        categories: Recipe categories this station supports
        level: Station level (higher = more advanced recipes)
        efficiency_bonus: Time reduction multiplier
        quality_bonus: Quality improvement bonus

    Usage:
        @crafting_station(name="Blacksmith Forge", level=2)
        class BlacksmithForge:
            pass
    """
    def decorator(cls: type) -> type:
        nonlocal name, station_id

        # Validate name
        if name == "":
            raise ValueError("Station name must be non-empty")

        # Get names
        station_name = name or getattr(cls, "name", cls.__name__)
        # Generate ID from name if not provided
        sid = station_id or station_name.lower().replace(" ", "_")

        # Get attributes
        station_recipes = recipes or getattr(cls, "recipes", [])
        station_categories = categories or getattr(cls, "categories", ())
        station_level = level if level != 1 else getattr(cls, "level", 1)
        station_efficiency = efficiency_bonus if efficiency_bonus != 0.0 else getattr(cls, "efficiency_bonus", 0.0)
        station_quality = quality_bonus if quality_bonus != 0.0 else getattr(cls, "quality_bonus", 0.0)

        # Create CraftingStation object
        station = CraftingStation(
            station_id=sid,
            name=station_name,
            categories=tuple(station_categories) if station_categories else (),
            level=station_level,
            efficiency_bonus=station_efficiency,
            quality_bonus=station_quality,
        )

        # Set class metadata
        cls._crafting_station = True
        cls._station_id = sid
        cls._station_name = station_name
        cls._station_recipes = station_recipes
        cls._station_categories = station_categories
        cls._station_level = station_level
        cls._station_definition = station

        # Track applied decorators
        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()
        cls._applied_decorators.add("crafting_station")

        # Register in module storage
        _registered_stations.append(station)
        _registered_station_classes.append(cls)

        # Register with CraftingRegistry
        CraftingRegistry.instance().register_station(station)

        # Register with Foundation Registry
        reg = _get_registry()
        if reg is not None:
            reg.register(cls)
            reg.add_tag(cls, "crafting_station")
            reg.set_metadata(cls, "station_id", sid)
            reg.set_metadata(cls, "name", station_name)
            reg.set_metadata(cls, "level", station_level)
            if station_recipes:
                reg.set_metadata(cls, "recipes", station_recipes)

        return cls

    return decorator


def ingredient(
    item_type: str = "",
    quantity: int = 1,
    consumed: bool = True,
    quality_min: Optional[str] = None,
    **kwargs: Any,
) -> Callable[[type], type]:
    """
    Decorator to add ingredient metadata to a class.

    Can be stacked to define multiple ingredients.

    Args:
        item_type: Item ID required
        quantity: Quantity required (must be >= 1)
        consumed: Whether consumed on craft
        quality_min: Minimum quality required

    Usage:
        @ingredient(item_type="iron_ore", quantity=3)
        @ingredient(item_type="coal", quantity=1)
        class SmeltingRecipe:
            pass
    """
    # Validate inputs
    if item_type == "":
        raise ValueError("Ingredient item_type must be non-empty")
    if quantity < 1:
        raise ValueError("Ingredient quantity must be at least 1")

    def decorator(cls: type) -> type:
        # Initialize ingredients list if not present
        if not hasattr(cls, "_ingredients"):
            cls._ingredients = []

        # Insert at the beginning (first decorator in stack is first in list)
        cls._ingredients.insert(0, {
            "item_type": item_type,
            "quantity": quantity,
            "consumed": consumed,
            "quality_min": quality_min,
        })

        return cls

    return decorator


def economy(
    economy_type: str = "",
    currency_id: Optional[str] = None,
    base_value: float = 0.0,
    tradeable: bool = True,
    **kwargs: Any,
) -> Callable[[type], type]:
    """
    Decorator to mark a class as part of the economy system.

    Integrates with Foundation Registry with 'economy' tag.

    Args:
        economy_type: Type of economy component (e.g., "currency", "trade")
        currency_id: Unique currency identifier (for currency types)
        base_value: Base value for trading
        tradeable: Whether this can be traded

    Usage:
        @economy(economy_type="currency", currency_id="gold")
        class GoldCurrency:
            pass
    """
    # Validate
    if economy_type == "":
        raise ValueError("Economy type must be non-empty")

    def decorator(cls: type) -> type:
        # Set class metadata
        cls._economy = True
        cls._economy_type = economy_type
        if currency_id:
            cls._currency_id = currency_id
        if base_value:
            cls._base_value = base_value
        cls._tradeable = tradeable

        # Store additional kwargs
        for key, value in kwargs.items():
            setattr(cls, f"_{key}", value)

        # Track applied decorators
        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()
        cls._applied_decorators.add("economy")

        # Register in module storage
        _economy_classes.append(cls)

        # Register with Foundation Registry
        reg = _get_registry()
        if reg is not None:
            reg.register(cls)
            reg.add_tag(cls, "economy")
            reg.add_tag(cls, f"economy:{economy_type}")
            reg.set_metadata(cls, "economy_type", economy_type)
            if currency_id:
                reg.set_metadata(cls, "currency_id", currency_id)

        return cls

    return decorator


# Valid quality values for validation
_VALID_QUALITIES = {"POOR", "NORMAL", "GOOD", "EXCELLENT", "MASTERWORK", "LEGENDARY"}


def crafting(
    quality_curve: str = "linear",
    base_quality: str = "NORMAL",
    craftable_by: Optional[List[str]] = None,
    required_tools: Optional[List[str]] = None,
    **kwargs: Any,
) -> Callable[[type], type]:
    """
    Decorator to mark a class as a craftable item.

    Integrates with Foundation Registry with 'crafting' tag.

    Args:
        quality_curve: Quality calculation curve (linear, exponential, step)
        base_quality: Base quality of crafted item
        craftable_by: List of professions that can craft this
        required_tools: List of tools required to craft

    Usage:
        @crafting(quality_curve="linear", base_quality="GOOD")
        class CraftableArmor:
            pass
    """
    # Validate base_quality
    if base_quality not in _VALID_QUALITIES:
        raise ValueError(f"Invalid base_quality: {base_quality}. Must be one of {_VALID_QUALITIES}")

    def decorator(cls: type) -> type:
        # Set class metadata
        cls._crafting = True
        cls._quality_curve = quality_curve
        cls._base_quality = base_quality
        cls._base_quality_value = CraftingQuality[base_quality]
        cls._craftable_by = craftable_by or []
        cls._required_tools = required_tools or []

        # Track applied decorators
        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = set()
        cls._applied_decorators.add("crafting")

        # Register in module storage
        _craftable_classes.append(cls)

        # Register with Foundation Registry
        reg = _get_registry()
        if reg is not None:
            reg.register(cls)
            reg.add_tag(cls, "crafting")
            reg.add_tag(cls, f"quality_curve:{quality_curve}")
            reg.set_metadata(cls, "quality_curve", quality_curve)
            reg.set_metadata(cls, "base_quality", base_quality)

        return cls

    return decorator


def get_registered_recipes() -> List[type]:
    """Get all recipe classes registered via decorators."""
    return list(_registered_recipe_classes)


def get_registered_stations() -> List[type]:
    """Get all station classes registered via decorators."""
    return list(_registered_station_classes)


def get_economy_classes(economy_type: Optional[str] = None) -> List[type]:
    """
    Get all classes decorated with @economy.

    Args:
        economy_type: Filter by economy type (optional)

    Returns:
        List of economy classes
    """
    if economy_type is None:
        return list(_economy_classes)
    return [
        cls for cls in _economy_classes
        if getattr(cls, "_economy_type", None) == economy_type
    ]


def get_recipes_for_station_from_registry(station_id: str) -> List[type]:
    """Get all decorator-registered recipe classes for a station."""
    return [
        cls for cls in _registered_recipe_classes
        if getattr(cls, "_recipe_station", None) == station_id
    ]


def get_recipes_by_skill_from_registry(skill_id: str) -> List[type]:
    """Get all decorator-registered recipe classes requiring a skill."""
    result = []
    for cls in _registered_recipe_classes:
        skill_req = getattr(cls, "_recipe_skill_req", None)
        if skill_req and skill_id in skill_req:
            result.append(cls)
    return result


def get_craftable_items() -> List[type]:
    """Get all craftable item classes."""
    return list(_craftable_classes)


def clear_registered() -> None:
    """Clear all decorator-registered recipes and stations."""
    _registered_recipes.clear()
    _registered_recipe_classes.clear()
    _registered_stations.clear()
    _registered_station_classes.clear()
    _economy_classes.clear()
    _craftable_classes.clear()


# =============================================================================
# Serialization Helpers
# =============================================================================


def ingredient_from_dict(data: Dict[str, Any]) -> IngredientRequirement:
    """
    Create an ingredient requirement from a dictionary.

    Args:
        data: Dictionary containing ingredient data.
              Must have either 'item_id' (for Ingredient) or 'category' (for IngredientCategory).

    Returns:
        Ingredient or IngredientCategory instance

    Raises:
        ValueError: If neither item_id nor category is provided
    """
    if "item_id" in data:
        return Ingredient(
            item_id=data["item_id"],
            quantity=data.get("quantity", 1),
            consumed=data.get("consumed", True),
            quality_min=CraftingQuality(data["quality_min"]) if "quality_min" in data else None,
        )
    elif "category" in data:
        return IngredientCategory(
            category=data["category"],
            quantity=data.get("quantity", 1),
            consumed=data.get("consumed", True),
        )
    else:
        raise ValueError(f"Cannot determine ingredient type from data: {data}")


def recipe_output_from_dict(data: Dict[str, Any]) -> RecipeOutput:
    """
    Create a recipe output from a dictionary.

    Args:
        data: Dictionary containing output data

    Returns:
        RecipeOutput instance
    """
    return RecipeOutput(
        item_id=data["item_id"],
        base_quantity=data.get("base_quantity", data.get("quantity", 1)),
        bonus_quantity_chance=data.get("bonus_quantity_chance", 0.0),
        max_bonus_quantity=data.get("max_bonus_quantity", 0),
        quality_variance=data.get("quality_variance", True),
    )


def skill_requirement_from_dict(data: Dict[str, Any]) -> SkillRequirement:
    """
    Create a skill requirement from a dictionary.

    Args:
        data: Dictionary containing skill requirement data

    Returns:
        SkillRequirement instance
    """
    return SkillRequirement(
        skill_id=data["skill_id"],
        level=data.get("level", 1),
        grants_xp=data.get("grants_xp", 0),
    )


def recipe_from_dict(data: Dict[str, Any]) -> Recipe:
    """
    Create a recipe from a dictionary.

    Args:
        data: Dictionary containing recipe data

    Returns:
        Recipe instance
    """
    ingredients = tuple(
        ingredient_from_dict(i) for i in data.get("ingredients", [])
    )
    outputs = tuple(
        recipe_output_from_dict(o) for o in data.get("outputs", [])
    )
    skill_reqs = tuple(
        skill_requirement_from_dict(s) for s in data.get("skill_requirements", [])
    )

    return Recipe(
        recipe_id=data["recipe_id"],
        name=data.get("name", data["recipe_id"]),
        category=data.get("category", "misc"),
        ingredients=ingredients,
        outputs=outputs,
        station_required=data.get("station_required"),
        station_level=data.get("station_level", 1),
        skill_requirements=skill_reqs,
        crafting_time=data.get("crafting_time", 1.0),
        description=data.get("description", ""),
        is_discoverable=data.get("is_discoverable", True),
        discovered_by_default=data.get("discovered_by_default", True),
    )


def crafting_station_from_dict(data: Dict[str, Any]) -> CraftingStation:
    """
    Create a crafting station from a dictionary.

    Args:
        data: Dictionary containing station data

    Returns:
        CraftingStation instance
    """
    categories = data.get("categories", ())
    if isinstance(categories, list):
        categories = tuple(categories)

    return CraftingStation(
        station_id=data["station_id"],
        name=data.get("name", data["station_id"]),
        categories=categories,
        level=data.get("level", 1),
        efficiency_bonus=data.get("efficiency_bonus", 0.0),
        quality_bonus=data.get("quality_bonus", 0.0),
    )
