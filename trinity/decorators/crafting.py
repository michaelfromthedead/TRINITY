"""
Trinity Pattern - Tier 52: CRAFTING Decorators

Crafting system decorators for recipes, ingredients, loot tables, and salvaging.
All decorators use the ops-based system via make_decorator().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

# Type variable for decorators
T = TypeVar("T")


# ============================================================================
# Configuration dataclasses
# ============================================================================


@dataclass(frozen=True)
class CraftingStationConfig:
    """Crafting station configuration."""

    id: str
    categories: tuple[str, ...]


@dataclass(frozen=True)
class RecipeConfig:
    """Recipe configuration."""

    result: str
    result_count: int
    station: Optional[str]
    category: str
    unlock_condition: Optional[Callable]


@dataclass(frozen=True)
class IngredientConfig:
    """Ingredient configuration."""

    categories: tuple[str, ...]
    properties: dict[str, Any]


@dataclass(frozen=True)
class LootTableConfig:
    """Loot table configuration."""

    id: str
    rolls: int


@dataclass(frozen=True)
class SalvageRecipeConfig:
    """Salvage recipe configuration."""

    source: str
    station: Optional[str]
    skill_requirement: Optional[str]


# ============================================================================
# Step builders
# ============================================================================


def _crafting_station_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @crafting_station decorator."""
    station_id = params.get("id", "")
    categories = params.get("categories", [])

    return [
        Step(Op.TAG, {"key": "crafting_station", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "crafting_station_config",
                "value": CraftingStationConfig(
                    id=station_id, categories=tuple(categories)
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "crafting"}),
    ]


def _recipe_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @recipe decorator."""
    result = params.get("result", "")
    result_count = params.get("result_count", 1)
    station = params.get("station")
    category = params.get("category", "misc")
    unlock_condition = params.get("unlock_condition")

    return [
        Step(Op.TAG, {"key": "recipe", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "recipe_config",
                "value": RecipeConfig(
                    result=result,
                    result_count=result_count,
                    station=station,
                    category=category,
                    unlock_condition=unlock_condition,
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "crafting"}),
    ]


def _ingredient_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @ingredient decorator."""
    categories = params.get("categories", [])
    properties = params.get("properties", {})

    return [
        Step(Op.TAG, {"key": "ingredient", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "ingredient_config",
                "value": IngredientConfig(
                    categories=tuple(categories), properties=dict(properties)
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "crafting"}),
    ]


def _loot_table_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @loot_table decorator."""
    loot_id = params.get("id", "")
    rolls = params.get("rolls", 1)

    return [
        Step(Op.TAG, {"key": "loot_table", "value": True}),
        Step(
            Op.TAG,
            {"key": "loot_table_config", "value": LootTableConfig(id=loot_id, rolls=rolls)},
        ),
        Step(Op.REGISTER, {"registry": "crafting"}),
    ]


def _salvage_recipe_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @salvage_recipe decorator."""
    source = params.get("source", "")
    station = params.get("station")
    skill_requirement = params.get("skill_requirement")

    return [
        Step(Op.TAG, {"key": "salvage_recipe", "value": True}),
        Step(
            Op.TAG,
            {
                "key": "salvage_recipe_config",
                "value": SalvageRecipeConfig(
                    source=source, station=station, skill_requirement=skill_requirement
                ),
            },
        ),
        Step(Op.REGISTER, {"registry": "crafting"}),
    ]


# ============================================================================
# Validators
# ============================================================================


def _validate_crafting_station_params(**kwargs: Any) -> None:
    """Validate @crafting_station parameters."""
    station_id = kwargs.get("id", "")
    if not station_id:
        raise ValueError("id must be a non-empty string")


def _validate_recipe_params(**kwargs: Any) -> None:
    """Validate @recipe parameters."""
    result = kwargs.get("result", "")
    if not result:
        raise ValueError("result must be a non-empty string")

    result_count = kwargs.get("result_count", 1)
    if not isinstance(result_count, int) or result_count <= 0:
        raise ValueError(f"result_count must be > 0, got {result_count}")


def _validate_loot_table_params(**kwargs: Any) -> None:
    """Validate @loot_table parameters."""
    loot_id = kwargs.get("id", "")
    if not loot_id:
        raise ValueError("id must be a non-empty string")

    rolls = kwargs.get("rolls", 1)
    if not isinstance(rolls, int) or rolls <= 0:
        raise ValueError(f"rolls must be > 0, got {rolls}")


def _validate_salvage_recipe_params(**kwargs: Any) -> None:
    """Validate @salvage_recipe parameters."""
    source = kwargs.get("source", "")
    if not source:
        raise ValueError("source must be a non-empty string")


# ============================================================================
# After-apply functions
# ============================================================================


def _crafting_station_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @crafting_station is applied."""
    station_id = params.get("id", "")
    categories = params.get("categories", [])

    obj._crafting_station = True
    obj._crafting_station_id = station_id
    obj._crafting_station_categories = tuple(categories)
    obj._crafting_station_config = CraftingStationConfig(
        id=station_id, categories=tuple(categories)
    )


def _recipe_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @recipe is applied."""
    result = params.get("result", "")
    result_count = params.get("result_count", 1)
    station = params.get("station")
    category = params.get("category", "misc")
    unlock_condition = params.get("unlock_condition")

    obj._recipe = True
    obj._recipe_result = result
    obj._recipe_result_count = result_count
    obj._recipe_station = station
    obj._recipe_category = category
    obj._recipe_unlock_condition = unlock_condition
    obj._recipe_config = RecipeConfig(
        result=result,
        result_count=result_count,
        station=station,
        category=category,
        unlock_condition=unlock_condition,
    )


def _ingredient_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @ingredient is applied."""
    categories = params.get("categories", [])
    properties = params.get("properties", {})

    obj._ingredient = True
    obj._ingredient_categories = tuple(categories)
    obj._ingredient_properties = dict(properties)
    obj._ingredient_config = IngredientConfig(
        categories=tuple(categories), properties=dict(properties)
    )


def _loot_table_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @loot_table is applied."""
    loot_id = params.get("id", "")
    rolls = params.get("rolls", 1)

    obj._loot_table = True
    obj._loot_table_id = loot_id
    obj._loot_table_rolls = rolls
    obj._loot_table_config = LootTableConfig(id=loot_id, rolls=rolls)


def _salvage_recipe_after_apply(obj: Any, params: dict[str, Any]) -> None:
    """Set attributes after @salvage_recipe is applied."""
    source = params.get("source", "")
    station = params.get("station")
    skill_requirement = params.get("skill_requirement")

    obj._salvage_recipe = True
    obj._salvage_recipe_source = source
    obj._salvage_recipe_station = station
    obj._salvage_recipe_skill_requirement = skill_requirement
    obj._salvage_recipe_config = SalvageRecipeConfig(
        source=source, station=station, skill_requirement=skill_requirement
    )


# ============================================================================
# Decorator creation
# ============================================================================

crafting_station = make_decorator(
    name="crafting_station",
    steps=_crafting_station_steps,
    validate=_validate_crafting_station_params,
    after_steps=_crafting_station_after_apply,
)

recipe = make_decorator(
    name="recipe",
    steps=_recipe_steps,
    validate=_validate_recipe_params,
    after_steps=_recipe_after_apply,
)

ingredient = make_decorator(
    name="ingredient",
    steps=_ingredient_steps,
    after_steps=_ingredient_after_apply,
)

loot_table = make_decorator(
    name="loot_table",
    steps=_loot_table_steps,
    validate=_validate_loot_table_params,
    after_steps=_loot_table_after_apply,
)

salvage_recipe = make_decorator(
    name="salvage_recipe",
    steps=_salvage_recipe_steps,
    validate=_validate_salvage_recipe_params,
    after_steps=_salvage_recipe_after_apply,
)


# ============================================================================
# Registry registration
# ============================================================================

_REGISTRY_ENTRIES = [
    ("crafting_station", crafting_station, ("class",)),
    ("recipe", recipe, ("class",)),
    ("ingredient", ingredient, ("class",)),
    ("loot_table", loot_table, ("class",)),
    ("salvage_recipe", salvage_recipe, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.CRAFTING,
            func=_func,
            unique=False,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.CRAFTING].append(_spec)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "crafting_station",
    "recipe",
    "ingredient",
    "loot_table",
    "salvage_recipe",
    "CraftingStationConfig",
    "RecipeConfig",
    "IngredientConfig",
    "LootTableConfig",
    "SalvageRecipeConfig",
]
