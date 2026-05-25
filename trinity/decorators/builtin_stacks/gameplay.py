"""Gameplay domain composite stacks."""
from __future__ import annotations
from trinity.decorators.stacks import Stack, parameterized_stack, stack


@parameterized_stack
def full_destruction(
    health: float = 100.0,
    fracture_pattern: str = "voronoi",
    pool_size: int = 256,
) -> Stack:
    """Destructible entity with damage, fracture, and physics."""
    from trinity.decorators.destruction import damage_resistance, damage_type, destructible, fracture, physics_material
    from trinity.decorators.memory import pooled
    return stack(
        destructible(health=health),
        damage_type(id="physical"),
        damage_resistance(resistances={"physical": 0.0}),
        fracture(pattern=fracture_pattern),
        physics_material(),
        pooled(initial_size=pool_size),
    )


@parameterized_stack
def gameplay_ability(
    cooldown: float = 1.0,
    max_stacks: int = 1,
) -> Stack:
    """Ability with buff, cooldown-like tracking, and gameplay tags."""
    from trinity.decorators.data_flow import serializable
    from trinity.decorators.debug_safety import track_changes
    from trinity.decorators.gameplay import ability, buff, gameplay_tag
    return stack(
        ability(cooldown=cooldown),
        buff(max_stacks=max_stacks),
        gameplay_tag(hierarchy="ability"),
        serializable(format="binary"),
        track_changes,
    )


@parameterized_stack
def crafting_system(
    station_id: str = "workbench",
) -> Stack:
    """Crafting station with recipes, loot, and salvage."""
    from trinity.decorators.crafting import crafting_station, ingredient, loot_table, recipe, salvage_recipe
    from trinity.decorators.data_flow import serializable
    return stack(
        crafting_station(id=station_id),
        recipe(result="default"),
        ingredient(),
        loot_table(id="default", rolls=1),
        salvage_recipe(source="default"),
        serializable(format="binary"),
    )


__all__ = ["full_destruction", "gameplay_ability", "crafting_system"]
