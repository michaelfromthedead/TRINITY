#!/usr/bin/env python3
"""
Demo of Trinity Pattern Tier 37: PROCEDURAL decorators.

Demonstrates seeded generation, procedural content, and constraints.
"""

from trinity.decorators.procedural import constraint, procedural, seeded
from trinity.decorators.registry import inspect_decorated, registry


# Example 1: World generator with seeded randomness
@seeded(seed_source="world")
@procedural(cache=True)
class WorldGenerator:
    """Generate world terrain from a world seed."""

    seed: int
    noise_scale: float


# Example 2: Chunk generator with chunk-based seeding
@seeded(seed_source="chunk")
@procedural(cache=True)
class ChunkGenerator:
    """Generate individual chunks with deterministic seeding."""

    chunk_x: int
    chunk_z: int
    biome: str


# Example 3: Entity generator with constraints
def valid_health(entity):
    """Health must be positive."""
    return entity.get("health", 0) > 0


def valid_level(entity):
    """Level must be between 1 and 100."""
    level = entity.get("level", 1)
    return 1 <= level <= 100


@constraint(rules=[valid_health, valid_level])
@seeded(seed_source="entity")
@procedural(cache=False)
class EntityGenerator:
    """Generate entities with validation constraints."""

    entity_type: str
    level_range: tuple


# Example 4: Dungeon generator with custom validation
def validate_room_size(dungeon):
    """Rooms must have valid dimensions."""
    return dungeon.get("width", 0) > 0 and dungeon.get("height", 0) > 0


@constraint(rules=[validate_room_size])
@procedural(cache=True, validate=validate_room_size)
@seeded(seed_source="explicit")
class DungeonGenerator:
    """Generate dungeons with explicit seeds and validation."""

    num_rooms: int
    difficulty: str


# Example 5: Loot table with multiple constraints
def valid_rarity(loot):
    """Rarity must be valid."""
    return loot.get("rarity", "") in {"common", "rare", "epic", "legendary"}


def valid_quantity(loot):
    """Quantity must be positive."""
    return loot.get("quantity", 0) > 0


def valid_value(loot):
    """Value must be non-negative."""
    return loot.get("value", 0) >= 0


@constraint(rules=[valid_rarity, valid_quantity, valid_value])
@procedural(cache=True)
@seeded(seed_source="world")
class LootTableGenerator:
    """Generate loot drops with multiple validation rules."""

    enemy_level: int
    loot_tier: str


# Example 6: Simple procedural content without seeding
@procedural(cache=False)
class DynamicNameGenerator:
    """Generate random names without deterministic seeding."""

    name_pattern: str


def main():
    """Demonstrate the PROCEDURAL decorators."""
    print("=" * 70)
    print("Trinity Pattern - Tier 37: PROCEDURAL Decorators Demo")
    print("=" * 70)

    # Show world generator
    print("\n1. World Generator (Seeded + Procedural):")
    print(f"   Seeded: {WorldGenerator._seeded}")
    print(f"   Seed Source: {WorldGenerator._seed_source}")
    print(f"   Procedural: {WorldGenerator._procedural}")
    print(f"   Cache Enabled: {WorldGenerator._procedural_cache}")

    # Show chunk generator
    print("\n2. Chunk Generator:")
    print(f"   Seed Source: {ChunkGenerator._seed_source}")
    print(f"   Cache: {ChunkGenerator._procedural_cache}")

    # Show entity generator with constraints
    print("\n3. Entity Generator (With Constraints):")
    print(f"   Constraint Rules: {len(EntityGenerator._constraint_rules)}")
    print(f"   Seed Source: {EntityGenerator._seed_source}")
    print(f"   Cache: {EntityGenerator._procedural_cache}")
    info = inspect_decorated(EntityGenerator)
    print(f"   Applied Decorators: {', '.join(info.decorators)}")

    # Show dungeon generator
    print("\n4. Dungeon Generator:")
    print(f"   Seed Source: {DungeonGenerator._seed_source}")
    print(f"   Has Validator: {DungeonGenerator._procedural_validate is not None}")
    print(f"   Constraint Rules: {len(DungeonGenerator._constraint_rules)}")

    # Show loot table
    print("\n5. Loot Table Generator (Multiple Constraints):")
    print(f"   Constraint Rules: {len(LootTableGenerator._constraint_rules)}")
    print("   Validations:")
    print("     - Rarity check")
    print("     - Quantity check")
    print("     - Value check")

    # Test constraint rules
    print("\n6. Testing Constraint Rules:")
    test_loot = {"rarity": "epic", "quantity": 5, "value": 100}
    all_valid = all(rule(test_loot) for rule in LootTableGenerator._constraint_rules)
    print(f"   Test loot: {test_loot}")
    print(f"   All constraints pass: {all_valid}")

    invalid_loot = {"rarity": "invalid", "quantity": 0, "value": -10}
    results = [rule(invalid_loot) for rule in LootTableGenerator._constraint_rules]
    print(f"   Invalid loot: {invalid_loot}")
    print(f"   Constraint results: {results}")

    # Show dynamic generator
    print("\n7. Dynamic Name Generator (No Seeding):")
    print(f"   Procedural: {DynamicNameGenerator._procedural}")
    print(f"   Seeded: {hasattr(DynamicNameGenerator, '_seeded')}")
    print(f"   Cache: {DynamicNameGenerator._procedural_cache}")

    # Show registry stats
    print("\n8. Registry Statistics:")
    procedural_specs = registry.by_tier(37)  # Tier.PROCEDURAL = 37
    print(f"   PROCEDURAL decorators registered: {len(procedural_specs)}")
    print(f"   Decorator names: {', '.join(s.name for s in procedural_specs)}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
