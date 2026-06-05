# PHASE 1 ARCHITECTURE: Economy Module Testing

## Overview

Comprehensive test coverage for engine/gameplay/economy module (~4,217 lines across 4 files).

## Components Under Test

### 1. Inventory System (inventory.py, 1,225 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| ItemDefinition | Item template with validation | Schema validation, required fields |
| ItemInstance | Stack management | Merge, split, quantity bounds |
| InventorySlot | Slot filtering and locking | Filter rules, lock/unlock transitions |
| InventoryContainer | Full container operations | Weight limits, auto-stacking, transactions |

### 2. Crafting System (crafting.py, 947 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| Recipe | Ingredient/output specification | Requirement validation |
| IngredientCategory | Flexible matching | Category membership |
| QualityRoll | Quality determination | Skill bonuses, station bonuses |
| CraftingQueue | Timed crafting | Progress tracking, cancellation |
| RecipeBuilder | Fluent construction | Build validation |

### 3. Loot System (loot.py, 884 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| WeightedTable | Weighted random selection | Normalization, nesting |
| ConditionSystem | Drop conditions | Level, quest, flag, attribute, random |
| PitySystem | Bad luck protection | Threshold tracking, reset, boost |
| LuckBonuses | Luck stat effects | Multiplier application |
| LootRoller | Simulation/preview | Determinism, seeding |

### 4. Equipment System (equipment.py, 767 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| StatModifier | Flat/percent/multiplier | Stacking rules, ordering |
| ResistanceModifier | Resistance with caps | Cap enforcement |
| EquipmentContainer | Equip/unequip operations | Exclusivity, requirements |
| SetBonus | Set detection and application | Partial/full sets |
| DurabilitySystem | Wear and repair | Zero durability behavior |

## Architecture Decisions

### ADR-E1: Test Isolation via reset_instance()

All singleton managers must have reset_instance() called in test teardown. Tests must not depend on order.

### ADR-E2: Deterministic Randomness

LootRoller and QualityRoll tests must inject seeded RandomSource to ensure reproducibility.

### ADR-E3: Transaction Test Fixtures

Create a TransactionTestHarness that:
- Sets up container with known items
- Provides begin/commit/rollback helpers
- Validates invariants after each operation

### ADR-E4: Pity System State Inspection

PitySystem must expose counter state for test assertions (already has counters.get()).

## Test Structure

```
tests/
  economy/
    test_inventory_item.py          # ItemDefinition, ItemInstance
    test_inventory_slot.py          # InventorySlot
    test_inventory_container.py     # InventoryContainer
    test_inventory_transactions.py  # Transaction semantics
    test_crafting_recipe.py         # Recipe, IngredientCategory
    test_crafting_quality.py        # Quality rolling
    test_crafting_queue.py          # CraftingQueue
    test_loot_tables.py             # WeightedTable
    test_loot_conditions.py         # Condition evaluation
    test_loot_pity.py               # Pity system
    test_equipment_modifiers.py     # StatModifier, ResistanceModifier
    test_equipment_container.py     # EquipmentContainer
    test_equipment_sets.py          # SetBonus
```

## Dependencies

- pytest for test framework
- pytest-mock for mocking RandomSource
- No external runtime dependencies

## Risks

| Risk | Mitigation |
|------|------------|
| Flaky tests from randomness | Mandatory seeding for all random operations |
| Slow tests from complex setup | Factory functions for common fixtures |
| Missing edge cases | Coverage report + mutation testing |
