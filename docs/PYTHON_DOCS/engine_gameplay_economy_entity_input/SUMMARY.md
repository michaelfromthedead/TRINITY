# SUMMARY: engine/gameplay/{economy,entity,input}

---

## Metrics

| Metric | Value |
|--------|-------|
| **Total Lines** | 12,699 |
| **Classification** | REAL |
| **Confidence** | 99% |
| **Primary Files** | 13 |
| **Constants Files** | 3 |
| **Modules** | 3 (economy, entity, input) |

### Per-Module Breakdown

| Module | Lines | Files | Classification |
|--------|-------|-------|----------------|
| economy | 4,217 | 5 (inventory, crafting, loot, equipment, constants) | REAL |
| entity | 4,418 | 6 (actor, possession, prefab, lifecycle, __init__, constants) | REAL |
| input | 4,064 | 6 (devices, action_mapper, axis_mapper, processing, __init__, constants) | REAL |

---

## Algorithm Inventory

| Algorithm | File | Lines | Status | Description |
|-----------|------|-------|--------|-------------|
| Stack Merging | inventory.py | 182-205 | REAL | Merge item stacks with overflow handling |
| Stack Splitting | inventory.py | 156-180 | REAL | Split stacks with quantity validation |
| Auto-Stacking | inventory.py | 400-450 | REAL | Automatically merge stackable items |
| Container Sorting | inventory.py | 500-550 | REAL | Sort by type/rarity/name/weight |
| Quality Rolling | crafting.py | 573-597 | REAL | Weighted quality selection with skill bonus |
| Crafting Queue | crafting.py | 638-715 | REAL | Timed crafting with progress tracking |
| Ingredient Matching | crafting.py | 380-413 | REAL | Category-based flexible ingredients |
| Weighted Selection | loot.py | 524-570 | REAL | Weighted random with nested tables |
| Pity System | loot.py | 306-340 | REAL | Guaranteed rare drops after N failures |
| Luck Bonus | loot.py | 440-480 | REAL | Luck stat affects drop rates |
| Condition Evaluation | loot.py | 100-198 | REAL | Level, quest, flag, attribute conditions |
| Actor Type Assignment | actor.py | 95-141 | REAL | Metaclass assigns unique type IDs |
| Component Collection | actor.py | 143-158 | REAL | Collect component declarations from class |
| Character Movement | actor.py | 1126-1147 | REAL | Walk/run movement with input |
| Jump Physics | actor.py | 1063-1080 | REAL | Ground check, velocity application |
| AI Movement | possession.py | 741-779 | REAL | Movement-to-location with acceptance radius |
| Possession Binding | possession.py | 854-885 | REAL | Controller/pawn binding with callbacks |
| Prefab Inheritance | prefab.py | 280-330 | REAL | Recursive inheritance resolution |
| Lifecycle State Machine | lifecycle.py | 200-300 | REAL | State transitions with validation |
| Radial Dead Zone | processing.py | 71-101 | REAL | 2D dead zone with smooth rescaling |
| Cross Dead Zone | processing.py | 104-131 | REAL | Axis-aligned cross dead zone |
| S-Curve Response | processing.py | 195-222 | REAL | Tanh-based S-curve with rescaling |
| Power Curve | processing.py | 160-174 | REAL | Exponent-based response curve |
| Hold Trigger | action_mapper.py | 173-228 | REAL | Time-based hold detection |
| Tap Trigger | action_mapper.py | 236-294 | REAL | Quick press/release detection |
| DoubleTap Trigger | action_mapper.py | 297-370 | REAL | Double quick press detection |
| Combo Trigger | action_mapper.py | 400-500 | REAL | Multi-input combo sequences |
| Motion Smoothing | devices.py | 800-850 | REAL | Gyroscope/accelerometer smoothing |
| Vector2 Dead Zone | axis_mapper.py | 300-350 | REAL | 2D axis with radial normalization |

---

## File Summary

### economy/

| File | Lines | Purpose |
|------|-------|---------|
| inventory.py | 1,225 | ItemDefinition, ItemInstance, InventoryContainer |
| crafting.py | 947 | Recipe, CraftingSystem, CraftingQueue |
| loot.py | 884 | LootTable, LootRoller, PityTracker |
| equipment.py | 767 | EquipmentStats, EquipmentContainer, SetBonus |
| constants.py | 394 | Enums, thresholds, default values |

### entity/

| File | Lines | Purpose |
|------|-------|---------|
| actor.py | 1,167 | ActorMeta, Actor, Pawn, Character |
| possession.py | 899 | ControllerMeta, Controller, AI/PlayerController |
| prefab.py | 774 | PrefabRegistry, PrefabBuilder, inheritance |
| lifecycle.py | 630 | LifecycleManager, state transitions, callbacks |
| __init__.py | 687 | Alternative/simplified implementations |
| constants.py | 261 | Enums, entity limits, defaults |

### input/

| File | Lines | Purpose |
|------|-------|---------|
| devices.py | 1,503 | Device types, DeviceManager, hot-plug |
| action_mapper.py | 834 | TriggerEvaluators, ActionMapper |
| axis_mapper.py | 782 | AxisMapper, Vector2Mapper, digital-to-analog |
| processing.py | 747 | Dead zones, response curves, smoothing |
| constants.py | 198 | Thresholds, defaults, key codes |

---

## Key Patterns Found

1. **Trinity Pattern**: Metaclass + Descriptor + Decorator across all modules
2. **Fluent Builders**: RecipeBuilder, LootTableBuilder, PrefabBuilder
3. **Testable Singletons**: reset_instance() for test isolation
4. **Protocol Abstractions**: RandomSource for deterministic testing
5. **Event Systems**: Callbacks for extensibility
6. **Transaction Support**: begin/commit/rollback for atomic operations
