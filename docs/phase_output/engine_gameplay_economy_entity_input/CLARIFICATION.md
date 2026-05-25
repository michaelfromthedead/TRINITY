# CLARIFICATION: engine/gameplay/{economy,entity,input}

## Philosophical Framing

### Why These Modules Matter

The economy, entity, and input modules represent the **gameplay spine** of any interactive simulation. They answer three fundamental questions:

1. **Economy**: "What can the player own, create, and exchange?"
2. **Entity**: "What exists in the world and how does it behave?"
3. **Input**: "How does the player communicate intent to the simulation?"

These modules are deliberately decoupled yet composable. An actor can exist without an inventory. An inventory can exist without a controller. But when combined, they form the complete gameplay loop.

### Design Rationale

#### UE5-Style Actor Hierarchy

The entity module adopts Unreal Engine 5's Actor->Pawn->Character hierarchy because it solves the most common composition problems in games:

- **StaticActor**: Environmental objects (no tick overhead)
- **DynamicActor**: Physics-enabled objects (velocity, forces)
- **Pawn**: Possessable entities (controller abstraction)
- **Character**: Full movement suite (walk/run/jump/crouch)

This is not arbitrary inheritance — each level adds exactly the capabilities needed for that class of entity, and nothing more.

#### Component Composition Over Deep Inheritance

The ActorMeta metaclass and ComponentContainer pattern enable:
- Type-indexed component lookup (O(1) access by type)
- Runtime component addition/removal
- Inspection without knowledge of specific component types

This avoids the "diamond of death" problem while maintaining strong typing.

#### Input Abstraction Layers

The input module has four layers because each solves a different problem:

1. **Devices** — Hardware abstraction (what physical devices exist)
2. **Processing** — Signal conditioning (dead zones, curves, smoothing)
3. **Action Mapper** — Intent abstraction (pressed/held/combo triggers)
4. **Axis Mapper** — Movement abstraction (digital-to-analog conversion)

A gyroscope reading flows: Device -> Processor -> AxisMapper -> Character.move().

#### Economy Transactions

The transaction system (begin/commit/rollback) exists because:
- Inventory operations can fail mid-way (weight limit, slot full)
- Crafting consumes multiple items atomically (all or nothing)
- Trading requires two-phase commit semantics

Without transactions, partial failures corrupt inventory state.

### Trinity Pattern Integration

All three modules use Trinity Pattern constructs:

| Module | Metaclasses | Descriptors | Decorators |
|--------|-------------|-------------|------------|
| Economy | ItemMeta | — | @recipe, @loot_table |
| Entity | ActorMeta, ControllerMeta | PossessionDescriptor, LifecycleStateDescriptor | @prefab, @extends, @lifecycle_hook |
| Input | — | — | @input_action, @input_axis |

This consistency enables:
- Automatic registration in registries
- State validation at assignment time
- Metadata attachment for tooling

### Singleton Patterns

Managers (PrefabRegistry, PossessionManager, LifecycleManager, DeviceManager) use singletons with reset_instance() because:

1. Games have exactly one of these systems active
2. Tests need isolation between test cases
3. Hot reload needs controlled teardown

The reset_instance() contract: after reset, the next get_instance() returns a fresh manager.

### Pity System Philosophy

The loot pity system implements "bad luck protection" — a player-friendly mechanic that:

1. Tracks consecutive failures per rarity tier
2. Boosts weight when threshold is reached
3. Resets counter on successful drop

This ensures that even with fair RNG, no player experiences unbounded bad luck.

### Dead Zone Philosophy

Dead zone processing solves the fundamental problem that physical controllers are imperfect:

- **Axial dead zone**: Ignores small values per axis (WASD-like)
- **Radial dead zone**: Ignores small magnitudes (stick-like)
- **Cross dead zone**: Ignores small values near axes (prevents drift on diagonals)

The rescaling step is critical — without it, movement would "jump" from 0 to dead_zone threshold.

## Key Design Decisions

### Why Weight-Based Loot Tables

Weights are more intuitive than raw probabilities:
- "Epic is 5x rarer than Rare" vs "Epic is 0.02 and Rare is 0.10"
- Nested tables compose naturally (parent weight * child weight)
- Adding items doesn't require rebalancing others

### Why Fluent Builders

RecipeBuilder, PrefabBuilder, LootTableBuilder use fluent APIs because:
- Complex objects need many optional parameters
- Method chaining makes intent clear
- IDEs can autocomplete the next valid method
- Immutability until build() ensures validity

### Why Lifecycle State Machines

Entities need strict lifecycle management because:
- Resources must be acquired/released in order
- Systems depend on consistent state (physics needs position, rendering needs mesh)
- Debugging requires knowing current state
- Deferred transitions prevent mid-frame inconsistency

### Why Controller Abstraction

The Pawn/Controller split enables:
- Same pawn, different controllers (player vs AI)
- Same controller, different pawns (possession transfer)
- Unpossessed pawns (idle NPCs)
- Multiple pawns per player (RTS unit selection)

## Open Questions

1. Should prefab instantiation support async component loading?
2. Should input actions support input layering (UI eats input before game)?
3. Should economy support networked inventory with conflict resolution?
4. Should entity lifecycle support deterministic ordering for replay?
