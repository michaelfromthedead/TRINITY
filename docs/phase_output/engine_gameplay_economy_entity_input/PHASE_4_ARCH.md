# PHASE 4 ARCHITECTURE: Cross-Module Integration Testing

## Overview

Integration testing between economy<->entity and entity<->input modules, validating cross-cutting concerns.

## Integration Points

### 1. Economy <-> Entity Integration

| Integration | Components | Test Focus |
|-------------|------------|------------|
| Inventory Ownership | InventoryContainer.owner_id <-> Actor.id | Owner resolution |
| Equipment Stats | EquipmentContainer <-> Character stats | Stat application |
| Item-Actor Binding | ItemInstance <-> Actor | Pickup/drop lifecycle |

### 2. Entity <-> Input Integration

| Integration | Components | Test Focus |
|-------------|------------|------------|
| Player Input | PlayerController <-> ActionMapper | Binding propagation |
| Character Movement | Character.movement_input <-> AxisMapper | Value translation |
| Pawn Possession | Pawn <-> Controller <-> Input devices | Ownership chain |

### 3. Cross-Cutting Concerns

| Concern | Components | Test Focus |
|---------|------------|------------|
| Singleton Reset | All managers | Test isolation |
| Event Propagation | Event systems | Cross-module events |
| Error Contexts | Error messages | Context preservation |
| Trinity Pattern | Decorators/Descriptors | Pattern consistency |

## Architecture Decisions

### ADR-INT-1: Integration Test Fixtures

Create IntegrationTestHarness with:
- Full module initialization
- Coordinated reset for all singletons
- Event listener tracking across modules

### ADR-INT-2: Ownership Chain Verification

Test complete ownership chains:
- Player -> PlayerController -> Pawn -> InventoryContainer -> ItemInstance
- Verify bidirectional resolution at each link

### ADR-INT-3: Input-to-Movement Pipeline

Test complete input processing pipeline:
- KeyDown(W) -> AxisMapper -> Vector2(0,1) -> Character.movement_input -> position change

### ADR-INT-4: Event Cross-Module Propagation

Test events that cross module boundaries:
- Equipment change -> stat recalculation -> movement speed change
- Item pickup -> inventory event -> potential possession change (full inventory)

### ADR-INT-5: Error Context Preservation

Verify error messages include context from all relevant modules:
- "Cannot equip item X to slot Y for actor Z: requirement not met (stat A < B)"

## Test Structure

```
tests/
  integration/
    test_economy_entity.py          # Inventory-actor binding
    test_entity_input.py            # Controller-input binding
    test_equipment_stats.py         # Equipment stat application
    test_input_movement.py          # Input-to-movement pipeline
    test_singleton_isolation.py     # Manager reset for test isolation
    test_event_propagation.py       # Cross-module events
    test_error_contexts.py          # Error message quality
```

## Test Scenarios

### Scenario 1: Character Picks Up Item

1. Character walks to item (input -> movement)
2. Pickup action triggered (input -> action)
3. Item added to inventory (economy)
4. If equipment, stat modifiers applied (economy -> entity)
5. Movement speed updated (entity)

### Scenario 2: Player Equips Weapon

1. Player opens inventory (UI, out of scope)
2. Equip action on weapon (input -> action)
3. Weapon equipped (economy)
4. Old weapon unequipped if slot occupied (economy)
5. Stats recalculated (economy -> entity)
6. Attack damage updated (entity)

### Scenario 3: AI Takes Over Pawn

1. Player unpossesses pawn (entity)
2. AI controller possesses pawn (entity)
3. Input bindings cleared (input)
4. AI blackboard initialized (entity)
5. AI begins behavior tree (entity)

### Scenario 4: Controller Swap During Combat

1. Player A possessing pawn (entity + input)
2. Player A disconnects (input device removal)
3. Pawn becomes unpossessed (entity)
4. AI takes over temporarily (entity)
5. Player A reconnects (input device addition)
6. Player A repossesses pawn (entity + input)

## Dependencies

- All three modules (economy, entity, input)
- pytest for test framework
- pytest-mock for event verification

## Risks

| Risk | Mitigation |
|------|------------|
| Test order dependence | Mandatory singleton reset in fixtures |
| Slow integration tests | Minimal setup, focused assertions |
| Flaky async events | Synchronous event dispatch in tests |
