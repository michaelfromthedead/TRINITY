# PHASE 4 TODO: Cross-Module Integration Testing

## Summary

Integration testing between economy, entity, and input modules.

---

## T-INT-4.1: Inventory-Actor Binding Tests

**File**: `tests/integration/test_economy_entity.py`

### Tasks

- [ ] Test InventoryContainer.owner_id resolves to correct Actor
- [ ] Test actor destruction clears inventory ownership
- [ ] Test inventory transfer updates owner_id
- [ ] Test multiple inventories per actor (main, bank, etc.)
- [ ] Test item pickup adds to actor's inventory
- [ ] Test item drop removes from actor's inventory

### Acceptance Criteria

- All tests pass with `uv run pytest tests/integration/test_economy_entity.py`
- Ownership chain: Actor.id == InventoryContainer.owner_id
- No orphaned inventories after actor destruction

---

## T-INT-4.2: Equipment Stat Application Tests

**File**: `tests/integration/test_equipment_stats.py`

### Tasks

- [ ] Test equipping item applies StatModifiers to Character
- [ ] Test unequipping item removes StatModifiers
- [ ] Test modifier stacking order across equipment pieces
- [ ] Test set bonus triggers at threshold
- [ ] Test resistance modifiers affect damage calculation
- [ ] Test equipment-induced max_walk_speed change

### Acceptance Criteria

- All tests pass
- Stats: final_stat = base + flat_sum * (1 + percent_sum) * multiplier_product
- Speed: equipping heavy armor reduces max_walk_speed

---

## T-INT-4.3: Controller-Input Binding Tests

**File**: `tests/integration/test_entity_input.py`

### Tasks

- [ ] Test PlayerController receives ActionMapper callbacks
- [ ] Test possess clears old controller bindings
- [ ] Test unpossess clears current bindings
- [ ] Test input consumption prevents duplicate handling
- [ ] Test modifier keys affect action triggering
- [ ] Test device hot-plug updates controller bindings

### Acceptance Criteria

- All tests pass
- Binding lifecycle: possess -> bind, unpossess -> unbind
- No duplicate action firing

---

## T-INT-4.4: Input-to-Movement Pipeline Tests

**File**: `tests/integration/test_input_movement.py`

### Tasks

- [ ] Test WASD keys -> AxisMapper -> movement_input
- [ ] Test movement_input -> velocity -> position change
- [ ] Test dead zone prevents drift from small input
- [ ] Test response curve affects movement feel
- [ ] Test smoothing affects movement responsiveness
- [ ] Test complete pipeline: key press -> position change

### Acceptance Criteria

- All tests pass
- Pipeline: KeyDown(W) -> AxisMapper(0,1) -> Character.movement_input(0,1) -> velocity -> position
- Dead zone: small stick deflection -> no movement

---

## T-INT-4.5: Singleton Isolation Tests

**File**: `tests/integration/test_singleton_isolation.py`

### Tasks

- [ ] Test PrefabRegistry.reset_instance() clears all prefabs
- [ ] Test PossessionManager.reset_instance() clears all possessions
- [ ] Test LifecycleManager.reset_instance() clears all entities
- [ ] Test DeviceManager.reset_instance() clears all devices
- [ ] Test sequential test runs with reset are isolated
- [ ] Test cross-module reset in correct order

### Acceptance Criteria

- All tests pass
- No state leakage between tests
- Reset order: input -> entity -> economy (dependencies)

---

## T-INT-4.6: Event Propagation Tests

**File**: `tests/integration/test_event_propagation.py`

### Tasks

- [ ] Test inventory add event reaches actor listeners
- [ ] Test equipment change event triggers stat recalculation
- [ ] Test possession change event updates input bindings
- [ ] Test lifecycle state change event reaches global listeners
- [ ] Test device connect event reaches controller listeners
- [ ] Test event listener removal stops propagation

### Acceptance Criteria

- All tests pass
- Event delivery verified with mock listeners
- No memory leaks from unremoved listeners

---

## T-INT-4.7: Error Context Tests

**File**: `tests/integration/test_error_contexts.py`

### Tasks

- [ ] Test inventory full error includes container and item info
- [ ] Test equip requirement error includes stat values
- [ ] Test possession error includes controller and pawn info
- [ ] Test lifecycle invalid transition error includes current and target states
- [ ] Test input binding error includes action and binding info
- [ ] Test all errors are actionable (say what to do)

### Acceptance Criteria

- All tests pass
- Error messages include relevant IDs and values
- No generic "operation failed" messages
