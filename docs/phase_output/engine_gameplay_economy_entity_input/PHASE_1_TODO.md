# PHASE 1 TODO: Economy Module Testing

## Summary

Test coverage for engine/gameplay/economy (~4,217 lines).

---

## T-ECON-1.1: Inventory Item Tests

**File**: `tests/economy/test_inventory_item.py`

### Tasks

- [ ] Test ItemDefinition validates required fields (id, name, max_stack)
- [ ] Test ItemInstance.merge_from() transfers quantity correctly
- [ ] Test ItemInstance.merge_from() raises when items cannot stack
- [ ] Test ItemInstance.space_remaining returns correct value
- [ ] Test ItemInstance.quantity cannot exceed max_stack
- [ ] Test ItemInstance.quantity cannot go negative
- [ ] Test ItemInstance.can_stack_with() checks item_id match
- [ ] Test ItemInstance.can_stack_with() checks definition compatibility

### Acceptance Criteria

- All tests pass with `uv run pytest tests/economy/test_inventory_item.py`
- 100% coverage of merge_from() method
- Edge case: merge with zero quantity source

---

## T-ECON-1.2: Inventory Container Tests

**File**: `tests/economy/test_inventory_container.py`

### Tasks

- [ ] Test auto_stack finds best existing stack
- [ ] Test auto_stack creates new slot when no stackable exists
- [ ] Test auto_stack respects weight limit
- [ ] Test split_stack creates new instance with correct quantity
- [ ] Test split_stack validates quantity <= source.quantity
- [ ] Test compact() merges partial stacks
- [ ] Test sort_by_type() groups items correctly
- [ ] Test sort_by_rarity() orders by rarity enum value
- [ ] Test transfer() moves items between containers
- [ ] Test event listeners fire on add/remove/update

### Acceptance Criteria

- All tests pass
- Weight limit edge cases covered (at limit, over limit, exactly at)
- Event listener tests verify callback arguments

---

## T-ECON-1.3: Inventory Transaction Tests

**File**: `tests/economy/test_inventory_transactions.py`

### Tasks

- [ ] Test begin_transaction() marks transaction active
- [ ] Test commit_transaction() applies all changes
- [ ] Test rollback_transaction() reverts all changes
- [ ] Test nested transactions raise error
- [ ] Test operations outside transaction apply immediately
- [ ] Test rollback after add restores original state
- [ ] Test rollback after remove restores original state
- [ ] Test rollback after modify restores original quantity

### Acceptance Criteria

- All tests pass
- Invariant: rollback leaves container in pre-transaction state
- Invariant: commit leaves container in post-operation state

---

## T-ECON-1.4: Crafting Quality Tests

**File**: `tests/economy/test_crafting_quality.py`

### Tasks

- [ ] Test base quality distribution matches QUALITY_BASE_CHANCES
- [ ] Test skill excess increases quality bonus
- [ ] Test station bonus adds to quality bonus
- [ ] Test context quality_bonus adds to total
- [ ] Test quality roll respects cumulative probability
- [ ] Test deterministic output with seeded RNG
- [ ] Test zero bonus produces base distribution
- [ ] Test maximum bonus skews heavily toward legendary

### Acceptance Criteria

- All tests pass
- Seeded RNG produces identical results across runs
- Statistical tests verify distribution within tolerance (N=1000)

---

## T-ECON-1.5: Loot Pity System Tests

**File**: `tests/economy/test_loot_pity.py`

### Tasks

- [ ] Test check_pity() returns False below threshold
- [ ] Test check_pity() returns True at threshold
- [ ] Test check_pity() returns True above threshold
- [ ] Test counter increments on failure
- [ ] Test counter resets on success
- [ ] Test PITY_WEIGHT_BOOST applies when pity triggers
- [ ] Test different rarity thresholds from RARITY_PITY_THRESHOLDS
- [ ] Test zero threshold rarity never triggers pity

### Acceptance Criteria

- All tests pass
- Counter state verified after each operation
- Edge case: threshold = 1 (immediate pity)

---

## T-ECON-1.6: Loot Table Tests

**File**: `tests/economy/test_loot_tables.py`

### Tasks

- [ ] Test weighted selection respects weights
- [ ] Test nested table recursion resolves correctly
- [ ] Test empty table returns no items
- [ ] Test single-entry table always returns that entry
- [ ] Test condition evaluation filters entries
- [ ] Test luck bonus modifies weights
- [ ] Test deterministic output with seeded RNG
- [ ] Test LootTableBuilder produces valid tables

### Acceptance Criteria

- All tests pass
- Statistical tests verify weight distribution (N=1000)
- Nested tables: verify recursion depth limit

---

## T-ECON-1.7: Equipment Modifier Tests

**File**: `tests/economy/test_equipment_modifiers.py`

### Tasks

- [ ] Test flat modifier adds to base
- [ ] Test percent modifier multiplies base
- [ ] Test multiplier modifier multiplies total
- [ ] Test modifier stacking order: flat -> percent -> multiplier
- [ ] Test resistance modifier respects cap
- [ ] Test negative modifiers work correctly
- [ ] Test zero modifiers have no effect
- [ ] Test modifier removal restores base value

### Acceptance Criteria

- All tests pass
- Stacking order verified with multi-modifier setup
- Cap enforcement verified at exact cap value

---

## T-ECON-1.8: Equipment Container Tests

**File**: `tests/economy/test_equipment_container.py`

### Tasks

- [ ] Test equip() places item in correct slot
- [ ] Test equip() two-hand weapon clears both hand slots
- [ ] Test unequip() removes item and returns it
- [ ] Test requirement check blocks under-level equip
- [ ] Test requirement check blocks under-stat equip
- [ ] Test set bonus detection with partial set
- [ ] Test set bonus detection with full set
- [ ] Test durability decreases on use
- [ ] Test repair restores durability
- [ ] Test zero durability triggers broken state

### Acceptance Criteria

- All tests pass
- Exclusive slot handling verified (two-hand clears offhand)
- Set bonus tests cover 2/4/6 piece thresholds
