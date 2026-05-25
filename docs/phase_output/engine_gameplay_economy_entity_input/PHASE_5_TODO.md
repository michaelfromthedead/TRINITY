# PHASE 5 TODO: Serialization Completion

## Summary

Complete from_dict() methods for all serializable types.

---

## T-SER-5.1: Economy from_dict() Implementation

**Files**: `engine/gameplay/economy/*.py`

### Tasks

- [ ] Implement ItemDefinition.from_dict() with version field
- [ ] Implement ItemInstance.from_dict() with stack quantity
- [ ] Implement InventorySlot.from_dict() with lock state
- [ ] Implement InventoryContainer.from_dict() with all slots
- [ ] Implement Recipe.from_dict() with ingredients list
- [ ] Implement CraftingQueue.from_dict() with progress state
- [ ] Implement StatModifier.from_dict() with modifier type
- [ ] Implement EquipmentContainer.from_dict() with equipped items

### Acceptance Criteria

- All from_dict() methods pass round-trip test
- Version field present in all serialized output
- Enum values serialize as strings

---

## T-SER-5.2: Entity from_dict() Implementation

**Files**: `engine/gameplay/entity/*.py`

### Tasks

- [ ] Implement Transform.from_dict() with position, rotation, scale
- [ ] Implement Actor.from_dict() with components
- [ ] Implement Pawn.from_dict() with controller reference
- [ ] Implement Character.from_dict() with movement state
- [ ] Implement PrefabDefinition.from_dict() with inheritance
- [ ] Implement ControllerState.from_dict() with blackboard

### Acceptance Criteria

- All from_dict() methods pass round-trip test
- Controller reference stored as ID, resolved on load
- Components deserialized by type registry

---

## T-SER-5.3: Input from_dict() Implementation

**Files**: `engine/gameplay/input/*.py`

### Tasks

- [ ] Implement ActionBinding.from_dict() with trigger type
- [ ] Implement AxisBinding.from_dict() with binding type
- [ ] Implement InputSettings.from_dict() with all settings

### Acceptance Criteria

- All from_dict() methods pass round-trip test
- Trigger types serialize as strings
- Device references handled correctly

---

## T-SER-5.4: Round-Trip Tests

**File**: `tests/serialization/test_roundtrip.py`

### Tasks

- [ ] Test ItemDefinition round-trip with all fields
- [ ] Test ItemInstance round-trip with various quantities
- [ ] Test InventoryContainer round-trip with mixed slots
- [ ] Test Actor round-trip with components
- [ ] Test Character round-trip with movement state
- [ ] Test complete inventory with items round-trip
- [ ] Test complete actor hierarchy round-trip

### Acceptance Criteria

- All tests pass
- Equality check: original == deserialized
- Deep equality for nested objects

---

## T-SER-5.5: Version Migration Tests

**File**: `tests/serialization/test_version_migration.py`

### Tasks

- [ ] Test loading v1 data with current code
- [ ] Test migration stub fills default values
- [ ] Test unknown version raises VersionError
- [ ] Test migration preserves existing data
- [ ] Test migration chain v1 -> v2 -> v3

### Acceptance Criteria

- All tests pass
- Old saves load without data loss
- Migration is idempotent

---

## T-SER-5.6: DeserializationContext Implementation

**File**: `engine/serialization/context.py`

### Tasks

- [ ] Create DeserializationContext class
- [ ] Implement resolve_actor(id) for actor references
- [ ] Implement resolve_item(id) for item references
- [ ] Implement resolve_controller(id) for controller references
- [ ] Implement deferred resolution for forward references
- [ ] Implement cycle detection for reference resolution

### Acceptance Criteria

- Context resolves all reference types
- Forward references resolve after full load
- Cycles detected and reported

---

## T-SER-5.7: Error Handling Tests

**File**: `tests/serialization/test_errors.py`

### Tasks

- [ ] Test missing required field raises KeyError
- [ ] Test invalid type raises TypeError
- [ ] Test invalid enum value raises ValueError
- [ ] Test unresolvable reference raises ReferenceError
- [ ] Test corrupted data raises SerializationError
- [ ] Test error messages include field path

### Acceptance Criteria

- All tests pass
- Errors are specific and actionable
- Field path: "inventory.slots[3].item.quantity"
