# PHASE 2 TODO: Entity Module Testing

## Summary

Test coverage for engine/gameplay/entity (~4,418 lines).

---

## T-ENT-2.1: Actor Metaclass Tests

**File**: `tests/entity/test_actor_meta.py`

### Tasks

- [ ] Test ActorMeta assigns unique type IDs to each actor class
- [ ] Test ActorMeta collects components from class definition
- [ ] Test type ID is stable across interpreter restarts (or document if not)
- [ ] Test subclass inherits parent components
- [ ] Test subclass can override parent components

### Acceptance Criteria

- All tests pass with `uv run pytest tests/entity/test_actor_meta.py`
- Type ID uniqueness verified across 10+ actor classes
- Component inheritance verified with 3-level hierarchy

---

## T-ENT-2.2: Component Container Tests

**File**: `tests/entity/test_actor_components.py`

### Tasks

- [ ] Test add_component() stores component by type
- [ ] Test get_component() retrieves by exact type
- [ ] Test get_component() returns None for missing type
- [ ] Test has_component() returns correct boolean
- [ ] Test remove_component() removes and returns component
- [ ] Test multiple components of different types coexist
- [ ] Test replacing component of same type

### Acceptance Criteria

- All tests pass
- O(1) lookup verified (no iteration in implementation)
- Type safety: get returns correct subclass

---

## T-ENT-2.3: Actor Hierarchy Tests

**File**: `tests/entity/test_actor_hierarchy.py`

### Tasks

- [ ] Test StaticActor has position but no tick
- [ ] Test DynamicActor inherits StaticActor + adds velocity
- [ ] Test Pawn inherits DynamicActor + adds controller slot
- [ ] Test Character inherits Pawn + adds movement modes
- [ ] Test parent/child attachment uses weak references
- [ ] Test destroying parent orphans children (no crash)
- [ ] Test destroying child removes from parent.children
- [ ] Test attach_to() updates both parent and child references

### Acceptance Criteria

- All tests pass
- Weak reference behavior verified (parent gone -> child.parent is None)
- Memory: no reference cycles detected

---

## T-ENT-2.4: Character Movement Tests

**File**: `tests/entity/test_character_movement.py`

### Tasks

- [ ] Test movement_input sets _is_walking flag
- [ ] Test tick() applies velocity based on input and speed
- [ ] Test current_max_speed returns walk_speed when walking
- [ ] Test current_max_speed returns run_speed when running
- [ ] Test jump() sets vertical velocity when grounded
- [ ] Test jump() does nothing when airborne
- [ ] Test crouch() reduces max speed
- [ ] Test movement_input resets to (0,0) after tick

### Acceptance Criteria

- All tests pass
- Movement physics verified: position += velocity * delta_time
- Jump: verified vertical velocity application

---

## T-ENT-2.5: Possession Manager Tests

**File**: `tests/entity/test_possession_manager.py`

### Tasks

- [ ] Test possess() links controller to pawn
- [ ] Test possess() updates PossessionManager tracking
- [ ] Test unpossess() unlinks controller from pawn
- [ ] Test repossess() transfers pawn to new controller
- [ ] Test possessing already-possessed pawn unpossesses previous
- [ ] Test get_possessed_pawn() returns correct pawn
- [ ] Test get_controller_for() returns correct controller
- [ ] Test reset_instance() clears all possessions

### Acceptance Criteria

- All tests pass
- Singleton behavior verified
- No dangling references after unpossess

---

## T-ENT-2.6: AI Controller Tests

**File**: `tests/entity/test_possession_ai.py`

### Tasks

- [ ] Test blackboard.set() and blackboard.get() work correctly
- [ ] Test move_to_location() sets move_target in blackboard
- [ ] Test _process_movement() moves pawn toward target
- [ ] Test movement stops within acceptance_radius
- [ ] Test stop_movement() clears move_target
- [ ] Test movement respects pawn.max_walk_speed
- [ ] Test direction calculation is correct (normalized)

### Acceptance Criteria

- All tests pass
- Movement math verified: distance calculation, normalization
- Edge case: target at current position (distance = 0)

---

## T-ENT-2.7: Prefab Inheritance Tests

**File**: `tests/entity/test_prefab_inheritance.py`

### Tasks

- [ ] Test child prefab inherits parent actor_class
- [ ] Test child prefab inherits parent components
- [ ] Test child prefab can override parent components
- [ ] Test child prefab inherits parent properties
- [ ] Test child prefab can override parent properties
- [ ] Test child prefab inherits parent tags (union)
- [ ] Test inheritance chain resolves correctly (grandparent->parent->child)
- [ ] Test MAX_PREFAB_INHERITANCE_DEPTH raises RecursionError
- [ ] Test circular inheritance detected and raises error

### Acceptance Criteria

- All tests pass
- Depth limit verified at exact threshold
- Circular detection verified with A->B->A chain

---

## T-ENT-2.8: Lifecycle State Tests

**File**: `tests/entity/test_lifecycle_states.py`

### Tasks

- [ ] Test valid transition: NONE -> SPAWNING -> ACTIVE
- [ ] Test valid transition: ACTIVE -> DESTROYING -> DESTROYED
- [ ] Test invalid transition raises error (e.g., NONE -> ACTIVE)
- [ ] Test LifecycleStateDescriptor validates on assignment
- [ ] Test state is readable after valid transition
- [ ] Test @on_spawn decorator fires during SPAWNING
- [ ] Test @begin_play decorator fires at ACTIVE entry
- [ ] Test @end_play decorator fires at DESTROYING entry
- [ ] Test @on_destroy decorator fires at DESTROYED entry

### Acceptance Criteria

- All tests pass
- All valid state transitions enumerated and tested
- All invalid transitions verified to raise

---

## T-ENT-2.9: Lifecycle Manager Batching Tests

**File**: `tests/entity/test_lifecycle_manager.py`

### Tasks

- [ ] Test deferred_spawn() queues spawn for frame end
- [ ] Test multiple deferred_spawn() in one tick all process together
- [ ] Test deferred_destroy() queues destroy for frame end
- [ ] Test destroy during tick processes after tick completes
- [ ] Test flush_deferred() processes all queued transitions
- [ ] Test global callbacks fire for all entities
- [ ] Test state counting returns correct counts per state
- [ ] Test reset_instance() clears all entities and queues

### Acceptance Criteria

- All tests pass
- Batching verified: spawn order preserved
- No mid-tick state inconsistency
