# PROJECT: engine/gameplay/{economy,entity,input}

## Overview

Integration and test coverage completion for three production-quality gameplay modules:
- **economy** (~4,217 lines): RPG economy systems (inventory, crafting, loot, equipment)
- **entity** (~4,418 lines): UE5-style actor framework (actors, possession, prefabs, lifecycle)
- **input** (~4,064 lines): Professional input processing (devices, actions, axes, processing)

**Total Lines**: ~12,699 across 13 files

## Scope

### In Scope
- Test coverage for edge cases (pity system, stack merging, dead zones)
- Serialization completion (from_dict methods for save/load)
- Networking preparation (replication flags for multiplayer)
- Performance optimization (object pooling for high-churn items)
- Integration validation between economy<->entity and entity<->input

### Out of Scope
- Rewriting existing implementations (all are production-ready)
- Adding new features beyond recommendations
- UI/rendering layer integration

## Goals

1. Achieve comprehensive test coverage for all algorithmic edge cases
2. Complete serialization APIs for full save/load support
3. Add replication infrastructure for future multiplayer
4. Optimize memory patterns for high-frequency operations
5. Validate cross-module integration points

## Constraints

- Python 3.13 required (not 3.14)
- Must preserve existing API signatures
- Must maintain Trinity Pattern compatibility (decorators, descriptors, metaclasses)
- Singleton patterns must retain reset_instance() for testing

## Acceptance Criteria

### Phase 1 (Economy Testing)
- [ ] Pity system tests: threshold tracking, counter reset, boost application
- [ ] Stack merging tests: overflow, zero quantity, incompatible items
- [ ] Weighted random tests: nested tables, empty tables, single entry
- [ ] Transaction tests: begin/commit/rollback, nested transactions

### Phase 2 (Entity Testing)
- [ ] Prefab inheritance tests: max depth, circular reference detection
- [ ] Lifecycle state tests: invalid transitions, deferred batching
- [ ] Actor hierarchy tests: attachment, detachment, orphaning
- [ ] Possession tests: controller swap, unpossess during tick

### Phase 3 (Input Testing)
- [ ] Dead zone tests: axial, radial, cross, zero division
- [ ] Response curve tests: power, exponential, S-curve edge values
- [ ] Trigger state machine tests: hold, tap, double-tap, combo
- [ ] Device hot-plug tests: connect, disconnect, reconnect

### Phase 4 (Integration)
- [ ] Economy<->Entity: inventory owner_id, equipment stat application
- [ ] Entity<->Input: PlayerController binding, character movement from axes
- [ ] Cross-cutting: singleton reset, event propagation, error contexts

### Phase 5 (Serialization)
- [ ] from_dict() methods for all serializable types
- [ ] Round-trip tests: to_dict() -> from_dict() == original
- [ ] Version migration stubs for future schema changes

### Phase 6 (Networking Prep)
- [ ] Replication flags on mutable state
- [ ] Authority markers for server/client ownership
- [ ] Delta compression hints for frequently changing values

### Phase 7 (Performance)
- [ ] Object pool for ItemInstance
- [ ] Object pool for input events
- [ ] Weak reference audit for actor hierarchy
- [ ] ID generation lock contention analysis
