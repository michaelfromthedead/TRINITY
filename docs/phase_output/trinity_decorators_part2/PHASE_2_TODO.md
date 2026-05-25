# PHASE 2 TODO: Gameplay & World Systems

## Overview

Validate gameplay, world building, AI, state machine, procedural, and spatial decorators.

---

## T2.1: Validate Gameplay Decorators

**File**: `trinity/decorators/gameplay.py`

**Tasks**:
- [ ] Verify `@ability` validates cooldown >= 0
- [ ] Verify `@ability` accepts cost dict[str, float]
- [ ] Verify `@ability` accepts tags and blocked_by sets
- [ ] Verify `@buff` validates stacking mode against VALID_STACKING
- [ ] Verify `@gameplay_tag` tags target with gameplay category
- [ ] Verify `@spawner` registers in gameplay registry
- [ ] Verify `@interactable` sets interaction hooks
- [ ] Verify `@quest` sets quest metadata

**Acceptance Criteria**:
- All 6 decorators produce correct steps
- VALID_STACKING = {"replace", "stack", "refresh", "unique"}
- Error messages include decorator name and valid options

---

## T2.2: Validate World Building Decorators

**File**: `trinity/decorators/world_building.py`

**Tasks**:
- [ ] Verify `@foliage_type` validates against VALID_FOLIAGE_TYPES
- [ ] Verify `@procedural_placement` sets placement rules
- [ ] Verify `@level_instance` marks level streaming
- [ ] Verify `@water_body` validates against VALID_WATER_TYPES
- [ ] Verify `@navmesh_modifier` validates against VALID_NAVMESH_MODIFIERS
- [ ] Verify `@trigger_volume` sets trigger callbacks

**Acceptance Criteria**:
- All 6 decorators follow 6-part pattern
- VALID_* constants defined as frozensets
- All register in "world" registry

---

## T2.3: Validate Game AI Decorators

**File**: `trinity/decorators/game_ai.py`

**Tasks**:
- [ ] Verify `@behavior_tree` accepts root node parameter
- [ ] Verify `@behavior_tree` accepts tick_rate (default 0.1)
- [ ] Verify `@utility_ai` sets scoring system
- [ ] Verify `@blackboard` creates shared AI state
- [ ] Verify `@ai_debug` enables AI visualization
- [ ] Verify `@perception` configures sensory systems

**Acceptance Criteria**:
- All 5 decorators produce ops chain
- All register in "ai" registry
- Behavior tree supports tick rate configuration

---

## T2.4: Validate State Machine Decorators

**File**: `trinity/decorators/state_machine.py`

**Tasks**:
- [ ] Verify `@state_machine` requires states list
- [ ] Verify `@state_machine` validates initial state is in states
- [ ] Verify `@on_enter` registers state entry callback
- [ ] Verify `@on_exit` registers state exit callback
- [ ] Verify transition validation

**Error Message Test**:
```python
@state_machine(states=["idle", "walk"], initial="run")  # Should raise:
# ValueError: @state_machine: initial state 'run' is not in states ['idle', 'walk']
```

**Acceptance Criteria**:
- State validation is compile-time (decoration time)
- Entry/exit hooks use Op.HOOK
- Invalid initial state produces actionable error

---

## T2.5: Validate Procedural Decorators

**File**: `trinity/decorators/procedural.py`

**Tasks**:
- [ ] Verify `@seeded` validates seed_source against VALID_SEED_SOURCES
- [ ] Verify `@seeded` sets deterministic flag
- [ ] Verify `@procedural` marks procedural generation
- [ ] Verify `@constraint` sets generation constraints

**Valid Seed Sources**:
- [ ] "world" - world seed
- [ ] "local" - local coordinates
- [ ] "instance" - instance ID
- [ ] "fixed" - fixed seed value

**Acceptance Criteria**:
- VALID_SEED_SOURCES = {"world", "local", "instance", "fixed"}
- Invalid seed source produces actionable error
- All register in appropriate registry

---

## T2.6: Validate Spatial Decorators

**File**: `trinity/decorators/spatial.py`

**Tasks**:
- [ ] Verify `@spatial` validates structure against VALID_SPATIAL_STRUCTURES
- [ ] Verify `@partitioned` sets partitioning scheme
- [ ] Verify spatial query optimization hints

**Valid Structures**:
- [ ] "octree" - 3D spatial partitioning
- [ ] "bvh" - bounding volume hierarchy
- [ ] "grid" - uniform grid
- [ ] "kdtree" - k-dimensional tree

**Acceptance Criteria**:
- VALID_SPATIAL_STRUCTURES = {"octree", "bvh", "grid", "kdtree"}
- Invalid structure produces actionable error
- Both decorators set spatial metadata

---

## Summary

| Task | File | Decorators | Lines |
|------|------|------------|-------|
| T2.1 | gameplay.py | 6 | 349 |
| T2.2 | world_building.py | 6 | 339 |
| T2.3 | game_ai.py | 5 | 247 |
| T2.4 | state_machine.py | 3 | 196 |
| T2.5 | procedural.py | 3 | 179 |
| T2.6 | spatial.py | 2 | 165 |

**Total**: 25 decorators, 1,475 lines
