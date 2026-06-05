# PHASE 2 ARCHITECTURE: Entity Module Testing

## Overview

Comprehensive test coverage for engine/gameplay/entity module (~4,418 lines across 5 files).

## Components Under Test

### 1. Actor System (actor.py, 1,167 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| ActorMeta | Metaclass with type ID, component collection | Registration, ID uniqueness |
| ComponentContainer | Type-indexed component lookup | Add, remove, get by type |
| Transform | 3D position, rotation (quaternion), scale | Operations, normalization |
| StaticActor | Non-moving base actor | No tick overhead |
| DynamicActor | Physics-enabled (velocity, forces) | Force application, integration |
| Pawn | Possessable entity | Controller attachment |
| Character | Full movement suite | Walk/run/jump/crouch states |

### 2. Possession System (possession.py, 899 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| ControllerMeta | Metaclass for controller registration | Registration |
| PossessionDescriptor | Trinity Pattern state tracking | State validation |
| PlayerController | Input binding, camera control | Binding, camera updates |
| AIController | Blackboard, behavior tree, movement | Blackboard access, move-to |
| PossessionManager | Singleton tracking possessions | Possess/unpossess lifecycle |

### 3. Prefab System (prefab.py, 774 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| PrefabRegistry | Singleton with inheritance resolution | Registration, lookup |
| PrefabInstantiator | Deferred and immediate instantiation | Spawn timing |
| PrefabBuilder | Fluent API for construction | Build validation |
| @prefab decorator | Registration | Decorator behavior |
| @extends decorator | Inheritance | Parent resolution |

### 4. Lifecycle System (lifecycle.py, 630 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| LifecycleStateDescriptor | State transition validation | Invalid transitions |
| LifecycleManager | Singleton, deferred transitions | Batching, ordering |
| LifecycleMixin | Base class for lifecycle objects | Mixin composition |
| @on_spawn, @begin_play, @tick, @end_play, @on_destroy | Hook decorators | Callback timing |

### 5. Module Init (__init__.py, 687 lines)

| Component | Responsibility | Test Focus |
|-----------|---------------|------------|
| Alternative implementations | Simplified Actor/Pawn/Character | API compatibility |
| spawn() function | Prefab instantiation | Return type, component setup |

## Architecture Decisions

### ADR-ENT-1: Actor Hierarchy Test Fixtures

Create ActorTestHarness with:
- Pre-built actors of each type
- Transform manipulation helpers
- Tick simulation (advance time by delta)

### ADR-ENT-2: Possession State Machine

Test all valid transitions:
- None -> Possessed (possess)
- Possessed -> None (unpossess)
- Possessed -> Possessed (repossess with different controller)

Test all invalid transitions are rejected.

### ADR-ENT-3: Prefab Inheritance Depth

Verify MAX_PREFAB_INHERITANCE_DEPTH is enforced:
- Chain at max depth succeeds
- Chain at max+1 depth raises RecursionError

### ADR-ENT-4: Lifecycle Deferred Batching

Verify that transitions within a frame are batched:
- Multiple spawn() calls in one tick -> all process at frame end
- Destroy during tick -> processed after tick completes

### ADR-ENT-5: Weak Reference Correctness

Verify parent/child actor relationships use weak references:
- Parent destroyed -> child.parent returns None, no crash
- Child destroyed -> parent.children no longer contains child

## Test Structure

```
tests/
  entity/
    test_actor_meta.py              # ActorMeta, type IDs
    test_actor_components.py        # ComponentContainer
    test_actor_transform.py         # Transform operations
    test_actor_hierarchy.py         # StaticActor->DynamicActor->Pawn->Character
    test_actor_physics.py           # DynamicActor forces, velocity
    test_character_movement.py      # Walk/run/jump/crouch
    test_possession_manager.py      # PossessionManager
    test_possession_player.py       # PlayerController
    test_possession_ai.py           # AIController, blackboard, move-to
    test_prefab_registry.py         # PrefabRegistry
    test_prefab_inheritance.py      # Inheritance resolution
    test_prefab_instantiation.py    # Spawn, deferred spawn
    test_lifecycle_states.py        # State transitions
    test_lifecycle_manager.py       # LifecycleManager batching
    test_lifecycle_decorators.py    # Hook decorator timing
```

## Dependencies

- pytest for test framework
- pytest-mock for mocking tick/frame boundaries

## Risks

| Risk | Mitigation |
|------|------------|
| Circular references in actor hierarchy | Weak reference tests |
| Non-deterministic tick ordering | Explicit tick group assertions |
| Metaclass side effects | reset_instance() in teardown |
