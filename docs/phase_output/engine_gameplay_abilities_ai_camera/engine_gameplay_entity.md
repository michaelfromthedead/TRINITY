# Investigation: engine/gameplay/entity

## Summary
The entity subsystem is a **fully implemented** Actor/Pawn framework following UE5-inspired architecture. It provides complete lifecycle management, controller possession, prefab templates, and component composition with real state machines, thread-safe singleton managers, and proper hierarchy handling. This is production-quality code with ~3,500 lines of substantive implementation.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 688 | REAL | Exports + simplified Actor/Pawn/Character/Controller classes |
| `actor.py` | 1168 | REAL | Full actor hierarchy with metaclass, ComponentContainer, Transform |
| `lifecycle.py` | 631 | REAL | Complete lifecycle state machine with descriptors, decorators, manager |
| `constants.py` | 262 | REAL | All constants, enums (LifecycleState, ActorType, TickGroup) |
| `possession.py` | 900 | REAL | Controller classes, possession management, AI controller with blackboard |
| `prefab.py` | 775 | REAL | Prefab registry, instantiator, builder API, decorators |

## Entity Components
- **Actor hierarchy**: Actor -> StaticActor -> DynamicActor -> Pawn -> Character
- **Controller system**: Controller -> PlayerController, AIController
- **Lifecycle management**: LifecycleManager singleton, LifecycleMixin, state descriptors
- **Prefab system**: PrefabRegistry, PrefabInstantiator, PrefabBuilder fluent API
- **Component system**: ComponentContainer with type-based lookup
- **Transform system**: Position/rotation/scale with hierarchy support

## Implementation
- Real entity management? **YES** - ComponentContainer, weak-ref hierarchy, ActorMeta registry
- Real spawning? **YES** - PrefabInstantiator with deferred/immediate modes, spawn_prefab()
- Real lifecycle? **YES** - 10-state machine (UNINITIALIZED->DESTROYED), validated transitions
- Real tick system? **YES** - TickGroup enum, delta_time handling, Character movement
- Real possession? **YES** - possess/unpossess with callbacks, PossessionManager tracking
- Real AI? **YES** - AIController with blackboard, move_to_location, behavior tree hooks

## Verdict
**REAL IMPLEMENTATION** - Complete, production-quality entity system

## Evidence

### Lifecycle State Machine (constants.py:31-42)
```python
VALID_LIFECYCLE_TRANSITIONS: dict[LifecycleState, frozenset[LifecycleState]] = {
    LifecycleState.UNINITIALIZED: frozenset({LifecycleState.CREATED}),
    LifecycleState.CREATED: frozenset({LifecycleState.INITIALIZING, LifecycleState.DESTROYING}),
    LifecycleState.INITIALIZING: frozenset({LifecycleState.INITIALIZED, LifecycleState.DESTROYING}),
    LifecycleState.INITIALIZED: frozenset({LifecycleState.BEGINNING_PLAY, LifecycleState.DESTROYING}),
    LifecycleState.BEGINNING_PLAY: frozenset({LifecycleState.ACTIVE, LifecycleState.DESTROYING}),
    LifecycleState.ACTIVE: frozenset({LifecycleState.DEACTIVATING, LifecycleState.DESTROYING}),
    # ...
}
```

### ComponentContainer (actor.py:201-275)
```python
class ComponentContainer:
    __slots__ = ("_components", "_owner", "_type_to_name")

    def add(self, name: str, component: Any) -> None:
        if len(self._components) >= MAX_COMPONENTS_PER_ENTITY:
            raise ValueError(f"Maximum component count ({MAX_COMPONENTS_PER_ENTITY}) reached")
        self._components[name] = component
        self._type_to_name[type(component)] = name
        # Notify owner
        owner = self._owner()
        if owner and hasattr(owner, "_on_component_added"):
            owner._on_component_added(name, component)
```

### Character Movement (actor.py:1063-1081)
```python
def jump(self) -> bool:
    if not self._is_grounded or self._is_jumping:
        return False
    self._is_jumping = True
    self._is_grounded = False
    self._velocity = (
        self._velocity[0],
        self._velocity[1] + self._jump_velocity,
        self._velocity[2],
    )
    return True
```

### Prefab Builder Fluent API (prefab.py:566-671)
```python
class PrefabBuilder(Generic[T]):
    def with_component(self, name: str, component_type: type, **properties) -> "PrefabBuilder[T]":
        self._components[name] = ComponentDefinition(name=name, component_type=component_type, ...)
        return self

    def instantiate(self, overrides=None, transform=None) -> T:
        self.build()
        return PrefabInstantiator().instantiate(self._name, overrides=overrides, immediate=True)
```

### AI Controller Movement (possession.py:741-779)
```python
def _process_movement(self, delta_time: float) -> None:
    target = self._blackboard.get("move_target")
    if target is None:
        return
    pawn = self.pawn
    current_pos = pawn.position
    dx = target[0] - current_pos[0]
    # ... distance calculation and movement
    if distance > 0:
        speed = pawn.max_walk_speed if hasattr(pawn, "max_walk_speed") else DEFAULT_AI_MOVE_SPEED
        move_dist = min(speed * delta_time, distance)
        factor = move_dist / distance
        pawn.position = (current_pos[0] + dx * factor, ...)
```

## Architecture Notes
- Uses Trinity Pattern: metaclasses for registration, descriptors for state tracking, decorators for hooks
- Thread-safe singletons for managers (LifecycleManager, PrefabRegistry, PossessionManager)
- Weak references for parent/child/controller relationships to prevent memory leaks
- Deferred operations batch to end-of-frame for performance
- Integrates with trinity.decorators.ops (Op, Step) and trinity.metaclasses.engine_meta
