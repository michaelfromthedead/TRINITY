# PHASE 4 ARCHITECTURE: Layer 4 Integration and ShellLang

## Overview

Phase 4 implements the integration layer that bridges Foundation with Trinity's component system, plus the complete ShellLang ECS DSL. This is the highest layer with full dependency on all lower layers.

## Integration Components

### bridge.py (241 lines)

Trinity to Foundation integration adapter.

**Functions**:
- `get_trinity_registry() -> Dict[str, Type]`: Pulls ComponentMeta.all_components() into Foundation
- `create_ai_interface(world) -> AIInterface`: Creates AI command executor for Trinity world

**Classes**:
- `TrinityWorldAdapter`: Bidirectional sync between Trinity and ShellLang
  - `__init__(trinity_world, shelllang_world)`: Connect both worlds
  - `sync_to_shelllang()`: Copy Trinity state to ShellLang
  - `sync_from_shelllang()`: Copy ShellLang state to Trinity
  - `on_trinity_change(entity, component)`: React to Trinity changes
  - `on_shelllang_change(entity, field, value)`: React to ShellLang changes

**Integration Points**:
1. Type registration: Trinity components registered in Foundation registry
2. Instance tracking: Trinity instances tracked for queries
3. Change propagation: Changes in either system sync to the other
4. AI access: AI agents manipulate Trinity via ShellLang

## ShellLang Subsystem

### shelllang/__init__.py (63 lines)

Package exports.

**Exports**:
- World, Entity, Snapshot, Change (from core)
- EntityProxy, QueryResult, TypeQuery, TimeManager (from sugar)
- AIInterface (from ai)
- Shell, Feedback (from repl)

### shelllang/core.py (395 lines)

The 5 semantic primitives.

**1. Entity (uint64 identifier)**
```python
class Entity:
    __slots__ = ('_id',)
    
    def __init__(self, entity_id: int):
        self._id = entity_id
    
    def __hash__(self):
        return hash(self._id)
    
    def __eq__(self, other):
        return isinstance(other, Entity) and self._id == other._id
```

- Lightweight handle (single int)
- Hashable for use as dict key
- Comparable for sorting
- IDs auto-increment from 1

**2. Component (type alias)**
```python
Component = Type[Any]  # Any Python class
```

- Any Python class can be a component
- No marker interface required
- Attached to entities via World.attach()

**3. World (ECS container)**
```python
class World:
    _entities: Dict[int, Set[str]]      # entity_id -> component names
    _components: Dict[int, Dict[str, Any]]  # entity_id -> {name: instance}
    _next_entity_id: int
    _history: List[Change]
```

- `create() -> Entity`: Spawn new entity
- `destroy(entity)`: Remove entity and components
- `attach(entity, component)`: Add component to entity
- `detach(entity, ComponentType)`: Remove component from entity
- `get(entity, ComponentType)`: Get component instance
- `has(entity, ComponentType)`: Check component presence
- `set(entity, ComponentType, field, value)`: Mutate field
- `query(*ComponentTypes) -> List[Entity]`: Find entities

**4. Mutate (tracked change)**
```python
@dataclass
class Change:
    entity: Entity
    component_name: str
    field: str
    old_value: Any
    new_value: Any
    tick: int
```

- All mutations recorded as Change objects
- Enables undo/redo
- Enables network sync

**5. Snapshot (frozen state)**
```python
class Snapshot:
    _data: Dict[int, Dict[str, Dict[str, Any]]]  # Frozen copy
    _tick: int
    _name: Optional[str]
```

- `snap(name=None) -> Snapshot`: Create snapshot
- `restore(snapshot)`: Revert to snapshot
- `diff(a, b) -> List[Change]`: Compute changes between snapshots

### shelllang/sugar.py (541 lines)

Fluent DSL layer.

**EntityProxy**: Dot access to components
```python
class EntityProxy:
    def __init__(self, entity: Entity, world: World):
        self._entity = entity
        self._world = world
    
    def __getattr__(self, component_name: str) -> ComponentProxy:
        return ComponentProxy(self._entity, component_name, self._world)
```

Usage: `e.health.current` -> gets Health.current field

**ComponentProxy**: Field access and mutation
```python
class ComponentProxy:
    def __getattr__(self, field: str) -> Any:
        return self._world.get(self._entity, self._component)[field]
    
    def __setattr__(self, field: str, value: Any):
        self._world.set(self._entity, self._component, field, value)
```

Usage: `e.health.current = 100` -> tracked mutation

**QueryResult**: Chainable query results
```python
class QueryResult:
    def where(self, predicate: Callable) -> 'QueryResult': ...
    def near(self, target: Entity, radius: float) -> 'QueryResult': ...
    def first(self) -> Optional[EntityProxy]: ...
    def all(self) -> List[EntityProxy]: ...
    def count(self) -> int: ...
    def set(self, **fields) -> int: ...  # Bulk update
```

Usage: `enemies.where(lambda e: e.health.current < 50).set(health__current=0)`

**TypeQuery**: Type-based query entry
```python
class TypeQuery:
    def __init__(self, component_type: Type):
        self._type = component_type
    
    @property
    def all(self) -> QueryResult:
        return QueryResult(world.query(self._type), world)
```

Usage: `Enemy.all.where(...)`

**TimeManager**: Named snapshots and undo/redo
```python
class TimeManager:
    def mark(self, name: str) -> Snapshot: ...
    def rewind(self, name: str): ...
    def undo(self, steps: int = 1): ...
    def redo(self, steps: int = 1): ...
    def history() -> List[str]: ...
```

Usage: `mark("before_fight")` ... `rewind("before_fight")`

### shelllang/ai.py (515 lines)

Structured JSON command interface.

**AIInterface**: Main entry point
```python
class AIInterface:
    def execute(self, command: Dict) -> Dict: ...
    def validate(self, command: Dict) -> Dict: ...
    def dry_run(self, command: Dict) -> Dict: ...
```

**Operations (10 total)**:

| Operation | Command | Result |
|-----------|---------|--------|
| query | `{"op": "query", "components": ["Health", "Enemy"]}` | `{"entities": [1, 2, 3]}` |
| set | `{"op": "set", "entity": 1, "component": "Health", "field": "current", "value": 50}` | `{"success": true}` |
| spawn | `{"op": "spawn", "components": {"Health": {"current": 100}, "Enemy": {}}}` | `{"entity": 42}` |
| destroy | `{"op": "destroy", "entity": 42}` | `{"success": true}` |
| snap | `{"op": "snap", "name": "checkpoint"}` | `{"tick": 5}` |
| restore | `{"op": "restore", "name": "checkpoint"}` | `{"success": true}` |
| inspect | `{"op": "inspect", "entity": 1}` | `{"components": {...}}` |
| schema | `{"op": "schema", "component": "Health"}` | `{"fields": {...}}` |
| list_types | `{"op": "list_types"}` | `{"types": ["Health", "Enemy", ...]}` |
| count | `{"op": "count", "components": ["Enemy"]}` | `{"count": 5}` |

**Validation**: Each operation validates parameters before execution.

**Dry Run**: Returns preview of effects without mutating world.

### shelllang/repl.py (274 lines)

Interactive shell.

**Shell**: REPL implementation
```python
class Shell:
    def __init__(self, world: World):
        self._world = world
        self._namespace = self._setup_namespace()
    
    def run(self, prompt=">>> "):
        while self._running:
            line = input(prompt)
            result = self.execute(line)
            if result is not None:
                print(result)
    
    def execute(self, code: str) -> Any:
        try:
            return eval(code, self._namespace)
        except SyntaxError:
            exec(code, self._namespace)
            return None
```

**Feedback**: Operation echo system
```python
class Feedback:
    def echo(self, message: str): ...
    def error(self, message: str): ...
    def success(self, message: str): ...
```

**Namespace Setup**: Pre-populated with:
- World instance
- Sugar functions (mark, rewind, undo, redo)
- Type queries for registered components

## Data Flow

```
Trinity World <-- TrinityWorldAdapter --> ShellLang World
                         |
                         v
                    Change Events
                         |
            +------------+------------+
            |            |            |
            v            v            v
        Inspector    AI Interface   REPL
```

## Dependencies

Layer 4 depends on:
- Layer 3: inspector (for inspect command), shell (for REPL base)
- Layer 2: tracker (change tracking), query (filtering)
- Layer 1: registry (type registration)
- Layer 0: all modules for various utilities
- External: Trinity component system (ComponentMeta)

## Testing Strategy

### Bridge Tests
- get_trinity_registry returns all components
- TrinityWorldAdapter syncs both directions
- Changes propagate correctly
- create_ai_interface returns working interface

### ShellLang Core Tests
- Entity creation and identity
- Component attach/detach/get/has
- World query returns correct entities
- Mutation records Change
- Snapshot save/restore preserves state
- Diff computes correct changes

### ShellLang Sugar Tests
- EntityProxy provides attribute access
- ComponentProxy reads and writes fields
- QueryResult chaining works
- TypeQuery returns correct entities
- TimeManager mark/rewind works

### ShellLang AI Tests
- All 10 operations execute correctly
- Validation rejects invalid commands
- Dry run previews without mutation
- Error responses are structured

### ShellLang REPL Tests
- Expression evaluation returns value
- Statement execution modifies namespace
- Namespace includes world and sugar
- Feedback echoes correctly
