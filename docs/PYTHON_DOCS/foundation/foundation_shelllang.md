# Investigation: foundation/shelllang

## Summary
ShellLang is a fully implemented minimal ECS shell providing dual-interface access for humans (fluent Python DSL) and AI (structured JSON commands). The 5 semantic primitives (Entity, Component, Query, Mutate, Snapshot) are all real, working implementations with 1174 lines of tests proving functionality.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 63 | REAL | Exports all public API |
| `core.py` | 396 | REAL | World, Entity, Snapshot, Change - full ECS core |
| `sugar.py` | 542 | REAL | EntityProxy, QueryResult, TypeQuery, TimeManager |
| `ai.py` | 516 | REAL | AIInterface with execute/validate/dry_run |
| `repl.py` | 275 | REAL | Shell, Feedback - interactive REPL |

## ShellLang Components

### Core (core.py)
- **World**: ECS container holding all entities and components
- **Entity**: Lightweight uint64 identifier handle
- **Component**: Any Python class (type alias)
- **Snapshot**: Frozen world state for save/restore
- **Change**: Recorded mutation for undo/redo

### Sugar Layer (sugar.py)
- **EntityProxy**: Dot access to components (`e.health.current`)
- **ComponentProxy**: Tracks field access and mutation
- **QueryResult**: Chainable query results with fluent interface
- **TypeQuery**: Type-based query entry point (`Enemy.all`)
- **TimeManager**: Named snapshots and undo/redo

### AI Interface (ai.py)
- **AIInterface**: Structured JSON command execution
  - `execute()`: Run a command
  - `validate()`: Validate without executing
  - `dry_run()`: Preview effects

### REPL (repl.py)
- **Shell**: Interactive Python REPL with namespace setup
- **Feedback**: Echo system for operation feedback

## The 5 Primitives

1. **ENTITY** - uint64 identifier (line 33-51 in core.py)
   - Lightweight handle, hashable, comparable
   - IDs auto-increment from 1

2. **COMPONENT** - typed data attached to entity (line 55 in core.py)
   - Any Python class can be a component
   - Attached via `world.attach(entity, component)`

3. **QUERY** - entity predicate -> [entity] (line 219-238 in core.py)
   - `world.query(*ComponentTypes)` returns entities with all specified components
   - Sugar: `Enemy.all.where(lambda e: e.health.current < 50)`

4. **MUTATE** - (entity, field, value) -> tracked change (line 180-198 in core.py)
   - `world.set(entity, ComponentType, field_name, value)`
   - All mutations recorded as Change objects

5. **SNAPSHOT** - frozen world state (line 74-87, 244-278 in core.py)
   - `world.snap(name)` creates immutable snapshot
   - `world.restore(snapshot)` reverts to snapshot
   - `world.diff(a, b)` computes changes between snapshots

## Implementation

- **Real REPL?** YES - Shell class (repl.py:83-258) has `run()` method with input() loop, help system, expression/statement evaluation via Python's eval()/exec()
- **Real eval/exec?** YES - Shell.execute() (repl.py:159-188) uses Python's `eval()` for expressions and `exec()` for statements
- **Real AI interface?** YES - AIInterface class (ai.py:51-504) with 10 operations (query, set, spawn, destroy, snap, restore, inspect, schema, list_types, count)

## Verdict
**REAL IMPLEMENTATION**

This is a complete, production-ready ECS shell with:
- 1792 total lines of implementation
- 1174 lines of comprehensive tests (82 test cases)
- Dual interface (human-friendly sugar + AI-friendly JSON)
- Full undo/redo with named snapshots
- Change tracking for debugging

## Evidence

### Real Entity Creation (core.py:114-120)
```python
def create(self) -> Entity:
    """Create a new entity and return its handle."""
    entity_id = self._next_entity_id
    self._next_entity_id += 1
    self._entities[entity_id] = set()
    self._components[entity_id] = {}
    return Entity(entity_id)
```

### Real Query Implementation (core.py:222-238)
```python
def query(self, *Cs: Type) -> List[Entity]:
    if not Cs:
        return [Entity(eid) for eid in self._entities]

    required = {C.__name__ for C in Cs}
    results = []

    for entity_id, component_names in self._entities.items():
        if required <= component_names:
            results.append(Entity(entity_id))

    return results
```

### Real REPL Loop (repl.py:190-212)
```python
def run(self, prompt: str = DEFAULT_PROMPT) -> None:
    self._running = True
    print("ShellLang REPL. Type 'help' for commands, 'quit' to exit.")

    while self._running:
        try:
            line = input(prompt)
            result = self.execute(line)
            if result is not None:
                print(result)
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        ...
```

### Real AI Command Execution (ai.py:67-100)
```python
def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
    op = command.get("op")

    if op == "query":
        return self._query(command)
    elif op == "set":
        return self._set(command)
    elif op == "spawn":
        return self._spawn(command)
    ...
```

### Fluent Sugar DSL (sugar.py)
```python
# Human-friendly syntax enabled by sugar layer:
enemies.where(lambda e: e.health.current < 50).near(player, 10).set(health__current=0)
mark("before_fight")
rewind("before_fight")
```
