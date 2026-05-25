# PHASE 4 TODO: Layer 4 Integration and ShellLang

## T-FND-4.1: Trinity Bridge Functions

### Description
Implement bridge functions connecting Trinity to Foundation.

### Tasks
- [ ] Implement get_trinity_registry() function
- [ ] Pull ComponentMeta.all_components() into Foundation registry
- [ ] Implement create_ai_interface(world) function
- [ ] Return configured AIInterface instance
- [ ] Handle missing Trinity gracefully (optional dependency)

### Acceptance Criteria
- get_trinity_registry returns dict of component names to types
- All Trinity components registered in Foundation registry
- create_ai_interface returns working AIInterface
- Works without Trinity installed (returns empty/stub)

---

## T-FND-4.2: Trinity World Adapter

### Description
Implement bidirectional sync between Trinity and ShellLang worlds.

### Tasks
- [ ] Implement TrinityWorldAdapter class
- [ ] Implement __init__(trinity_world, shelllang_world)
- [ ] Implement sync_to_shelllang() copying state
- [ ] Implement sync_from_shelllang() copying state
- [ ] Implement on_trinity_change() handler
- [ ] Implement on_shelllang_change() handler
- [ ] Register change handlers on both worlds

### Acceptance Criteria
- sync_to_shelllang copies all entities and components
- sync_from_shelllang copies changes back
- Change handlers fire on modifications
- No infinite loops from sync
- Handles entity creation/destruction

---

## T-FND-4.3: ShellLang Entity

### Description
Implement lightweight entity identifier.

### Tasks
- [ ] Implement Entity class with __slots__
- [ ] Store single int _id
- [ ] Implement __hash__ returning hash(_id)
- [ ] Implement __eq__ comparing _id
- [ ] Implement __repr__ for debugging
- [ ] Implement __lt__ for sorting

### Acceptance Criteria
- Entity is hashable (usable as dict key)
- Entity equality based on _id
- Entity is comparable for sorting
- Memory efficient (single int)
- repr shows entity ID

---

## T-FND-4.4: ShellLang World

### Description
Implement ECS world container.

### Tasks
- [ ] Implement World class
- [ ] Implement create() -> Entity
- [ ] Implement destroy(entity)
- [ ] Implement attach(entity, component)
- [ ] Implement detach(entity, ComponentType)
- [ ] Implement get(entity, ComponentType)
- [ ] Implement has(entity, ComponentType)
- [ ] Implement set(entity, ComponentType, field, value)
- [ ] Implement query(*ComponentTypes)
- [ ] Track changes in _history list

### Acceptance Criteria
- create returns new Entity with unique ID
- destroy removes entity and all components
- attach adds component instance to entity
- detach removes component from entity
- get returns component instance or raises
- has returns bool
- set mutates field and records Change
- query returns entities with all specified components

---

## T-FND-4.5: ShellLang Change Tracking

### Description
Implement Change dataclass for mutation tracking.

### Tasks
- [ ] Implement Change dataclass
- [ ] Fields: entity, component_name, field, old_value, new_value, tick
- [ ] Record Change on every World.set() call
- [ ] Store tick number for temporal ordering
- [ ] Implement __repr__ for debugging

### Acceptance Criteria
- Every mutation creates Change record
- old_value captured before mutation
- new_value captured after mutation
- tick monotonically increases
- Changes stored in World._history

---

## T-FND-4.6: ShellLang Snapshot

### Description
Implement frozen world state snapshots.

### Tasks
- [ ] Implement Snapshot class
- [ ] Implement World.snap(name=None) creating snapshot
- [ ] Deep copy all entity/component data
- [ ] Store tick number at snapshot time
- [ ] Implement World.restore(snapshot)
- [ ] Implement World.diff(a, b) -> List[Change]

### Acceptance Criteria
- snap creates immutable copy of world state
- restore reverts world to snapshot state
- diff computes changes between two snapshots
- Named snapshots retrievable by name
- Snapshot tick matches creation time

---

## T-FND-4.7: EntityProxy

### Description
Implement dot access proxy for entities.

### Tasks
- [ ] Implement EntityProxy class
- [ ] Implement __init__(entity, world)
- [ ] Implement __getattr__ returning ComponentProxy
- [ ] Cache ComponentProxy instances
- [ ] Implement __repr__

### Acceptance Criteria
- e.health returns ComponentProxy for Health
- Attribute access creates proxy on demand
- Same attribute returns same proxy (cached)
- Works with any component name

---

## T-FND-4.8: ComponentProxy

### Description
Implement field access proxy for components.

### Tasks
- [ ] Implement ComponentProxy class
- [ ] Implement __init__(entity, component_name, world)
- [ ] Implement __getattr__ reading field value
- [ ] Implement __setattr__ writing field value
- [ ] Route writes through World.set() for tracking
- [ ] Implement __repr__

### Acceptance Criteria
- e.health.current reads Health.current field
- e.health.current = 50 writes via World.set()
- Writes are tracked as Changes
- AttributeError for missing fields

---

## T-FND-4.9: QueryResult

### Description
Implement chainable query result wrapper.

### Tasks
- [ ] Implement QueryResult class
- [ ] Implement __init__(entities, world)
- [ ] Implement where(predicate) -> QueryResult
- [ ] Implement near(target, radius) -> QueryResult
- [ ] Implement first() -> Optional[EntityProxy]
- [ ] Implement all() -> List[EntityProxy]
- [ ] Implement count() -> int
- [ ] Implement set(**fields) -> int (bulk update)
- [ ] Implement __iter__ for iteration

### Acceptance Criteria
- where filters by predicate
- near filters by spatial proximity
- first returns first match or None
- all returns list of EntityProxy
- count returns entity count
- set updates all matching entities
- Iteration yields EntityProxy instances

---

## T-FND-4.10: TypeQuery

### Description
Implement type-based query entry point.

### Tasks
- [ ] Implement TypeQuery class
- [ ] Implement __init__(component_type, world)
- [ ] Implement all property returning QueryResult
- [ ] Register TypeQuery on component types
- [ ] Support Enemy.all.where(...) syntax

### Acceptance Criteria
- Enemy.all returns QueryResult for all Enemy entities
- QueryResult contains correct entities
- Chaining works: Enemy.all.where(...).near(...)
- Works with any component type

---

## T-FND-4.11: TimeManager

### Description
Implement named snapshots and undo/redo.

### Tasks
- [ ] Implement TimeManager class
- [ ] Implement mark(name) -> Snapshot
- [ ] Implement rewind(name) restoring snapshot
- [ ] Implement undo(steps=1)
- [ ] Implement redo(steps=1)
- [ ] Implement history() -> List[str]
- [ ] Store named snapshots in dict
- [ ] Maintain undo/redo stacks

### Acceptance Criteria
- mark creates named snapshot
- rewind restores named snapshot
- undo reverts last N changes
- redo reapplies reverted changes
- history lists snapshot names
- New changes clear redo stack

---

## T-FND-4.12: AIInterface Core

### Description
Implement AI command execution interface.

### Tasks
- [ ] Implement AIInterface class
- [ ] Implement __init__(world)
- [ ] Implement execute(command) -> Dict
- [ ] Implement validate(command) -> Dict
- [ ] Implement dry_run(command) -> Dict
- [ ] Route to operation handlers by "op" field

### Acceptance Criteria
- execute runs command and returns result
- validate checks command without executing
- dry_run previews effects without mutation
- Unknown op returns error response
- All responses are JSON-serializable dicts

---

## T-FND-4.13: AIInterface Operations

### Description
Implement all 10 AI command operations.

### Tasks
- [ ] Implement _query(command) operation
- [ ] Implement _set(command) operation
- [ ] Implement _spawn(command) operation
- [ ] Implement _destroy(command) operation
- [ ] Implement _snap(command) operation
- [ ] Implement _restore(command) operation
- [ ] Implement _inspect(command) operation
- [ ] Implement _schema(command) operation
- [ ] Implement _list_types(command) operation
- [ ] Implement _count(command) operation

### Acceptance Criteria
- query returns matching entity IDs
- set mutates field and returns success
- spawn creates entity with components
- destroy removes entity
- snap creates named snapshot
- restore reverts to snapshot
- inspect returns entity's components
- schema returns component field info
- list_types returns all registered types
- count returns entity count for query

---

## T-FND-4.14: AIInterface Validation

### Description
Implement command validation for all operations.

### Tasks
- [ ] Validate op field exists and is string
- [ ] Validate query: components is list of strings
- [ ] Validate set: entity, component, field, value present
- [ ] Validate spawn: components is dict
- [ ] Validate destroy: entity is int
- [ ] Validate snap: name is string (optional)
- [ ] Validate restore: name is string
- [ ] Validate inspect: entity is int
- [ ] Validate schema: component is string
- [ ] Return structured error for invalid commands

### Acceptance Criteria
- Missing required fields return error
- Wrong types return error
- Unknown operations return error
- Valid commands return {valid: true}
- Error messages are descriptive

---

## T-FND-4.15: ShellLang REPL

### Description
Implement interactive shell for ShellLang.

### Tasks
- [ ] Implement Shell class
- [ ] Implement __init__(world)
- [ ] Implement _setup_namespace() with imports
- [ ] Implement run(prompt) with input loop
- [ ] Implement execute(code) with eval/exec
- [ ] Handle KeyboardInterrupt gracefully
- [ ] Implement quit command

### Acceptance Criteria
- run() starts interactive loop
- Expressions evaluated and printed
- Statements executed
- Ctrl+C doesn't crash
- "quit" exits cleanly
- Namespace includes world and sugar

---

## T-FND-4.16: ShellLang Feedback

### Description
Implement operation feedback system.

### Tasks
- [ ] Implement Feedback class
- [ ] Implement echo(message) for info
- [ ] Implement error(message) for errors
- [ ] Implement success(message) for confirmations
- [ ] Color output if terminal supports
- [ ] Integrate with Shell

### Acceptance Criteria
- echo prints neutral message
- error prints with error indicator
- success prints with success indicator
- Works in non-TTY mode (no colors)
- Shell uses Feedback for output
