# PHASE 6 ARCHITECTURE: Objectives and Quest Flow

**Scope**: Objective tracking, state machines, composite objectives, flow patterns  
**Files**: `objectives.py`, `quest_flow.py`  
**Lines**: ~1,803

---

## Architecture Overview

Phase 6 covers the objective tracking system and quest flow control that structure quest progression.

```
                    +------------------+
                    |      Quest       |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                                       |
   +-----v-----+                          +------v------+
   | Objectives|                          |    Flow     |
   +-----------+                          +-------------+
         |                                       |
   +-----v-----+                          +------v------+
   |   State   |                          | Sequential  |
   |  Machine  |                          |  Parallel   |
   +-----------+                          |  Branching  |
         |                                |  Optional   |
   +-----v-----+                          |   Mixed     |
   |   Kill    |                          +-------------+
   |  Collect  |
   |   Talk    |
   |   Reach   |
   |  Escort   |
   | Composite |
   +-----------+
```

---

## Component Architecture

### Objective System (`objectives.py`)

```
ObjectiveState (enum)
├── INACTIVE
├── IN_PROGRESS
├── COMPLETE
└── FAILED

Objective (base)
├── id: str
├── state: ObjectiveState
├── description: str
├── activate()
├── update(event)
├── complete()
├── fail()
├── get_progress() -> (current, total)
└── on_state_change(callback)

KillObjective extends Objective
├── target_type: str
├── required_count: int
├── current_count: int
├── streak_tracking: bool
└── streak_count: int

CollectObjective extends Objective
├── item_id: str
├── required_count: int
├── current_count: int
└── auto_remove: bool  # Remove items on completion

TalkObjective extends Objective
├── npc_id: str
└── talked: bool

ReachObjective extends Objective
├── location: Vector3
├── radius: float
├── stay_duration: float  # Optional
└── time_in_area: float

EscortObjective extends Objective
├── npc_id: str
├── destination: Vector3
├── npc_alive: bool
└── npc_at_destination: bool

InteractObjective extends Objective
├── object_id: str
└── interacted: bool

UseObjective extends Objective
├── ability_id: str
├── required_uses: int
└── current_uses: int

CraftObjective extends Objective
├── recipe_id: str
├── required_count: int
└── current_count: int

DefendObjective extends Objective
├── location: Vector3
├── radius: float
├── duration: float
├── elapsed: float
└── enemies_in_area: int

TimedObjective extends Objective
├── wrapped: Objective
├── time_limit: float
└── elapsed: float

CompositeObjective extends Objective
├── objectives: List[Objective]
├── mode: "all" | "any" | "sequential"
└── _current_index: int  # For sequential
```

### Quest Flow System (`quest_flow.py`)

```
FlowNode (base)
├── id: str
├── objectives: List[Objective]
├── children: List[FlowNode]
├── is_complete() -> bool
├── is_active() -> bool
├── get_current_objectives() -> List[Objective]
└── advance()

SequentialFlow extends FlowNode
├── current_index: int
└── advance() -> Move to next child

ParallelFlow extends FlowNode
├── require_all: bool
└── advance() -> Complete when condition met

BranchingFlow extends FlowNode
├── branches: Dict[str, FlowNode]
├── selected_branch: str | None
├── select_branch(choice: str)
└── auto_advance: bool

OptionalFlow extends FlowNode
├── bonus_objectives: List[Objective]
├── bonus_rewards: List[Reward]
└── is_optional: True

MixedFlow extends FlowNode
├── (combines all patterns)
└── children: List[FlowNode]  # May be different types

FlowBuilder
├── sequential() -> Self
├── parallel(require_all=True) -> Self
├── branch(choices) -> Self
├── optional(objectives, rewards) -> Self
├── add_objective(objective) -> Self
├── add_child(flow) -> Self
└── build() -> FlowNode
```

---

## Algorithm Details

### Objective State Machine

```
             activate()
INACTIVE ─────────────────> IN_PROGRESS
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              complete()                     fail()
                    │                           │
                    v                           v
               COMPLETE                      FAILED
```

**Transitions**:
- `activate()`: INACTIVE -> IN_PROGRESS
- `complete()`: IN_PROGRESS -> COMPLETE
- `fail()`: IN_PROGRESS -> FAILED

**Invalid Transitions**:
- Cannot activate from COMPLETE/FAILED
- Cannot complete/fail from INACTIVE
- No transitions from terminal states

### Composite Objective Evaluation

```
mode: "all"
  is_complete = all(o.state == COMPLETE for o in objectives)
  is_failed = any(o.state == FAILED for o in objectives)

mode: "any"
  is_complete = any(o.state == COMPLETE for o in objectives)
  is_failed = all(o.state == FAILED for o in objectives)

mode: "sequential"
  is_complete = _current_index >= len(objectives)
  is_failed = objectives[_current_index].state == FAILED
  
  on child complete:
    _current_index += 1
    if _current_index < len(objectives):
      objectives[_current_index].activate()
```

### Flow Traversal

```
SequentialFlow:
  get_current_objectives():
    return children[current_index].get_current_objectives()
  
  advance():
    if children[current_index].is_complete():
      current_index += 1
      if current_index < len(children):
        children[current_index].activate()

ParallelFlow:
  get_current_objectives():
    result = []
    for child in children:
      if child.is_active():
        result.extend(child.get_current_objectives())
    return result
  
  advance():
    if require_all:
      if all(c.is_complete() for c in children):
        mark_complete()
    else:
      if any(c.is_complete() for c in children):
        mark_complete()

BranchingFlow:
  select_branch(choice):
    selected_branch = choice
    branches[choice].activate()
  
  get_current_objectives():
    if selected_branch:
      return branches[selected_branch].get_current_objectives()
    return []  # No branch selected yet
```

---

## Architectural Decisions

### ADR-QST-011: Explicit State Machine

**Context**: Objectives have clear lifecycle states.

**Decision**: Use enum-based state machine with explicit transitions.

**Rationale**:
- States are mutually exclusive
- Transitions are auditable
- Invalid transitions are rejected
- State change events are hookable

**Consequences**:
- No partial completion (use progress instead)
- Terminal states are final
- Must track progress separately from state

### ADR-QST-012: Typed Objective Classes

**Context**: Different objectives track different data.

**Decision**: Inheritance hierarchy with specific objective types.

**Rationale**:
- KillObjective needs kill count + streak
- ReachObjective needs location + duration
- Each type has specific completion logic

**Consequences**:
- Adding new types requires code
- Type-specific update handling
- Serialization must handle all types

### ADR-QST-013: Composite Pattern for Objectives

**Context**: Quest objectives often group (all/any/sequential).

**Decision**: CompositeObjective wraps multiple objectives.

**Rationale**:
- Naturally represents objective trees
- Mode determines completion rule
- Recursive composition possible

**Consequences**:
- Must propagate events to children
- State depends on children states
- UI must handle composite display

### ADR-QST-014: Named Flow Patterns

**Context**: Quest structure has common patterns.

**Decision**: Named flow types (Sequential, Parallel, Branching, Optional, Mixed).

**Rationale**:
- Self-documenting quest structure
- Tool support for authoring
- Clear semantics per pattern

**Consequences**:
- New patterns require new classes
- Mixed allows arbitrary nesting
- Must validate flow structure

### ADR-QST-015: Fluent FlowBuilder

**Context**: Programmatic flow construction should be convenient.

**Decision**: FlowBuilder with method chaining.

**Rationale**:
- Readable construction code
- Enforces valid nesting
- build() can validate

**Consequences**:
- Nested builders for nested flows
- Error handling in build step

---

## Objective Types Reference

### Combat Objectives

| Type | Tracking | Completion |
|------|----------|------------|
| KillObjective | Kill count, streak | Count >= required |
| DefendObjective | Time in area, enemies | Duration met, no breach |

### Collection Objectives

| Type | Tracking | Completion |
|------|----------|------------|
| CollectObjective | Item count | Count >= required |
| CraftObjective | Craft count | Count >= required |

### Interaction Objectives

| Type | Tracking | Completion |
|------|----------|------------|
| TalkObjective | Talked flag | Talked == true |
| InteractObjective | Interacted flag | Interacted == true |
| UseObjective | Use count | Count >= required |

### Movement Objectives

| Type | Tracking | Completion |
|------|----------|------------|
| ReachObjective | Position, time | In area for duration |
| EscortObjective | NPC status, position | NPC alive + at dest |

### Meta Objectives

| Type | Tracking | Completion |
|------|----------|------------|
| TimedObjective | Wrapped + time | Wrapped complete before limit |
| CompositeObjective | Children states | Mode-dependent (all/any/seq) |

---

## Flow Patterns Reference

### SequentialFlow

```
[Objective A] -> [Objective B] -> [Objective C]
```

- Objectives activate one at a time
- Must complete in order
- Quest fails if any objective fails

### ParallelFlow

```
[Objective A]
[Objective B]  (any order)
[Objective C]
```

- All objectives active simultaneously
- Complete when require_all met (or any complete)
- More flexible than sequential

### BranchingFlow

```
         ┌─> [Branch A]
[Choice] ┼─> [Branch B]
         └─> [Branch C]
```

- Player selects branch
- Only selected branch executes
- Unused branches never activate

### OptionalFlow

```
[Main Objectives]
  └─ [Bonus Objectives] (optional)
```

- Main objectives required
- Bonus objectives for extra rewards
- No penalty for skipping bonus

### MixedFlow

```
[Sequential]
  ├─ [Parallel]
  │    ├─ [Objective]
  │    └─ [Objective]
  └─ [Branching]
       ├─ [Branch A]
       └─ [Branch B]
```

- Arbitrary nesting
- Combines all patterns
- Most flexible, most complex

---

## Performance Considerations

### Objective Updates

- **Event filtering**: Only update relevant objectives per event
- **State caching**: Cache computed state, invalidate on change
- **Batch updates**: Group multiple events before recalculating

### Flow Traversal

- **Active tracking**: Maintain set of active nodes
- **Lazy evaluation**: Only compute current objectives on demand
- **Index caching**: Cache current index for sequential

### Memory

- **Objective pooling**: Reuse objective objects
- **Flow flattening**: Consider flattening for simple quests
- **Event cleanup**: Remove completed objective listeners
