# PHASE 5 ARCHITECTURE: Dialogue Effects and Variables

**Scope**: Transactional effects, rollback pattern, variable scoping  
**Files**: `dialogue_effects.py`, `dialogue_variables.py`  
**Lines**: ~2,428

---

## Architecture Overview

Phase 5 covers the effect system that modifies game state from dialogue, and the variable system that tracks dialogue/quest state across scopes.

```
                    +------------------+
                    |  Dialogue Event  |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                                       |
   +-----v-----+                          +------v------+
   |  Effects  |                          |  Variables  |
   +-----------+                          +-------------+
         |                                       |
   +-----v-----+                          +------v------+
   |  Execute  |                          |    Local    |
   |  Rollback |                          |   Global    |
   |   Batch   |                          |    Quest    |
   +-----------+                          +-------------+
         |                                       |
   +-----v-----+                          +------v------+
   | Variable  |                          |  Observers  |
   |   Item    |                          |   History   |
   |   Quest   |                          +-------------+
   |Reputation |
   |   Event   |
   +-----------+
```

---

## Component Architecture

### Effect System (`dialogue_effects.py`)

```
Effect (base)
├── execute(context: EffectContext) -> bool
├── rollback(context: EffectContext) -> None
└── description() -> str

VariableEffect extends Effect
├── SetVariable
│   ├── variable: str
│   ├── value: Any
│   └── _previous_value: Any  # For rollback
├── IncrementVariable
│   ├── variable: str
│   └── amount: int
└── DecrementVariable
    ├── variable: str
    └── amount: int

ItemEffect extends Effect
├── GiveItem
│   ├── item_id: str
│   └── quantity: int
└── TakeItem
    ├── item_id: str
    └── quantity: int

QuestEffect extends Effect
├── SetQuestState
│   ├── quest_id: str
│   ├── state: str
│   └── _previous_state: str
├── SetQuestProgress
│   ├── quest_id: str
│   └── progress: int
├── StartQuest
│   └── quest_id: str
├── CompleteQuest
│   └── quest_id: str
└── FailQuest
    └── quest_id: str

ReputationEffect extends Effect
├── ChangeReputation
│   ├── faction: str
│   └── amount: int
└── SetReputation
    ├── faction: str
    ├── value: int
    └── _previous_value: int

EventEffect extends Effect
├── TriggerEvent
│   ├── event_type: str
│   └── event_data: Dict
├── PlaySound
│   └── sound_id: str
├── PlayAnimation
│   ├── target: str
│   └── animation_id: str
└── StartDialogue
    └── dialogue_id: str

EffectBatch
├── effects: List[Effect]
├── execute(context) -> bool
│   └── Rollback all on any failure
└── rollback(context) -> None
```

### Variable System (`dialogue_variables.py`)

```
VariableStore (base)
├── get(name: str) -> Any
├── set(name: str, value: Any)
├── has(name: str) -> bool
├── delete(name: str)
└── get_all() -> Dict[str, Any]

LocalVariableStore extends VariableStore
├── variables: Dict[str, Any]
├── dialogue_id: str
└── (ephemeral, per-conversation)

GlobalVariableStore extends VariableStore
├── variables: Dict[str, Any]
├── increment(name: str, amount: int)
├── toggle(name: str) -> bool
├── append_to_list(name: str, value: Any)
├── remove_from_list(name: str, value: Any)
├── get_change_history(name: str) -> List[Change]
├── add_observer(name: str, callback)
└── (persistent, game-wide)

QuestVariableStore extends VariableStore
├── variables: Dict[str, Any]
├── quest_id: str
├── sync_to_quest_state()
├── sync_from_quest_state()
└── (linked to quest lifecycle)

VariableManager
├── local: LocalVariableStore
├── global_: GlobalVariableStore
├── quest: Dict[str, QuestVariableStore]
├── resolve(name: str) -> Any
│   └── Check local -> quest -> global
└── set(name: str, value: Any, scope: Scope)
```

---

## Algorithm Details

### Transactional Execute/Rollback

```
Input: EffectBatch with effects [E1, E2, E3]
  |
  v
Execute Loop:
  executed = []
  for effect in effects:
    if effect.execute(context):
      executed.append(effect)
    else:
      # Failure: rollback all executed
      for e in reversed(executed):
        e.rollback(context)
      return False
  return True
```

**Rollback Mechanism**:
- Effects store previous state on execute
- Rollback restores previous state
- Order reversed (last executed, first rolled back)

### Variable Resolution

```
Input: Variable name (may be scoped or unscoped)
  |
  v
Parse Scope Prefix:
  "local:foo"  -> LocalVariableStore
  "global:foo" -> GlobalVariableStore
  "quest:foo"  -> QuestVariableStore
  "foo"        -> Resolution chain
  |
  v
Resolution Chain (unscoped):
  1. Check LocalVariableStore
  2. Check current QuestVariableStore
  3. Check GlobalVariableStore
  4. Return None if not found
  |
  v
Output: Variable value
```

### Observer Pattern

```
GlobalVariableStore.set(name, value):
  old_value = variables.get(name)
  variables[name] = value
  
  # Notify observers
  for observer in observers.get(name, []):
    observer.on_change(name, old_value, value)
  
  # Record history
  history[name].append(Change(timestamp, old_value, value))
```

---

## Architectural Decisions

### ADR-QST-006: Execute/Rollback Pattern

**Context**: Effects must either all succeed or all fail (atomic).

**Decision**: Each effect has `execute()` and `rollback()` methods.

**Rationale**:
- Explicit rollback logic per effect type
- Batch execution can rollback partial progress
- No need for external transaction manager

**Consequences**:
- Effects must track previous state
- Rollback must be idempotent
- Cannot rollback external side effects (sounds, animations)

### ADR-QST-007: Three Variable Scopes

**Context**: Variables have different lifecycles and visibility.

**Decision**: Local, Global, and Quest scopes.

**Rationale**:
- **Local**: Ephemeral dialogue state (choices made this conversation)
- **Global**: Persistent world state (flags, counters)
- **Quest**: Quest-specific state (synced with quest system)

**Consequences**:
- Must resolve scope for unscoped access
- Quest variables must sync bidirectionally
- Save/load must handle all scopes

### ADR-QST-008: Resolution Chain for Unscoped Variables

**Context**: Authors may not always specify scope.

**Decision**: Check Local -> Quest -> Global in order.

**Rationale**:
- Most specific scope takes precedence
- Common pattern in programming languages
- Reduces boilerplate in dialogue scripts

**Consequences**:
- Shadowing can cause confusion
- Should warn on shadowed variables

### ADR-QST-009: Observer Pattern for Variables

**Context**: UI and other systems need to react to variable changes.

**Decision**: Observer pattern with callbacks.

**Rationale**:
- Loose coupling between variable store and consumers
- Multiple observers per variable
- No polling needed

**Consequences**:
- Must manage observer lifecycle (unsubscribe)
- Observers must not mutate during notification

### ADR-QST-010: Change History Tracking

**Context**: Debugging and some game features need variable history.

**Decision**: Record all changes with timestamps.

**Rationale**:
- Debug: trace how variable reached current value
- Features: "has this ever been true?" queries
- Save replay: recreate state from history

**Consequences**:
- Memory cost grows with changes
- May need pruning for long sessions
- History itself must serialize

---

## Effect Types Reference

### Variable Effects

| Effect | Execute | Rollback |
|--------|---------|----------|
| SetVariable | Store new value | Restore previous |
| IncrementVariable | Add amount | Subtract amount |
| DecrementVariable | Subtract amount | Add amount |

### Item Effects

| Effect | Execute | Rollback |
|--------|---------|----------|
| GiveItem | Add to inventory | Remove from inventory |
| TakeItem | Remove from inventory | Add to inventory |

### Quest Effects

| Effect | Execute | Rollback |
|--------|---------|----------|
| SetQuestState | Change state | Restore previous state |
| SetQuestProgress | Change progress | Restore previous progress |
| StartQuest | Activate quest | Deactivate quest |
| CompleteQuest | Mark complete | Mark in-progress |
| FailQuest | Mark failed | Mark in-progress |

### Reputation Effects

| Effect | Execute | Rollback |
|--------|---------|----------|
| ChangeReputation | Add/subtract | Reverse add/subtract |
| SetReputation | Set value | Restore previous |

### Event Effects

| Effect | Execute | Rollback |
|--------|---------|----------|
| TriggerEvent | Fire event | (Cannot rollback) |
| PlaySound | Play audio | (Cannot rollback) |
| PlayAnimation | Start animation | (Cannot rollback) |
| StartDialogue | Open dialogue | (Cannot rollback) |

**Note**: Event effects are non-reversible. Batch containing them should place them last.

---

## Performance Considerations

### Effect Execution

- **Batch size**: Keep batches small for faster rollback
- **State capture**: Store minimal state for rollback
- **Validation**: Validate before execute to avoid rollback

### Variable Access

- **Scope resolution**: Cache resolved scope for repeated access
- **History**: Limit history depth or prune periodically
- **Observers**: Batch notifications to avoid callback storms

### Memory

- **Local stores**: Clear when conversation ends
- **History pruning**: Configurable depth limit
- **Observer cleanup**: Remove dead references
