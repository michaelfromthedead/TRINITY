# PHASE 4 ARCHITECTURE: Dialogue Graph System

**Scope**: Dialogue graph structure, node types, validation, conditions  
**Files**: `dialogue.py`, `dialogue_conditions.py`  
**Lines**: ~2,531

---

## Architecture Overview

Phase 4 covers the dialogue graph system that drives non-linear narrative conversations in the quest subsystem.

```
                    +------------------+
                    |  DialogueGraph   |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
   +-----v-----+       +-----v-----+       +-----v-----+
   |   Nodes   |       | Validation|       | Conditions|
   +-----------+       +-----------+       +-----------+
         |                   |                   |
   +-----v-----+       +-----v-----+       +-----v-----+
   |   Text    |       |BFS Reach  |       |  Compare  |
   |  Choice   |       |Cycle Detect|      |   Item    |
   |  Branch   |       |Orphan Find |      |   Quest   |
   |  Event    |       +-----------+       | Reputation|
   |  Random   |                           | Compound  |
   |   Entry   |                           +-----------+
   |   Exit    |
   +-----------+
```

---

## Component Architecture

### Dialogue Graph (`dialogue.py`)

```
DialogueGraph
├── nodes: Dict[NodeId, DialogueNode]
├── entry_node: NodeId
├── add_node(node) -> NodeId
├── remove_node(node_id)
├── connect(from_id, to_id, condition?)
├── validate() -> List[ValidationError]
├── find_path(from_id, to_id) -> List[NodeId]
└── get_reachable(from_id) -> Set[NodeId]

DialogueNode (base)
├── id: NodeId
├── edges: List[Edge]
└── metadata: Dict

TextNode extends DialogueNode
├── speaker: str
├── text: str
└── portrait: str?

ChoiceNode extends DialogueNode
├── choices: List[Choice]
└── Choice
    ├── text: str
    ├── condition: Condition?
    └── target: NodeId

BranchNode extends DialogueNode
├── branches: List[Branch]
└── Branch
    ├── condition: Condition
    └── target: NodeId

EventNode extends DialogueNode
├── event_type: str
└── event_data: Dict

RandomNode extends DialogueNode
├── weights: List[float]
└── targets: List[NodeId]

EntryNode extends DialogueNode
└── (no additional fields)

ExitNode extends DialogueNode
└── (no additional fields)
```

### DialogueGraphBuilder

```
DialogueGraphBuilder
├── entry() -> Self
├── text(speaker, text) -> Self
├── choice(choices) -> Self
├── branch(conditions) -> Self
├── event(type, data) -> Self
├── random(weights) -> Self
├── connect_to(node_id) -> Self
├── exit() -> Self
└── build() -> DialogueGraph
```

### Condition System (`dialogue_conditions.py`)

```
Condition (base)
├── evaluate(context) -> bool
└── __and__, __or__, __invert__ (operator overloading)

VariableCondition extends Condition
├── variable: str
├── operator: ==, !=, <, >, <=, >=, contains, regex
└── value: Any

HasItem extends Condition
├── item_id: str
└── quantity: int

HasQuest extends Condition
├── quest_id: str
└── state: QuestState?

QuestState extends Condition
├── quest_id: str
└── state: str

ReputationCondition extends Condition
├── faction: str
├── operator: ==, !=, <, >, <=, >=
└── value: int

And extends Condition
├── left: Condition
└── right: Condition

Or extends Condition
├── left: Condition
└── right: Condition

Not extends Condition
└── condition: Condition

Xor extends Condition
├── left: Condition
└── right: Condition
```

---

## Algorithm Details

### BFS Reachability Analysis

```
Input: DialogueGraph, start_node_id
  |
  v
Initialize: visited = {}, queue = [start_node_id]
  |
  v
Loop:
  node = queue.pop()
  if node in visited: continue
  visited.add(node)
  for edge in node.edges:
    queue.append(edge.target)
  |
  v
Output: visited (all reachable nodes)
```

**Use Cases**:
- Find orphaned nodes: nodes not reachable from entry
- Validate graph connectivity
- Generate dialogue previews

### Cycle Detection

```
Input: DialogueGraph
  |
  v
DFS with color marking:
  - White: unvisited
  - Gray: in current path
  - Black: fully processed
  |
  v
Cycle exists if we visit a Gray node
  |
  v
Output: List of cycle-forming edges
```

**Note**: Cycles may be valid (player can re-enter dialogue sections). Detection is for author awareness, not automatic rejection.

### Condition Evaluation (Short-Circuit)

```
And.evaluate(context):
  if not left.evaluate(context):
    return False  # Short-circuit
  return right.evaluate(context)

Or.evaluate(context):
  if left.evaluate(context):
    return True   # Short-circuit
  return right.evaluate(context)
```

### Operator Overloading

```python
# condition_a & condition_b
def __and__(self, other: Condition) -> And:
    return And(self, other)

# condition_a | condition_b
def __or__(self, other: Condition) -> Or:
    return Or(self, other)

# ~condition
def __invert__(self) -> Not:
    return Not(self)
```

**Result**: DSL-like condition composition:
```python
can_enter = HasItem("key") & ~QuestState("door", "locked") | HasSkill("lockpick", 5)
```

---

## Architectural Decisions

### ADR-QST-001: Graph-Based Dialogue

**Context**: Dialogue must support branching, conditions, and non-linear flow.

**Decision**: Use directed graph with typed nodes.

**Rationale**:
- Graphs naturally represent branching narrative
- Node types capture different dialogue behaviors
- Edges represent transitions (may be conditional)
- Maps to visual node editor tools

**Consequences**:
- Must validate graph connectivity
- Cycles are allowed but flagged
- Serialization must preserve graph structure

### ADR-QST-002: Typed Node Classes

**Context**: Different dialogue behaviors need different data.

**Decision**: Use inheritance hierarchy with specific node types.

**Rationale**:
- TextNode: Simple speaker/text
- ChoiceNode: Player choices with conditions
- BranchNode: Automatic branching based on conditions
- EventNode: Triggers external systems
- RandomNode: Weighted random selection
- Entry/Exit: Graph boundaries

**Consequences**:
- Must handle each type in traversal
- New types require code changes
- Type safety in graph operations

### ADR-QST-003: Operator Overloading for Conditions

**Context**: Condition composition should be readable.

**Decision**: Overload &, |, ~ operators.

**Rationale**:
- Pythonic DSL for conditions
- Readable by designers
- Composable arbitrarily deep

**Consequences**:
- Operator precedence follows Python (& before |)
- Must document for non-Python authors
- Serialization must flatten expression tree

### ADR-QST-004: Short-Circuit Evaluation

**Context**: Compound conditions should optimize.

**Decision**: Implement short-circuit for And/Or.

**Rationale**:
- Standard boolean logic optimization
- Avoids expensive evaluations when unnecessary
- Matches programmer expectations

**Consequences**:
- Condition order matters for performance
- Side effects in conditions would be problematic (avoid)

### ADR-QST-005: Fluent Builder Pattern

**Context**: Programmatic dialogue construction should be convenient.

**Decision**: DialogueGraphBuilder with method chaining.

**Rationale**:
- Readable construction code
- Enforces valid construction order
- `build()` can validate before returning

**Consequences**:
- Builder is mutable during construction
- Error handling in build step

---

## Validation Rules

### Structural Validation

| Rule | Severity | Description |
|------|----------|-------------|
| Has entry node | Error | Graph must have exactly one entry |
| Has exit node | Error | Graph must have at least one exit |
| No orphans | Warning | All nodes reachable from entry |
| No dead ends | Warning | All non-exit nodes have outgoing edges |
| Cycles annotated | Info | Cycles flagged for author review |

### Semantic Validation

| Rule | Severity | Description |
|------|----------|-------------|
| Conditions reference valid variables | Error | Variable names must exist |
| Items reference valid item IDs | Error | Item IDs must be in database |
| Quests reference valid quest IDs | Error | Quest IDs must be in database |
| Choice text not empty | Warning | Choices should have visible text |
| Random weights sum to 1 | Warning | Probabilities should normalize |

---

## Performance Considerations

### Graph Operations

- **Node lookup**: O(1) via dict by ID
- **Edge iteration**: O(E) where E is edges from node
- **Reachability**: O(V + E) BFS
- **Path finding**: O(V + E) BFS/DFS

### Condition Evaluation

- **Simple conditions**: O(1) variable lookup
- **Compound conditions**: O(depth) with short-circuit
- **Avoid**: Deep nesting, expensive sub-conditions

### Memory

- **Graph storage**: Proportional to V + E
- **Condition trees**: Proportional to condition complexity
- **Consider**: Condition caching for repeated evaluation
