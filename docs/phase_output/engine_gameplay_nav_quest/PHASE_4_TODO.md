# PHASE 4 TODO: Dialogue Graph System

**Scope**: Dialogue graph structure, node types, validation, conditions  
**Files**: `dialogue.py`, `dialogue_conditions.py`

---

## T-QST-4.1: Verify Dialogue Node Types

**Priority**: P0  
**Estimate**: 1.5 hours

### Description
Verify all dialogue node types correctly store and expose their data.

### Tasks
- [ ] Review TextNode (speaker, text, portrait)
- [ ] Review ChoiceNode (choices list, conditions)
- [ ] Review BranchNode (branch conditions, targets)
- [ ] Review EventNode (event type, data)
- [ ] Review RandomNode (weights, targets)
- [ ] Review Entry/Exit nodes

### Acceptance Criteria
- All node types store their fields correctly
- Edges connect to valid target IDs
- Metadata is accessible
- Serialization round-trips all fields

---

## T-QST-4.2: Verify Graph Operations

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify DialogueGraph add/remove/connect operations work correctly.

### Tasks
- [ ] Review `add_node()` ID generation
- [ ] Review `remove_node()` edge cleanup
- [ ] Review `connect()` edge creation
- [ ] Test conditional edges

### Acceptance Criteria
- Added nodes get unique IDs
- Removed nodes disconnect from graph
- Connections create valid edges
- Conditional edges store condition references

---

## T-QST-4.3: Verify BFS Reachability Analysis

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify `get_reachable()` correctly identifies all reachable nodes.

### Tasks
- [ ] Review BFS implementation
- [ ] Test on linear graph
- [ ] Test on branching graph
- [ ] Test on graph with cycles
- [ ] Test orphan detection (nodes not in reachable set)

### Acceptance Criteria
- All reachable nodes returned
- Cycles do not cause infinite loop
- Orphan nodes identified as not in set
- Performance acceptable for large graphs

---

## T-QST-4.4: Verify Path Finding

**Priority**: P1  
**Estimate**: 1 hour

### Description
Verify `find_path()` returns valid paths between nodes.

### Tasks
- [ ] Review path finding algorithm (BFS/DFS)
- [ ] Test direct connection
- [ ] Test multi-hop path
- [ ] Test no path case
- [ ] Test multiple paths (verify one is returned)

### Acceptance Criteria
- Valid paths returned for connected nodes
- None/empty for disconnected nodes
- Path is actually walkable
- Cycles handled correctly

---

## T-QST-4.5: Verify Cycle Detection

**Priority**: P1  
**Estimate**: 1 hour

### Description
Verify validation correctly detects cycles in dialogue graphs.

### Tasks
- [ ] Review DFS color-marking implementation
- [ ] Test acyclic graph (no cycles detected)
- [ ] Test simple cycle (A->B->A)
- [ ] Test complex cycles (nested)
- [ ] Verify cycle edges reported

### Acceptance Criteria
- Acyclic graphs pass
- Cycles detected and reported
- Cycle-forming edges identified
- No false positives

---

## T-QST-4.6: Verify Condition Operator Overloading

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify &, |, ~ operators compose conditions correctly.

### Tasks
- [ ] Review `__and__` creates And
- [ ] Review `__or__` creates Or
- [ ] Review `__invert__` creates Not
- [ ] Test nested compositions
- [ ] Verify operator precedence

### Acceptance Criteria
- `a & b` creates `And(a, b)`
- `a | b` creates `Or(a, b)`
- `~a` creates `Not(a)`
- `a & b | c` respects precedence (& before |)
- Arbitrarily deep nesting works

---

## T-QST-4.7: Verify Condition Types

**Priority**: P0  
**Estimate**: 1.5 hours

### Description
Verify all condition types evaluate correctly.

### Tasks
- [ ] Review VariableCondition operators (==, !=, <, >, <=, >=, contains, regex)
- [ ] Review HasItem evaluation
- [ ] Review HasQuest evaluation
- [ ] Review QuestState evaluation
- [ ] Review ReputationCondition evaluation
- [ ] Review Xor evaluation

### Acceptance Criteria
- All comparison operators work correctly
- Item checks reference item database
- Quest checks reference quest state
- Reputation checks faction values
- Xor returns true when exactly one operand is true

---

## T-QST-4.8: Verify Short-Circuit Evaluation

**Priority**: P1  
**Estimate**: 0.5 hours

### Description
Verify And/Or short-circuit to avoid unnecessary evaluation.

### Tasks
- [ ] Review And.evaluate() short-circuit
- [ ] Review Or.evaluate() short-circuit
- [ ] Create condition with side effects (for testing)
- [ ] Verify short-circuit prevents side effect execution

### Acceptance Criteria
- `False & X` does not evaluate X
- `True | X` does not evaluate X
- Performance benefit measurable for expensive conditions

---

## T-QST-4.9: Verify DialogueGraphBuilder

**Priority**: P2  
**Estimate**: 1 hour

### Description
Verify fluent builder creates valid graphs.

### Tasks
- [ ] Review method chaining
- [ ] Test simple linear dialogue
- [ ] Test branching dialogue
- [ ] Test `build()` validation
- [ ] Test error cases

### Acceptance Criteria
- Method chaining returns Self
- Built graph matches builder calls
- Validation errors thrown on invalid graphs
- Builder is reusable after build()

---

## T-QST-4.10: Verify Graph Validation

**Priority**: P1  
**Estimate**: 1 hour

### Description
Verify `validate()` catches all defined validation rules.

### Tasks
- [ ] Test missing entry node detection
- [ ] Test missing exit node detection
- [ ] Test orphan node detection
- [ ] Test dead end detection
- [ ] Test invalid references (items, quests, variables)

### Acceptance Criteria
- Errors returned for structural issues
- Warnings returned for semantic issues
- All invalid references detected
- Valid graphs return empty error list

---

## Summary

| Task | Priority | Estimate | Status |
|------|----------|----------|--------|
| T-QST-4.1 | P0 | 1.5h | Pending |
| T-QST-4.2 | P0 | 1h | Pending |
| T-QST-4.3 | P0 | 1h | Pending |
| T-QST-4.4 | P1 | 1h | Pending |
| T-QST-4.5 | P1 | 1h | Pending |
| T-QST-4.6 | P0 | 1h | Pending |
| T-QST-4.7 | P0 | 1.5h | Pending |
| T-QST-4.8 | P1 | 0.5h | Pending |
| T-QST-4.9 | P2 | 1h | Pending |
| T-QST-4.10 | P1 | 1h | Pending |

**Total Estimate**: 10.5 hours
