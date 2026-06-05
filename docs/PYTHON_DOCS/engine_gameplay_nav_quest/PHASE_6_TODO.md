# PHASE 6 TODO: Objectives and Quest Flow

**Scope**: Objective tracking, state machines, composite objectives, flow patterns  
**Files**: `objectives.py`, `quest_flow.py`

---

## T-QST-6.1: Verify Objective State Machine

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify the ObjectiveState enum and state transitions.

### Tasks
- [ ] Review ObjectiveState enum values
- [ ] Verify activate() transition
- [ ] Verify complete() transition
- [ ] Verify fail() transition
- [ ] Test invalid transition rejection
- [ ] Test state change callbacks

### Acceptance Criteria
- All states defined (INACTIVE, IN_PROGRESS, COMPLETE, FAILED)
- Valid transitions succeed
- Invalid transitions raise errors or are ignored
- State change callbacks invoked

---

## T-QST-6.2: Verify KillObjective

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify KillObjective tracks kills and streaks correctly.

### Tasks
- [ ] Review kill count incrementing
- [ ] Review streak tracking logic
- [ ] Test completion at required count
- [ ] Test progress reporting
- [ ] Test target type filtering

### Acceptance Criteria
- Only matching target types count
- Count increments correctly
- Streak resets on non-matching kill (if enabled)
- Completes when count >= required
- Progress returns (current, required)

---

## T-QST-6.3: Verify CollectObjective

**Priority**: P0  
**Estimate**: 0.5 hours

### Description
Verify CollectObjective tracks item collection.

### Tasks
- [ ] Review item count tracking
- [ ] Test auto_remove flag
- [ ] Test completion at required count
- [ ] Test inventory integration

### Acceptance Criteria
- Counts items in inventory
- Auto-remove removes items on completion
- Completes when count >= required
- Handles item loss (count decreases)

---

## T-QST-6.4: Verify Movement Objectives

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify ReachObjective and EscortObjective.

### Tasks
- [ ] Review ReachObjective position checking
- [ ] Review ReachObjective stay_duration tracking
- [ ] Review EscortObjective NPC tracking
- [ ] Review EscortObjective destination checking
- [ ] Test failure conditions

### Acceptance Criteria
- ReachObjective completes when in area for duration
- EscortObjective tracks NPC alive state
- EscortObjective completes when NPC at destination
- EscortObjective fails if NPC dies

---

## T-QST-6.5: Verify Interaction Objectives

**Priority**: P1  
**Estimate**: 0.5 hours

### Description
Verify TalkObjective, InteractObjective, UseObjective.

### Tasks
- [ ] Review TalkObjective NPC reference
- [ ] Review InteractObjective object reference
- [ ] Review UseObjective ability tracking
- [ ] Test completion conditions

### Acceptance Criteria
- TalkObjective completes on conversation with correct NPC
- InteractObjective completes on interaction with correct object
- UseObjective counts ability uses correctly

---

## T-QST-6.6: Verify Crafting and Defense Objectives

**Priority**: P1  
**Estimate**: 0.5 hours

### Description
Verify CraftObjective and DefendObjective.

### Tasks
- [ ] Review CraftObjective recipe tracking
- [ ] Review DefendObjective area monitoring
- [ ] Review DefendObjective duration tracking
- [ ] Test failure conditions

### Acceptance Criteria
- CraftObjective counts crafted items correctly
- DefendObjective tracks enemies in area
- DefendObjective fails if enemies breach
- DefendObjective completes after duration

---

## T-QST-6.7: Verify TimedObjective Wrapper

**Priority**: P1  
**Estimate**: 0.5 hours

### Description
Verify TimedObjective wraps objectives with time limits.

### Tasks
- [ ] Review time tracking
- [ ] Review wrapped objective forwarding
- [ ] Test completion before time limit
- [ ] Test failure on timeout

### Acceptance Criteria
- Time counts down correctly
- Wrapped objective updates forwarded
- Completes if wrapped completes in time
- Fails if time limit exceeded

---

## T-QST-6.8: Verify CompositeObjective

**Priority**: P0  
**Estimate**: 1.5 hours

### Description
Verify CompositeObjective with all modes.

### Tasks
- [ ] Review mode="all" completion logic
- [ ] Review mode="any" completion logic
- [ ] Review mode="sequential" progression
- [ ] Test nested composites
- [ ] Test failure propagation

### Acceptance Criteria
- "all" completes when all children complete
- "any" completes when any child completes
- "sequential" advances through children in order
- Failure propagates correctly per mode
- Nested composites work

---

## T-QST-6.9: Verify SequentialFlow

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify SequentialFlow traverses children in order.

### Tasks
- [ ] Review current_index tracking
- [ ] Review advance() logic
- [ ] Test get_current_objectives()
- [ ] Test completion detection
- [ ] Test failure handling

### Acceptance Criteria
- Only current child is active
- Advance moves to next child
- Current objectives from current child only
- Completes when all children complete
- Fails if any child fails

---

## T-QST-6.10: Verify ParallelFlow

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify ParallelFlow handles concurrent objectives.

### Tasks
- [ ] Review require_all flag handling
- [ ] Review get_current_objectives() aggregation
- [ ] Test completion with require_all=true
- [ ] Test completion with require_all=false
- [ ] Test failure conditions

### Acceptance Criteria
- All children active simultaneously
- Current objectives from all active children
- require_all=true: complete when all done
- require_all=false: complete when any done
- Failure conditions per mode

---

## T-QST-6.11: Verify BranchingFlow

**Priority**: P1  
**Estimate**: 1 hour

### Description
Verify BranchingFlow handles player choices.

### Tasks
- [ ] Review select_branch() method
- [ ] Review auto_advance flag
- [ ] Test branch selection
- [ ] Test current objectives after selection
- [ ] Test invalid branch selection

### Acceptance Criteria
- Branch selection activates chosen branch
- Only selected branch executes
- Unselected branches remain inactive
- Auto-advance triggers on completion
- Invalid branch selection rejected

---

## T-QST-6.12: Verify OptionalFlow

**Priority**: P2  
**Estimate**: 0.5 hours

### Description
Verify OptionalFlow handles bonus objectives.

### Tasks
- [ ] Review bonus objective tracking
- [ ] Review bonus reward granting
- [ ] Test completion without bonus
- [ ] Test completion with bonus

### Acceptance Criteria
- Main objectives required
- Bonus objectives tracked separately
- Rewards granted for completed bonuses
- No penalty for skipping bonus

---

## T-QST-6.13: Verify MixedFlow

**Priority**: P2  
**Estimate**: 1 hour

### Description
Verify MixedFlow allows arbitrary nesting.

### Tasks
- [ ] Test nested Sequential in Parallel
- [ ] Test nested Parallel in Sequential
- [ ] Test nested Branching in Sequential
- [ ] Test deeply nested structures

### Acceptance Criteria
- All patterns can nest in MixedFlow
- Traversal handles mixed types
- Completion propagates correctly
- No restrictions on nesting depth

---

## T-QST-6.14: Verify FlowBuilder

**Priority**: P2  
**Estimate**: 0.5 hours

### Description
Verify FlowBuilder creates valid flow structures.

### Tasks
- [ ] Review method chaining
- [ ] Test sequential construction
- [ ] Test nested construction
- [ ] Test build() validation

### Acceptance Criteria
- Method chaining returns Self
- Built flow matches builder calls
- Validation errors on invalid structure
- Builder reusable after build()

---

## Summary

| Task | Priority | Estimate | Status |
|------|----------|----------|--------|
| T-QST-6.1 | P0 | 1h | Pending |
| T-QST-6.2 | P0 | 1h | Pending |
| T-QST-6.3 | P0 | 0.5h | Pending |
| T-QST-6.4 | P0 | 1h | Pending |
| T-QST-6.5 | P1 | 0.5h | Pending |
| T-QST-6.6 | P1 | 0.5h | Pending |
| T-QST-6.7 | P1 | 0.5h | Pending |
| T-QST-6.8 | P0 | 1.5h | Pending |
| T-QST-6.9 | P0 | 1h | Pending |
| T-QST-6.10 | P0 | 1h | Pending |
| T-QST-6.11 | P1 | 1h | Pending |
| T-QST-6.12 | P2 | 0.5h | Pending |
| T-QST-6.13 | P2 | 1h | Pending |
| T-QST-6.14 | P2 | 0.5h | Pending |

**Total Estimate**: 11.5 hours
