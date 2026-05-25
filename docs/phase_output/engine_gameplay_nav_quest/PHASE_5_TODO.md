# PHASE 5 TODO: Dialogue Effects and Variables

**Scope**: Transactional effects, rollback pattern, variable scoping  
**Files**: `dialogue_effects.py`, `dialogue_variables.py`

---

## T-QST-5.1: Verify Effect Base Pattern

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify the Effect base class and execute/rollback pattern.

### Tasks
- [ ] Review Effect base class interface
- [ ] Verify execute() returns bool for success/failure
- [ ] Verify rollback() restores previous state
- [ ] Test effect description generation

### Acceptance Criteria
- All effects implement execute/rollback
- Return values indicate success/failure
- Rollback is idempotent
- Descriptions are human-readable

---

## T-QST-5.2: Verify Variable Effects

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify SetVariable, IncrementVariable, DecrementVariable effects.

### Tasks
- [ ] Review SetVariable state capture
- [ ] Review IncrementVariable amount handling
- [ ] Review DecrementVariable amount handling
- [ ] Test rollback restores original values

### Acceptance Criteria
- SetVariable captures previous value on execute
- Increment/Decrement correctly modify values
- Rollback restores exact previous state
- Works with various value types

---

## T-QST-5.3: Verify Item Effects

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify GiveItem and TakeItem effects.

### Tasks
- [ ] Review GiveItem inventory integration
- [ ] Review TakeItem inventory integration
- [ ] Test quantity handling
- [ ] Test rollback inventory state

### Acceptance Criteria
- GiveItem adds items to inventory
- TakeItem removes items from inventory
- Quantity respected
- Rollback restores inventory state
- TakeItem fails if insufficient quantity

---

## T-QST-5.4: Verify Quest Effects

**Priority**: P0  
**Estimate**: 1.5 hours

### Description
Verify all quest-related effects.

### Tasks
- [ ] Review SetQuestState integration with quest system
- [ ] Review SetQuestProgress integration
- [ ] Review StartQuest activation
- [ ] Review CompleteQuest/FailQuest terminal states
- [ ] Test rollback for each effect type

### Acceptance Criteria
- Quest state changes reflect in quest system
- Progress updates correctly
- StartQuest activates inactive quest
- Complete/Fail mark terminal states
- Rollback restores previous quest state

---

## T-QST-5.5: Verify Reputation Effects

**Priority**: P1  
**Estimate**: 0.5 hours

### Description
Verify ChangeReputation and SetReputation effects.

### Tasks
- [ ] Review ChangeReputation delta handling
- [ ] Review SetReputation absolute handling
- [ ] Verify faction reference validation
- [ ] Test rollback for both types

### Acceptance Criteria
- ChangeReputation adds/subtracts correctly
- SetReputation sets absolute value
- Invalid factions rejected
- Rollback restores previous reputation

---

## T-QST-5.6: Verify Event Effects

**Priority**: P1  
**Estimate**: 1 hour

### Description
Verify event effects (TriggerEvent, PlaySound, PlayAnimation, StartDialogue).

### Tasks
- [ ] Review TriggerEvent dispatch
- [ ] Review PlaySound audio integration
- [ ] Review PlayAnimation integration
- [ ] Review StartDialogue recursion handling
- [ ] Verify non-reversibility documented

### Acceptance Criteria
- Events fire correctly
- Sound/animation play correctly
- StartDialogue opens new dialogue
- Rollback is no-op (documented)
- No crashes on rollback

---

## T-QST-5.7: Verify EffectBatch Transactions

**Priority**: P0  
**Estimate**: 1.5 hours

### Description
Verify EffectBatch provides transactional execution.

### Tasks
- [ ] Review batch execute loop
- [ ] Test all-succeed case
- [ ] Test middle-failure case (rollback partial)
- [ ] Test first-failure case
- [ ] Verify rollback order (reversed)

### Acceptance Criteria
- All effects execute if all succeed
- Partial execution rolls back on failure
- Rollback order is reversed (LIFO)
- Final state is consistent (all or nothing)

---

## T-QST-5.8: Verify LocalVariableStore

**Priority**: P0  
**Estimate**: 0.5 hours

### Description
Verify LocalVariableStore is ephemeral and per-conversation.

### Tasks
- [ ] Review get/set/has/delete operations
- [ ] Verify isolation between conversations
- [ ] Test cleanup on conversation end
- [ ] Verify no persistence

### Acceptance Criteria
- Variables set are retrievable
- Different conversations have different stores
- Store clears when conversation ends
- No persistence to save file

---

## T-QST-5.9: Verify GlobalVariableStore

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify GlobalVariableStore persistence and special operations.

### Tasks
- [ ] Review get/set/has/delete operations
- [ ] Verify increment/toggle/list operations
- [ ] Test change history tracking
- [ ] Test observer notification
- [ ] Verify persistence/serialization

### Acceptance Criteria
- Basic operations work
- Increment adds to numeric values
- Toggle flips boolean values
- List operations append/remove correctly
- History records all changes
- Observers notified on change
- Persists through save/load

---

## T-QST-5.10: Verify QuestVariableStore

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify QuestVariableStore syncs with quest system.

### Tasks
- [ ] Review bidirectional sync
- [ ] Test sync_to_quest_state()
- [ ] Test sync_from_quest_state()
- [ ] Verify quest lifecycle integration
- [ ] Test concurrent access

### Acceptance Criteria
- Changes sync to quest system
- Quest system changes sync back
- Sync triggers on appropriate events
- Quest completion clears store appropriately

---

## T-QST-5.11: Verify Variable Resolution Chain

**Priority**: P1  
**Estimate**: 0.5 hours

### Description
Verify unscoped variable resolution follows Local -> Quest -> Global.

### Tasks
- [ ] Review VariableManager.resolve()
- [ ] Test local shadows quest
- [ ] Test quest shadows global
- [ ] Test fallback to global

### Acceptance Criteria
- Local takes precedence
- Quest takes precedence over global
- Global is fallback
- Scoped access bypasses resolution

---

## T-QST-5.12: Verify Observer Pattern

**Priority**: P2  
**Estimate**: 0.5 hours

### Description
Verify observer pattern for variable changes.

### Tasks
- [ ] Review add_observer/remove_observer
- [ ] Test observer called on change
- [ ] Test multiple observers per variable
- [ ] Test observer removal

### Acceptance Criteria
- Observer callback invoked on change
- Old and new values passed to callback
- Multiple observers all called
- Removed observers not called

---

## Summary

| Task | Priority | Estimate | Status |
|------|----------|----------|--------|
| T-QST-5.1 | P0 | 1h | Pending |
| T-QST-5.2 | P0 | 1h | Pending |
| T-QST-5.3 | P0 | 1h | Pending |
| T-QST-5.4 | P0 | 1.5h | Pending |
| T-QST-5.5 | P1 | 0.5h | Pending |
| T-QST-5.6 | P1 | 1h | Pending |
| T-QST-5.7 | P0 | 1.5h | Pending |
| T-QST-5.8 | P0 | 0.5h | Pending |
| T-QST-5.9 | P0 | 1h | Pending |
| T-QST-5.10 | P0 | 1h | Pending |
| T-QST-5.11 | P1 | 0.5h | Pending |
| T-QST-5.12 | P2 | 0.5h | Pending |

**Total Estimate**: 11 hours
