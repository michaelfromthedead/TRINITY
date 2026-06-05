# Investigation: engine/gameplay/quest

## Summary
The quest system is a **production-grade, fully-featured implementation** providing comprehensive quest management, objective tracking, branching dialogue graphs, quest journals, and world markers. This is real, working code with proper state machines, event-driven progress tracking, serialization, and localization support.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 0 | EMPTY | Just a package marker |
| `constants.py` | 278 | REAL | Comprehensive constants for dialogue, quests, objectives |
| `dialogue.py` | ~1070 | REAL | Full dialogue graph system with 6 node types |
| `dialogue_conditions.py` | ~860 | REAL | Condition evaluation system |
| `dialogue_effects.py` | ~1150 | REAL | Effect execution system |
| `dialogue_variables.py` | ~670 | REAL | Variable scoping and storage |
| `journal.py` | ~585 | REAL | Quest journal, HUD tracker, world markers |
| `objectives.py` | 937 | REAL | 11 objective types with full tracking |
| `quest.py` | 432 | REAL | Quest definition, state machine, registry |
| `quest_flow.py` | ~685 | REAL | Quest flow controller |
| `tracker.py` | 640 | REAL | Player quest progress tracking |

**Total: ~7,307 lines of substantial implementation code**

## Quest Components

### Quest System (`quest.py`)
- `QuestState` enum: UNAVAILABLE, AVAILABLE, ACTIVE, COMPLETE, TURNED_IN, FAILED
- `QuestType` enum: 13 types (MAIN, SIDE, DAILY, WEEKLY, WORLD, DUNGEON, RAID, PVP, HIDDEN, TUTORIAL, EVENT, BOUNTY, EXPLORATION)
- `QuestDefinition` dataclass with prerequisites, rewards, level requirements, time limits, repeatable flags
- `Quest` class with full state machine (accept, complete, turn_in, fail, reset, abandon)
- `QuestRegistry` singleton for quest lookup by type, zone, giver, tag

### Objective System (`objectives.py`)
- **11 Objective Types**: Kill, Collect, Talk, Reach, Escort, Interact, Use, Craft, Defend, Timed, Composite
- `Objective` base class with states (INACTIVE, IN_PROGRESS, COMPLETE, FAILED)
- Event-driven `update()` method for game event processing
- Progress tracking with callbacks (on_complete, on_fail, on_progress)
- `CompositeObjective` supports all/any/sequential completion modes
- `ObjectiveFactory` for creating objectives from config data

### Tracker (`tracker.py`)
- Per-player quest progress tracking
- Prerequisite checking (quests, level, items, reputation)
- Event dispatching with listener pattern
- Automatic objective activation and completion
- Time-based quest updates (time limits)
- Full serialization/deserialization for save games

### Dialogue System (`dialogue.py`)
- **6 Node Types**: TextNode, ChoiceNode, BranchNode, EventNode, RandomNode, Entry/Exit
- Conditional branching with priority evaluation
- Player choices with conditions and effects
- Weighted random node selection
- Localization support with fallback
- Full graph serialization to/from dict

### Journal (`journal.py`)
- `QuestJournal` for quest log UI
- `HUDTracker` for on-screen quest tracking
- `WorldMarker` for in-world objective markers
- Journal filtering and sorting
- Category organization (Main, Side, Daily, Completed, Failed)

## Implementation

- Real quest tracking? **YES** - Full state machine with transitions, events, serialization
- Real objectives? **YES** - 11 types with counters, flags, progress callbacks, composite objectives
- Real branching? **YES** - BranchNode with priority conditions, ChoiceNode with conditional choices, RandomNode with weighted selection

## Verdict
**REAL IMPLEMENTATION**

This is a complete, production-quality quest system suitable for a commercial RPG. Key evidence:
1. Proper state machines with all transitions
2. Event-driven architecture for game integration
3. Full serialization for save/load
4. Localization support
5. Extensible factory patterns
6. Comprehensive validation

## Evidence

### Quest State Machine (quest.py:174-235)
```python
def accept(self, timestamp: float) -> bool:
    """Accept the quest."""
    if self.state != QuestState.AVAILABLE:
        return False
    self.state = QuestState.ACTIVE
    self.accepted_at = timestamp
    return True

def complete(self, timestamp: float) -> bool:
    """Mark quest as complete (objectives done)."""
    if self.state != QuestState.ACTIVE:
        return False
    self.state = QuestState.COMPLETE
    self.completed_at = timestamp
    return True
```

### Kill Objective Event Handling (objectives.py:203-235)
```python
def update(self, event_type: str, event_data: dict[str, Any]) -> bool:
    if event_type != "kill":
        return False
    if self.state != ObjectiveState.IN_PROGRESS:
        return False

    if event_data.get("target_type") != self.target_type:
        self.kill_streak = 0
        return False

    if self.location and event_data.get("location") != self.location:
        self.kill_streak = 0
        return False

    self.current += event_data.get("count", 1)
    self.kill_streak += 1
    self._notify_progress()

    if self.current >= self.required:
        self.complete()
    return True
```

### Conditional Branching (dialogue.py:552-584)
```python
def evaluate(self, context: Any) -> Tuple[Optional[str], ConditionResult]:
    """Evaluate branches and return the selected path."""
    sorted_branches = sorted(
        self.branches,
        key=lambda b: b.priority,
        reverse=True
    )

    for branch in sorted_branches:
        result = branch.condition.evaluate(context)
        if result.success:
            return branch.target_node, result

    if self.default_node:
        return self.default_node, ConditionResult(
            success=True,
            message="Using default branch"
        )
    return None, ConditionResult(success=False, message="No branch matched")
```

### Tracker Serialization (tracker.py:593-615)
```python
def serialize(self) -> dict[str, Any]:
    """Serialize tracker state for saving."""
    return {
        "player_id": self.player_id,
        "current_time": self._current_time,
        "completed_quests": list(self._completed_quests),
        "failed_quests": list(self._failed_quests),
        "player_level": self._player_level,
        "player_items": self._player_items.copy(),
        "player_reputation": self._player_reputation.copy(),
        "tracked_quests": {
            qid: {
                "state": tq.quest.state.name,
                "accepted_at": tq.quest.accepted_at,
                "completed_at": tq.quest.completed_at,
                "times_completed": tq.quest.times_completed,
                "objective_progress": tq.quest.objective_progress,
            }
            for qid, tq in self._tracked.items()
        },
    }
```
