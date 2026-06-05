# Phase 9: Quest & Dialogue Systems — Architecture

## Overview

Complete quest system (definitions, objectives, journal, tracker) and graph-based dialogue system with conditions, effects, and variable scoping.

## Component Breakdown

### Quest Core (`quest/quest.py`)

```
QuestDefinition (immutable template)
├── id, name, description, quest_type (12 types: MAIN through EXPLORATION)
├── level_requirement, level_cap, time_limit
├── repeatable, cooldown, auto_accept, auto_complete
├── hidden, shareable, abandon_penalty
├── Prerequisites (quest IDs), required_items, required_reputation
├── Rewards list
└── Metadata: category, zone, giver_id, turn_in_id, tags

Quest (active player instance)
├── 6 states: UNAVAILABLE → AVAILABLE → ACTIVE → COMPLETE → TURNED_IN + FAILED
├── State transition methods: make_available(), accept(), complete(), turn_in(), fail()
├── reset() for repeatable quests, abandon()
├── Timestamps: accepted_at, completed_at, turned_in_at, failed_at
├── times_completed, last_completed_at
├── objective_progress (player-specific)
└── Properties: is_active, is_complete, is_finished, is_available, can_repeat

QuestRegistry (singleton)
├── Register/unregister quest definitions
├── Lookup by ID, type, zone, giver, tag
└── Full quest enumeration

@quest(id, name, type, level_requirement, prerequisites, rewards) decorator
├── Attaches metadata to class
└── Auto-registers with QuestRegistry
```

### Quest Objectives (`quest/objectives.py`)

```
6 Objective Types
├── KillObjective — counter on entity kills (target faction/type, count)
├── CollectObjective — counter on item pickup (item_id, count)
├── TalkObjective — flag on dialogue completion (npc_id)
├── ReachObjective — flag on position trigger (location, radius)
├── EscortObjective — status of escorted entity (from → to, health threshold)
└── InteractObjective — flag on interaction trigger (target_id)

4 Flow Patterns
├── Sequential — objectives complete in order
├── Parallel — all must complete, any order
├── Branching — one of N paths completes
└── Optional — not required for completion

Progress tracking per objective type
```

### Quest Rewards (`quest/quest_rewards.py`)

```
Reward System
├── XPReward — experience points
├── ItemReward — item + quantity (goes to inventory)
├── CurrencyReward — currency type + amount
├── QuestUnlockReward — unlocks new quests
├── AbilityUnlockReward — unlocks abilities
├── ReputationReward — faction reputation
├── TitleReward — player title
└── Granted on COMPLETE → TURNED_IN transition
```

### Quest Tracking

```
QuestTracker (player entity component)
├── Active quest list
├── Per-quest progress (objectives, counters, flags)
├── Completed/failed history
└── EventLog integration

QuestJournal
├── Text log of quest events
├── Quest state history
└── Display-ready formatting

QuestFlow
├── Narrative flow control
├── Quest chains and sequences
└── Conditional quest availability
```

### Dialogue System (`quest/dialogue.py`)

```
Graph-Based Dialogue
├── Root node → traversal
├── Node types:
│   ├── TextNode — NPC line + optional player response
│   ├── ChoiceNode — player options with conditions
│   ├── BranchNode — condition-based flow control
│   ├── EventNode — triggers game action
│   └── RandomNode — randomized variation

DialogueConditions (quest/dialogue_conditions.py)
├── Condition checking system (1078 lines)
├── Quest state, player level, inventory, reputation checks
└── Composite conditions (AND/OR/NOT)

DialogueEffects (quest/dialogue_effects.py)
├── Game effect execution from dialogue (1486 lines)
├── Quest accept/turn-in/update
├── Item give/remove, currency, XP
└── World state changes, ability unlocks

DialogueVariables (quest/dialogue_variables.py)
├── 4 scopes (942 lines):
│   ├── Local — scoped to conversation
│   ├── Global — persistent across saves
│   ├── Quest-linked — read/write quest state
│   └── World state — boolean facts
├── Branch conditions
└── Text substitution

Presentation
├── Text box with typewriter effect
├── Portrait display (configurable per NPC/line)
├── Choice buttons for player responses
├── Skip/delay controls
└── Voice sync integration point
```

### Quest/Dialogue Integration

```
Dialogue → Quest
├── Accept quest
├── Turn in quest
├── Update quest state
└── Quest dialogue conditions filter options

Quest → Dialogue
├── Check player's quest state
├── Show/hide dialogue options based on quest progress
└── Objective updates trigger dialogue changes
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `quest/quest.py` | 432 | QuestDefinition, Quest, QuestRegistry, @quest |
| `quest/objectives.py` | 936 | 6 objective types, 4 flow patterns |
| `quest/quest_rewards.py` | — | Reward types (XP, item, currency, etc.) |
| `quest/journal.py` | 721 | Quest journal |
| `quest/tracker.py` | 639 | QuestTracker component |
| `quest/quest_flow.py` | 867 | Quest flow control |
| `quest/dialogue.py` | 1453 | Dialogue graph, node types, presentation |
| `quest/dialogue_conditions.py` | 1078 | Dialogue condition checking |
| `quest/dialogue_effects.py` | 1486 | Dialogue game effects |
| `quest/dialogue_variables.py` | 942 | Variable scoping and resolution |

## Dependencies

- Phase 1 entity framework (Actor, ComponentStore)
- Phase 8 (Inventory for item rewards)
- Foundation: EventLog
