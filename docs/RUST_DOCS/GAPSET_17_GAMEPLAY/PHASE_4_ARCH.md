# Phase 4: AI Tier 2 — GOAP Planner — Architecture

## Overview

Goal-Oriented Action Planning (GOAP) for high-level AI decision-making. Provides A*-based planning over actions with goal arbitration, plan execution, and monitoring.

## Component Breakdown

### World State

```
GOAP WorldState (immutable)
├── Typed facts: boolean, int, float, Vec3
├── set(fact, value)
├── get(fact)
├── matches(conditions) — precondition checking
├── diff(target_state) — heuristic distance
└── Separate from BT WorldState (shared concept, different implementation)
```

### Actions

```
GOAPAction
├── Preconditions (facts that must be true)
├── Effects (facts that change after execution)
├── Cost value (float, for A* heuristic)
├── execute(agent, world_state)
├── Serializable
└── Reusable across multiple goals
```

### Goals

```
Goal
├── Target state (desired WorldState)
├── Priority (for arbitration)
├── Insistence (urgency modifier)
└── Highest-priority achievable goal = active
```

### Planner

```
GOAPPlanner (A* search)
├── Forward search: current state → applicable actions → new state
├── Heuristic: distance from current to goal state
├── Plan caching: 100 entries, 5s TTL
├── Returns action plan or failure
└── Configurable per agent
```

### Plan Execution

```
GOAPAgent
├── Plan executor (sequential action execution)
├── On action failure → replan
├── On world state change → re-evaluate plan
├── On goal change → replan for new goal
└── Goal arbitration (highest-priority achievable)
```

### Target Selection

```
TargetSelector
├── Threat score (damage output, proximity)
├── Priority score (objective relevance)
├── Opportunity score (exposed, low HP)
└── Weighted combination → current target
```

### Advanced Combat Behaviors (GOAP-expressed)

```
Flank  → navigate to flanking position (uses nav system)
Support → assist allies, suppress enemies
Cover  → find/use/traverse cover points (uses SmartObjects)
```

## Data Flow

```
Frame Tick
  └─→ AI System
       ├─→ Perception update (Phase 3)
       ├─→ GOAPAgent
       │    ├─→ Goal arbitration (re-evaluate priorities)
       │    ├─→ Plan validity check (world state changed?)
       │    ├─→ Replan if needed (A* search)
       │    └─→ Plan step execution
       └─→ BT/Utility fallback (if no plan found)
```

## Key Files

| File | Purpose |
|------|---------|
| `ai/goap.py` | WorldState, GOAPAction, Goal, GOAPPlanner, GOAPAgent |

## Dependencies

- Phase 3: BT World State, Perception, CombatAI (basic)
- Phase 5: Navigation system (for movement actions)
- Phase 7: Ability system (for combat actions)
