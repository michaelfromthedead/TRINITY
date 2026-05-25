# Phase 3: AI Tier 1 — Behavior Trees + Utility AI + Perception — Architecture

## Overview

Foundation AI layer providing reactive behavior (BT), scored decision-making (Utility AI), environmental awareness (Perception), and knowledge representation (Blackboard, World State).

## Component Breakdown

### Behavior Trees (`ai/behavior_tree.py`)

```
Composite Nodes
├── Sequence (all children succeed)
├── Selector (first child succeeds)
└── Parallel (3 policies: ALL, ONE, ANY)

Decorator Nodes
├── Invert (flip child result)
├── Repeat (run N times)
├── Timeout (fail after duration)
├── Cooldown (rate-limit)
├── ForceSuccess / ForceFailure
└── ConditionCheck

Leaf Nodes
├── Action (callable)
├── Condition (predicate)
├── Wait (delay)
├── Log (debug)
└── Custom (extensible)

Runtime
├── Tick-based traversal
├── Blackboard read/write
├── Abort on condition change
└── Depth limit: 100

@behavior_tree(id, debug_name) decorator
Node type registry (Foundation Registry)
```

### Blackboard (`ai/blackboard.py`)

```
Hierarchical key-value store
├── Observers (value change notification)
├── TTL expiry
├── Glob pattern matching
├── Parent-child scoping
├── TypedBlackboard (bool, int, float, Vec3, EntityRef, string)
├── Serialization support
└── @blackboard decorator
```

### Utility AI (`ai/utility.py`)

```
Scoring System
├── Options (1+ considerations)
├── Considerations (input → curve → score [0,1])
├── 9 response curves: Linear, Quadratic, Exponential, Logistic, Sine, Inverse, Step, Smoothstep, Custom
├── Compensation factor scoring
└── Selector: deterministic (highest) or weighted random

@utility_ai(id, update_rate) decorator
```

### Perception (`ai/perception.py`)

```
Sight
├── FOV cone raycast
├── Configurable range/angle
├── Occlusion via physics raycast
└── Stimuli: entity, last known position, timestamp

Hearing
├── Sound propagation (distance falloff)
├── Occlusion reduces loudness
└── Threshold-based triggering

Damage/Squad
├── Combat events → auto-stimuli
├── Faction/allegiance detection
└── Passive awareness

Perception Memory
├── Stimuli storage with timestamps
├── Aged stimuli decay → removal
├── Last known positions (unseen targets)
└── 3x persistence multiplier

@perception(sense, range, fov) decorator
```

### Knowledge / World State

```
World State — Boolean facts
├── has("enemy_sighted"), set/clear
└── Integration with Blackboard

Influence Maps — NOT IMPLEMENTED (constants only)
├── Grid overlay with layers (threat, resource, interest)
├── Propagation with decay
└── sample(layer, position)
```

### Combat AI (Basic)

```
CombatAI class
├── Attack (approach, use weapon/ability)
├── Defend (block, dodge, cover)
├── Retreat (flee to safety)
└── BT subtree implementations
```

### Social AI

```
FactionComponent
├── team_id, faction string
├── attitude: hostile/neutral/friendly
├── IFF: is_enemy, is_ally, is_neutral
└── Runtime team change
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `ai/behavior_tree.py` | — | BT composite, decorator, leaf, runtime |
| `ai/blackboard.py` | — | Hierarchical blackboard with observers |
| `ai/utility.py` | — | Utility AI scoring, curves, selector |
| `ai/perception.py` | — | Sight, hearing, damage perception |
| `ai/constants.py` | — | Configuration constants (incl. influence map prefs) |
| `ai/__init__.py` | 1185 | Combined module (bus factor — needs decomposition) |

## Dependencies

- Phase 1 entity framework (Actor, lifecycle, ComponentStore)
- Physics system (raycast for sight)
- Audio system (hearing perception)
