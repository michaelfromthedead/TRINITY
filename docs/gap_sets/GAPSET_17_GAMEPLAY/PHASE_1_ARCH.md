# Phase 1: Entity Framework & Core — Architecture

## Overview

UE5-inspired Actor/Component entity model with lifecycle management, controller hierarchy, prefab system, FSM/HFSM state machines, and movement components.

## Component Breakdown

### Entity Layer (`entity/`)

```
Actor (base)
├── DynamicActor (+ physics body)
├── Pawn (+ possessable, Controller ref)
└── Character (+ MovementComponent dependency)

Controller (base)
├── PlayerController (+ input bindings)
└── AIController (+ AI decision system)

LifecycleManager (singleton)
├── Deferred state transitions (batched to end of frame)
├── Spawn / BeginPlay / Tick / EndPlay / Destroy
└── @on_spawn / @on_add / @on_remove / @on_despawn hooks

PrefabRegistry (singleton)
├── PrefabInstantiator
├── @prefab / @extends decorators
├── PrefabBuilder (fluent API)
└── spawn_prefab() / register_prefab()
```

### State Machine Layer (`fsm/`)

```
FSM
├── StateMeta (metaclass)
├── @on_enter / @on_exit hooks
├── Transition validation
└── Execution: StateMachineSystem (UPDATE phase)

HFSM
├── register_substate
├── Parent/child nesting
└── Auto-enter parent / auto-exit children

Pushdown Automaton
├── State stack (push/pop)
├── Temporary states (e.g., stunned)
└── Pop returns to previous state
```

### Movement Layer

```
MovementComponent
├── velocity: Vec3, speed: float
├── movement_mode: enum (walking, running, sprinting, crouching, prone, swimming, flying, custom)
└── Character dependency
```

## Data Flow

```
Frame Tick
  └─→ LifecycleManager (deferred ops)
       └─→ StateMachineSystem (UPDATE phase)
            ├─→ FSM.tick() → transition evaluation → hook firing
            ├─→ HFSM.tick() → hierarchy-aware state checks
            └─→ PDA.tick() → stack management
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `entity/actor.py` | — | Actor, DynamicActor, Pawn, Character classes |
| `entity/controllers.py` | — | Controller, PlayerController, AIController |
| `entity/lifecycle.py` | — | LifecycleManager, deferred transitions |
| `entity/prefab.py` | — | Prefab system, @prefab decorator |
| `entity/movement.py` | — | MovementComponent |
| `fsm/fsm.py` | — | FSM, HFSM, pushdown automaton, StateMachineSystem |

## Dependencies

- Foundation: EngineMeta, ComponentMeta, @component, EventLog, TrackedDescriptor
- Phase 1 provides base for all subsequent phases
