# System Interconnectivity

How Trinity, Foundation, and FlowForge connect to each other.

---

## The Three Systems

```
┌─────────────────────────────────────────────────────────────────┐
│                         FLOWFORGE                               │
│                    Visual node editor                           │
│              "See and edit code as graphs"                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FOUNDATION                               │
│                    Runtime services                             │
│         "Track, record, introspect everything"                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          TRINITY                                │
│                  Metaprogramming layer                          │
│          "Define components, systems, resources"                │
└─────────────────────────────────────────────────────────────────┘
```

---

## What Each System Does

| System | Role | When It Runs |
|--------|------|--------------|
| **Trinity** | Define the shape of things (classes, fields, decorators) | Definition time |
| **Foundation** | Observe and record what happens (changes, events, state) | Runtime |
| **FlowForge** | Visualize and edit the code | Edit time |

---

## The Folders

```
AI_GAME_ENGINE/
├── trinity/           # Metaprogramming (metaclasses, descriptors, decorators)
├── foundation/        # Runtime services (registry, tracker, eventlog, mirror)
└── flowforge/         # Visual editor (Tauri + ComfyUI + Python sidecar)
```

Three folders because three different concerns.

---

## How They Connect

### Trinity → Foundation (automatic)

Trinity components automatically notify Foundation when things happen:

```python
# When you define a component...
@component
class Player:
    health: int = 100

# ComponentMeta registers it with Foundation
registry.register(Player)

# When you change a field...
player.health = 50

# TrackedDescriptor notifies Foundation
tracker.mark_dirty(player, "health", 100, 50)
eventlog.record_change(player, "health", 100, 50)
```

**Connection point:** `trinity/` imports from `foundation/` (with try/except so Trinity works standalone)

### Foundation → Trinity (observation)

Foundation observes Trinity but doesn't control it:

```python
# Foundation's registry knows about Trinity types
registry.all_types()  # Returns [Player, Enemy, Weapon, ...]

# Foundation's tracker knows what changed
tracker.get_dirty_fields(player)  # Returns ["health"]

# Foundation's eventlog knows the history
eventlog.by_entity(player.id)  # Returns all events for this player
```

**Connection point:** `foundation/bridge.py` pulls from Trinity's metaclasses

### FlowForge → Foundation (planned for Part 2)

FlowForge should talk to Foundation, not Trinity directly:

```python
# FlowForge asks Foundation what exists
adapter = FoundationAdapter()
for type_info in adapter.get_all_types():
    render_node(type_info)

# FlowForge subscribes to live changes
adapter.subscribe_changes(update_canvas)

# FlowForge can undo/redo via Foundation
adapter.undo()
```

**Connection point:** `flowforge/foundation_adapter.py` (Part 2)

---

## Why This Layering?

```
         EDIT TIME                    RUNTIME
              │                          │
              ▼                          ▼
┌─────────────────────┐      ┌─────────────────────┐
│     FLOWFORGE       │      │     YOUR GAME       │
│  (visual editing)   │      │   (running code)    │
└──────────┬──────────┘      └──────────┬──────────┘
           │                            │
           │         ┌──────────────────┘
           │         │
           ▼         ▼
┌─────────────────────────────────────────────────┐
│                  FOUNDATION                      │
│   Single source of truth for all runtime state  │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│                    TRINITY                       │
│         Class definitions and metadata           │
└─────────────────────────────────────────────────┘
```

Both FlowForge (editor) and your game (runtime) go through Foundation. This means:

- FlowForge sees the same state as the running game
- Changes in the game appear in FlowForge automatically
- Undo/redo works across both
- One system to debug, not two

---

## Data Flow Examples

### Defining a Component

```
You write code
      │
      ▼
@component class Player    ──► Trinity metaclass processes it
      │
      ▼
ComponentMeta.__new__()    ──► Foundation registry.register(Player)
      │
      ▼
FlowForge sees Player      ◄── Foundation adapter.get_all_types()
```

### Changing a Field

```
player.health = 50
      │
      ▼
TrackedDescriptor.__set__  ──► Foundation tracker.mark_dirty()
      │                    ──► Foundation eventlog.record()
      ▼
FlowForge highlights field ◄── Foundation adapter.subscribe_changes()
```

### Viewing History

```
Click "why is health 50?"
      │
      ▼
FlowForge requests history ──► Foundation eventlog.by_entity(player.id)
      │
      ▼
Shows causal chain         ◄── [spawn → take_damage → health changed]
```

---

## Summary

| From | To | How | Why |
|------|----|-----|-----|
| Trinity | Foundation | Metaclasses/descriptors call Foundation | Record everything automatically |
| Foundation | Trinity | Registry/mirror read Trinity metadata | Know what types exist |
| FlowForge | Foundation | Adapter wraps Foundation APIs | Single source of truth |
| FlowForge | Trinity | ~~Direct access~~ | ❌ Bypasses Foundation |

**The rule:** FlowForge → Foundation → Trinity (never skip Foundation)
