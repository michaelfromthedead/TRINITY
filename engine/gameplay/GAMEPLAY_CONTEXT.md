# GAMEPLAY_CONTEXT.md — Gameplay Systems Layer

> **Layer**: engine/gameplay/ — High-level game logic, AI, input, abilities, quests, economy
> **Purpose**: Implements all gameplay mechanics on top of ECS, Trinity, and Foundation
> **Dependencies**: engine/core/ (ECS, scheduling), engine/common/ (types), engine/simulation/ (physics), engine/engine/ (main loop)

---

## 1. Architecture Summary

The Gameplay Layer provides high-level systems implementing game logic, player interaction, AI behavior, and game mechanics. It sits above the simulation/rendering layers and below tooling.

```
┌─────────────────────────────────────────────────────────────┐
│              GAMEPLAY LAYER (this layer)                     │
│  Entity, AI, Navigation, Input, Camera, Abilities, Quests   │
└─────────────────────────────────────────────────────────────┘
           │
  ┌────────┼────────┐
  ▼        ▼        ▼
SIMULATION RENDERING AUDIO
```

### 14 Subsystems

| # | Subsystem | Scope |
|---|-----------|-------|
| 1 | **Entity & Object Model** | Actor/Pawn, prefabs, lifecycle, possession |
| 2 | **Scripting & Logic** | Events, visual scripting, script bindings |
| 3 | **AI Systems** | Perception, knowledge, decision-making, combat AI, social AI |
| 4 | **Navigation** | NavMesh, pathfinding (A*, HPA*), steering, avoidance, nav links |
| 5 | **Input System** | Devices, raw processing, action mapping, contexts, rebinding |
| 6 | **Camera System** | First/third person, orbit, follow, collision, effects, rails |
| 7 | **Ability System** | Abilities, effects, attributes, targeting, gameplay tags |
| 8 | **Inventory System** | Items, containers, equipment, loot, crafting |
| 9 | **Dialogue System** | Nodes, conditions, variables, presentation, voice |
| 10 | **Quest System** | Objectives, flow, states, tracking, rewards |
| 11 | **State Machines** | FSM, HFSM, pushdown automaton |
| 12 | **Combat System** | Damage, health, death, teams, scoring |
| 13 | **Game Modes** | Rules, spawning, scoring, time limits |
| 14 | **Economy** | Currencies, transactions, trading, crafting |

### Entity Lifecycle

```
CREATE → INITIALIZE → ACTIVE → DEACTIVATE → DESTROY
```

Deferred operations: spawn, destroy, and structural modifications are batched until end of frame via `@deferred`.

### Gameplay System Execution Order (within simulation tick)

```
1.  InputSystem         — Gather and process player input
2.  AISystem            — AI decisions (BT, utility, GOAP)
3.  AbilitySystem       — Activate/tick abilities and effects
4.  EffectSystem        — Apply gameplay effects (buffs, debuffs)
5.  MovementSystem      — Character movement, steering
6.  StateMachineSystem  — FSM transitions and updates
7.  DamageSystem        — Damage calculation, health reduction
8.  DeathSystem         — Death detection, cleanup triggers
9.  CleanupSystem       — Deferred destroy, component removal
10. TriggerSystem       — Area triggers, interaction checks
```

### Component Types

| Type | Description | Example |
|------|-------------|---------|
| Data Components | Pure state storage | Transform, Health, Velocity |
| Tag Components | Zero-size markers | IsPlayer, IsEnemy, NeedsUpdate |
| Shared Components | Grouped entities | Faction, Team, RenderLayer |
| Singletons | Global state | GameState, WorldTime |

### Actor/Pawn Framework

| Type | Description |
|------|-------------|
| Static Actor | Non-moving world objects |
| Dynamic Actor | Movable, physics-enabled |
| Pawn | Possessable entity |
| Character | Humanoid pawn with movement |

Possession system: Controller → Player Controller / AI Controller → Runtime possession switch.

---

## 2. All Decorators

### Core ECS Decorators (trinity/decorators/ecs_core.py)

1. **@component(name=None)** — Mark class as ECS component.
   - Steps: `TAG("component", True)`, `TAG("component_name", name)`, `REGISTER("ecs")`
   - Excludes: `resource`, `event`

2. **@tag()** — Zero-sized component for filtering.
   - Steps: `TAG("tag_component", True)`, `REGISTER("ecs")`
   - Excludes: `resource`, `event`, `serializable`

3. **@system(phase="update")** — Mark function/class as ECS system.
   - Steps: `TAG("system", True)`, `TAG("system_phase", phase)`, `REGISTER("ecs")`

4. **@resource(name=None)** — Singleton resource.
   - Steps: `TAG("resource", True)`, `REGISTER("ecs")`
   - Excludes: `component`, `tag`, `event`

5. **@event()** — Event type for event bus.
   - Steps: `TAG("event", True)`, `REGISTER("events")`
   - Excludes: `component`, `tag`, `resource`

6. **@query(components=(), with_=(), without=(), maybe=())** — Declarative query.

7. **@bundle()** — Components that spawn together.

8. **@relation(kind="one_to_many", exclusive=False)** — Entity relationships.
   - Valid kinds: "one_to_one", "one_to_many", "many_to_many"

9. **@derived(from_components: tuple, cache=True)** — Computed component.

### Gameplay Decorators (trinity/decorators/gameplay.py)

10. **@ability(cost={}, cooldown=0.0, tags=set(), blocked_by=set())** — Gameplay ability.
    - Steps: `TAG("ability", True)`, `TAG("ability_cost", cost)`, `TAG("ability_cooldown", cooldown)`, `TAG("ability_tags", tags)`, `TAG("ability_blocked_by", blocked_by)`, `REGISTER("gameplay")`
    - Stores: `_ability_cost`, `_ability_cooldown`, `_ability_tags`, `_ability_blocked_by`
    - Validates: cooldown >= 0

11. **@buff(duration=None, stacking="none", max_stacks=1, tick_rate=0.0)** — Buff/debuff.
    - Valid stacking: "none", "duration", "intensity", "independent"
    - Stores: `_buff_duration`, `_buff_stacking`, `_buff_max_stacks`, `_buff_tick_rate`
    - Validates: max_stacks > 0, tick_rate >= 0

12. **@gameplay_tag(hierarchy: str)** — Hierarchical gameplay tag.
    - Stores: `_tag_hierarchy`
    - Validates: hierarchy non-empty

13. **@spawner(prefab: str, pool_size=10, spawn_rate=1.0, max_alive=None)** — Entity spawner.
    - Stores: `_spawner_prefab`, `_spawner_pool_size`, `_spawner_spawn_rate`, `_spawner_max_alive`
    - Validates: prefab non-empty, pool_size > 0, spawn_rate > 0

14. **@interactable(prompt: str, range=2.0, hold_time=0.0)** — Player-interactable object.
    - Stores: `_interactable_prompt`, `_interactable_range`, `_interactable_hold_time`
    - Validates: prompt non-empty, range > 0, hold_time >= 0

15. **@quest(id: str, prerequisites=[], rewards=[])** — Quest definition.
    - Stores: `_quest_id`, `_quest_prerequisites`, `_quest_rewards`
    - Validates: id non-empty

### AI Decorators (trinity/decorators/game_ai.py)

16. **@behavior_tree(id: str, debug_name=None)** — Behavior tree definition.
    - Stores: `_bt_id`, `_bt_debug_name`
    - Validates: id non-empty

17. **@utility_ai(id: str, update_rate=0.5)** — Utility AI system.
    - Stores: `_utility_id`, `_utility_update_rate`
    - Validates: id non-empty, update_rate > 0

18. **@blackboard** — AI blackboard (shared data store).
    - Sets `_blackboard = True`

19. **@ai_debug** — AI debug visualization.
    - Sets `_ai_debug = True`

20. **@perception(sense="sight", range=10.0, fov=None)** — AI perception.
    - Valid senses: "sight", "hearing", "damage", "squad"
    - Stores: `_perception_sense`, `_perception_range`, `_perception_fov`
    - Validates: range > 0

### Input Decorators (trinity/decorators/input.py)

21. **@input_action(name: str, default_bindings: list)** — Input action with key bindings.
    - Stores: `_action_name`, `_action_bindings`
    - Validates: name non-empty, bindings non-empty

22. **@input_axis(name: str, positive: list, negative: list)** — Input axis.
    - Stores: `_axis_name`, `_axis_positive`, `_axis_negative`
    - Validates: name non-empty, positive non-empty, negative non-empty

### State Machine Decorators (trinity/decorators/state_machine.py)

23. **@state_machine(initial: str, states: set, transitions: dict = None)** — FSM definition.
    - Validates: initial in states, states non-empty, all transition targets are valid states
    - Stores: `_sm_initial`, `_sm_states` (frozenset), `_sm_transitions`, `_sm_current_state`

24. **@on_enter(state: str)** — Hook called when entering a state.
    - Stores: `_on_enter_state`, `_lifecycle_hook = "enter"`

25. **@on_exit(state: str)** — Hook called when exiting a state.
    - Stores: `_on_exit_state`, `_lifecycle_hook = "exit"`

### Lifecycle Decorators (trinity/decorators/lifecycle.py)

26. **@on_add(component=ComponentType)** — Called when component added to entity.
27. **@on_remove(component=ComponentType)** — Called when component removed.
28. **@on_change(component=ComponentType)** — Called when component state changes.
29. **@on_spawn()** — Called when entity is spawned.
30. **@on_despawn()** — Called when entity is despawned.

### Scheduling Decorators (trinity/decorators/scheduling.py)

31. **@phase(name, after=(), before=())** — Execution phase.
32. **@parallel(chunk_size=64, min_batch=256)** — Multi-threaded. Excludes: `exclusive`.
33. **@exclusive()** — Sole world access. Excludes: `parallel`.
34. **@after(*systems)** — Run after specified systems.
35. **@before(*systems)** — Run before specified systems.
36. **@run_if(condition: Callable)** — Conditional execution.
37. **@fixed(hz=60)** — Fixed timestep. Excludes: `throttle`.
38. **@throttle(max_hz=None, max_ms=None)** — Rate limit. Excludes: `fixed`.
39. **@job(priority=0, affinity="any", stack_size=65536)** — Job properties.
40. **@async_system()** — Async system.
41. **@deferred()** — Defer structural changes.
42. **@chain(*systems)** — Explicit pipeline.

### Time Decorators (trinity/decorators/time.py)

43. **@deterministic** — Replay-safe marking.
44. **@time_scale(layer="gameplay", min_scale=0.0, max_scale=10.0)** — Per-system time scale.
45. **@pausable(pause_layers={"gameplay"})** — Pause/resume support.
46. **@rewindable(history_seconds=5.0, interpolation="linear")** — State rewind. Valid: "linear", "hermite", "none".

### Data Flow Decorators (trinity/decorators/data_flow.py)

47. **@serializable(format="binary", version=1)** — Serialization. Valid: "binary", "json", "msgpack".
48. **@track_changes** — Change tracking via Tracker.
49. **@networked(relevance="spatial", authority="server", priority=0, unreliable=False, delta=False, predicted=False, interpolated="none")** — Network replication.
    - Valid relevance: "global", "spatial", "owner"
    - Valid authority: "server", "client", "owner"
    - Valid interpolation: "linear", "hermite", "none"
50. **@snapshot(history_frames=60)** — State snapshot/restore. Requires @serializable.
51. **@versioned(version=1, migrations=None)** — Version migration. Requires @serializable.

### Dev Decorators (trinity/decorators/dev.py)

52. **@profile(name=None, warn_ms=None, track_allocations=False)** — Profiling.
53. **@trace(level="debug")** — Tracing. Valid: "debug", "info", "warn".
54. **@reloadable(enabled=True, preserve=[], reinitialize=[], validate=None)** — Hot reload.
55. **@editor(category="General", hidden=False)** — Editor integration.
56. **@test(cases=[], fuzz=False, property_based=False)** — Test config.
57. **@bench(iterations=1000, warmup=100)** — Benchmarking.
58. **@invariant(check: Callable, when="debug")** — Runtime invariant.
59. **@deprecated(since, replacement=None, remove_in=None)** — Deprecation.

---

## 3. All Metaclasses

### 1. ComponentMeta(EngineMeta)
- Generates unique `_component_id`
- Processes field annotations → `_field_types`, `_field_offsets`, `_field_defaults`
- Installs descriptors from Annotated types (Tracked, Validated, Networked, etc.)
- Pool management: pooled allocation via `__call__`, `return_to_pool()`, `pool_stats()`
- Budget tracking: `instance_count()`, enforces `max_instances`
- Layout: `get_layout_mode()` (SoA/AoS), `get_layout_arrays()`
- Registry: `_registry: dict[int, type]`, `_name_to_id: dict[str, int]`
- Methods: `get_by_id()`, `get_by_name()`, `all_components()`, `component_count()`

### 2. SystemMeta(EngineMeta)
- Generates unique `_system_id`
- Extracts `_reads`, `_writes`, `_resources` from decorators
- Analyzes dependencies (write-after-read conflicts)
- Checks parallelization safety
- Registry: `_registry: dict[int, type]`, `_phases: dict[SystemPhase, list[int]]`
- Methods: `get_phase_systems()`, `get_phase_order()` (topological sort), `get_parallel_groups()`, `hot_reload()`, `reload_system()`

### 3. EventMeta(EngineMeta)
- Generates unique `_event_id`
- Collects `_event_fields`, tracks inheritance via `_event_parent_ids`
- Validates data-only (no custom methods)
- Event pooling: `acquire()`, `release()`, `pool_stats()`
- Serialization: `serialize()`, `deserialize()`
- Registry: `_registry`, `_name_to_id`, `_event_pools`
- Methods: `is_subtype()`, `get_subtypes()`, `get_by_channel()`

### 4. StateMeta(EngineMeta)
- Generates unique `_state_id`
- Registers states with machines
- Validates transitions, tracks hierarchy (parent/children)
- Registry: `_registry: dict[type, dict[str, type]]`, `_state_history`
- Methods: `can_transition()`, `validate_transitions()`, `register_substate()`, `get_substates()`, `is_active_in_hierarchy()`, `record_transition()`, `get_previous_state()`, `get_history()`

### 5. ResourceMeta(EngineMeta)
- Singleton pattern, priority-based init order
- Registry: `_registry`, `_instances`
- Methods: `initialize_all()`, `shutdown_all()`, `get_or_create()`, `is_lazy()`

### 6. EngineMeta (base)
- `_all_engine_types: dict[str, type]`
- Auto-registers, processes Annotated fields, registers with Foundation Registry

---

## 4. All Descriptors

### Validation Descriptors (trinity/descriptors/validation.py)

1. **ValidatedDescriptor[T]** — Custom validators on write.
   - `validators: list[Callable]`
   - `add_validator(validator)` — add dynamically
   - `pre_set`: runs all validators, raises ValueError

2. **RangeDescriptor[T]** — Numeric range clamp/validate.
   - `min_val`, `max_val`, `clamp: bool`
   - `pre_set`: clamps if `clamp=True`, else raises ValueError

3. **TypeDescriptor[T]** — Runtime type enforcement.
   - `expected_type`, `coerce: bool`
   - `pre_set`: coerces or raises TypeError

4. **ChoiceDescriptor[T]** — Value must be in allowed set.
   - `choices: frozenset`
   - `pre_set`: raises ValueError if not in choices

5. **PatternDescriptor[T]** — String regex validation.
   - `pattern: str`
   - `pre_set`: raises ValueError if no fullmatch

### Tracking Descriptors (trinity/descriptors/tracking.py)

6. **TrackedDescriptor[T]** — Dirty flag tracking.
   - `post_set`: marks dirty in `_dirty_fields` set or `_dirty_mask` bitmask
   - Integrates with Foundation Tracker and EventLog
   - Helpers: `is_dirty()`, `get_dirty_fields()`, `clear_dirty()`, `clear_dirty_field()`

7. **VersionedDescriptor[T]** — Per-field version counter.
   - `post_set`: increments `_version_{field}`
   - `get_version(obj) -> int`

8. **DiffDescriptor[T]** — Previous value and change detection.
   - Strategies: "shallow", "deep", "structural", "custom"
   - `get_previous()`, `has_changed()`

### Observable Descriptors (trinity/descriptors/observable.py)

9. **ObservableDescriptor[T]** — Notifies observers on change.
   - `subscribe(obj, callback)`, `unsubscribe(obj, callback)`

10. **BoundDescriptor[T]** — Two-way binding to external data.

### Networking Descriptors (trinity/descriptors/networking.py)

11. **InterpolatedDescriptor[T]** — Smooth between network snapshots.
    - Modes: "linear", "hermite"
    - `get_interpolated(obj, t: float) -> T`

12. **PredictedDescriptor[T]** — Client-side prediction with rollback.
    - `rollback(obj, frames)`, `get_history(obj) -> list`
    - Default max_history: 30 frames

### Base Descriptor

```python
class BaseDescriptor(Generic[T]):
    descriptor_id: str = "base"
    accepts_inner: tuple[str, ...] = ("*",)
    accepts_outer: tuple[str, ...] = ("*",)
    excludes: tuple[str, ...] = ()
    
    __get__, __set__, __delete__, __set_name__
    pre_get, post_get, pre_set, post_set   # Lifecycle hooks
    get_metadata(), get_chain()             # Introspection
```

---

## 5. Foundation Integration Points

### 1. Registry — System & Component Discovery
- All gameplay systems register via SystemMeta → Foundation Registry
- All gameplay components register via ComponentMeta → Foundation Registry
- AI behavior tree nodes register via Foundation Registry
- Input mappings register as Resources via ResourceMeta

### 2. Tracker — Change Tracking
- TrackedDescriptor fields feed into Tracker automatically
- Health changes, ability cooldowns, attribute modifications all tracked
- Powers: undo/redo, dirty detection, network delta sync

### 3. EventLog — Decision Logging
- `@trace` on systems → EventLog records all system executions
- AI decisions logged for post-hoc debugging: "Why did the AI attack?"
- Quest progression logged as events
- Entity lifecycle events (spawn, destroy) recorded
- Query: `EventLog.query(entity=X, operation="AISystem")` → decision history

### 4. Mirror — Gameplay State Reflection
- Game state queryable through ShellLang: `Enemies.all.where(lambda e: e.ai.state == "aggressive")`
- Inspector shows ability cooldowns, buff stacks, AI state
- Editor integration via `@editor` decorator

### 5. Serializer / ContentStore — Save/Load
- Inventory, quest progress, player state all serializable
- `@serializable(format="binary")` on gameplay components
- ContentStore for structural sharing of save data

### 6. Bridge — ShellLang ↔ Gameplay
- ShellLang commands can spawn entities, modify components, trigger abilities
- AIInterface commands → engine operations
- Live debugging: pause game, inspect AI blackboard, step behavior trees

---

## 6. Architecture Spec Details

### AI Systems Detail

#### Perception
| Sense | Implementation |
|-------|----------------|
| Sight | Line-of-sight raycast, FOV cone |
| Hearing | Sound propagation, loudness falloff, occlusion |
| Damage | Hit detection notifications |
| Squad | Faction/allegiance detection |

Perception Memory: stimuli with timestamp, age-based decay, last known position.

#### Knowledge Representation
- **Blackboard**: Named key-value storage, observers, per-agent or shared scope
- **World State**: Boolean facts, query current conditions, diff vs goal state
- **Influence Maps**: Spatial grid overlay, propagation, decay, multiple layers

#### Decision Making
- **Behavior Trees**: Selector, Sequence, Parallel composites; Invert, Repeat, Timeout, Cooldown decorators; Action and Condition leaves; Blackboard for state; Abort on condition change
- **Utility AI**: Score options, response curves, select highest, multiple considerations per option
- **GOAP**: Current state → A* search through actions → goal state; action preconditions and effects
- **HTN**: Compound tasks decompose into primitives; recursive expansion; partial planning
- **MCTS**: Selection (UCB1), Expansion, Simulation (random playout), Backpropagation

#### Combat AI
- Attack, Defend, Flank, Retreat, Support behaviors
- Cover system, flanking, suppression, coordination
- Target selection: threat assessment, priority, opportunity, assignment

#### Social AI
- Factions, reputation, personal relationships
- Conversation, trade, follow, avoid behaviors

### Navigation Detail

#### NavMesh
- Generation: voxelization → region building → contour tracing → mesh building
- Parameters: agent radius, agent height, step height, max slope
- Runtime: static (pre-built), dynamic (runtime updates), tiled (streaming), obstacles (carving)

#### Pathfinding
| Algorithm | Description |
|-----------|-------------|
| A* | Optimal with heuristic |
| Jump Point Search | Grid optimization |
| Theta* | Any-angle paths |
| HPA* | Hierarchical for large worlds |

Path modification: smoothing, funnel (string pulling), corridor (formation width).

#### Steering
- Basic: Seek, Flee, Arrive, Pursue
- Group: Separation, Alignment, Cohesion → Flocking
- Avoidance: RVO, ORCA, Force-Based

#### Nav Links
| Type | Description |
|------|-------------|
| Jump Link | Gap crossing |
| Drop Link | Ledge descent |
| Climb Link | Ladder, wall climb |
| Teleport Link | Portal, elevator |

Smart Objects: definition, slots, reservation, context animation.

### Input System Detail

#### Devices
| Device | Input Types |
|--------|-------------|
| Keyboard | Key states, text input |
| Mouse | Position, buttons, scroll |
| Gamepad | Buttons, sticks, triggers |
| Touch | Points, gestures |
| Motion | Gyro, accelerometer |
| XR Controllers | VR/AR tracked input |

#### Raw Processing
- Dead zone (ignore small inputs)
- Response curve (non-linear mapping)
- Smoothing (filter noise)
- Invert (flip axis direction)

#### Action Mapping
```
Raw Input          Action Mapping         Gameplay
Spacebar/A ───► "Jump" Action    ───►  Character.Jump()
W/S/Stick  ───► "MoveForward"    ───►  Character.Move()
Mouse Delta───► "Look"           ───►  Camera.Rotate()
```

Action Triggers: Pressed, Released, Hold, Tap, Combo (sequence).

#### Input Contexts
- Context stack: push/pop based on game state
- Higher priority contexts consume input
- Passthrough allows lower context handling
- Examples: OnFoot, InVehicle, Menu, Dialogue

### Camera System Detail

| Type | Description |
|------|-------------|
| First Person | Through character's eyes |
| Third Person | Behind character |
| Top-Down | Overhead view |
| Isometric | Angled overhead |
| Free Cam | Debug camera |
| Cinematic | Scripted sequences |

Follow camera: target, offset, lag, lead.
Collision: raycast/sphere cast detection, pull-in/push-out response.
Effects: shake (Perlin noise, wave, trauma decay), FOV change, tilt, DOF focus.
Rails & volumes: spline rails, trigger volumes, blend regions.

### Ability System Detail

#### Execution Flow
```
ACTIVATE → COMMIT → EXECUTE → END
Check Can → Pay Costs → Run Logic → Cleanup
Start CD    Consume    Apply Effects  End CD
Grant Tags  Resources  Spawn Actors   Remove Tags
```

#### Effect Types
| Type | Description |
|------|-------------|
| Instant | One-time application |
| Duration | Time-limited |
| Infinite | Until removed |
| Periodic | Repeating tick |

Modifiers: Add (flat), Multiply (%), Override, Stacking.
Executions: Damage, Heal, Buff, Debuff.

#### Attributes
- Base value (unmodified), Current value (after modifiers), Min/Max bounds
- Common: Health, Mana, Stamina, Speed, Damage, Armor
- Derived: calculated from others, formula-based, cached with dirty flag

### Inventory Detail

#### Item Properties
- Stackable (max stack), Weight, Value, Rarity, Level requirement
- Types: Equipment, Consumable, Material, Key Item, Currency

#### Equipment
- Slots: Head, Chest, Hands, Legs, Feet, Weapon, Off-hand
- Effects: attribute bonuses, resistances, special effects
- Visuals: socket attachment, skin override, show/hide

#### Loot
- Tables: entries, weight (probability), conditions, nested tables
- Rolling: RNG selection, pity system, luck stat bonus

#### Crafting
- Recipes: ingredients, output, skill requirement, station
- Process: check requirements → consume ingredients → create output → quality variance

### Quest System Detail

#### Quest States
```
UNAVAILABLE → AVAILABLE → ACTIVE → COMPLETE → TURNED IN
                             ↓
                          FAILED
```

#### Objectives
- Kill, Collect, Talk, Reach, Escort, Interact
- Flow: Sequential, Parallel, Branching, Optional
- Tracking: counters (X of Y), flags, event listeners

### Dialogue System Detail

Node Types: Text, Choice, Branch (condition), Event (trigger), Random (variation).
Variables: local (per-conversation), global (persistent), quest-linked, world state facts.
Presentation: text box, portrait, choice buttons, voice sync, lip sync, camera shots.

### Game Modes

- Deathmatch, Team Deathmatch, Capture the Flag, King of the Hill, Battle Royale
- Components: rules, spawning logic, scoring, time limits, team configuration
- Scoring events: Kill, Death, Objective, Assist, Bonus

---

## 7. Decorator Stacks

### gameplay_ability(cooldown=1.0, max_stacks=1)
```python
@parameterized_stack
def gameplay_ability(cooldown=1.0, max_stacks=1) -> Stack:
    return stack(
        ability(cooldown=cooldown),
        buff(stacking="intensity", max_stacks=max_stacks),
        gameplay_tag(hierarchy="ability"),
        serializable(format="binary"),
        track_changes,
    )
```

### full_destruction(health=100.0, fracture_pattern="voronoi", pool_size=256)
```python
@parameterized_stack
def full_destruction(health=100.0, fracture_pattern="voronoi", pool_size=256) -> Stack:
    return stack(
        destructible(health=health),
        damage_type(),
        damage_resistance(),
        fracture(pattern=fracture_pattern),
        physics_material(),
        pooled(size=pool_size),
    )
```

### crafting_system(station_id="workbench")
```python
@parameterized_stack
def crafting_system(station_id="workbench") -> Stack:
    return stack(
        crafting_station(id=station_id),
        recipe(),
        ingredient(),
        loot_table(),
        salvage_recipe(),
        serializable(format="binary"),
    )
```

### production_component(pool_size=1024, layout="soa", category="gameplay")
```python
@parameterized_stack
def production_component(pool_size=DEFAULT_POOL_SIZE, layout="soa", category="gameplay") -> Stack:
    return stack(
        component,
        pool(size=pool_size, layout=layout),
        serializable(format="binary"),
        track_changes,
        editor(category=category),
    )
```

### multiplayer_character(pool_size=64, history_frames=30, version=1)
```python
@parameterized_stack
def multiplayer_character(pool_size=64, history_frames=30, version=1) -> Stack:
    return stack(
        production_component(pool_size=pool_size),
        networked(authority="server"),
        snapshot(history_frames=history_frames),
        versioned(version=version),
    )
```

### competitive_entity(pool_size=128, history_frames=600)
```python
@parameterized_stack
def competitive_entity(pool_size=128, history_frames=600) -> Stack:
    return stack(
        deterministic_data(),
        snapshot(history_frames=history_frames),
        networked(authority="shared", prediction=True),
        replay_ready(history_frames=history_frames),
    )
```

### safe_system(phase="update", read=(), write=())
```python
@parameterized_stack
def safe_system(phase="update", read=(), write=()) -> Stack:
    return stack(
        system(phase=phase),
        trace(level="debug"),
        profile(),
        invariant(check=lambda: True, when="debug"),
    )
```

### saveable_data(version=1, format="binary", migrations=None)
```python
@parameterized_stack
def saveable_data(version=1, format="binary", migrations=None) -> Stack:
    return stack(
        component,
        serializable(format=format),
        versioned(version=version),
        track_changes,
    )
```

---

## 8. TODO Checklist

### 9.1 Entity & Object Model
- [ ] Implement Actor class (entity + transform + lifecycle)
- [ ] Implement entity prefab instantiation from `@prefab` definitions
- [ ] Implement entity lifecycle hooks (spawn, begin play, tick, end play, destroy)
- [ ] Wire `@lifecycle` decorators → lifecycle event registration
- [ ] Integrate Foundation EventLog — log entity lifecycle events

### 9.2 Behavior Trees
- [ ] Implement BT runtime (selector, sequence, parallel, decorator nodes)
- [ ] Implement blackboard (shared AI state)
- [ ] Wire `@behavior_tree` decorator → BT definition
- [ ] Register BT node types via Foundation Registry

### 9.3 AI Systems
- [ ] Implement GOAP planner (goals, actions, world state)
- [ ] Implement utility AI (scoring, curves, considerations)
- [ ] Implement perception system (sight, hearing, awareness)
- [ ] Wire `@game_ai` decorators → AI agent configuration
- [ ] Integrate Foundation EventLog — log AI decisions for debugging

### 9.4 Navigation
- [ ] Implement NavMesh generation and pathfinding (A*, string pulling)
- [ ] Implement navigation agent (steering, avoidance, crowds)
- [ ] Wire `@navmesh` decorator → navigation mesh configuration

### 9.5 Input System
- [ ] Implement input action mapping (action → key bindings)
- [ ] Implement input contexts (gameplay, UI, menu)
- [ ] Implement input buffering and combo detection
- [ ] Wire `@input` decorators → input action definitions
- [ ] Register input mappings as Resources via ResourceMeta

### 9.6 Ability System
- [ ] Implement Gameplay Ability System (abilities, effects, attributes)
- [ ] Implement cooldowns, costs, targeting
- [ ] Wire `@ability`, `@buff`, `@gameplay_tag` decorators
- [ ] Wire TrackedDescriptor → attribute change tracking

### 9.7 Quest & Narrative
- [ ] Implement quest system (objectives, conditions, rewards)
- [ ] Implement dialogue system
- [ ] Wire `@narrative` and `@cinematics` decorators
- [ ] Integrate Foundation EventLog — log quest progression

### 9.8 Economy & Crafting
- [ ] Implement inventory system
- [ ] Implement crafting system (`@recipe`, `@ingredient`, `@crafting_station`)
- [ ] Implement economy (currencies, transactions, trading)
- [ ] Wire `@economy`, `@crafting` decorators
- [ ] Wire `@transactions` decorator → atomic inventory operations

### Camera System
- [ ] Implement camera controller (first person, third person, orbit, follow)
- [ ] Implement camera collision detection and response
- [ ] Implement camera effects (shake, FOV, tilt, DOF)
- [ ] Implement camera rails and trigger volumes

### State Machines
- [ ] Implement FSM runtime using StateMeta
- [ ] Implement HFSM (hierarchical states via register_substate)
- [ ] Implement pushdown automaton (state stack)
- [ ] Wire @state_machine, @on_enter, @on_exit decorators

### Combat & Scoring
- [ ] Implement damage calculation system
- [ ] Implement health management with attribute system
- [ ] Implement death detection and cleanup
- [ ] Implement teams and faction system
- [ ] Implement scoring (kills, objectives, assists)

### Game Modes
- [ ] Implement game mode base class (rules, spawning, scoring, time)
- [ ] Implement common modes (deathmatch, CTF, KOTH, battle royale)
- [ ] Implement match lifecycle (lobby → countdown → play → end → results)

---

## 9. Directory Structure

```
engine/gameplay/
├── __init__.py                    # Public API exports
├── GAMEPLAY_CONTEXT.md            # THIS FILE
│
├── entity/                        # Entity & Object Model
│   ├── __init__.py
│   ├── actor.py                   # Actor, DynamicActor, Pawn, Character
│   ├── prefab.py                  # Prefab/Blueprint instantiation
│   ├── lifecycle.py               # Entity lifecycle manager (create → destroy)
│   ├── possession.py              # Controller, PlayerController, AIController
│   └── spawner.py                 # SpawnerSystem — pooled entity creation
│
├── ai/                            # AI Systems
│   ├── __init__.py
│   ├── behavior_tree.py           # BT runtime (Selector, Sequence, Parallel, Decorator, Leaf)
│   ├── blackboard.py              # Blackboard key-value storage
│   ├── utility_ai.py              # Utility AI (scoring, curves, selection)
│   ├── goap.py                    # Goal-Oriented Action Planning
│   ├── perception.py              # Perception system (sight, hearing, damage, squad)
│   ├── knowledge.py               # World state, influence maps
│   ├── combat_ai.py               # Combat behaviors (attack, defend, flank, retreat)
│   └── social.py                  # Social AI (factions, reputation, relationships)
│
├── nav/                           # Navigation
│   ├── __init__.py
│   ├── navmesh.py                 # NavMesh generation and queries
│   ├── pathfinding.py             # A*, JPS, Theta*, HPA*
│   ├── steering.py                # Steering behaviors (seek, flee, arrive, flocking)
│   ├── avoidance.py               # Local avoidance (RVO, ORCA)
│   └── nav_links.py               # Jump, drop, climb, teleport links
│
├── input/                         # Input System
│   ├── __init__.py
│   ├── devices.py                 # Device management, hot-plug
│   ├── processing.py              # Dead zone, response curve, smoothing
│   ├── action_mapper.py           # Action mapping (raw input → gameplay action)
│   ├── axis_mapper.py             # Axis mapping (positive/negative bindings)
│   ├── context.py                 # Input context stack
│   ├── combo.py                   # Input buffering and combo detection
│   └── rebinding.py               # Runtime rebinding, save/load bindings
│
├── camera/                        # Camera System
│   ├── __init__.py
│   ├── controller.py              # Camera controllers (FP, TP, orbit, follow, free)
│   ├── collision.py               # Camera collision detection and response
│   ├── effects.py                 # Shake, FOV, tilt, DOF
│   └── rails.py                   # Spline rails, trigger volumes, blend regions
│
├── abilities/                     # Ability System
│   ├── __init__.py
│   ├── ability.py                 # Ability definition, activation, execution flow
│   ├── effects.py                 # Gameplay effects (instant, duration, infinite, periodic)
│   ├── attributes.py              # Attribute system (base, current, modifiers, derived)
│   ├── buff_system.py             # Buff/debuff stacking, duration, tick processing
│   ├── targeting.py               # Self, actor, point, area, confirmation targeting
│   └── tags.py                    # Gameplay tag system (hierarchical matching)
│
├── components/                    # Common Gameplay Components
│   ├── __init__.py
│   ├── transform.py               # Position, Rotation, Scale, Hierarchy
│   ├── health.py                  # Health, MaxHealth, Regeneration, DamageResistance
│   ├── movement.py                # Velocity, Speed, MovementMode
│   ├── team.py                    # Team, Faction, Allegiance, IFF tags
│   ├── stats.py                   # Attributes, Modifiers, DerivedValues
│   └── state_machine.py           # FSM runner component, HFSM, pushdown automaton
│
├── combat/                        # Combat System
│   ├── __init__.py
│   ├── damage.py                  # DamageSystem — calculation, application
│   ├── death.py                   # DeathSystem — detection, cleanup triggers
│   ├── scoring.py                 # ScoreSystem — kills, objectives, assists
│   └── game_mode.py               # GameMode — rules, spawning, match lifecycle
│
├── quest/                         # Quest & Narrative
│   ├── __init__.py
│   ├── quest.py                   # Quest definition, states, objectives, rewards
│   ├── tracker.py                 # QuestTracker — progress, counters, flags
│   ├── dialogue.py                # Dialogue graph, nodes, conditions, variables
│   └── journal.py                 # Quest log, HUD tracker, world markers
│
└── economy/                       # Economy & Crafting
    ├── __init__.py
    ├── inventory.py               # Inventory container, slots, stacking
    ├── equipment.py               # Equipment slots, attribute bonuses, visuals
    ├── loot.py                    # Loot tables, weighted rolling, pity system
    ├── crafting.py                # Recipes, ingredients, stations, quality
    ├── currency.py                # Currency types, transactions
    └── trading.py                 # Trading between entities
```

Estimated: ~50+ files across 10 subdirectories.

---

## 10. Canonical Usage Examples

### Example 1: Health Component with Tracked Attributes
```python
from typing import Annotated
from trinity.decorators import component, track_changes, serializable, editor
from trinity.descriptors import Tracked, Range, Validated

@component
@track_changes
@serializable(format="binary")
@editor(category="Combat")
class Health:
    current: Annotated[float, Tracked(), Range(0, 10000)] = 100.0
    max_hp: Annotated[float, Tracked(), Validated(gt=0)] = 100.0
    regen_rate: Annotated[float, Range(0, 100)] = 0.0
    is_invulnerable: bool = False
```

### Example 2: Damage System
```python
from trinity.decorators import system, phase, trace, after

@system(phase="update")
@phase(SystemPhase.UPDATE)
@after(AbilitySystem)
@trace(level="debug")
class DamageSystem:
    def update(self, world: World, dt: float):
        for eid, (health, dmg) in world.query(Health, PendingDamage):
            if not health.is_invulnerable:
                health.current = max(0, health.current - dmg.amount)
            world.remove(eid, PendingDamage)
            
            if health.current <= 0:
                self.emit(EntityDied(entity=eid))
```

### Example 3: AI with Behavior Tree + Perception
```python
from trinity.decorators.game_ai import behavior_tree, blackboard, perception

@component
@behavior_tree(id="guard_ai", debug_name="Guard Patrol")
@blackboard
@perception(sense="sight", range=20.0, fov=90.0)
class GuardAI:
    alert_level: float = 0.0
    last_known_target: Optional[EntityRef] = None
    patrol_index: int = 0
```

### Example 4: State Machine (Enemy AI States)
```python
from trinity.decorators.state_machine import state_machine, on_enter, on_exit

@state_machine(
    initial="idle",
    states={"idle", "patrol", "chase", "attack", "flee"},
    transitions={
        "idle": ["patrol"],
        "patrol": ["idle", "chase"],
        "chase": ["attack", "flee", "patrol"],
        "attack": ["chase", "flee"],
        "flee": ["idle"],
    }
)
class EnemyState:
    aggression: float = 0.5

@on_enter(state="chase")
def begin_chase(entity):
    entity.movement.speed *= 1.5

@on_exit(state="chase")
def end_chase(entity):
    entity.movement.speed /= 1.5
```

### Example 5: Input Action Mapping
```python
from trinity.decorators.input import input_action, input_axis

@input_action(name="jump", default_bindings=["Space", "Gamepad_A"])
def on_jump(player):
    if player.grounded:
        player.velocity.y = player.jump_force

@input_axis(name="move_forward", positive=["W", "Up"], negative=["S", "Down"])
def on_move_forward(player, value: float):
    player.velocity.z = value * player.move_speed
```

### Example 6: Ability with Cooldown and Cost
```python
from trinity.stacks import gameplay_ability

@gameplay_ability(cooldown=5.0, max_stacks=1)
@gameplay_tag(hierarchy="ability.offensive.fireball")
class Fireball:
    damage: Annotated[float, Tracked()] = 100.0
    radius: float = 5.0
    mana_cost: float = 50.0
    
    # Execution flow: ACTIVATE → COMMIT → EXECUTE → END
```

### Example 7: Inventory Item
```python
@component
@serializable(format="binary")
@editor(category="Inventory")
class Item:
    id: str = ""
    name: str = ""
    stack_count: Annotated[int, Range(1, 999)] = 1
    max_stack: int = 99
    weight: float = 0.0
    rarity: str = "common"  # common, uncommon, rare, epic, legendary
```

### Example 8: Quest Definition
```python
@quest(id="main_quest_01", prerequisites=[], rewards=[("gold", 500), ("xp", 1000)])
class DefendTheVillage:
    objectives = [
        {"type": "kill", "target": "bandit", "count": 10},
        {"type": "talk", "target": "village_elder"},
    ]
    
    # States: UNAVAILABLE → AVAILABLE → ACTIVE → COMPLETE → TURNED_IN
```

### Example 9: Multiplayer Character
```python
from trinity.stacks import multiplayer_character

@multiplayer_character(pool_size=64, history_frames=30, version=1)
class PlayerCharacter:
    position: Annotated[Fixed32, PredictedDescriptor(max_history=30)] = Fixed32(0)
    health: Annotated[int, Tracked()] = 100
    team_id: int = 0
```

### Example 10: Competitive Entity (Rollback Networking)
```python
from trinity.stacks import competitive_entity

@competitive_entity(pool_size=128, history_frames=600)
class FighterState:
    x: Annotated[Fixed32, PredictedDescriptor()] = Fixed32(0)
    y: Annotated[Fixed32, PredictedDescriptor()] = Fixed32(0)
    state: str = "idle"
    hitstun_frames: int = 0
```

---

## 11. Integration Patterns

### Pattern 1: Gameplay ↔ Foundation (AI Debug)
```
AI makes decision (Behavior Tree tick)
  → @trace logs to EventLog
  → EventLog.query(entity=X, operation="AISystem")
  → ShellLang: Enemies.all.where(lambda e: e.ai.state == "aggressive")
  → Mirror reflects AI blackboard for Inspector
```

### Pattern 2: Gameplay ↔ ECS (System Execution)
```
Engine main loop → SystemScheduler
  → SystemPhase.UPDATE
    → InputSystem (reads raw input, writes actions)
    → AISystem (reads perception, writes decisions)
    → AbilitySystem (reads actions/decisions, writes effects)
    → MovementSystem (reads velocity, writes position)
    → DamageSystem (reads pending damage, writes health)
    → DeathSystem (reads health <= 0, emits EntityDied)
    → CleanupSystem (processes deferred destroys)
```

### Pattern 3: Gameplay ↔ Abilities
```
Player presses ability key
  → InputSystem maps to action
  → AbilitySystem checks: can activate? (cooldown, cost, tags)
  → COMMIT: pay costs, start cooldown, grant tags
  → EXECUTE: play animation, spawn projectile, apply effect
  → Effect applies via TrackedDescriptor → Tracker detects change
  → END: remove tags, cleanup
```

### Pattern 4: Gameplay ↔ State Machine
```
StateMeta validates transitions at definition time
  → Runtime: StateMachineSystem checks conditions each tick
  → Transition fires: @on_exit(old_state) → @on_enter(new_state)
  → StateMeta.record_transition() → history tracked
  → EventLog records transition for replay/debug
```

### Pattern 5: Gameplay ↔ Networking
```
Authoritative server:
  → Server runs gameplay systems (abilities, damage, death)
  → @networked components replicated to clients
  → Clients predict locally (PredictedDescriptor)
  → On correction: rollback via SnapshotManager
  → Remote entities smoothed (InterpolatedDescriptor)
```

### Pattern 6: Gameplay ↔ Save/Load
```
Save Game:
  → Inventory: @serializable items serialized
  → Quest progress: @serializable quest state
  → Player stats: @versioned attributes with migration
  → World state: entity positions, health, AI states
  → All via Foundation Serializer → ContentStore

Load Game:
  → Deserialize → restore components
  → Reconnect systems → resume gameplay
```

---

## 12. Quick Reference Tables

### Gameplay Decorators Summary
| Decorator | File | Purpose |
|-----------|------|---------|
| @ability | gameplay.py | Ability with cost, cooldown, tags |
| @buff | gameplay.py | Buff/debuff with stacking |
| @gameplay_tag | gameplay.py | Hierarchical gameplay tag |
| @spawner | gameplay.py | Entity spawner with pooling |
| @interactable | gameplay.py | Player interaction |
| @quest | gameplay.py | Quest definition |
| @behavior_tree | game_ai.py | BT definition |
| @utility_ai | game_ai.py | Utility AI system |
| @blackboard | game_ai.py | AI shared data |
| @perception | game_ai.py | AI perception sense |
| @ai_debug | game_ai.py | AI debug visualization |
| @input_action | input.py | Input action binding |
| @input_axis | input.py | Input axis binding |
| @state_machine | state_machine.py | FSM definition |
| @on_enter | state_machine.py | State enter hook |
| @on_exit | state_machine.py | State exit hook |

### Gameplay Stacks Summary
| Stack | Purpose |
|-------|---------|
| gameplay_ability | Ability + buff + tags + serializable + tracked |
| full_destruction | Destructible + damage + fracture + physics + pooled |
| crafting_system | Station + recipe + ingredient + loot + salvage + serializable |
| multiplayer_character | Production + networked + snapshot + versioned |
| competitive_entity | Deterministic + snapshot + networked(predicted) + replay |

### System Execution Order
| Order | System | Phase | Reads | Writes |
|-------|--------|-------|-------|--------|
| 1 | InputSystem | PRE_UPDATE | RawInput | Actions |
| 2 | AISystem | UPDATE | Perception, Blackboard | Decisions |
| 3 | AbilitySystem | UPDATE | Actions, Decisions | Effects |
| 4 | EffectSystem | UPDATE | Effects | Attributes |
| 5 | MovementSystem | UPDATE | Velocity, Actions | Position |
| 6 | StateMachineSystem | UPDATE | Conditions | State |
| 7 | DamageSystem | UPDATE | PendingDamage | Health |
| 8 | DeathSystem | POST_UPDATE | Health | DeathEvent |
| 9 | CleanupSystem | POST_UPDATE | Deferred | Entities |
| 10 | TriggerSystem | POST_UPDATE | Position | Events |

### Component → Descriptor Mapping
| Component Field | Descriptor | Purpose |
|----------------|------------|---------|
| Health.current | TrackedDescriptor + RangeDescriptor | Track changes, clamp 0-max |
| Ability.cooldown | TrackedDescriptor | Track cooldown state |
| Attribute.value | TrackedDescriptor + ValidatedDescriptor | Track + validate |
| Position (MP) | PredictedDescriptor | Client prediction |
| Remote entity | InterpolatedDescriptor | Smooth rendering |
| Item.stack_count | RangeDescriptor | Clamp 1-max_stack |
| Score | TrackedDescriptor | Change notification |

### AI Decision Architecture
| Method | Best For | Complexity |
|--------|----------|------------|
| FSM | Simple agents, few states | Low |
| Behavior Tree | Complex sequences, hierarchical | Medium |
| Utility AI | Many options, scoring | Medium |
| GOAP | Goal-directed, planning | High |
| HTN | Task decomposition | High |
| MCTS | Adversarial search | Very High |

### Buff Stacking Modes
| Mode | Behavior |
|------|----------|
| none | Single instance only |
| duration | Refresh/extend duration |
| intensity | Stack effect magnitude |
| independent | Each instance separate |

### Quest States
| State | Transitions To |
|-------|---------------|
| UNAVAILABLE | AVAILABLE (prerequisites met) |
| AVAILABLE | ACTIVE (accepted) |
| ACTIVE | COMPLETE (objectives done), FAILED (conditions) |
| COMPLETE | TURNED_IN (claimed rewards) |
| FAILED | ACTIVE (retry), UNAVAILABLE (reset) |

### Perception Senses
| Sense | Config | Detection |
|-------|--------|-----------|
| Sight | range, fov | Raycast + FOV cone |
| Hearing | range | Sound propagation, occlusion |
| Damage | (auto) | Hit notification |
| Squad | (auto) | Faction detection |
