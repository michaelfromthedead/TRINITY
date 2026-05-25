# GAPSET_17_GAMEPLAY — PHASE_N_TODO.md

> **Task format**: T-GP-{PHASE}.{N}
> **Checkboxes**: `[ ]` = pending, `[X]` = done
> **Status**: All tasks pending (0% implemented)

---

## Phase 1: Entity Framework & Core (Foundation)

### Entity & Object Model

- [ ] **T-GP-1.1** — Implement Actor base component (entity ID + transform + lifecycle flags)
  - **Acceptance**: `@component` class `Actor` defined with position, rotation, scale fields. Field types use Trinity descriptors. Registers via ComponentMeta.
  - **Deps**: Foundation ComponentMeta, @component
  - **Effort**: Medium

- [ ] **T-GP-1.2** — Implement DynamicActor, Pawn, Character as component variants
  - **Acceptance**: DynamicActor adds physics body. Pawn adds possessable flag (Controller reference). Character adds movement component dependency.
  - **Deps**: T-GP-1.1
  - **Effort**: Medium

- [ ] **T-GP-1.3** — Implement entity prefab instantiation from decorators
  - **Acceptance**: Prefab defined via decorator stack can be instantiated to spawn a configured entity with all components. Supports parameter overrides at spawn time.
  - **Deps**: T-GP-1.1
  - **Effort**: Medium

- [ ] **T-GP-1.4** — Implement entity lifecycle hooks (spawn, begin play, tick, end play, destroy)
  - **Acceptance**: Entity advances through lifecycle stages. `@on_spawn`, `@on_add`, `@on_remove`, `@on_despawn` hooks fire at correct times. Deferred operations via `@deferred` batched to end of frame.
  - **Deps**: T-GP-1.1
  - **Effort**: High

- [ ] **T-GP-1.5** — Implement Controller, PlayerController, AIController
  - **Acceptance**: Controller component references a possessed entity. PlayerController binds to input actions. AIController plugs into AI decision system. Possession can switch at runtime.
  - **Deps**: T-GP-1.2
  - **Effort**: Medium

- [ ] **T-GP-1.6** — Implement @spawner decorator with object pooling
  - **Acceptance**: `@spawner(prefab, pool_size, spawn_rate, max_alive)` defines pooled spawner. Pool recycles entities. Max alive limit enforced. Spawn rate regulates per-frame spawns.
  - **Deps**: T-GP-1.3, T-GP-1.4
  - **Effort**: Medium

- [ ] **T-GP-1.7** — Wire Foundation EventLog for entity lifecycle events
  - **Acceptance**: Spawn, destroy, possession, state change events logged to EventLog. Queryable per entity.
  - **Deps**: T-GP-1.4, Foundation EventLog
  - **Effort**: Low

### State Machines

- [ ] **T-GP-1.8** — Implement FSM runtime using StateMeta
  - **Acceptance**: FSM defined by `@state_machine(initial, states, transitions)` executes transitions correctly. `@on_enter` and `@on_exit` hooks fire. Invalid transitions rejected.
  - **Deps**: Foundation StateMeta
  - **Effort**: Medium

- [ ] **T-GP-1.9** — Implement HFSM via StateMeta register_substate
  - **Acceptance**: Hierarchical states with parent/child nesting. Entering child auto-enters parent. Exiting parent auto-exits children. State active check includes hierarchy.
  - **Deps**: T-GP-1.8
  - **Effort**: Medium

- [ ] **T-GP-1.10** — Implement pushdown automaton (state stack)
  - **Acceptance**: States push/pop from stack. Pop returns to previous state. Supports temporary states (e.g., stunned overlay on combat state).
  - **Deps**: T-GP-1.8
  - **Effort**: Low

- [ ] **T-GP-1.11** — Register state machine system in gameplay execution order
  - **Acceptance**: StateMachineSystem runs in UPDATE phase, ticks all active FSM/HFSM/pushdown components, processes transitions, fires hooks.
  - **Deps**: T-GP-1.8, T-GP-1.9, T-GP-1.10
  - **Effort**: Medium

### Movement Components

- [ ] **T-GP-1.12** — Implement Movement component (velocity, speed, movement mode)
  - **Acceptance**: Movement component with velocity Vec3, speed float, movement mode enum (walking, running, sprinting, crouching, prone, swimming, flying, custom).
  - **Deps**: T-GP-1.2 (Character)
  - **Effort**: Low

---

## Phase 2: Input & Camera Systems

### Input System (4-Layer)

- [ ] **T-GP-2.1** — Implement device manager with hot-plug support
  - **Acceptance**: DeviceManager discovers and manages keyboard, mouse, gamepad, touch, motion, XR devices. Hot-plug events fire. Devices provide raw input events.
  - **Deps**: T-GP-1.1
  - **Effort**: High

- [ ] **T-GP-2.2** — Implement raw input processing (dead zone, response curve, smoothing, invert)
  - **Acceptance**: Processing pipeline chain: dead zone ignores small values; response curve maps linearly or exponentially; smoothing filters with moving average; inverts axis. Each processor configurable per-axis.
  - **Deps**: T-GP-2.1
  - **Effort**: Medium

- [ ] **T-GP-2.3** — Implement action mapper (@input_action decorator + resolution)
  - **Acceptance**: `@input_action(name, default_bindings)` registers action mapping. Each frame, mapper resolves raw input → action events. Trigger types: Pressed, Released, Hold, Tap, Combo. Multiple bindings per action.
  - **Deps**: T-GP-2.1, Foundation @input_action decorator
  - **Effort**: High

- [ ] **T-GP-2.4** — Implement axis mapper (@input_axis decorator + resolution)
  - **Acceptance**: `@input_axis(name, positive, negative)` registers axis mapping. Positive bindings produce +1, negative produce -1. Combined value in [-1, 1].
  - **Deps**: T-GP-2.1
  - **Effort**: Medium

- [ ] **T-GP-2.5** — Implement input context stack with priority
  - **Acceptance**: Context stack with push/pop. Higher-priority contexts consume matched inputs. Passthrough flag allows lower contexts to also receive. Contexts: OnFoot, InVehicle, Menu, Dialogue, Cutscene.
  - **Deps**: T-GP-2.3, T-GP-2.4
  - **Effort**: Medium

- [ ] **T-GP-2.6** — Implement input buffering and combo detection
  - **Acceptance**: Input buffer stores recent inputs (configurable window, default 500ms). Combo definitions match sequences. On match, combo action fires.
  - **Deps**: T-GP-2.3
  - **Effort**: Medium

- [ ] **T-GP-2.7** — Implement runtime rebinding with save/load
  - **Acceptance**: Binding can be changed at runtime. Rebinding UI support (receive new key/button). Bindings serializable to disk. Load on session start.
  - **Deps**: T-GP-2.3
  - **Effort**: Medium

- [ ] **T-GP-2.8** — Register input mappings as Resources via ResourceMeta
  - **Acceptance**: Input actions and axes registered as Engine resources. Auto-discovered on system init. Available for editor inspection.
  - **Deps**: T-GP-2.3, T-GP-2.4, Foundation ResourceMeta
  - **Effort**: Low

### Camera System (6 Modes)

- [ ] **T-GP-2.9** — Implement camera base component (mode, target, offset, parameters)
  - **Acceptance**: Camera component holds active mode enum, target entity ID, offset Vec3, lag float, FOV float. Registers via @component.
  - **Deps**: T-GP-1.1
  - **Effort**: Low

- [ ] **T-GP-2.10** — Implement camera controllers (first person, third person, orbit, follow, free)
  - **Acceptance**: Each controller implements `compute_transform(camera_component) -> Transform`. FP: ties to character head socket. TP: spring arm orbit. Orbit: rotates around pivot. Follow: tracks target with lag. Free: unconstrained WASD + mouse.
  - **Deps**: T-GP-2.9
  - **Effort**: High

- [ ] **T-GP-2.11** — Implement cinematic camera controller
  - **Acceptance**: Cinematic camera reads keyframes or spline path. Supports ease-in/out, cut transitions, crossfade. Control handoff at sequence start/end.
  - **Deps**: T-GP-2.9
  - **Effort**: Medium

- [ ] **T-GP-2.12** — Implement camera collision detection and response (spring arm)
  - **Acceptance**: Spring arm raycasts from target to desired offset. On collision: camera pulled to hit point. On clearance: camera pushes out to restore. Sphere cast for body radius.
  - **Deps**: T-GP-2.10
  - **Effort**: Medium

- [ ] **T-GP-2.13** — Implement camera effects (shake, FOV, tilt, DOF)
  - **Acceptance**: CameraEffects component holds active effects. Shake: configurable Perlin noise amplitude/frequency/decay. FOV: smooth zoom target/transition. Tilt: rotation roll. DOF: focus target distance.
  - **Deps**: T-GP-2.9
  - **Effort**: Medium

- [ ] **T-GP-2.14** — Implement camera rails and trigger volumes
  - **Acceptance**: Spline-based camera rail with position/timing control. Trigger volumes detect entity enter/exit. Blend regions transition smoothly between camera states.
  - **Deps**: T-GP-2.9
  - **Effort**: Medium

---

## Phase 3: AI Tier 1 — Behavior Trees + Utility AI + Perception

### Behavior Trees

- [ ] **T-GP-3.1** — Implement BT composite nodes (Selector, Sequence, Parallel)
  - **Acceptance**: Selector runs children until one succeeds. Sequence runs children until one fails. Parallel runs all children concurrently (configurable success/failure threshold).
  - **Deps**: T-GP-1.1
  - **Effort**: Medium

- [ ] **T-GP-3.2** — Implement BT decorator nodes (Invert, Repeat, Timeout, Cooldown)
  - **Acceptance**: Invert flips child result. Repeat runs child N times. Timeout fails child after duration. Cooldown rate-limits child execution.
  - **Deps**: T-GP-3.1
  - **Effort**: Medium

- [ ] **T-GP-3.3** — Implement BT leaf nodes (Action, Condition)
  - **Acceptance**: Action executes a callable (Python function or bound method). Condition evaluates a boolean predicate. Both can read/write blackboard.
  - **Deps**: T-GP-3.1
  - **Effort**: Medium

- [ ] **T-GP-3.4** — Implement BT runtime (tick-based traversal, blackboard binding)
  - **Acceptance**: Root node ticked each frame. Traversal descends through active branches. Blackboard values read/written during tick. Abort on condition change triggers re-evaluation.
  - **Deps**: T-GP-3.1, T-GP-3.2, T-GP-3.3
  - **Effort**: High

- [ ] **T-GP-3.5** — Wire @behavior_tree decorator to BT definition
  - **Acceptance**: `@behavior_tree(id, debug_name)` on a component class registers tree structure. Debug name appears in logging/tooling.
  - **Deps**: T-GP-3.4
  - **Effort**: Low

- [ ] **T-GP-3.6** — Register BT node types via Foundation Registry
  - **Acceptance**: All node types (Selector, Sequence, Parallel, Invert, Repeat, Timeout, Cooldown, Action, Condition) registered. Available for runtime node creation from data.
  - **Deps**: T-GP-3.1, T-GP-3.2, T-GP-3.3
  - **Effort**: Low

### Blackboard

- [ ] **T-GP-3.7** — Implement Blackboard (key-value store with observers)
  - **Acceptance**: Typed key-value store (bool, int, float, Vec3, EntityRef, string). Observers notify on value change. Scoping: per-agent, squad-shared, global. Serialization support.
  - **Deps**: T-GP-1.1
  - **Effort**: Medium

- [ ] **T-GP-3.8** — Wire @blackboard decorator
  - **Acceptance**: `@blackboard` on component class marks it as blackboard-using. Blackboard initialized on entity spawn.
  - **Deps**: T-GP-3.7
  - **Effort**: Low

### Utility AI

- [ ] **T-GP-3.9** — Implement Utility AI scoring system (options, considerations, curves)
  - **Acceptance**: Option contains 1+ considerations. Each consideration maps input value through response curve to score [0,1]. Options scored per tick (or configurable interval). Highest score selected.
  - **Deps**: T-GP-1.1
  - **Effort**: High

- [ ] **T-GP-3.10** — Implement response curve types (linear, exponential, logistic, power, constant)
  - **Acceptance**: Each curve type maps float [0,1] → [0,1] with configurable parameters (slope, midpoint, etc.). Curves are assets, serializable.
  - **Deps**: T-GP-3.9
  - **Effort**: Medium

- [ ] **T-GP-3.11** — Implement Utility AI selector (deterministic, weighted random)
  - **Acceptance**: Deterministic: picks highest score. Weighted random: higher scores more likely but not guaranteed. Configurable per agent.
  - **Deps**: T-GP-3.9
  - **Effort**: Low

- [ ] **T-GP-3.12** — Wire @utility_ai decorator
  - **Acceptance**: `@utility_ai(id, update_rate)` registers utility AI definition. Update rate controls re-evaluation frequency.
  - **Deps**: T-GP-3.9
  - **Effort**: Low

### Perception

- [ ] **T-GP-3.13** — Implement sight perception (raycast, FOV cone)
  - **Acceptance**: Periodic raycast sweep within FOV cone. Configurable range and FOV angle. Occlusion checked via physics raycast. Stimuli generated on detection (entity, last known position, timestamp).
  - **Deps**: T-GP-1.1
  - **Effort**: High

- [ ] **T-GP-3.14** — Implement hearing perception (sound propagation, loudness, occlusion)
  - **Acceptance**: Sound events propagate with distance-based falloff. Occlusion reduces effective loudness. Stimuli generated when loudness exceeds threshold.
  - **Deps**: T-GP-1.1
  - **Effort**: Medium

- [ ] **T-GP-3.15** — Implement damage and squad perception (passive senses)
  - **Acceptance**: Damage perception: combat events auto-generate stimuli. Squad perception: faction/allegiance system auto-detects nearby entities.
  - **Deps**: T-GP-1.1
  - **Effort**: Low

- [ ] **T-GP-3.16** — Implement perception memory (stimuli decay, last known positions)
  - **Acceptance**: Stimuli stored with timestamp. Aged stimuli decay in priority and eventually removed. Last known position tracked for unseen targets.
  - **Deps**: T-GP-3.13, T-GP-3.14, T-GP-3.15
  - **Effort**: Medium

- [ ] **T-GP-3.17** — Wire @perception decorator
  - **Acceptance**: `@perception(sense, range, fov)` decorates AI component. System reads config to initialize perception subsystem.
  - **Deps**: T-GP-3.13
  - **Effort**: Low

### Knowledge

- [ ] **T-GP-3.18** — Implement World State (boolean facts)
  - **Acceptance**: Set of boolean facts about game world. Queries: `world_state.has("enemy_sighted")`. Events can set/clear facts.
  - **Deps**: T-GP-1.1
  - **Effort**: Low

- [ ] **T-GP-3.19** — Implement Influence Maps (spatial grid with propagation)
  - **Acceptance**: Grid overlay on world. Values propagate (decay with distance). Multiple layers (threat, resource, interest). Queries: `influence_map.sample(layer, position)`.
  - **Deps**: T-GP-1.1
  - **Effort**: High

### Combat AI (Basic)

- [ ] **T-GP-3.20** — Implement basic combat behaviors (Attack, Defend, Retreat)
  - **Acceptance**: Attack behavior: approach target, use weapon/ability. Defend behavior: block, dodge, take cover. Retreat behavior: flee to safe position. Each is a BT subtree.
  - **Deps**: T-GP-3.1, T-GP-3.7
  - **Effort**: Medium

### Social AI (Basic)

- [ ] **T-GP-3.21** — Implement faction system (teams, IFF)
  - **Acceptance**: Faction component with team_id, faction string, attitude (hostile/neutral/friendly). IFF queries: is_enemy, is_ally, is_neutral.
  - **Deps**: T-GP-1.1
  - **Effort**: Low

### AI System Integration

- [ ] **T-GP-3.22** — Integrate Foundation EventLog — log AI decisions
  - **Acceptance**: Each AI decision (BT tick, utility selection, perception event) logged with entity ID, timestamp, decision type, relevant data. Queryable for debugging.
  - **Deps**: T-GP-3.4, T-GP-3.9, T-GP-3.16, Foundation EventLog
  - **Effort**: Low

- [ ] **T-GP-3.23** — Wire @ai_debug decorator for AI visualization
  - **Acceptance**: `@ai_debug` enables debug visualization: BT node states (running/succeeded/failed), perception ranges (wireframe cone/circle), influence map overlay.
  - **Deps**: T-GP-3.4, T-GP-3.13
  - **Effort**: Medium

---

## Phase 4: AI Tier 2 — GOAP Planner

- [ ] **T-GP-4.1** — Implement GOAP WorldState (boolean + numeric facts)
  - **Acceptance**: WorldState is a set of typed facts (bool, int, float, Vec3). Supports: `set(fact, value)`, `get(fact)`, `matches(conditions)`, `diff(target_state)`.
  - **Deps**: T-GP-3.18 (shared concept with BT WorldState)
  - **Effort**: Medium

- [ ] **T-GP-4.2** — Implement GOAP Action (preconditions + effects)
  - **Acceptance**: Action specifies preconditions (facts that must be true) and effects (facts that change). Cost value. Executable: `execute(agent, world_state)`. Serializable.
  - **Deps**: T-GP-4.1
  - **Effort**: Medium

- [ ] **T-GP-4.3** — Implement GOAP Goal (desired world state with priority)
  - **Acceptance**: Goal has target state and priority. Goal arbitration: highest-priority achievable goal selected. Achievable = planner found valid plan.
  - **Deps**: T-GP-4.1
  - **Effort**: Low

- [ ] **T-GP-4.4** — Implement GOAP Planner (A* search over actions)
  - **Acceptance**: A* search: current state → actions with applicable preconditions → new state → goal. Forward search. Heuristic: distance to goal. Returns action plan or failure.
  - **Deps**: T-GP-4.2, T-GP-4.3
  - **Effort**: High

- [ ] **T-GP-4.5** — Implement GOAP plan execution and monitoring
  - **Acceptance**: Plan executor runs actions sequentially. On action failure: replan. On world state change: re-evaluate plan validity. On goal change: replan for new goal.
  - **Deps**: T-GP-4.4
  - **Effort**: Medium

### Combat AI (Expanded)

- [ ] **T-GP-4.6** — Implement advanced combat behaviors (Flank, Support, Cover)
  - **Acceptance**: Flank behavior: navigate to flanking position (uses nav system). Support behavior: assist allies, suppress enemies. Cover behavior: find/use/traverse cover points. Each as GOAP-expressible action.
  - **Deps**: T-GP-3.20, T-GP-4.4
  - **Effort**: High

- [ ] **T-GP-4.7** — Implement target selection system (threat assessment, priority, opportunity)
  - **Acceptance**: TargetSelector evaluates each known entity: threat score (damage output, proximity), priority score (objective relevance), opportunity score (exposed, low HP). Weighted combination.
  - **Deps**: T-GP-3.16, T-GP-4.6
  - **Effort**: Medium

---

## Phase 5: Navigation System

### NavMesh Generation

- [ ] **T-GP-5.1** — Implement NavMesh voxelization (walkable surface rasterization)
  - **Acceptance**: Rasterize world geometry into voxel grid. Agent radius/height/step_height/max_slope configurable. Output: voxel field.
  - **Deps**: Foundation math types (Vec3, AABB, Ray)
  - **Effort**: High

- [ ] **T-GP-5.2** — Implement NavMesh region building (contiguous walkable regions)
  - **Acceptance**: Walkable voxels merged into contiguous regions. Watershed or monotone partitioning algorithm. Regions are connected sets.
  - **Deps**: T-GP-5.1
  - **Effort**: High

- [ ] **T-GP-5.3** — Implement NavMesh contour tracing and mesh building
  - **Acceptance**: Region boundaries extracted as simplified polylines. Polylines triangulated into navigation mesh polygons. Edge classification: walkable/obstacle.
  - **Deps**: T-GP-5.2
  - **Effort**: High

- [ ] **T-GP-5.4** — Implement static NavMesh (pre-built, compile-time)
  - **Acceptance**: NavMesh generated from world geometry at level load. Immutable at runtime. Fast query via polygon graph.
  - **Deps**: T-GP-5.3
  - **Effort**: Medium

- [ ] **T-GP-5.5** — Implement dynamic NavMesh (runtime obstacle carving)
  - **Acceptance**: Obstacles carve holes in existing NavMesh at runtime. Re-connection of surrounding polygons after carving. Performance within frame budget.
  - **Deps**: T-GP-5.3
  - **Effort**: High

- [ ] **T-GP-5.6** — Implement tiled NavMesh (streaming for large worlds)
  - **Acceptance**: NavMesh partitioned into tiles. Tiles loaded/unloaded by proximity (player/AI location). Seamless transitions between tiles.
  - **Deps**: T-GP-5.3
  - **Effort**: High

### Pathfinding

- [ ] **T-GP-5.7** — Implement A* pathfinder over NavMesh graph
  - **Acceptance**: A* search on NavMesh polygon graph. Heuristic: Euclidean distance. Returns polygon path + waypoint positions. Configurable heuristic weight.
  - **Deps**: T-GP-5.4
  - **Effort**: Medium

- [ ] **T-GP-5.8** — Implement Jump Point Search pathfinder (grid optimization)
  - **Acceptance**: JPS for grid-based navigation. Jump points: forced neighbors, straight/Diagonal pruning. 10-100x fewer nodes expanded than A* on grids.
  - **Deps**: Foundation math types (Vec2, Vec3)
  - **Effort**: High

- [ ] **T-GP-5.9** — Implement Theta* pathfinder (any-angle)
  - **Acceptance**: Theta* on grids: line-of-sight checks replace parent with grandparent if visible. Produces shorter, smoother paths than A* on grids. O(n^2) worst case.
  - **Deps**: Foundation math types (Vec2, Vec3, line-of-sight)
  - **Effort**: High

- [ ] **T-GP-5.10** — Implement HPA* hierarchical planner
  - **Acceptance**: HPA* abstracts grid into hierarchy: clusters → abstract graph → intra-cluster paths. Multi-level planning for large worlds (e.g., 1000x1000+). Abstract then refine.
  - **Deps**: T-GP-5.7
  - **Effort**: High

- [ ] **T-GP-5.11** — Implement path smoothing (funnel algorithm / string pulling)
  - **Acceptance**: Funnel algorithm produces shortest path through polygon sequence. Also: path simplification (reduce waypoint count), curve smoothing (Catmull-Rom spline).
  - **Deps**: T-GP-5.7
  - **Effort**: Medium

- [ ] **T-GP-5.12** — Implement path corridor (path width for formation movement)
  - **Acceptance**: Path represented as corridor (left/right edges) instead of single line. Agents stay within corridor. Formation offsets within corridor width.
  - **Deps**: T-GP-5.7
  - **Effort**: Medium

### Steering

- [ ] **T-GP-5.13** — Implement basic steering behaviors (Seek, Flee, Arrive, Pursue)
  - **Acceptance**: Seek: force toward target. Flee: force away. Arrive: decelerate approaching target. Pursue: seek intercept point (predict target position).
  - **Deps**: T-GP-1.1
  - **Effort**: Medium

- [ ] **T-GP-5.14** — Implement group steering (Separation, Alignment, Cohesion → Flocking)
  - **Acceptance**: Separation: avoid crowding neighbors. Alignment: match velocity. Cohesion: move toward group center. Weighted sum = flocking behavior.
  - **Deps**: T-GP-5.13
  - **Effort**: Medium

### Local Avoidance

- [ ] **T-GP-5.15** — Implement RVO (Reciprocal Velocity Obstacles)
  - **Acceptance**: Each agent selects collision-free velocity assuming others reciprocate. Solves oscillation problem of basic VO. Configurable neighbor radius and time horizon.
  - **Deps**: T-GP-5.13
  - **Effort**: High

- [ ] **T-GP-5.16** — Implement ORCA (Optimal Reciprocal Collision Avoidance)
  - **Acceptance**: ORCA computes optimal collision-free velocity via linear optimization. Guaranteed collision avoidance for all agents. Faster and smoother than RVO.
  - **Deps**: T-GP-5.13
  - **Effort**: High

- [ ] **T-GP-5.17** — Implement force-based avoidance
  - **Acceptance**: Social forces model: repulsive force from each agent proportional to proximity. Less precise, more natural-looking than RVO/ORCA. Good for crowds.
  - **Deps**: T-GP-5.13
  - **Effort**: Medium

### Navigation Links

- [ ] **T-GP-5.18** — Implement NavLink types (Jump, Drop, Climb, Teleport)
  - **Acceptance**: Each link type connects two NavMesh polygons with cost multiplier. Validation function checks if link is traversable (e.g., jump height/distance within limits).
  - **Deps**: T-GP-5.4
  - **Effort**: Medium

- [ ] **T-GP-5.19** — Implement Smart Objects (definition, slots, reservation)
  - **Acceptance**: Smart object has N slots. Agents reserve slots before use. Context animation plays while occupying. Reservation prevents conflicts.
  - **Deps**: T-GP-5.18
  - **Effort**: Medium

---

## Phase 6: AI Tier 3 — MCTS Solver

- [ ] **T-GP-6.1** — Implement MCTS tree node (state, visits, score, children)
  - **Acceptance**: Node stores game state (reference or clone), visit count, total score, child nodes. Children indexed by action.
  - **Deps**: Phase 1 entity framework
  - **Effort**: Medium

- [ ] **T-GP-6.2** — Implement MCTS Selection phase (UCB1)
  - **Acceptance**: UCB1 formula: `score = Q + C * sqrt(ln(N) / n)`. Q = average reward, N = parent visits, n = node visits, C = exploration constant. Selects child with highest UCB1.
  - **Deps**: T-GP-6.1
  - **Effort**: Medium

- [ ] **T-GP-6.3** — Implement MCTS Expansion phase
  - **Acceptance**: On first visit to a node, expand all legal actions as child nodes. One child per action. Expansion only occurs once per node.
  - **Deps**: T-GP-6.1
  - **Effort**: Low

- [ ] **T-GP-6.4** — Implement MCTS Simulation phase (random playout / heuristic)
  - **Acceptance**: From expanded state, play out game to terminal state. Default: random action selection. Pluggable: domain-specific heuristic policy for better playouts.
  - **Deps**: T-GP-6.1
  - **Effort**: Medium

- [ ] **T-GP-6.5** — Implement MCTS Backpropagation phase
  - **Acceptance**: Terminal result (win/loss/draw, score) propagated up tree. Each node updates: visit count += 1, total score += result. Average = total/visits.
  - **Deps**: T-GP-6.1
  - **Effort**: Low

- [ ] **T-GP-6.6** — Implement MCTS iteration budget and best-action selection
  - **Acceptance**: Configurable iteration budget (time-based in ms, or count-based). After budget exhausted, selects most-visited (not highest-scored) child as best action.
  - **Deps**: T-GP-6.2, T-GP-6.3, T-GP-6.4, T-GP-6.5
  - **Effort**: Medium

- [ ] **T-GP-6.7** — Implement MCTS integration with combat AI (domain adapter)
  - **Acceptance**: GameState interface implemented for combat scenarios: legal actions (movement, attack, defend, use ability), terminal conditions (all enemies dead, time elapsed), reward function.
  - **Deps**: T-GP-6.6, Phase 4 combat AI
  - **Effort**: High

- [ ] **T-GP-6.8** — Implement MCTS fallback chain (MCTS → HTN → GOAP → Utility → BT)
  - **Acceptance**: Agent auto-selects AI tier: try MCTS first; if budget insufficient or scenario unsuitable, fall back to HTN, then GOAP, then Utility, then BT. Configurable per agent.
  - **Deps**: T-GP-4.4, T-GP-6.6
  - **Effort**: Medium

---

## Phase 7: Ability System (GAS-Style)

- [ ] **T-GP-7.1** — Implement ability definition and activation flow (ACTIVATE → COMMIT → EXECUTE → END)
  - **Acceptance**: 4-phase state machine. ACTIVATE: checks can-activate (cooldown, cost, tags, blocked_by). COMMIT: pays costs, starts cooldown. EXECUTE: runs effects. END: cleanup. Each phase fires events.
  - **Deps**: T-GP-1.1
  - **Effort**: High

- [ ] **T-GP-7.2** — Wire @ability decorator to ability registration and execution
  - **Acceptance**: `@ability(cost, cooldown, tags, blocked_by)` decorates an ability class. Class registered with AbilitySystem. Execution dispatches to decorated class's methods.
  - **Deps**: T-GP-7.1
  - **Effort**: Low

- [ ] **T-GP-7.3** — Implement gameplay effects (Instant, Duration, Infinite, Periodic)
  - **Acceptance**: Instant: apply once. Duration: apply for N seconds, then remove. Infinite: apply until removed. Periodic: tick every N seconds. Each effect has modifier(s).
  - **Deps**: T-GP-7.1
  - **Effort**: High

- [ ] **T-GP-7.4** — Wire @buff decorator with stacking modes
  - **Acceptance**: `@buff(duration, stacking, max_stacks, tick_rate)` defines buff. Stacking modes: none, duration, intensity, independent. Each mode correctly manages multiple instances.
  - **Deps**: T-GP-7.3
  - **Effort**: Medium

- [ ] **T-GP-7.5** — Implement attribute system (base, current, modifiers, derived)
  - **Acceptance**: Attribute has base value and modifier list (flat add, multiply, override). Current value = (base + flat sums) * multiply product. Derived attributes with formula and dirty cache. `TrackedDescriptor` for change tracking.
  - **Deps**: T-GP-7.1
  - **Effort**: High

- [ ] **T-GP-7.6** — Implement targeting modes (Self, Actor, Point, Area, Confirmation)
  - **Acceptance**: Self: caster only. Actor: raycast/overlap for target entity. Point: world position. Area: radius query. Confirmation: player must confirm (UI reticle). All configurable per ability.
  - **Deps**: T-GP-7.1
  - **Effort**: Medium

- [ ] **T-GP-7.7** — Implement gameplay tag system (hierarchical matching)
  - **Acceptance**: Tags are hierarchical: `ability.offensive.fireball`. Parent matches children unless explicitly excluded. Queries: `has_tag(tag)`, `matches(tag_query)`, `blocked_by(tags)`.
  - **Deps**: T-GP-7.1
  - **Effort**: Medium

- [ ] **T-GP-7.8** — Wire @gameplay_tag decorator
  - **Acceptance**: `@gameplay_tag(hierarchy)` tags a class. Tags inherited from decorator stacks. Serializable.
  - **Deps**: T-GP-7.7
  - **Effort**: Low

- [ ] **T-GP-7.9** — Implement cooldown and cost management
  - **Acceptance**: Cooldown tracks per-ability, persistent across activations. Costs: mana, stamina, health, resources. Cost validation in ACTIVATE phase. Cost payment in COMMIT phase.
  - **Deps**: T-GP-7.1
  - **Effort**: Medium

- [ ] **T-GP-7.10** — Register ability system in gameplay execution order
  - **Acceptance**: AbilitySystem runs in UPDATE phase (order 3, after Input, after AI). Tick: process active ability activations, tick active effects, update cooldowns.
  - **Deps**: T-GP-7.1
  - **Effort**: Medium

- [ ] **T-GP-7.11** — Wire TrackedDescriptor → ability attribute change tracking
  - **Acceptance**: Attribute fields with `TrackedDescriptor` automatically report changes to Foundation Tracker. UI binding, network replication hooks.
  - **Deps**: T-GP-7.5, Foundation TrackedDescriptor
  - **Effort**: Low

---

## Phase 8: Inventory, Equipment, Loot, Crafting & Economy

### Inventory

- [ ] **T-GP-8.1** — Implement inventory container (slots, item stacks, capacity)
  - **Acceptance**: Inventory component with N slots. Each slot holds item ID + stack count. Add/remove/transfer operations. Weight capacity limit. Stack size limit per item type.
  - **Deps**: T-GP-1.1
  - **Effort**: High

- [ ] **T-GP-8.2** — Implement Item component (id, name, stack_count, max_stack, weight, rarity)
  - **Acceptance**: Item ECS component with all properties. `@component` + `@serializable(format="binary")`. Rarity: common/uncommon/rare/epic/legendary.
  - **Deps**: T-GP-1.1
  - **Effort**: Low

- [ ] **T-GP-8.3** — Implement item types (Equipment, Consumable, Material, KeyItem, Currency)
  - **Acceptance**: Each type is a sub-component or tag on Item. Equipment adds slot, stats, visuals. Consumable adds effect-on-use. Material tags for crafting. KeyItem marks non-disposable.
  - **Deps**: T-GP-8.2
  - **Effort**: Medium

### Equipment

- [ ] **T-GP-8.4** — Implement equipment slots (head, chest, hands, legs, feet, weapon, off-hand)
  - **Acceptance**: Equipment component with slot array. Equip item to slot (validates slot type compatibility). Unequip returns item to inventory. Preview support (swap without committing).
  - **Deps**: T-GP-8.1
  - **Effort**: Medium

- [ ] **T-GP-8.5** — Implement equipment attribute bonuses (modifier application)
  - **Acceptance**: Equipment items carry attribute modifiers. On equip: modifiers applied to entity attributes. On unequip: modifiers removed. Modifier operators: add, multiply, override.
  - **Deps**: T-GP-8.4, T-GP-7.5 (attribute system)
  - **Effort**: Medium

- [ ] **T-GP-8.6** — Implement equipment visuals (socket attachment, skin override)
  - **Acceptance**: Socket system: item attaches to named bone/socket on character skeleton. Skin override: equipment replaces character material/texture. Show/hide slot toggles.
  - **Deps**: T-GP-8.4
  - **Effort**: Medium

### Loot

- [ ] **T-GP-8.7** — Implement loot tables (entries, weights, conditions, nested tables)
  - **Acceptance**: LootTable asset: list of entries with item_ref + weight + conditions. Nested tables for groups. Conditions: player level, quest state, game progress.
  - **Deps**: T-GP-8.2
  - **Effort**: Medium

- [ ] **T-GP-8.8** — Implement loot rolling (RNG selection, pity system, luck bonus)
  - **Acceptance**: RNG selection from weighted entries. Pity system: increases probability of rare drops after consecutive failures. Luck stat bonus modifies weights. Returns item list.
  - **Deps**: T-GP-8.7
  - **Effort**: Medium

### Crafting

- [ ] **T-GP-8.9** — Implement crafting recipes and stations
  - **Acceptance**: Recipe: ingredient list (item + count) + output item + quantity + station requirement. Station: workbench, forge, cooking fire, etc. Recipe discovery system.
  - **Deps**: T-GP-8.1, T-GP-8.2
  - **Effort**: Medium

- [ ] **T-GP-8.10** — Implement crafting process (check → consume → create → quality)
  - **Acceptance**: Validate requirements (has ingredients, correct station, sufficient skill). Consume ingredients from inventory. Create output item(s). Roll quality variance based on skill.
  - **Deps**: T-GP-8.9
  - **Effort**: Medium

### Economy

- [ ] **T-GP-8.11** — Implement currency system (multiple currency types)
  - **Acceptance**: Currency component tracks amounts per currency type (gold, silver, copper, tokens, etc.). Add/remove/transfer operations. Exchange rates between types.
  - **Deps**: T-GP-8.1
  - **Effort**: Low

- [ ] **T-GP-8.12** — Implement trading between entities
  - **Acceptance**: TradeOffer: items + currencies from each party. Offer lifetime (accept/reject/timeout). Atomic execution: both sides transfer simultaneously on acceptance.
  - **Deps**: T-GP-8.1, T-GP-8.11
  - **Effort**: Medium

### Wiring

- [ ] **T-GP-8.13** — Wire @economy, @crafting, @recipe, @ingredient, @crafting_station decorators
  - **Acceptance**: All economy/crafting decorators registered and functional. Each provides validation at decoration time (not runtime).
  - **Deps**: T-GP-8.9, T-GP-8.11
  - **Effort**: Low

- [ ] **T-GP-8.14** — Wire @serializable on all inventory/economy components
  - **Acceptance**: Inventory, equipment, item, currency components serializable. Save/load player inventory. ContentStore integration for structural sharing.
  - **Deps**: T-GP-8.1, Foundation Serializer
  - **Effort**: Medium

---

## Phase 9: Quest & Dialogue Systems

### Quests

- [ ] **T-GP-9.1** — Implement quest definition and state machine (5 states)
  - **Acceptance**: Quest with states: UNAVAILABLE → AVAILABLE → ACTIVE → COMPLETE → TURNED_IN, plus FAILED. Prerequisites gate UNAVAILABLE→AVAILABLE. Objectives gate ACTIVE→COMPLETE.
  - **Deps**: T-GP-1.1
  - **Effort**: Medium

- [ ] **T-GP-9.2** — Implement objective types (Kill, Collect, Talk, Reach, Escort, Interact)
  - **Acceptance**: Each objective type tracks progress. Kill: counter on entity kills. Collect: counter on item pickup. Talk: flag on dialogue completion. Reach: flag on position trigger. Escort: status of escorted entity. Interact: flag on interaction trigger.
  - **Deps**: T-GP-9.1
  - **Effort**: High

- [ ] **T-GP-9.3** — Implement objective flow patterns (Sequential, Parallel, Branching, Optional)
  - **Acceptance**: Sequential: objectives must complete in order. Parallel: all must complete, any order. Branching: one of N paths completes. Optional: not required for completion.
  - **Deps**: T-GP-9.2
  - **Effort**: Medium

- [ ] **T-GP-9.4** — Implement quest rewards (items, currency, XP, unlocks)
  - **Acceptance**: Reward list attached to quest. On COMPLETE → TURNED_IN transition, rewards granted. Item rewards go to inventory. Currency added. XP applied. Unlocks (abilities, quests, areas) activated.
  - **Deps**: T-GP-9.1, T-GP-8.1 (inventory for item rewards)
  - **Effort**: Medium

- [ ] **T-GP-9.5** — Wire @quest decorator and Foundation EventLog integration
  - **Acceptance**: `@quest(id, prerequisites, rewards)` decorator registers quest. EventLog logs: quest state changes, objective progress, reward grants.
  - **Deps**: T-GP-9.1, T-GP-9.4
  - **Effort**: Low

### Dialogue

- [ ] **T-GP-9.6** — Implement dialogue graph and node types (Text, Choice, Branch, Event, Random)
  - **Acceptance**: Graph of nodes. Text: NPC line + optional response. Choice: player options with conditions. Branch: condition-based flow control. Event: triggers game action. Random: randomized variation. Graph traversal from root.
  - **Deps**: T-GP-1.1
  - **Effort**: High

- [ ] **T-GP-9.7** — Implement dialogue variables (local, global, quest-linked, world state)
  - **Acceptance**: Local: scoped to conversation. Global: persistent across saves. Quest-linked: read/write quest state. World state: query boolean facts. Variables used in branch conditions and text substitution.
  - **Deps**: T-GP-9.6
  - **Effort**: Medium

- [ ] **T-GP-9.8** — Implement dialogue presentation (text box, portrait, choice buttons)
  - **Acceptance**: Text box with typewriter effect. Portrait display (configurable per NPC/line). Choice buttons for player responses. Skip/delay controls. Voice sync integration point.
  - **Deps**: T-GP-9.6
  - **Effort**: Medium

### Quest/Dialogue Integration

- [ ] **T-GP-9.9** — Implement QuestTracker component (active quests, progress, journal)
  - **Acceptance**: QuestTracker on player entity. Active quest list. Per-quest progress (objectives, counters, flags). Completed/failed quest history. Journal: text log of quest events.
  - **Deps**: T-GP-9.1
  - **Effort**: Medium

- [ ] **T-GP-9.10** — Implement quest→dialogue integration
  - **Acceptance**: Dialogue nodes can: accept quest, turn in quest, update quest state. Quest dialogue conditions: check player's quest state to show/hide dialogue options.
  - **Deps**: T-GP-9.5, T-GP-9.6
  - **Effort**: Medium

---

## Phase 10: Combat & Game Modes

### Combat System

- [ ] **T-GP-10.1** — Implement Health component (current, max_hp, regen, invulnerability)
  - **Acceptance**: Health component: current (RangeDescriptor 0-max), max_hp, regen_rate (per-second), is_invulnerable flag. TrackedDescriptor on current for change tracking.
  - **Deps**: T-GP-1.1
  - **Effort**: Low

- [ ] **T-GP-10.2** — Implement DamageSystem (pending damage → calculation → health reduction)
  - **Acceptance**: DamageSystem reads PendingDamage component each tick. Formula: damage = base × modifiers × (1 - resistance) × vulnerability. Applies to Health.current. Removes PendingDamage after processing.
  - **Deps**: T-GP-10.1
  - **Effort**: Medium

- [ ] **T-GP-10.3** — Implement DeathSystem (health ≤ 0 detection, cleanup triggers)
  - **Acceptance**: Each tick, query entities with health ≤ 0. Emit EntityDied event. Trigger death cleanup (remove components, play death animation/fx, mark for respawn).
  - **Deps**: T-GP-10.2
  - **Effort**: Medium

- [ ] **T-GP-10.4** — Implement team and faction system (IFF, friendly fire)
  - **Acceptance**: Team component: team_id, faction string. IFF queries: is_enemy, is_ally, is_neutral. Configurable friendly fire per game mode. Team change at runtime.
  - **Deps**: T-GP-1.1
  - **Effort**: Low

- [ ] **T-GP-10.5** — Implement damage modifiers (resistance, vulnerability, damage type)
  - **Acceptance**: Damage resistance (flat + percentage). Damage vulnerability (multiplier for specific types). Damage type enum: physical, fire, ice, lightning, poison, holy, dark, etc.
  - **Deps**: T-GP-10.2
  - **Effort**: Medium

- [ ] **T-GP-10.6** — Register combat systems in execution order
  - **Acceptance**: DamageSystem (order 7), DeathSystem (order 8), CleanupSystem (order 9) registered in correct execution order relative to other gameplay systems.
  - **Deps**: T-GP-10.2, T-GP-10.3
  - **Effort**: Low

### Scoring

- [ ] **T-GP-10.7** — Implement ScoreSystem (kills, objectives, assists, bonuses)
  - **Acceptance**: ScoreSystem listens to scoring events (Kill, Death, Objective, Assist, Bonus). Per-entity and per-team scores tracked. Score multipliers for streaks, objectives.
  - **Deps**: T-GP-10.3
  - **Effort**: Medium

### Game Modes

- [ ] **T-GP-10.8** — Implement GameModeBase (match lifecycle, rules, spawning)
  - **Acceptance**: GameModeBase class: match lifecycle (Lobby → Countdown → Playing → Match End → Results). Spawn logic (player spawn points, respawn timer). Rule hooks: can_player_respawn, is_match_over, get_winner.
  - **Deps**: T-GP-1.1, T-GP-10.4
  - **Effort**: High

- [ ] **T-GP-10.9** — Implement Deathmatch and Team Deathmatch modes
  - **Acceptance**: Deathmatch: free-for-all, kill = 1 point, first to score limit wins. Team Deathmatch: team scores, most kills wins. Respawn on death. Configurable time limit + score limit.
  - **Deps**: T-GP-10.8, T-GP-10.7
  - **Effort**: Medium

- [ ] **T-GP-10.10** — Implement Capture the Flag mode
  - **Acceptance**: CTF: each team has flag. Capture enemy flag at own base. Flag dropped on carrier death. Return after timeout. First to capture limit wins.
  - **Deps**: T-GP-10.8
  - **Effort**: High

- [ ] **T-GP-10.11** — Implement King of the Hill mode
  - **Acceptance**: KOTH: hill zone on map. Team/player in hill earns points. Contest: both teams in hill = no points. Hill rotates on timer. First to score limit wins.
  - **Deps**: T-GP-10.8
  - **Effort**: Medium

- [ ] **T-GP-10.12** — Implement Battle Royale mode
  - **Acceptance**: BR: large player count, spawn with no gear, loot weapons, shrinking safe zone, last alive wins. Circle logic, loot distribution, final showdown.
  - **Deps**: T-GP-10.8
  - **Effort**: High

---

## Summary

| Phase | Description | Task Count | Key Dependencies |
|-------|-------------|-----------|-----------------|
| 1 | Entity Framework & Core | 12 | Foundation systems |
| 2 | Input & Camera | 14 | Phase 1 |
| 3 | AI Tier 1 (BT + Utility + Perception) | 23 | Phase 1, Phase 2 |
| 4 | AI Tier 2 (GOAP + HTN) | 7 | Phase 3 |
| 5 | Navigation (8 methods) | 19 | Phase 3 (for AI movement) |
| 6 | AI Tier 3 (MCTS) | 8 | Phase 4, Phase 5 |
| 7 | Ability System (GAS-style) | 11 | Phase 1, Phase 2 |
| 8 | Inventory & Economy | 14 | Phase 1, Phase 7 |
| 9 | Quests & Dialogue | 10 | Phase 1, Phase 8 |
| 10 | Combat & Game Modes | 12 | Phase 1, Phase 2, Phase 7, Phase 8 |
| **Total** | | **130** | |
