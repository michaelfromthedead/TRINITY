# SIMULATION_CONTEXT.md — Simulation & Physics Layer

> **Purpose**: Complete implementation reference for the engine/simulation/ layer.  
> Read this file and ONLY this file when implementing simulation and physics systems.

**Implementation Status:**
| Layer | Status | Notes |
|-------|--------|-------|
| Python | ✅ COMPLETE | GJK/EPA, XPBD, SPH fluids, cloth, vehicles (~49K lines) |
| Rust | ⚠️ 65% | GAPSET_19_PHYSICS — Core solvers wired |
| Wired | ✅ Partial | Rigid body and collision mostly functional |

*See `docs/STATUS.md` for current progress. See `docs/gap_sets/GAPSET_19_PHYSICS/` for tasks.*

---

## 1. Architecture Summary

The simulation layer provides all physics and physical simulation systems for the engine. It sits between gameplay (above) and core systems (below), and feeds transform data into animation and rendering. It is deterministic, substep-capable, and scales from simple rigid bodies to full destruction, cloth, hair, soft bodies, fluids, and vehicles.

**Core Subsystems (10):**
1. **Rigid Body Dynamics** — Body types (static/kinematic/dynamic), forces, integration, sleeping, islands
2. **Collision Detection** — Broadphase (SAP/BVH/grid/octree), narrowphase (GJK/EPA/SAT/MPR), contact manifolds, CCD
3. **Constraint Solver** — Sequential Impulse, PGS, TGS, XPBD; joints (fixed/hinge/slider/ball/spring/distance/D6), motors, limits
4. **Destruction** — Voronoi/radial/slice/custom fracture, damage types, damage resistance, support structures, debris
5. **Cloth** — Particles/edges/triangles, structural/shear/bending constraints, PBD/XPBD/mass-spring/FEM solvers, wind, self-collision
6. **Hair & Fur** — Guide/interpolated/clump strands, Follow-The-Leader/DER/PBD, LOD, GPU simulation
7. **Soft Body & Deformables** — FEM, corotational FEM, shape matching, PBD/XPBD, muscle simulation
8. **Fluid** — SPH, PBF, FLIP/PIC/APIC, Eulerian grid, shallow water, surface reconstruction
9. **Vehicles** — Wheeled (Pacejka tire model), tracked, hover, aircraft, watercraft; suspension, drivetrain
10. **Character Physics** — Kinematic/dynamic/hybrid controllers, ground detection, ragdoll, active ragdoll

**Simulation Domains (from @simulation_domain):**
| Domain | Description |
|--------|-------------|
| `rigid_body` | Rigid body dynamics (default physics) |
| `soft_body` | Deformable volumes (FEM, shape matching) |
| `cloth` | Surface-based particle simulation |
| `fluid` | Particle or grid-based fluid |
| `vehicle` | Specialized vehicle dynamics |

**Solver Types (from @solver_hint):**
| Solver | Description |
|--------|-------------|
| `pgs` | Projected Gauss-Seidel (default, general purpose) |
| `tgs` | Temporal Gauss-Seidel (improved convergence) |
| `xpbd` | Extended Position Based Dynamics (compliance-based) |

**Simulation Phase in Engine Loop:**
```
SystemPhase.PRE_PHYSICS (0) — Input processing, force accumulation, CCD setup
SystemPhase.PHYSICS (1)     — Broadphase, narrowphase, solve, integrate
SystemPhase.POST_PHYSICS (2) — Contact callbacks, destruction, event dispatch
```

**Dependency Chain:**
```
Platform (RHI/Window/Threading)
    -> Core (Memory/Math/ECS/Task)
        -> SIMULATION <- THIS LAYER
            -> Animation (skeletal, ragdoll blend)
                -> Rendering (transform data, debug draw)
```

**Determinism Requirement:** All simulation state uses Fixed16 (Q8.8) or Fixed32 (Q16.16) for reproducibility. See `engine/determinism/DETERMINISM_CONTEXT.md` for fixed-point math details, snapshot system, checksums, and replay architecture.

---

## 2. Decorators

### 2.1 Physics Simulation Decorators (Tier 46 — physics_sim.py)

#### @simulation_domain
```python
@simulation_domain(
    domain: str,  # REQUIRED: "rigid_body" | "soft_body" | "cloth" | "fluid" | "vehicle"
)
```
- **Steps:** TAG(simulation_domain=True), TAG(physics_domain=domain), REGISTER(physics_sim)
- **After-Apply:** `_simulation_domain=True`, `_physics_domain=domain`
- **Validation:** `domain` must be in `VALID_DOMAINS`

#### @substep
```python
@substep(
    min_hz: int = 60,        # > 0, minimum substep rate
    max_hz: int = 240,       # >= min_hz, maximum substep rate
    max_substeps: int = 4,   # > 0, cap on substeps per frame
)
```
- **Steps:** TAG(substep=True), TAG(substep_min_hz), TAG(substep_max_hz), TAG(substep_max_substeps), REGISTER(physics_sim)
- **After-Apply:** `_substep=True`, `_substep_min_hz`, `_substep_max_hz`, `_substep_max_substeps`
- **Validation:** `min_hz > 0`, `max_hz >= min_hz`, `max_substeps > 0`

#### @solver_hint
```python
@solver_hint(
    type: str = "pgs",            # "pgs" | "tgs" | "xpbd"
    iterations: int = 4,          # > 0
    warm_starting: bool = True,
)
```
- **Steps:** TAG(solver_hint=True), TAG(solver_type), TAG(solver_iterations), TAG(solver_warm_starting), REGISTER(physics_sim)
- **After-Apply:** `_solver_hint=True`, `_solver_type`, `_solver_iterations`, `_solver_warm_starting`
- **Validation:** `type` in `VALID_SOLVER_TYPES`, `iterations > 0`

#### @sleep_threshold
```python
@sleep_threshold(
    linear: float = 0.1,    # >= 0, linear velocity threshold
    angular: float = 0.05,  # >= 0, angular velocity threshold
    time: float = 0.5,      # >= 0, seconds below threshold before sleep
)
```
- **Steps:** TAG(sleep_threshold=True), TAG(sleep_linear), TAG(sleep_angular), TAG(sleep_time), REGISTER(physics_sim)
- **After-Apply:** `_sleep_threshold=True`, `_sleep_linear`, `_sleep_angular`, `_sleep_time`
- **Validation:** all values `>= 0`

#### @continuous_collision
```python
@continuous_collision(
    mode: str = "none",  # "none" | "speculative" | "sweep"
)
```
- **Steps:** TAG(continuous_collision=True), TAG(ccd_mode=mode), REGISTER(physics_sim)
- **After-Apply:** `_continuous_collision=True`, `_ccd_mode=mode`
- **Validation:** `mode` in `VALID_CCD_MODES`

#### @buoyancy
```python
@buoyancy(
    density: float = 1.0,        # > 0, object density relative to water
    drag: float = 0.5,           # >= 0, linear drag in water
    angular_drag: float = 0.1,   # >= 0, angular drag in water
)
```
- **Steps:** TAG(buoyancy=True), TAG(buoyancy_density), TAG(buoyancy_drag), TAG(buoyancy_angular_drag), REGISTER(physics_sim)
- **After-Apply:** `_buoyancy=True`, `_buoyancy_density`, `_buoyancy_drag`, `_buoyancy_angular_drag`
- **Validation:** `density > 0`, `drag >= 0`, `angular_drag >= 0`

#### @wind_affected
```python
@wind_affected(
    drag_coefficient: float = 1.0,   # > 0
    area: float | str = "auto",      # > 0 or "auto" (computed from mesh)
)
```
- **Steps:** TAG(wind_affected=True), TAG(wind_drag_coefficient), TAG(wind_area), REGISTER(physics_sim)
- **After-Apply:** `_wind_affected=True`, `_wind_drag_coefficient`, `_wind_area`
- **Validation:** `drag_coefficient > 0`; if `area` is float, must be `> 0`; if str, must be `"auto"`

### 2.2 Destruction Decorators (Tier 43 — destruction.py)

#### @destructible
```python
@destructible(
    health: float = 100.0,         # > 0, total hit points
    fracture_depth: int = 2,       # >= 0, levels of recursive fracture
    debris_lifetime: float = 10.0, # >= 0, seconds before debris cleanup
)
```
- **Config:** `DestructibleConfig(health, fracture_depth, debris_lifetime)`
- **Steps:** TAG(destructible=True), TAG(destructible_config=DestructibleConfig(...)), REGISTER(destruction)
- **After-Apply:** `_destructible=True`, `_destructible_health`, `_destructible_fracture_depth`, `_destructible_debris_lifetime`, `_destructible_config`
- **Validation:** `health > 0`, `fracture_depth >= 0`, `debris_lifetime >= 0`

#### @damage_type
```python
@damage_type(
    id: str,                        # REQUIRED, non-empty damage type identifier
    base_multiplier: float = 1.0,   # > 0
)
```
- **Config:** `DamageTypeConfig(id, base_multiplier)`
- **Steps:** TAG(damage_type=True), TAG(damage_type_config=DamageTypeConfig(...)), REGISTER(destruction)
- **After-Apply:** `_damage_type=True`, `_damage_type_id`, `_damage_type_multiplier`, `_damage_type_config`
- **Validation:** `id` must be non-empty string, `base_multiplier > 0`

#### @damage_resistance
```python
@damage_resistance(
    resistances: dict[str, float],  # REQUIRED, non-empty; damage_type_id -> multiplier
)
```
- **Config:** `DamageResistanceConfig(resistances)`
- **Steps:** TAG(damage_resistance=True), TAG(damage_resistance_config=DamageResistanceConfig(...)), REGISTER(destruction)
- **After-Apply:** `_damage_resistance=True`, `_damage_resistance_values`, `_damage_resistance_config`
- **Validation:** `resistances` must be non-empty dict

#### @fracture
```python
@fracture(
    pattern: str = "voronoi",          # "voronoi" | "radial" | "slice" | "custom"
    min_size: float = 0.1,             # > 0, minimum fragment size
    interior_material: str | None = None,
)
```
- **Config:** `FractureConfig(pattern, min_size, interior_material)`
- **Steps:** TAG(fracture=True), TAG(fracture_config=FractureConfig(...)), REGISTER(destruction)
- **After-Apply:** `_fracture=True`, `_fracture_pattern`, `_fracture_min_size`, `_fracture_interior_material`, `_fracture_config`
- **Validation:** `pattern` in `VALID_FRACTURE_PATTERN`, `min_size > 0`

#### @physics_material
```python
@physics_material(
    friction: float = 0.5,      # >= 0
    restitution: float = 0.3,   # >= 0 (0 = inelastic, 1 = elastic)
    density: float = 1.0,       # >= 0
)
```
- **Config:** `PhysicsMaterialConfig(friction, restitution, density)`
- **Steps:** TAG(physics_material=True), TAG(physics_material_config=PhysicsMaterialConfig(...)), REGISTER(destruction)
- **After-Apply:** `_physics_material=True`, `_physics_friction`, `_physics_restitution`, `_physics_density`, `_physics_material_config`
- **Validation:** all values `>= 0`

#### @joint
```python
@joint(
    type: str = "fixed",                    # "fixed" | "hinge" | "slider" | "ball" | "spring"
    break_force: float | None = None,       # > 0 or None (unbreakable)
    break_torque: float | None = None,      # > 0 or None (unbreakable)
)
```
- **Config:** `JointConfig(type, break_force, break_torque)`
- **Steps:** TAG(joint=True), TAG(joint_config=JointConfig(...)), REGISTER(destruction)
- **After-Apply:** `_joint=True`, `_joint_type`, `_joint_break_force`, `_joint_break_torque`, `_joint_config`
- **Validation:** `type` in `VALID_JOINT_TYPE`

### 2.3 Spatial Decorators (Tier — spatial.py)

#### @spatial
```python
@spatial(
    structure: str,          # REQUIRED: "grid" | "quadtree" | "octree" | "bvh" | "hash"
    cell_size: float = 1.0,  # > 0
)
```
- **Steps:** TAG(spatial=True), TAG(spatial_structure), TAG(spatial_cell_size), REGISTER(spatial)
- **After-Apply:** `_spatial=True`, `_spatial_structure`, `_spatial_cell_size`
- **Validation:** `structure` in `VALID_SPATIAL_STRUCTURES`, `cell_size > 0`

#### @partitioned
```python
@partitioned(
    dimensions: int = 2,         # 2 | 3
    max_entities: int = 1000,    # > 0
)
```
- **Steps:** TAG(partitioned=True), TAG(partition_dimensions), TAG(partition_max_entities), REGISTER(spatial)
- **After-Apply:** `_partitioned=True`, `_partition_dimensions`, `_partition_max_entities`
- **Validation:** `dimensions` in `{2, 3}`, `max_entities > 0`

### 2.4 Supporting Decorators (from other modules)

| Decorator | Module | Simulation Role |
|-----------|--------|----------------|
| `@system(phase="physics")` | ecs_core | Register physics systems in PHYSICS phase |
| `@system(phase="pre_physics")` | ecs_core | Force accumulation, CCD prep |
| `@system(phase="post_physics")` | ecs_core | Contact callbacks, destruction events |
| `@component` | ecs_core | RigidBody, Collider, Joint, Cloth, Vehicle components |
| `@resource` | ecs_core | PhysicsWorld, CollisionDispatcher as singleton Resources |
| `@query` | ecs_core | Query entities with physics components |
| `@fixed(hz=60)` | scheduling | Fixed timestep for deterministic physics |
| `@parallel(chunk_size=64)` | scheduling | Parallel broadphase, island solving |
| `@exclusive` | scheduling | Constraint solver (requires write access) |
| `@after(systems)` | scheduling | Run after broadphase |
| `@before(systems)` | scheduling | Run before integration |
| `@pooled(initial_size=1024)` | memory | Pooled contact manifolds, collision pairs |
| `@packed(layout="soa")` | memory | SoA for SIMD-friendly physics data |
| `@aligned(bytes=64)` | memory | Cache-line alignment for physics arrays |
| `@arena(name="physics")` | memory | Arena allocator for per-frame physics temp data |
| `@budget(category="physics")` | memory | Physics memory budgeting |
| `@atomic` | memory | Thread-safe counters for parallel physics |
| `@serializable(format="binary")` | data_flow | Physics state persistence |
| `@networked(authority="server")` | data_flow | Network-replicated physics |
| `@snapshot(history_frames=60)` | data_flow | Snapshot for rollback networking |
| `@versioned` | data_flow | Schema migration for physics components |
| `@on_add(component)` | lifecycle | Initialize physics body on component add |
| `@on_remove(component)` | lifecycle | Remove from physics world on component remove |
| `@on_change(component)` | lifecycle | Re-sync physics params on component change |
| `@profile(name="Physics")` | dev | CPU timing for physics systems |
| `@trace` | dev | EventLog physics operations |
| `@invariant(check=fn)` | dev | Validate physics constraints at runtime |
| `@bench(iterations=1000)` | dev | Benchmark solver performance |
| `@server_authoritative` | security | Physics state is server-authoritative |
| `@validated(rules=[...])` | security | Validate physics inputs from clients |
| `@rate_limited(max_per_second=60)` | security | Rate-limit physics commands |

### 2.5 Determinism Decorators (from time.py)

| Decorator | Signature | Role |
|-----------|-----------|------|
| `@deterministic` | no params | Mark system as deterministic (no floats, no side effects) |
| `@time_scale(scale=1.0)` | `scale: float` | Apply time scaling to system |
| `@pausable` | no params | Allow system pause/resume |
| `@rewindable(max_frames=300)` | `max_frames: int` | Enable frame rewind for debugging |
| `@fixed(hz=60)` | `hz: int` | Fixed timestep (critical for determinism) |

See `engine/determinism/DETERMINISM_CONTEXT.md` for full determinism decorator details.

---

## 3. Metaclasses

### ComponentMeta (PRIMARY for physics components)

```python
class ComponentMeta(EngineMeta):
    _registry: dict[int, type]
    _name_to_id: dict[str, int]
    _next_id: int
    _lock: threading.Lock
```

**`__new__(mcs, name, bases, namespace, **kwargs)` — 7 step groups:**
1. **Generate unique ID:** `_component_id`, `_component_name`; TAG(component_id), TAG(component_name)
2. **Process fields:** `_process_fields(cls)` — extracts `Annotated[]` types, unwraps base type, extracts descriptors from metadata, stores `_field_types`, `_field_offsets`, `_field_defaults`, `_field_descriptors`; DESCRIBE(field, type) per field
3. **Install descriptors:** `_install_descriptors(cls)` — builds descriptor chain from markers: StorageDescriptor (innermost) -> ValidatedDescriptor -> TrackedDescriptor -> NetworkedDescriptor (outermost); auto-installs TrackedDescriptor from TRACK steps, ValidatedDescriptor from VALIDATE steps; INTERCEPT(field, descriptor) per field
4. **Validate component:** `_validate_component(cls)` — warns on non-data methods; VALIDATE(component_rules)
5. **Register:** stores in `_registry`, `_name_to_id`; REGISTER(component_registry)
6. **Foundation integration:** `_register_with_foundation(cls)` — registers with Foundation Registry; REGISTER(foundation)
7. **Initialize pool/budget:** if `_pooled_config` set, creates `_pool`; TAG(pooled), HOOK(on_create). If `_budget_config` set, creates `_instance_count`; TAG(budgeted), VALIDATE(budget_limit)

**Key Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `get_by_id` | `(component_id: int) -> type | None` | Lookup by ID |
| `get_by_name` | `(name: str) -> type | None` | Lookup by qualified name |
| `all_components` | `() -> list[type]` | All registered components |
| `component_count` | `() -> int` | Count of registered components |
| `clear_registry` | `() -> None` | Clear (for testing) |
| `__call__` | `(cls, *args, **kwargs) -> Any` | Pool + budget enforcement on instantiation |
| `get_layout_mode` | `(cls) -> str` | "soa" or "aos" |
| `get_layout_arrays` | `(cls, instances) -> dict[str, list]` | SoA extraction |
| `return_to_pool` | `(cls, instance) -> None` | Return to pool, decrement budget |
| `pool_stats` | `(cls) -> dict` | `{enabled, available, max_size, config}` |
| `instance_count` | `(cls) -> int` | Live instance count (budget) |

**Simulation usage:** RigidBody, Collider (Box/Sphere/Capsule/Mesh/ConvexHull), PhysicsMaterial, Joint, ClothParticle, FluidParticle, VehicleConfig, CharacterController components.

### SystemMeta (for physics systems)

```python
class SystemMeta(EngineMeta):
    _registry: dict[int, type]
    _phases: dict[SystemPhase, list[int]]
    _next_id: int
    _lock: threading.Lock
```

**`__new__` — 5 step groups:**
1. **Generate ID:** `_system_id`, `_system_name`; TAG(system_id), TAG(system_name)
2. **Set defaults:** `_system_phase` (default UPDATE), `_reads`, `_writes`, `_resources`, `_exclusive`, `_priority`; TAG per attribute
3. **Validate declarations:** checks `_reads`/`_writes` reference component types; VALIDATE(system_declarations)
4. **Analyze dependencies:** builds `_dependencies` (write-read conflicts), `_can_parallelize`; DESCRIBE(dependencies, can_parallelize)
5. **Register:** stores in `_registry`, `_phases[phase]`; REGISTER(system_registry)

**Key Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `get_by_id` | `(system_id: int) -> type | None` | Lookup by ID |
| `get_by_name` | `(name: str) -> type | None` | Lookup by qualified name |
| `all_systems` | `() -> list[type]` | All registered systems |
| `get_phase_systems` | `(phase: SystemPhase) -> list[type]` | Systems in phase |
| `get_phase_order` | `(phase: SystemPhase) -> list[type]` | Topologically sorted (Kahn's algorithm) |
| `get_parallel_groups` | `(phase: SystemPhase) -> list[list[type]]` | Non-conflicting groups for parallel execution |
| `hot_reload` | `(old_cls, new_cls) -> type` | Replace system, re-analyze dependencies |
| `reload_system` | `(name: str) -> type | None` | Re-validate and refresh dependencies |
| `clear_registry` | `() -> None` | Clear (for testing) |

**Simulation usage:** All `@system(phase="physics")`, `@system(phase="pre_physics")`, and `@system(phase="post_physics")` systems.
- `get_phase_order(SystemPhase.PHYSICS)` returns topologically sorted physics systems
- `get_parallel_groups(SystemPhase.PHYSICS)` returns parallelizable physics system groups

### EventMeta (for physics events)

```python
class EventMeta(EngineMeta):
    _registry: dict[int, type]
    _name_to_id: dict[str, int]
    _next_id: int
    _lock: threading.Lock
    _event_pools: dict[type, list]
    _event_pool_max_size: int  # = 64
```

**`__new__` — 6 step groups:**
1. **Generate ID:** `_event_id`, `_event_name`
2. **Collect fields:** `_event_fields` from annotations
3. **Track inheritance:** `_event_parent_ids` (transitive)
4. **Set defaults:** `_event_priority`, `_event_channels`, `_event_pooled`
5. **Validate:** events must be data-only (no methods)
6. **Register:** stores in `_registry`, `_name_to_id`

**Key Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `get_by_id` | `(event_id: int) -> type | None` | Lookup by ID |
| `get_by_name` | `(name: str) -> type | None` | Lookup by qualified name |
| `all_events` | `() -> list[type]` | All events |
| `is_subtype` | `(event_id, parent_id) -> bool` | Inheritance check |
| `get_subtypes` | `(parent_id) -> list[type]` | All subtypes |
| `get_by_channel` | `(channel: str) -> list[type]` | Events on channel |
| `acquire` | `(event_cls, **kwargs) -> Any` | Pool acquire |
| `release` | `(instance) -> None` | Pool release |
| `pool_stats` | `(event_cls) -> dict` | `{pooled, current_size, max_size}` |
| `serialize` | `(event_instance) -> dict` | Recursive serialization |
| `deserialize` | `(event_cls, data) -> Any` | Recursive deserialization |
| `clear_registry` | `() -> None` | Clear (for testing) |

**Simulation usage:** CollisionBeginEvent, CollisionEndEvent, CollisionPersistEvent, DestructionEvent, JointBreakEvent, DamageEvent.

### StateMeta (for physics state machines)

```python
class StateMeta(EngineMeta):
    _registry: dict[type, dict[str, type]]  # machine -> {name -> state}
    _global_registry: dict[int, type]
    _next_id: int
    _lock: threading.Lock
    _state_history: dict[type, list[str]]
```

**`__new__` — 4 step groups:**
1. **Generate ID:** `_state_id`, `_state_name`, `_state_qualified_name`
2. **Set defaults:** `_state_transitions`, `_state_on_enter`, `_state_on_exit`, `_state_machine_cls`, `_state_parent`, `_state_children`
3. **Register globally:** in `_global_registry`
4. **Register with machine:** if `_state_machine_cls` is set

**Key Methods:**
| Method | Signature | Description |
|--------|-----------|-------------|
| `get_by_id` | `(state_id: int) -> type | None` | Lookup by ID |
| `get_by_name` | `(machine_cls, name) -> type | None` | Lookup within machine |
| `all_states` | `() -> list[type]` | All states |
| `get_machine_states` | `(machine_cls) -> dict[str, type]` | States for machine |
| `can_transition` | `(from_state, to_state) -> bool` | Validate transition |
| `validate_transitions` | `(machine_cls) -> list[str]` | Validate all transitions |
| `register_with_machine` | `(state_cls, machine_cls) -> None` | Post-creation registration |
| `get_enter_hook` | `(state_cls) -> Callable | None` | Enter callback |
| `get_exit_hook` | `(state_cls) -> Callable | None` | Exit callback |
| `register_substate` | `(parent, child) -> None` | Hierarchical states |
| `get_substates` | `(state_cls) -> set[str]` | Child state names |
| `get_parent_state` | `(state_cls) -> str | None` | Parent state name |
| `is_active_in_hierarchy` | `(state_cls, current, machine_cls) -> bool` | Hierarchical check |
| `record_transition` | `(machine_cls, from_state, to_state, max_history=10) -> None` | Record history |
| `get_previous_state` | `(machine_cls) -> str | None` | Previous state |
| `get_history` | `(machine_cls, limit=10) -> list[str]` | Transition history |
| `clear_registry` | `() -> None` | Clear (for testing) |

**Simulation usage:** Vehicle gear states, character physics states (grounded/airborne/swimming/climbing), ragdoll activation states (animated/ragdoll/blending), destruction states (intact/damaged/fractured/debris).

---

## 4. Descriptors

### TrackedDescriptor (tracking.py) — Physics State Dirty Flags

```python
TrackedDescriptor(
    field_type: type = object,
    inner: BaseDescriptor | None = None,
    field_offset: int = 0,
    use_bitmask: bool = False,
)
```
- **descriptor_id:** `"tracked"`
- **accepts_inner:** `("storage", "validated", "range")`
- **accepts_outer:** `("networked", "observable", "cached")`
- **excludes:** `("computed",)`
- **post_set():** if value changed, marks dirty via set or bitmask; calls `_notify_foundation_tracker(obj, old, new)` -> `tracker.mark_dirty()`; calls `_notify_eventlog(obj, old, new)` -> `Change(entity, field, old, new)`
- **Simulation usage:** Every mutable physics property (velocity, force, transform, mass, inertia). Changes trigger dirty flag -> physics world sync.

### VersionedDescriptor (tracking.py) — Per-Field Version Counter

```python
VersionedDescriptor(
    field_type: type = object,
    inner: BaseDescriptor | None = None,
)
```
- **descriptor_id:** `"versioned"`
- **post_set():** increments `_version_{name}` counter
- **get_version(obj) -> int:** current version
- **Simulation usage:** Optimistic concurrency for networked physics state.

### DiffDescriptor (tracking.py) — Previous Value Storage

```python
DiffDescriptor(
    field_type: type = object,
    inner: BaseDescriptor | None = None,
    strategy: str = "shallow",  # "shallow" | "deep" | "structural" | "custom"
    custom_differ: Callable | None = None,
)
```
- **descriptor_id:** `"diff"`
- **post_set():** stores old value in `_prev_{name}`
- **get_previous(obj) -> T:** previous value
- **has_changed(obj) -> bool:** compare current vs previous using strategy
- **Simulation usage:** Delta compression for networked physics; detect velocity changes for sleeping.

### NetworkedDescriptor (networking.py) — Network Replication Queue

```python
NetworkedDescriptor(
    field_type: type = object,
    inner: BaseDescriptor | None = None,
    authority: str = "server",       # "server" | "client" | "owner"
    interpolated: bool = False,
    priority: int = 1,               # DEFAULT_NETWORK_PRIORITY
    update_frequency: int = 0,       # DEFAULT_UPDATE_FREQUENCY (0 = every change)
)
```
- **descriptor_id:** `"networked"`
- **accepts_inner:** `("tracked", "observable", "validated", "range", "storage")`
- **excludes:** `("transient", "local_only")`
- **post_set():** if changed, appends to `_network_queue` with field, value, old_value, priority
- **Simulation usage:** Replicated transforms, velocities, forces for networked physics.

### InterpolatedDescriptor (networking.py) — Smooth Network Updates

```python
InterpolatedDescriptor(
    field_type: type = object,
    inner: BaseDescriptor | None = None,
    mode: str = "linear",  # "linear" | "hermite"
)
```
- **descriptor_id:** `"interpolated"`
- **post_set():** buffers snapshots (ring buffer, size `INTERPOLATION_BUFFER_SIZE=3`)
- **get_interpolated(obj, t: float) -> T:** interpolate between last two snapshots; linear or hermite
- **Simulation usage:** Smooth rendering of networked physics bodies between server updates.

### PredictedDescriptor (networking.py) — Client-Side Prediction

```python
PredictedDescriptor(
    field_type: type = object,
    inner: BaseDescriptor | None = None,
    max_history: int = 30,  # DEFAULT_PREDICTION_HISTORY
)
```
- **descriptor_id:** `"predicted"`
- **post_set():** appends to history buffer (ring buffer, size `max_history`)
- **rollback(obj, frames=1) -> T:** roll back to previous state, truncate history
- **get_history(obj) -> list:** copy of history buffer
- **Simulation usage:** Client-predicted physics (player movement, projectiles). Rollback on server correction.

### ThrottledNetworkDescriptor (networking.py) — Rate-Limited Updates

```python
ThrottledNetworkDescriptor(
    field_type: type = object,
    inner: BaseDescriptor | None = None,
    max_updates_per_second: float = 20.0,  # DEFAULT_MAX_UPDATES_PER_SECOND
)
```
- **descriptor_id:** `"throttled_network"`
- **post_set():** token bucket rate limiting via `_min_interval = 1/max_updates_per_second`
- **has_pending(obj) -> bool:** check for throttled pending update
- **flush(obj) -> None:** force send
- **Simulation usage:** Rate-limit non-critical physics updates (debris velocity, cloth particles).

### ObservableDescriptor (observable.py) — Observer Notifications

```python
ObservableDescriptor(
    field_type: type = object,
    inner: BaseDescriptor | None = None,
)
```
- **descriptor_id:** `"observable"`
- **accepts_inner:** `("tracked", "storage", "validated", "range")`
- **post_set():** notifies `_observers[field_name]` callbacks: `(obj, field_name, old, new)`
- **Simulation usage:** Physics parameter changes trigger reconfiguration (e.g., mass change -> inertia tensor recompute).

### BoundDescriptor (observable.py) — Two-Way Binding

```python
BoundDescriptor(
    field_type: type = object,
    inner: BaseDescriptor | None = None,
    source: Any = None,
    getter: Callable = None,
    setter: Callable = None,
)
```
- **descriptor_id:** `"bound"`
- **Simulation usage:** Bind UI sliders to physics parameters for live tuning.

### BaseDescriptor (base.py) — Base Class

```python
BaseDescriptor(
    field_type: type = object,
    inner: BaseDescriptor | None = None,
    **config: Any,
)
```
- **__slots__:** `("_name", "_field_type", "_inner", "_owner", "_config")`
- **Lifecycle hooks:** `pre_get(obj)`, `post_get(obj, value) -> T`, `pre_set(obj, value) -> T`, `post_set(obj, value, old_value)`
- **Chain introspection:** `get_chain() -> list[BaseDescriptor]` (outermost first)
- **Read tracking:** integrates with `_current_computation` ContextVar for reactive computation
- **Provenance:** integrates with `foundation.provenance` for read recording

### Descriptor Chain for Simulation Fields

```
StorageDescriptor           <- innermost (raw obj.__dict__ storage)
  -> ValidatedDescriptor    <- bounds checking (mass > 0, velocity clamped)
    -> TrackedDescriptor    <- dirty flags -> Foundation Tracker -> physics world sync
      -> NetworkedDescriptor <- network queue -> replication
        -> PredictedDescriptor <- prediction history + rollback (outermost)
```

---

## 5. Foundation Integration Points

### Foundation Registry -> Physics Discovery
```
registry.subclasses(Collider)          # Discover all collider types
registry.instances(PhysicsWorld)       # Get active physics world
registry.get_all("physics_sim")        # All @simulation_domain components
ComponentMeta.all_components()         # All registered components
SystemMeta.get_phase_systems(PHYSICS)  # All physics-phase systems
```

### Foundation Tracker -> Physics Dirty Flags
```
rigidbody.velocity = Vec3(1, 0, 0)
    -> TrackedDescriptor.post_set()
    -> Tracker.mark_dirty(rigidbody, "velocity")
    -> Per-frame: Tracker.all_dirty()
    -> Physics world collects dirty bodies
    -> Re-syncs velocity with internal solver state
```

### Foundation EventLog -> Physics Operation Audit
```
@system(phase="physics")
@trace
class SolverSystem(System):
    def execute(self, dt):
        # EventLog.record("SolverSystem.execute", tick=N)
        solver.solve(constraints, iterations=4)
        # EventLog.record("solve_complete", contacts=len(contacts))
```

### Foundation Mirror -> Physics Introspection
```python
m = mirror(rigidbody)
m.get("mass")                    # Read mass
m.get("velocity")                # Read velocity
m.set("linear_damping", 0.5)    # Live edit
m.to_dict()                      # Export all physics state

m = mirror(physics_world)
m.get("body_count")              # Active body count
m.get("contact_count")           # Active contacts
```

### Foundation Bridge -> ShellLang Physics Debugging
```
QUERY RigidBody WHERE body_type == "dynamic" AND sleeping == False
QUERY Collider WHERE shape == "sphere" AND radius > 5.0
MUTATE entity=42 COMPONENT=RigidBody SET mass=10.0
QUERY Joint WHERE break_force IS NOT None AND stress > 0.8
```

### Foundation ContentStore -> Physics Snapshot Deduplication
```
content_store.store(world_snapshot)    # Hash-based, structural sharing
snapshot.save()                        # Accordion strategy (dense recent, sparse old)
snapshot.restore(frame_id)             # Rollback to snapshot
```

### Foundation DeltaSync -> Efficient Physics State Transfer
```
delta = DeltaSync.diff(old_state, new_state)   # Minimal diff
DeltaSync.apply(target_state, delta)            # Apply correction
# Used for: rollback networking, replay correction, desync repair
```

---

## 6. Architecture Spec Details

### 6.1 Physics World

**World Structure:**
| Component | Description |
|-----------|-------------|
| Scene | Container for physics bodies |
| Gravity | Global force (default: -9.81 m/s^2) |
| Timestep | Fixed dt for determinism (1/60 = 16.67ms) |
| Substeps | Accuracy vs speed (configurable via @substep) |

**Body Types:**
| Type | Description | Mass | Forces |
|------|-------------|------|--------|
| Static | Immovable world geometry | Infinite | No |
| Kinematic | Scripted motion (platforms, doors) | Infinite | No (user-controlled) |
| Dynamic | Fully simulated | Finite | Yes |

**Body Properties:** Mass, Inertia Tensor, Center of Mass, Linear Damping, Angular Damping, Use Gravity, Enable CCD, Lock Axes.

**Body State:** Transform (position + orientation), Velocity (linear + angular), Accumulated Forces, Sleep State.

**Collision Shapes:**
| Category | Shapes |
|----------|--------|
| Primitive | Sphere, Box, Capsule, Cylinder, Cone, Plane |
| Complex | Convex Hull, Triangle Mesh (static only), Heightfield, Compound |
| Implicit | Signed Distance Field, Custom support function |

**Shape Properties:** Local transform offset, Physics material, Collision filter (layers + mask), Trigger volume flag.

### 6.2 Collision Detection

**Broadphase Algorithms:**
| Algorithm | Complexity | Best For |
|-----------|-----------|----------|
| SAP (Sweep and Prune) | O(n log n) update | Mostly-static scenes |
| Dynamic BVH | O(n log n) build, O(log n) query | General purpose |
| Uniform Grid | O(1) per cell | Dense, uniform distributions |
| Spatial Hash | O(1) amortized | Variable density |
| Octree | O(log n) query | Large worlds, sparse objects |

**Narrowphase Algorithms:**
| Algorithm | Use Case |
|-----------|----------|
| GJK | General convex-convex distance/intersection |
| EPA | Penetration depth extraction (post-GJK) |
| SAT | Box-box (15-axis), convex-convex |
| MPR | Alternative to GJK+EPA |
| Sphere-Sphere | Direct distance check |
| Capsule-Capsule | Segment distance |

**Contact Manifold:** Up to 4 contact points (persistent across frames for stability). Each point: position, normal, penetration depth, friction impulse cache.

**CCD Methods (from @continuous_collision):**
| Mode | Description | Use Case |
|------|-------------|----------|
| `none` | No CCD (default) | Slow objects |
| `speculative` | Expanded AABB, speculative contacts | Medium-speed objects |
| `sweep` | Full time-of-impact sweep | Fast/thin objects (bullets) |

**Collision Queries:** Raycast (line vs world), Shape Cast (swept volume), Overlap (shape vs world), Closest Point. Options: any-hit, closest-hit, all-hits, filter callback. Batch: multi-ray, async.

**Collision Events:** Begin Overlap, Persist, End Overlap, Hit Event (includes impulse data).

### 6.3 Constraint Solver

**Solver Pipeline:**
```
Prepare (build Jacobians) -> Velocity Solve (iterative impulses) -> Position Solve (fix penetration)
    Warm Starting: use previous frame impulses as initial guess
    Relaxation: SOR factor for convergence
```

**Solver Types (from @solver_hint):**
| Type | Description | Use Case |
|------|-------------|----------|
| Sequential Impulse / PGS | Iterative impulses, Projected Gauss-Seidel | General rigid body |
| TGS | Temporal Gauss-Seidel, better convergence | Stacking, ragdolls |
| XPBD | Position-based with compliance | Cloth, soft bodies, ropes |

**Joint Types (from @joint):**
| Joint | DOF Removed | Description |
|-------|-------------|-------------|
| `fixed` | All 6 | No relative motion |
| `hinge` | 5 (1 rotation free) | Door hinge, wheel axle |
| `slider` | 5 (1 translation free) | Piston, sliding door |
| `ball` | 3 (3 rotations free) | Shoulder, ragdoll |
| `spring` | Soft | Spring-damper connection |
| Distance | Maintains separation | Rope, chain |
| D6/Configurable | Per-axis limits | General purpose |

**Joint Features:** Linear/angular limits (soft/hard), velocity/position motors, max force/torque, breakable (via @joint break_force/break_torque).

**Contact Constraints:** Normal (non-penetration), Friction (tangent, Coulomb model), Rolling friction, Spinning friction.

### 6.4 Sleeping System (from @sleep_threshold)

**Sleep Criteria:** Linear velocity below threshold, Angular velocity below threshold, Time below threshold exceeded.

**Wake Conditions:** Force applied, Collision with active body, Joint partner wakes, Explicit wake call.

**Island Sleeping:** Connected component detection -> entire island sleeps/wakes together. Parallel island solving for independent groups.

### 6.5 Destruction System (from destruction.py decorators)

**Destruction Pipeline:**
```
Damage Accumulation -> Support Evaluation -> Fracture -> Debris Spawn -> Cleanup
```

**Fracture Patterns (from @fracture):**
| Pattern | Description |
|---------|-------------|
| `voronoi` | Cell decomposition (natural looking) |
| `radial` | From impact point outward |
| `slice` | Planar cuts |
| `custom` | Artist-defined cuts |

**Damage System (from @destructible, @damage_type, @damage_resistance):**
- Per-chunk health accumulation
- Damage types with multipliers (e.g., "explosive" x2.0, "fire" x0.5)
- Resistance map per object
- Propagation to neighbors via support graph

**Support Structures:** Connection graph, anchor points, stress chains, collapse detection when support lost.

**Debris:** Impact debris (small particles), secondary debris (falling pieces), dust VFX. Optimized via LOD, count limits, lifetime, pooling.

### 6.6 Cloth Simulation

**Representation:** Particles (vertex positions), Edges (structural links), Triangles (surface faces). Rest shape, mass distribution, fixed/pinned particles, weight painting.

**Constraint Types:**
| Type | Description |
|------|-------------|
| Distance/Structural | Edge length preservation |
| Shear | Diagonal resistance |
| Bending | Dihedral angle resistance |
| Long-range attachment | Prevent excessive stretch |
| Anchor | Fixed point attachment |
| Tether | Maximum distance constraint |

**Solver Types:** PBD, XPBD, Mass-Spring, FEM.

**Cloth Collision:** World collision (mesh, capsule, SDF), Self collision (particle-particle with spatial hash).

**Cloth Forces:** Gravity, Wind (aerodynamic), Drag, Lift.

**GPU Cloth:** Compute shaders for parallel particle update, optional direct render without CPU readback, batching multiple cloths, LOD, sleeping.

### 6.7 Hair & Fur Simulation

**Strand Types:** Guide hairs (simulated), Interpolated (between guides), Clumps (grouped).

**Simulation Methods:**
| Method | Description |
|--------|-------------|
| Follow-The-Leader | Sequential constraints, fast |
| Discrete Elastic Rods | Physical model, accurate |
| PBD | Position-based, stable |
| Mass-Spring | Force-based, tunable |

**Hair Collision:** Body collision (capsules, SDF, mesh), Self collision (voxelization, grid-based).

**Hair LOD:** Full guides -> Reduced guides -> Interpolation blend -> Shell/card fallback.

### 6.8 Soft Body & Deformables

**Representation:** Particle-based (point cloud), Tetrahedral mesh (volume elements), Voxel grid.

**Simulation Methods:** FEM, Corotational FEM (large deformation), Shape Matching (geometric), PBD/XPBD.

**Material Properties:** Young's Modulus (stiffness), Poisson's Ratio (compressibility), Plasticity (permanent deformation), Damping.

**Muscle Simulation:** Fibers (contraction direction), Activation (tension control), Attachment points, Volume-preserving bulging, Jiggle (secondary motion).

### 6.9 Fluid Simulation

**SPH (Smoothed Particle Hydrodynamics):** Neighbor search -> Density estimate -> Pressure compute -> Forces (pressure, viscosity, surface tension, external).

**Position Based Fluids (PBF):** Predict -> Neighbor search -> Density constraint -> Position correction.

**FLIP/PIC/APIC:** Particles carry velocity, Grid for pressure solve, Particle-to-grid and grid-to-particle transfer. PIC/FLIP ratio controls smoothness vs detail.

**Eulerian:** Velocity field (staggered grid), Semi-Lagrangian advection, Pressure projection, Solid boundary.

**Shallow Water:** Height field (2D surface), Flow field, Terrain boundary.

**Surface Reconstruction:** Marching Cubes, Screen-Space, Anisotropic kernels.

**GPU Acceleration:** Compute shader particles, Parallel spatial hash sort, GPU neighbor query.

### 6.10 Vehicles

**Wheeled Vehicles:** Raycast/Shape-sweep/Rigid wheel models. Suspension (spring + damper, anti-roll bar). Tire model (Pacejka magic formula, linear, brush). Drivetrain (engine torque curve, transmission, differential, clutch). Drive layouts (FWD/RWD/AWD/4WD).

**Aircraft:** Lift, Drag, Thrust, Weight. Control surfaces (ailerons, elevator, rudder, flaps).

**Watercraft:** Buoyancy (point/voxel/mesh sampling), Water drag, Wave forces, Propulsion.

**Vehicle Feel:** Assists (steering, ABS, traction control, ESC). Tuning levels (arcade, simcade, simulation).

### 6.11 Character Physics

**Controller Types:** Kinematic (sweep-based), Dynamic (physics-driven), Hybrid (kinematic + physics response).

**Movement:** Ground detection (shape sweep, multi-ray), Slope handling (walkable angle, slide, step up), Platform attachment.

**Ragdoll:** Bone mapping (skeleton -> physics), Constraint limits per joint, Per-bone collision shapes. Activation: death (full), hit reaction (partial), blend in/out.

**Active Ragdoll:** Animation target pose -> Powered joints (PD controller) -> Physical response. Balance (COM tracking, support polygon, recovery).

---

## 7. Decorator Stacks

### Builtin Composite Stacks (composite.py)

#### @competitive_entity — Competitive Game Physics
```python
from trinity.decorators.builtin_stacks.composite import competitive_entity

@competitive_entity(pool_size=128, history_frames=600)
class CompetitivePhysicsBody(Component): ...
```
**Combines:** production_component + deterministic_data + replay_ready + predicted_entity + secure_multiplayer

#### @multiplayer_character — Networked Character
```python
from trinity.decorators.builtin_stacks.composite import multiplayer_character

@multiplayer_character(pool_size=64, history_frames=30, version=1)
class NetworkedCharacter(Component): ...
```
**Combines:** production_component + predicted_entity + versioned_saveable + secure_multiplayer

### Builtin Network Stacks (network.py)

#### @predicted_entity — Client-Side Prediction
```python
from trinity.decorators.builtin_stacks.network import predicted_entity

@predicted_entity(history_frames=30, max_reconcile_frames=10, snap_threshold=0.5)
class PredictedBody(Component): ...
```
**Combines:** @networked(authority="server", predicted=True, interpolated="hermite") + @snapshot(history_frames) + @server_reconcile(...) + @diff(strategy="shallow")

#### @secure_multiplayer — Anti-Cheat Hardened
```python
from trinity.decorators.builtin_stacks.network import secure_multiplayer

@secure_multiplayer(rate_limit=10)
class SecureBody(Component): ...
```
**Combines:** @server_authoritative + @validated(rules=[]) + @rate_limited(max_per_second=rate_limit, per="player")

#### @networked_entity — Basic Replication
```python
from trinity.decorators.builtin_stacks.network import networked_entity

@networked_entity(authority="server", relevance="spatial", priority=10, pool_size=64)
class ReplicatedBody(Component): ...
```
**Combines:** @component + @packed("soa") + @pooled + @networked(...) + @serializable("binary") + @track_changes

#### @bandwidth_efficient — Optimized Networking
```python
from trinity.decorators.builtin_stacks.network import bandwidth_efficient

@bandwidth_efficient(radius=5000, max_updates_per_second=20.0, priority=50)
class EfficientBody(Component): ...
```
**Combines:** @networked(relevance="spatial", delta=True) + @diff("structural") + @interest + @bandwidth_priority + @throttle_network + @batch

### Proposed Simulation Stacks

#### physics_body — Standard Rigid Body
```python
# Proposed:
@component
@simulation_domain(domain="rigid_body")
@physics_material(friction=0.5, restitution=0.3, density=1.0)
@sleep_threshold(linear=0.1, angular=0.05, time=0.5)
@packed(layout="soa")
@pooled(initial_size=1024)
@budget(category="physics")
class RigidBodyBase(Component): ...
```

#### destructible_object — Full Destruction
```python
# Proposed:
@component
@simulation_domain(domain="rigid_body")
@destructible(health=100.0, fracture_depth=2, debris_lifetime=10.0)
@fracture(pattern="voronoi", min_size=0.1)
@physics_material(friction=0.5, restitution=0.3)
@pooled(initial_size=512)
class DestructibleBase(Component): ...
```

#### networked_physics — Replicated Physics Body
```python
# Proposed:
@component
@simulation_domain(domain="rigid_body")
@physics_material(friction=0.5, restitution=0.3)
@networked(authority="server", predicted=True, interpolated="hermite")
@snapshot(history_frames=30)
@serializable(format="binary")
@packed(layout="soa")
@pooled(initial_size=256)
class NetworkedPhysicsBase(Component):
    position: Annotated[Fixed32, TrackedDescriptor(), PredictedDescriptor(max_history=30)]
    velocity: Annotated[Fixed32, TrackedDescriptor(), NetworkedDescriptor(authority="server")]
```

#### cloth_component — Cloth Simulation
```python
# Proposed:
@component
@simulation_domain(domain="cloth")
@solver_hint(type="xpbd", iterations=8)
@wind_affected(drag_coefficient=1.0, area="auto")
@pooled(initial_size=64)
class ClothBase(Component): ...
```

---

## 8. TODO Checklist (from GAME_ENGINE_INTEGRATION_TODO.md S6 + S17)

### 6.1 Physics Core
- [ ] Implement (or integrate) rigid body dynamics
- [ ] Implement broadphase collision (BVH or sweep-and-prune)
- [ ] Implement narrowphase collision (GJK/EPA, SAT)
- [ ] Implement constraint solver (sequential impulse or PGS)
- [ ] Wire `@simulation_domain` decorator -> physics world assignment
- [ ] Wire `@substep` decorator -> substep count configuration
- [ ] Wire `@solver_hint` decorator -> solver iteration count
- [ ] Wire `@sleep_threshold` decorator -> sleep parameters
- [ ] Wire `@continuous_collision` decorator -> CCD mode

### 6.2 Physics Components
- [ ] Implement RigidBody component (mass, inertia, velocity, forces)
- [ ] Implement Collider components (box, sphere, capsule, mesh, convex hull)
- [ ] Implement physics materials (friction, restitution, density)
- [ ] Wire `@physics_material` decorator -> material properties
- [ ] Register physics components via ComponentMeta -> Foundation Registry
- [ ] Wire TrackedDescriptor -> Tracker for physics state changes

### 6.3 Constraints
- [ ] Implement joint types (hinge, ball, prismatic, fixed, distance, spring)
- [ ] Wire `@joint` decorator -> joint configuration
- [ ] Implement motors and limits on joints

### 6.4 Advanced Simulation
- [ ] Implement destruction system (fracture, debris, damage propagation)
- [ ] Wire `@destructible`, `@fracture`, `@damage_type`, `@damage_resistance` decorators
- [ ] Implement buoyancy (`@buoyancy` decorator)
- [ ] Implement wind system (`@wind_affected` decorator)
- [ ] Integrate Foundation EventLog -- log collision events, destruction events

### 17.1 Deterministic Core
- [ ] Implement simulation boundary (deterministic kernel vs presentation)
- [ ] Implement fixed-point math types (no floats in simulation)
- [ ] Implement deterministic RNG (seeded, reproducible)
- [ ] Implement command-based input (ordered, timestamped)

### 17.2 Snapshot & Rollback
- [ ] Implement hierarchical checksums (per-entity, per-archetype, global)
- [ ] Implement snapshot system (Foundation ContentStore -> efficient snapshots)
- [ ] Implement rollback (Foundation Tracker.undo() -> revert to snapshot)
- [ ] Implement desync detection (compare checksums across clients)

### 17.3 Replay
- [ ] Implement frame-perfect replay from EventLog
- [ ] Implement replay scrubbing (jump to any frame)
- [ ] Implement replay comparison (divergence detection)

### 17.4 Network Determinism
- [ ] Implement lockstep networking (wait for all inputs)
- [ ] Implement rollback networking (predict -> rollback on mismatch)
- [ ] Wire Foundation DeltaSync -> efficient state correction

---

## 9. Directory Structure

```
engine/simulation/
├── __init__.py                       # Public API exports
├── physics/
│   ├── __init__.py
│   ├── physics_world.py              # World container, gravity, timestep, substeps
│   ├── rigid_body.py                 # Body types, state, forces, integration
│   ├── collision_shapes.py           # Sphere, Box, Capsule, Convex, Mesh, Compound
│   ├── physics_material.py           # Friction, restitution, density, combine modes
│   ├── sleeping.py                   # Sleep criteria, wake conditions, island sleeping
│   ├── body_flags.py                 # CCD, gravity, axis locks, gyroscopic
│   └── queries.py                    # Raycast, shape cast, overlap, closest point
├── collision/
│   ├── __init__.py
│   ├── broadphase.py                 # SAP, Dynamic BVH, Uniform Grid, Hash, Octree
│   ├── narrowphase.py                # GJK, EPA, SAT, MPR, specialized tests
│   ├── contact_manifold.py           # Persistent contacts, manifold reduction
│   ├── ccd.py                        # Linear sweep, time-of-impact, speculative
│   ├── collision_filter.py           # Layer/mask filtering, trigger detection
│   └── collision_events.py           # Begin/persist/end overlap, hit events
├── solver/
│   ├── __init__.py
│   ├── constraint_solver.py          # Sequential Impulse, PGS, warm starting
│   ├── tgs_solver.py                 # Temporal Gauss-Seidel
│   ├── xpbd_solver.py               # Extended Position Based Dynamics
│   ├── jacobian.py                   # Jacobian computation for constraints
│   └── island_manager.py            # Connected component detection, parallel islands
├── constraints/
│   ├── __init__.py
│   ├── joint_fixed.py                # Fixed joint (weld)
│   ├── joint_hinge.py                # Hinge/revolute joint
│   ├── joint_slider.py               # Prismatic/slider joint
│   ├── joint_ball.py                 # Ball/spherical joint
│   ├── joint_spring.py               # Spring-damper joint
│   ├── joint_distance.py             # Distance constraint
│   ├── joint_d6.py                   # Configurable 6-DOF joint
│   ├── joint_motors.py               # Velocity/position motors, servo
│   ├── joint_limits.py               # Linear/angular limits, soft/hard
│   └── contact_constraint.py         # Normal, friction, rolling, spinning
├── destruction/
│   ├── __init__.py
│   ├── destruction_system.py         # Damage accumulation, support evaluation
│   ├── fracture_voronoi.py           # Voronoi cell decomposition
│   ├── fracture_radial.py            # Radial fracture from impact point
│   ├── fracture_slice.py             # Planar slice cuts
│   ├── damage_types.py               # Damage type registry, multipliers
│   ├── support_graph.py              # Connection network, anchors, stress chains
│   └── debris.py                     # Debris spawning, LOD, lifetime, pooling
├── cloth/
│   ├── __init__.py
│   ├── cloth_simulation.py           # Particle system, constraint iteration
│   ├── cloth_constraints.py          # Distance, shear, bending, tether
│   ├── cloth_collision.py            # World collision, self collision
│   ├── cloth_wind.py                 # Aerodynamic forces
│   └── gpu_cloth.py                  # Compute shader cloth
├── hair/
│   ├── __init__.py
│   ├── hair_simulation.py            # Guide strands, FTL, DER
│   ├── hair_constraints.py           # Length, shape, root
│   ├── hair_collision.py             # Body collision, self collision
│   └── hair_lod.py                   # LOD system, card fallback
├── softbody/
│   ├── __init__.py
│   ├── fem_solver.py                 # Finite element method
│   ├── shape_matching.py             # Geometric shape matching
│   ├── soft_body_pbd.py              # PBD/XPBD soft body
│   ├── muscle.py                     # Muscle fiber simulation
│   └── deformable_mesh.py            # Tetrahedral mesh, embedding
├── fluid/
│   ├── __init__.py
│   ├── sph.py                        # Smoothed Particle Hydrodynamics
│   ├── pbf.py                        # Position Based Fluids
│   ├── flip_pic.py                   # FLIP/PIC/APIC hybrid
│   ├── eulerian.py                   # Grid-based Eulerian solver
│   ├── shallow_water.py              # Height field water
│   ├── surface_reconstruction.py     # Marching cubes, screen-space
│   └── gpu_fluid.py                  # Compute shader fluid
├── vehicles/
│   ├── __init__.py
│   ├── vehicle_system.py             # Vehicle update, input processing
│   ├── wheeled_vehicle.py            # Car, truck, bike
│   ├── suspension.py                 # Spring-damper, anti-roll bar
│   ├── tire_model.py                 # Pacejka, linear, brush
│   ├── drivetrain.py                 # Engine, transmission, differential
│   ├── tracked_vehicle.py            # Tank, excavator
│   ├── hover_vehicle.py              # Hovercraft
│   ├── aircraft.py                   # Plane, helicopter aerodynamics
│   └── watercraft.py                 # Boat, ship buoyancy + propulsion
├── character/
│   ├── __init__.py
│   ├── character_controller.py       # Kinematic/dynamic/hybrid controller
│   ├── ground_detection.py           # Shape sweep, slope handling, steps
│   ├── ragdoll.py                    # Bone mapping, activation, blending
│   ├── active_ragdoll.py             # Powered joints, PD controller, balance
│   └── platform_handling.py          # Moving platform attachment
└── components/
    ├── __init__.py
    ├── rigid_body_component.py       # RigidBody ECS component
    ├── collider_components.py        # BoxCollider, SphereCollider, etc.
    ├── joint_component.py            # Joint ECS component
    ├── cloth_component.py            # Cloth ECS component
    ├── vehicle_component.py          # Vehicle ECS component
    ├── character_component.py        # CharacterController ECS component
    ├── destruction_component.py      # Destructible ECS component
    └── fluid_component.py            # Fluid emitter ECS component
```

---

## 10. Canonical Usage Examples

### Example 1: Rigid Body Component
```python
from typing import Annotated
from trinity.base import Component
from trinity.types import Fixed32
from trinity.descriptors.tracking import TrackedDescriptor
from trinity.descriptors.networking import NetworkedDescriptor
from trinity.decorators.ecs_core import component
from trinity.decorators.physics_sim import simulation_domain, sleep_threshold, continuous_collision
from trinity.decorators.destruction import physics_material
from trinity.decorators.memory import pooled, packed
from trinity.decorators.data_flow import serializable

@component
@simulation_domain(domain="rigid_body")
@physics_material(friction=0.5, restitution=0.3, density=1.0)
@sleep_threshold(linear=0.1, angular=0.05, time=0.5)
@continuous_collision(mode="speculative")
@packed(layout="soa")
@pooled(initial_size=1024)
@serializable(format="binary")
class RigidBody(Component):
    """Dynamic rigid body with full physics simulation."""
    mass: Annotated[Fixed32, TrackedDescriptor()] = Fixed32(1.0)
    linear_velocity: Annotated[Fixed32, TrackedDescriptor()] = Fixed32(0.0)
    angular_velocity: Annotated[Fixed32, TrackedDescriptor()] = Fixed32(0.0)
    linear_damping: Annotated[float, TrackedDescriptor()] = 0.01
    angular_damping: Annotated[float, TrackedDescriptor()] = 0.05
    gravity_scale: Annotated[float, TrackedDescriptor()] = 1.0
    body_type: str = "dynamic"  # "static" | "kinematic" | "dynamic"
```

### Example 2: Destructible Object
```python
from trinity.decorators.destruction import destructible, fracture, damage_type, damage_resistance, physics_material

@component
@simulation_domain(domain="rigid_body")
@destructible(health=200.0, fracture_depth=3, debris_lifetime=15.0)
@fracture(pattern="voronoi", min_size=0.05, interior_material="concrete_interior")
@damage_resistance(resistances={"bullet": 0.8, "explosive": 1.5, "fire": 0.3})
@physics_material(friction=0.6, restitution=0.2, density=2.4)
@pooled(initial_size=256)
class DestructibleWall(Component):
    """Concrete wall that fractures under damage."""
    current_health: Annotated[float, TrackedDescriptor()] = 200.0
    is_fractured: bool = False
```

### Example 3: Physics System with Substeps
```python
from trinity.decorators.ecs_core import system
from trinity.decorators.physics_sim import substep, solver_hint
from trinity.decorators.scheduling import fixed, parallel, exclusive
from trinity.decorators.dev import profile

@system(phase="physics")
@fixed(hz=60)
@substep(min_hz=60, max_hz=240, max_substeps=4)
@solver_hint(type="pgs", iterations=8, warm_starting=True)
@exclusive
@profile(name="PhysicsSolver", warn_ms=8.0)
class PhysicsSolverSystem(System):
    """Main physics solver: broadphase, narrowphase, solve, integrate."""

    def execute(self, dt: float):
        # 1. Broadphase: generate collision pairs
        # 2. Narrowphase: generate contacts (GJK/EPA)
        # 3. Build islands (connected components)
        # 4. Per-island: warm start, iterate solver, position correction
        # 5. Integrate velocities -> positions
        # 6. Update sleeping (per-island)
        pass
```

### Example 4: Networked Physics with Prediction
```python
from trinity.decorators.builtin_stacks.network import predicted_entity

@component
@simulation_domain(domain="rigid_body")
@physics_material(friction=0.5, restitution=0.3)
@predicted_entity(history_frames=30, max_reconcile_frames=10, snap_threshold=0.5)
@pooled(initial_size=128)
class NetworkedProjectile(Component):
    """Server-authoritative projectile with client prediction."""
    position: Annotated[Fixed32,
        TrackedDescriptor(),
        PredictedDescriptor(max_history=30),
    ] = Fixed32(0.0)

    velocity: Annotated[Fixed32,
        TrackedDescriptor(),
        NetworkedDescriptor(authority="server", priority=5),
    ] = Fixed32(0.0)

    damage: float = 25.0
```

### Example 5: Vehicle Component
```python
@component
@simulation_domain(domain="vehicle")
@substep(min_hz=120, max_hz=480, max_substeps=8)
@solver_hint(type="pgs", iterations=6)
@serializable(format="binary")
class WheeledVehicle(Component):
    """4-wheeled vehicle with Pacejka tire model."""
    engine_rpm: Annotated[float, TrackedDescriptor()] = 0.0
    throttle: Annotated[float, TrackedDescriptor()] = 0.0
    brake: Annotated[float, TrackedDescriptor()] = 0.0
    steering: Annotated[float, TrackedDescriptor()] = 0.0
    gear: int = 0
    drive_type: str = "rwd"  # "fwd" | "rwd" | "awd"
```

### Example 6: Cloth Component
```python
@component
@simulation_domain(domain="cloth")
@solver_hint(type="xpbd", iterations=10, warm_starting=True)
@wind_affected(drag_coefficient=1.2, area="auto")
@pooled(initial_size=32)
class ClothSimulation(Component):
    """Cloth simulation with XPBD solver and wind."""
    stiffness: Annotated[float, TrackedDescriptor()] = 0.9
    damping: Annotated[float, TrackedDescriptor()] = 0.01
    gravity_scale: Annotated[float, TrackedDescriptor()] = 1.0
    self_collision: bool = True
    collision_margin: float = 0.01
```

---

## 11. Integration Patterns

### Pattern 1: Physics Dirty Flag -> World Sync
```python
# Physics property change flow:
rigidbody.mass = Fixed32(10.0)
# 1. TrackedDescriptor.post_set() fires
# 2. Tracker.mark_dirty(rigidbody, "mass")
# 3. Per-frame pre_physics system:
#    dirty_bodies = Tracker.all_dirty(type=RigidBody)
#    for body in dirty_bodies:
#        physics_world.sync_body(body)  # Re-upload mass, inertia
#    Tracker.clear_dirty(type=RigidBody)
```

### Pattern 2: Collision Event -> Destruction
```python
# Collision triggers destruction:
# 1. Narrowphase detects collision -> CollisionEvent(impulse=500)
# 2. Post-physics system dispatches event
# 3. DestructionSystem handles event:
#    if entity has @destructible:
#        damage = impulse * damage_type_multiplier / resistance
#        entity.current_health -= damage
#        if entity.current_health <= 0:
#            fracture(entity, pattern="voronoi")
#            spawn_debris(entity)
```

### Pattern 3: Deterministic Substep Loop
```python
# Fixed-timestep physics with substeps:
# 1. Accumulate frame time
# 2. While accumulated >= fixed_dt:
#    a. Pre-physics: force accumulation, CCD setup
#    b. Physics: broadphase -> narrowphase -> solve -> integrate
#    c. Post-physics: events, destruction, sleeping
#    d. accumulated -= fixed_dt
# 3. Store interpolation alpha = accumulated / fixed_dt
# All math uses Fixed32 for determinism.
```

### Pattern 4: Networked Physics Prediction + Rollback
```python
# Client-side prediction with server reconciliation:
# 1. Client predicts locally: apply input -> simulate -> store in PredictedDescriptor history
# 2. Server sends authoritative state
# 3. Client compares server state vs predicted state at same tick
# 4. If diverged beyond snap_threshold:
#    PredictedDescriptor.rollback(frames=delta)
#    Re-simulate from server state with buffered inputs
# 5. InterpolatedDescriptor smooths rendering between corrections
```

### Pattern 5: Spatial Indexing -> Broadphase
```python
# @spatial decorator integration:
# 1. @spatial(structure="bvh") on PhysicsWorld Resource
# 2. Broadphase uses BVH for AABB pair detection
# 3. @partitioned(dimensions=3, max_entities=10000) on collision layer
# 4. Island manager uses spatial partition for parallel solving
```

### Pattern 6: Foundation Bridge -> Live Physics Debug
```python
# ShellLang debugging:
# > QUERY RigidBody WHERE body_type == "dynamic" AND sleeping == False
#   -> "Found 847 active dynamic bodies"
# > QUERY Joint WHERE type == "hinge" AND stress > break_force * 0.9
#   -> Shows joints near breaking
# > MUTATE entity=42 COMPONENT=RigidBody SET gravity_scale=0.0
#   -> Zero gravity on specific entity
# > QUERY DestructibleWall WHERE current_health < 50
#   -> Find damaged walls
```

---

## 12. Quick Reference Tables

### Simulation Decorators Summary
| Decorator | Module | Tier | Key Params | Purpose |
|-----------|--------|------|------------|---------|
| @simulation_domain | physics_sim | 46 | domain | Physics world assignment |
| @substep | physics_sim | 46 | min_hz, max_hz, max_substeps | Substep configuration |
| @solver_hint | physics_sim | 46 | type, iterations, warm_starting | Solver configuration |
| @sleep_threshold | physics_sim | 46 | linear, angular, time | Sleep parameters |
| @continuous_collision | physics_sim | 46 | mode | CCD mode |
| @buoyancy | physics_sim | 46 | density, drag, angular_drag | Water buoyancy |
| @wind_affected | physics_sim | 46 | drag_coefficient, area | Wind response |
| @destructible | destruction | 43 | health, fracture_depth, debris_lifetime | Destructible object |
| @damage_type | destruction | 43 | id, base_multiplier | Damage type definition |
| @damage_resistance | destruction | 43 | resistances | Damage resistance map |
| @fracture | destruction | 43 | pattern, min_size, interior_material | Fracture configuration |
| @physics_material | destruction | 43 | friction, restitution, density | Surface properties |
| @joint | destruction | 43 | type, break_force, break_torque | Joint definition |
| @spatial | spatial | — | structure, cell_size | Spatial indexing |
| @partitioned | spatial | — | dimensions, max_entities | Spatial partitioning |

### SystemPhase for Simulation
| Phase | Value | Simulation Systems |
|-------|-------|-------------------|
| PRE_PHYSICS | 0 | Force accumulation, CCD setup, input processing |
| PHYSICS | 1 | Broadphase, narrowphase, solve, integrate |
| POST_PHYSICS | 2 | Contact callbacks, destruction, event dispatch |

### Physics Solver Comparison
| Solver | Convergence | Stability | Use Case |
|--------|-------------|-----------|----------|
| PGS | Medium | Good | General rigid body (default) |
| TGS | Better | Very good | Stacking, ragdolls, complex scenes |
| XPBD | Compliance-based | Excellent | Cloth, soft body, ropes |

### Joint Types Quick Reference
| Joint | Free DOF | Params | Example |
|-------|----------|--------|---------|
| fixed | 0 | break_force/torque | Welded objects |
| hinge | 1 rotation | limits, motor, break | Door, wheel axle |
| slider | 1 translation | limits, motor, break | Piston, sliding door |
| ball | 3 rotation | cone limit, break | Ragdoll shoulder |
| spring | soft | stiffness, damping | Suspension, rope |

### Collision Shape Performance
| Shape | Broadphase | Narrowphase | Best For |
|-------|-----------|-------------|----------|
| Sphere | Fastest | Fastest | Projectiles, particles |
| Capsule | Fast | Fast | Characters, limbs |
| Box | Fast | Fast | Crates, buildings |
| Convex Hull | Medium | Medium | Props, vehicles |
| Triangle Mesh | Slow | Slow | Static world geometry only |

### Descriptor Stacking for Physics Fields
```
StorageDescriptor           <- innermost (raw storage)
  -> ValidatedDescriptor    <- bounds (mass > 0, velocity clamped)
    -> TrackedDescriptor    <- dirty flags -> Foundation Tracker -> world sync
      -> NetworkedDescriptor <- network queue -> replication
        -> PredictedDescriptor <- prediction history + rollback (outermost)
```

### Fixed-Point Types for Deterministic Physics
| Type | Format | Range | Precision | Use |
|------|--------|-------|-----------|-----|
| Fixed16 | Q8.8 | -128.0 to 127.996 | ~0.0039 | Normalized values, angles |
| Fixed32 | Q16.16 | -32768.0 to 32767.999 | ~0.000015 | Positions, velocities, forces |

See `engine/determinism/DETERMINISM_CONTEXT.md` for full fixed-point arithmetic, snapshot, checksum, and replay details.

### Constants Reference (from trinity/constants.py)
| Constant | Value | Usage |
|----------|-------|-------|
| DEFAULT_PHYSICS_HZ | 60 | Standard physics update rate |
| DEFAULT_CHUNK_SIZE | 64 | SIMD-friendly batch size |
| DEFAULT_MIN_BATCH | 256 | Minimum for parallelization |
| DEFAULT_POOL_SIZE | 1024 | Initial pool allocation |
| DEFAULT_PREDICTION_HISTORY | 30 | Frames for rollback |
| INTERPOLATION_BUFFER_SIZE | 3 | Snapshots for interpolation |
| CACHE_LINE_BYTES | 64 | Memory alignment |
| DEFAULT_MAX_UPDATES_PER_SECOND | 20.0 | Network throttle rate |
