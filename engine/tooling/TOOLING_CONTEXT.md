# TOOLING_CONTEXT.md — Tooling & Editor Layer

> **Purpose**: Complete implementation reference for the engine/tooling/ layer.
> Read this file and ONLY this file when implementing tooling and editor systems.

---

## 1. Architecture Summary

The tooling layer is the **heaviest Foundation consumer** in the engine. It observes, inspects, edits, and controls every other engine layer through Foundation's six core systems plus VIPER extensions. It does not participate in the runtime game loop directly; instead, it wraps the runtime in an editor shell that provides content creation, debugging, profiling, and build pipeline workflows.

**Core Subsystems (16):**

1. **Editor Framework** -- Application shell (window, menus, docking, panels), viewport (perspective/ortho, fly/orbit/pan/zoom cameras), render modes (lit/unlit/wireframe/debug views), editor modes (edit/play/simulate/prefab), plugin system (lifecycle: load/init/enable/disable/unload, extension points for UI/menus/tools/importers)
2. **Level Editor** -- Entity inspector (Mirror-backed property panel), transform gizmos (translate/rotate/scale in world/local/view/parent spaces), snapping (grid/surface/vertex/edge/pivot/increment), scene hierarchy (tree view with search/filter/sort, parent/group/isolate), object placement (drag-drop from content browser, surface place, paint brush, distribution arrays)
3. **Asset Browser** -- Content browser (folder tree, thumbnails, list/column views), import pipeline (source file detection, format-specific settings, process, validate, engine asset output), reference management (find references, dependency graph, redirectors), asset validation (reference check, size check, naming convention, quality check)
4. **Profiler** -- CPU profiler (flame graph, timeline, hotspots, thread view), GPU profiler (render pass timing, draw call analysis, shader cost, VRAM), memory profiler (allocation tracking, leak detection, category breakdown, snapshots), network profiler (bandwidth, RTT, packet loss, per-actor/per-property)
5. **Debug Draw** -- Primitives (lines, arrows, points, spheres, boxes, capsules, cylinders, cones, planes, meshes), text (screen text, world text, billboard), draw options (color, duration, thickness, depth test), overlay categories (physics, navigation, AI, audio, rendering, network)
6. **Console** -- UI modes (overlay, full screen, mini bar), command types (exec commands, console variables, aliases, script execution), CVar system (int/float/bool/string, flags: ReadOnly/Cheat/Config/Scalability), command categories (r.* rendering, p.* physics, ai.* AI, net.* network, stat * statistics, show * flags)
7. **Hot-Reload** -- `@reloadable` decorator (preserve/reinitialize field lists, validate callback), schema hash detection via `Mirror.schema_hash()` (hash changed = migration needed), `SystemMeta.hot_reload()` / `SystemMeta.reload_system()` for live system replacement, `AssetMeta.check_changes()` for file watcher integration
8. **Undo/Redo** -- Foundation Tracker transactions (`begin_transaction`/`commit_transaction`/`rollback_transaction`), atomic grouping of changes, coalescing within transactions, `undo()`/`redo()` with bounded stacks
9. **Material Editor** -- Node-based material graph, material outputs (base_color, metallic, roughness, normal, emissive, AO), input nodes (texture sample, parameters, UVs, vertex data, time), math nodes (arithmetic, trig, vector, interpolation), preview (sphere/cube/plane/custom mesh, studio/outdoor/HDR environment), material instances (scalar/vector/texture/switch overrides, parent-child hierarchy)
10. **Animation Tools** -- Sequencer/timeline (tracks, keyframes, curves, playhead), track types (transform, property, event, audio, camera), skeleton editor (bone hierarchy, sockets, virtual bones, retarget, mirror, bone LOD), montage editor (sections, links, loops, slots), animation graph (state machines, blend nodes, IK nodes, bone modify)
11. **Visual Scripting (FlowForge)** -- Graph editor (pan/zoom, box select, minimap), node types (flow control: branch/loop/switch/sequence; data: variables/constants/math/conversion; events: begin-play/tick/input/collision/custom; actions: function-call/spawn/destroy/print), blueprint features (components, functions, local variables, pure functions), debugging (breakpoints, watch values, step-through, execution flow visualization), Foundation Capabilities for sandbox permissions
12. **Build & Cook** -- Build types (editor/debug/development/shipping), build targets (Windows/Linux/macOS/console/mobile), cook process (discover assets, resolve dependencies, convert, optimize, package), incremental cook (content hash change detection, cook cache, derived data cache), packaging (executable + content paks + config + splash, compression/encryption/chunking/patch generation)
13. **VCS Integration** -- Supported systems (Git, Perforce, SVN, Plastic SCM), editor UI (file status, diff view, history, blame), operations (checkout/commit/revert/update, branch/switch/merge/resolve), binary file handling (lock/unlock, LFS), Foundation ContentStore for content-addressable asset versioning
14. **Terrain Tools** -- Sculpt modes (sculpt/smooth/flatten/erosion/noise/ramp), brush settings (size/strength/falloff/shape), painting (layer weights, holes, alpha), import/export (heightmap RAW/PNG, weight maps)
15. **Localization** -- Dashboard (languages, progress, issues), workflow (gather text from source, export PO/XLIFF, import translations, compile archive), preview (live language switch, pseudo-loc, visual layout check)
16. **Automation & CI/CD** -- Build pipeline (trigger, compile, cook, package, deploy), build agents (local/farm/cloud), automated testing (unit/functional/playtest/performance), scripted automation (commandlets for CLI cook/build/import, Python API for editor access)

**Architecture Patterns:**

- **Retained-Mode UI** -- Widget tree persisted across frames; updates via dirty flags, not full rebuild
- **Editor-Runtime Separation** -- Editor wraps runtime via Foundation Bridge; play mode creates isolated World snapshot; stop restores editor state
- **Command Pattern for Undo** -- Every user action is a Tracker transaction; commit stores in undo stack; rollback reverts atomically
- **Plugin Lifecycle** -- Load -> Initialize -> Enable -> [Active] -> Disable -> Unload; extension points for registration, hooks, UI panels, menu items
- **Foundation as Backbone** -- Registry for type browsing, Tracker for undo/dirty, EventLog for action history, Mirror for property display, Bridge for editor-runtime communication, Serializer for scene persistence, ContentStore for asset deduplication, DeltaSync for live editing, Provenance for data lineage

**Dependency Chain:**
```
ALL ENGINE LAYERS (Platform, Core, Resource, Simulation, Animation,
                   Rendering, Audio, Gameplay, UI, Networking, World)
    |
    v (observed/controlled via Foundation)
TOOLING & EDITOR ← THIS LAYER
```

---

## 2. Decorators (Complete Reference)

### 2.1 Dev Decorators (Tier 6 -- dev.py) — 9 Decorators

#### @profile
```python
@profile(
    name: Optional[str] = None,           # Display name; defaults to target.__name__
    warn_ms: Optional[float] = None,      # Threshold in ms; emits warning if exceeded
    track_allocations: bool = False,       # Track memory allocations (future)
)
```
- **Steps:** TAG(profiled={name, warn_ms, track_allocations}), HOOK(before_call), HOOK(after_call), REGISTER(dev)
- **After-Apply (class):** `_profiled=True`, `_profile_name`
- **After-Apply (function):** Wraps with timing; adds `profile_stats() -> dict`, `profile_reset()`
- **Targets:** function, class

#### @gpu_profile
```python
@gpu_profile(
    category: str,                        # REQUIRED, non-empty string (e.g., "culling", "shadows")
    include_memory: bool = False,         # Include VRAM tracking
)
```
- **Steps:** TAG(gpu_profiled={category, include_memory}), HOOK(before_call), HOOK(after_call), REGISTER(dev)
- **After-Apply:** `_gpu_profiled=True`, `_gpu_profile_category`, `_gpu_profile_include_memory`, `gpu_stats() -> dict`
- **Targets:** function, class

#### @trace
```python
@trace(
    level: str = "debug",                 # "debug" | "info" | "warn"
)
```
- **Steps:** TAG(traced={level}), HOOK(before_call), HOOK(after_call), REGISTER(dev)
- **After-Apply (class):** `_traced=True`, `_trace_level`
- **After-Apply (function):** Wraps with ENTER/EXIT/ERROR logging to `trinity.trace` logger
- **Targets:** function, class

#### @reloadable
```python
@reloadable(
    enabled: bool = True,                 # Enable/disable hot-reload
    preserve: list[str] = [],             # Field names to preserve across reload
    reinitialize: list[str] = [],         # Field names to reinitialize after reload
    validate: Optional[Callable] = None,  # Post-reload validation callback
)
```
- **Steps:** TAG(reloadable={enabled, preserve, reinitialize}), REGISTER(dev)
- **After-Apply:** `_reloadable=True`, `_reload_preserve`, `_reload_reinitialize`, `_reload_validate`
- **Targets:** class, function

#### @editor
```python
@editor(
    category: str = "General",            # Editor category for grouping
    hidden: bool = False,                 # Hide from editor UI
)
```
- **Steps:** TAG(editor={category, hidden}), REGISTER(dev)
- **After-Apply:** `_editor=True`, `_editor_category`, `_editor_hidden`
- **Targets:** any

#### @test
```python
@test(
    cases: list[dict] = [],               # Test case configurations
    fuzz: bool = False,                   # Enable fuzz testing
    property_based: bool = False,         # Enable property-based testing
)
```
- **Steps:** TAG(test={cases, fuzz, property_based}), REGISTER(dev)
- **After-Apply:** `_test=True`, `_test_cases`, `_test_fuzz`, `_test_property_based`
- **Targets:** function, class

#### @bench
```python
@bench(
    iterations: int = 1000,               # > 0, number of benchmark iterations
    warmup: int = 100,                    # >= 0, warmup iterations before measurement
)
```
- **Steps:** TAG(bench={iterations, warmup}), REGISTER(dev)
- **After-Apply:** `_bench=True`, `_bench_iterations`, `_bench_warmup`
- **Targets:** function, class

#### @invariant
```python
@invariant(
    check: Callable,                      # REQUIRED, callable that validates state
    when: str = "debug",                  # "debug" | "always"
)
```
- **Steps:** TAG(invariant={check, when}), VALIDATE(invariant_check), REGISTER(dev)
- **After-Apply:** Appends to `_invariants` list (accumulates, not unique)
- **Targets:** class, function

#### @deprecated
```python
@deprecated(
    since: str,                           # REQUIRED, non-empty version string
    replacement: Optional[str] = None,    # Suggested replacement
    remove_in: Optional[str] = None,      # Version when removal is planned
)
```
- **Steps:** TAG(deprecated={since, replacement, remove_in}), HOOK(before_call), REGISTER(dev)
- **After-Apply (class):** `_deprecated=True`, `_deprecated_since`, `_deprecated_replacement`, `_deprecated_remove_in`
- **After-Apply (function):** Wraps with DeprecationWarning emission
- **Targets:** function, class

### 2.2 Debug/Cheat Decorators (Tier -- debug_cheat.py) — 3 Decorators

#### @cheat
```python
@cheat(
    name: str,                            # REQUIRED, non-empty command name
    category: str = "general",            # Cheat category for grouping
    requires_confirmation: bool = False,  # Prompt before execution
)
```
- **Steps:** TAG(cheat=True), TAG(cheat_name), TAG(cheat_category), TAG(cheat_requires_confirmation), REGISTER(debug_cheat)
- **After-Apply:** `_cheat=True`, `_cheat_name`, `_cheat_category`, `_cheat_requires_confirmation`
- **Targets:** function

#### @debug_draw
```python
@debug_draw(
    color: Any = None,                    # Draw color (tuple, hex, or color object)
    duration: float = 0.0,               # >= 0, seconds to display (0 = one frame)
    depth_test: bool = True,             # Whether to depth-test against scene geometry
)
```
- **Steps:** TAG(debug_draw=True), TAG(debug_draw_color), TAG(debug_draw_duration), TAG(debug_draw_depth_test), REGISTER(debug_cheat)
- **After-Apply:** `_debug_draw=True`, `_debug_draw_color`, `_debug_draw_duration`, `_debug_draw_depth_test`
- **Targets:** class, function

#### @inspector
```python
@inspector(
    category: str = "default",            # Inspector panel category
    readonly: bool = False,               # Prevent editing in inspector
    range: Optional[tuple[float, float]] = None,  # (min, max) for numeric sliders
)
```
- **Steps:** TAG(inspector=True), TAG(inspector_category), TAG(inspector_readonly), TAG(inspector_range), REGISTER(debug_cheat)
- **After-Apply:** `_inspector=True`, `_inspector_category`, `_inspector_readonly`, `_inspector_range`
- **Targets:** class, function

### 2.3 Debug Safety Decorators (Tiers 10-11 -- debug_safety.py) — 4 Decorators

#### @reads
```python
@reads(*components: type)
```
- **Steps:** TAG(reads=True), TAG(reads_components=components), REGISTER(debug_safety)
- **After-Apply:** `_reads=True`, `_reads_components`
- **Targets:** function
- **Usage:** Declare read access on system functions for dependency analysis and parallel scheduling

#### @writes
```python
@writes(*components: type)
```
- **Steps:** TAG(writes=True), TAG(writes_components=components), REGISTER(debug_safety)
- **After-Apply:** `_writes=True`, `_writes_components`
- **Targets:** function
- **Usage:** Declare write access on system functions for conflict detection

#### @trace_stack
```python
@trace_stack(
    depth: int = 3,                       # > 0, number of stack frames to show
    show_decorator_chain: bool = True,    # Include applied decorators in trace
)
```
- **Steps:** TAG(trace_stack=True), TAG(trace_stack_depth), TAG(trace_stack_show_chain), HOOK(on_error), REGISTER(debug_safety)
- **After-Apply:** `_trace_stack=True`, `_trace_stack_depth`, `_trace_stack_show_chain`
- **Targets:** function

#### @track_changes
```python
@track_changes(
    fields: Optional[list[str]] = None,   # Field names to track, or None for all
)
```
- **Steps:** TAG(track_changes=True), TAG(track_changes_fields), TRACK(), REGISTER(change_detection)
- **After-Apply:** `_tracked=True`, `_tracked_fields`
- **Targets:** class (requires @component)

### 2.4 Debug Extended Decorators (Tier 51 -- debug_extended.py) — 2 Decorators

#### @network_debug
```python
@network_debug(
    log_packets: bool = False,            # Log all packet data
    simulate_latency: float = 0.0,        # >= 0, simulated latency in seconds
    simulate_loss: float = 0.0,           # 0.0-1.0, simulated packet loss ratio
)
```
- **Steps:** TAG(network_debug=True), TAG(network_debug_config=NetworkDebugConfig(...)), REGISTER(debug_extended)
- **After-Apply:** `_network_debug=True`, `_network_debug_log_packets`, `_network_debug_simulate_latency`, `_network_debug_simulate_loss`, `_network_debug_config`
- **Targets:** class

#### @automation_test
```python
@automation_test(
    category: str,                        # REQUIRED, non-empty test category
    timeout_seconds: float = 30.0,        # > 0, test timeout
    required_features: set[str] = set(),  # Platform features required to run
)
```
- **Steps:** TAG(automation_test=True), TAG(automation_test_config=AutomationTestConfig(...)), REGISTER(debug_extended)
- **After-Apply:** `_automation_test=True`, `_automation_test_category`, `_automation_test_timeout_seconds`, `_automation_test_required_features`, `_automation_test_config`
- **Targets:** class

### 2.5 Supporting Decorators Referenced by Tooling

| Decorator | Module | Tier | Tooling Role |
|-----------|--------|------|--------------|
| @component | ecs_core | 1 | Editor-visible data types; inspector displays fields |
| @system(phase=...) | ecs_core | 1 | Systems shown in scheduler view; phase assignment |
| @resource | ecs_core | 1 | Singleton Resources shown in resource browser |
| @event | ecs_core | 1 | Events shown in event timeline/log |
| @parallel(chunk_size=) | scheduling | 3 | Scheduler debug view: parallelizable systems |
| @exclusive | scheduling | 3 | Scheduler debug view: exclusive systems |
| @pooled(initial_size=) | memory | 2 | Memory profiler: pool stats |
| @packed(layout=) | memory | 2 | Memory profiler: layout info |
| @budget(category=) | memory | 2 | Memory profiler: budget tracking |
| @serializable(format=) | data_flow | 4 | Scene save/load, prefab serialization |
| @versioned(version=) | data_flow | 4 | Schema migration on load |
| @networked(authority=) | data_flow | 4 | Network profiler: replication stats |
| @on_add/@on_remove/@on_change | lifecycle | 7 | Event timeline visualization |
| @on_spawn/@on_despawn | lifecycle | 7 | Entity lifecycle timeline |

---

## 3. Metaclasses

All metaclasses are in `trinity/metaclasses/`. Tooling uses their class-level APIs for type browsing, system inspection, and live reload.

### EngineMeta (`engine_meta.py`) — Base Metaclass

```python
class EngineMeta(type):
    _all_engine_types: ClassVar[dict[int, type]]

    @classmethod
    def get_all_types(mcs) -> list[type]
        # Returns every class created by any EngineMeta subclass

    @classmethod
    def get_types_by_metaclass(mcs, meta: type) -> list[type]
        # Filter by specific metaclass (e.g., ComponentMeta)

    @classmethod
    def clear_registry(mcs) -> None
        # Reset for testing
```

### ComponentMeta (`component_meta.py`)

```python
class ComponentMeta(EngineMeta):
    _registry: ClassVar[dict[int, type]]        # id -> type
    _name_to_id: ClassVar[dict[str, int]]        # name -> id
    _next_id: ClassVar[int]
    _lock: ClassVar[threading.Lock]

    @classmethod
    def all_components(mcs) -> list[type]
        # All registered component types

    @classmethod
    def get_by_id(mcs, component_id: int) -> Optional[type]

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[type]

    @classmethod
    def component_count(mcs) -> int

    @classmethod
    def clear_registry(mcs) -> None

    # Hot-reload support (tooling-critical):
    @classmethod
    def _process_fields(mcs, cls: type) -> None
        # Re-extract __annotations__, rebuild _field_types/_field_defaults/_field_offsets

    @classmethod
    def _install_descriptors(mcs, cls: type) -> None
        # Re-compose descriptor chains from _applied_steps

    # Pool/instance methods (inspector/profiler):
    @classmethod
    def pool_stats(mcs) -> dict[str, Any]
        # Pool allocation statistics per component type

    @classmethod
    def instance_count(mcs, cls: type) -> int
        # Via Foundation Registry instance tracking
```

**Tooling usage:** Content browser type listing, inspector field display, hot-reload detection, memory profiler pool stats.

### SystemMeta (`system_meta.py`)

```python
class SystemMeta(EngineMeta):
    _registry: ClassVar[dict[int, type]]
    _phases: ClassVar[dict[SystemPhase, list[int]]]
    _next_id: ClassVar[int]
    _lock: ClassVar[threading.Lock]

    @classmethod
    def all_systems(mcs) -> list[type]

    @classmethod
    def get_by_id(mcs, system_id: int) -> Optional[type]

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[type]

    @classmethod
    def get_phase_systems(mcs, phase: SystemPhase) -> list[type]
        # All systems in a given phase

    @classmethod
    def get_phase_order(mcs, phase: SystemPhase) -> list[type]
        # Topologically sorted systems for a phase

    @classmethod
    def get_parallel_groups(mcs, phase: SystemPhase) -> list[list[type]]
        # Groups of non-conflicting systems that can run in parallel

    @classmethod
    def hot_reload(mcs) -> None
        # Re-analyze all system dependencies and rebuild phase order

    @classmethod
    def reload_system(mcs, system_cls: type) -> None
        # Hot-replace a single system while preserving state

    @classmethod
    def clear_registry(mcs) -> None
```

**Tooling usage:** System scheduler visualization, dependency graph view, parallel group debugging, hot-reload of game logic.

### EventMeta (`event_meta.py`)

```python
class EventMeta(EngineMeta):
    _registry: ClassVar[dict[int, type]]
    _name_to_id: ClassVar[dict[str, int]]

    @classmethod
    def all_events(mcs) -> list[type]

    @classmethod
    def get_by_id(mcs, event_id: int) -> Optional[type]

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[type]

    @classmethod
    def get_by_channel(mcs, channel: str) -> list[type]
        # Events assigned to a specific channel

    @classmethod
    def get_subtypes(mcs, event_cls: type) -> list[type]

    @classmethod
    def is_subtype(mcs, child: type, parent: type) -> bool

    @classmethod
    def pool_stats(mcs) -> dict[str, Any]
        # Event pool statistics

    # Serialization for network/replay:
    @classmethod
    def serialize(mcs, event_instance: Any) -> dict

    @classmethod
    def deserialize(mcs, data: dict) -> Any

    @classmethod
    def clear_registry(mcs) -> None
```

**Tooling usage:** Event timeline visualization, event channel browser, network event debugging.

### StateMeta (`state_meta.py`)

```python
class StateMeta(EngineMeta):
    _registry: ClassVar[dict[int, type]]
    _name_to_id: ClassVar[dict[str, int]]

    @classmethod
    def all_states(mcs) -> list[type]

    @classmethod
    def get_by_id(mcs, state_id: int) -> Optional[type]

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[type]

    @classmethod
    def get_machine_states(mcs, machine_name: str) -> list[type]
        # All states belonging to a named state machine

    @classmethod
    def can_transition(mcs, from_state: type, to_state: type) -> bool
        # Validate a specific transition

    @classmethod
    def validate_transitions(mcs, machine_name: str) -> list[str]
        # Return list of invalid/unreachable transitions

    @classmethod
    def get_history(mcs, machine_name: str) -> list[type]
        # State transition history for replay/debug

    @classmethod
    def get_enter_hook(mcs, state_cls: type) -> Optional[Callable]

    @classmethod
    def get_exit_hook(mcs, state_cls: type) -> Optional[Callable]

    @classmethod
    def register_with_machine(mcs, state_cls: type, machine_name: str) -> None

    @classmethod
    def clear_registry(mcs) -> None
```

**Tooling usage:** State machine visualizer (graph of states + transitions), animation graph editor, AI behavior debugging.

### ResourceMeta (`resource_meta.py`)

```python
class ResourceMeta(EngineMeta):
    _registry: ClassVar[dict[int, type]]

    @classmethod
    def all_resources(mcs) -> list[type]

    @classmethod
    def get_by_id(mcs, resource_id: int) -> Optional[type]

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[type]

    @classmethod
    def get_instance(mcs, resource_cls: type) -> Optional[Any]
        # Returns the singleton instance

    @classmethod
    def has_instance(mcs, resource_cls: type) -> bool

    @classmethod
    def initialize_all(mcs) -> None
        # Dependency-ordered initialization of all resources

    @classmethod
    def shutdown_all(mcs) -> None
        # Reverse-order shutdown

    @classmethod
    def reset_instance(mcs, resource_cls: type) -> None
        # Destroy and re-create a singleton (for hot-reload)

    @classmethod
    def clear_registry(mcs) -> None
```

**Tooling usage:** Resource browser, singleton inspector, dependency-ordered init/shutdown debugging.

---

## 4. Descriptors

All descriptors are in `trinity/descriptors/`. Tooling uses them for data binding, undo/redo, dirty flags, change tracking, debug profiling, and conditional breakpoints.

### ObservableDescriptor (`observable.py`) — UI Data Binding

```python
class ObservableDescriptor(BaseDescriptor[T]):
    descriptor_id = "observable"
    accepts_inner = ("tracked", "storage", "validated", "range")
    accepts_outer = ("networked", "cached")
    excludes = ("transient",)

    def post_set(self, obj, value, old_value) -> None:
        # Fires all registered observers: callback(obj, field_name, old_value, new_value)
```

**Helper functions:**
```python
add_observer(obj: Any, field_name: str, callback: Observer) -> None
remove_observer(obj: Any, field_name: str, callback: Observer) -> None
clear_observers(obj: Any, field_name: Optional[str] = None) -> None
```
**Tooling usage:** Inspector panel auto-refresh when component fields change. UI data binding for editor panels.

### BoundDescriptor (`observable.py`) — Two-Way Data Binding

```python
class BoundDescriptor(BaseDescriptor[T]):
    descriptor_id = "bound"
    accepts_inner = ("*",)
    accepts_outer = ("*",)

    def __init__(self, field_type=object, inner=None,
                 source=None,              # External data source object
                 getter: Callable = None,  # (source) -> value
                 setter: Callable = None,  # (source, value) -> None
                 **config)
```
**Tooling usage:** Bind editor UI widget directly to a component field; changes flow bidirectionally.

### TrackedDescriptor (`tracking.py`) — Undo/Redo, Dirty Flags

```python
class TrackedDescriptor(BaseDescriptor[T]):
    descriptor_id = "tracked"
    accepts_inner = ("storage", "validated", "range")
    accepts_outer = ("networked", "observable", "cached")
    excludes = ("computed",)

    def __init__(self, field_type=object, inner=None,
                 field_offset: int = 0,    # Bit position for bitmask tracking
                 use_bitmask: bool = False, # Use bitmask instead of set
                 **config)

    def post_set(self, obj, value, old_value) -> None:
        # 1. Adds field to obj._dirty_fields (set) or obj._dirty_mask (bitmask)
        # 2. Calls tracker.mark_dirty(obj, field, old, new) -- Foundation Tracker
        # 3. Calls add_change_to_current_event(Change(...)) -- Foundation EventLog
```

**Helper functions:**
```python
is_dirty(obj, field_name: str) -> bool
get_dirty_fields(obj) -> set[str]
clear_dirty(obj) -> None
clear_dirty_field(obj, field_name: str) -> None
```
**Tooling usage:** Undo/redo via Tracker transactions. Scene save optimization (only serialize dirty fields). Inspector dirty-flag indicators.

### DiffDescriptor (`tracking.py`) — Change Tracking

```python
class DiffDescriptor(BaseDescriptor[T]):
    descriptor_id = "diff"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    VALID_STRATEGIES = {"shallow", "deep", "structural", "custom"}

    def __init__(self, field_type=object, inner=None,
                 strategy: str = "shallow",
                 custom_differ: Optional[Callable[[Any, Any], bool]] = None,
                 **config)

    def get_previous(self, obj) -> Optional[T]
    def has_changed(self, obj) -> bool
```
**Tooling usage:** "What changed?" queries in inspector, diff view for undo history.

### VersionedDescriptor (`tracking.py`) — Per-Field Version Counter

```python
class VersionedDescriptor(BaseDescriptor[T]):
    descriptor_id = "versioned"
    accepts_inner = ("*",)
    accepts_outer = ("*",)

    def get_version(self, obj) -> int
        # Returns increment count for this field on obj
```
**Tooling usage:** Cache invalidation for computed views, optimistic concurrency in collaborative editing.

### ProfiledDescriptor (`debug.py`) — Timing Get/Set (DEBUG ONLY)

```python
class ProfiledDescriptor(BaseDescriptor[T]):
    descriptor_id = "profiled"
    accepts_inner = ("*",)
    accepts_outer = ("*",)

    def __init__(self, field_type=object, inner=None,
                 max_samples: int = 100,   # Ring buffer size for timing samples
                 **config)

    def get_stats(self, obj) -> dict:
        # Returns {"get": {count, avg_ns, max_ns, min_ns}, "set": {count, avg_ns, max_ns, min_ns}}
```
**Tooling usage:** Field access profiler. Identify hot fields causing performance issues.

### LoggedDescriptor (`debug.py`) — Logging All Accesses (DEBUG ONLY)

```python
class LoggedDescriptor(BaseDescriptor[T]):
    descriptor_id = "logged"
    accepts_inner = ("*",)
    accepts_outer = ("*",)

    def __init__(self, field_type=object, inner=None,
                 log_level: str = "DEBUG", # "DEBUG" | "INFO" | "WARNING" | "ERROR"
                 **config)
```
**Tooling usage:** Audit trail for inspector field reads/writes.

### WatchedDescriptor (`debug.py`) — Conditional Breakpoints (DEBUG ONLY)

```python
class WatchedDescriptor(BaseDescriptor[T]):
    descriptor_id = "watched"
    accepts_inner = ("*",)
    accepts_outer = ("*",)

    def __init__(self, field_type=object, inner=None,
                 condition: Optional[Callable[[Any], bool]] = None,  # Trigger when True
                 callback: Optional[Callable[[Any, str, Any], None]] = None,  # Or pdb.set_trace()
                 **config)
```
**Tooling usage:** Conditional data breakpoints. Watch a field and break when it hits a value (e.g., `health <= 0`).

---

## 5. Foundation Integration

The tooling layer consumes ALL Foundation systems. This section details the complete API surface used by each.

### 5.1 Registry — Type Browsing

```python
from foundation.registry import registry

registry.all_types() -> list[type]                       # Content browser: list all types
registry.subclasses(base) -> list[type]                  # Type hierarchy browser
registry.types_with_decorator("editor") -> list[type]    # Find @editor-marked types
registry.types_where(predicate) -> list[type]            # Custom filters
registry.instances(cls) -> Iterator[object]              # Live instance browser
registry.instance_count(cls) -> int                      # Instance count display
registry.describe(cls) -> str                            # Human-readable type description
registry.get(name) -> type                               # Lookup by name
registry.get_name(cls) -> str                            # Name for display
registry.is_registered(cls) -> bool                      # Existence check
registry.set_metadata(cls, key, value)                   # Attach editor metadata
registry.get_metadata(cls, key) -> Any                   # Read editor metadata
```

### 5.2 Tracker — Undo/Redo, Dirty Flags, Subscriptions

```python
from foundation.tracker import tracker

# Undo/Redo (command pattern for editor):
tracker.begin_transaction(name: str) -> None             # Start atomic group
tracker.commit_transaction() -> None                     # Commit to undo stack
tracker.rollback_transaction() -> None                   # Revert all changes in group
tracker.undo() -> bool                                   # Undo last transaction
tracker.redo() -> bool                                   # Redo last undone transaction
tracker.can_undo -> bool                                 # Undo stack non-empty
tracker.can_redo -> bool                                 # Redo stack non-empty
tracker.undo_stack -> list[Transaction]                  # Full undo history
tracker.redo_stack -> list[Transaction]                  # Full redo history
tracker.in_transaction -> bool                           # Currently in transaction

# Dirty flags:
tracker.mark_dirty(obj, field, old, new) -> None         # Called by TrackedDescriptor
tracker.is_dirty(obj) -> bool                            # Any dirty field on obj
tracker.dirty_fields(obj) -> set[str]                    # Which fields are dirty
tracker.all_dirty() -> list[Any]                         # All dirty objects
tracker.mark_clean(obj) -> None                          # Clear dirty flags

# Subscriptions (UI refresh, inspector update):
tracker.on_change(callback) -> None                      # Global: any change anywhere
tracker.on_change(obj, callback) -> None                 # Per-object: changes on specific obj
tracker.on_change(cls, callback) -> None                 # Per-type: changes on any instance of cls
tracker.off_change(callback) -> None                     # Unsubscribe
```

### 5.3 EventLog — Debug Timeline, Causal Chains

```python
from foundation.eventlog import EventLog, Event, Change

event_log = EventLog()

event_log.record(event: Event) -> None                   # Record operation
event_log.events_at(tick: int) -> list[Event]            # Timeline: events at frame N
event_log.events_for_entity(entity_id) -> list[Event]    # Entity history panel
event_log.events_for_operation(op_name) -> list[Event]   # Operation search
event_log.events_caused_by(entity_id) -> list[Event]     # Causal chain: "why did X happen?"
event_log.events_where(**filters) -> list[Event]         # General query
# Event.changes: list[Change] for per-field detail
# Event.root_cause / root_cause_entity for causal chain visualization
# Event.depth for nesting visualization
```

**Tooling views:**
- **HistoryView** -- Scrollable timeline of all operations per frame
- **CausalityView** -- Tree visualization of root_cause -> child operations -> field changes

### 5.4 Mirror — Entity Inspector, Type Browser

```python
from foundation.mirror import mirror, schema_hash, ObjectMirror, ClassMirror, FieldInfo

# Object inspection (inspector panel):
m = mirror(obj)                     # ObjectMirror
m.type_name -> str                  # Display name
m.type_class -> type                # Actual class
m.fields -> dict[str, FieldInfo]    # All fields with metadata
m.methods -> dict[str, MethodInfo]  # All methods
m.get(name) -> Any                  # Read field value
m.set(name, value) -> None          # Write field value (triggers descriptors!)
m.has(name) -> bool                 # Field exists
m.to_dict() -> dict                 # Export all fields
m.get_path("a.b[0].c") -> Any      # Deep path access
m.set_path("a.b[0].c", val)        # Deep path mutation
m.describe() -> str                 # Human-readable summary

# Class inspection (type browser):
cm = mirror(MyClass)                # ClassMirror
cm.fields -> dict[str, FieldInfo]   # Field types and defaults
cm.methods -> dict[str, MethodInfo] # Method signatures

# Schema hashing (hot-reload detection):
schema_hash(cls) -> str             # 16-char hex hash of class schema
# Changed hash = schema migration needed

# FieldInfo metadata keys (for editor widgets):
#   "hidden": bool       -- Hide from inspector
#   "readonly": bool     -- Disable editing
#   "range": (min, max)  -- Slider range for numerics
#   "choices": [...]     -- Dropdown choices
#   "widget": str        -- Custom widget type
#   "label": str         -- Display label override
#   "description": str   -- Tooltip text
#   "transient": bool    -- Skip serialization
#   "replicated": bool   -- Network replicated
#   "authority": str     -- Network authority
#   "version_added": int -- Schema version
```

### 5.5 Bridge — Editor <-> Runtime

```python
from foundation.bridge import (
    get_trinity_registry,            # -> dict[str, Type] of all component types
    create_world_from_trinity,       # -> World pre-populated with Trinity types
    create_shell,                    # -> Shell connected to World
    create_ai_interface,             # -> AIInterface connected to World
    TrinityWorldAdapter,             # Bidirectional sync
)

# TrinityWorldAdapter:
adapter = TrinityWorldAdapter()
adapter.add_instance(component_instance) -> Entity       # Trinity -> ShellLang
adapter.get_instance(entity, ComponentType) -> instance   # ShellLang -> Trinity
adapter.sync_from_foundation_registry()                  # Bulk sync

# Editor live edit flow:
# 1. User edits field in inspector
# 2. Mirror.set(name, value) -> triggers TrackedDescriptor -> Tracker.mark_dirty
# 3. If in play mode: DeltaSync computes delta, pushes to running game
# 4. If in edit mode: Tracker transaction records for undo

# Play/Pause/Step:
# Play:  snapshot editor World -> create runtime World -> run game loop
# Pause: suspend game loop, inspect live state via Mirror
# Step:  advance one frame, inspect results
# Stop:  restore editor World from snapshot
```

### 5.6 ContentStore — Asset Browser, Scene Deduplication

```python
from foundation.content_store import ContentStore, MemoryBackend, FileBackend, ContentDiffer

store = ContentStore(backend=FileBackend(path))
store.store(data) -> hash                  # Content-addressable storage
store.retrieve(hash) -> data               # Retrieve by hash
store.has(hash) -> bool                    # Existence check

# ContentDiffer: hash-based diffing -- O(differences) not O(size)
differ = ContentDiffer(store)
diff = differ.diff(old_hash, new_hash)     # Structural diff

# Tooling usage:
# - Asset deduplication (same texture stored once regardless of references)
# - Session save with structural sharing (identical subtrees share storage)
# - Undo history compression (store snapshots efficiently)
```

### 5.7 DeltaSync — Live Editing

```python
from foundation.delta_sync import DeltaSync, DeltaPatch

sync = DeltaSync()
patch = sync.compute_delta(old_state, new_state) -> DeltaPatch
sync.apply_delta(target_state, patch) -> new_state

# Tooling usage:
# - Push inspector edits to running game in play mode
# - Incremental session saves (only store what changed)
# - Network replication delta compression
```

### 5.8 Provenance — Data Lineage

```python
from foundation.provenance import track_provenance, derivation_tree

@track_provenance
def compute_damage(base, multiplier):
    return base * multiplier

derivation_tree(obj, "damage") -> tree
# Returns full dependency tree: "damage = base(50) * multiplier(1.5)"

# Tooling usage:
# - Inspector "Why is this value X?" feature
# - Debug computed/derived fields
# - Track PCG derivation chains
```

### 5.9 Serializer — Scene Save/Load, Prefabs, Copy/Paste

```python
from foundation.serializer import to_dict, from_dict, to_file, from_file, register_type

# Scene save/load:
data = to_dict(scene_root)                 # Recursive serialization
scene = from_dict(data)                    # Recursive deserialization
to_file(scene_root, Path("level.scene"))   # File I/O
scene = from_file(Path("level.scene"))     # File I/O

# Prefab editing:
# - Serialize prefab to dict
# - Diff against parent prefab (only store overrides)
# - Patch parent changes into child instances

# Copy/paste:
# - to_dict(selected_entities) -> clipboard
# - from_dict(clipboard) -> new entities (deep copy with new IDs)

# Schema migration:
# - schema_hash(cls) embedded in save file header
# - On load: compare hash -> if mismatch, run migration via VersionedDescriptor
```

---

## 6. Architecture Spec Details

### 6.1 Editor Framework

**Application Shell:**
- Main window with docking system (panels can float, dock, tab, split)
- Menu bar (File, Edit, View, Tools, Window, Help)
- Toolbar (common actions, tool selection)
- Status bar (selection info, FPS, memory, build status)

**Viewport System:**
- **Perspective** -- 3D view with depth, FOV control
- **Orthographic** -- 2D views (top/front/side), parallel projection
- **Navigation:** Fly (WASD+mouse), Orbit (around selection), Pan (middle drag), Zoom (scroll/dolly), Focus (frame selected)
- **Render Modes:** Lit (full PBR), Unlit (base color only), Wireframe (edges), Debug views (normals, UV, overdraw, shader complexity, lightmap density, buffer visualizations)
- **Overlays:** Grid, Gizmos, Actor type icons, Statistics

**Editor Modes and Transitions:**

| Mode | Description | Enter | Exit |
|------|-------------|-------|------|
| Edit | Level editing, object placement | Default | Switch to Play/Simulate/Prefab |
| Play | Test game in editor, runtime active | Snapshot World, create runtime | Stop: restore snapshot |
| Simulate | Physics preview without gameplay | Snapshot, run physics only | Stop: restore |
| Prefab | Isolated editing of prefab asset | Open prefab, hide world | Close: return to Edit |

```
Edit <---> Play
Edit <---> Simulate
Edit <---> Prefab
```
Transition: Save State -> Setup Mode -> Active -> Save State -> Exit -> Cleanup

**Plugin System:**

| Plugin Type | Purpose | Extension Points |
|-------------|---------|-----------------|
| Editor Plugins | UI extensions, custom panels | Panel registration, menu items |
| Asset Plugins | New asset types | AssetMeta registration, import/export |
| Tool Plugins | Custom editing tools | Viewport tool registration |
| Importer Plugins | New file format support | Format handler registration |

Lifecycle: Load -> Initialize -> Enable -> [Active] -> Disable -> Unload

### 6.2 Level Editor

**Transform Gizmos:**

| Mode | Function | Key |
|------|----------|-----|
| Select | Pick objects | Q |
| Move | Translate position | W |
| Rotate | Rotation | E |
| Scale | Resize | R |
| Universal | All combined | T |

**Transform Spaces:** World (global axes), Local (object-relative), View (camera-relative), Parent (parent-relative)

**Snapping:** Grid snap (configurable intervals), Vertex snap (mesh vertices), Surface snap (align to surface normal), Pivot snap (object pivots), Increment (fixed rotation/scale steps)

**Pivot Modes:** Center (bounding box center), Origin (object origin point), 3D Cursor (custom position), Individual (per-object)

**Scene Hierarchy:** Tree view (parent-child), Search (by name), Filter (by type/tag), Sort (alphabetical/type), Operations: Parent/Unparent/Group/Duplicate, Visibility: Show/Hide toggle, Lock (prevent selection), Isolate (hide all others)

**Object Placement:** Drag & Drop (from content browser), Spawn Point (at camera), Surface Place (on ground), Paint Brush (brush-based), Distribution: Array (linear), Radial (circular), Random Scatter, Even Spacing

### 6.3 Asset Browser

**Views:** Folder tree, Asset grid (thumbnails), List view (details), Columns (metadata)

**Search & Filter:** Text search (name, path), Type filter (by asset class via `AssetMeta` registry), Tag filter (custom tags), Recent (last modified)

**Reference Management:** Find References (what uses this asset), Dependencies (what this asset uses), Redirectors (track moved assets), Fix Redirectors (resolve references)

**Import Pipeline:**
```
Source File -> Detect Type (AssetMeta.get_for_extension) -> Import Settings -> Process -> Validate -> Engine Asset
```

| Category | Formats |
|----------|---------|
| 3D Models | FBX, glTF, OBJ |
| Textures | PNG, TGA, EXR, PSD |
| Audio | WAV, OGG, MP3 |
| Data | JSON, CSV, XML |

### 6.4 Profiler

**CPU Profiler:**
- Frame time (per-frame total cost, target: 16.67ms at 60fps)
- Call hierarchy (function tree with inclusive/exclusive times)
- Hotspots (most expensive functions)
- Thread view (multi-threaded analysis)
- Flame graph visualization
- Timeline view per-frame
- `@profile` decorator data: call_count, total_ms, min_ms, max_ms, avg_ms

**GPU Profiler:**
- Render pass timing (shadows, G-Buffer, lighting, post-process)
- Draw call analysis (batch counts, state changes)
- Shader cost (GPU time per shader)
- GPU memory (VRAM usage)
- `@gpu_profile` decorator data: category, include_memory
- External: RenderDoc, PIX, Nsight, RGP integration

**Memory Profiler:**
- Total memory / heap / VRAM / pool breakdown
- Memory tags: [Rendering], [Physics], [Audio], [Gameplay]
- Allocation tracking (who allocated what, stack traces, lifetime)
- Leak detection (allocated but never freed)
- Snapshots (point-in-time capture, diff between snapshots)
- `@budget` decorator data: category, max_bytes, warn_at
- `@pooled` decorator data: initial_size, grow_factor, max_size, pool usage

**Network Profiler:**
- Bandwidth (KB/s sent/received)
- Packets (count, per-second)
- RTT (round-trip latency)
- Packet loss percentage
- Per-actor bandwidth breakdown
- Per-property bytes breakdown
- `@network_debug` data: log_packets, simulate_latency, simulate_loss

### 6.5 Debug Draw

**Primitives:**

| Primitive | Description |
|-----------|-------------|
| Line | Single or multi-segment |
| Arrow | Directional indicator |
| Point | Location marker |
| Sphere | Radius visualization |
| Box | AABB/OBB bounds |
| Capsule | Character shapes |
| Cylinder | Volume shapes |
| Cone | Direction + angle |
| Plane | Flat surfaces |
| Mesh | Custom geometry |

**Text Types:** Screen Text (2D overlay), World Text (3D positioned), Billboard (faces camera)

**Draw Options:** Color (RGBA), Duration (seconds, 0 = one frame), Thickness (line width), Depth Test (behind/in front of geometry)

**Overlay Categories:**

| Category | Visualizes |
|----------|------------|
| Physics | Collision shapes, contacts, joints, raycasts, velocities, center of mass |
| Navigation | NavMesh polygons, active paths, off-mesh links, agents, obstacles |
| AI | Perception cones, BT state, EQS results, blackboard, current path |
| Audio | Sound source positions, attenuation spheres, active voices |
| Rendering | Wireframe, unlit, buffer views, overdraw, shader complexity, lightmap density |
| Network | Replication status, ownership, relevancy |

### 6.6 Console

**UI Modes:** Overlay (transparent over game), Full Screen (dedicated view), Mini (small output bar)

**Command Types:**
- Exec commands: `god`, `teleport 100 200 50`, `spawn enemy`, `kill all`
- Console variables: `r.VSync 0`, `p.Gravity -980.0`
- Aliases: `alias noclip "god; fly"`
- Script execution: `exec debug.cfg`

**CVar System:**

| Type | Example |
|------|---------|
| Integer | `r.ShadowQuality 3` |
| Float | `p.Gravity -980.0` |
| Boolean | `r.VSync 1` |
| String | `log.OutputFile "game.log"` |

**CVar Flags:**

| Flag | Description |
|------|-------------|
| ReadOnly | Cannot modify at runtime |
| Cheat | Requires cheats enabled |
| Config | Saved to configuration file |
| Scalability | Part of quality settings |

**Command Categories:** `r.*` rendering, `p.*` physics, `ai.*` AI, `net.*` network, `stat *` statistics, `show *` debug flags

### 6.7 Introspection Tools

```python
from trinity.decorators.ops import decompose, decompose_layered
from trinity.decorators.introspection import primitives, composites, chain, expand

# decompose(cls) -> list[Step]
#   Flat list of every Step (TAG/HOOK/REGISTER/VALIDATE/TRACK/INTERCEPT/DESCRIBE) on a class

# decompose_layered(cls) -> dict
#   {
#     "decorator_steps": [...],
#     "metaclass_steps": [...],
#     "descriptor_steps": {"field_name": [...], ...}
#   }

# expand(cls) -> str
#   Human-readable layered trace (tree format)

# primitives(cls) -> set[Op]          # Which primitive Ops are on this class
# primitives(cls, "field") -> set[Op]  # Field-specific Ops
# composites(cls) -> list[str]         # Which composite decorators were applied
# chain(cls, "field") -> list          # Full descriptor chain for a field
```

### 6.8 Trinity Tools

```python
from trinity.tools.doctor import doctor
from trinity.tools.step_trace import trace

# doctor() -> dict
#   Validates ALL registered Trinity classes against composition rules.
#   Returns {total, passed, failed, errors: {class_name: [error_messages]}}

# trace(cls) -> str
#   Formatted Step trace grouped by layer:
#   [Decorator] (N steps) ...
#   [Descriptor] (N steps) per field ...
#   [Metaclass] (N steps) ...
```

---

## 7. Decorator Stacks

Pre-composed decorator bundles relevant to tooling. Each stack expands to a specific combination of decorators.

### Built-in: @production_component

```python
@production_component(pool_size=128, layout="soa", category="gameplay")
class MyComponent(Component): ...
```
**Expands to:** `@track_changes` + `@budget(category=category)` + `@pooled(initial_size=pool_size)` + `@packed(layout=layout)` + `@component`

### Built-in: @reactive_ui_component

```python
@reactive_ui_component(pool_size=256, batch_delay_ms=16.0)
class UIData(Component): ...
```
**Expands to:** `@production_component(pool_size=pool_size)` + `@observable(notify="batched", batch_delay_ms=batch_delay_ms)` + `@diff(strategy="shallow")` + `@lazy(init_on="first_access")` + `@serializable(format="json")`

### Built-in: @moddable_content

```python
@moddable_content(namespace="mygame", version=1)
class ModdableItem(Component): ...
```
**Expands to:** `@component` + `@moddable(namespace=namespace)` + `@serializable(format="json")` + `@versioned(version=version)` + `@track_changes` + `@observable()`

### Built-in: @saveable_data

```python
@saveable_data(version=1, format="binary")
class SaveData: ...
```
**Expands to:** `@track_changes` + `@versioned(version=version, migrations=migrations)` + `@serializable(format=format)`

### Proposed: @editor_component

```python
# Proposed stack for editor-visible components
@editor_component(category="Gameplay", pool_size=256)
class EditorVisible(Component): ...
```
**Would expand to:** `@component` + `@editor(category=category)` + `@track_changes` + `@observable()` + `@serializable(format="binary")` + `@pooled(initial_size=pool_size)`

### Proposed: @profiled_system

```python
# Proposed stack for fully profiled systems
@profiled_system(phase="render", name="MyRenderSystem")
class MySystem(System): ...
```
**Would expand to:** `@system(phase=phase)` + `@profile(name=name)` + `@trace(level="debug")`

### Proposed: @debuggable_entity

```python
# Proposed stack for fully debuggable entities
@debuggable_entity(category="AI")
class AIAgent(Component): ...
```
**Would expand to:** `@component` + `@editor(category=category)` + `@track_changes` + `@debug_draw()` + `@inspector(category=category)` + `@serializable(format="binary")`

---

## 8. TODO Checklist

### Section 13: Tooling Layer (from GAME_ENGINE_INTEGRATION_TODO.md)

#### 13.1 Editor Framework
- [ ] Implement editor application shell (window, menus, panels, docking)
- [ ] Implement viewport rendering (3D scene view, camera controls)
- [ ] Implement transform gizmos (translate, rotate, scale)
- [ ] Implement selection system (click, box select, multi-select)
- [ ] Wire Foundation Inspector -> property panel for selected objects
- [ ] Wire Foundation Shell -> editor console
- [ ] Wire Foundation Mirror -> object property display

#### 13.2 Level Editor
- [ ] Implement entity placement, duplication, deletion in viewport
- [ ] Implement snapping (grid, surface, vertex, edge)
- [ ] Implement undo/redo via Foundation Tracker.undo()/redo()
- [ ] Wire Foundation EventLog -> editor action history

#### 13.3 Asset Tools
- [ ] Implement content browser (asset listing, search, thumbnails)
- [ ] Implement asset import pipeline (mesh, texture, audio, animation)
- [ ] Wire AssetMeta registry -> content browser type filtering

#### 13.4 FlowForge Integration (Visual Scripting)
- [ ] Connect FlowForge backend to editor (embed visual scripting panel)
- [ ] Wire FlowForge trinity_adapter -> live decorator introspection in editor
- [ ] Implement FlowForge node execution (run visual scripts at runtime)
- [ ] Wire Foundation Capabilities -> FlowForge sandbox permissions

#### 13.5 Material Editor
- [ ] Implement node-based material editor
- [ ] Implement material preview (sphere, plane, mesh)
- [ ] Wire material parameters to ValidatedDescriptor for live editing

#### 13.6 Animation Tools
- [ ] Implement timeline/sequencer
- [ ] Implement animation curve editor
- [ ] Implement animation preview

#### 13.7 Build & Cook
- [ ] Implement build pipeline (source -> cooked -> packaged)
- [ ] Implement asset cooking (platform-specific optimization)
- [ ] Implement incremental builds
- [ ] Wire @build_deploy decorators -> build configuration

#### 13.8 Version Control
- [ ] Implement VCS integration (lock/unlock, diff, merge)
- [ ] Wire Foundation ContentStore -> content-addressable asset versioning

### Section 14: Debug Layer (from GAME_ENGINE_INTEGRATION_TODO.md)

#### 14.1 Logging
- [ ] Implement logging system (categories, levels, output targets)
- [ ] Implement log filtering (per-category, per-level)
- [ ] Implement log output to file, console, network
- [ ] Wire Foundation EventLog -> structured log entries

#### 14.2 Console System
- [ ] Implement in-game console (CVars, commands)
- [ ] Implement CVar system (typed variables, change callbacks)
- [ ] Wire Foundation Shell -> console command execution
- [ ] Wire Foundation Tracker.on_change -> CVar change notifications

#### 14.3 Visual Debugging
- [ ] Implement debug draw (lines, boxes, spheres, text, arrows)
- [ ] Implement debug overlays (wireframe, collision, navmesh, audio)
- [ ] Wire @debug_cheat decorators -> cheat command registration

#### 14.4 Profiling
- [ ] Implement CPU profiler (hierarchical timer, flame graph)
- [ ] Implement GPU profiler (timestamp queries, pipeline stats)
- [ ] Implement memory profiler (allocation tracking, leak detection)
- [ ] Implement network profiler (bandwidth, latency, packet loss)
- [ ] Wire Foundation EventLog -> profiling event capture
- [ ] Wire @debug_safety decorators -> read/write tracking

#### 14.5 Crash Handling
- [ ] Implement crash reporter (minidump, callstack, state capture)
- [ ] Implement assertion system (verify, check, ensure)
- [ ] Wire Foundation Session -> crash state capture

#### 14.6 Replay & Recording
- [ ] Implement input replay (record inputs -> deterministic replay)
- [ ] Implement state recording (periodic snapshots + deltas)
- [ ] Wire Foundation EventLog -> replay event source
- [ ] Wire Foundation DeltaSync -> efficient replay storage
- [ ] Wire @replay decorators -> replay system configuration

#### 14.7 Testing Framework
- [ ] Implement in-engine test runner (unit, functional, integration, stress)
- [ ] Implement automated testing (bots, scenarios)
- [ ] Wire Foundation Shell -> test execution from console

---

## 9. Directory Structure

```
engine/tooling/
├── __init__.py                          # Public API exports
├── TOOLING_CONTEXT.md                   # THIS FILE
├── editor/
│   ├── __init__.py
│   ├── app_shell.py                     # Main window, menus, docking, panels
│   ├── viewport.py                      # 3D/2D viewport, camera controls, render modes
│   ├── selection.py                     # Click, box, multi-select
│   ├── gizmos.py                        # Transform gizmos (translate/rotate/scale)
│   ├── modes.py                         # Edit/Play/Simulate/Prefab mode management
│   ├── commands.py                      # Command pattern: all editor actions
│   ├── shortcuts.py                     # Keyboard shortcut mapping
│   ├── preferences.py                   # User settings, layout persistence
│   └── plugins.py                       # Plugin lifecycle, extension points
├── leveleditor/
│   ├── __init__.py
│   ├── placement.py                     # Drag-drop, surface place, paint brush
│   ├── snapping.py                      # Grid, surface, vertex, edge, pivot snapping
│   ├── hierarchy.py                     # Scene tree view, search, filter, parent/group
│   ├── alignment.py                     # Align to ground, normal, object, view
│   └── distribution.py                  # Array, radial, scatter, even spacing
├── assettools/
│   ├── __init__.py
│   ├── content_browser.py               # Folder tree, thumbnails, search, filter
│   ├── import_pipeline.py               # Source -> detect -> settings -> process -> validate
│   ├── reference_manager.py             # Find refs, dependencies, redirectors
│   └── asset_validation.py              # Reference/size/naming/quality checks
├── material_editor/
│   ├── __init__.py
│   ├── material_graph.py                # Node-based material authoring
│   ├── material_preview.py              # Preview rendering (sphere/cube/plane)
│   └── material_nodes.py                # Input, math, output nodes
├── animation_tools/
│   ├── __init__.py
│   ├── sequencer.py                     # Timeline, tracks, keyframes
│   ├── curve_editor.py                  # Animation curve editing
│   ├── skeleton_editor.py               # Bone hierarchy, sockets, retarget
│   ├── montage_editor.py                # Sections, links, slots
│   └── anim_graph_editor.py             # State machine, blend nodes, IK
├── visual_scripting/
│   ├── __init__.py
│   ├── graph_editor.py                  # FlowForge graph canvas
│   ├── node_library.py                  # Node type registry
│   ├── blueprint_runtime.py             # Visual script execution
│   └── blueprint_debug.py              # Breakpoints, watch, step-through
├── profiling/
│   ├── __init__.py
│   ├── cpu_profiler.py                  # Flame graph, timeline, hotspots
│   ├── gpu_profiler.py                  # Render pass timing, shader cost
│   ├── memory_profiler.py               # Allocation tracking, leak detection, snapshots
│   └── network_profiler.py              # Bandwidth, RTT, packet loss
├── debug/
│   ├── __init__.py
│   ├── debug_draw.py                    # Primitives, text, draw options
│   ├── debug_overlays.py                # Physics, nav, AI, audio, rendering overlays
│   ├── debug_camera.py                  # Free camera, eject camera
│   └── gameplay_debug.py                # Cheats, time control, debug camera
├── console/
│   ├── __init__.py
│   ├── console_ui.py                    # Overlay/fullscreen/mini modes
│   ├── console_commands.py              # Command registration and dispatch
│   ├── cvar_system.py                   # Typed variables, flags, change callbacks
│   └── command_history.py               # Command history, autocomplete
├── logging/
│   ├── __init__.py
│   ├── log_system.py                    # Categories, levels, structured logging
│   ├── log_targets.py                   # File, console, network, IDE output
│   └── log_filter.py                    # Level/category/keyword/regex filtering
├── crash/
│   ├── __init__.py
│   ├── crash_reporter.py                # Minidump, callstack, state capture
│   ├── assertions.py                    # check, verify, ensure, checkSlow
│   └── crash_upload.py                  # Remote reporting (Sentry/Backtrace/custom)
├── replay/
│   ├── __init__.py
│   ├── input_recorder.py                # Record/replay input streams
│   ├── state_recorder.py                # Periodic snapshots + deltas
│   └── replay_playback.py              # Playback controls, camera modes
├── testing/
│   ├── __init__.py
│   ├── test_runner.py                   # Unit, functional, integration, stress
│   ├── automation_framework.py          # Bots, scenarios, CI integration
│   └── test_reporting.py                # Results, coverage, notifications
├── build/
│   ├── __init__.py
│   ├── build_pipeline.py                # Compile -> cook -> package -> deploy
│   ├── cook_system.py                   # Asset cooking, incremental, derived data cache
│   └── packaging.py                     # Executable + paks + config, compression/encryption
├── vcs/
│   ├── __init__.py
│   ├── vcs_integration.py               # Git/Perforce/SVN abstraction
│   ├── file_operations.py               # Checkout, commit, revert, update
│   └── merge_tools.py                   # Branch, merge, conflict resolution
├── terrain/
│   ├── __init__.py
│   ├── sculpt_tools.py                  # Sculpt, smooth, flatten, erosion, noise, ramp
│   ├── paint_tools.py                   # Layer painting, holes, alpha
│   └── terrain_import.py                # Heightmap/weight map import/export
├── localization/
│   ├── __init__.py
│   ├── loc_dashboard.py                 # Languages, progress, issues
│   ├── loc_workflow.py                  # Gather, export, import, compile
│   └── loc_preview.py                   # Language switch, pseudo-loc, visual check
└── automation/
    ├── __init__.py
    ├── commandlets.py                   # CLI cook, build, import, custom
    ├── python_api.py                    # Python editor automation API
    └── ci_integration.py                # Build farm, cloud agents, reporting
```

---

## 10. Canonical Examples

### Example 1: Editor-Visible Component with Inspector Integration

```python
from typing import Annotated
from trinity.base import Component
from trinity.types import Vec3
from trinity.decorators.ecs_core import component
from trinity.decorators.dev import editor
from trinity.decorators.debug_safety import track_changes
from trinity.decorators.debug_cheat import inspector
from trinity.descriptors.tracking import TrackedDescriptor
from trinity.descriptors.validation import RangeDescriptor
from trinity.descriptors.observable import ObservableDescriptor

@component
@editor(category="Gameplay")
@track_changes
@inspector(category="Gameplay")
class Health(Component):
    """Health component visible in editor inspector."""
    current: Annotated[float,
        TrackedDescriptor(),
        RangeDescriptor(min_val=0, max_val=1000, clamp=True),
        ObservableDescriptor(),
        {"label": "Current HP", "widget": "slider", "range": (0, 1000)},
    ] = 100.0

    max_hp: Annotated[float,
        TrackedDescriptor(),
        RangeDescriptor(min_val=1, max_val=10000, clamp=True),
        {"label": "Max HP", "widget": "slider", "range": (1, 10000)},
    ] = 100.0

    is_invulnerable: Annotated[bool,
        TrackedDescriptor(),
        {"label": "Invulnerable", "description": "Prevents all damage"},
    ] = False
```

### Example 2: Profiled System with GPU Profiling

```python
from trinity.decorators.ecs_core import system
from trinity.decorators.scheduling import parallel
from trinity.decorators.dev import profile, gpu_profile, trace
from trinity.decorators.debug_safety import reads, writes

@system(phase="render")
@parallel(chunk_size=512)
@profile(name="GPUCullingSystem", warn_ms=2.0)
@gpu_profile(category="culling", include_memory=True)
@trace(level="debug")
@reads("Transform", "MeshRenderer")
@writes("IndirectDrawBuffer")
class GPUCullingSystem:
    """GPU frustum + occlusion culling. Profiled on both CPU and GPU."""

    def execute(self, dt: float):
        # @profile wraps this: records call_count, total_ms, min/max/avg
        # @gpu_profile records GPU timestamp queries
        # @trace logs ENTER/EXIT to trinity.trace logger
        pass
```

### Example 3: Hot-Reloadable System with Preserve/Reinitialize

```python
from trinity.decorators.ecs_core import system
from trinity.decorators.dev import reloadable

@system(phase="gameplay")
@reloadable(
    preserve=["_cache", "_config"],        # Keep these across reload
    reinitialize=["_temp_buffer"],         # Reset these after reload
    validate=lambda cls: hasattr(cls, "execute"),  # Post-reload check
)
class AISystem:
    """AI system that can be hot-reloaded during development."""
    _cache: dict = {}
    _config: dict = {"aggression": 0.5}
    _temp_buffer: list = []

    def execute(self, dt: float):
        pass

# Hot-reload flow:
# 1. File watcher detects AISystem source change
# 2. Mirror.schema_hash(AISystem) compared to stored hash
# 3. If changed: SystemMeta.reload_system(AISystem)
#    a. Preserve _cache and _config from old instance
#    b. Re-import module, get new class
#    c. ComponentMeta._process_fields() + _install_descriptors()
#    d. Create new instance, copy preserved fields
#    e. Reinitialize _temp_buffer to default
#    f. Run validate callback
# 4. SystemMeta.hot_reload() rebuilds phase order
```

### Example 4: Undo/Redo Transaction for Entity Manipulation

```python
from foundation.tracker import tracker

def move_entities(entities: list, offset):
    """Move multiple entities with a single undoable action."""
    tracker.begin_transaction("Move Entities")
    try:
        for entity in entities:
            # Each setattr triggers TrackedDescriptor -> tracker.mark_dirty
            entity.transform.position.x += offset.x
            entity.transform.position.y += offset.y
            entity.transform.position.z += offset.z
        tracker.commit_transaction()
    except Exception:
        tracker.rollback_transaction()
        raise

# Later in editor:
tracker.undo()   # Reverts ALL entity positions atomically
tracker.redo()   # Re-applies ALL entity positions atomically

# Undo stack inspection:
for txn in tracker.undo_stack:
    print(f"{txn.name}: {len(txn.changes)} changes at {txn.timestamp}")
```

### Example 5: Debug Draw for Physics Collision Shapes

```python
from trinity.decorators.ecs_core import system
from trinity.decorators.dev import profile
from trinity.decorators.debug_cheat import debug_draw

@system(phase="post_physics")
@profile(name="PhysicsDebugDraw")
@debug_draw(color=(0, 255, 0), duration=0.0, depth_test=True)
class PhysicsDebugDrawSystem:
    """Draw collision shapes as wireframes for debugging."""

    def execute(self, dt: float):
        for entity in self.query("RigidBody", "Collider"):
            collider = entity.collider
            if collider.shape == "box":
                self.draw_box(entity.transform.position, collider.half_extents)
            elif collider.shape == "sphere":
                self.draw_sphere(entity.transform.position, collider.radius)
            elif collider.shape == "capsule":
                self.draw_capsule(entity.transform.position, collider.radius, collider.height)
```

### Example 6: Console Command (CVar + Cheat)

```python
from trinity.decorators.debug_cheat import cheat

@cheat(name="god", category="player", requires_confirmation=False)
def god_mode():
    """Toggle god mode (invulnerability)."""
    player = get_player()
    player.health.is_invulnerable = not player.health.is_invulnerable
    return f"God mode: {'ON' if player.health.is_invulnerable else 'OFF'}"

@cheat(name="sethealth", category="player", requires_confirmation=False)
def set_health(value: float = 100.0):
    """Set player health to a specific value."""
    player = get_player()
    player.health.current = value
    return f"Health set to {value}"

# CVar example (runtime):
# > r.ShadowQuality 3
# > god
# > sethealth 999
```

### Example 7: Asset Browser Type Filtering via Registry

```python
from foundation.registry import registry
from trinity.metaclasses.asset_meta import AssetMeta

def get_assets_for_filter(filter_type: str) -> list[type]:
    """Content browser type filter using Foundation Registry."""
    if filter_type == "all":
        return registry.all_types()
    elif filter_type == "components":
        return registry.types_with_decorator("component")
    elif filter_type == "editor_visible":
        return registry.types_with_decorator("editor")
    elif filter_type == "by_extension":
        return [AssetMeta.get_for_extension(ext) for ext in [".png", ".fbx", ".wav"]]
    elif filter_type == "custom":
        return registry.types_where(lambda cls: hasattr(cls, "_editor") and not cls._editor_hidden)
    return []

def get_asset_info(asset_type: type) -> dict:
    """Get asset info for content browser display."""
    return {
        "name": registry.get_name(asset_type),
        "instance_count": registry.instance_count(asset_type),
        "description": registry.describe(asset_type),
    }
```

### Example 8: Entity Inspector via Mirror

```python
from foundation.mirror import mirror, schema_hash, FieldInfo

def render_inspector(selected_entity):
    """Render inspector panel for a selected entity using Mirror."""
    m = mirror(selected_entity)

    print(f"Type: {m.type_name}")
    print(f"Schema Hash: {schema_hash(m.type_class)}")
    print()

    for name, field_info in m.fields.items():
        meta = field_info.metadata
        if meta.get("hidden", False):
            continue

        label = meta.get("label", name)
        value = m.get(name)
        readonly = meta.get("readonly", False)
        widget = meta.get("widget", "default")

        if widget == "slider" and "range" in meta:
            min_val, max_val = meta["range"]
            print(f"  {label}: [{min_val}--({value})--{max_val}] {'(readonly)' if readonly else ''}")
        elif widget == "color":
            print(f"  {label}: Color({value})")
        else:
            print(f"  {label}: {value!r} {'(readonly)' if readonly else ''}")

    # Edit a field (triggers full descriptor chain):
    if not readonly:
        m.set("current", 75.0)
        # This calls TrackedDescriptor -> tracker.mark_dirty -> observer callbacks
```

### Example 9: Live Edit via DeltaSync

```python
from foundation.delta_sync import DeltaSync
from foundation.mirror import mirror

sync = DeltaSync()

def live_edit_field(entity, field_name, new_value):
    """Push an inspector edit to the running game via DeltaSync."""
    m = mirror(entity)
    old_state = m.to_dict()

    # Apply edit
    m.set(field_name, new_value)

    new_state = m.to_dict()

    # Compute minimal delta
    patch = sync.compute_delta(old_state, new_state)

    # In play mode: push patch to runtime game World
    if editor_is_in_play_mode():
        runtime_entity = get_runtime_entity(entity)
        runtime_state = mirror(runtime_entity).to_dict()
        sync.apply_delta(runtime_state, patch)
```

### Example 10: Data Lineage Query via Provenance

```python
from foundation.provenance import track_provenance, derivation_tree

@track_provenance
def compute_effective_damage(base_damage: float, multiplier: float, armor: float) -> float:
    """Compute final damage with full provenance tracking."""
    raw = base_damage * multiplier
    reduced = max(0, raw - armor)
    return reduced

# In inspector "Why?" panel:
tree = derivation_tree(entity, "last_damage_dealt")
# Returns:
# last_damage_dealt = 35.0
#   = compute_effective_damage(base_damage=50.0, multiplier=1.5, armor=40.0)
#     base_damage = WeaponComponent.damage (50.0)
#     multiplier = BuffSystem.get_multiplier() (1.5)
#     armor = ArmorComponent.rating (40.0)
```

---

## 11. Integration Patterns

### Pattern 1: Tooling Observes All Layers via Foundation

```
Every Engine Layer (Platform, Core, Resource, Simulation, Animation,
                    Rendering, Audio, Gameplay, UI, Networking, World)
    |
    | (registers types via metaclasses -> Foundation Registry)
    | (tracks changes via descriptors -> Foundation Tracker)
    | (records operations via @trace -> Foundation EventLog)
    | (reflects metadata via annotations -> Foundation Mirror)
    |
    v
TOOLING reads Foundation to observe, inspect, and control everything
```

The tooling layer never imports from engine layers directly. It accesses everything through Foundation.

### Pattern 2: Editor Data Flow

```
User Action (click/drag/type in UI)
    -> Editor Command (Command pattern, reversible)
        -> Foundation Mirror.set(name, value)
            -> TrackedDescriptor.__set__()
                -> ValidatedDescriptor (clamp/reject)
                -> StorageDescriptor (store value)
                -> TrackedDescriptor.post_set()
                    -> Foundation Tracker.mark_dirty(obj, field, old, new)
                    -> Foundation EventLog.add_change_to_current_event()
                -> ObservableDescriptor.post_set()
                    -> UI refresh callbacks fire
        -> Tracker transaction commit
            -> Undo stack updated
```

### Pattern 3: Play/Pause/Step via Snapshot/Restore

```
EDIT MODE:
    Editor World (live entities, inspector, placement)

ENTER PLAY:
    1. Serializer.to_dict(editor_world) -> snapshot (stored in ContentStore)
    2. Create runtime World from snapshot
    3. Start game loop
    4. Editor UI switches to play-mode panels (console, profiler, debug draw)

PAUSE:
    5. Suspend game loop
    6. Mirror still works on runtime entities (inspect live state)
    7. DeltaSync can push edits to paused runtime

STEP:
    8. Advance exactly one frame
    9. Inspect results via Mirror/EventLog

STOP:
    10. Destroy runtime World
    11. Serializer.from_dict(snapshot) -> restore editor World
    12. Editor UI switches back to edit-mode panels
```

### Pattern 4: Hot-Reload via Schema Hash Detection

```
1. File watcher detects source file change
2. For each class in changed module:
   a. old_hash = stored schema_hash
   b. Re-import module (new class definition)
   c. new_hash = schema_hash(new_class)
   d. If old_hash != new_hash:
      - Schema migration needed
      - Serialize all instances of old class
      - Register new class with ComponentMeta
      - Deserialize instances into new class (with migration)
   e. If old_hash == new_hash:
      - Logic-only change (no field changes)
      - SystemMeta.reload_system(new_class) if system
      - Swap class in registry
3. @reloadable preserve/reinitialize fields handled
4. SystemMeta.hot_reload() rebuilds dependency graph
5. EventLog records reload event for debugging
```

### Pattern 5: FlowForge Visual Scripting via Foundation Capabilities

```
FlowForge Graph Editor
    -> User creates/connects nodes
    -> FlowForge AST (trinity_adapter reads Trinity metadata)
    -> Foundation Capabilities check: can this script access GPU? write entities? call RPCs?
    -> If permitted: generate Python code from AST
    -> Code runs through Foundation Bridge -> normal descriptor chains
    -> All changes tracked, undoable, observable
```

### Pattern 6: Debug Stripping (DEBUG_BUILD Conditional)

```python
# Debug-only decorators and descriptors are stripped in shipping builds:
# - @debug_draw, @cheat, @inspector -> no-ops
# - ProfiledDescriptor, LoggedDescriptor, WatchedDescriptor -> not installed
# - Console commands with Cheat flag -> removed
# - Debug overlay systems -> not registered

# Implementation: conditional compilation via build config
# if DEBUG_BUILD:
#     install debug descriptors
#     register cheat commands
#     enable debug draw
# else:
#     skip all debug decorators (they become identity functions)
#     strip debug descriptor chains
```

---

## 12. Quick Reference Tables

### All Tooling Decorators

| Decorator | Module | Tier | Key Params | Purpose |
|-----------|--------|------|------------|---------|
| @profile | dev | 6 | name, warn_ms, track_allocations | CPU timing |
| @gpu_profile | dev | 6 | category, include_memory | GPU timing |
| @trace | dev | 6 | level | Execution tracing |
| @reloadable | dev | 6 | enabled, preserve, reinitialize, validate | Hot-reload |
| @editor | dev | 6 | category, hidden | Editor visibility |
| @test | dev | 6 | cases, fuzz, property_based | Test config |
| @bench | dev | 6 | iterations, warmup | Benchmarking |
| @invariant | dev | 6 | check, when | Runtime invariants |
| @deprecated | dev | 6 | since, replacement, remove_in | Deprecation |
| @cheat | debug_cheat | - | name, category, requires_confirmation | Cheat commands |
| @debug_draw | debug_cheat | - | color, duration, depth_test | Visual debug |
| @inspector | debug_cheat | - | category, readonly, range | Inspector config |
| @reads | debug_safety | 10 | *components | Read access decl |
| @writes | debug_safety | 10 | *components | Write access decl |
| @trace_stack | debug_safety | 10 | depth, show_decorator_chain | Error traces |
| @track_changes | debug_safety | 11 | fields | Change tracking |
| @network_debug | debug_extended | 51 | log_packets, simulate_latency, simulate_loss | Network debug |
| @automation_test | debug_extended | 51 | category, timeout_seconds, required_features | Automation |

### Foundation Systems and Their Tooling Roles

| Foundation System | Module | Tooling Role |
|-------------------|--------|-------------|
| Registry | `foundation/registry.py` | Type browsing, instance counting, decorator filtering |
| Tracker | `foundation/tracker.py` | Undo/redo, dirty flags, change subscriptions |
| EventLog | `foundation/eventlog.py` | Debug timeline, causal chain analysis |
| Mirror | `foundation/mirror.py` | Inspector property display, schema hashing |
| Bridge | `foundation/bridge.py` | Editor-runtime communication, World sync |
| Serializer | `foundation/serializer.py` | Scene save/load, prefab editing, copy/paste |
| ContentStore | `foundation/content_store.py` | Asset deduplication, snapshot storage |
| DeltaSync | `foundation/delta_sync.py` | Live editing, incremental saves |
| Provenance | `foundation/provenance.py` | Data lineage ("why is this value X?") |
| Inspector | `foundation/inspector.py` | Property panel views, UI context |
| Capabilities | `foundation/capabilities.py` | Mod sandboxing, FlowForge permissions |

### Editor Modes and Transitions

| From | To | Action | State Management |
|------|----|--------|-----------------|
| Edit | Play | Click Play | Snapshot World, create runtime |
| Edit | Simulate | Click Simulate | Snapshot World, run physics only |
| Edit | Prefab | Open Prefab | Hide world, load prefab |
| Play | Edit | Click Stop | Restore World from snapshot |
| Simulate | Edit | Click Stop | Restore World from snapshot |
| Prefab | Edit | Close Prefab | Restore world visibility |

### Profiler Categories

| Category | Source | Metrics |
|----------|--------|---------|
| CPU | @profile, EventLog | call_count, total_ms, min_ms, max_ms, avg_ms |
| GPU | @gpu_profile | category, pass_time, draw_calls, shader_cost, VRAM |
| Memory | @budget, @pooled | category, used_bytes, max_bytes, pool_stats |
| Network | @network_debug | bandwidth, RTT, packet_loss, per_actor, per_property |

### Debug Draw Primitive Types

| Primitive | Params | Default Duration |
|-----------|--------|-----------------|
| Line | start, end, color, thickness | 0 (one frame) |
| Arrow | start, end, color, head_size | 0 |
| Point | position, color, size | 0 |
| Sphere | center, radius, color, segments | 0 |
| Box | center, half_extents, color, rotation | 0 |
| Capsule | center, radius, height, color | 0 |
| Cylinder | center, radius, height, color | 0 |
| Cone | apex, direction, angle, length, color | 0 |
| Plane | center, normal, size, color | 0 |
| Text (screen) | position_2d, text, color, size | 0 |
| Text (world) | position_3d, text, color, size | 0 |
| Text (billboard) | position_3d, text, color, size | 0 |

### CVar Types and Flags

| Type | Python Type | Example |
|------|-------------|---------|
| Integer | int | `r.ShadowQuality 3` |
| Float | float | `p.Gravity -980.0` |
| Boolean | bool | `r.VSync 1` |
| String | str | `log.OutputFile "game.log"` |

| Flag | Value | Description |
|------|-------|-------------|
| ReadOnly | 0x01 | Cannot modify at runtime |
| Cheat | 0x02 | Requires cheats enabled |
| Config | 0x04 | Saved to configuration file |
| Scalability | 0x08 | Part of quality settings |

### FieldInfo Metadata Keys (Editor Widgets)

| Key | Type | Description |
|-----|------|-------------|
| `hidden` | bool | Hide from inspector |
| `readonly` | bool | Disable editing |
| `range` | (min, max) | Numeric slider range |
| `choices` | list | Dropdown choices |
| `widget` | str | Custom widget type ("slider", "color", "dropdown", "text", "checkbox") |
| `label` | str | Display label override |
| `description` | str | Tooltip text |
| `transient` | bool | Skip serialization |
| `replicated` | bool | Network replicated |
| `authority` | str | Network authority |
| `interpolated` | str | Interpolation mode |
| `version_added` | int | Schema version when added |
| `serialize_as` | str | Serialization format override |
| `required` | bool | Must have a value |
| `validator` | Callable | Custom validation function |

### Trinity Introspection Functions

| Function | Signature | Returns |
|----------|-----------|---------|
| `decompose` | `(cls: type) -> list[Step]` | Flat list of all Steps on a class |
| `decompose_layered` | `(cls: type) -> dict` | Steps grouped by layer (decorator/metaclass/descriptor) |
| `expand` | `(cls: type) -> str` | Human-readable layered trace |
| `primitives` | `(cls, field=None) -> set[Op]` | Which primitive Ops are present |
| `composites` | `(cls, field=None) -> list[str]` | Which composite decorators were applied |
| `chain` | `(cls, field: str) -> list` | Full descriptor chain for a field |
| `validate_steps` | `(steps: list[Step]) -> dict` | Validate composition rules, return errors |
| `validate_combination` | `(ops: list[Op]) -> dict` | Check if Op combination is valid |
| `all_rules` | `() -> list[Rule]` | All composition rules |
| `doctor` | `() -> dict` | Validate all registered classes |
| `trace` | `(cls: type) -> str` | Formatted Step trace by layer |

### The 7 Primitive Ops

| Op | Signature | Effect |
|----|-----------|--------|
| TAG | `(target, key, value)` | Attach metadata; queryable via Mirror |
| HOOK | `(event, callback)` | Attach callback to lifecycle event |
| REGISTER | `(class, registry)` | Add class to named registry |
| DESCRIBE | `(class)` | Extract schema from annotations |
| TRACK | `(field)` | Install TrackedDescriptor; notify Tracker on change |
| VALIDATE | `(field, constraint)` | Install ValidatedDescriptor; check constraint on set |
| INTERCEPT | `(field, get, set, delete)` | Wrap field access with custom logic |
