# DEBUG_CONTEXT.md — Engine Debug & Diagnostics Layer

> **Purpose:** Collect ALL Trinity decorators, metaclasses, descriptors, Foundation
> integration points, architecture specs, decorator stacks, and canonical usage examples
> needed to implement `engine/debug/`. This document is the single source of truth —
> implementation should require zero external references.

**Implementation Status:**
| Layer | Status | Notes |
|-------|--------|-------|
| Python | ✅ COMPLETE | Profiling, hot-reload, logging, diagnostics |
| Rust | ⚠️ 50% | GAPSET_20_CROSS_CUTTING — Debug utilities |
| Wired | ⚠️ Partial | Profiling hooks connected |

*See `docs/STATUS.md` for current progress.*

---

## 1. Architecture Summary

The **Debug** layer (`engine/debug/`) is a **cross-cutting concern** that interfaces
with ALL other engine layers, providing observability, control, and validation at
runtime. It is NOT a layer in the traditional stack — it wraps around and penetrates
through every other layer.

**Debug provides 8 subsystems:**
- **Logging** — Categorized, leveled, structured logging with multiple output targets
- **Console** — In-game console with CVars, commands, autocomplete, scripting
- **Visual Debugging** — Debug draw (lines, spheres, boxes, text), overlays, gizmos, render views
- **Profiling** — CPU profiler, GPU profiler, memory profiler, network profiler, stats system
- **Crash Handling** — Crash capture (minidump, callstack, state), assertions, crash reporting
- **Replay & Recording** — Input recording, state recording, network recording, playback controls
- **Testing** — Unit, integration, functional, automation, stress testing, in-engine test runner
- **Debug Tools** — Cheats, time control, debug camera, AI/physics/network debugging

**Position in Architecture:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                     DEBUG & DIAGNOSTICS                              │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Logging │ Console │ Visual │ Profiling │ Crash │ Replay │ Test ││
│  └─────────────────────────────────────────────────────────────────┘│
│        ↕           ↕         ↕         ↕         ↕         ↕       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ GAMEPLAY │ UI │ AUDIO │ RENDERING │ ANIMATION │ SIMULATION     ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ CORE SYSTEMS (engine/core + engine/common)                     ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │ PLATFORM │ FOUNDATION + TRINITY                                ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

**Debug is the heaviest consumer of Foundation:**
- **EventLog** → operation history, causal chains, profiling, replay
- **Tracker** → change detection, undo/redo, dirty flags
- **Mirror** → runtime introspection, inspector, property panels
- **Registry** → type discovery, instance tracking
- **Bridge/Shell** → live REPL debugging
- **Serializer** → crash state capture, session snapshots
- **ContentStore** → efficient snapshot storage
- **DeltaSync** → incremental replay storage
- **Provenance** → "why did this happen?" debugging

**Build Configurations:**

| Config | Optimization | Symbols | Assertions | Debug Features | Use Case |
|--------|-------------|---------|------------|----------------|----------|
| DEBUG | -O0 | Full | All | All active | Development |
| DEVELOPMENT | -O2 | Included | Enabled | Available | Daily work |
| TEST | -O2 | Included | Enabled | + Automation hooks | CI/CD |
| SHIPPING | -O3 | None | Disabled | Stripped | Release |

---

## 2. Trinity Decorators of Interest

### 2.1 Dev/Optimization Decorators (Tier: DEV) — 9 decorators

| Decorator | Parameters | Op Types | Purpose |
|-----------|-----------|----------|---------|
| `@profile` | `name=None, gpu=False, warn_ms=None, track_allocations=False` | TAG, HOOK, REGISTER | CPU/GPU profiling with timing stats |
| `@gpu_profile` | `category (required), include_memory=False` | TAG, HOOK, REGISTER | GPU profiling per category |
| `@trace` | `level="debug"` (debug/info/warn) | TAG, HOOK, REGISTER | Execution tracing with log level |
| `@reloadable` | `enabled=True, preserve=[], reinitialize=[], validate=None` | TAG, REGISTER | Hot reload support |
| `@editor` | `category="General", hidden=False` | TAG, REGISTER | Editor integration metadata |
| `@test` | `cases=[], fuzz=False, property_based=False` | TAG, REGISTER | Test case declaration |
| `@bench` | `iterations=1000, warmup=100` | TAG, REGISTER | Benchmark configuration |
| `@invariant` | `check (required callable), when="debug"` | TAG, VALIDATE, REGISTER | Runtime invariant checking |
| `@deprecated` | `since (required), replacement=None, remove_in=None` | TAG, HOOK, REGISTER | Deprecation warnings |

**Detailed signatures:**

**`@profile(name=None, warn_ms=None)`**
- Sets: `_profiled=True`, `_profile_name`
- Adds: `profile_stats()` → `{name, call_count, total_ms, min_ms, max_ms, avg_ms}`
- Adds: `profile_reset()` — clears stats
- Runtime: Wraps function with `time.perf_counter_ns()` timing, warns if `warn_ms` exceeded

**`@gpu_profile(category, include_memory=False)`**
- Sets: `_gpu_profiled=True`, `_gpu_profile_category`, `_gpu_profile_include_memory`
- Adds: `gpu_stats()` → stub dict
- Runtime: Tags for GPU profiler integration

**`@trace(level="debug")`**
- Valid levels: "debug", "info", "warn"
- Sets: `_traced=True`, `_trace_level`
- Runtime: Logs ENTER/EXIT/ERROR messages via `trinity.trace` logger

**`@invariant(check, when="debug")`**
- Unique: False (accumulates in `_invariants` list)
- Valid when: "debug", "always"
- Sets: `_invariants=[{"check": fn, "when": "debug"}, ...]`
- Runtime: Enforced at runtime in debug builds; "always" checks run in all builds

**`@deprecated(since, replacement=None, remove_in=None)`**
- Sets: `_deprecated=True`, `_deprecated_since`, `_deprecated_replacement`, `_deprecated_remove_in`
- Runtime: Emits `DeprecationWarning` on each call with formatted message

### 2.2 Debug/Safety Decorators (Tier: DEBUG_SAFETY / CHANGE_DETECTION) — 4 decorators

| Decorator | Parameters | Tier | Purpose |
|-----------|-----------|------|---------|
| `@reads` | `*components: type` | DEBUG_SAFETY (10) | Declare read access for scheduler/debug |
| `@writes` | `*components: type` | DEBUG_SAFETY (10) | Declare write access for scheduler/debug |
| `@trace_stack` | `depth=3, show_decorator_chain=True` | DEBUG_SAFETY (10) | Enhanced error traces |
| `@track_changes` | `fields: Optional[list[str]]` | CHANGE_DETECTION (11) | Enable dirty flags (requires @component) |

**`@trace_stack(depth=3, show_decorator_chain=True)`**
- Sets: `_trace_stack=True`, `_trace_stack_depth`, `_trace_stack_show_chain`
- Runtime: On error, shows N stack frames + applied decorator chain

**`@track_changes(fields=None)`**
- Requires: `("component",)` — must be applied after @component
- Sets: `_tracked=True`, `_tracked_fields`
- Runtime: Enables Foundation Tracker dirty flag integration

### 2.3 ECS Core Decorators (used by debug systems)

| Decorator | Purpose in Debug |
|-----------|-----------------|
| `@resource` | Console system as singleton resource |
| `@event` | Debug events (log, profile, crash) |
| `@system` | Debug systems (profiler update, log flush, replay tick) |
| `@component` | Debug components (DebugDraw, ProfileMarker) |

### 2.4 Scheduling Decorators (for debug systems)

| Decorator | Purpose in Debug |
|-----------|-----------------|
| `@phase` | Debug phase (runs after all gameplay) |
| `@throttle` | Rate-limit debug overlay updates |
| `@run_if` | Skip debug systems in shipping builds |
| `@exclusive` | Crash handler needs sole access |

### 2.5 Data Flow Decorators

| Decorator | Purpose in Debug |
|-----------|-----------------|
| `@serializable` | Crash state capture, replay data |
| `@snapshot` | Replay checkpoints |

### 2.6 Time/Determinism Decorators

| Decorator | Purpose in Debug |
|-----------|-----------------|
| `@deterministic` | Mark systems for replay compatibility |
| `@rewindable` | Replay scrubbing |
| `@pausable` | Pause game for debugging |
| `@time_scale` | Slow motion for analysis |

---

## 3. Metaclasses of Interest

### 3.1 EngineMeta (Base)
- Global debug registry, common `__repr__`
- Debug use: All debug types register here for discovery

### 3.2 ComponentMeta
- Debug use: Debug draw components (DebugLine, DebugSphere, ProfileMarker)
- Registry: `_components` — Inspector enumerates all component types

### 3.3 SystemMeta
- Debug use: Debug systems (LogFlushSystem, ProfileUpdateSystem, ReplayTickSystem)
- Registry: `_systems` — Profiler discovers all systems for timing

### 3.4 ResourceMeta
- Debug use: Console, Profiler, Logger as singleton resources
- Registry: `_resources` — Debug tools discover these at startup

### 3.5 EventMeta
- Debug use: LogEvent, CrashEvent, ProfileEvent, DebugCommandEvent
- Registry: `_events` — EventLog records these with causal chains

---

## 4. Descriptors of Interest

### 4.1 Debug-Specific Descriptors (from `trinity/descriptors/debug.py`)

**ProfiledDescriptor**
- ID: `"profiled"`
- Parameters: `max_samples=100`
- Accepts inner/outer: `("*",)` / `("*",)`
- Runtime: Times every `__get__` and `__set__` using `perf_counter_ns()`. Maintains rolling window of `max_samples` timings.
- Sets on instance: `_profile_get_{name}` (list), `_profile_set_{name}` (list)
- Helper: `get_stats(obj)` → `{get: {count, avg_ns, max_ns, min_ns}, set: {...}}`
- Debug use: Field-level access profiling for hot path identification

**LoggedDescriptor**
- ID: `"logged"`
- Parameters: `log_level="DEBUG"`
- Accepts inner/outer: `("*",)` / `("*",)`
- Runtime: Logs every `__get__` as `"ClassName.field → value"` and every `__set__` as `"ClassName.field: old → new"` via `trinity.descriptors.debug` logger
- Debug use: Trace individual field accesses during debugging

**WatchedDescriptor**
- ID: `"watched"`
- Parameters: `condition: Optional[Callable], callback: Optional[Callable]`
- Accepts inner/outer: `("*",)` / `("*",)`
- Runtime: On `__set__`, if `condition(value)` is True → calls `callback(obj, name, value)` OR triggers `pdb.set_trace()` breakpoint if no callback
- Debug use: Conditional breakpoints on field values (e.g., break when health < 0)

### 4.2 Audit Descriptor (from `trinity/descriptors/audit.py`)

**AuditDescriptor**
- ID: `"audit"`
- Parameters: `max_entries=1000, log_reads=False`
- Accepts inner/outer: `("*",)` / `("*",)`
- Excludes: `()`
- Runtime: Append-only timestamped audit trail. Set entries: `(timestamp, "set", old, new)`. Get entries (if `log_reads`): `(timestamp, "get", value)`. Auto-truncates to `max_entries` (FIFO).
- Sets on instance: `_audit_{name}` (list)
- Helpers: `get_audit_log(obj, field, limit=None)`, `clear_audit_log(obj, field)`
- Debug use: Complete field access history for post-mortem analysis

### 4.3 Observable Descriptor (from `trinity/descriptors/observable.py`)

**ObservableDescriptor**
- ID: `"observable"`
- Accepts inner: `("tracked", "storage", "validated", "range")`
- Accepts outer: `("networked", "cached")`
- Excludes: `("transient",)`
- Runtime: Notifies registered observers on value change. Observer: `(obj, field, old, new) → None`
- Helpers: `add_observer()`, `remove_observer()`, `clear_observers()`
- Debug use: Inspector property panels subscribe to observe changes

**BoundDescriptor**
- ID: `"bound"`
- Parameters: `source, getter, setter`
- Accepts inner/outer: `("*",)` / `("*",)`
- Runtime: Two-way binding to external data source via getter/setter
- Debug use: Bind UI widget ↔ component field

### 4.4 Other Descriptors Relevant to Debug

| Descriptor | ID | Debug Use |
|-----------|-----|----------|
| `TrackedDescriptor` | `"tracked"` | Tracker integration — dirty flags for inspector refresh |
| `VersionedDescriptor` | `"versioned"` | Per-field version counter — cache invalidation |
| `SerializableDescriptor` | `"serializable"` | Crash state, replay snapshots |
| `TransientDescriptor` | `"transient"` | Excluded from serialization — debug-only fields |
| `ExpiringDescriptor` | `"expiring"` | TTL debug messages, temp markers |
| `AtomicDescriptor` | `"atomic"` | Thread-safe debug counters |
| `EventSourcedDescriptor` | `"event_sourced"` | Full event replay per field |

### 4.5 Annotated Field Syntax for Debug Components

```python
from typing import Annotated
from trinity.descriptors import Profiled, Logged, Watched, Audit, Observable

class DebugHealth(Component):
    hp: Annotated[float, Profiled(max_samples=50), Audit(log_reads=True)]
    # Every get/set is timed AND audited

class WatchedPosition(Component):
    x: Annotated[float, Watched(condition=lambda v: v > 1000, callback=on_oob)]
    # Triggers callback if x exceeds 1000 (out of bounds detection)

class InspectedValue(Component):
    value: Annotated[float, Observable(), Logged(log_level="INFO")]
    # Inspector subscribes to changes + all access logged
```

---

## 5. Foundation Integration Points

### 5.1 EventLog — **Primary debug backbone**
- **API:** `record(event)`, `events_at(tick)`, `events_for_entity(id)`, `events_for_operation(op)`, `events_caused_by(entity)`, `events_where(**kwargs)`, `changes_where(**kwargs)`, `@traced` decorator
- **Debug uses:**
  - **Profiling:** `@traced` on system.execute() records timing per tick
  - **Replay:** Replay `events_at(tick)` to reproduce exact game state
  - **Causal debugging:** `events_caused_by(entity)` answers "why did X happen?"
  - **Structured logging:** Events with typed fields, queryable post-mortem
  - **Frame boundaries:** Record frame start/end events for timeline view

### 5.2 Tracker — Change detection & undo/redo
- **API:** `mark_dirty()`, `is_dirty()`, `dirty_fields()`, `mark_clean()`, `all_dirty()`, `on_change()`, `off_change()`, `begin_transaction()`, `commit_transaction()`, `rollback_transaction()`, `undo()`, `redo()`
- **Debug uses:**
  - **Inspector:** `on_change(obj, callback)` refreshes property panels on edit
  - **CVar system:** `on_change(cvar, callback)` fires on CVar changes
  - **Editor undo/redo:** `undo()`/`redo()` for editor operations
  - **Dirty visualization:** `all_dirty()` highlights changed entities in scene view

### 5.3 Mirror — Runtime introspection
- **API:** `mirror(obj_or_cls)`, `get(name)`, `set(name, value)`, `has(name)`, `to_dict()`, `get_path(dotted)`, `set_path(dotted, value)`, `fields`, `methods`, `describe()`, `schema_hash(cls)`
- **Debug uses:**
  - **Inspector:** Enumerate fields with types, render appropriate widgets
  - **Console:** Read/write any field by name
  - **Watch window:** Monitor field values in real-time
  - **Schema check:** `schema_hash()` detects desync in replay

### 5.4 Registry — Type discovery
- **API:** `all_types()`, `subclasses(base)`, `types_with_decorator(name)`, `instances(cls)`, `instance_count(cls)`, `describe(cls)`
- **Debug uses:**
  - **Stats:** `instance_count(Component)` for entity statistics
  - **Type browser:** `all_types()` enumerates all engine types
  - **Decorator query:** `types_with_decorator("profiled")` finds all profiled types

### 5.5 Bridge/Shell — Live debugging REPL
- **API:** `create_shell()`, `create_ai_interface()`, `TrinityWorldAdapter`
- **Debug uses:**
  - **Console:** `create_shell()` provides Python REPL with engine namespace
  - **Live manipulation:** Create/destroy entities, modify components in real-time
  - **Scripting:** Execute debug scripts (`exec debug.cfg`)

### 5.6 Serializer — State capture
- **API:** `to_dict()`, `from_dict()`, `to_bytes()`, `from_bytes()`, `to_file()`, `from_file()`, `deep_copy()`, `diff()`, `patch()`
- **Debug uses:**
  - **Crash capture:** `to_dict(world)` snapshots entire state at crash
  - **Replay storage:** `to_bytes()` for efficient replay data
  - **Diff view:** `diff(state_a, state_b)` shows what changed between frames

### 5.7 ContentStore — Efficient snapshots
- **Debug uses:**
  - **Replay:** Content-addressable snapshots with structural sharing
  - **Checkpoint:** Periodic snapshots for crash recovery
  - **Diff:** Efficient delta between states

### 5.8 Provenance — Causal debugging
- **Debug uses:**
  - **"Why?" queries:** Derivation trees explain why a value is what it is
  - **Divergence debugging:** "Why did simulation diverge at tick N?"

---

## 6. Architecture Spec Details

> **Reference:** `DIAGRAMS/ARCHITECTURE_DEBUG.md` (1,266 lines)

### 6.1 Logging System

**Log Levels:** Verbose, Debug, Info, Warning, Error, Fatal

**Log Categories:**
- Engine: LogEngine, LogRendering, LogPhysics, LogAI, LogNetwork, LogAudio, LogAnimation, LogInput
- Game: LogGameplay, LogPlayer, LogUI, LogQuest, LogInventory

**Log Format:**
```
[2024-01-15 14:32:15.123] [Warning] [LogPhysics] Object fell through floor
```

**Structured Log Format:**
```json
{
  "timestamp": "2024-01-15T14:32:15.123Z",
  "level": "info",
  "category": "LogPlayer",
  "message": "Health changed",
  "fields": {
    "player_id": "player_1",
    "old_health": 100,
    "new_health": 75,
    "damage_source": "enemy_1"
  }
}
```

**Output Targets:** Console (in-game), File (rotated), IDE (VS output), Remote (network), System Log (OS)

**Filtering:** By level, by category, by keyword, by regex

**File Strategy:** Rotation per session, size limits, compression, crash-safe buffer

### 6.2 Console System

**UI Components:** Output window with scrollback, input line with autocomplete, command history

**Console Modes:** Overlay (transparent), Full Screen (dedicated), Mini (small bar)

**Command Types:**
- Exec commands: `god`, `teleport 100 200 50`, `spawn enemy`, `kill all`
- CVars: `r.VSync`, `r.VSync 0`, `p.Gravity -980.0`
- Aliases: `alias noclip "god; fly"`
- Script execution: `exec debug.cfg`

**Command Categories:**
- Rendering: `r.*`, `sg.*` (VSync, Shadows, etc.)
- Physics: `p.*` (ShowCollision, Gravity)
- AI: `ai.*` (Debug, ForceState)
- Network: `net.*` (SimulateLatency, PacketLoss)
- Statistics: `stat *` (fps, memory, gpu, physics)
- Show flags: `show *` (collision, bounds, navmesh)

**CVar Types:** Integer, Float, Boolean, String
**CVar Flags:** ReadOnly, Cheat, Config, Scalability

**Cheat Commands:**

| Command | Effect |
|---------|--------|
| `god` | Invulnerability |
| `fly` | Pass through geometry |
| `ghost` | No collision |
| `teleport X Y Z` | Move to location |
| `spawn Actor` | Create actor |
| `kill Target` | Destroy actor |
| `slomo 0.5` | Time dilation |
| `give Item` | Add to inventory |
| `sethealth 100` | Modify health |

### 6.3 Visual Debugging

**Debug Draw Primitives:**
- Shapes: Line, Arrow, Point, Sphere, Box, Capsule, Cylinder, Cone, Plane, Mesh
- Text: Screen (2D overlay), World (3D positioned), Billboard (faces camera)
- Options: Color, Duration (0=one frame), Thickness, Depth Test (behind/in-front)

**Debug Overlays:**
- Physics: Collision shapes, contact points, joints, raycasts, velocities, center of mass
- Navigation: NavMesh polygons, paths, off-mesh links, agents, obstacles
- Rendering: Wireframe, unlit, buffer views (normal, roughness), overdraw, shader complexity
- AI: Perception cones, behavior tree state, EQS results, blackboard values
- Audio: Sound positions, attenuation spheres, active voices
- Network: Replication status, ownership, relevancy

**Render Debug Views:**

| View | Description |
|------|-------------|
| Wireframe | Mesh edges only |
| Unlit | No lighting |
| Base Color | Albedo only |
| Normals | Surface direction |
| Roughness | Surface roughness |
| Metallic | Metal/dielectric |
| AO | Ambient occlusion |
| Overdraw | Pixel complexity |
| Shader Complexity | Instruction count |

**Debug Gizmos:** Transform (translate/rotate/scale axes), Bounds (AABB/OBB), Light (radius/cone), Camera (frustum), Audio (attenuation radius), Trigger (volume)

### 6.4 Profiling

**CPU Profiling:**
- Metrics: Frame time (target 16.67ms@60fps), game thread, render thread, worker threads
- Methods: Sampling (low overhead), Instrumentation (precise), Scoped timers (`PROFILE_SCOPE()`), Trace events (Chrome format)
- Analysis: Hotspots, flame graph, timeline per-frame view
- Hierarchy:
  ```
  └─ Main (16.5ms)
     ├─ Update (8.2ms)
     │  ├─ Physics (3.1ms)
     │  ├─ AI (2.8ms)
     │  └─ Animation (2.3ms)
     └─ Render (8.3ms)
  ```

**GPU Profiling:**
- Metrics: GPU frame time, per-pass timing, draw calls, state changes
- Counters: Vertices, triangles, fill rate, bandwidth, occupancy
- Tools: Built-in, RenderDoc, PIX, Nsight, RGP

**Memory Profiling:**
- Metrics: Total memory, heap usage, VRAM, per-pool breakdown
- Tracking: Memory tags ([Rendering], [Physics], [Audio], [Gameplay]), allocation tracking, stack traces, lifetime analysis
- Issues: Leak detection, corruption, fragmentation, buffer overrun

**Network Profiling:**
- Metrics: Bandwidth (KB/s), packet count, RTT, packet loss, per-actor bandwidth, per-property bytes

**Statistics System:**
- `stat fps` — Frame rate and timing
- `stat unit` — Per-system breakdown
- `stat engine` — Engine internals
- `stat memory` — Memory usage
- `stat gpu` — GPU timing
- `stat physics` — Physics simulation
- Display types: Counters, Timers, Graphs, Bars

### 6.5 Assertions & Checks

**Runtime Assertions:**

| Macro | Behavior |
|-------|----------|
| `check(cond)` | Fatal if false, always active |
| `checkf(cond, msg)` | Fatal with formatted message |
| `verify(cond)` | Like check, but returns the value |
| `ensure(cond)` | Non-fatal, logs once, continues |
| `ensureAlways(cond)` | Non-fatal, logs every time |
| `checkSlow(cond)` | Only in debug builds |
| `verifySlow(cond)` | Only in debug builds |

**Assertion Responses:** Break (debugger), Log (record), Crash (terminate), Continue (ensure)

**Validation Systems:**
- Pointer validation: Null check, dangling detection, valid object check
- Bounds checking: Array index, buffer size
- State validation: Valid state, invariant checks

**Sanitizers:** ASAN (buffer overflow, use-after-free), TSAN (data races), UBSAN (integer overflow, null deref), MSAN (uninitialized reads)

### 6.6 Crash Handling

**Captured Data:** Stack trace, CPU registers/memory, recent logs, screenshot, save data

**Minidump Levels:**
- Mini: Stack + threads (~100KB)
- Medium: + Referenced memory (~1-10MB)
- Full: Complete process memory (~100MB+)

**Crash Report:** Exception type/address, stack trace with symbols, thread states, module list, system info, game version, recent log buffer

**Reporting:** Local (crash files, dialog, text log), Remote (upload, symbol server, aggregation)
**Services:** Sentry, Crashlytics, Backtrace, custom
**Metrics:** Crash count, unique crashes, affected users, trends

### 6.7 Replay & Recording

**Recording Types:**
- Input recording: Player inputs, small files, requires determinism
- State recording: World snapshots, larger files, always works
- Network recording: Network traffic, replication debugging
- Demo recording: Full replay capability, any camera

**Recording Modes:** Continuous, Triggered (on event/crash), Rolling (last N seconds)

**Playback Controls:** Play/Pause, Speed (0.1x–4x), Seek, Frame step, Reverse

**Camera Modes:** Follow, Free Cam, POV, Orbit

**Features:** Slow motion, X-ray (see-through), highlights, annotations

**Frame Capture:** Screenshot, Video, GIF

### 6.8 Testing Framework

**Test Types:**
- Unit: Individual functions, fast, isolated
- Integration: Multiple systems together
- Functional: Game features (player can jump, inventory works)
- Automation: Smoke, regression, stress

**Execution Modes:** Editor, Game, Command Line, CI

**Assertions:** EXPECT_EQ, EXPECT_NE, EXPECT_TRUE, EXPECT_FALSE, EXPECT_NEAR, EXPECT_THROW

**Fixtures:** SetUp/TearDown per test, shared fixtures

### 6.9 Debug Tools

**Gameplay Debugging:**
- Cheats: God mode, infinite ammo, no clip, speed mult, teleport, spawn items
- Time control: Pause, slow motion (0.5x), fast forward (2x/4x), frame step
- Debug camera: Free cam, orbit, actor follow, cycle views

**AI Debugging:**
- Perception debug: Sight cones, hearing radius
- Behavior tree viewer: Active nodes, execution flow
- EQS viewer: Scored positions, winner
- Blackboard view: All key-value pairs
- AI control: Pause AI, step, override behavior, possess

**Physics Debugging:** Pause physics, step, apply force, query visualization, body inspection

**Network Debugging:**
- Simulation: Add latency (100-500ms), packet loss (5-20%), jitter, bandwidth limit
- Tools: Packet logging, connection inspect, replication debug

---

## 7. Decorator Stacks of Interest

### 7.1 Development Stack (from `builtin_stacks/development.py`)

**`profiled_dev(name, warn_ms=2.0)`**
Composes: `@profile(name=name, warn_ms=warn_ms)` + `@trace(level="debug")` + `@build_only(configurations={"debug", "development"})`
→ Dev-only profiling + tracing, stripped in shipping builds.

### 7.2 Proposed Debug-Specific Stacks

```python
from trinity.decorators.stacks import stack, parameterized_stack
from trinity.decorators.dev import profile, trace, invariant, bench
from trinity.decorators.debug_safety import trace_stack, track_changes
from trinity.decorators.ecs_core import system, resource, event, component
from trinity.decorators.scheduling import run_if, throttle, phase

# Debug system — runs only in debug builds, rate-limited
@parameterized_stack
def debug_system(name, phase="post_update", max_hz=30):
    return stack(
        system(phase=phase),
        throttle(max_hz=max_hz),
        run_if(lambda: is_debug_build()),
        profile(name=name),
        name=f"debug_system_{name}"
    )

# Profiled system — full profiling + trace stack for production systems
@parameterized_stack
def profiled_system(name, phase="update", warn_ms=2.0):
    return stack(
        system(phase=phase),
        profile(name=name, warn_ms=warn_ms),
        trace(level="info"),
        trace_stack(depth=5),
        name=f"profiled_{name}"
    )

# Debug component — fully observable for inspector
@parameterized_stack
def debug_component(name=None):
    return stack(
        component(name=name),
        track_changes,
        name="debug_component"
    )

# Debug resource — singleton debug tool
debug_resource = stack(resource, name="debug_resource")

# Debug event — for debug system communication
debug_event = stack(event, name="debug_event")

# Crash-safe component — serializable + tracked for crash capture
@parameterized_stack
def crash_safe(version=1):
    return stack(
        component,
        track_changes,
        serializable(format="binary", version=version),
        name="crash_safe"
    )
```

### 7.3 Existing Stacks Used by Debug

**`production_component`** — includes `@track_changes` for Inspector integration
**`safe_system`** — includes `@reads`/`@writes` for access tracking
**`saveable_data`** — includes `@track_changes` + `@serializable` for crash capture

---

## 8. TODO Checklist

> From `docs/GAME_ENGINE_INTEGRATION_TODO.md` section 14 + sections 17-18

### 8.1 Logging
- [ ] Implement logging system (categories, levels, output targets)
- [ ] Implement log filtering (per-category, per-level)
- [ ] Implement log output to file, console, network
- [ ] Implement structured logging (JSON format with typed fields)
- [ ] Wire Foundation EventLog → structured log entries
- [ ] Wire LoggedDescriptor → per-field access logging

### 8.2 Console System
- [ ] Implement in-game console (CVars, commands, autocomplete)
- [ ] Implement CVar system (typed variables, change callbacks, flags)
- [ ] Implement command categories (r.*, p.*, ai.*, net.*, stat, show)
- [ ] Implement console modes (overlay, full screen, mini)
- [ ] Implement script execution (`exec file.cfg`)
- [ ] Wire Foundation Shell → console command execution
- [ ] Wire Foundation Tracker.on_change → CVar change notifications
- [ ] Wire Foundation Mirror → console property inspection

### 8.3 Visual Debugging
- [ ] Implement debug draw (lines, arrows, points, spheres, boxes, capsules, cylinders, cones, planes, meshes)
- [ ] Implement debug text (screen, world, billboard)
- [ ] Implement draw options (color, duration, thickness, depth test)
- [ ] Implement debug overlays (physics, navigation, rendering, AI, audio, network)
- [ ] Implement render debug views (wireframe, unlit, buffer views, overdraw, shader complexity)
- [ ] Implement debug gizmos (transform, bounds, light, camera, audio, trigger)
- [ ] Wire @editor decorator → gizmo configuration

### 8.4 Profiling
- [ ] Implement CPU profiler (hierarchical timer, flame graph, timeline)
- [ ] Implement GPU profiler (timestamp queries, pipeline stats, per-pass timing)
- [ ] Implement memory profiler (allocation tracking, leak detection, per-tag breakdown)
- [ ] Implement network profiler (bandwidth, latency, packet loss, per-actor)
- [ ] Implement statistics system (stat fps, stat unit, stat memory, stat gpu, stat physics)
- [ ] Implement stat display types (counters, timers, graphs, bars)
- [ ] Wire Foundation EventLog → profiling event capture
- [ ] Wire @profile decorator → CPU profiler integration
- [ ] Wire @gpu_profile decorator → GPU profiler integration
- [ ] Wire ProfiledDescriptor → field-level access profiling
- [ ] Wire @reads/@writes → access tracking visualization

### 8.5 Assertions & Crash Handling
- [ ] Implement assertion system (check, checkf, verify, ensure, ensureAlways)
- [ ] Implement assertion responses (break, log, crash, continue)
- [ ] Implement crash capture (stack trace, context, recent logs, screenshot, save data)
- [ ] Implement minidump generation (mini, medium, full levels)
- [ ] Implement crash reporter (local files, dialog, upload)
- [ ] Implement crash metrics (count, unique signatures, affected users)
- [ ] Wire @invariant decorator → runtime invariant checking
- [ ] Wire Foundation Session → crash state capture
- [ ] Wire Foundation Serializer → crash snapshot

### 8.6 Replay & Recording
- [ ] Implement input recording (record inputs → deterministic replay)
- [ ] Implement state recording (periodic snapshots + deltas)
- [ ] Implement network recording (capture traffic for replay)
- [ ] Implement playback controls (play, pause, speed, seek, frame step, reverse)
- [ ] Implement replay camera modes (follow, free cam, POV, orbit)
- [ ] Implement recording modes (continuous, triggered, rolling)
- [ ] Wire Foundation EventLog → replay event source
- [ ] Wire Foundation DeltaSync → efficient replay storage
- [ ] Wire Foundation ContentStore → content-addressable snapshots
- [ ] Wire @deterministic → replay-compatible systems
- [ ] Wire @rewindable → replay scrubbing support
- [ ] Wire AuditDescriptor → per-field access history for replay analysis

### 8.7 Testing Framework
- [ ] Implement in-engine test runner (unit, integration, functional, stress)
- [ ] Implement test assertions (EXPECT_EQ, EXPECT_NE, EXPECT_TRUE, EXPECT_NEAR, EXPECT_THROW)
- [ ] Implement test fixtures (SetUp, TearDown, shared state)
- [ ] Implement test execution modes (editor, game, command line, CI)
- [ ] Implement automated testing (bots, scenarios)
- [ ] Wire Foundation Shell → test execution from console
- [ ] Wire @test decorator → test discovery and configuration
- [ ] Wire @bench decorator → benchmark suite
- [ ] Wire @invariant decorator → property-based testing

### 8.8 Debug Tools
- [ ] Implement cheat system (god, fly, ghost, teleport, spawn, kill, slomo, give)
- [ ] Implement time control (pause, slow motion, fast forward, frame step)
- [ ] Implement debug camera (free cam, orbit, actor follow, cycle views)
- [ ] Implement AI debugging tools (perception viz, BT viewer, EQS viz, blackboard)
- [ ] Implement physics debugging tools (pause, step, force, query viz, body inspect)
- [ ] Implement network debugging tools (latency sim, packet loss, jitter, bandwidth limit)
- [ ] Wire @deprecated decorator → deprecation tracking dashboard
- [ ] Wire WatchedDescriptor → conditional breakpoints on field values

---

## 9. Directory Structure

```
engine/debug/
├── __init__.py              # Public API exports
├── DEBUG_CONTEXT.md         # This file
│
├── logging/
│   ├── __init__.py          # Re-export: Logger, LogLevel, LogCategory
│   ├── logger.py            # Logger, LogLevel enum, LogCategory enum
│   ├── sinks.py             # LogSink, ConsoleSink, FileSink, NetworkSink, IDESink
│   ├── filters.py           # LevelFilter, CategoryFilter, KeywordFilter, RegexFilter
│   ├── structured.py        # StructuredLogger (JSON format)
│   └── rotation.py          # Log file rotation, compression, archival
│
├── console/
│   ├── __init__.py          # Re-export: Console, CVar, Command
│   ├── console.py           # Console (overlay/fullscreen/mini modes)
│   ├── cvar.py              # CVar<T>, CVarFlags, CVar registry
│   ├── commands.py          # Command registry, command handler
│   ├── autocomplete.py      # Tab completion, history
│   ├── aliases.py           # Command aliases
│   └── scripting.py         # Script file execution (exec)
│
├── visual/
│   ├── __init__.py          # Re-export: DebugDraw, DebugOverlay, Gizmo
│   ├── draw.py              # DebugDraw (lines, arrows, spheres, boxes, text)
│   ├── overlays.py          # DebugOverlay (physics, nav, rendering, AI, audio, network)
│   ├── render_views.py      # Debug render modes (wireframe, unlit, buffer views)
│   └── gizmos.py            # Transform, bounds, light, camera, audio, trigger gizmos
│
├── profiling/
│   ├── __init__.py          # Re-export: CPUProfiler, GPUProfiler, MemoryProfiler
│   ├── cpu.py               # CPUProfiler (scoped timers, flame graph, hierarchy)
│   ├── gpu.py               # GPUProfiler (timestamp queries, pass timing)
│   ├── memory.py            # MemoryProfiler (allocation tracking, leak detection)
│   ├── network.py           # NetworkProfiler (bandwidth, RTT, packet loss)
│   └── stats.py             # Statistics system (stat fps, stat memory, display types)
│
├── crash/
│   ├── __init__.py          # Re-export: CrashHandler, Assertion
│   ├── assertions.py        # check, checkf, verify, ensure, ensureAlways
│   ├── handler.py           # CrashHandler (capture, minidump, report)
│   ├── minidump.py          # Minidump generation (mini, medium, full)
│   └── reporter.py          # CrashReporter (local, remote, metrics)
│
├── replay/
│   ├── __init__.py          # Re-export: Recorder, Player
│   ├── recorder.py          # InputRecorder, StateRecorder, NetworkRecorder
│   ├── player.py            # ReplayPlayer (play, pause, seek, speed, reverse)
│   ├── camera.py            # ReplayCamera (follow, free, POV, orbit)
│   ├── storage.py           # ReplayStorage (ContentStore, DeltaSync integration)
│   └── capture.py           # FrameCapture (screenshot, video, GIF)
│
├── testing/
│   ├── __init__.py          # Re-export: TestRunner, TestSuite
│   ├── runner.py            # TestRunner (editor, game, CLI, CI modes)
│   ├── assertions.py        # EXPECT_EQ, EXPECT_NE, EXPECT_TRUE, EXPECT_NEAR, EXPECT_THROW
│   ├── fixtures.py          # TestFixture, SetUp, TearDown
│   ├── automation.py        # AutomationBot, TestScenario
│   └── benchmarks.py        # BenchmarkSuite (from @bench)
│
└── tools/
    ├── __init__.py          # Re-export debug tools
    ├── cheats.py            # Cheat commands (god, fly, teleport, etc.)
    ├── time_control.py      # Pause, slow motion, fast forward, frame step
    ├── debug_camera.py      # Free cam, orbit, follow, cycle
    ├── ai_debug.py          # Perception viz, BT viewer, EQS, blackboard
    ├── physics_debug.py     # Physics pause/step, force, query viz
    └── network_debug.py     # Latency sim, packet loss, jitter, bandwidth limit
```

---

## 10. Canonical Usage Examples

### 10.1 Logging

```python
from engine.debug.logging import Logger, LogLevel, LogCategory

log = Logger()

# Basic logging
log.info(LogCategory.Engine, "Engine initialized")
log.warning(LogCategory.Physics, "Object fell through floor")
log.error(LogCategory.Network, "Connection lost to server")

# Structured logging
log.structured(LogLevel.Info, LogCategory.Player, "Health changed", {
    "player_id": "player_1",
    "old_health": 100,
    "new_health": 75,
    "damage_source": "enemy_1"
})

# Filtering
log.set_level(LogCategory.Physics, LogLevel.Warning)  # Only warnings+
log.add_filter(CategoryFilter(exclude=[LogCategory.Audio]))

# Output targets
log.add_sink(FileSink("game.log", rotate_size_mb=10))
log.add_sink(NetworkSink("localhost:9000"))
```

### 10.2 Console with CVars

```python
from engine.debug.console import Console, CVar, CVarFlags

# Register CVars
r_vsync = CVar("r.VSync", default=1, flags=CVarFlags.Config)
r_shadows = CVar("r.ShadowQuality", default=3, flags=CVarFlags.Scalability)
p_gravity = CVar("p.Gravity", default=-980.0, flags=CVarFlags.Cheat)

# CVar change callback (wired through Foundation Tracker)
r_vsync.on_change(lambda old, new: renderer.set_vsync(new))

# Register commands
console = Console()
console.register("teleport", lambda x, y, z: player.set_position(Vec3(x, y, z)))
console.register("god", lambda: player.set_invulnerable(True), flags=CVarFlags.Cheat)

# Execute from Shell
console.execute("r.VSync 0")
console.execute("stat fps")
console.execute("teleport 100 200 50")
```

### 10.3 Visual Debugging

```python
from engine.debug.visual import DebugDraw, Color

# One-frame draw
DebugDraw.line(start=Vec3(0, 0, 0), end=Vec3(10, 0, 0), color=Color.RED)
DebugDraw.sphere(center=entity.position, radius=5.0, color=Color.GREEN, duration=2.0)
DebugDraw.box(center=aabb.center, extent=aabb.extents, color=Color.BLUE)
DebugDraw.arrow(origin=entity.position, direction=entity.forward, color=Color.YELLOW)

# Text
DebugDraw.screen_text("FPS: 60", x=10, y=10, color=Color.WHITE)
DebugDraw.world_text(f"HP: {health.hp}", position=entity.position + Vec3(0, 2, 0))

# Overlays
DebugDraw.enable_overlay("physics")  # Show collision shapes
DebugDraw.enable_overlay("navmesh")  # Show navigation mesh
DebugDraw.set_render_view("wireframe")
```

### 10.4 Profiling

```python
from engine.debug.profiling import CPUProfiler, Stats

# Scoped profiling
with CPUProfiler.scope("Physics.Update"):
    physics_world.step(dt)

# System profiling via decorator
@profile(name="AI.BehaviorTree", warn_ms=5.0)
def update_behavior_trees(world, dt):
    ...

# Statistics
Stats.counter("Entities.Active", world.entity_count())
Stats.timer("Frame.Update", update_ms)
Stats.graph("FPS", current_fps, max_history=300)

# Memory profiling
from engine.debug.profiling import MemoryProfiler
MemoryProfiler.snapshot("before_load")
load_level("level_01")
MemoryProfiler.snapshot("after_load")
diff = MemoryProfiler.diff("before_load", "after_load")
```

### 10.5 Assertions

```python
from engine.debug.crash import check, checkf, ensure, verify

# Fatal assertions
check(entity is not None)
checkf(index < len(array), f"Index {index} out of bounds (size {len(array)})")

# Non-fatal
if not ensure(health >= 0):
    health = 0  # recover

# Verify (returns value)
ptr = verify(allocator.allocate(size))

# Invariants via decorator
@invariant(check=lambda self: self.hp <= self.max_hp, when="debug")
class Health(Component):
    hp: float = 100.0
    max_hp: float = 100.0
```

### 10.6 Replay

```python
from engine.debug.replay import Recorder, ReplayPlayer

# Record
recorder = Recorder(mode="rolling", keep_seconds=30)
recorder.start()
# ... gameplay ...
recorder.save("replay.bin")  # Saves last 30 seconds

# Playback
player = ReplayPlayer.load("replay.bin")
player.play()
player.set_speed(0.25)   # Slow motion
player.seek(tick=1500)   # Jump to tick
player.step_frame()      # Single frame advance
player.set_camera("free")  # Free camera mode
```

### 10.7 Testing

```python
from engine.debug.testing import TestRunner, TestSuite, expect_eq, expect_near

class MathTests(TestSuite):
    def test_vector_add(self):
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        expect_eq(a + b, Vec3(5, 7, 9))
    
    def test_quat_normalize(self):
        q = Quat(1, 2, 3, 4)
        expect_near(q.normalized().length(), 1.0, epsilon=1e-6)

# Run from console
runner = TestRunner(mode="cli")
runner.discover("tests/")
runner.run(filter="Math*")

# Benchmarking
@bench(iterations=10000, warmup=100)
def bench_ecs_query():
    world.for_each(Transform, Velocity, fn=lambda t, v: None)
```

### 10.8 Foundation Integration

```python
from foundation.eventlog import get_event_log, traced, set_current_tick
from foundation.tracker import tracker
from foundation.mirror import mirror
from foundation.bridge import create_shell

# EventLog for profiling
@traced
@system(phase="gameplay")
class DamageSystem(System):
    def execute(self, world, dt):
        ...
# → EventLog records: tick, operation="DamageSystem.execute", timing, entity changes

# Tracker for inspector
tracker.on_change(health_component, callback=lambda obj, field, old, new:
    inspector.refresh_panel(obj))

# Mirror for console
m = mirror(player_entity)
print(m.fields)  # {'health': FieldInfo(...), 'transform': ...}
m.set_path("health.hp", 100)

# Shell for live debugging
shell = create_shell()
shell.execute("e = create()")
shell.execute("world.attach(e, Health(hp=80))")
shell.execute("query(Health)")

# Causal debugging
events = get_event_log().events_caused_by(entity_id=42)
# → "Entity 42 died because: GravitySystem → CollisionSystem → DamageSystem → EntityDied"
```

---

## 11. Key Integration Patterns

### Pattern 1: EventLog → Profiling Timeline
```
Each frame:
  set_current_tick(frame)
  @traced on each system.execute():
    → EventLog records: {tick, operation, entity, changes, timing}
  
  Profiler reads EventLog:
    → events_at(tick) → hierarchical timing tree
    → flame graph from nested operations
    → timeline view across frames
```

### Pattern 2: Mirror + Tracker → Inspector
```
User selects entity in editor:
  → mirror(entity) → enumerate fields with FieldInfo
  → For each field: render widget (float input, bool checkbox, etc.)
  → tracker.on_change(entity, callback=refresh_panel)
  
User edits field in inspector:
  → mirror.set(field, value)
    → descriptor chain fires (Tracked → Validated → Storage)
    → tracker.mark_dirty(entity, field)
    → inspector panel refreshes
```

### Pattern 3: Shell → Console Commands
```
User types command in console:
  → Console.parse(input)
  → If CVar: get/set via CVar registry
  → If command: lookup in command registry, execute handler
  → If script: load file, execute line by line

Foundation Shell integration:
  → Console delegates to Shell.execute(command)
  → Shell has pre-configured namespace: create(), query(), World, etc.
  → Results displayed in console output
```

### Pattern 4: EventLog → Deterministic Replay
```
Recording:
  EventLog records every @traced operation per tick
  → {tick, operation, entity, changes, args, result}

Replay:
  For each tick in recording:
    events = event_log.events_at(tick)
    For each event:
      Re-execute operation with recorded args
      Verify changes match (desync detection via schema_hash)

Scrubbing:
  Foundation ContentStore provides snapshots at keyframes
  → Seek to nearest keyframe → replay forward to target tick
```

### Pattern 5: Descriptors → Debug Tools
```
ProfiledDescriptor:
  → Every field access timed
  → Profiler aggregates: "Transform.x averages 12ns per access"

WatchedDescriptor:
  → condition(value) triggers on set
  → callback fires OR pdb.set_trace() breakpoint
  → "Break when health.hp < 0"

AuditDescriptor:
  → Append-only audit trail per field
  → Post-mortem: "Who changed health.hp at tick 500?"
  → get_audit_log(entity.health, "hp", limit=10)

LoggedDescriptor:
  → Every access logged to trinity.descriptors.debug
  → "Transform.x: 10.0 → 15.5"
```

### Pattern 6: Crash Handler → Foundation
```
Crash occurs:
  1. CrashHandler catches signal/exception
  2. Foundation Serializer.to_dict(world) → snapshot entire state
  3. Foundation EventLog → last N events for recent history
  4. Stack trace captured with debug symbols
  5. Recent log buffer saved
  6. Minidump generated (mini/medium/full)
  7. All packaged into CrashReport
  8. Uploaded to crash service (Sentry/Backtrace/custom)
```

---

## 12. Quick Reference Tables

### Decorator Quick Reference (Debug-Specific)

| Decorator | Tier | Parameters | Target | Debug Use |
|-----------|------|-----------|--------|-----------|
| `@profile` | DEV | name, warn_ms | fn/cls | CPU timing |
| `@gpu_profile` | DEV | category, include_memory | fn/cls | GPU timing |
| `@trace` | DEV | level | fn/cls | Execution tracing |
| `@reloadable` | DEV | enabled, preserve, reinit, validate | cls/fn | Hot reload |
| `@editor` | DEV | category, hidden | any | Editor integration |
| `@test` | DEV | cases, fuzz, property_based | fn/cls | Test discovery |
| `@bench` | DEV | iterations, warmup | fn/cls | Benchmarks |
| `@invariant` | DEV | check, when | cls/fn | Runtime invariants |
| `@deprecated` | DEV | since, replacement, remove_in | fn/cls | Deprecation |
| `@reads` | DEBUG_SAFETY | *components | fn | Access tracking |
| `@writes` | DEBUG_SAFETY | *components | fn | Access tracking |
| `@trace_stack` | DEBUG_SAFETY | depth, show_decorator_chain | fn | Error traces |
| `@track_changes` | CHANGE_DET | fields | cls | Dirty flags |

### Descriptor Quick Reference (Debug-Specific)

| Descriptor | ID | Parameters | Debug Use |
|-----------|-----|-----------|-----------|
| `ProfiledDescriptor` | profiled | max_samples=100 | Field access timing |
| `LoggedDescriptor` | logged | log_level="DEBUG" | Field access logging |
| `WatchedDescriptor` | watched | condition, callback | Conditional breakpoints |
| `AuditDescriptor` | audit | max_entries=1000, log_reads=False | Access audit trail |
| `ObservableDescriptor` | observable | — | Inspector change notifications |
| `BoundDescriptor` | bound | source, getter, setter | Two-way UI binding |

### Foundation System → Debug Use

| Foundation | Debug Subsystem | Use |
|-----------|----------------|-----|
| EventLog | Profiling | Operation timing per tick |
| EventLog | Replay | Event source for deterministic replay |
| EventLog | Logging | Structured log entries |
| EventLog | Crash | Recent event history |
| Tracker | Inspector | Change notifications for panels |
| Tracker | Console | CVar change callbacks |
| Tracker | Crash | Undo/redo state for recovery |
| Mirror | Inspector | Field enumeration + editing |
| Mirror | Console | Property read/write |
| Mirror | Replay | schema_hash desync detection |
| Registry | Stats | Instance counts, type browsing |
| Registry | Profiling | System discovery for timing |
| Shell/Bridge | Console | Live REPL execution |
| Serializer | Crash | World state snapshot |
| Serializer | Replay | State recording/playback |
| ContentStore | Replay | Efficient snapshot storage |
| DeltaSync | Replay | Incremental replay storage |
| Provenance | Debugging | "Why did X happen?" queries |

### Debug Stats Commands

| Command | Shows |
|---------|-------|
| `stat fps` | Frame rate, timing |
| `stat unit` | Per-system breakdown |
| `stat engine` | Engine internals |
| `stat memory` | Memory usage per tag |
| `stat gpu` | GPU timing per pass |
| `stat physics` | Physics simulation |
| `stat streaming` | Asset streaming |
| `stat particles` | Particle systems |
| `stat network` | Bandwidth, RTT, packet loss |

### Assertion Macros

| Macro | Fatal? | Logs? | Active In |
|-------|--------|-------|-----------|
| `check(cond)` | Yes | Yes | All builds |
| `checkf(cond, msg)` | Yes | Yes | All builds |
| `verify(cond)` | Yes | Yes | All (returns value) |
| `ensure(cond)` | No | Once | All builds |
| `ensureAlways(cond)` | No | Every time | All builds |
| `checkSlow(cond)` | Yes | Yes | Debug only |
| `verifySlow(cond)` | Yes | Yes | Debug only |
