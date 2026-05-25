# PHASE_6_ARCH.md — Grail Debug Architecture

## Overview

Phase 6 implements advanced debugging features: shader debug compilation (WGSL source through naga IR to backend bytecode), time-travel debugging with input recording, ECS snapshots, binary search navigation, and fix-and-continue. This phase is partially blocked on S15 (Core ECS) for snapshot state capture. Python replay tools exist but need Rust engine integration.

## Current State

| Task | Status | What Exists | What's Missing |
|------|--------|-------------|----------------|
| T-TL-6.1 | [-] NOT STARTED | Nothing | Shader debug compiler |
| T-TL-6.2 | [-] NOT STARTED | Nothing | Shader reflection browser |
| T-TL-6.3 | [~] PARTIAL | Python input_recorder.py | Rust input recorder |
| T-TL-6.4 | [~] PARTIAL | Python state_recorder.py, replay_file.py | Rust snapshot manager |
| T-TL-6.5 | [-] NOT STARTED | Nothing | Binary search navigator |
| T-TL-6.6 | [~] PARTIAL | Python replay_timeline.py, replay_playback.py | Rust/egui time-travel UI |
| T-TL-6.7 | [-] NOT STARTED | Nothing | Causal chain preservation |
| T-TL-6.8 | [~] PARTIAL | Python hot_reload.py | Fix-and-continue during replay |
| T-TL-6.9 | [-] NOT STARTED | Nothing | Debugger breakpoints |

## Architecture: Shader Debug Compiler (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Shader Debug Compiler                                │
│                                                       │
│  Pipeline:                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │ WGSL     │→ │ naga     │→ │ naga IR  │→ │ Back │ │
│  │ Source   │  │ Parse    │  │ Dump     │  │ Byte │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────┘ │
│                                                     │
│  WGSL Source → naga IR dump visualization:          │
│  ┌──────────────────────────────────────────────┐   │
│  │ fn vs_main(@location(0) position: vec3<f32>) │   │
│  │ -> FragmentOutput {                          │   │
│  │   // naga IR:                                │   │
│  │   //   %1 = FunctionArgument position        │   │
│  │   //   %2 = Access %1.x                      │   │
│  │   //   %3 = Access %1.y                      │   │
│  │   //   %4 = Access %1.z                      │   │
│  │   //   %5 = Construct vec3(%2, %3, %4)       │   │
│  │ }                                            │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  Features:                                           │
│  ├── Parse WGSL with naga::front::wgsl               │
│  ├── Dump naga IR as human-readable text             │
│  ├── Error annotation with source line numbers       │
│  └── Backend bytecode visualization per-target       │
│      (SPIR-V, MSL, HLSL, DXIL)                      │
└──────────────────────────────────────────────────────┘
```

## Architecture: Input Log Recorder (Gap)

```
┌──────────────────────────────────────────────────────┐
│  InputLogRecorder (Rust)                              │
│                                                       │
│  Captures all input events with tick counters:        │
│  ┌──────────────────────────────────────────────┐     │
│  │ struct InputRecord {                         │     │
│  │     tick: u64,                               │     │
│  │     timestamp: Instant,                      │     │
│  │     event: InputEvent,                       │     │
│  │ }                                            │     │
│  │                                               │     │
│  │ enum InputEvent {                            │     │
│  │     KeyPress { key: KeyCode, modifiers },     │     │
│  │     KeyRelease { key: KeyCode, modifiers },  │     │
│  │     MouseMove { x: f32, y: f32 },           │     │
│  │     MousePress { button, x, y },            │     │
│  │     MouseRelease { button, x, y },          │     │
│  │     MouseScroll { delta_x, delta_y },       │     │
│  │     WindowResize { width, height },         │     │
│  │     WindowFocus { focused: bool },          │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Serialization:                                       │
│  ├── Binary format with delta compression             │
│  ├── Per-frame chunking for seeks                     │
│  └── Header with metadata (total ticks, duration)     │
│                                                       │
│  Storage:                                             │
│  ├── Rolling buffer (last N seconds in memory)        │
│  └── File persistence for long recordings             │
└──────────────────────────────────────────────────────┘
```

## Architecture: Snapshot Manager (Gap — Blocked on S15)

```
┌──────────────────────────────────────────────────────┐
│  SnapshotManager                                      │
│                                                       │
│  Periodic ECS world snapshots:                        │
│  ┌──────────────────────────────────────────────┐     │
│  │ Snapshot {                                   │     │
│  │     tick: u64,                               │     │
│  │     timestamp: u64,                          │     │
│  │     entities: Vec<EntitySnapshot>,           │     │
│  │     // GPU resource state (readback)         │     │
│  │     gpu_snapshots: Vec<GpuResourceSnapshot>, │     │
│  │     checksum: [u8; 32],  // SHA-256          │     │
│  │ }                                            │     │
│  │                                               │     │
│  │ EntitySnapshot {                             │     │
│  │     entity_id: u64,                          │     │
│  │     archetype_id: u32,                       │     │
│  │     components: Vec<(u32, Vec<u8>)>,         │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Snapshot schedule:                                   │
│  ├── Every 60 ticks (1 second at 60fps)               │
│  ├── On explicit trigger (before scene load)          │
│  └── On checkpoint command from debugger              │
│                                                       │
│  Storage:                                             │
│  ├── Configurable max snapshots (ring buffer)          │
│  ├── Incremental snapshots (store full + deltas)      │
│  └── Hashed for integrity verification                │
│                                                       │
│  GPU state capture:                                   │
│  ├── Readback GPU resource state (buffers, textures)  │
│  ├── QuerySet results snapshot                        │
│  └── Pipeline state dump                              │
└──────────────────────────────────────────────────────┘
```

## Architecture: Binary Search Navigator (Gap)

```
┌──────────────────────────────────────────────────────┐
│  BinarySearchNavigator                                 │
│                                                       │
│  Algorithm:                                           │
│  ┌──────────────────────────────────────────────┐     │
│  │ fn find_nearest_snapshot_before(target_tick) │     │
│  │     // Binary search in sorted snapshot list │     │
│  │     // Returns snapshot with max tick < tgt │     │
│  │                                               │     │
│  │ fn restore(snapshot)                         │     │
│  │     // 1. Reset ECS to snapshot state        │     │
│  │     // 2. Restore GPU resources              │     │
│  │     // 3. Verify checksum                    │     │
│  │                                               │     │
│  │ fn replay_to_tick(from_snapshot, target_tick) │     │
│  │     // Replay input events from snapshot      │     │
│  │     // Advance frame by frame to target       │     │
│  │     // Verify intermediate state at each step │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Optimizations:                                       │
│  ├── Snapshot index for O(log N) lookup               │
│  ├── Incremental replay (reuse GPU state when poss.)  │
│  └── Lazy GPU resource restore (on-demand)            │
└──────────────────────────────────────────────────────┘
```

## Architecture: Time-Travel UI (Gap)

```
┌──────────────────────────────────────────────┐
│  Time-Travel UI (egui viewport overlay)       │
│                                               │
│  ┌──────────────────────────────────────┐     │
│  │  Timeline                            │     │
│  │  ──────▪─────────▪────●──────▪───    │     │
│  │  Tick:     10230   10240 10250 10260 │     │
│  │  Frame:      170     171   172   173 │     │
│  │  Mode:   [▶ Live] [⏸ Pause] [⏪ Replay]│   │
│  │                                         │     │
│  │  Controls:                              │     │
│  │  [⏮ Start] [◀ Step Bk] [▶▶ Play]      │     │
│  │  [▶ Step Fwd] [⏭ End]                 │     │
│  │  Speed: [1x ▼]                         │     │
│  └──────────────────────────────────────┘     │
│                                               │
│  Features:                                    │
│  ├── Timeline bar with snapshot markers       │
│  ├── Tick/frame counter display               │
│  ├── Play/pause/step controls                 │
│  ├── Adjustable replay speed                  │
│  ├── Viewport shows replayed state            │
│  └── Visual indicator when replaying ("REPLAY")│
└──────────────────────────────────────────────┘
```

## Architecture: Debugger Breakpoints (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Debugger Breakpoint System                           │
│                                                       │
│  Breakpoint types:                                    │
│  ┌──────────────────────────────────────────────┐     │
│  │ enum Breakpoint {                            │     │
│  │     Tick(u64),                               │     │
│  │     EntityState {                            │     │
│  │         entity_id: u64,                      │     │
│  │         component_id: u32,                   │     │
│  │         field_offset: u32,                   │     │
│  │         expected_value: Vec<u8>,             │     │
│  │     },                                       │     │
│  │     WatchExpression {                        │     │
│  │         expression: String,  // e.g. "hp<0"  │     │
│  │     },                                       │     │
│  │     FrameType(FrameType),  // break on shadow │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Breakpoint evaluation:                               │
│  ├── Checked at end of each frame                     │
│  ├── On match: pause replay, show state               │
│  ├── Resume: continue from breakpoint                 │
│  └── Hit count tracking                               │
│                                                       │
│  Breakpoint UI:                                       │
│  ├── List of active breakpoints                       │
│  ├── Add/remove/edit                                  │
│  ├── Enable/disable                                   │
│  └── Hit count display                                │
└──────────────────────────────────────────────────────┘
```

## Architecture: Fix-and-Continue (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Fix-and-Continue                                     │
│                                                       │
│  Flow:                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │ Pause    │→│ Edit     │→│ Hot-     │→│ Resume│ │
│  │ Replay   │  │ Shader   │  │ Reload   │  │ Replay│ │
│  └──────────┘  └──────────┘  └──────────┘  └──────┘ │
│                                                       │
│  Hot-reload mechanics:                                │
│  ├── Recompile modified WGSL source                   │
│  ├── Create new wgpu shader module                    │
│  ├── Update pipeline layouts if compatible            │
│  └── Replace in render graph nodes                    │
│                                                       │
│  Continuity:                                          │
│  ├── GPU state from snapshot is valid                 │
│  ├── Only shader changes are applied                  │
│  ├── Resource bindings remain intact                  │
│  └── Visual change visible on resume                  │
│                                                       │
│  Limitations:                                         │
│  ├── Cannot change pipeline layout                    │
│  ├── Cannot add/remove resources                      │
│  └── Shader interface must remain compatible          │
└──────────────────────────────────────────────────────┘
```

## Dependency Chain

```
Phase 3 (Viewport, GPUQueryManager)
  │
  ├──► T-TL-6.1 Shader Debug Compiler ──► T-TL-6.2 Shader Browser
  │
  └──► T-TL-6.3 Input Recorder
         │
         ▼
S15 Core ECS (external)
  │
  ▼
T-TL-6.4 Snapshot Manager
  │
  ├──► T-TL-6.5 Binary Search Navigator
  │     │
  │     ├──► T-TL-6.6 Time-Travel UI
  │     ├──► T-TL-6.7 Causal Chain
  │     ├──► T-TL-6.8 Fix-and-Continue ──► Phase 5 (Material Editor)
  │     └──► T-TL-6.9 Debugger Breakpoints
```

## Implementation Order

1. T-TL-6.3: Input recorder (capture → binary serialization)
2. T-TL-6.4: Snapshot manager (full ECS snapshots, GPU state capture) — requires S15
3. T-TL-6.1: Shader debug compiler (naga IR dump)
4. T-TL-6.5: Binary search navigator (find/restore/replay)
5. T-TL-6.6: Time-travel UI (timeline, controls, viewport)
6. T-TL-6.2: Shader reflection browser
7. T-TL-6.7: Causal chain preservation in replay
8. T-TL-6.8: Fix-and-continue (edit shader during replay)
9. T-TL-6.9: Debugger breakpoints (tick, entity state, watch expression)

## Success Criteria

- Shader debug compiler shows naga IR with source-line annotations
- Input events are recorded and replayable deterministically
- ECS snapshots capture complete world state every N ticks
- Binary search finds nearest snapshot to target tick
- Time-travel UI allows pause/step/play through frame history
- Fix-and-continue: edit WGSL during replay and see updated rendering
- Breakpoints pause replay on tick match, entity state match, or expression match
