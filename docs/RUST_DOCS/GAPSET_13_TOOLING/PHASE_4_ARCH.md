# PHASE_4_ARCH.md — Debug Integration Architecture

## Overview

Phase 4 integrates GPU debugging tools into the editor: RenderDoc integration for frame capture, GPU memory tracking with per-resource-type allocation tracking, frame-perfect profiling with a 300-frame ring buffer, Chrome tracing export, profiling budgets, and waterfall visualization. Python profiling tools exist but need Rust/wgpu integration.

## Current State

| Task | Status | What Exists | What's Missing |
|------|--------|-------------|----------------|
| T-TL-4.1 | [~] PARTIAL | Python frame_profiler.py mentions RenderDoc | wgpu::RenderDoc integration |
| T-TL-4.2 | [~] PARTIAL | Python memory_profiler.py | Rust GPU memory tracker |
| T-TL-4.3 | [~] PARTIAL | Python frame_profiler.py (FrameProfile) | Rust 300-frame ring buffer |
| T-TL-4.4 | [~] PARTIAL | Python profiler_export.py | Rust Chrome trace JSON export |
| T-TL-4.5 | [-] NOT STARTED | Nothing | Profiling budget system |
| T-TL-4.6 | [~] PARTIAL | Python profiler_overlay.py | Rust waterfall visualization |

## Architecture: RenderDoc Integration (Gap)

```
┌──────────────────────────────────────────────────────┐
│  RenderDoc Integration                                │
│                                                       │
│  Initialization:                                      │
│  ┌──────────────────────────────────────────────┐     │
│  │ if let Some(rdoc) = wgpu::RenderDoc::get() { │     │
│  │     rdoc.set_capture_keys(true);             │     │
│  │     rdoc.set_overlay_settings(...);          │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Frame Graph Annotation:                              │
│  ┌──────────────────────────────────────────────┐     │
│  │ fn begin_debug_marker(label: &str) {         │     │
│  │     encoder.push_debug_group(label);         │     │
│  │ }                                            │     │
│  │ fn end_debug_marker() {                      │     │
│  │     encoder.pop_debug_group();               │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Capture Trigger:                                     │
│  ├── F11 hotkey → rdoc.trigger_capture()              │
│  ├── Programmatic API for automation                  │
│  └── Conditional compilation (no-op without RenderDoc) │
└──────────────────────────────────────────────────────┘
```

## Architecture: GPU Memory Tracker (Gap)

```
┌──────────────────────────────────────────────────────┐
│  GpuMemoryTracker                                     │
│                                                       │
│  Per-resource-type tracking:                          │
│  ┌──────────────────────────────────────────────┐     │
│  │ BufferTracker {                               │     │
│  │     total_allocated: u64,    // bytes         │     │
│  │     allocation_count: u64,                    │     │
│  │     peak_allocated: u64,                      │     │
│  │     per_frame_deltas: Vec<i64>,               │     │
│  │     resource_map: HashMap<wgpu::Id,           │     │
│  │                        ResourceInfo>,         │     │
│  │ }                                            │     │
│  │                                               │     │
│  │ TextureTracker { ... }                       │     │
│  │ BindGroupTracker { ... }                     │     │
│  │ SamplerTracker { ... }                       │     │
│  │ PipelineTracker { ... }                      │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Budget enforcement:                                   │
│  ├── Configurable thresholds per resource type         │
│  ├── Warning when approaching budget                   │
│  └── Violation callback (log, overlay, crash)          │
│                                                       │
│  Features:                                            │
│  ├── Per-frame delta tracking                         │
│  ├── Leak detection (allocated but not freed)         │
│  ├── Categorized by system (renderer, editor, etc.)   │
│  └── Snapshot comparison (before/after scene load)    │
└──────────────────────────────────────────────────────┘
```

## Architecture: Frame-Perfect Profiler (Gap)

```
┌──────────────────────────────────────────────────────┐
│  FrameProfiler                                        │
│                                                       │
│  300-frame ring buffer:                               │
│  ┌──────────────────────────────────────────────┐     │
│  │ struct FrameProfiler {                       │     │
│  │     buffer: [Option<FrameProfile>; 300],     │     │
│  │     write_index: usize,                      │     │
│  │     frame_count: u64,                        │     │
│  │ }                                            │     │
│  │                                               │     │
│  │ struct FrameProfile {                        │     │
│  │     frame_index: u64,                        │     │
│  │     frame_type: FrameType,                   │     │
│  │     cpu_start: Instant,                      │     │
│  │     cpu_duration: Duration,                  │     │
│  │     gpu_duration: Option<Duration>,          │     │
│  │     passes: Vec<PassTiming>,                 │     │
│  │     memory_allocated: u64,                   │     │
│  │     draw_calls: u32,                         │     │
│  │     triangles: u32,                          │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  FrameType discrimination:                            │
│  ┌──────────────────────────────────────────────┐     │
│  │ enum FrameType {                             │     │
│  │     Normal,                                  │     │
│  │     Loading,                                 │     │
│  │     Editor,       // viewport inactive       │     │
│  │     Shadow,       // shadow map render       │     │
│  │     Reflection,   // reflection probe        │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Methods:                                             │
│  ├── begin_frame(frame_type) -> FrameHandle           │
│  ├── end_frame(handle)                                │
│  ├── begin_pass(name)                                 │
│  ├── end_pass()                                       │
│  ├── record_gpu_timings(query_results)                 │
│  ├── get_frame_history(n) -> Vec<FrameProfile>        │
│  └── get_latest_frame() -> Option<FrameProfile>       │
└──────────────────────────────────────────────────────┘
```

## Architecture: Chrome Tracing Export (Gap)

```
┌──────────────────────────────────────────────────────┐
│  ChromeTraceExporter                                  │
│                                                       │
│  Serializes FrameProfiler ring buffer to              │
│  Chrome Trace Event Format (JSON):                    │
│                                                       │
│  {                                                    │
│    "traceEvents": [                                   │
│      {                                                │
│        "name": "OpaquePass",                          │
│        "cat": "gpu",                                  │
│        "ph": "X",           // Complete event         │
│        "ts": 1234567890,    // microseconds           │
│        "dur": 1234,         // duration in us         │
│        "pid": 1,            // process id             │
│        "tid": 1,            // thread id              │
│        "args": {                                      │
│          "frame_type": "normal",                      │
│          "draw_calls": 42                             │
│        }                                              │
│      },                                               │
│      ...                                              │
│    ],                                                 │
│    "displayTimeUnit": "ms",                           │
│    "systemTraceEvents": []                            │
│  }                                                    │
│                                                       │
│  Export triggers:                                     │
│  ├── Profiler UI "Export" button                      │
│  ├── Programmatic API for CI/CD                       │
│  └── Automatic on frame spike (>16ms)                 │
└──────────────────────────────────────────────────────┘
```

## Architecture: Waterfall Visualization (Gap)

```
┌──────────────────────────────────────────────┐
│  Waterfall Timeline (viewport overlay)        │
│                                               │
│  F5 toggle:                                   │
│  ┌──────────────────────────────────────┐     │
│  │ Frame 12345          CPU  ████████   │     │
│  │ ─────────────────────────────────    │     │
│  │ ShadowPass          GPU  ████       │     │
│  │ OpaquePass          GPU  █████████  │     │
│  │ TransparentPass     GPU     ████    │     │
│  │ PostProcess         GPU       ██    │     │
│  │ UI Overlay          CPU  ██         │     │
│  │                     ──────────────  │     │
│  │                     0  4  8  12  16ms│     │
│  │                                     │     │
│  │ [<] Frame 12344  [>] Frame 12346   │     │
│  │ Budget: [████████░░░░] 8.2/16.0ms  │     │
│  └──────────────────────────────────────┘     │
│                                               │
│  Features:                                    │
│  ├── GPU/CPU timeline side by side             │
│  ├── Pass-level breakdown bars                │
│  ├── Color-coded by pass type                 │
│  ├── Frame navigation (left/right arrows)     │
│  └── Budget bar with violation highlight      │
└──────────────────────────────────────────────┘
```

## Data Flow

```
wgpu Device/Queue
  │ timestamp queries, resource allocation
  ▼
┌──────────────────────────────────────────────┐
│  GPUQueryManager        GpuMemoryTracker      │
│  (Phase 3 / T-TL-3.6)  (T-TL-4.2)            │
│         │                      │              │
│         ▼                      ▼              │
│  FrameProfiler (T-TL-4.3)                     │
│  ├── Collects GPU timings                     │
│  ├── Collects CPU timings                     │
│  └── Collects memory stats                    │
│         │                                     │
│         ▼                                     │
│  300-frame ring buffer                        │
│         │                                     │
│         ├──► ChromeTraceExporter (T-TL-4.4)   │
│         ├──► BudgetSystem (T-TL-4.5)          │
│         └──► WaterfallOverlay (T-TL-4.6)      │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  Overlay rendering via EguiUIContext          │
│  RenderDoc capture via wgpu::RenderDoc        │
└──────────────────────────────────────────────┘
```

## Dependencies

- **Blocked on Phase 3** (T-TL-3.5 viewport scene rendering, T-TL-3.6 GPUQueryManager).
- T-TL-4.1 RenderDoc: depends on wgpu::RenderDoc (available in wgpu 0.19+).
- T-TL-4.2 Memory tracker: can wrap wgpu resource allocation callbacks.
- T-TL-4.3 Frame profiler: depends on GPUQueryManager for GPU timings.
- T-TL-4.5 Budget system: depends on frame profiler data.
- T-TL-4.6 Waterfall: depends on frame profiler + DebugOverlayController.

## Implementation Order

1. T-TL-4.1: RenderDoc initialization and frame graph annotation
2. T-TL-4.3: FrameProfiler with 300-frame ring buffer
3. T-TL-4.2: GPU memory tracker wrapping wgpu allocations
4. T-TL-4.4: Chrome tracing export from ring buffer
5. T-TL-4.5: Profiling budget system with violation detection
6. T-TL-4.6: Waterfall visualization overlay

## Success Criteria

- RenderDoc captures frames on F11 with debug markers visible
- GPU memory tracker reports per-resource-type allocation with per-frame deltas
- FrameProfiler maintains 300-frame rolling history
- Chrome trace file loads correctly in chrome://tracing
- Budget violations trigger visible overlay warnings
- Waterfall visualization shows CPU/GPU timeline with pass breakdown
