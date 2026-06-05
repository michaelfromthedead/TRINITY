# PHASE_1_ARCH.md — Foundation Infrastructure Architecture

## Overview

Phase 1 establishes the communication infrastructure between the Python tooling layer, the Rust/egui editor, and the wgpu renderer. It covers bridge protocol expansion, the EguiUIContext adapter, and GPU instrumentation plumbing.

## Current State

| Task | Status | What Exists | What's Missing |
|------|--------|-------------|----------------|
| T-TL-1.1 | [x] DONE | Bridge protocol: 22 endpoints, 4 channels, full serde, 1300 test lines | Nothing |
| T-TL-1.2 | [x] DONE | All 22 endpoints (type/data/command/system), Trinity introspection commands | Nothing |
| T-TL-1.3 | [~] PARTIAL | bridge.rs stub with TODO comments (8 lines) | Full PyO3 egui adapter |
| T-TL-1.4 | [~] PARTIAL | Python @gpu_profile decorator in gpu_profiler.py | wgpu QuerySet management |
| T-TL-1.5 | [x] DONE | Typed error responses, sidecar restart_if_crashed, retry logic | Nothing |
| T-TL-1.6 | [-] NOT STARTED | Nothing | egui input mapping |

## Architecture: Bridge Protocol (Already Implemented)

```
┌──────────────────────────────────────────────────────────────┐
│                      Bridge Protocol                          │
│              flowforge/src-tauri/src/bridge_protocol.rs        │
│                                                              │
│  TYPE CHANNEL (5)        DATA CHANNEL (5)                    │
│  ┌─────────────────┐    ┌─────────────────┐                  │
│  │ type.register   │    │ data.read       │                  │
│  │ type.list       │    │ data.write      │                  │
│  │ type.get        │    │ data.delete     │                  │
│  │ type.remove     │    │ data.batch_read │                  │
│  │ type.count      │    │ data.batch_write│                  │
│  └─────────────────┘    └─────────────────┘                  │
│                                                              │
│  COMMAND CHANNEL (6)      SYSTEM CHANNEL (6)                 │
│  ┌─────────────────┐    ┌─────────────────┐                  │
│  │ command.create  │    │ system.connect  │                  │
│  │ command.spawn   │    │ system.status   │                  │
│  │ command.despawn │    │ system.inspect  │                  │
│  │ command.query   │    │ system.inspector_get              │
│  │ command.reset   │    │ system.events_recent              │
│  │ command.stats   │    │ system.checksum │                  │
│  └─────────────────┘    └─────────────────┘                  │
│                                                              │
│  Transport: JSON-RPC 2.0 over stdin/stdout (sidecar)         │
│  Routing: METHOD_TABLE, channel_for_method()                 │
│  Tests: ~1300 lines including W1-W10 whitebox sections       │
└──────────────────────────────────────────────────────────────┘
```

## Architecture: Sidecar Process Lifecycle (Already Implemented)

```
┌────────────────────────────┐
│      Tauri App (Rust)      │
│                            │
│  PythonSidecar              │
│  ├── spawn()               │
│  │   └── child process     │
│  ├── send_request()        │
│  │   └── stdin JSON-RPC    │
│  ├── is_running()          │
│  ├── restart_if_crashed()  │
│  └── shutdown()            │
│                            │
│  SidecarState (managed)     │
│  ├── auto-restart logic     │
│  └── connection health      │
└────────────────────────────┘
         │ stdin/stdout JSON-RPC
         ▼
┌────────────────────────────┐
│  Python Trinity Runtime     │
│  (engine/tooling/)          │
│                            │
│  Handles:                  │
│  - type.register calls     │
│  - data.read/write ops     │
│  - command.spawn/despawn   │
│  - system.inspect/status   │
└────────────────────────────┘
```

## Architecture: EguiUIContext Adapter (Gap — Needs Implementation)

This is the primary remaining work in Phase 1. The adapter must bridge Python UI tool calls to egui::Ui rendering.

```
┌────────────────────────────────────────┐
│          Python Tool (existing)         │
│                                        │
│  editor/viewport.py                    │
│  editor/inspector.py                   │
│  editor/hierarchy.py                   │
│  debug/debug_overlays.py               │
│  profiling/profiler_overlay.py         │
│                                        │
│  Output: UI commands via JSON-RPC      │
│  {method: "ui.label", params: {...}}   │
└──────────────┬─────────────────────────┘
               │ JSON-RPC over sidecar
               ▼
┌────────────────────────────────────────┐
│      EguiUIContext (Rust — NEW)         │
│                                         │
│  Receives UI commands from Python       │
│  Maps to egui::Ui rendering calls       │
│                                         │
│  Protocol methods:                      │
│  ┌─────────────────────────────────┐   │
│  │ ui.label(text)                  │   │
│  │ ui.button(text) -> clicked      │   │
│  │ ui.input(label) -> text         │   │
│  │ ui.tree_node(label) -> open     │   │
│  │ ui.collapsible(label) -> open   │   │
│  │ ui.separator()                  │   │
│  │ ui.group(label)                 │   │
│  │ ui.slider(label, min, max) -> f │   │
│  │ ui.color_picker(label) -> rgba  │   │
│  │ ui.image(texture_id, size)      │   │
│  │ ui.viewport(content_rect)       │   │
│  └─────────────────────────────────┘   │
└──────────────┬─────────────────────────┘
               │ egui::Ui calls
               ▼
┌────────────────────────────────────────┐
│          egui/WGPU Surface              │
│                                         │
│  Renders UI to wgpu swapchain           │
│  Handles input events                   │
│  Frame scheduling                       │
└─────────────────────────────────────────┘
```

## Architecture: GPU Instrumentation (Gap — Partial)

```
┌──────────────────────────────────────┐
│  Python @gpu_profile decorator       │
│  (engine/tooling/profiling/)          │
│                                      │
│  GPUProfilerState                    │
│  GPUProfileSample                    │
│  RenderPassType enum                 │
│  profiler logic                      │
└──────┬───────────────────────────────┘
       │ JSON-RPC
       ▼
┌──────────────────────────────────────┐
│  wgpu QuerySet Manager (Rust — NEW)  │
│                                      │
│  wgpu::QuerySet pool (TIMESTAMP)     │
│  Per-pass begin/end pairs            │
│  Deferred readback (1-3 frame delay) │
│  Results → FrameProfile              │
│  Pool lifecycle management           │
└──────────────────────────────────────┘
```

## Dependencies

- **No external dependencies for Phase 1** — bridge protocol and sidecar already work.
- T-TL-1.3 (EguiUIContext) is self-contained: add egui and wgpu dependencies to flowforge.
- T-TL-1.4 (GPU instrumentation) needs wgpu::QuerySet API (available in wgpu 0.19+).

## Implementation Order

1. Add wgpu-egui and egui dependencies to flowforge Cargo.toml
2. Implement EguiUIContext as a JSON-RPC receiver on the sidecar protocol
3. Define the Python-side `UIContext` protocol class in engine/tooling/editor/
4. Implement the protocol methods mapping to egui::Ui
5. Implement wgpu QuerySet pool management in renderer-backend
6. Wire input events through the EguiUIContext (keyboard, mouse, modifiers)
7. Integration test: Python tool -> sidecar -> EguiUIContext -> egui rendering

## Success Criteria

- Python tool can send a `ui.label` command and see text rendered in egui
- Python tool can render a hierarchy tree via `ui.tree_node` commands
- wgpu timestamp queries return valid GPU timing data
- Editor input (keyboard, mouse) is received by both egui and Python tools
- Bridge protocol error handling detects and recovers from sidecar crashes
