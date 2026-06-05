# PROJECT.md — GAPSET_13_TOOLING Project Structure

## Repository Layout

```
TRINITY/
├── engine/tooling/                    # Python tooling layer (20 subsystems, ~130 files)
│   ├── __init__.py
│   ├── TOOLING_CONTEXT.md             # Comprehensive implementation reference (81KB)
│   ├── animation_tools/               # Sequencer, skeleton editor, IK, curves, montage
│   ├── assettools/                    # Import pipeline, browser, search, thumbnails
│   ├── automation/                    # CI/CD, build agents, commandlets, Python API
│   ├── build/                         # Build pipeline, cache, config, cook, packaging
│   ├── console/                       # Console UI, commands, CVar system, history
│   ├── crash/                         # Crash reporter, analytics, assertions, upload
│   ├── debug/                         # Debug camera, console, draw, overlays, watch
│   ├── editor/                        # App shell, commands, gizmos, modes, plugins
│   ├── hotreload/                     # Hot-reload, module watcher, state preservation
│   ├── leveleditor/                   # Hierarchy, placement, snapping, prefabs, layers
│   ├── localization/                  # String table, translation, dashboard
│   ├── logging/                       # Structured logging, targets, filters
│   ├── material_editor/               # Node graph, compiler, library, parameters
│   ├── profiling/                     # CPU/GPU/memory/network profilers, export
│   ├── replay/                        # Input recording, state capture, timeline
│   ├── terrain/                       # Sculpt, paint, foliage, LOD, import
│   ├── testing/                       # Test framework, assertions, fixtures, mocking
│   ├── undo/                          # Undo/redo, command pattern, transactions
│   ├── vcs/                           # Git/Perforce providers, merge, lock
│   └── visual_scripting/              # Blueprint compiler, graph editor, node types
│
├── tests/tooling/                     # Test suite mirroring engine/tooling/ structure
│   ├── __init__.py
│   ├── animation_tools/               # 9 test files (test_anim_graph_editor, etc.)
│   ├── assettools/                    # 9 test files
│   ├── automation/                    # 5 test files
│   ├── build/                         # 7 test files
│   ├── console/                       # 4 test files
│   ├── crash/                         # 4 test files
│   ├── debug/                         # 9 test files
│   ├── editor/                        # 9 test files
│   ├── hotreload/                     # 6 test files
│   ├── leveleditor/                   # 9 test files
│   ├── localization/                  # 6 test files
│   ├── logging/                       # 5 test files
│   ├── material_editor/               # 9 test files
│   ├── profiling/                     # 9 test files
│   ├── replay/                        # 9 test files
│   ├── terrain/                       # 7 test files
│   ├── testing/                       # 6 test files
│   ├── undo/                          # 5 test files
│   ├── vcs/                           # 6 test files
│   └── visual_scripting/             # 9 test files
│
├── crates/renderer-backend/src/       # Rust renderer-backend crate
│   ├── editor.rs                      # Editor struct, EditorState, selection (265 lines)
│   ├── bridge.rs                      # PyO3 bridge module (stub, 10 lines)
│   ├── lib.rs                         # Module declarations, re-exports
│   └── ...                            # GPU-driven, frame graph, renderer, etc.
│
├── flowforge/apps/desktop/src-tauri/  # Tauri desktop application
│   ├── src/
│   │   ├── main.rs                    # App entry, command registration
│   │   ├── bridge_protocol.rs         # 4-channel bridge protocol (2188 lines)
│   │   ├── sidecar/mod.rs             # Python sidecar process management (419 lines)
│   │   ├── commands/
│   │   │   ├── mod.rs                 # Command module declarations
│   │   │   ├── editor.rs              # Open in editor, detect editors (246 lines)
│   │   │   ├── trinity.rs             # Trinity introspection commands (918 lines)
│   │   │   ├── python.rs              # Python file parsing
│   │   │   ├── assets.rs              # Asset import/URL commands
│   │   │   ├── nodes.rs               # FlowForge node search/definition
│   │   │   ├── files.rs               # File I/O commands
│   │   │   ├── workflow.rs            # Workflow execution
│   │   │   ├── system.rs              # System info, ping
│   │   │   ├── codegen.rs             # Code generation
│   │   │   └── ipc.rs                 # IPC utilities
│   │   ├── plugins/mod.rs             # Plugin system
│   │   └── state/mod.rs               # Application state
│   └── tests/
│       ├── blackbox_bridge_contract.rs # Bridge protocol contract tests
│       └── sidecar_tests.rs            # Sidecar integration tests
│
└── docs/gap_sets/GAPSET_13_TOOLING/   # This gap set documentation
    ├── PHASE_N_TODO.md                 # RDC-annotated task breakdown
    ├── GAP_13_SUMMARY.md               # Gap analysis summary
    ├── PROJECT.md                      # This file
    ├── CLARIFICATION.md                # Clarification requests
    ├── PHASE_1_ARCH.md                 # Foundation Infrastructure architecture
    ├── PHASE_2_ARCH.md                 # Core Editor Panels architecture
    ├── PHASE_3_ARCH.md                 # GPU Foundation architecture
    ├── PHASE_4_ARCH.md                 # Debug Integration architecture
    ├── PHASE_5_ARCH.md                 # Advanced Editor architecture
    ├── PHASE_6_ARCH.md                 # Grail Debug architecture
    ├── PHASE_7_ARCH.md                 # Pharo Environment architecture
    ├── PHASE_8_ARCH.md                 # FlowForge architecture
    └── PHASE_9_ARCH.md                 # Polish & Hardening architecture
```

## Key Architecture: Three-Tier Tooling System

```
┌─────────────────────────────────────────────────────────┐
│                   TIER 1: Python Tools                   │
│                  (engine/tooling/*.py)                    │
│                                                          │
│  Editor │ Level Editor │ Debug │ Console │ Profiling     │
│  Assets │ Material Ed  │ Anim  │ Terrain │ Visual Script │
│  Build  │ VCS │ HotReload │ Undo │ Loc │ Automation     │
│                                                          │
│  Status: [x] COMPLETE — 20 subsystems, ~130 files        │
└──────────────────────┬──────────────────────────────────┘
                       │ Foundation Bridge (JSON-RPC)
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  TIER 2: Rust Bridge                      │
│             (flowforge/src-tauri/*.rs)                    │
│                                                          │
│  bridge_protocol.rs — 4 channels, 22 endpoints           │
│  trinity.rs — 7 introspection commands                   │
│  sidecar/mod.rs — Python process lifecycle               │
│  editor.rs — Open in external editors                    │
│                                                          │
│  Status: [x] MOSTLY COMPLETE — protocol + commands done  │
└──────────────────────┬──────────────────────────────────┘
                       │ wgpu + egui integration (GAP)
                       ▼
┌─────────────────────────────────────────────────────────┐
│              TIER 3: Engine Integration                   │
│           (crates/renderer-backend/src/)                  │
│                                                          │
│  editor.rs — Editor + EditorState                        │
│  [GAP] EguiUIContext adapter (PyO3 -> egui)             │
│  [GAP] wgpu-egui viewport                               │
│  [GAP] GPUQueryManager (wgpu timestamps)                 │
│  [GAP] 14 debug visualization passes                    │
│  [GAP] RenderDoc integration                             │
│                                                          │
│  Status: [~] PARTIAL — editor.rs only                    │
└─────────────────────────────────────────────────────────┘
```

## Python Subsystem Details

Each subsystem in `engine/tooling/` follows a consistent pattern:
- `__init__.py` with package exports
- Feature-specific Python files (400-700 lines each)
- Corresponding test files in `tests/tooling/`
- Integration with Foundation decorator system (Tracker, EventLog, Mirror, Bridge)

## Rust Crate Dependencies

```
flowforge (Tauri app)
  ├── tauri (desktop framework)
  ├── serde / serde_json (bridge protocol)
  ├── tracing (logging)
  └── thiserror (error types)

renderer-backend
  ├── wgpu (GPU API)
  ├── parking_lot (RwLock)
  └── ... (other engine modules)
```
