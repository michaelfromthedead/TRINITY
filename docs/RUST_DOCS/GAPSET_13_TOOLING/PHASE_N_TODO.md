# GAPSET_13_TOOLING: Task Breakdown

## Task ID Convention

Format: `T-TL-{PHASE}.{N}` where:
- `T` = Task
- `TL` = Tooling gap set
- `{PHASE}` = Phase number (1-9)
- `{N}` = Task number within phase

Priority markers:
- **P0:** Blocking -- must complete before next phase can start
- **P1:** Critical -- core functionality
- **P2:** Important -- significant feature
- **P3:** Nice-to-have -- completion/polish

---

## Phase 1: Foundation Infrastructure (P0)

**Bridge expansion, EguiUIContext adapter, GPU instrumentation.**

| Task ID | Priority | Gap | Description | Effort | Dependencies |
|---------|----------|-----|-------------|--------|--------------|
| T-TL-1.1 | P0 | S18-G3 | Design and implement Type/Data/Command three-channel bridge protocol. Define schema for all 15+ function endpoints. | 2w | None |
| T-TL-1.2 | P0 | S18-G3 | Expand bridge from 7 to 15+ functions: add `bridge.*`, `entity.*`, `component.*`, `frame.*`, `profiler.*`, `editor.*`, `material.*`, `asset.*` namespaces. | 2w | T-TL-1.1 |
| T-TL-1.3 | P0 | S18-G4 | Build EguiUIContext adapter: Python UIContext protocol implementation in Rust via PyO3, mapping Python UI calls to egui::Ui. | 3w | T-TL-1.2 |
| T-TL-1.4 | P0 | S17-G1 | Implement `@gpu_profile` decorator: wgpu::QuerySet management, timestamp insertion, pool lifecycle, deferred readback (1-3 frame), results wiring to DebugOverlayController. | 2w | None (wgpu available) |
| T-TL-1.5 | P1 | S18-G3 | Bridge error handling: typed error responses, timeout handling, connection health monitoring, reconnection logic. | 1w | T-TL-1.2 |
| T-TL-1.6 | P1 | S18-G4 | EguiUIContext input handling: keyboard, mouse, modifier keys, window events mapped through Python protocol to egui. | 1w | T-TL-1.3 |

**Phase 1 total: ~11 weeks**

---

## Phase 2: Core Editor Panels (P1)

**Inspector, Hierarchy, EditorCamera, REPL enhancements.**

| Task ID | Priority | Gap | Description | Effort | Dependencies |
|---------|----------|-----|-------------|--------|--------------|
| T-TL-2.1 | P1 | S18-G5 | Implement EditorCamera: orbit/pan/zoom, WASD/fly controls, focus-on-entity (F key), save/restore camera pose. Pure math (glam/nalgebra). | 2w | T-TL-1.2 |
| T-TL-2.2 | P1 | S18-G6 | Build Inspector panel: JSON rendering with type-aware formatting, collapsible components, inline editing of primitive fields, SelectionState integration. | 2w | T-TL-1.2, T-TL-1.3 |
| T-TL-2.3 | P1 | S18-G7 | Build Hierarchy panel: tree view from SQLite parent_id, expand/collapse, drag-drop reparenting, right-click context menu, search/filter, anti-cycle enforcement. | 2w | T-TL-1.2, T-TL-1.3 |
| T-TL-2.4 | P1 | S18-G6 | Inspector moldable views: type-specific rendering for components (transform, material, mesh, camera, light, physics). | 2w | T-TL-2.2 |
| T-TL-2.5 | P2 | S18-G6 | Inspector multi-select and comparison: select two entities, diff components side by side. | 1w | T-TL-2.2 |
| T-TL-2.6 | P2 | --- | REPL enhancements: multiline input, history persistence to SQLite, syntax highlighting, output formatting (rich text, error highlighting). | 1w | T-TL-1.3 |
| T-TL-2.7 | P2 | --- | SelectionState shared data structure: selected_entity_id, hovered_entity_id, gizmo_mode, gizmo_space -- wired to all panels. | 1w | T-TL-2.1, T-TL-2.2, T-TL-2.3 |

**Phase 2 total: ~11 weeks**

---

## Phase 3: GPU Foundation (P2)

**wgpu viewport, GPU timestamps, 14 debug visualization passes.**

**BLOCKED on S1-S9 (GPU Pipeline) and S10+ (Frame Graph).**

| Task ID | Priority | Gap | Description | Effort | Dependencies |
|---------|----------|-----|-------------|--------|--------------|
| T-TL-3.1 | P1 | S18-G2 | wgpu-egui integration: surface creation, resize handling, swapchain management, wgpu context sharing with engine. | 2w | S1-S9, T-TL-1.3 |
| T-TL-3.2 | P1 | S18-G2 | 4-up viewport layout: Perspective, Top, Front, Side. Orthographic camera projections for 3 auxiliary views. Grid, axis lines, selection highlight. | 2w | T-TL-3.1, T-TL-2.1 |
| T-TL-3.3 | P1 | S18-G2 | Gizmo rendering: 2D projected translate/rotate/scale handles in viewport. Mouse interaction for gizmo manipulation. | 2w | T-TL-3.2 |
| T-TL-3.4 | P1 | S18-G2 | GPU picking: render entity IDs to offscreen texture, readback pixel under cursor, resolve to entity ID. Selection highlight in viewport. | 1w | T-TL-3.2 |
| T-TL-3.5 | P1 | S18-G2 | 3D scene rendering in viewport: submit scene draw calls to viewport's wgpu surface. Share frame graph output. | 2w | T-TL-3.1, S10+ |
| T-TL-3.6 | P1 | S17-G4 | GPUQueryManager: wgpu::QuerySet pool (TIMESTAMP type), per-pass begin/end pairs, deferred readback, results to FrameProfile. | 2w | T-TL-1.4, T-TL-3.5 |
| T-TL-3.7 | P2 | S17-G2 | Implement debug visualization passes (14 total): GBuffer views (depth, normal, Albedo, metallic, roughness, AO, emissive, velocity), lighting views (direct, indirect, shadows), post-fx views (bloom, SSAO, SSR), wireframe overlay. | 4w | S10+, T-TL-3.6 |
| T-TL-3.8 | P2 | S17-G2 | DebugOverlayController API: set_visualization, add_visualization, remove_visualization, clear, set_heatmap_scale, set_opacity. Hotkeys: F2 (toggle), F3 (cycle), F4 (HUD), F5 (waterfall). | 1w | T-TL-3.7 |
| T-TL-3.9 | P2 | S18-G2 | Shadow map visualization toggle in viewport (overlay shadow cascade textures). | 1w | T-TL-3.7 |

**Phase 3 total: ~17 weeks (blocked on S1-S9, S10+)**

---

## Phase 4: Debug Integration (P2)

**RenderDoc, GPU memory tracking, frame-perfect profiling.**

| Task ID | Priority | Gap | Description | Effort | Dependencies |
|---------|----------|-----|-------------|--------|--------------|
| T-TL-4.1 | P2 | S17-G6 | RenderDoc integration: wgpu::RenderDoc::get() initialization, frame graph node annotation (push/pop markers), F11 capture trigger. Conditional no-op. | 1w | T-TL-3.5 |
| T-TL-4.2 | P2 | S17-G7 | GPU memory tracker: per-resource-type allocation tracking, per-frame deltas, budget enforcement (configurable thresholds). | 2w | T-TL-3.5 |
| T-TL-4.3 | P1 | S17-G5 | Frame-perfect profiler: 300-frame ring buffer of FrameProfile structs, frame type discrimination (normal, loading, editor, shadow, reflection). | 2w | T-TL-3.6 |
| T-TL-4.4 | P1 | S17-G5 | Chrome tracing export: serialize ring buffer to Chrome trace JSON format. Export triggered from profiler UI. | 1w | T-TL-4.3 |
| T-TL-4.5 | P2 | S17-G5 | Profiling budget system: configurable per-pass CPU/GPU budgets, violation detection, overlay warnings. | 1w | T-TL-4.3 |
| T-TL-4.6 | P2 | S17-G5 | Waterfall visualization: per-frame GPU/CPU timeline view as viewport overlay (F5 hotkey). | 2w | T-TL-4.3, T-TL-3.8 |

**Phase 4 total: ~9 weeks**

---

## Phase 5: Advanced Editor (P2)

**Material Editor, Asset Browser, Profiler Overlay.**

| Task ID | Priority | Gap | Description | Effort | Dependencies |
|---------|----------|-----|-------------|--------|--------------|
| T-TL-5.1 | P1 | S18-G8 | Material DSL compiler: Python DSL parser (asteval or custom), AST construction, WGSL code generation, naga validation integration. | 3w | T-TL-1.2, T-TL-1.3 |
| T-TL-5.2 | P1 | S18-G8 | Material Editor panel: syntax-highlighted text editor, compilation error display (inlined), parameter sliders, preview sphere in viewport. | 2w | T-TL-5.1, T-TL-3.5 |
| T-TL-5.3 | P1 | S18-G9 | Asset Browser panel: filesystem tree, file type filters, icon rendering, search/filter, drag-drop import. | 2w | T-TL-1.3 |
| T-TL-5.4 | P2 | S18-G9 | Thumbnail cache: automatic texture thumbnail generation on import, mipmap-based downscaling, LRU eviction. | 2w | T-TL-5.3, T-TL-3.1 |
| T-TL-5.5 | P2 | S18-G10 | Profiler Overlay panel: in-editor frame timing graph (sparkline), per-pass breakdown, budget bar, frame rate history, memory usage, tooltips. | 2w | T-TL-4.3, T-TL-4.5 |
| T-TL-5.6 | P2 | S18-G8 | Material library: save/load materials from asset library, material parameter inheritance, material instances. | 2w | T-TL-5.1, T-TL-5.3 |
| T-TL-5.7 | P3 | S18-G8 | Material node graph visualization: convert text DSL to visual node graph (read-only initially, editable later). | 2w | T-TL-5.1 |

**Phase 5 total: ~15 weeks**

---

## Phase 6: Grail Debug Features (P3)

**Time-travel debugging, shader debug compilation.**

**BLOCKED on S15 (Core ECS) for time-travel state capture.**

| Task ID | Priority | Gap | Description | Effort | Dependencies |
|---------|----------|-----|-------------|--------|--------------|
| T-TL-6.1 | P2 | S17-G8 | Shader debug compiler: WGSL source -> naga IR dump -> backend bytecode visualization. Error annotation with source lines. | 2w | T-TL-5.1 |
| T-TL-6.2 | P2 | S17-G8 | Shader reflection browser: bind group layout, vertex attributes, push constants, pipeline statistics display. | 1w | T-TL-6.1 |
| T-TL-6.3 | P3 | S17-G3 | Input log recorder: capture all inputs (keyboard, mouse, window events) with tick counters. Deterministic replay foundation. | 2w | T-TL-3.5 |
| T-TL-6.4 | P3 | S17-G3 | Snapshot manager: periodic ECS world snapshots (every 60 ticks), GPU resource state capture (readback), snapshot storage. | 3w | T-TL-6.3, S15 |
| T-TL-6.5 | P3 | S17-G3 | Binary search navigator: FIND_NEAREST_SNAPSHOT_BEFORE, restore and replay from snapshot to target tick, hash verification. | 2w | T-TL-6.4 |
| T-TL-6.6 | P3 | S17-G3 | Time-travel UI: timeline bar with tick marks, play/pause, step forward/backward, tick counter display. Viewport shows replayed state. | 2w | T-TL-6.5, T-TL-3.5 |
| T-TL-6.7 | P3 | S17-G3 | Causal chain preservation: ensure EventLog events maintain causal links across replay. | 2w | T-TL-6.5 |
| T-TL-6.8 | P3 | S17-G3 | Fix-and-continue: edit shader/material during replay, hot-reload, continue replay with modified code. | 3w | T-TL-6.6, T-TL-5.2 |
| T-TL-6.9 | P3 | S17-G3 | Debugger breakpoints: conditional breakpoints on specific ticks, entity state match, watch expressions on component fields. | 2w | T-TL-6.6 |

**Phase 6 total: ~19 weeks (partially blocked on S15)**

---

## Phase 7: Pharo Environment (P3)

**Live environment pillars.**

| Task ID | Priority | Gap | Description | Effort | Dependencies |
|---------|----------|-----|-------------|--------|--------------|
| T-TL-7.1 | P2 | S18-G11 | Foundation Serializer: session persistence to SQLite (evolves to Foundation format), save/restore editor state, panel layout, REPL history, auto-save, crash recovery. | 3w | T-TL-1.3, T-TL-2.2, T-TL-2.3, T-TL-2.6 |
| T-TL-7.2 | P2 | S18-G11 | Mirror implementation: uniform reflection API over Python objects (inspect), Rust objects (PyO3 introspection), ECS components (component registry). Query interfaces across all object types. | 3w | T-TL-1.2, S15 |
| T-TL-7.3 | P2 | S18-G11 | Moldable inspectors: data-driven visualization templates per object type (list, graph, image, code, timeline, custom). | 2w | T-TL-7.2, T-TL-2.2 |
| T-TL-7.4 | P3 | S18-G11 | Spotter (universal search): Ctrl+P search across entities, components, assets, settings, commands. Regex mode. | 2w | T-TL-7.2 |
| T-TL-7.5 | P3 | S18-G11 | Change management: undo/redo for editor operations (entity create/delete, component edits, scene reparent). | 2w | T-TL-2.2, T-TL-2.3 |
| T-TL-7.6 | P3 | S18-G11 | Finder (query-by-example): find entities by component pattern. Save searches as smart folders. Full-text search. | 2w | T-TL-7.2 |
| T-TL-7.7 | P3 | S18-G11 | Pharo Shell: enhanced REPL with workspace support, multi-file execution, script save/load, interactive object display. | 2w | T-TL-2.6, T-TL-7.2 |

**Phase 7 total: ~16 weeks**

---

## Phase 8: FlowForge Visual Scripting (P3)

**Node graph visual scripting.**

| Task ID | Priority | Gap | Description | Effort | Dependencies |
|---------|----------|-----|-------------|--------|--------------|
| T-TL-8.1 | P3 | S18-G12 | FlowForge node graph editor: egui graph widget, node palette, drag-drop connections, property panel, categorized node types. | 4w | T-TL-1.3 |
| T-TL-8.2 | P3 | S18-G12 | Node type implementation: Events, Actions, Conditions, Math, Flow, ECS, Debug nodes. 40+ node implementations. | 4w | T-TL-8.1 |
| T-TL-8.3 | P3 | S18-G12 | Python bytecode compilation: FlowForge graph -> Python AST -> bytecode. Interpreted execution mode. | 3w | T-TL-8.2 |
| T-TL-8.4 | P3 | S18-G12 | Sub-graph macros: reusable sub-graphs as macro nodes. Parameterized inputs/outputs. Node library and categorization. | 2w | T-TL-8.2 |
| T-TL-8.5 | P3 | S18-G12 | FlowForge entity integration: attach FlowForge scripts to ECS entities, trigger on events (begin, update, collision, input). | 2w | T-TL-8.2, S15 |

**Phase 8 total: ~15 weeks (partially blocked on S15)**

---

## Phase 9: Polish & Hardening (P3)

| Task ID | Priority | Gap | Description | Effort | Dependencies |
|---------|----------|-----|-------------|--------|--------------|
| T-TL-9.1 | P2 | --- | Error boundary system: isolate each panel in its own error scope. One panel crash does not crash editor. | 2w | All phases |
| T-TL-9.2 | P3 | --- | Panel layout persistence: save/restore docking, sizing, visibility per session. | 2w | T-TL-7.1 |
| T-TL-9.3 | P3 | --- | Theme system: light/dark mode, custom theme support, consistent color tokens across all panels. | 2w | T-TL-1.3 |
| T-TL-9.4 | P3 | --- | Keyboard shortcut system: customizable shortcuts, shortcut conflict detection, cheat sheet UI. | 1w | All phases |
| T-TL-9.5 | P3 | --- | Editor performance profiling: track editor frame time, identify slow panels, optimize render loops. | 2w | All phases |
| T-TL-9.6 | P3 | --- | First-run tutorial: guided onboarding for panel layout and core workflows. | 1w | All phases |

**Phase 9 total: ~10 weeks**

---

## Summary

| Phase | Tasks | Total Effort | Blocked By | Parallelizable |
|-------|-------|-------------|------------|----------------|
| 1: Foundation | 6 | 11w | None | Yes |
| 2: Core Panels | 7 | 11w | Phase 1 | Yes |
| 3: GPU Foundation | 9 | 17w | S1-S9, S10+ | Partial |
| 4: Debug Integration | 6 | 9w | Phase 3 | Yes |
| 5: Advanced Editor | 7 | 15w | Phase 2, 3 | Partial |
| 6: Grail Debug | 9 | 19w | S15, Phase 3 | Partial |
| 7: Pharo Environment | 7 | 16w | Phase 1, 2, S15 | Partial |
| 8: FlowForge | 5 | 15w | Phase 1, S15 | Yes |
| 9: Polish | 6 | 10w | All prior | Yes |
| **Total** | **62** | **123w** | | |

### Adjusted for Swarm Parallelization

With 4-6 parallel agents working across independent phases:

| Wave | Phases | Workers | Wall Time |
|------|--------|---------|-----------|
| Wave A | Phase 1 + Phase 2 + Phase 8 | 3 | ~11w |
| Wave B | Phase 3 + Phase 7 (partial) | 2 | ~17w* |
| Wave C | Phase 4 + Phase 5 | 3 | ~12w |
| Wave D | Phase 6 | 2 | ~10w* |
| Wave E | Phase 9 | 2 | ~5w |

**Estimated wall time: 28-36 weeks** (swarm-parallel, dependent on S1-S9 and S15 availability)

*Blocked on external dependencies (S1-S9 GPU pipeline, S15 Core ECS)
