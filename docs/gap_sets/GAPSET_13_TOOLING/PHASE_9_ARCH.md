# PHASE_9_ARCH.md — Polish and Hardening Architecture

## Overview

Phase 9 focuses on editor stability, UX polish, and performance: error boundary system (isolated panel crash recovery), panel layout persistence, theme system (light/dark mode), keyboard shortcut system (customizable bindings), editor performance profiling, and first-run tutorial. All tasks are genuinely new with no existing Rust implementation.

## Current State

| Task | Status | What Exists | What's Missing |
|------|--------|-------------|----------------|
| T-TL-9.1 | [-] NOT STARTED | Nothing | Error boundary system |
| T-TL-9.2 | [-] NOT STARTED | Nothing | Panel layout persistence |
| T-TL-9.3 | [-] NOT STARTED | Nothing | Theme system |
| T-TL-9.4 | [~] PARTIAL | Python shortcuts.py | Rust shortcut system |
| T-TL-9.5 | [-] NOT STARTED | Nothing | Editor performance profiling |
| T-TL-9.6 | [-] NOT STARTED | Nothing | First-run tutorial |

## Architecture: Error Boundary System (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Error Boundary System                                │
│                                                       │
│  Each panel runs in its own error scope:              │
│  ┌──────────────────────────────────────────────┐     │
│  │ struct PanelContext {                        │     │
│  │     panel_id: &'static str,                  │     │
│  │     state: PanelState,                       │     │
│  │     error: Option<PanelError>,               │     │
│  │     crash_count: u32,                        │     │
│  │     last_crash: Option<Instant>,             │     │
│  │ }                                            │     │
│  │                                               │     │
│  │ enum PanelState {                            │     │
│  │     Normal,                                  │     │
│  │     Error { message: String, recoverable: bool },│
│  │     Crashed { reason: String },              │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Error boundary wrapper:                              │
│  ┌──────────────────────────────────────────────┐     │
│  │ fn error_boundary(ui: &mut egui::Ui,        │     │
│  │                   panel: &mut dyn EditorPanel, │   │
│  │                   ctx: &mut PanelContext)    │     │
│  │ {                                            │     │
│  │     match ctx.state {                        │     │
│  │         PanelState::Normal => {               │     │
│  │             std::panic::catch_unwind(        │     │
│  │                 || panel.ui(ui)              │     │
│  │             ).unwrap_or_else(|e| {           │     │
│  │                 ctx.state = PanelState::     │     │
│  │                     Crashed {                │     │
│  │                         reason: format!(    │     │
│  │                             "{:?}", e)      │     │
│  │                     };                       │     │
│  │                 render_error_state(ui, ctx); │     │
│  │             });                              │     │
│  │         }                                    │     │
│  │         PanelState::Crashed => {             │     │
│  │             render_error_state(ui, ctx);     │     │
│  │         }                                    │     │
│  │         PanelState::Error { .. } => {        │     │
│  │             render_error_state(ui, ctx);     │     │
│  │         }                                    │     │
│  │     }                                        │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Error state rendering:                               │
│  ┌──────────────────────────────────────────────┐     │
│  │ ┌──────────────────────────────────────┐      │     │
│  │ │  ⚠ Panel Error                      │      │     │
│  │ │                                     │      │     │
│  │ │  Inspector.cpp:42: panic:            │      │     │
│  │ │  index out of bounds                │      │     │
│  │ │                                     │      │     │
│  │ │  [Retry] [Reset Panel] [Report Bug] │      │     │
│  │ └──────────────────────────────────────┘      │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Crash count circuit breaker:                         │
│  ├── N crashes in M seconds → disable panel           │
│  ├── User must manually re-enable                      │
│  └── Prevent crash loops                              │
└──────────────────────────────────────────────────────┘
```

## Architecture: Panel Layout Persistence (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Panel Layout Persistence                             │
│                                                       │
│  Layout serialization format (JSON):                  │
│  ┌──────────────────────────────────────────────┐     │
│  │ {                                            │     │
│  │   "version": 1,                              │     │
│  │   "docks": [                                 │     │
│  │     {"area": "left", "panel": "hierarchy",   │     │
│  │      "visible": true, "size": 250},          │     │
│  │     {"area": "center", "panel": "viewport",  │     │
│  │      "visible": true},                       │     │
│  │     {"area": "right", "panel": "inspector",  │     │
│  │      "visible": true, "size": 300},          │     │
│  │     {"area": "bottom", "panel": "console",   │     │
│  │      "visible": false}                       │     │
│  │   ],                                          │     │
│  │   "floating": [],                             │     │
│  │   "active_panel": "viewport"                  │     │
│  │ }                                             │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Docking areas:                                       │
│  ┌──────────────────────────────────────────────┐     │
│  │  ┌────────┬──────────────────┬────────┐       │     │
│  │  │        │                  │        │       │     │
│  │  │ Left   │    Center        │ Right  │       │     │
│  │  │ Panel  │   (Viewport)     │ Panel  │       │     │
│  │  │        │                  │        │       │     │
│  │  ├────────┴──────────────────┴────────┤       │     │
│  │  │  Bottom Panel (Console)            │       │     │
│  │  └────────────────────────────────────┘       │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Persistence flow:                                    │
│  ├── On layout change → serialize → store in ses.     │
│  ├── On startup → load from Foundation Serializer     │
│  ├── Manual reset → restore default layout             │
│  └── Per-workspace layouts saved separately            │
└──────────────────────────────────────────────────────┘
```

## Architecture: Theme System (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Theme System                                         │
│                                                       │
│  Color tokens (consistent across all panels):         │
│  ┌──────────────────────────────────────────────┐     │
│  │ struct Theme {                              │     │
│  │     name: String,                            │     │
│  │     is_dark: bool,                           │     │
│  │     // Base colors                           │     │
│  │     background: egui::Color32,               │     │
│  │     surface: egui::Color32,                  │     │
│  │     text: egui::Color32,                     │     │
│  │     text_secondary: egui::Color32,           │     │
│  │     // Panel colors                          │     │
│  │     panel_header: egui::Color32,             │     │
│  │     panel_border: egui::Color32,             │     │
│  │     panel_hover: egui::Color32,              │     │
│  │     // Accent colors                          │     │
│  │     accent: egui::Color32,                   │     │
│  │     accent_hover: egui::Color32,             │     │
│  │     accent_active: egui::Color32,            │     │
│  │     // Semantic colors                       │     │
│  │     success: egui::Color32,                  │     │
│  │     warning: egui::Color32,                  │     │
│  │     error: egui::Color32,                    │     │
│  │     info: egui::Color32,                     │     │
│  │     // Code editor                           │     │
│  │     syntax_keyword: egui::Color32,           │     │
│  │     syntax_string: egui::Color32,            │     │
│  │     syntax_comment: egui::Color32,           │     │
│  │     syntax_number: egui::Color32,            │     │
│  │     syntax_type: egui::Color32,              │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Themes:                                               │
│  ├── Dark (default) — dark backgrounds, light text     │
│  ├── Light — light backgrounds, dark text              │
│  └── Custom — user-defined via JSON file               │
│                                                       │
│  Theme application:                                    │
│  ├── egui visual style (colors, rounding, spacing)     │
│  ├── Custom widget styling (node graph, viewport)     │
│  └── All panels use Theme tokens via const lookup      │
└──────────────────────────────────────────────────────┘
```

## Architecture: Keyboard Shortcut System (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Keyboard Shortcut System                             │
│                                                       │
│  Shortcut definition:                                 │
│  ┌──────────────────────────────────────────────┐     │
│  │ struct Shortcut {                            │     │
│  │     id: &'static str,                        │     │
│  │     name: &'static str,                      │     │
│  │     description: &'static str,               │     │
│  │     default_binding: KeyBind,                 │     │
│  │     current_binding: KeyBind,                 │     │
│  │     action: ShortcutAction,                   │     │
│  │     context: ShortcutContext,                 │     │
│  │ }                                            │     │
│  │                                               │     │
│  │ struct KeyBind {                             │     │
│  │     key: KeyCode,                            │     │
│  │     modifiers: Modifiers,  // Ctrl, Shift,   │     │
│  │ }                                            │     │
│  │                                               │     │
│  │ enum ShortcutContext {                       │     │
│  │     Global,      // always active            │     │
│  │     Viewport,    // only in 3D viewport       │     │
│  │     GraphEditor, // only in node graph        │     │
│  │     TextEditor,  // only in code editor       │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Built-in shortcuts:                                  │
│  ┌──────────────────────────────────────────────┐     │
│  │ Ctrl+P    Spotter (universal search)          │     │
│  │ Ctrl+S    Save (session + current asset)      │     │
│  │ Ctrl+Z    Undo                                │     │
│  │ Ctrl+Shift+Z  Redo                            │     │
│  │ F2        Toggle debug visualization          │     │
│  │ F3        Cycle debug visualization           │     │
│  │ F4        Toggle HUD overlay                  │     │
│  │ F5        Toggle waterfall timeline           │     │
│  │ F11       RenderDoc capture                   │     │
│  │ W/E/R     Gizmo mode (translate/rotate/scale) │     │
│  │ F         Focus on selected entity            │     │
│  │ Ctrl+`    Toggle console                      │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Conflict detection:                                  │
│  ├── On shortcut change → check all contexts          │
│  ├── Warn on conflicts                                │
│  └── Priority: context-specific > global              │
│                                                       │
│  Cheat sheet UI:                                      │
│  ├── Ctrl+Shift+/ → overlay with all shortcuts        │
│  ├── Searchable                                       │
│  └── Category groupings                               │
└──────────────────────────────────────────────────────┘
```

## Architecture: Editor Performance Profiling (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Editor Performance Profiling                         │
│                                                       │
│  Track per-frame editor metrics:                      │
│  ┌──────────────────────────────────────────────┐     │
│  │ struct EditorFrameMetrics {                  │     │
│  │     frame_time: Duration,                     │     │
│  │     panel_times: Vec<(&'static str, Duration)>,│   │
│  │     ui_passes: u32,  // egui tessellations    │     │
│  │     memory_allocated: u64,                    │     │
│  │     input_events: u32,                        │     │
│  │     bridge_calls: u32,                        │     │
│  │     viewport_fps: f32,                        │     │
│  │ }                                             │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Slow panel detection:                                │
│  ├── Per-panel timing                                 │
│  ├── Threshold: panel > 4ms = warning                 │
│  ├── Log slow panels to console                       │
│  └── Visual indicator in panel title bar              │
│                                                       │
│  Optimizations:                                       │
│  ├── Lazy panel update (only when visible)            │
│  ├── Panel rendering budget                           │
│  ├── Bridge call batching                             │
│  └── Viewport render skipping when minimized          │
└──────────────────────────────────────────────────────┘
```

## Architecture: First-Run Tutorial (Gap)

```
┌──────────────────────────────────────────────────────┐
│  First-Run Tutorial                                   │
│                                                       │
│  Onboarding flow:                                     │
│  ┌──────────────────────────────────────────────┐     │
│  │ Step 1: Welcome                              │     │
│  │   "Welcome to FlowForge Editor!               │     │
│  │    This quick tutorial will show you the      │     │
│  │    basics. [Start] [Skip]"                   │     │
│  │                                               │     │
│  │ Step 2: Panel Layout                         │     │
│  │   Highlight: Hierarchy panel                  │     │
│  │   Text: "Entities are organized in a          │     │
│  │          hierarchy. Click to select."        │     │
│  │                                               │     │
│  │ Step 3: Inspector                            │     │
│  │   Highlight: Inspector panel                  │     │
│  │   Text: "Edit component properties here."    │     │
│  │                                               │     │
│  │ Step 4: Viewport                             │     │
│  │   Highlight: 3D viewport                     │     │
│  │   Text: "Orbit: click+drag. Zoom: scroll."  │     │
│  │                                               │     │
│  │ Step 5: FlowForge                            │     │
│  │   Highlight: FlowForge tab                   │     │
│  │   Text: "Create visual scripts here."        │     │
│  │                                               │     │
│  │ Step 6: Done                                 │     │
│  │   "You're ready! Open Help for more."        │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Implementation:                                      │
│  ├── Show once (flag in session settings)             │
│  ├── Overlay arrows + highlight boxes on existing UI  │
│  ├── Non-modal (can skip at any step)                 │
│  └── Reset via Help menu ("Show tutorial again")      │
└──────────────────────────────────────────────────────┘
```

## Dependency Chain

```
All Prior Phases
  │
  ├──► T-TL-9.1 Error Boundaries ──→ wraps all panels
  ├──► T-TL-9.2 Layout Persistence ──► Phase 7 (Serializer)
  ├──► T-TL-9.3 Theme System ──► Phase 1 (egui styling)
  ├──► T-TL-9.4 Keyboard Shortcuts ──→ all panels
  ├──► T-TL-9.5 Editor Profiling ──→ panel optimization
  └──► T-TL-9.6 First-Run Tutorial ──→ overlay on all panels
```

## Implementation Order

1. T-TL-9.4: Keyboard shortcut system (basic global shortcuts first)
2. T-TL-9.3: Theme system (dark mode default, light mode toggle)
3. T-TL-9.1: Error boundaries (wrap each panel in catch_unwind)
4. T-TL-9.2: Panel layout persistence (save/restore dock positions)
5. T-TL-9.5: Editor performance profiling (per-panel timing, optimization)
6. T-TL-9.6: First-run tutorial (onboarding overlay sequence)

## Success Criteria

- One panel crash does not crash the entire editor
- Panel layout persists across sessions
- Light/dark mode toggle updates all panels consistently
- All keyboard shortcuts are configurable with conflict detection
- Editor frame time is tracked per-panel with slow panel warnings
- First-run tutorial guides new users through core workflows
- Cheat sheet overlay (Ctrl+Shift+/) shows all available shortcuts
