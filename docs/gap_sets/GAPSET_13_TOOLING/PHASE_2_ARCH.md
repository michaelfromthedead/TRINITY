# PHASE_2_ARCH.md — Core Editor Panels Architecture

## Overview

Phase 2 builds the core editor panels: EditorCamera (orbit/pan/zoom), Inspector (component inspection/editing), Hierarchy (entity tree), REPL (console), and shared SelectionState. All panels are implemented as egui widgets that consume data from the existing Rust EditorState and bridge protocol.

## Current State

| Task | Status | What Exists | What's Missing |
|------|--------|-------------|----------------|
| T-TL-2.1 | [~] PARTIAL | Python debug_camera.py; Rust Editor struct | Rust EditorCamera with glam math |
| T-TL-2.2 | [~] PARTIAL | Python selection.py; Rust Editor::selected_components() | egui Inspector panel |
| T-TL-2.3 | [~] PARTIAL | Python hierarchy.py (full tree ops) | egui Hierarchy panel |
| T-TL-2.4 | [-] NOT STARTED | Nothing | Moldable views |
| T-TL-2.5 | [-] NOT STARTED | Nothing | Multi-select diff |
| T-TL-2.6 | [~] PARTIAL | Python console_ui.py, command_history.py | Rust/egui REPL |
| T-TL-2.7 | [~] PARTIAL | Rust EditorState (basic); Python SelectionState (rich) | Unified SelectionState |

## Architecture: EditorCamera (Gap)

```
┌──────────────────────────────────────┐
│  EditorCamera (Rust — NEW)            │
│                                      │
│  Camera state:                       │
│  ├── position: Vec3                  │
│  ├── target: Vec3                    │
│  ├── up: Vec3                        │
│  ├── fov_y: f32                      │
│  ├── orbit_angles: (f32, f32)        │
│  ├── orbit_distance: f32             │
│  └── projection: Perspective or Ortho│
│                                      │
│  Controls:                           │
│  ├── orbit(delta_pitch, delta_yaw)   │
│  ├── pan(delta_x, delta_y)           │
│  ├── zoom(delta)                     │
│  ├── focus_on(entity_pos)            │
│  ├── wasd_movement(delta_time)       │
│  └── save/restore pose               │
│                                      │
│  Matrix computation (glam):          │
│  ├── view_matrix() -> Mat4           │
│  └── projection_matrix() -> Mat4     │
│                                      │
│  Serialization:                      │
│  └── serde Serialize/Deserialize     │
└──────────────────────────────────────┘
```

**Implementation notes:**
- Use `glam::Vec3` and `glam::Mat4` for math (already in renderer-backend deps).
- Camera state is independent of wgpu — can be unit tested.
- `save/restore` pose enables bookmarking camera positions.
- Reuse Python `debug_camera.py` logic but implement in Rust.

## Architecture: Inspector Panel (Gap)

```
┌──────────────────────────────────────────────┐
│  Inspector Panel (egui — NEW)                 │
│                                               │
│  Data source: Editor::selected_components()   │
│  Returns: Vec<(component_id, name, bytes)>    │
│                                               │
│  Layout:                                      │
│  ┌────────────────────────────────────┐       │
│  │ ▸ Transform (collapsible)          │       │
│  │   Position: [1.0] [2.0] [3.0]     │       │
│  │   Rotation: [0.0] [0.0] [0.0]     │       │
│  │   Scale:    [1.0] [1.0] [1.0]     │       │
│  ├────────────────────────────────────┤       │
│  │ ▸ MeshRenderer                    │       │
│  │   Mesh: [character_model ▼]       │       │
│  │   Material: [character_mat ▼]     │       │
│  └────────────────────────────────────┘       │
│                                               │
│  Features:                                    │
│  ├── Collapsible component sections            │
│  ├── Type-aware field rendering (f32, i32)    │
│  ├── Inline editing of primitive fields        │
│  ├── Add Component button                     │
│  └── Null/empty state when nothing selected    │
└──────────────────────────────────────────────┘
```

**Panels as egui widgets pattern:**

```rust
// All panels follow this pattern:
pub trait EditorPanel {
    /// Unique panel identifier (for layout persistence).
    fn id(&self) -> &'static str;

    /// Panel title displayed in the tab/title bar.
    fn title(&self) -> &str;

    /// Render the panel contents into the given egui Ui.
    fn ui(&mut self, ui: &mut egui::Ui, editor: &Editor);

    /// Called once per frame regardless of visibility (for background updates).
    fn update(&mut self, _ctx: &egui::Context, _editor: &Editor) {}
}
```

## Architecture: Hierarchy Panel (Gap)

```
┌──────────────────────────────────────────────┐
│  Hierarchy Panel (egui — NEW)                 │
│                                               │
│  Data source: Editor::entity_ids()            │
│  Plus parent_id from ComponentStore           │
│                                               │
│  Layout:                                      │
│  ┌────────────────────────────────────┐       │
│  │ 🔍 [Search entities...       ]     │       │
│  ├────────────────────────────────────┤       │
│  │ ▼ Scene Root                      │       │
│  │   ├ ▼ Character                   │       │
│  │   │  ├ Mesh (selected)            │       │
│  │   │  └ Camera                    │       │
│  │   └ ▼ Lights                     │       │
│  │      ├ DirectionalLight           │       │
│  │      └ PointLight                 │       │
│  └────────────────────────────────────┘       │
│                                               │
│  Features:                                    │
│  ├── Tree view using egui::collapsible         │
│  ├── Single-click selection                    │
│  ├── Right-click context menu (rename, delete) │
│  ├── Search/filter with text matching          │
│  └── Empty state when no entities              │
└──────────────────────────────────────────────┘
```

## Architecture: SelectionState (Extension)

```rust
// Extend EditorState with gizmo and hover support:
pub struct EditorState {
    pub selected_entity: Option<u64>,
    pub hovered_entity: Option<u64>,
    pub show_hierarchy: bool,
    pub show_inspector: bool,
    pub show_viewport: bool,
    pub gizmo_mode: GizmoMode,      // Translate | Rotate | Scale
    pub gizmo_space: GizmoSpace,    // Local | World
}

pub enum GizmoMode { Translate, Rotate, Scale }
pub enum GizmoSpace { Local, World }
```

## Architecture: REPL Panel (Gap)

```
┌──────────────────────────────────────────────┐
│  REPL Panel (egui — NEW)                      │
│                                               │
│  Uses bridge protocol for Trinity commands    │
│                                               │
│  Layout:                                      │
│  ┌────────────────────────────────────┐       │
│  │ >>> registry_list                  │       │
│  │ [{"componentId":1,"name":"Pos"}]  │       │
│  │ >>> _                             │       │
│  ├────────────────────────────────────┤       │
│  │ [entry line] ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  │       │
│  └────────────────────────────────────┘       │
│                                               │
│  Features:                                    │
│  ├── Command history (up/down arrows)          │
│  ├── Output formatting (JSON pretty-print)    │
│  ├── Error highlighting                       │
│  └── History persistence to SQLite             │
└──────────────────────────────────────────────┘
```

## Data Flow

```
Python Tools (existing)
  │ JSON-RPC over sidecar
  ▼
Rust Bridge Protocol
  │ deserialize
  ▼
Editor (crates/renderer-backend/src/editor.rs)
  │ shared Arc<RwLock<ComponentStore>>
  ▼
EditorState (selection, visibility flags)
  │ read by all panels
  ▼
┌────────────┬─────────────┬──────────┬──────────┐
│ Inspector  │ Hierarchy   │ Camera   │ REPL     │
│ Panel      │ Panel       │          │ Panel    │
└────────────┴─────────────┴──────────┴──────────┘
  │ egui::Ui
  ▼
egui Frame → wgpu Surface
```

## Dependencies

- **Blocked on Phase 1** (T-TL-1.3 EguiUIContext) for egui rendering infrastructure.
- T-TL-2.1 (EditorCamera) can proceed independently — pure math, no egui dependency.
- T-TL-2.7 (SelectionState) should be implemented first as it is consumed by all panels.

## Implementation Order

1. T-TL-2.7: Extend EditorState with gizmo_mode, gizmo_space, hovered_entity
2. T-TL-2.1: Implement EditorCamera with glam math, orbit/pan/zoom/WASD controls
3. T-TL-2.3: Build Hierarchy panel (tree view from entity hierarchy)
4. T-TL-2.2: Build Inspector panel (component data display + editing)
5. T-TL-2.6: Build REPL panel (command input + output display)
6. T-TL-2.4: Add moldable views (type-specific component rendering)
7. T-TL-2.5: Add multi-select and comparison (two-pane diff)

## Success Criteria

- EditorCamera orbits around entities with mouse drag
- Hierarchy tree shows all entities with expand/collapse
- Inspector displays component fields with type-aware formatting
- Inline editing changes propagate to ComponentStore via bridge protocol
- REPL accepts Python commands and displays formatted results
- SelectionState synchronizes across all panels
