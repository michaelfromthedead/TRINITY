# PHASE_7_ARCH.md — Pharo Environment Architecture

## Overview

Phase 7 implements the Pharo-inspired live programming environment features: Foundation Serializer (session persistence to SQLite), Mirror (unified reflection API over Python, Rust, and ECS objects), Moldable Inspectors (data-driven visualization templates), Spotter (universal search), Change Management (undo/redo), Finder (query-by-example), and Pharo Shell (enhanced REPL). This phase is predominantly new implementation with minimal existing Rust code.

## Current State

| Task | Status | What Exists | What's Missing |
|------|--------|-------------|----------------|
| T-TL-7.1 | [-] NOT STARTED | Nothing | Foundation Serializer |
| T-TL-7.2 | [-] NOT STARTED | Trinity inspection commands (partial) | Unified Mirror API |
| T-TL-7.3 | [-] NOT STARTED | Nothing | Moldable inspectors |
| T-TL-7.4 | [-] NOT STARTED | Nothing | Spotter universal search |
| T-TL-7.5 | [~] PARTIAL | Python undo_system.py, command_pattern.py | Rust undo system |
| T-TL-7.6 | [-] NOT STARTED | Nothing | Finder query-by-example |
| T-TL-7.7 | [-] NOT STARTED | Nothing | Pharo Shell |

## Architecture: Foundation Serializer (Gap)

```
┌──────────────────────────────────────────────────────┐
│  FoundationSerializer                                 │
│                                                       │
│  SQLite-based session persistence:                    │
│  ┌──────────────────────────────────────────────┐     │
│  │ -- Schema                                    │     │
│  │ CREATE TABLE sessions (                      │     │
│  │     id TEXT PRIMARY KEY,                     │     │
│  │     name TEXT,                               │     │
│  │     created_at TEXT,                         │     │
│  │     updated_at TEXT                          │     │
│  │ );                                           │     │
│  │ CREATE TABLE panel_layouts (                 │     │
│  │     session_id TEXT REFERENCES sessions(id), │     │
│  │     panel_id TEXT,                           │     │
│  │     position BLOB,  // JSON {x,y,w,h}       │     │
│  │     visible BOOLEAN,                         │     │
│  │     config BLOB     // JSON panel state     │     │
│  │ );                                           │     │
│  │ CREATE TABLE repl_history (                  │     │
│  │     session_id TEXT REFERENCES sessions(id), │     │
│  │     command TEXT,                            │     │
│  │     timestamp TEXT                           │     │
│  │ );                                           │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Serialization:                                       │
│  ├── EditorState → SQLite row                          │
│  ├── Panel layout/config → JSON blob                  │
│  ├── REPL history → command rows                      │
│  └── Camera poses → JSON blob                         │
│                                                       │
│  Save/restore:                                        │
│  ├── Auto-save on panel layout change                 │
│  ├── Auto-save on editor state change                 │
│  ├── Manual save via Ctrl+S                           │
│  ├── Restore on startup (last session or picker)      │
│  └── Crash recovery (auto-save on interval)           │
└──────────────────────────────────────────────────────┘
```

## Architecture: Mirror (Unified Reflection API) (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Mirror Reflection API                                │
│                                                       │
│  Unified interface over three object kinds:           │
│  ┌──────────────────────────────────────────────┐     │
│  │ trait MirrorObject {                        │     │
│  │     fn kind(&self) -> MirrorKind;            │     │
│  │     fn name(&self) -> &str;                  │     │
│  │     fn properties(&self) -> Vec<Property>;   │     │
│  │     fn methods(&self) -> Vec<MethodDesc>;    │     │
│  │     fn invoke(&self, method: &str,           │     │
│  │                args: &[Value]) -> Result;    │     │
│  │ }                                            │     │
│  │                                               │     │
│  │ enum MirrorKind {                            │     │
│  │     PythonObject,   // via PyO3               │     │
│  │     RustObject,     // via introspection      │     │
│  │     EcsComponent,   // via ComponentRegistry  │     │
│  │     EcsSystem,      // via SystemRegistry     │     │
│  │ }                                            │     │
│  │                                               │     │
│  │ struct Property {                            │     │
│  │     name: String,                            │     │
│  │     type_code: String,                       │     │
│  │     value: Value,  // JSON value             │     │
│  │     read_only: bool,                         │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Bridge integration:                                  │
│  ├── Python objects: PyO3 reflection                  │
│  │   (type(), dir(), getattr())                       │
│  ├── Rust objects: manual Mirror impl + derive macro  │
│  ├── ECS components: via ComponentRegistry            │
│  └── ECS systems: via SystemRegistry                  │
│                                                       │
│  Query interface:                                     │
│  ├── find_objects(predicate: fn(&MirrorObject) -> bool│
│  ├── query_objects("name contains 'Player'")          │
│  └── browse_object_tree(root: &dyn MirrorObject)      │
└──────────────────────────────────────────────────────┘
```

## Architecture: Moldable Inspectors (Gap)

```
┌──────────────────────────────────────────────────────┐
│  MoldableInspector                                    │
│                                                       │
│  Data-driven visualization templates:                 │
│  ┌──────────────────────────────────────────────┐     │
│  │ struct VisualizationTemplate {               │     │
│  │     name: String,                            │     │
│  │     applies_to: fn(&MirrorObject) -> bool,   │     │
│  │     render: fn(&mut egui::Ui, &MirrorObject), │     │
│  │ }                                            │     │
│  │                                               │     │
│  │ // Built-in templates:                       │     │
│  │ Template::List     // show as list            │     │
│  │ Template::Graph    // show as graph/edges     │     │
│  │ Template::Image    // show as image           │     │
│  │ Template::Code     // show source code        │     │
│  │ Template::Timeline // show timeline           │     │
│  │ Template::Custom   // user-defined egui       │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Inspector uses template matching:                    │
│  ┌──────────────────────────────────────────────┐     │
│  │ If inspecting a Texture:                     │     │
│  │ → matches Image template                     │     │
│  │ → renders as thumbnail with metadata         │     │
│  │                                               │     │
│  │ If inspecting a graph:                       │     │
│  │ → matches Graph template                     │     │
│  │ → renders as node-edge visualization         │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Custom template registration:                        │
│  ├── Editor plugins can register templates            │
│  ├── Templates match on object type via predicate     │
│  └── Priority ordering (most specific wins)           │
└──────────────────────────────────────────────────────┘
```

## Architecture: Spotter (Universal Search) (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Spotter (Ctrl+P Universal Search)                    │
│                                                       │
│  ┌──────────────────────────────────────────────┐     │
│  │  Spotter                                     │     │
│  │  ┌────────────────────────────────────────┐  │     │
│  │  │ 🔍 [camera position               ]   │  │     │
│  │  ├────────────────────────────────────────┤  │     │
│  │  │ Entities (3)                          │  │     │
│  │  │   📷 Camera                           │  │     │
│  │  │   📦 CameraController                 │  │     │
│  │  ├────────────────────────────────────────┤  │     │
│  │  │ Components (2)                        │  │     │
│  │  │   ▤ CameraComponent                   │  │     │
│  │  │   ▤ CameraShakeComponent              │  │     │
│  │  ├────────────────────────────────────────┤  │     │
│  │  │ Assets (5)                            │  │     │
│  │  │   🖼 camera_icon.png                  │  │     │
│  │  │   📄 camera_rig.fbx                   │  │     │
│  │  └────────────────────────────────────────┘  │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Search domains:                                      │
│  ├── Entities (by name or component)                  │
│  ├── Components (by name or field)                    │
│  ├── Assets (by filename or metadata)                  │
│  ├── Settings (by key or description)                 │
│  ├── Commands (by name or shortcut)                   │
│  └── Help (by keyword)                                │
│                                                       │
│  Features:                                            │
│  ├── Fuzzy text matching                               │
│  ├── Regex mode (toggle)                              │
│  ├── Category groupings                               │
│  ├── Action on select (navigate, open, execute)       │
│  └── Recent searches                                  │
└──────────────────────────────────────────────────────┘
```

## Architecture: Change Management (Rust undo/redo) (Gap)

```
┌──────────────────────────────────────────────────────┐
│  ChangeManager (Undo/Redo)                            │
│                                                       │
│  Command pattern (mirrors Python undo_system.py):    │
│  ┌──────────────────────────────────────────────┐     │
│  │ trait UndoCommand {                         │     │
│  │     fn execute(&mut self);                   │     │
│  │     fn undo(&mut self);                      │     │
│  │     fn redo(&mut self) { self.execute() }    │     │
│  │     fn name(&self) -> &str;                  │     │
│  │     fn merge(&mut self, other: &Self) -> bool│     │
│  │ }                                            │     │
│  │                                               │     │
│  │ struct ChangeManager {                       │     │
│  │     undo_stack: Vec<Box<dyn UndoCommand>>,   │     │
│  │     redo_stack: Vec<Box<dyn UndoCommand>>,   │     │
│  │     max_entries: usize,  // 100 default      │     │
│  │     transaction_depth: usize,                 │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Editor commands (undoable):                          │
│  ├── EntityCreate(components)                         │
│  ├── EntityDelete(entity_id, snapshot)                │
│  ├── ComponentEdit(entity_id, component_id, old, new) │
│  ├── SceneReparent(entity_id, old_parent, new_parent) │
│  └── BatchCommand(commands)  // for multi-select      │
│                                                       │
│  Transaction support:                                 │
│  ├── begin_transaction(label)                         │
│  ├── commit_transaction() → single undo step          │
│  └── rollback_transaction() → undo all                │
└──────────────────────────────────────────────────────┘
```

## Architecture: Finder (Query-by-Example) (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Finder (Query-by-Example)                            │
│                                                       │
│  Find entities by component pattern:                  │
│  ┌──────────────────────────────────────────────┐     │
│  │ Example: find all entities with              │     │
│  │   Position.y > 0 AND Health.current < 50    │     │
│  │                                               │     │
│  │ Query format (JSON):                          │     │
│  │ {                                             │     │
│  │   "must_have": [1, 3],  // component IDs     │     │
│  │   "conditions": [                            │     │
│  │     {"comp": 1, "field": "y", "op": "gt",    │     │
│  │      "value": 0.0},                          │     │
│  │     {"comp": 3, "field": "current",          │     │
│  │      "op": "lt", "value": 50.0}             │     │
│  │   ]                                          │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Features:                                            │
│  ├── Save searches as smart folders                   │
│  ├── Full-text search across entity names             │
│  ├── Regex mode for text fields                       │
│  └── Query result streaming (paginated)               │
└──────────────────────────────────────────────────────┘
```

## Architecture: Pharo Shell (Enhanced REPL) (Gap)

```
┌──────────────────────────────────────────────┐
│  Pharo Shell (Enhanced REPL)                  │
│                                               │
│  ┌──────────────────────────────────────┐     │
│  │ Workspace (multi-file editor)         │     │
│  │ ┌──────────────────────────────────┐ │     │
│  │ │ # Explore the world              │ │     │
│  │ │ registry_list()                  │ │     │
│  │ │ entity_count = query_all().len() │ │     │
│  │ │ print(f"{entity_count} entities")│ │     │
│  │ └──────────────────────────────────┘ │     │
│  │ [Run All] [Run Selection] [Save]     │     │
│  ├──────────────────────────────────────┤     │
│  │ Output                              │     │
│  │ [{"componentId":1,"name":"Pos"},...]│     │
│  │ 42 entities                          │     │
│  ├──────────────────────────────────────┤     │
│  │ Interactive REPL                     │     │
│  │ >>> entity_count                     │     │
│  │ 42                                   │     │
│  │ >>> _                                │     │
│  └──────────────────────────────────────┘     │
│                                               │
│  Features:                                    │
│  ├── Workspace with multi-line editing         │
│  ├── File: save/load scripts (Python)         │
│  ├── Interactive object display (Mirror)      │
│  ├── History persistence                      │
│  └── Syntax highlighting                      │
└──────────────────────────────────────────────┘
```

## Dependency Chain

```
Phase 1 (EguiUIContext)
  │
  ├──► T-TL-7.1 Foundation Serializer ──► Phase 9 (layout persistence)
  │
  ├──► T-TL-7.2 Mirror ──► S15 (ECS registry)
  │     │
  │     ├──► T-TL-7.3 Moldable Inspectors ──► Phase 2 (Inspector panel)
  │     ├──► T-TL-7.4 Spotter
  │     ├──► T-TL-7.6 Finder
  │     └──► T-TL-7.7 Pharo Shell ──► Phase 2 (REPL)
  │
  └──► T-TL-7.5 Change Management ──► Phase 2 (panels)
```

## Implementation Order

1. T-TL-7.5: Change management (undo/redo command pattern)
2. T-TL-7.1: Foundation Serializer (SQLite session persistence)
3. T-TL-7.2: Mirror (unified reflection API over Python/Rust/ECS)
4. T-TL-7.7: Pharo Shell (enhanced REPL with workspace)
5. T-TL-7.4: Spotter (universal search with fuzzy matching)
6. T-TL-7.3: Moldable inspectors (data-driven templates)
7. T-TL-7.6: Finder (query-by-example with smart folders)

## Success Criteria

- Editor session persists to SQLite and restores on restart
- Mirror API can introspect Python objects, Rust structs, and ECS components
- Moldable inspectors render different templates for different object types
- Spotter searches across entities, components, assets, settings, and commands
- Undo/redo works for entity create, delete, and component edit operations
- Finder returns entities matching component field conditions
- Pharo Shell executes Python code with workspace support and object display
