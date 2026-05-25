# PHASE_8_ARCH.md — FlowForge Visual Scripting Architecture

## Overview

Phase 8 implements the FlowForge visual scripting system: node graph editor, categorized node types (40+), Python bytecode compilation, sub-graph macros, and ECS entity integration. Python visual scripting tools exist but need Rust graph widget and ECS integration.

## Current State

| Task | Status | What Exists | What's Missing |
|------|--------|-------------|----------------|
| T-TL-8.1 | [~] PARTIAL | Python graph_editor.py (pan/zoom, minimap, selection) | Rust/egui graph widget |
| T-TL-8.2 | [~] PARTIAL | Python node_types.py, node_library.py; Tauri nodes.rs | Rust/egui node implementations |
| T-TL-8.3 | [~] PARTIAL | Python blueprint_compiler.py | Python bytecode compilation |
| T-TL-8.4 | [~] PARTIAL | Python blueprint_serializer.py | Sub-graph macros |
| T-TL-8.5 | [-] NOT STARTED | Python blueprint_runtime.py exists | ECS entity integration |

## Architecture: Node Graph Editor (Gap)

```
┌──────────────────────────────────────────────────────┐
│  FlowForge Node Graph Editor (egui custom widget)     │
│                                                       │
│  ┌──────────────────────────────────────────────┐     │
│  │  Node Palette    │  Graph Canvas              │     │
│  │  ┌──────────────┐│  ┌──────────────────────┐ │     │
│  │  │ ▶ Events     ││  │                      │ │     │
│  │  │   On Begin   ││  │  [On Begin]          │ │     │
│  │  │   On Update  ││  │  ┌──────────────┐    │ │     │
│  │  │   On Collide ││  │  │ Output: Tick │    │ │     │
│  │  │ ▶ Actions    ││  │  └──────┬───────┘    │ │     │
│  │  │   Move To    ││  │         │            │ │     │
│  │  │   Rotate     ││  │         ▼            │ │     │
│  │  │   Spawn      ││  │  [Move To]           │ │     │
│  │  │ ▶ Conditions ││  │  ┌──────────────┐    │ │     │
│  │  │   If>        ││  │  │ Target: [   ]│    │ │     │
│  │  │   Compare    ││  │  │ Speed:  5.0  │    │ │     │
│  │  │ ▶ Math       ││  │  └──────────────┘    │ │     │
│  │  │   Add        ││  │                      │ │     │
│  │  │   Multiply   ││  │  [Add]→[Log]         │ │     │
│  │  └──────────────┘│  └──────────────────────┘ │     │
│  │                  │  Zoom: 100%  [Fit] [Mini] │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Graph widget features:                               │
│  ├── Pan/zoom canvas                                   │
│  ├── Drag-drop nodes from palette                     │
│  ├── Click-drag connections between pins              │
│  ├── Node selection (single, box select)              │
│  ├── Property panel for selected node                 │
│  ├── Minimap in corner                                │
│  └── Grid background with snap                        │
└──────────────────────────────────────────────────────┘
```

**egui custom widget approach:**

```rust
// The graph editor is implemented as a custom egui widget
// using egui's painter for rendering and sense for interaction.

pub struct NodeGraphEditor {
    nodes: Vec<Node>,
    connections: Vec<Connection>,
    canvas_offset: Vec2,
    zoom: f32,
    selected_nodes: Vec<NodeId>,
    drag_state: DragState,
    palette_filter: String,
}

impl NodeGraphEditor {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // 1. Render grid background
        // 2. Render connections (Bezier curves between pins)
        // 3. Render nodes (rounded rects with title, pins, fields)
        // 4. Handle input (click, drag, scroll)
        // 5. Render minimap in corner
        // 6. Render connection drag preview
    }

    fn render_node(&self, painter: &egui::Painter, node: &Node, rect: Rect) {
        // Title bar (colored by category)
        // Input pins on left edge
        // Output pins on right edge
        // Property fields inside body
        // Selection highlight
    }

    fn render_connection(&self, painter: &egui::Painter, conn: &Connection) {
        // Bezier curve from output pin to input pin
        // Color by data type
        // Highlight on hover
    }
}
```

## Architecture: Node Type System (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Node Type Hierarchy                                  │
│                                                       │
│  Categories (implemented in Python node_types.py):    │
│  ┌──────────────────────────────────────────────┐     │
│  │ Events (6):                                  │     │
│  │   OnBegin, OnUpdate, OnEnd, OnCollision,     │     │
│  │   OnInput, OnTimer                           │     │
│  │                                               │     │
│  │ Actions (10):                                │     │
│  │   MoveTo, Rotate, Scale, Spawn, Destroy,     │     │
│  │   SetVariable, PlayAnimation, PlaySound,     │     │
│  │   SetMaterial, Teleport                      │     │
│  │                                               │     │
│  │ Conditions (6):                              │     │
│  │   If, Compare, InRange, HasTag,              │     │
│  │   IsVisible, IsPlaying                       │     │
│  │                                               │     │
│  │ Math (8):                                    │     │
│  │   Add, Subtract, Multiply, Divide,           │     │
│  │   Lerp, Clamp, Random, VectorOp              │     │
│  │                                               │     │
│  │ Flow (5):                                    │     │
│  │   Sequence, Branch, Loop, Wait,              │     │
│  │   Delay                                      │     │
│  │                                               │     │
│  │ ECS (5):                                     │     │
│  │   GetComponent, SetComponent,                │     │
│  │   AddComponent, RemoveComponent,             │     │
│  │   Query                                      │     │
│  │                                               │     │
│  │ Debug (3):                                   │     │
│  │   Print, Log, Breakpoint                     │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Each node type defines:                              │
│  ├── Input pins (typed)                               │
│  ├── Output pins (typed)                              │
│  ├── Properties (default values, constraints)         │
│  ├── Validation rules                                 │
│  └── Code generation template                         │
└──────────────────────────────────────────────────────┘
```

## Architecture: Python Bytecode Compilation (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Blueprint Compiler Pipeline                          │
│                                                       │
│  FlowForge Graph → Executable Code:                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │ Graph    │→ │ IR       │→ │ Python   │→ │ Exec │ │
│  │ Editor   │  │ Builder  │  │ Codegen  │  │      │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────┘ │
│                                                       │
│  Graph IR (intermediate representation):              │
│  ┌──────────────────────────────────────────────┐     │
│  │ struct GraphIR {                            │     │
│  │     nodes: Vec<NodeIR>,                      │     │
│  │     connections: Vec<ConnectionIR>,          │     │
│  │     entry_points: Vec<EntryPoint>,           │     │
│  │     variables: Vec<Variable>,                 │     │
│  │ }                                            │     │
│  │                                               │     │
│  │ struct NodeIR {                              │     │
│  │     id: NodeId,                              │     │
│  │     node_type: String,                       │     │
│  │     properties: HashMap<String, Value>,      │     │
│  │     input_pins: Vec<PinIR>,                   │     │
│  │     output_pins: Vec<PinIR>,                  │     │
│  │ }                                            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Code generation:                                     │
│  ├── Topological sort of graph nodes                  │
│  ├── Each node → Python function call                 │
│  ├── Connections → variable assignments               │
│  ├── Entry points → event handlers                    │
│  └── Output: valid Python AST → bytecode via compile()│
└──────────────────────────────────────────────────────┘
```

## Architecture: Sub-Graph Macros (Gap)

```
┌──────────────────────────────────────────────────────┐
│  Sub-Graph Macros                                     │
│                                                       │
│  Macro definition:                                    │
│  ┌──────────────────────────────────────────────┐     │
│  │ Node palette → right-click "Create Macro"    │     │
│  │ Selected nodes collapsed to single node      │     │
│  │                                              │     │
│  │ Macro node:                                  │     │
│  │  ┌──────────────────────────────────┐        │     │
│  │  │  [MyCustomLogic]                 │        │     │
│  │  │  ┌──────────┐                    │        │     │
│  │  │  │ Input: x │──►[internal]──►   │        │     │
│  │  │  │          │                    │        │     │
│  │  │  │          │──►[internal]──►   │        │     │
│  │  │  │ Output: y│                    │        │     │
│  │  │  └──────────┘                    │        │     │
│  │  └──────────────────────────────────┘        │     │
│  │                                              │     │
│  │  Macro expansion:                            │     │
│  │  ├── Inline: expand macro in place           │     │
│  │  ├── Call: keep as call node                 │     │
│  │  └── Nested: macros within macros            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Macro library:                                       │
│  ├── Saved in asset library                           │
│  ├── Categorized in node palette                      │
│  ├── Versioned (compatibility check)                  │
│  └── Shared between projects                          │
└──────────────────────────────────────────────────────┘
```

## Architecture: FlowForge Entity Integration (Gap — Blocked on S15)

```
┌──────────────────────────────────────────────────────┐
│  FlowForge ECS Integration                            │
│                                                       │
│  Attach FlowForge script to entity:                   │
│  ┌──────────────────────────────────────────────┐     │
│  │ struct FlowForgeScript {                     │     │
│  │     blueprint_id: AssetId,                    │     │
│  │     compiled: Option<CompiledBlueprint>,       │     │
│  │     variables: HashMap<String, Value>,         │     │
│  │     state: ScriptState,                       │     │
│  │ }                                             │     │
│  │                                               │     │
│  │ enum ScriptState { Running, Paused, Stopped } │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  Event triggers:                                      │
│  ┌──────────────────────────────────────────────┐     │
│  │ OnBegin:   Called when entity is spawned     │     │
│  │ OnUpdate:  Called every frame                │     │
│  │ OnEnd:     Called when entity is despawned   │     │
│  │ OnCollision: Called on collision events      │     │
│  │ OnInput:   Called on input events            │     │
│  └──────────────────────────────────────────────┘     │
│                                                       │
│  System integration:                                  │
│  ├── FlowForgeSystem runs in ECS schedule             │
│  ├── Iterates entities with FlowForgeScript           │
│  ├── Executes compiled blueprints per event           │
│  └── Communicates via ECS component access            │
│                                                       │
│  Blueprint runtime:                                   │
│  ├── Compiled to Python (interpreted)                 │
│  ├── Or compiled to native (future)                   │
│  └── Variable scope tied to entity                    │
└──────────────────────────────────────────────────────┘
```

## Dependency Chain

```
Phase 1 (EguiUIContext)
  │
  └──► T-TL-8.1 Node Graph Editor
         │
         ├──► T-TL-8.2 Node Types (40+)
         │     │
         │     ├──► T-TL-8.3 Python Bytecode Compiler
         │     │                              │
         │     │                              └──► T-TL-8.4 Sub-Graph Macros
         │     │
         │     └──► T-TL-8.5 Entity Integration ──► S15 Core ECS
         │
         └──► Tauri nodes.rs commands (already exist)
```

## Implementation Order

1. T-TL-8.2: Node type definitions (port Python node_types.py to Rust)
2. T-TL-8.1: Node graph editor egui widget (canvas, nodes, connections)
3. T-TL-8.3: Blueprint compiler (graph → IR → Python code)
4. T-TL-8.4: Sub-graph macros (selection → macro node)
5. T-TL-8.5: ECS entity integration (FlowForgeScript component, FlowForgeSystem) — requires S15

## Success Criteria

- Node graph canvas renders with pan/zoom and grid background
- Nodes can be dragged from palette and connected with Bezier curves
- Graph IR correctly represents all node types and connections
- Compiled Python code executes and produces expected results
- Sub-graph macros collapse/expand correctly
- FlowForgeScript on ECS entity triggers OnBegin/OnUpdate events
- 40+ node types across Events, Actions, Conditions, Math, Flow, ECS, Debug categories
