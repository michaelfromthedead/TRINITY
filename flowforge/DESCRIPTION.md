# FlowForge

**Visual Programming Interface for Trinity Python Metaprogramming**

*ComfyUI's Canvas Engine + Tauri + Python/Trinity*

---

## Executive Summary

FlowForge is a visual programming environment that lets you **see and edit** Trinity Python code as a node graph. It takes ComfyUI's battle-tested LiteGraph canvas and connects it to the Trinity metaprogramming system, enabling visual exploration and modification of decorators, metaclasses, and descriptors.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  TRINITY PYTHON CODE                    FLOWFORGE VISUALIZATION             │
│  ═══════════════════                    ═══════════════════════             │
│                                                                             │
│  @component                             ┌─────────────────────┐             │
│  class Player:                          │   Player            │             │
│      health: int = 100        ───►      │   ═══════           │             │
│      position: Vec3                     │   health: int       │             │
│      velocity: Vec3                     │   position: Vec3    │             │
│                                         │   velocity: Vec3    │             │
│                                         └─────────────────────┘             │
│                                                   │                         │
│  @system                                          ▼                         │
│  class MovementSystem:                  ┌─────────────────────┐             │
│      def update(self,           ───►    │   MovementSystem    │             │
│          query: Query[Player]):         │   ═══════════════   │             │
│          ...                            │   query ──► result  │             │
│                                         └─────────────────────┘             │
│                                                                             │
│  BIDIRECTIONAL: Edit nodes ──► regenerate Python                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Four Projects

FlowForge is structured as four sequential projects:

| Project | Goal | Status |
|---------|------|--------|
| **1** | Native desktop shell (Tauri + ComfyUI frontend) | ~40% |
| **2** | Python AST parser → node graph conversion | Not started |
| **3** | View-only mode for Trinity visualization | Not started |
| **4** | Bidirectional editing with Python regeneration | Not started |

---

## Why This Architecture

### The Problem

Trinity is a powerful Python metaprogramming system with ~275 decorators, ~30 descriptor families, and 5-10 metaclasses. Understanding the relationships between components, systems, and their interactions requires reading through Python files and mentally constructing the dependency graph.

### The Solution

FlowForge visualizes Trinity code as an interactive node graph:

- **See** all components, systems, and their relationships at a glance
- **Navigate** by clicking on nodes to jump to source code
- **Edit** visually and regenerate valid Python (Project 4)
- **Understand** the Trinity architecture through spatial representation

### Why Tauri + Python (Not Bun)

| Approach | Pros | Cons |
|----------|------|------|
| **Bun/TypeScript** | Fast, modern | Can't introspect Python, would need to reimplement Trinity |
| **Python sidecar** | Direct Trinity access, AST parsing built-in | Slightly more complex IPC |

Since Trinity **is** Python, we need Python to introspect it. Tauri provides the native desktop shell, Python provides the Trinity integration.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FLOWFORGE ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      NATIVE WINDOW (Tauri WebView)                   │   │
│  │                                                                      │   │
│  │   LiteGraph Canvas ←───→ Vue 3 Components ←───→ Pinia Stores        │   │
│  │         │                                             │              │   │
│  │         └─────────────── bridge.ts ──────────────────┘              │   │
│  │                              │                                       │   │
│  │                        invoke() / listen()                           │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                           │
│                          Tauri IPC (JSON)                                   │
│                                 │                                           │
│  ┌──────────────────────────────┼──────────────────────────────────────┐   │
│  │                    TAURI CORE (Rust)                                 │   │
│  │                                                                      │   │
│  │   Window Manager  │  File Dialogs  │  Python Sidecar Manager        │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                           │
│                           stdio / JSON                                      │
│                                 │                                           │
│  ┌──────────────────────────────┼──────────────────────────────────────┐   │
│  │                    PYTHON SIDECAR                                    │   │
│  │                                                                      │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │   │ AST Parser   │  │   Trinity    │  │   Code       │             │   │
│  │   │              │  │   Adapter    │  │   Generator  │             │   │
│  │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│  │                              │                                       │   │
│  │   ┌──────────────────────────┴──────────────────────────────────┐  │   │
│  │   │                    TRINITY + FOUNDATION                      │  │   │
│  │   │                                                              │  │   │
│  │   │  Decorators  │  Metaclasses  │  Descriptors  │  Foundation  │  │   │
│  │   └──────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Frontend Layer (Forked from ComfyUI)

ComfyUI's frontend provides:
- **LiteGraph.js** - High-performance canvas with 60fps at 1000+ nodes
- **Vue 3 + Pinia** - Modern reactive UI framework
- **Node palette** - Searchable node library
- **Property panel** - Edit node properties

We fork it and:
- Remove Stable Diffusion-specific code
- Replace `fetch()` API calls with Tauri IPC
- Add Trinity-specific node types
- Add Python source navigation

### 2. Tauri Shell (Rust)

Tauri provides:
- Native window with OS integration
- File dialogs (open/save Python files)
- Python sidecar process management
- IPC routing between frontend and Python

### 3. Python Sidecar

The Python backend provides:
- **AST Parser** - Parse Python files into node graphs
- **Trinity Adapter** - Introspect Trinity decorators, metaclasses, descriptors
- **Code Generator** - Convert node graphs back to valid Python
- **Foundation Integration** - Access Registry, Mirror, Inspector, etc.

---

## Data Flow

### Viewing Python Code as Nodes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1. User: "Open trinity/components.py"                                      │
│                                    │                                        │
│  2. Tauri: Native file dialog      │                                        │
│                                    ▼                                        │
│  3. Python: ast.parse(source)  ──────►  TrinityASTVisitor                  │
│                                              │                              │
│  4. Python: Extract @component, @system      │                              │
│                                              ▼                              │
│  5. Python: Build node graph JSON   { nodes: [...], edges: [...] }         │
│                                              │                              │
│  6. Frontend: graph.configure(json)          │                              │
│                                              ▼                              │
│  7. LiteGraph: Render nodes on canvas   ┌─────────┐    ┌─────────┐        │
│                                         │ Player  │───►│Movement │        │
│                                         │Component│    │ System  │        │
│                                         └─────────┘    └─────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Editing Nodes and Regenerating Python

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1. User: Add field "mana: int = 50" to Player node                        │
│                                    │                                        │
│  2. LiteGraph: Update node.data.fields                                      │
│                                    │                                        │
│  3. Frontend: invoke('generate_python', {graph})                           │
│                                    │                                        │
│  4. Python: graph_to_ast.py        │                                        │
│                                    ▼                                        │
│  5. Python: Build AST from node graph                                       │
│                                    │                                        │
│  6. Python: ast.unparse(module)    │                                        │
│                                    ▼                                        │
│  7. Frontend: Show diff preview                                             │
│                                                                             │
│     - health: int = 100                                                     │
│     + health: int = 100                                                     │
│     + mana: int = 50                                                        │
│                                                                             │
│  8. User: Confirm ──► Python file updated                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Trinity Node Types

FlowForge represents Trinity constructs as visual nodes:

| Trinity Construct | Node Type | Visual Representation |
|-------------------|-----------|----------------------|
| `@component` class | Component Node | Box with field list |
| `@system` class | System Node | Box with query → result flow |
| `@resource` class | Resource Node | Singleton indicator |
| `@event` class | Event Node | Event payload fields |
| Metaclass inheritance | Edge | Dashed line showing hierarchy |
| Descriptor binding | Property indicator | Icon showing tracking status |
| Registry connection | Edge | Solid line to registry node |

---

## Project Structure

```
flowforge/
├── apps/
│   └── desktop/                    # Tauri desktop application
│       ├── src/                    # Frontend (forked from ComfyUI)
│       │   ├── litegraph/          # Canvas engine (COPY FROM COMFYUI)
│       │   ├── components/         # Vue components (COPY FROM COMFYUI)
│       │   ├── stores/             # Pinia stores (ADAPT)
│       │   └── bridge/             # Tauri IPC layer (NEW)
│       └── src-tauri/              # Rust backend
│           ├── src/
│           │   ├── main.rs
│           │   ├── commands/       # IPC command handlers
│           │   └── sidecar/        # Python process manager
│           └── Cargo.toml
│
├── flowforge_backend/              # Python sidecar (NEW)
│   ├── __main__.py                 # Entry point
│   ├── ipc/                        # JSON protocol handling
│   ├── ast_parser/                 # Python → node graph
│   ├── trinity_adapter/            # Trinity introspection
│   └── codegen/                    # Node graph → Python
│
└── docs/
    └── adr/                        # Architecture Decision Records
```

---

## Development Workflow

### Prerequisites

```bash
# Rust (for Tauri)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python 3.10+ (for sidecar)
# Already installed for Trinity

# Node.js / Bun (for frontend build)
curl -fsSL https://bun.sh/install | bash
```

### Setup

```bash
# Clone
git clone https://github.com/yourorg/flowforge
cd flowforge

# Install frontend dependencies
bun install

# Development mode
bun run dev

# The app will:
# 1. Start Tauri native window
# 2. Spawn Python sidecar (python -m flowforge_backend)
# 3. Connect frontend to Python via Tauri IPC
```

---

## Key Differences from Original Plan

| Aspect | Original (Bun) | New (Python) |
|--------|----------------|--------------|
| **Backend** | Bun/TypeScript sidecar | Python sidecar |
| **Node execution** | TypeScript node implementations | Trinity Python introspection |
| **Purpose** | Generic visual programming | Trinity code visualization |
| **Nodes** | Math, Logic, String, etc. | Components, Systems, Decorators |
| **Data flow** | Execute nodes as pipeline | Parse/generate Python code |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Open Python file | < 500ms to render as nodes |
| Navigate 100+ node graph | 60fps pan/zoom |
| Generate Python from graph | Syntactically valid, runnable |
| Round-trip (parse → edit → generate) | Zero data loss |

---

## Summary

FlowForge bridges visual programming and Python metaprogramming:

1. **See** Trinity code as interactive node graphs
2. **Navigate** complex relationships visually
3. **Edit** components and systems through nodes
4. **Generate** valid Python from visual changes

The architecture uses Tauri for native desktop integration and Python for direct Trinity introspection—because you can't understand Python metaprogramming from TypeScript.
