# FlowForge Architecture

**Visual Programming Interface for Trinity Python Metaprogramming**

*ComfyUI's Canvas Engine + Tauri + Python/Trinity*

---

## Table of Contents

1. [System Overview](#system-overview)
2. [The Four Projects](#the-four-projects)
3. [Layer-by-Layer Architecture](#layer-by-layer-architecture)
4. [IPC Protocol Design](#ipc-protocol-design)
5. [Trinity Integration](#trinity-integration)
6. [AST-to-Node Conversion](#ast-to-node-conversion)
7. [Data Flow Diagrams](#data-flow-diagrams)
8. [Security Model](#security-model)

---

## System Overview

### Original ComfyUI Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMFYUI ORIGINAL ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         BROWSER (Client)                             │   │
│  │                                                                      │   │
│  │   Vue 3 App ←───→ LiteGraph Canvas ←───→ Pinia Stores               │   │
│  │        │                                       │                     │   │
│  │        └──────────── api.ts ──────────────────┘                     │   │
│  │                         │                                            │   │
│  │                    fetch() / WebSocket                               │   │
│  └─────────────────────────┼───────────────────────────────────────────┘   │
│                            │                                                │
│                      HTTP / WS                                              │
│                            │                                                │
│  ┌─────────────────────────┼───────────────────────────────────────────┐   │
│  │                    PYTHON SERVER                                     │   │
│  │                                                                      │   │
│  │   aiohttp ←───→ Execution Engine ←───→ Node Registry                │   │
│  │      │                  │                    │                       │   │
│  │      │                  ▼                    ▼                       │   │
│  │      │           PyTorch / CUDA        nodes/*.py                   │   │
│  │      │                                                               │   │
│  │      └──────→ /object_info (GET)                                    │   │
│  │      └──────→ /queue (GET/POST)                                     │   │
│  │      └──────→ /prompt (POST)                                        │   │
│  │      └──────→ /ws (WebSocket)                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### FlowForge Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FLOWFORGE TARGET ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      NATIVE WINDOW (Tauri WebView)                   │   │
│  │                                                                      │   │
│  │   Vue 3 App ←───→ LiteGraph Canvas ←───→ Pinia Stores               │   │
│  │        │                                       │                     │   │
│  │        └──────────── bridge.ts ───────────────┘                     │   │
│  │                         │                                            │   │
│  │                   invoke() / listen()                                │   │
│  └─────────────────────────┼───────────────────────────────────────────┘   │
│                            │                                                │
│                     Tauri IPC (JSON)                                        │
│                            │                                                │
│  ┌─────────────────────────┼───────────────────────────────────────────┐   │
│  │                    TAURI CORE (Rust)                                 │   │
│  │                                                                      │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │   │   Window    │  │   Commands  │  │   Events    │                │   │
│  │   │   Manager   │  │   Router    │  │   Emitter   │                │   │
│  │   └─────────────┘  └──────┬──────┘  └─────────────┘                │   │
│  │                           │                                          │   │
│  │   ┌─────────────┐  ┌──────┴──────┐  ┌─────────────┐                │   │
│  │   │   Plugin    │  │   Sidecar   │  │   File      │                │   │
│  │   │   Loader    │  │   Manager   │  │   System    │                │   │
│  │   └─────────────┘  └──────┬──────┘  └─────────────┘                │   │
│  └───────────────────────────┼─────────────────────────────────────────┘   │
│                              │                                              │
│                        stdio / IPC                                          │
│                              │                                              │
│  ┌───────────────────────────┼─────────────────────────────────────────┐   │
│  │                    PYTHON SIDECAR PROCESS                            │   │
│  │                                                                      │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │   │   Graph     │  │   Trinity   │  │    AST      │                │   │
│  │   │   Executor  │  │   Adapter   │  │   Parser    │                │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘                │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                    TRINITY SYSTEMS                           │   │   │
│  │   │                                                              │   │   │
│  │   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │   │   │
│  │   │   │ Decorators  │  │ Metaclasses │  │ Descriptors │        │   │   │
│  │   │   │  (@component │  │ (Component  │  │ (Tracked    │        │   │   │
│  │   │   │   @system)   │  │   Meta)     │  │  Descriptor)│        │   │   │
│  │   │   └─────────────┘  └─────────────┘  └─────────────┘        │   │   │
│  │   │                                                              │   │   │
│  │   │   ┌─────────────────────────────────────────────────────┐  │   │   │
│  │   │   │               FOUNDATION LAYER                       │  │   │   │
│  │   │   │  Mirror │ Registry │ Tracker │ EventLog │ Inspector │  │   │   │
│  │   │   └─────────────────────────────────────────────────────┘  │   │   │
│  │   └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Four Projects

FlowForge is structured as four sequential projects:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE FOUR PROJECTS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROJECT 1: Native Desktop Shell                                            │
│  ════════════════════════════════                                           │
│  Replace ComfyUI's web-only model with Tauri native desktop                │
│  Status: ~60% complete (Tauri shell done, frontend pending)                 │
│                                                                             │
│  PROJECT 2: Python AST → Node Graph                                         │
│  ═══════════════════════════════════                                        │
│  Parse Trinity Python code and convert to visual node representation       │
│  Status: Not started                                                        │
│                                                                             │
│  PROJECT 3: View-Only Mode                                                  │
│  ═════════════════════════                                                  │
│  Display Trinity code structure as interactive node graph                   │
│  Status: Not started                                                        │
│                                                                             │
│  PROJECT 4: Bidirectional Editing                                           │
│  ═════════════════════════════════                                          │
│  Edit nodes visually → regenerate valid Python code                        │
│  Status: Not started                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Project Dependencies

```
PROJECT 1 ──────────────────────────────────────────────────────────────────►
           Native Shell (Tauri + ComfyUI Frontend + Python Sidecar)
                              │
                              ▼
PROJECT 2 ──────────────────────────────────────────────────────────────────►
           AST Parser (Python → Node Graph JSON)
                              │
                              ▼
PROJECT 3 ──────────────────────────────────────────────────────────────────►
           View Mode (Read-only visualization)
                              │
                              ▼
PROJECT 4 ──────────────────────────────────────────────────────────────────►
           Edit Mode (Bidirectional sync)
```

---

## Layer-by-Layer Architecture

### Layer 1: Frontend (Forked from ComfyUI)

The frontend is forked directly from ComfyUI's frontend repository with minimal modifications:

| Component | Source | Modifications |
|-----------|--------|---------------|
| **LiteGraph.js** | ComfyUI frontend | None - core canvas engine |
| **GraphCanvas.vue** | ComfyUI frontend | Remove SD-specific watchers |
| **Pinia Stores** | ComfyUI frontend | Replace API hydration with Tauri IPC |
| **UI Chrome** | ComfyUI frontend | Restyle, remove SD-specific panels |

**Critical Files to Fork:**
```
ComfyUI_frontend/src/
├── scripts/
│   ├── api.ts           → Replace with bridge/tauri-api.ts
│   ├── app.ts           → Modify bootstrap sequence
│   └── litegraph/       → PRESERVE ENTIRELY (the canvas!)
├── components/
│   ├── GraphCanvas.vue  → Remove server dependencies
│   ├── NodePalette.vue  → Keep for node selection
│   └── ...              → Preserve UI components
└── stores/
    ├── nodeDefStore.ts  → Load from Python via IPC
    └── workflowStore.ts → Adapt for local file operations
```

### Layer 2: Bridge Layer (New)

A thin TypeScript layer replacing fetch() calls with Tauri IPC:

```typescript
// src/bridge/tauri-api.ts

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

export class TauriAPI {
  // Node definitions from Trinity
  async getObjectInfo(): Promise<NodeDefinitions> {
    return invoke('get_object_info');
  }

  // Parse Python file to node graph
  async parsePythonFile(path: string): Promise<NodeGraph> {
    return invoke('parse_python_file', { path });
  }

  // Generate Python from node graph (Project 4)
  async generatePython(graph: NodeGraph): Promise<string> {
    return invoke('generate_python', { graph });
  }

  // Subscribe to Python backend events
  async subscribeToEvents(callback: (event: BackendEvent) => void) {
    return listen('backend_event', (event) => {
      callback(event.payload as BackendEvent);
    });
  }

  // Native file dialogs
  async openPythonFile(): Promise<string | null> {
    return invoke('open_file_dialog', {
      filters: [{ name: 'Python', extensions: ['py'] }]
    });
  }
}
```

### Layer 3: Tauri Core (Rust)

Handles OS integration, window management, and Python process orchestration:

```rust
// src-tauri/src/main.rs

mod commands;
mod sidecar;

use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            // Spawn Python sidecar on startup
            let python_sidecar = sidecar::PythonSidecar::spawn()?;
            app.manage(python_sidecar);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_object_info,
            commands::parse_python_file,
            commands::generate_python,
            commands::open_file_dialog,
            commands::save_file_dialog,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**Sidecar Manager:**

```rust
// src-tauri/src/sidecar/mod.rs

use std::process::{Child, Command, Stdio};
use std::io::{BufReader, BufWriter, BufRead, Write};

pub struct PythonSidecar {
    process: Child,
    stdin: BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
}

impl PythonSidecar {
    pub fn spawn() -> Result<Self, Box<dyn std::error::Error>> {
        let mut process = Command::new("python")
            .args(["-m", "flowforge_backend"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = BufWriter::new(process.stdin.take().unwrap());
        let stdout = BufReader::new(process.stdout.take().unwrap());

        Ok(Self { process, stdin, stdout })
    }

    pub fn send_request(&mut self, method: &str, params: Value) -> Result<Value> {
        let request = json!({
            "id": uuid::Uuid::new_v4().to_string(),
            "method": method,
            "params": params
        });

        writeln!(self.stdin, "{}", request.to_string())?;
        self.stdin.flush()?;

        let mut response = String::new();
        self.stdout.read_line(&mut response)?;

        Ok(serde_json::from_str(&response)?)
    }
}
```

### Layer 4: Python Sidecar (Trinity Integration)

The Python backend handles Trinity introspection and AST operations:

```
flowforge_backend/
├── __init__.py
├── __main__.py              # Entry point: python -m flowforge_backend
├── main.py                  # IPC loop
│
├── ipc/
│   ├── __init__.py
│   ├── protocol.py          # JSON message handling
│   └── handlers.py          # Route to appropriate handler
│
├── trinity_adapter/
│   ├── __init__.py
│   ├── introspector.py      # Examine Trinity classes at runtime
│   ├── schema_builder.py    # Build node schemas from decorators
│   └── executor.py          # Execute node operations via Trinity
│
├── ast_parser/
│   ├── __init__.py
│   ├── python_parser.py     # Parse .py files to AST
│   ├── decorator_extractor.py  # Extract @component, @system, etc.
│   ├── metaclass_analyzer.py   # Analyze metaclass relationships
│   └── graph_builder.py     # Convert AST to node graph JSON
│
└── codegen/
    ├── __init__.py
    ├── graph_to_ast.py      # Convert node graph back to AST
    └── python_emitter.py    # Emit valid Python code
```

---

## IPC Protocol Design

### Tauri ↔ Frontend (Tauri IPC)

```typescript
// Frontend → Tauri
invoke('method_name', { param1: 'value', param2: 123 })

// Tauri → Frontend (events)
listen('event_name', (event) => {
    console.log(event.payload);
});
```

### Tauri ↔ Python (stdio JSON)

**Message Format:**

```typescript
interface IPCMessage {
    id: string;           // Request ID for correlation
    type: 'request' | 'response' | 'event';

    // For requests
    method?: string;
    params?: any;

    // For responses
    result?: any;
    error?: { code: number; message: string };

    // For events
    event?: string;
    payload?: any;
}
```

**Protocol Flow:**

```
Tauri (Rust)                           Python
────────────                           ──────

spawn python process ──────────────→  start, listen stdin
        │
        │  {"id":"1","method":"get_object_info","params":{}}
        ├─────────────────────────────→
        │                              introspect Trinity
        │                                    │
        │  {"id":"1","result":{"nodes":{...}}}
        ←─────────────────────────────┤
        │
forward to frontend
```

---

## Trinity Integration

### How Trinity Maps to Nodes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRINITY → NODE GRAPH MAPPING                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRINITY CONCEPT              NODE REPRESENTATION                           │
│  ═══════════════              ═══════════════════                           │
│                                                                             │
│  @component decorator    →    Node Type: "Trinity/Component"               │
│                               Inputs: field definitions                     │
│                               Outputs: the component instance               │
│                                                                             │
│  @system decorator       →    Node Type: "Trinity/System"                  │
│                               Inputs: query parameters                      │
│                               Outputs: execution result                     │
│                                                                             │
│  ComponentMeta          →    Meta-node showing class hierarchy             │
│                               Connections: inheritance relationships        │
│                                                                             │
│  TrackedDescriptor      →    Property node with change tracking            │
│                               Shows: field name, type, validation          │
│                                                                             │
│  Registry.register()    →    Edge connecting class to registry             │
│                                                                             │
│  Foundation.Mirror      →    Introspection node showing live state         │
│                                                                             │
│  Foundation.Inspector   →    Debug node for runtime inspection             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Schema Export from Trinity

```python
# flowforge_backend/trinity_adapter/schema_builder.py

from trinity.decorators import get_all_decorators
from trinity.metaclasses import ComponentMeta, SystemMeta
from foundation import Registry

def build_node_schemas() -> dict:
    """Convert Trinity definitions to ComfyUI-compatible node schemas."""

    schemas = {}

    # Export component types
    for cls in Registry.get_all(ComponentMeta):
        schemas[f"Trinity/Component/{cls.__name__}"] = {
            "input": {
                "required": {
                    field.name: [field.type_name, {"default": field.default}]
                    for field in cls._trinity_fields_
                }
            },
            "output": [cls.__name__],
            "output_name": ["instance"],
            "category": "Trinity/Components",
            "display_name": cls.__name__,
            "description": cls.__doc__ or f"Component: {cls.__name__}",
        }

    # Export system types
    for cls in Registry.get_all(SystemMeta):
        schemas[f"Trinity/System/{cls.__name__}"] = {
            "input": {
                "required": {
                    "query": ["QUERY", {}],
                }
            },
            "output": ["RESULT"],
            "output_name": ["result"],
            "category": "Trinity/Systems",
            "display_name": cls.__name__,
            "description": cls.__doc__ or f"System: {cls.__name__}",
        }

    return schemas
```

---

## AST-to-Node Conversion

### Project 2: Parsing Python to Nodes

```python
# flowforge_backend/ast_parser/python_parser.py

import ast
from pathlib import Path
from typing import Dict, List

class TrinityASTVisitor(ast.NodeVisitor):
    """Extract Trinity patterns from Python AST."""

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_id = 0

    def visit_ClassDef(self, node: ast.ClassDef):
        # Check for Trinity decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        if 'component' in decorators:
            self._add_component_node(node)
        elif 'system' in decorators:
            self._add_system_node(node)

        # Check metaclass
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                self._add_metaclass_edge(node, keyword.value)

        self.generic_visit(node)

    def _add_component_node(self, node: ast.ClassDef):
        node_id = self._next_id()

        # Extract fields from class body
        fields = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                fields.append({
                    "name": item.target.id,
                    "type": ast.unparse(item.annotation),
                    "default": ast.unparse(item.value) if item.value else None
                })

        self.nodes.append({
            "id": node_id,
            "type": "Trinity/Component",
            "title": node.name,
            "pos": [100, 100 + node_id * 150],
            "data": {
                "class_name": node.name,
                "fields": fields,
                "decorators": [ast.unparse(d) for d in node.decorator_list],
                "lineno": node.lineno,
            }
        })

        return node_id

def parse_python_file(path: str) -> Dict:
    """Parse a Python file and return node graph JSON."""

    source = Path(path).read_text()
    tree = ast.parse(source)

    visitor = TrinityASTVisitor()
    visitor.visit(tree)

    return {
        "nodes": visitor.nodes,
        "edges": visitor.edges,
        "source_file": path,
    }
```

### Project 4: Generating Python from Nodes

```python
# flowforge_backend/codegen/python_emitter.py

import ast
from typing import Dict, List

def generate_python(graph: Dict) -> str:
    """Convert node graph back to Python source code."""

    module = ast.Module(body=[], type_ignores=[])

    # Add imports
    module.body.append(
        ast.ImportFrom(
            module='trinity.decorators',
            names=[ast.alias(name='component'), ast.alias(name='system')],
            level=0
        )
    )

    # Convert each node to AST
    for node in graph['nodes']:
        if node['type'] == 'Trinity/Component':
            class_def = _build_component_class(node)
            module.body.append(class_def)
        elif node['type'] == 'Trinity/System':
            class_def = _build_system_class(node)
            module.body.append(class_def)

    # Emit valid Python
    return ast.unparse(ast.fix_missing_locations(module))

def _build_component_class(node: Dict) -> ast.ClassDef:
    """Build a @component class from node data."""

    data = node['data']

    # Build field annotations
    body = []
    for field in data['fields']:
        body.append(ast.AnnAssign(
            target=ast.Name(id=field['name'], ctx=ast.Store()),
            annotation=ast.Name(id=field['type'], ctx=ast.Load()),
            value=ast.parse(field['default']).body[0].value if field['default'] else None,
            simple=1
        ))

    return ast.ClassDef(
        name=data['class_name'],
        bases=[],
        keywords=[],
        body=body or [ast.Pass()],
        decorator_list=[
            ast.Name(id='component', ctx=ast.Load())
        ]
    )
```

---

## Data Flow Diagrams

### View Mode (Project 3)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VIEW MODE DATA FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User: "Open player.py"                                                     │
│        │                                                                    │
│        ▼                                                                    │
│  Frontend: invoke('open_file_dialog', {filters: ['*.py']})                 │
│        │                                                                    │
│        ▼                                                                    │
│  Tauri: Native file picker → "/path/to/player.py"                          │
│        │                                                                    │
│        ▼                                                                    │
│  Frontend: invoke('parse_python_file', {path})                             │
│        │                                                                    │
│        ▼                                                                    │
│  Python: TrinityASTVisitor.visit(ast.parse(source))                        │
│        │                                                                    │
│        ▼                                                                    │
│  Python: Returns { nodes: [...], edges: [...] }                            │
│        │                                                                    │
│        ▼                                                                    │
│  Frontend: graph.configure(nodeGraph)                                       │
│        │                                                                    │
│        ▼                                                                    │
│  LiteGraph: Renders nodes on canvas                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edit Mode (Project 4)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EDIT MODE DATA FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User: Adds new field to Component node                                     │
│        │                                                                    │
│        ▼                                                                    │
│  LiteGraph: Updates node.data.fields                                        │
│        │                                                                    │
│        ▼                                                                    │
│  Frontend: invoke('generate_python', {graph})                              │
│        │                                                                    │
│        ▼                                                                    │
│  Python: graph_to_ast.py → python_emitter.py                               │
│        │                                                                    │
│        ▼                                                                    │
│  Python: Returns "from trinity import component\n\n@component\nclass..."   │
│        │                                                                    │
│        ▼                                                                    │
│  Frontend: Shows diff / preview                                             │
│        │                                                                    │
│        ▼                                                                    │
│  User: Confirms → invoke('write_file', {path, content})                    │
│        │                                                                    │
│        ▼                                                                    │
│  Tauri: Writes Python file to disk                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Security Model

### Sandboxing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SECURITY LAYERS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. TAURI CAPABILITY SYSTEM                                                 │
│     - File access scoped to user-selected files                             │
│     - No arbitrary filesystem access                                        │
│     - IPC commands explicitly allowlisted                                   │
│                                                                             │
│  2. PYTHON SIDECAR ISOLATION                                                │
│     - Runs as separate process                                              │
│     - Communicates only via stdio                                           │
│     - No direct access to Tauri APIs                                        │
│                                                                             │
│  3. AST-ONLY PARSING                                                        │
│     - Python files parsed as AST, never executed during parse               │
│     - No eval() or exec() of user code                                      │
│     - Safe introspection of Trinity structures                              │
│                                                                             │
│  4. CODE GENERATION VALIDATION                                              │
│     - Generated Python validated against AST before emit                    │
│     - Syntax errors caught before file write                                │
│     - Optional: diff review before save                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

FlowForge bridges the gap between visual programming and Python metaprogramming:

1. **Project 1:** Native desktop shell (Tauri + forked ComfyUI frontend + Python sidecar)
2. **Project 2:** Parse Trinity Python code into visual node graphs
3. **Project 3:** View-only mode for code exploration
4. **Project 4:** Bidirectional editing with Python code generation

The architecture preserves ComfyUI's proven canvas technology while connecting it to Trinity's powerful metaprogramming system.
