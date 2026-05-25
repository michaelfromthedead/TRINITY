# FlowForge

[![CI](https://github.com/yourorg/flowforge/actions/workflows/ci.yml/badge.svg)](https://github.com/yourorg/flowforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Visual Programming Interface for Trinity Python Metaprogramming**

*ComfyUI's Canvas Engine + Tauri + Python/Trinity*

## Overview

FlowForge is a visual programming environment that lets you **see and edit** Trinity Python code as a node graph. It takes ComfyUI's battle-tested LiteGraph canvas and connects it to the Trinity metaprogramming system, enabling visual exploration and modification of decorators, metaclasses, and descriptors.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  @component                         ┌─────────────────┐         │
│  class Player:                      │   Player        │         │
│      health: int = 100    ───►      │   ═══════       │         │
│      position: Vec3                 │   health: int   │         │
│                                     │   position: Vec3│         │
│                                     └─────────────────┘         │
│                                                                 │
│  BIDIRECTIONAL: Edit nodes ──► regenerate Python               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## The Four Projects

| Project | Goal | Status |
|---------|------|--------|
| **1** | Native desktop shell (Tauri + ComfyUI frontend + Python sidecar) | ~40% |
| **2** | Python AST parser → node graph conversion | Not started |
| **3** | View-only mode for Trinity visualization | Not started |
| **4** | Bidirectional editing with Python regeneration | Not started |

## Project Structure

```
flowforge/
├── apps/
│   └── desktop/                    # Tauri desktop application
│       ├── src/                    # Frontend (to be forked from ComfyUI)
│       │   ├── litegraph/          # Canvas engine
│       │   ├── components/         # Vue components
│       │   ├── stores/             # Pinia stores
│       │   └── bridge/             # Tauri IPC layer
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

## Prerequisites

```bash
# Rust (for Tauri)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python 3.10+ (for Trinity sidecar)
# Already installed for Trinity development

# Bun (for frontend build)
curl -fsSL https://bun.sh/install | bash
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourorg/flowforge.git
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

## Architecture

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
│  │   AST Parser  │  Trinity Adapter  │  Code Generator                 │   │
│  │                                                                      │   │
│  │   ┌──────────────────────────────────────────────────────────────┐  │   │
│  │   │                    TRINITY + FOUNDATION                       │  │   │
│  │   │  Decorators  │  Metaclasses  │  Descriptors  │  Foundation   │  │   │
│  │   └──────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

### Project 1: Native Desktop Shell
- Tauri-based native window with OS integration
- Forked ComfyUI LiteGraph canvas (60fps with 1000+ nodes)
- Python sidecar for Trinity integration
- Native file dialogs for .py files

### Project 2: AST Parser
- Parse Python files using `ast` module
- Extract Trinity decorators (@component, @system, @resource, @event)
- Build node graph JSON from Python AST
- Track class relationships and dependencies

### Project 3: View Mode
- Visual representation of Trinity code structure
- Click nodes to navigate to source code
- Filter and search nodes by type/name
- Live introspection via Foundation systems

### Project 4: Bidirectional Editing
- Edit component fields visually
- Generate valid Python from node graph
- Diff preview before applying changes
- Round-trip editing with zero data loss

## Trinity Node Types

| Trinity Construct | Node Representation |
|-------------------|---------------------|
| `@component` class | Component node with field list |
| `@system` class | System node with query → result flow |
| `@resource` class | Resource node (singleton indicator) |
| `@event` class | Event node with payload fields |
| Metaclass hierarchy | Dashed edge showing inheritance |
| Descriptor binding | Property icon showing tracking |

## Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Technical architecture details
- [DESCRIPTION.md](./DESCRIPTION.md) - Project overview and vision
- [ROADMAP.md](./ROADMAP.md) - Development roadmap and status
- [docs/adr/](./docs/adr/) - Architecture Decision Records

## Current Status

```
PROJECT 1: Native Shell     ████████████████░░░░░░░░░░░░░░░░░░░░  40%
PROJECT 2: AST Parser       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
PROJECT 3: View Mode        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
PROJECT 4: Edit Mode        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
```

### What's Done
- Tauri shell with Rust commands
- File dialog integration
- Basic monorepo structure
- CI/CD workflows

### Next Steps
1. Delete Bun engine packages (throwaway)
2. Fork ComfyUI frontend (LiteGraph + Vue)
3. Create Python sidecar package
4. Wire Tauri to Python via stdio IPC

## Why Python Sidecar (Not Bun)?

Trinity **is** Python. To introspect decorators, metaclasses, and descriptors, we need Python's `ast` module and direct access to Trinity's runtime. A TypeScript/Bun backend would require reimplementing Trinity's entire metaprogramming system.

| Approach | Can Introspect Trinity? |
|----------|------------------------|
| Bun/TypeScript | No - would need to reimplement |
| Python sidecar | Yes - direct access to Trinity |

## License

MIT
