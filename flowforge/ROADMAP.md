# FlowForge Roadmap

**Visual Programming Interface for Trinity Python Metaprogramming**

---

## Overview

FlowForge is structured as **four sequential projects**, each building on the previous:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROJECT TIMELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROJECT 1          PROJECT 2          PROJECT 3          PROJECT 4        │
│  Native Shell       AST Parser         View Mode          Edit Mode         │
│                                                                             │
│  ████████░░░░        ░░░░░░░░░░         ░░░░░░░░░░         ░░░░░░░░░░       │
│  ~40%                0%                 0%                 0%               │
│                                                                             │
│  Tauri + ComfyUI    Python → Nodes     Read-only viz      Bidirectional    │
│  + Python sidecar                                         code gen          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project 1: Native Desktop Shell

**Goal:** Tauri application with forked ComfyUI frontend and Python sidecar

### Phase 1.1: Pivot Cleanup (CURRENT)

| Task | Status | Description |
|------|--------|-------------|
| 1.1.1 | ✅ Done | Tauri shell with Rust commands |
| 1.1.2 | ✅ Done | File dialog integration |
| 1.1.3 | ⏳ TODO | Delete Bun engine (`packages/engine/`) |
| 1.1.4 | ⏳ TODO | Delete TypeScript nodes (`packages/nodes-builtin/`) |
| 1.1.5 | ⏳ TODO | Delete TypeScript SDK (`packages/sdk/`) |
| 1.1.6 | ⏳ TODO | Update sidecar manager: Bun → Python |

**Checkpoint:** Clean monorepo with only Tauri shell remaining

### Phase 1.2: Fork ComfyUI Frontend

| Task | Status | Description |
|------|--------|-------------|
| 1.2.1 | ⏳ TODO | Clone ComfyUI_frontend repository |
| 1.2.2 | ⏳ TODO | Copy LiteGraph.js to `apps/desktop/src/litegraph/` |
| 1.2.3 | ⏳ TODO | Copy Vue components to `apps/desktop/src/components/` |
| 1.2.4 | ⏳ TODO | Copy Pinia stores to `apps/desktop/src/stores/` |
| 1.2.5 | ⏳ TODO | Remove Stable Diffusion-specific code |
| 1.2.6 | ⏳ TODO | Strip Python API fetch calls |

**Checkpoint:** Frontend runs in Tauri, shows empty canvas

### Phase 1.3: Bridge Layer

| Task | Status | Description |
|------|--------|-------------|
| 1.3.1 | ⏳ TODO | Create `src/bridge/tauri-api.ts` |
| 1.3.2 | ⏳ TODO | Replace ComfyUI `api.ts` with Tauri IPC calls |
| 1.3.3 | ⏳ TODO | Wire Pinia stores to bridge layer |
| 1.3.4 | ⏳ TODO | Add file open/save via native dialogs |

**Checkpoint:** Frontend communicates with Tauri backend

### Phase 1.4: Python Sidecar

| Task | Status | Description |
|------|--------|-------------|
| 1.4.1 | ⏳ TODO | Create `flowforge_backend/` Python package |
| 1.4.2 | ⏳ TODO | Implement stdio JSON IPC protocol |
| 1.4.3 | ⏳ TODO | Update Rust sidecar manager to spawn Python |
| 1.4.4 | ⏳ TODO | Add basic request/response handling |
| 1.4.5 | ⏳ TODO | Implement `get_object_info` stub |

**Checkpoint:** Python sidecar starts, responds to requests

### Phase 1.5: End-to-End Integration

| Task | Status | Description |
|------|--------|-------------|
| 1.5.1 | ⏳ TODO | Frontend loads node definitions from Python |
| 1.5.2 | ⏳ TODO | Canvas renders with Trinity node palette |
| 1.5.3 | ⏳ TODO | Native file dialogs work for .py files |

**Milestone:** App opens, shows canvas, can open Python files

---

## Project 2: Python AST Parser

**Goal:** Parse Trinity Python code and convert to node graph format

### Phase 2.1: AST Visitor

| Task | Status | Description |
|------|--------|-------------|
| 2.1.1 | ⏳ TODO | Create `TrinityASTVisitor` class |
| 2.1.2 | ⏳ TODO | Detect `@component` decorated classes |
| 2.1.3 | ⏳ TODO | Detect `@system` decorated classes |
| 2.1.4 | ⏳ TODO | Detect `@resource` decorated classes |
| 2.1.5 | ⏳ TODO | Detect `@event` decorated classes |
| 2.1.6 | ⏳ TODO | Extract class fields with types and defaults |
| 2.1.7 | ⏳ TODO | Extract method signatures |

**Checkpoint:** Visitor extracts Trinity patterns from AST

### Phase 2.2: Node Graph Builder

| Task | Status | Description |
|------|--------|-------------|
| 2.2.1 | ⏳ TODO | Define node graph JSON schema |
| 2.2.2 | ⏳ TODO | Convert Component classes to nodes |
| 2.2.3 | ⏳ TODO | Convert System classes to nodes |
| 2.2.4 | ⏳ TODO | Build edges from import relationships |
| 2.2.5 | ⏳ TODO | Build edges from type references |
| 2.2.6 | ⏳ TODO | Layout algorithm for node positioning |

**Checkpoint:** AST → node graph JSON conversion works

### Phase 2.3: IPC Integration

| Task | Status | Description |
|------|--------|-------------|
| 2.3.1 | ⏳ TODO | Implement `parse_python_file` command |
| 2.3.2 | ⏳ TODO | Handle multi-file projects |
| 2.3.3 | ⏳ TODO | Cache parsed AST for performance |
| 2.3.4 | ⏳ TODO | Incremental re-parsing on file change |

**Milestone:** Open .py file → see nodes on canvas

---

## Project 3: View-Only Mode

**Goal:** Interactive visualization of Trinity code structure

### Phase 3.1: Node Rendering

| Task | Status | Description |
|------|--------|-------------|
| 3.1.1 | ⏳ TODO | Register Trinity node types with LiteGraph |
| 3.1.2 | ⏳ TODO | Custom rendering for Component nodes |
| 3.1.3 | ⏳ TODO | Custom rendering for System nodes |
| 3.1.4 | ⏳ TODO | Custom rendering for Resource nodes |
| 3.1.5 | ⏳ TODO | Custom rendering for Event nodes |
| 3.1.6 | ⏳ TODO | Edge styling (inheritance, references) |

**Checkpoint:** Nodes render with Trinity-specific appearance

### Phase 3.2: Navigation

| Task | Status | Description |
|------|--------|-------------|
| 3.2.1 | ⏳ TODO | Click node → highlight source line |
| 3.2.2 | ⏳ TODO | Double-click → open in external editor |
| 3.2.3 | ⏳ TODO | Search nodes by name |
| 3.2.4 | ⏳ TODO | Filter by node type |
| 3.2.5 | ⏳ TODO | Minimap for large graphs |

**Checkpoint:** Users can navigate and explore the codebase

### Phase 3.3: Trinity Introspection

| Task | Status | Description |
|------|--------|-------------|
| 3.3.1 | ⏳ TODO | Connect to live Trinity runtime |
| 3.3.2 | ⏳ TODO | Show Registry contents |
| 3.3.3 | ⏳ TODO | Show active instances (Mirror) |
| 3.3.4 | ⏳ TODO | Display EventLog events |
| 3.3.5 | ⏳ TODO | Inspector integration for debugging |

**Milestone:** Full read-only visualization of Trinity code and runtime

---

## Project 4: Bidirectional Editing

**Goal:** Edit nodes visually and regenerate valid Python code

### Phase 4.1: Edit Operations

| Task | Status | Description |
|------|--------|-------------|
| 4.1.1 | ⏳ TODO | Add field to Component node |
| 4.1.2 | ⏳ TODO | Remove field from Component node |
| 4.1.3 | ⏳ TODO | Rename class/field |
| 4.1.4 | ⏳ TODO | Change field type |
| 4.1.5 | ⏳ TODO | Change field default value |
| 4.1.6 | ⏳ TODO | Add new Component/System node |
| 4.1.7 | ⏳ TODO | Delete node (with confirmation) |

**Checkpoint:** Edit operations tracked in node graph

### Phase 4.2: Code Generation

| Task | Status | Description |
|------|--------|-------------|
| 4.2.1 | ⏳ TODO | Implement `graph_to_ast.py` converter |
| 4.2.2 | ⏳ TODO | Implement `python_emitter.py` |
| 4.2.3 | ⏳ TODO | Preserve comments and formatting where possible |
| 4.2.4 | ⏳ TODO | Validate generated Python syntax |
| 4.2.5 | ⏳ TODO | Round-trip tests (parse → modify → generate → parse) |

**Checkpoint:** Node graph → valid Python conversion works

### Phase 4.3: Diff & Apply

| Task | Status | Description |
|------|--------|-------------|
| 4.3.1 | ⏳ TODO | Generate diff between original and modified |
| 4.3.2 | ⏳ TODO | Display diff preview UI |
| 4.3.3 | ⏳ TODO | Apply changes to file |
| 4.3.4 | ⏳ TODO | Undo/redo support |
| 4.3.5 | ⏳ TODO | Conflict detection on external file changes |

**Milestone:** Full bidirectional editing: Python ↔ Nodes

---

## Current Status Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OVERALL PROGRESS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROJECT 1: Native Shell                                                    │
│  ████████████████░░░░░░░░░░░░░░░░░░░░  40%                                 │
│  └─ Tauri shell done, ComfyUI fork pending                                 │
│                                                                             │
│  PROJECT 2: AST Parser                                                      │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0%                                  │
│  └─ Blocked on Project 1 completion                                        │
│                                                                             │
│  PROJECT 3: View Mode                                                       │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0%                                  │
│  └─ Blocked on Project 2 completion                                        │
│                                                                             │
│  PROJECT 4: Edit Mode                                                       │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0%                                  │
│  └─ Blocked on Project 3 completion                                        │
│                                                                             │
│  TOTAL: ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~10%                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## What's Completed (Keep)

| Component | Status | Notes |
|-----------|--------|-------|
| Tauri shell | ✅ Keep | Window, plugins, state management |
| Rust commands | ✅ Keep | File dialogs, IPC routing |
| CI/CD workflows | ✅ Keep | Adapt for Python |
| Monorepo structure | ✅ Keep | Simplify after cleanup |
| ESLint/Prettier | ✅ Keep | For frontend code |

## What's Deleted (Throwaway)

| Component | Status | Notes |
|-----------|--------|-------|
| `packages/engine/` | ❌ Delete | Bun execution engine |
| `packages/nodes-builtin/` | ❌ Delete | TypeScript nodes |
| `packages/sdk/` | ❌ Delete | TypeScript plugin SDK |
| Bun sidecar manager | ❌ Replace | Change to Python |

## What's New (Remaining Work)

| Component | Status | Notes |
|-----------|--------|-------|
| ComfyUI frontend fork | ⏳ TODO | LiteGraph + Vue components |
| Tauri bridge layer | ⏳ TODO | Replace fetch() with IPC |
| `flowforge_backend/` | ⏳ TODO | Python sidecar package |
| AST parser | ⏳ TODO | Python → node graph |
| Code generator | ⏳ TODO | Node graph → Python |

---

## Next Steps (Immediate)

1. **Delete throwaway code:**
   ```bash
   rm -rf packages/engine packages/nodes-builtin packages/sdk
   ```

2. **Clone ComfyUI frontend:**
   ```bash
   git clone https://github.com/Comfy-Org/ComfyUI_frontend /tmp/comfyui
   cp -r /tmp/comfyui/src/scripts/litegraph apps/desktop/src/
   cp -r /tmp/comfyui/src/components apps/desktop/src/
   ```

3. **Create Python sidecar:**
   ```bash
   mkdir -p flowforge_backend
   touch flowforge_backend/__init__.py
   touch flowforge_backend/__main__.py
   ```

4. **Update Rust sidecar manager:**
   - Change spawn command from `bun` to `python -m flowforge_backend`

---

## Testing Strategy

### Project 1 Tests
- [ ] Tauri app launches
- [ ] Python sidecar spawns and responds
- [ ] File dialogs open/save .py files
- [ ] LiteGraph canvas renders

### Project 2 Tests
- [ ] Parse simple @component class
- [ ] Parse simple @system class
- [ ] Handle nested imports
- [ ] Handle circular references

### Project 3 Tests
- [ ] Node click → source highlight
- [ ] Search finds nodes
- [ ] Large graph (100+ nodes) performance

### Project 4 Tests
- [ ] Add field → valid Python
- [ ] Remove field → valid Python
- [ ] Round-trip: parse → modify → generate → parse matches

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LiteGraph API changes in ComfyUI | Pin to specific commit |
| Complex Python syntax in codegen | Start with simple patterns, expand |
| Performance with large codebases | Lazy loading, caching |
| Trinity API changes | Abstract behind adapter layer |
