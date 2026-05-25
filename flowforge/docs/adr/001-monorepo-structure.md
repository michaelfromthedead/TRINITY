# ADR-001: Monorepo Structure

**Status:** Accepted
**Date:** 2026-01-27
**Decision Makers:** System Architecture Team

## Context

FlowForge is a domain-agnostic visual programming environment derived from ComfyUI, targeting a Tauri + Bun architecture. We need to establish the foundational project structure that:

1. Supports multiple deployment targets (desktop, web)
2. Enables code sharing between frontend and backend
3. Facilitates plugin development with a clear SDK
4. Allows independent versioning of packages
5. Provides efficient build pipelines

## Decision

We will use a **Turborepo-based monorepo** with the following structure:

```
flowforge/
├── apps/
│   ├── desktop/           # Tauri 2 app (Vue 3 frontend + Rust backend)
│   └── web/               # Future web-only version
├── packages/
│   ├── core/              # Shared types, schemas, graph algorithms
│   ├── engine/            # Bun execution engine (sidecar)
│   ├── sdk/               # Plugin SDK (definePlugin, defineNode, defineType)
│   └── nodes-builtin/     # Built-in nodes (math, logic, string, flow-control)
├── config/                # Shared ESLint, Prettier configs
├── docs/                  # Documentation and ADRs
└── scripts/               # Build and development scripts
```

### Package Responsibilities

| Package | Purpose | Dependencies |
|---------|---------|--------------|
| `@flowforge/core` | Type definitions, Zod schemas, graph algorithms | zod |
| `@flowforge/engine` | Bun sidecar, IPC handler, executor | @flowforge/core |
| `@flowforge/sdk` | Plugin authoring utilities | @flowforge/core |
| `@flowforge/nodes-builtin` | Essential node implementations | @flowforge/core, @flowforge/sdk |
| `@flowforge/desktop` | Tauri desktop application | @flowforge/core, @tauri-apps/* |

### Key Interfaces (packages/core)

1. **NodeDefinition** - Schema for node types with inputs, outputs, widgets
2. **WorkflowSchema** - Complete workflow structure with nodes, links, groups
3. **ExecutionContext** - Runtime context passed to node execute functions
4. **IPCMessage** - JSON-RPC style messages for Tauri/Bun communication

### IPC Protocol

Communication between Tauri (Rust) and Bun uses a stdio-based JSON protocol:

```typescript
interface IPCMessage {
  id: string;
  type: 'request' | 'response' | 'event';
  timestamp: number;
  // request: method, params
  // response: requestId, result | error
  // event: event, payload
}
```

## Rationale

### Why Turborepo?

- Fast incremental builds with caching
- Parallel task execution
- Clear dependency graph
- Works well with Bun's package management

### Why Separate Engine from Desktop?

- Engine runs as a sidecar process for isolation
- Can be replaced with different execution backends
- Enables future serverless deployment

### Why SDK as Separate Package?

- Clear API surface for plugin authors
- Minimal dependencies for plugin development
- Version independently from core

## Consequences

### Positive

- Clear separation of concerns
- Independent versioning per package
- Efficient incremental builds
- Plugin authors only need SDK

### Negative

- More complex initial setup
- Need to manage workspace dependencies
- Build configuration spread across packages

### Neutral

- Requires Turborepo knowledge
- Additional package.json files to maintain

## Alternatives Considered

1. **Single package** - Rejected due to lack of modularity
2. **Lerna** - Rejected in favor of simpler Turborepo
3. **Nx** - Rejected as heavier than needed for this project

## Related Decisions

- ADR-002: IPC Protocol Design (pending)
- ADR-003: Plugin Security Model (pending)
- ADR-004: Graph Execution Strategy (pending)
