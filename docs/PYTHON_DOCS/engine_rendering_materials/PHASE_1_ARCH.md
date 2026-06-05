# PHASE 1 ARCHITECTURE: Core Material System Validation

## Overview

Phase 1 focuses on validating the core material system components that are already marked as REAL: material templates, instances, functions, layers, and the central registry.

## Components

### MaterialTemplate
- Base shader definition with parameter schema
- Defines the structure that instances override
- Contains default values for all parameters

### MaterialInstance
- Overrides specific parameters from a template
- Uses weak references to parent template
- Propagates dirty flags on parameter change

### MaterialFunction
- Reusable shader snippets
- Dependency tracking between functions
- Embeddable code blocks with uniform declarations

### MaterialLayer
- Composable material stacking
- Blend modes (alpha, additive, multiply)
- Layer ordering and priority

### MaterialSystem
- Central registry for templates, instances, functions
- Hot-reload support for development iteration
- Garbage collection of unused instances

### DirtyFlags
- `PARAMETERS` - Scalar uniform changes
- `TEXTURES` - Texture binding changes  
- `SHADER` - Graph recompilation required

## Architecture Decisions

### AD-1: Weak Reference Ownership
**Decision**: MaterialInstance holds weak references to MaterialTemplate.
**Rationale**: Enables safe hot-reload without dangling pointers. Template invalidation propagates cleanly.
**Consequences**: Instances must check template validity before use.

### AD-2: Dirty Flag Granularity
**Decision**: Three-level dirty flags (parameters, textures, shader).
**Rationale**: GPU upload cost varies; only update what changed.
**Consequences**: Renderer must query and clear flags appropriately.

### AD-3: Centralized Registry
**Decision**: MaterialSystem as single source of truth.
**Rationale**: Simplifies lookup, hot-reload, and garbage collection.
**Consequences**: Thread safety required for concurrent access.

## Validation Strategy

1. Unit test each component in isolation
2. Integration test template->instance->dirty flag flow
3. Stress test hot-reload with rapid template changes
4. Profile memory for weak reference cleanup
