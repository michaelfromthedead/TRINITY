# PHASE 4 ARCHITECTURE: VFX Graph System

## Overview
The VFX Graph provides node-based visual authoring of particle effects. It compiles to ParticleEmitter configurations at edit time, not runtime interpretation.

## Core Components

### VFXGraph
- Container for VFX nodes and connections
- Maintains dirty flag for incremental compilation
- Caches compiled ParticleEmitter

### VFXNode Types
- **VFXEmitterModule**: Emitter configuration (capacity, duration, loop)
- **VFXSpawnNode**: Spawn shape, rate, burst
- **VFXForceNode**: Gravity, wind, turbulence, vortex
- **VFXAttributeNode**: Size, color, rotation over lifetime
- **VFXRenderNode**: Billboard, mesh renderer configuration

### Compilation Pipeline
From `vfx_graph.py` (lines 830-870):
```python
def compile(self) -> ParticleEmitter:
    if not self._dirty and self._compiled_emitter:
        return self._compiled_emitter
    
    self._categorize_modules()  # Sort into spawn/update/render
    
    emitter_config = EmitterConfig()
    for module in self._spawn_modules:
        if isinstance(module, VFXEmitterModule):
            emitter_config = module.to_emitter_config()
            break
    
    emitter = ParticleEmitter(config=emitter_config)
    
    for module in self._spawn_modules:
        pm = module.to_particle_module()
        if pm and pm.stage == ModuleStage.SPAWN:
            emitter.add_spawn_module(pm)
    
    # ... add update and render modules ...
    
    self._dirty = False
    self._compiled_emitter = emitter
    return emitter
```

## Module Categorization

### Stage Assignment
Each VFX node maps to a particle module with explicit stage:
```
VFXSpawnNode      -> ModuleStage.SPAWN
VFXForceNode      -> ModuleStage.UPDATE
VFXAttributeNode  -> ModuleStage.UPDATE
VFXRenderNode     -> ModuleStage.RENDER
```

### Categorization Algorithm
1. Traverse all nodes in graph
2. Convert each VFXNode to corresponding ParticleModule
3. Query module stage
4. Append to stage-specific list

## Node Connections

### Connection Types
- **Data Flow**: Output attribute -> Input attribute
- **Execution Order**: Implicit from stage membership

### Validation
Connections validated at compile time:
- Type compatibility (Vec3 to Vec3, float to float)
- Stage ordering (spawn cannot depend on render)
- No cycles (DAG requirement)

## Caching Strategy

### Dirty Flag
```python
self._dirty = True   # Set when graph modified
self._dirty = False  # Set after compilation
```

### Cache Invalidation
Graph marked dirty when:
- Node added/removed
- Connection added/removed
- Node parameter changed

## Decisions

### ADR-VFX-001: Compile-Time Over Runtime
- **Context**: Graph could be interpreted at runtime or compiled
- **Decision**: Compile to static module list at edit time
- **Consequence**: Zero graph traversal overhead during simulation

### ADR-VFX-002: Stage-Based Module Organization
- **Context**: Modules must execute in correct order
- **Decision**: Explicit SPAWN/UPDATE/RENDER stages
- **Consequence**: Clear execution model, matches particle pipeline

### ADR-VFX-003: Dirty Flag Caching
- **Context**: Compilation may be expensive for complex graphs
- **Decision**: Cache compiled emitter, invalidate on change
- **Consequence**: Incremental workflow, no redundant compilation

### ADR-VFX-004: DAG Requirement
- **Context**: Cycles would cause infinite loops
- **Decision**: Validate acyclic graph at compile time
- **Consequence**: All valid graphs have finite evaluation
