# PHASE 1 ARCHITECTURE: GPU Cloth Completion

## Objective

Complete the GPU cloth simulation backend by implementing actual wgpu compute shader dispatch to replace the current PARTIAL STUB.

## Current State

### Existing Components (from gpu_cloth.py, 572 lines)

- `GPUBuffer`: Abstract GPU memory buffer interface
- `GPUClothBuffers`: Defines required buffers (positions, velocities, inverse_masses, constraints)
- `GPUComputePipeline`: Abstract compute shader dispatch interface
- `GPUClothSolverStub`: No-op solver with explicit warning
- Shader templates (GLSL):
  - `INTEGRATION_SHADER_TEMPLATE`
  - `DISTANCE_CONSTRAINT_SHADER_TEMPLATE`
  - `VELOCITY_UPDATE_SHADER_TEMPLATE`

### Missing Components

- wgpu device/queue acquisition from renderer-backend
- Shader compilation (SPIR-V or WGSL conversion)
- Buffer upload/download for CPU-GPU data transfer
- Actual compute dispatch calls
- Synchronization for collision handling

## Architecture Decisions

### ADR-GPU-CLOTH-001: Use renderer-backend wgpu Integration

**Decision**: Acquire wgpu device and queue from renderer-backend rather than creating independent GPU context.

**Rationale**: 
- Avoids multiple GPU contexts competing for resources
- Enables buffer sharing with render pipeline
- Consistent resource management lifecycle

**Consequence**: GPU cloth depends on renderer-backend initialization.

### ADR-GPU-CLOTH-002: WGSL Shader Language

**Decision**: Convert GLSL shader templates to WGSL for native wgpu compatibility.

**Rationale**:
- wgpu uses WGSL natively
- Avoids GLSL-to-SPIR-V compilation step
- Better error messages during development

**Consequence**: Shader templates must be rewritten in WGSL syntax.

### ADR-GPU-CLOTH-003: Double-Buffered Positions

**Decision**: Use double-buffered position arrays (current/predicted) for PBD iteration.

**Rationale**:
- PBD predict step writes to predicted buffer
- Constraint projection reads current, writes predicted
- Final update swaps buffers
- Avoids read-write hazards in compute shaders

**Consequence**: 2x position buffer memory requirement.

### ADR-GPU-CLOTH-004: Collision on CPU, Simulation on GPU

**Decision**: Download particle positions to CPU for collision detection, upload corrected positions back.

**Rationale**:
- Collision detection against world geometry requires scene data access
- CPU collision code (cloth_collision.py) is production-quality
- Hybrid approach avoids duplicating collision logic in shaders

**Consequence**: One GPU-CPU round-trip per simulation step.

## Component Design

```
renderer-backend
      |
      v
+---------------------+
| wgpu device/queue   |
+---------------------+
      |
      v
+---------------------+
| GPUClothSolver      |
| - compile_shaders() |
| - upload_buffers()  |
| - dispatch()        |
| - download_buffers()|
+---------------------+
      |
      v
+---------------------+      +---------------------+
| GPU Compute Passes  | ---> | ClothCollisionHandler|
| - integrate         |      | (CPU, existing)     |
| - distance_project  |      +---------------------+
| - velocity_update   |
+---------------------+
```

## Buffer Layout

| Buffer | Type | Size | Usage |
|--------|------|------|-------|
| positions_current | vec4<f32> | N particles | read |
| positions_predicted | vec4<f32> | N particles | read/write |
| velocities | vec4<f32> | N particles | read/write |
| inverse_masses | f32 | N particles | read |
| distance_constraints | Constraint | M edges | read |
| params | SimParams | 1 | uniform |

## Compute Pipeline Stages

1. **Integration**: Apply gravity and external forces, predict positions
2. **Distance Constraint Projection**: Iteratively satisfy edge length constraints
3. **Velocity Update**: Compute new velocities from position changes
4. **Download**: Copy positions to CPU for collision handling
5. **Collision** (CPU): Apply collision corrections via existing ClothCollisionHandler
6. **Upload**: Copy corrected positions back to GPU

## Integration Points

- **renderer-backend**: Provides wgpu device, queue, and potentially shared buffers
- **cloth_collision.py**: Existing CPU collision handler receives downloaded positions
- **cloth_simulation.py**: GPU solver must match CPU PBD interface for drop-in replacement

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| wgpu API changes | High | Pin wgpu version, abstract behind stable interface |
| Shader compilation failures | Medium | Extensive shader unit tests |
| Performance regression vs CPU | High | Benchmark against CPU baseline, optimize workgroup sizes |
| Collision sync overhead | Medium | Profile round-trip latency, consider async collision |
