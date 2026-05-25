# PHASE 2 ARCHITECTURE: GPU Particle Compute Integration

## Overview
Complete the GPU particle compute dispatch integration with RHI. The architecture is fully defined; GPU dispatch calls are placeholders awaiting RHI binding.

## Current State

### GPUParticleSystem Components
- **GPUParticleBuffer**: SoA (Structure of Arrays) attribute storage
- **GPUParticleAttributes**: Per-particle data layout
- **GPUParticleSimulator**: Compute shader orchestration (stubbed)
- **GPUParticleRenderer**: Draw call preparation (stubbed)

### Stubbed Dispatch Pattern
From `gpu_particles.py` (lines 452-472):
```python
def update(self, dt: float) -> None:
    if self._buffer.alive_count == 0:
        return
    dispatch_size = self._calculate_dispatch_size(self._buffer.alive_count)
    # In actual implementation, this would:
    # 1. Bind update compute shader
    # 2. Set uniforms (dt, gravity, wind, etc.)
    # 3. Bind attribute buffers
    # 4. Dispatch compute shader
    # 5. Memory barrier
    # Placeholder for simulation logic
    pass
```

## Target Architecture

### Compute Shader Pipeline
```
1. EMIT PASS
   - Read spawn parameters
   - Initialize new particle attributes
   - Atomic increment alive count

2. UPDATE PASS
   - Read current attributes
   - Apply forces (gravity, wind, turbulence)
   - Integrate velocity -> position
   - Update age, check lifetime
   - Write new attributes

3. COMPACT PASS (optional)
   - Stream compaction for dead particles
   - Rebuild indirect draw count

4. SORT PASS (optional)
   - Bitonic sort by depth for transparency
```

### Buffer Layout (SoA)
```
Buffer 0: positions[N]     - vec4 (xyz + padding)
Buffer 1: velocities[N]    - vec4 (xyz + padding)
Buffer 2: colors[N]        - vec4 (rgba)
Buffer 3: attributes[N]    - vec4 (size, rotation, age, lifetime)
Buffer 4: state[N]         - uint (alive/dead flags)
Buffer 5: indirect_args    - DrawIndirectCommand
```

### Dispatch Size Calculation
```python
workgroup_size = 256
dispatch_size = (alive_count + workgroup_size - 1) // workgroup_size
```

## RHI Integration Points

### Required RHI Calls
| Operation | RHI Method |
|-----------|------------|
| Create compute pipeline | `create_compute_pipeline(shader)` |
| Bind compute pipeline | `bind_compute_pipeline(pipeline)` |
| Set push constants | `set_push_constants(data)` |
| Bind storage buffers | `bind_storage_buffer(slot, buffer)` |
| Dispatch compute | `dispatch_compute(x, y, z)` |
| Memory barrier | `pipeline_barrier(COMPUTE_TO_COMPUTE)` |

### Buffer Synchronization
- COMPUTE_WRITE -> COMPUTE_READ barrier between emit and update
- COMPUTE_WRITE -> VERTEX_READ barrier before rendering
- Double/triple buffering for CPU readback if needed

## Decisions

### ADR-GPU-001: SoA Buffer Layout
- **Context**: Need cache-efficient GPU access patterns
- **Decision**: Structure of Arrays (SoA) over Array of Structures (AoS)
- **Consequence**: Each attribute in contiguous buffer, better coalescing

### ADR-GPU-002: Indirect Draw for Dynamic Count
- **Context**: Alive particle count varies each frame
- **Decision**: Use indirect draw with GPU-written count
- **Consequence**: No CPU readback latency for draw call setup

### ADR-GPU-003: CPU Fallback Path
- **Context**: Not all platforms support compute shaders
- **Decision**: Maintain CPU simulation as fallback
- **Consequence**: Graceful degradation, unified API surface

### ADR-GPU-004: Workgroup Size 256
- **Context**: Balance occupancy vs register pressure
- **Decision**: 256 threads per workgroup
- **Consequence**: Good occupancy on most GPUs, divisible by warp/wavefront size
