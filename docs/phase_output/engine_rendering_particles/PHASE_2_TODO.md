# PHASE 2 TODO: GPU Particle Compute Integration

## Tasks

### 2.1 Implement GPU Buffer Creation
**File**: `gpu_particles.py`
**Acceptance**:
- SoA buffers created with correct size and usage flags
- Position, velocity, color, attribute buffers allocated
- State buffer for alive/dead tracking
- Indirect draw argument buffer initialized

### 2.2 Implement Emit Compute Pass
**File**: `gpu_particles.py`
**Acceptance**:
- Compute shader bound with spawn parameters
- New particle attributes initialized correctly
- Atomic counter for alive count updated
- Respects spawn rate and burst configuration

### 2.3 Implement Update Compute Pass
**File**: `gpu_particles.py` (lines 452-472)
**Acceptance**:
- Compute shader binds all attribute buffers
- Delta time uniform set correctly
- Force uniforms (gravity, wind) passed
- Velocity integrated to position
- Age updated, lifetime checked
- Dead particles marked in state buffer

### 2.4 Implement Memory Barriers
**File**: `gpu_particles.py`
**Acceptance**:
- COMPUTE_TO_COMPUTE barrier between emit and update
- COMPUTE_TO_VERTEX barrier before rendering
- Correct barrier stages and access masks

### 2.5 Implement Dispatch Size Calculation
**File**: `gpu_particles.py`
**Acceptance**:
- Workgroup size constant (256)
- Dispatch count rounded up correctly
- Zero particles results in no dispatch

### 2.6 Implement Indirect Draw Setup
**File**: `gpu_particles.py`
**Acceptance**:
- DrawIndirectCommand populated from alive count
- Vertex count correct for particle quad/mesh
- Instance count matches alive particles
- First vertex/instance offsets correct

### 2.7 Implement CPU Fallback Detection
**File**: `gpu_particles.py`
**Acceptance**:
- Query RHI for compute shader support
- Fall back to CPU simulation if unsupported
- Unified API regardless of backend

### 2.8 Implement GPU-CPU Synchronization
**File**: `gpu_particles.py`
**Acceptance**:
- Optional CPU readback for debugging
- Fence/timeline semaphore for sync
- Double buffer to hide latency

### 2.9 Implement Compute Shader Bindings
**File**: `gpu_particles.py`
**Acceptance**:
- Storage buffer slots match shader layout
- Push constants for per-dispatch data
- Uniform buffer for per-frame constants

### 2.10 Implement Stream Compaction (Optional)
**File**: `gpu_particles.py`
**Acceptance**:
- Prefix sum for dead particle removal
- Compacts alive particles to contiguous range
- Updates indirect draw count

### 2.11 Implement Depth Sort (Optional)
**File**: `gpu_particles.py`
**Acceptance**:
- Bitonic sort by camera depth
- Enables correct transparency blending
- Only when blend mode requires sorting

### 2.12 Write WGSL/GLSL Emit Shader
**Acceptance**:
- Initialize particle position from spawn shape
- Initialize velocity with spread
- Initialize color, size, lifetime
- Atomic increment alive count

### 2.13 Write WGSL/GLSL Update Shader
**Acceptance**:
- Read current position, velocity
- Apply forces (gravity, wind, turbulence)
- Integrate: velocity += acceleration * dt, position += velocity * dt
- Update age, check lifetime, mark dead

### 2.14 Write WGSL/GLSL Compact Shader
**Acceptance**:
- Parallel prefix sum
- Scatter alive particles to output
- Update indirect draw argument

### 2.15 Profile GPU vs CPU Performance
**Acceptance**:
- Benchmark at 10K, 100K, 1M particles
- Measure dispatch overhead
- Identify crossover point where GPU wins
