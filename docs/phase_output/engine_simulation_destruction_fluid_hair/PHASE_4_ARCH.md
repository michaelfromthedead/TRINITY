# PHASE 4 ARCHITECTURE: Cross-Cutting Enhancements

**Scope**: All 16 files, ~10,973 lines  
**Classification**: Enhancement Opportunities

---

## Overview

This phase addresses cross-cutting concerns identified across all three simulation subsystems:
1. Numerical robustness patterns
2. Configuration-driven design
3. Performance optimization opportunities
4. Testing coverage gaps
5. Integration patterns

---

## Cross-Cutting Patterns (Already Implemented)

### 1. Numerical Robustness

All systems implement consistent numerical safety patterns:

**Division Guard Pattern**:
```python
denominator = compute_value()
if abs(denominator) > epsilon:
    result = numerator / denominator
else:
    result = fallback_value  # Domain-appropriate default
```

**Files Using This Pattern**:
- `fracture_voronoi.py`: Triangle clipping (edge-on-plane)
- `sph.py`: Kernel evaluation (zero distance)
- `pbf.py`: Lagrange multiplier (zero gradient sum)
- `hair_constraints.py`: Rodrigues formula (collinear vectors)

**NaN/Inf Guard Pattern**:
```python
if not np.isfinite(value):
    value = safe_default
    log_warning("Numerical instability detected")
```

**Degenerate Geometry Pattern**:
```python
if area < degenerate_threshold:
    skip_triangle()  # Or use centroid approximation
```

### 2. Configuration-Driven Design

All tunable parameters externalized to `config.py`:

| Subsystem | Config File | Parameter Categories |
|-----------|-------------|---------------------|
| Destruction | `destruction/config.py` | Thresholds, pool sizes, LOD distances |
| Fluid | `fluid/config.py` | Kernel radii, iteration counts, timesteps |
| Hair | `hair/config.py` | Stiffness, friction, LOD thresholds |

**Benefits**:
- Runtime parameter adjustment
- Per-scene overrides
- No magic numbers in algorithms
- Clear documentation of tunable values

### 3. Performance Patterns

**Object Pooling** (Debris, Particles):
```python
class ObjectPool:
    def __init__(self, size):
        self._objects = [create_object() for _ in range(size)]
        self._free_list = list(range(size))
    
    def acquire(self):
        return self._objects[self._free_list.pop()]
    
    def release(self, index):
        self._free_list.append(index)
```

**Files**: `debris.py`, particle systems in fluid

**Spatial Hashing** (O(1) Neighbor Queries):
```python
def hash_position(position, cell_size):
    ix = int(position.x / cell_size)
    iy = int(position.y / cell_size)
    iz = int(position.z / cell_size)
    return hash((ix, iy, iz))
```

**Files**: `sph.py`, `hair_collision.py`

**LOD Systems** (Distance-Based Quality):

| Subsystem | LOD Levels | Hysteresis |
|-----------|------------|------------|
| Destruction/Debris | FULL, REDUCED, SIMPLE, PARTICLE | Yes |
| Hair | HIGH, MEDIUM, LOW, SHELL | Yes |

**Files**: `debris.py`, `hair_lod.py`

### 4. Integration Patterns

**Protocol-Based Interfaces**:
```python
class PhysicsBodyProtocol(Protocol):
    def get_position(self) -> Vec3: ...
    def get_velocity(self) -> Vec3: ...
    def apply_force(self, force: Vec3) -> None: ...
```

**Benefits**:
- No inheritance requirements
- External physics engines integrate without modification
- Easy mocking for tests

**Callback Extensibility**:
```python
class SimulationCallbacks:
    on_fracture_complete: Callable[[FractureResult], None]
    on_debris_created: Callable[[Debris], None]
    on_fluid_surface_extracted: Callable[[Mesh], None]
```

---

## Enhancement Opportunities

### 1. GPU Acceleration

**Current State**:
- `gpu_fluid.py` provides abstract interface with CPU fallback
- Hair/destruction are CPU-only

**Recommended Architecture**:
```
gpu/
  +-- compute_interface.py (Abstract dispatch)
  +-- vulkan_backend.py
  +-- metal_backend.py
  +-- wgpu_backend.py
  +-- cpu_fallback.py
```

**Candidate Algorithms for GPU**:

| Algorithm | System | Parallelism |
|-----------|--------|-------------|
| Spatial hash build | Fluid, Hair | Per-particle |
| Kernel evaluation | Fluid | Per-particle-pair |
| FTL constraint solve | Hair | Per-strand |
| Density field build | Hair | Per-particle |
| Voronoi clipping | Destruction | Per-triangle |

### 2. Multi-Threading

**Embarrassingly Parallel Operations**:
| Operation | Granularity | Estimated Speedup |
|-----------|-------------|-------------------|
| Hair strand update | Per-strand | Linear with cores |
| Fluid particle forces | Per-particle | Linear with cores |
| Triangle clipping | Per-triangle | Linear with cores |
| Spatial hash query | Per-particle | Linear with cores |

**Synchronization Points**:
- Spatial hash rebuild (barrier before queries)
- Constraint projection (may need atomic updates)
- Output aggregation (reduction)

### 3. SIMD Optimization

**Candidate Operations**:
| Operation | Vector Width | Files |
|-----------|--------------|-------|
| vec3 operations | 4x float | All |
| Kernel batched eval | 8x float (AVX) | `sph.py` |
| Triangle-plane tests | 4x float | `fracture_*.py` |

**NumPy Already Provides**:
- Automatic SIMD for array operations
- Potential for explicit vectorization via numba/jax

### 4. Memory Layout Optimization

**Current**: Likely Array-of-Structures (AoS)
```python
particles = [Particle(pos, vel, ...), ...]
```

**Recommended**: Structure-of-Arrays (SoA) for hot paths
```python
positions = np.array([...])  # Contiguous
velocities = np.array([...])  # Contiguous
```

**Benefits**:
- Better cache utilization
- SIMD-friendly
- GPU-transfer friendly

---

## Testing Coverage Gaps

### Identified Test Categories Needed

**1. Edge Case Tests**:
| Subsystem | Edge Cases |
|-----------|------------|
| Destruction | Zero-area triangles, edge-on-plane, collinear vertices |
| Fluid | Zero particles, single particle, extreme velocities |
| Hair | Zero-length segments, collinear edges, extreme head motion |

**2. Numerical Stability Tests**:
- Input values near machine epsilon
- Very large input values
- Accumulated error over many timesteps

**3. Performance Regression Tests**:
- Baseline timing for standard scenarios
- Memory allocation tracking
- Scaling tests (10K, 100K, 1M elements)

**4. Integration Tests**:
- Cross-system interaction (debris + fluid)
- Multiple LOD transitions
- Callback sequence verification

---

## Dependency Consolidation

### Shared Dependencies

| Dependency | Usage | Version Constraint |
|------------|-------|-------------------|
| numpy | Vector math, arrays | >=1.20 |
| heapq | Dijkstra priority queue | stdlib |
| math | Trig functions | stdlib |
| typing | Protocols, type hints | stdlib |

### Potential Shared Modules

**Candidate for extraction**:
```
common/
  +-- vec3.py (Consistent vector operations)
  +-- spatial_hash.py (Shared implementation)
  +-- object_pool.py (Generic pooling)
  +-- config_base.py (Configuration pattern)
  +-- numerical.py (Epsilon, guards, clamp)
```

**Benefits**:
- Single source of truth for common patterns
- Consistent behavior across systems
- Easier testing of shared code

---

## Architecture Decisions Preserved

### What NOT to Change

1. **Separate subsystem codebases**: Destruction, fluid, hair remain independent
2. **Configuration externalization**: All params in config.py files
3. **Protocol-based integration**: No shared base classes
4. **Numerical guard patterns**: Division/NaN/degenerate checks

### What Could Be Unified

1. **Spatial hashing**: Single implementation, parameterized for each use
2. **Object pooling**: Generic pool class
3. **LOD framework**: Shared hysteresis logic
4. **Vector operations**: Consistent Vec3 type alias
