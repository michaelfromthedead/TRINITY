# ENGINE_WORLD - Phase 4 Architecture: Rust Bridge Preparation

## Architecture Decisions

### ADR-W4-001: Bridge Candidate Selection

**Context**: Phase 3 benchmarks identify slow operations. Not all slow operations are bridge candidates; some are I/O bound, some are already fast enough.

**Decision**: Bridge candidates must meet ALL criteria:
1. CPU-bound (not I/O or memory-bound)
2. Called frequently (hot path)
3. Self-contained (minimal Python object interaction)
4. >10x speedup expected from Rust

**Candidates Identified** (pending Phase 3 benchmarks):

| Component | Operation | Reason |
|-----------|-----------|--------|
| PCG | Noise sampling | Pure math, no GIL interaction |
| HLOD | QEM edge collapse | Tight loop, heap operations |
| Queries | A* pathfinding | Heap operations, tight loop |
| Foliage | Frustum culling | Vectorizable math |

**Not Candidates**:

| Component | Operation | Reason |
|-----------|-----------|--------|
| Partition | Cell loading | I/O bound |
| Terrain | Height query | Already fast (single interpolation) |
| Environment | Weather transition | Low frequency |

### ADR-W4-005: Testing Strategy

**Context**: Bridge functions must produce identical results to Python implementations.

**Decision**: Property-based testing with same inputs:
1. Generate random inputs
2. Call Python implementation
3. Call Rust implementation
4. Compare results (with floating-point tolerance)

**Tools**: `hypothesis` (Python), `quickcheck` (Rust)

### ADR-W4-006: Performance Verification

**Context**: Bridge overhead may negate Rust speedup for small inputs.

**Decision**: Establish crossover points:
- For each bridged function, find N where Rust becomes faster
- Document minimum batch size for bridge to be beneficial
- Python implementation remains fallback for small inputs

## Python Wrapper Pattern

```python
# engine/world/bridge.py

try:
    from ._world_bridge import perlin_noise_batch as _rust_perlin_batch
    HAS_RUST_BRIDGE = True
except ImportError:
    HAS_RUST_BRIDGE = False

def perlin_noise_batch(positions, seed):
    """Sample Perlin noise at multiple positions.
    
    Uses Rust implementation if available, falls back to Python.
    """
    if HAS_RUST_BRIDGE and len(positions) >= RUST_CROSSOVER_THRESHOLD:
        return _rust_perlin_batch(positions, seed)
    else:
        # Python fallback
        gen = PerlinNoise(seed)
        return [gen.sample(*pos) for pos in positions]
```

## Type Stub Generation

```python
# engine/world/_bridge.pyi

from typing import List, Tuple

def perlin_noise_batch(
    positions: List[Tuple[float, float]],
    seed: int
) -> List[float]: ...

def simplify_mesh_qem(
    vertices: List[Tuple[float, float, float]],
    indices: List[int],
    target_ratio: float
) -> Tuple[List[Tuple[float, float, float]], List[int]]: ...

def astar_pathfind(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    walkable: List[List[bool]],
    cell_size: float
) -> List[Tuple[float, float]]: ...

def frustum_cull_aabbs(
    frustum_planes: List[Tuple[float, float, float, float]],
    aabbs: List[Tuple[float, float, float, float, float, float]]
) -> List[bool]: ...
```
