# CLARIFICATION: engine/rendering/particles

## Philosophical Framing

The particle system represents a **layered simulation architecture** where CPU-side physics provides the ground truth, and GPU compute acts as an acceleration layer. This design philosophy ensures:

1. **Correctness First**: CPU implementation is complete and verifiable before GPU optimization
2. **Graceful Degradation**: System functions without GPU compute, falling back to CPU
3. **Modular Composition**: 17 discrete modules can be mixed arbitrarily via VFX graph

## Design Rationale

### Why CPU + GPU Dual Path?

The stubbed GPU dispatch is not a deficiency but a deliberate architectural boundary. The CPU path:
- Enables debugging without GPU shader complexity
- Provides reference implementation for GPU shader validation
- Supports platforms without compute shader support
- Allows deterministic replay for debugging

### Why O(1) Pool with Reverse Lookup?

The `_particle_to_index` dictionary trades memory for constant-time deallocation:
```python
self._particle_to_index: dict[int, int] = {
    id(p): i for i, p in enumerate(self._particles)
}
```
This is critical for high particle counts where linear search would be prohibitive.

### Why Spatial Hashing for Collision?

3D spatial hashing with 3x3x3 neighborhood queries provides:
- O(1) average-case nearby particle lookup
- Bounded memory usage (cells only for occupied regions)
- Cache-friendly access patterns for localized particles

### Why Volume-Corrected Sphere Sampling?

Naive sphere sampling clusters particles near the center. The cube-root correction:
```python
r = self._radius * (random.random() ** (1 / 3))
```
Ensures uniform volumetric distribution, critical for physically plausible effects.

### Why Catmull-Rom for Trails?

Catmull-Rom splines provide C1 continuity (smooth tangents) with local control - modifying one control point only affects nearby curve segments. This enables:
- Smooth ribbon geometry from discrete position samples
- Efficient incremental updates as new points are added
- Natural-looking trail deformation

### Why Shelf Packing for Decal Atlas?

Shelf packing (also known as Next-Fit Decreasing Height) provides:
- O(n) packing time vs O(n^2) for optimal algorithms
- Good space utilization (typically 80-90%)
- Simple online insertion (no need to repack entire atlas)

## Module Architecture

### Stage Pipeline
```
SPAWN -> UPDATE -> RENDER
```

- **SPAWN**: Shape emission, burst/rate timing
- **UPDATE**: Forces, collision, attribute interpolation
- **RENDER**: Billboard alignment, mesh instancing

### Module Composition Philosophy

Each module is a pure function of (particle state, delta time). Modules do not communicate directly - they apply transformations to particle attributes. This enables:
- Arbitrary module ordering within stages
- Easy addition of new modules without modifying existing code
- Parallelizable update (each particle independent)

## VFX Graph Compilation

The VFX graph is a **metaprogramming layer** that produces ParticleEmitter configurations. It does not interpret the graph at runtime - compilation happens once, producing a static module list.

This design:
- Eliminates graph traversal overhead during simulation
- Enables node-level validation before emission
- Supports caching compiled emitters

## Decal System Philosophy

Deferred decals operate in screen space, projecting onto the G-Buffer. This approach:
- Decouples decal count from scene geometry complexity
- Enables proper depth testing without Z-fighting
- Supports normal map and roughness modification of underlying surfaces
