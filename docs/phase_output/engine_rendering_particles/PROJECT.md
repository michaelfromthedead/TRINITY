# PROJECT: engine/rendering/particles

## Classification
**PARTIAL IMPLEMENTATION** - CPU particle system is production-ready. GPU compute is architecture-only with stubbed execution.

## Scope

### In Scope
- Complete CPU particle simulation with physics modules
- GPU particle system architecture completion (RHI integration)
- VFX graph node-based authoring system
- Trail rendering with Catmull-Rom spline interpolation
- Deferred decal system with G-Buffer support
- Particle pool management with O(1) allocation/deallocation
- Budget management system (ambient/gameplay/critical categories)

### Out of Scope
- Particle system asset pipeline (separate tooling concern)
- Editor integration (UI layer concern)
- Platform-specific GPU backend implementations

## File Inventory
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `particle_modules.py` | 1,016 | REAL | 17 modules: spawn, forces, collision, rendering |
| `vfx_graph.py` | 995 | REAL | Node-based graph compilation to emitter |
| `decal_system.py` | 946 | REAL | Deferred decals, G-Buffer, atlas packing |
| `particle_system.py` | 850 | REAL | Full particle emitter, pool, budget management |
| `gpu_particles.py` | 844 | PARTIAL | Architecture complete, GPU dispatch stubbed |
| `trail_renderer.py` | 815 | REAL | Full mesh generation, Catmull-Rom, caps |
| `constants.py` | 160 | REAL | Centralized config constants |

**Total**: 5,982 lines (main files) + 160 (constants) = 6,142 lines

## Goals
1. Complete GPU particle compute dispatch integration with RHI
2. Validate CPU particle physics modules against edge cases
3. Ensure VFX graph compilation produces correct emitter configurations
4. Verify trail renderer mesh generation correctness
5. Confirm decal system G-Buffer integration

## Constraints
- GPU dispatch requires RHI layer to be operational
- VectorFieldModule data loading remains stubbed (requires asset system)
- Must maintain O(1) pool operations under all workloads
- Budget system must respect category limits without starvation

## Acceptance Criteria

### Phase 1: CPU System Validation
- [ ] All 17 particle modules pass unit tests
- [ ] Pool allocation/deallocation maintains O(1) complexity
- [ ] Spatial hashing collision detection correct for 3x3x3 neighborhood
- [ ] Sphere sampling uses volume-corrected distribution

### Phase 2: GPU Integration
- [ ] GPU dispatch calls replaced with actual RHI bindings
- [ ] Compute shader bind/dispatch/barrier sequence implemented
- [ ] GPU particle buffer synchronization verified
- [ ] Fallback to CPU simulation when GPU unavailable

### Phase 3: Rendering Pipeline
- [ ] Trail renderer Catmull-Rom tangents produce smooth curves
- [ ] Ribbon mesh caps render correctly at trail endpoints
- [ ] Decal atlas shelf packing maximizes utilization
- [ ] G-Buffer decal projection correct for arbitrary box orientations

### Phase 4: VFX Graph
- [ ] Graph compilation produces valid ParticleEmitter
- [ ] Module categorization (spawn/update/render) correct
- [ ] Dirty flag prevents unnecessary recompilation
- [ ] All VFX node types convert to particle modules
