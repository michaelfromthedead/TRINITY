# RECOMMENDATIONS: engine/rendering/particles

## Rust Bridge Requirements

### High Priority

| Requirement | Rationale | Complexity |
|-------------|-----------|------------|
| ParticleBuffer SoA Struct | GPU-friendly memory layout for particle attributes (position, velocity, color, size, life) | Medium |
| Pool Manager Bridge | Expose allocation/deallocation to Python with O(1) guarantees | Low |
| Emitter Update Loop | Rust-side particle simulation for CPU path performance | High |
| Trail Vertex Buffer | Direct vertex buffer generation without Python intermediary | Medium |

### Medium Priority

| Requirement | Rationale | Complexity |
|-------------|-----------|------------|
| VFX Graph Evaluator | Compiled graph evaluation in Rust for complex effects | High |
| Decal Batch Builder | Aggregate decal instances for efficient GPU submission | Medium |
| Spatial Hash Grid | Native spatial hash for collision queries | Low |
| Force Field Sampling | Vectorized force module evaluation | Medium |

### Low Priority

| Requirement | Rationale | Complexity |
|-------------|-----------|------------|
| Particle Sorting | Bitonic or radix sort for transparency ordering | Medium |
| LOD Calculator | Distance-based particle count scaling | Low |
| Compaction Kernel | GPU-side or Rust-side dead particle removal | Medium |

## Integration Strategy

### Phase 1: Buffer Bridging (Week 1-2)

1. Define Rust `ParticleSoA` struct matching GPU expectations
2. Create PyO3 bindings for buffer allocation/mapping
3. Bridge `ParticlePool` to use Rust backing store
4. Validate CPU simulation continues working

### Phase 2: Simulation Bridging (Week 2-3)

1. Port force module logic to Rust (gravity, wind, collision)
2. Maintain Python API for module configuration
3. Implement batch particle update in Rust
4. Add spatial hash grid in Rust for collision

### Phase 3: Render Bridging (Week 3-4)

1. Bridge trail renderer vertex generation to Rust
2. Bridge decal instance batching to Rust
3. Connect to RHI for GPU buffer uploads
4. Implement GPU indirect draw path

### Phase 4: GPU Compute (Week 4-6)

1. Implement RHI compute pipeline abstraction
2. Write WGSL particle update shader
3. Write WGSL particle render shader
4. Add GPU-side sorting and compaction

## Testing Strategy

### Unit Tests

| Component | Test Focus |
|-----------|------------|
| ParticlePool | Alloc/dealloc correctness, O(1) guarantees, compaction |
| SpawnModules | Shape sampling distribution validation |
| ForceModules | Physics integration accuracy |
| SpatialHash | Neighbor query correctness, edge cases |
| TrailRenderer | Vertex generation, UV mapping, caps |
| DecalSystem | Atlas packing, projection math |
| VFXGraph | Compilation output, module ordering |

### Integration Tests

| Scenario | Validation |
|----------|------------|
| Emitter Lifecycle | Spawn, update, render, death, pool return |
| Multi-Emitter | Budget enforcement across categories |
| Trail + Particle | Combined effect rendering |
| Decal Spawning | Decal creation from particle death |
| VFX Playback | Graph-compiled effect matches expected |

### Performance Tests

| Benchmark | Target |
|-----------|--------|
| 10K particles CPU update | < 2ms |
| 100K particles GPU update | < 1ms |
| Trail mesh generation (1K points) | < 0.5ms |
| Decal atlas packing (100 decals) | < 0.1ms |
| VFX graph compilation | < 10ms |

### Visual Tests

| Test | Method |
|------|--------|
| Sphere spawn distribution | Screenshot comparison |
| Gravity simulation | Physics correctness |
| Trail smoothness | Catmull-Rom visual verification |
| Decal projection | G-Buffer accuracy |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU compute shader bugs | Medium | High | CPU reference path for validation |
| Memory fragmentation (pool) | Low | Medium | Periodic compaction, pool sizing |
| VFX graph perf regression | Low | Medium | Compilation caching, profiling |
| RHI integration delays | Medium | High | CPU path remains production-ready |
| Cross-platform compute issues | Medium | Medium | WGSL abstraction, feature detection |
| Trail vertex count explosion | Low | Medium | Point budget, distance culling |

### Critical Dependencies

| Dependency | Status | Risk |
|------------|--------|------|
| RHI Core | In Progress | Blocks GPU compute |
| G-Buffer System | Complete | Low risk |
| Material System | Complete | Low risk |
| Asset Pipeline | Complete | Low risk |

### Recommended Priority Order

1. **Rust ParticleBuffer bridge** - Foundation for all other work
2. **RHI compute pipeline** - Unblocks GPU particles
3. **Force module Rust port** - CPU performance win
4. **Trail renderer Rust bridge** - Reduce Python overhead
5. **GPU particle shaders** - Full GPU path
6. **VectorField loading** - Complete feature set
