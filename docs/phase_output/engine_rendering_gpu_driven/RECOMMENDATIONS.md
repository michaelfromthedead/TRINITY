# RECOMMENDATIONS: engine/rendering/gpu_driven

## Rust Bridge Requirements

### High Priority

| Requirement | Description | Effort |
|-------------|-------------|--------|
| **Culling Compute Shaders** | Port FrustumCuller and OcclusionCuller to WGSL compute shaders. Input: instance bounds buffer. Output: visibility bitmask buffer. | 2-3 days |
| **Instance Buffer Upload** | Create Rust function to upload Python InstanceData arrays to wgpu buffers. Use staging buffer for efficiency. | 1 day |
| **Indirect Command Buffer** | Create wgpu buffer for DrawIndexedIndirectArgs. Implement fill from Python DrawCommand list. | 1 day |
| **Visibility Buffer Render Pass** | Implement wgpu render pass that writes triangle+instance IDs to visibility buffer texture. | 2 days |

### Medium Priority

| Requirement | Description | Effort |
|-------------|-------------|--------|
| **Bindless Descriptor Integration** | Connect Python BindlessResourceSystem to Rust descriptor heap. Share handle allocation state. | 2 days |
| **Meshlet Data Upload** | Upload MeshletMesh data (vertices, indices, meshlet metadata) to GPU buffers. | 1 day |
| **Material Tile Dispatch** | Implement compute shader to classify materials per 8x8 tile. Generate per-material dispatch counts. | 1-2 days |
| **Deferred Texturing Pass** | Implement material-sorted shading pass using visibility buffer lookups. | 2 days |

### Low Priority

| Requirement | Description | Effort |
|-------------|-------------|--------|
| **Mesh Shader Support** | Detect mesh shader capability. Implement meshlet amplification + mesh shaders as alternative path. | 3-5 days |
| **GPU Scene Graph** | Move instance transforms to GPU. Implement GPU-side transform updates. | 3 days |
| **Two-Phase Occlusion** | Implement occlusion reprojection from previous frame. Add second-phase culling for newly visible. | 2 days |

## Integration Strategy

### Phase 1: Basic GPU Culling (Week 1)

1. Create Rust buffer types for instance bounds and visibility results
2. Port FrustumCuller to WGSL compute shader
3. Implement CPU-to-GPU instance data upload
4. Wire culling output to indirect draw generation
5. Verify correctness against Python reference

### Phase 2: Visibility Buffer Rendering (Week 2)

1. Implement visibility buffer texture creation
2. Create vertex/fragment shaders for triangle ID output
3. Wire indirect draw execution
4. Implement visibility buffer read-back for verification
5. Add material tile classification compute pass

### Phase 3: Bindless Materials (Week 3)

1. Integrate bindless texture manager with Rust descriptor heap
2. Implement material resource table upload
3. Create deferred texturing shader
4. Wire material-sorted dispatch
5. Full pipeline integration test

### Phase 4: Advanced Features (Week 4+)

1. Add HZB occlusion culling compute shader
2. Implement meshlet culling (if mesh shaders unavailable)
3. Add mesh shader path (if hardware supports)
4. Performance optimization pass

## Testing Strategy

### Unit Tests

| Test | Purpose |
|------|---------|
| Frustum extraction test | Verify plane normals and distances match known VP matrices |
| Sphere-frustum test | Verify inside/outside/intersect classification |
| HZB build test | Verify mip pyramid max reduction |
| Edge function test | Verify rasterization coverage |
| Handle generation test | Verify stale handle detection |
| Draw compaction test | Verify merge correctness |

### Integration Tests

| Test | Purpose |
|------|---------|
| CPU-GPU parity test | Compare Python culling results to GPU shader results |
| Visibility buffer test | Verify triangle IDs match expected for known geometry |
| Material dispatch test | Verify tile classification produces correct material counts |
| Full frame test | Render known scene, compare against reference image |

### Performance Tests

| Test | Target |
|------|--------|
| Culling throughput | 1M instances culled in <1ms on GPU |
| Indirect draw setup | 100K draw commands generated in <0.5ms |
| Visibility buffer fill | 1080p visibility buffer at >60fps |
| Material dispatch | 256 materials dispatched in <0.5ms |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| WGSL compute shader bugs | Medium | High | Extensive CPU-GPU parity testing |
| Descriptor heap exhaustion | Low | High | Implement descriptor heap growth or fixed maximum |
| Mesh shader unavailability | Medium | Medium | Maintain compute shader fallback path |
| Performance regression | Medium | Medium | Benchmark at each integration phase |
| Memory bandwidth limits | Medium | Medium | Implement buffer streaming, reduce per-instance data |
| Driver bugs with indirect | Low | High | Test on multiple GPU vendors, implement workarounds |

### Risk Details

**WGSL Compute Shader Bugs**: The Python algorithms are well-tested but WGSL translation may introduce subtle bugs (floating point precision, index overflow). Mitigation: run Python and GPU paths in parallel during development, compare results.

**Descriptor Heap Exhaustion**: Bindless architecture requires large descriptor heaps. If heap is undersized, rendering fails. Mitigation: track descriptor usage, implement heap growth or fail-safe with warning.

**Mesh Shader Unavailability**: Mesh shaders provide optimal meshlet rendering but require modern GPUs. Mitigation: maintain compute shader + vertex shader fallback that processes meshlets without hardware acceleration.

**Performance Regression**: GPU-driven should be faster than CPU-driven, but poor implementation can be slower. Mitigation: profile at each phase, compare against baseline.

## Success Criteria

1. GPU culling matches Python reference within floating-point tolerance
2. Visibility buffer produces correct triangle IDs for test scenes
3. Full GPU-driven pipeline renders at >60fps for 100K instances
4. No descriptor leaks after 10 minutes of continuous operation
5. Memory usage stable (no unbounded growth)
6. All unit and integration tests pass
