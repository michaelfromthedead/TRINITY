# engine/rendering/framegraph — Recommendations

## Rust Bridge Requirements

### High Priority

| Requirement | Description | Effort |
|-------------|-------------|--------|
| **RenderContext Protocol** | Define typed `RenderContext` trait replacing `context: Any`. Must expose `execute_barriers()`, `allocate_resource()`, `begin_pass()`, `end_pass()`. | Medium |
| **wgpu Barrier Mapping** | Map `ResourceState`, `PipelineStage`, `AccessFlags` to wgpu equivalents. wgpu uses implicit barriers; may need explicit transition commands. | Medium |
| **GPU Memory Allocator** | Implement allocator using aliasing groups from `compute_aliasing()`. Use offset-based suballocation within shared memory blocks. | High |
| **Command Buffer Generation** | Generate wgpu command buffers from compiled frame graph. Map `GraphicsPass` to render pass, `ComputePass` to compute pass, etc. | High |

### Medium Priority

| Requirement | Description | Effort |
|-------------|-------------|--------|
| **Rust Compile** | Port `compile()` to Rust for sub-millisecond frame setup. Python calls Rust via PyO3, receives compiled graph. | High |
| **Queue Mapping** | Map `QueueType` to wgpu queues. wgpu has single queue model; may need to flatten or use compute-graphics interleaving. | Medium |
| **Timestamp Queries** | Integrate GPU timestamp queries for validating overlap benefit estimates. | Low |
| **Debug Layer** | Implement barrier validation layer that checks for missing transitions. | Medium |

### Low Priority

| Requirement | Description | Effort |
|-------------|-------------|--------|
| **VRS Integration** | Add variable rate shading pass configuration. | Low |
| **Mesh Shaders** | Add `MeshPass` type for mesh shader pipelines. | Medium |
| **Bindless Resources** | Integrate bindless descriptor model for texture/buffer access. | High |

## Integration Strategy

### Phase 1: Minimal Viable Integration
1. Define `RenderContext` trait in Rust with stub implementations
2. Implement PyO3 binding to pass compiled JSON to Rust
3. Verify round-trip: Python build -> JSON -> Rust deserialize

### Phase 2: Barrier Execution
1. Implement `execute_barriers()` using wgpu transition commands
2. Map Python `ResourceState` enum to wgpu equivalents
3. Test with simple 2-pass frame graph (clear -> render)

### Phase 3: Resource Allocation
1. Implement aliasing-aware allocator using `alias_group` IDs
2. Allocate shared memory blocks per alias group
3. Suballocate transient resources within blocks

### Phase 4: Full Execution
1. Implement `begin_pass()` / `end_pass()` for all pass types
2. Generate wgpu command buffers from scheduled passes
3. Execute with proper queue synchronization

### Phase 5: Performance
1. Port `compile()` to Rust
2. Integrate timestamp queries
3. Profile and optimize hot paths

## Testing Strategy

### Unit Tests
| Test | Description |
|------|-------------|
| Dependency Graph | Verify producer/consumer tracking builds correct DAG |
| Topological Sort | Verify passes ordered correctly |
| Pass Culling | Verify unused passes removed |
| Resource Aliasing | Verify non-overlapping lifetimes share groups |
| Barrier Generation | Verify correct transitions generated |
| UAV Hazards | Verify read-after-write barriers inserted |
| Cross-Queue Sync | Verify sync points generated for cross-queue deps |

### Integration Tests
| Test | Description |
|------|-------------|
| JSON Round-Trip | Python -> JSON -> Rust -> execute |
| Simple Frame | Clear -> Render -> Present |
| Async Compute | Graphics + parallel compute |
| History Resources | Double-buffered temporal effects |
| External Resources | Imported backbuffer handling |

### Validation Tests
| Test | Description |
|------|-------------|
| Barrier Correctness | wgpu validation layer catches missing barriers |
| Memory Aliasing | No visual artifacts from aliasing bugs |
| Queue Sync | No GPU hangs from missing synchronization |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| wgpu barrier model mismatch | Medium | High | Study wgpu synchronization model; may need explicit buffer/texture transitions |
| Aliasing bugs | Medium | High | Conservative aliasing initially; enable progressively with validation |
| Queue synchronization | Low | Critical | Start with single-queue execution; add async compute later |
| Performance regression | Medium | Medium | Profile continuously; keep Python hot path minimal |
| PyO3 overhead | Low | Medium | Batch calls; minimize Python/Rust boundary crossings |

## Dependencies

### External
- wgpu (Rust GPU abstraction)
- pyo3 (Python-Rust interop)
- serde_json (JSON deserialization)

### Internal
- RHI backend (defines `RenderContext` trait)
- Memory management (GPU allocator)
- Shader system (pipeline state creation)

## Success Criteria

1. **Functional**: Frame graph compiles and executes on GPU hardware
2. **Correct**: No validation layer errors; visual output matches reference
3. **Performant**: Frame setup < 1ms; no CPU-side bottlenecks
4. **Robust**: Handles edge cases (empty passes, missing resources, etc.)
