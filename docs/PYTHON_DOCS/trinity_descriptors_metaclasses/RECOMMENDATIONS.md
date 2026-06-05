# RECOMMENDATIONS: Trinity Descriptors, Metaclasses, and Tools

## Rust Bridge Requirements

### High Priority

| Requirement | Current State | Target State | Rationale |
|-------------|---------------|--------------|-----------|
| **Expand RustStorageDescriptor** | Basic types (f32, i32, u8, string) | All descriptor behaviors | Full SoA performance for all field types |
| **NetworkedDescriptor Rust** | Python only | Interpolation/prediction in Rust | Tick-critical path performance |
| **ComponentMeta Pool Management** | Python pools | Rust-side pools | Memory efficiency in hot path |
| **SystemMeta Parallel Execution** | `get_parallel_groups()` computes groups | Wire to Rust thread pool | Actual parallel system execution |

### Medium Priority

| Requirement | Current State | Target State | Rationale |
|-------------|---------------|--------------|-----------|
| **TrackedDescriptor Rust** | Python dirty flags | Rust-side dirty tracking | Avoid Python/Rust boundary for change detection |
| **ValidationDescriptor Rust** | Python validation | Rust-side validation | Hot-path validation without Python call |
| **ComponentMeta Budgets** | Python `_instance_count` | Rust instance limits | Enforce budgets without Python overhead |
| **CachedDescriptor Rust** | Python TTL cache | Rust-side caching | Cache hits without Python call |

### Low Priority

| Requirement | Current State | Target State | Rationale |
|-------------|---------------|--------------|-----------|
| **AtomicDescriptor Rust** | Python threading.Lock | Rust atomics | True lock-free operations |
| **CompressedDescriptor Rust** | Python zlib/lz4 | Rust compression | Avoid Python for compression |
| **ObservableDescriptor Rust** | Python callbacks | Rust observer pattern | Event dispatch without Python |

---

## Integration Strategy

### Phase 1: Core Rust Bridge (Week 1-2)

1. **Extend `_omega` API** to support:
   - `component_get_tracked(entity_id, field_name) -> (value, dirty)`
   - `component_set_tracked(entity_id, field_name, value) -> old_dirty`
   - `component_validate(entity_id, field_name, value) -> Result`

2. **Update RustStorageDescriptor** to detect field annotations and route appropriately

3. **Add Rust-side dirty flag storage** in component store

### Phase 2: Network Descriptors (Week 3-4)

1. **Move interpolation math to Rust**:
   - Linear interpolation
   - Hermite interpolation with velocity

2. **Move prediction logic to Rust**:
   - Client-side prediction state machine
   - Rollback buffer management

3. **Wire NetworkedDescriptor to Rust** for tick-critical paths

### Phase 3: System Parallelization (Week 5-6)

1. **Export `get_parallel_groups()` results to Rust**

2. **Implement Rust thread pool** for system execution

3. **Wire SystemMeta phase execution** to Rust scheduler

### Phase 4: Pooling and Budgets (Week 7-8)

1. **Move ComponentMeta pools to Rust**

2. **Implement Rust-side instance counting** for budgets

3. **Wire Python pool API** to Rust pool manager

---

## Testing Strategy

### Unit Tests

| Test Area | Coverage Target | Approach |
|-----------|-----------------|----------|
| Descriptor Protocol | 100% | Test `__get__/__set__/__delete__` for each descriptor |
| Composition Chains | 100% | Test all valid/invalid chain combinations |
| Metaclass Registration | 100% | Test type registration, ID generation |
| Rust Fallback | 100% | Test both Rust-available and Rust-unavailable paths |

### Integration Tests

| Test Area | Coverage Target | Approach |
|-----------|-----------------|----------|
| Descriptor + Rust Storage | 100% | End-to-end field operations through Rust |
| Metaclass + Rust Registry | 100% | Type registration round-trip |
| Network Descriptor + Tick | 80% | Interpolation/prediction across ticks |
| System Parallel Execution | 80% | Verify parallel groups execute correctly |

### Performance Tests

| Test Area | Target | Approach |
|-----------|--------|----------|
| RustStorageDescriptor | <1us per access | Microbenchmark get/set |
| NetworkedDescriptor | <10us per tick | Microbenchmark interpolation |
| ComponentMeta pool | <100ns acquire | Microbenchmark pool ops |
| System execution | 10K entities/ms | Benchmark system tick |

### Regression Tests

1. **Step trace stability** — Ensure introspection output remains consistent
2. **Lint hook compatibility** — Validate import-time hooks don't break
3. **Thread safety** — Concurrent access to registries under load

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Rust API mismatch** | Medium | High | Define Rust API contract before implementation |
| **Performance regression** | Low | High | Benchmark before/after each change |
| **Python/Rust state desync** | Medium | Medium | Single source of truth in Rust, Python reads only |
| **Thread safety in bridge** | Medium | High | Use Rust ownership model, minimize shared state |
| **Descriptor chain complexity** | Low | Medium | Comprehensive composition tests |
| **Metaclass registry corruption** | Low | High | Clear isolation in tests, thread-safe registries |
| **Network descriptor timing** | Medium | Medium | Fuzz test with variable tick rates |
| **Pool exhaustion** | Low | Medium | Configurable pool sizes, soft/hard limits |

### Risk Mitigation Actions

1. **Define Rust API contract first** — Document expected `_omega` API before coding
2. **Benchmark suite** — Establish performance baselines before GRANDPHASE2 work
3. **State ownership documentation** — Clearly document which side owns each piece of state
4. **Integration test coverage** — 80%+ integration test coverage before Rust changes
5. **Incremental rollout** — Feature flags for Rust-side behaviors
