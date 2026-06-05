# RECOMMENDATIONS: trinity/decorators (Part 1)

## Rust Bridge Requirements

### High Priority

| Requirement | Rationale | Affected Components |
|-------------|-----------|---------------------|
| GPU layout serialization | wgpu requires exact struct alignment; Python layouts must export to Rust | gpu.py, bridge.rs |
| Component ID synchronization | ECS queries must resolve same IDs in Python and Rust | ecs_core.py, component_store.rs |
| System ordering export | Rust scheduler needs tier and chain information | scheduling.py, scheduler module |
| Validation rule export | Rust should enforce same constraints as Python | All validators |

### Medium Priority

| Requirement | Rationale | Affected Components |
|-------------|-----------|---------------------|
| Pool configuration export | Rust allocator needs size/alignment from Python decorators | memory.py, allocator |
| Serialization schema sync | Binary protocols must match across boundary | data_flow.py, bridge.rs |
| Profile instrumentation bridge | Unified timing across Python/Rust boundary | dev.py, tracing |

### Low Priority

| Requirement | Rationale | Affected Components |
|-------------|-----------|---------------------|
| Render pass metadata | Non-critical; can be Rust-only initially | rendering.py |
| Audio DSP configuration | Lower priority for MVP | audio_extended.py |
| Modding API export | Future extensibility feature | modding.py |

---

## Integration Strategy

### Phase 1: Metadata Export Protocol

1. Define JSON/MessagePack schema for decorator metadata
2. Implement `to_bridge_format()` on key decorators
3. Add `bridge.rs` endpoint to receive metadata
4. Test roundtrip: Python decorator -> JSON -> Rust struct

### Phase 2: ID Synchronization

1. Move component ID assignment to Rust (single source of truth)
2. Python queries Rust for IDs via bridge
3. Update ComponentMeta to defer ID assignment
4. Test: Same component decorated in Python resolves to same ID in Rust

### Phase 3: Layout Verification

1. Generate test cases from Python GPU layouts
2. Compare against Rust wgpu struct definitions
3. Implement automated layout comparison in CI
4. Document alignment rules for developers

### Phase 4: Runtime Reflection

1. Expose Rust query interface for decorator metadata
2. Enable Rust systems to introspect Python-decorated entities
3. Support bidirectional metadata updates
4. Document reflection API

---

## Testing Strategy

### Unit Tests (Python)

| Test Category | Coverage Target | Location |
|---------------|-----------------|----------|
| Ops composition | 100% of 7 ops | tests/decorators/test_ops.py |
| GPU layout math | All type combinations | tests/decorators/test_gpu.py |
| Thread safety | Concurrent decorator application | tests/decorators/test_registry.py |
| Validation errors | All parameter constraints | tests/decorators/test_validation.py |

### Integration Tests (Python + Rust)

| Test Category | Coverage Target | Location |
|---------------|-----------------|----------|
| Metadata roundtrip | All bridge-exported decorators | tests/integration/test_bridge_decorators.py |
| ID synchronization | Component/system IDs | tests/integration/test_component_ids.py |
| Layout verification | GPU struct alignment | tests/integration/test_gpu_layouts.py |

### Property Tests

| Property | Generators | Tool |
|----------|------------|------|
| GPU layout determinism | Random field orderings | hypothesis |
| Tier ordering correctness | Random decorator application order | hypothesis |
| Validation completeness | Fuzz parameter combinations | hypothesis |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Layout mismatch Python/Rust | Medium | High | Automated layout verification in CI |
| ID desynchronization | Medium | High | Single source of truth in Rust |
| Thread safety under load | Low | Medium | Stress tests with concurrent imports |
| Tier cycle detection | Low | Medium | Static analysis of tier dependencies |
| Validation bypass | Low | High | Fuzz testing of all validators |
| Breaking changes to Ops | Low | High | Versioned Op schema with migration |

---

## Action Items

### Immediate (GRANDPHASE2 Blockers)

1. [ ] Define bridge metadata JSON schema
2. [ ] Implement `to_bridge_format()` on `@component`, `@gpu_buffer`
3. [ ] Add `bridge.rs` metadata ingestion endpoint
4. [ ] Create layout verification test suite

### Short-Term (Next Sprint)

5. [ ] Move component ID assignment to Rust
6. [ ] Implement tier export for scheduler
7. [ ] Add validation rule export
8. [ ] Document bridge protocol

### Medium-Term (Quarter)

9. [ ] Complete all High Priority requirements
10. [ ] Implement runtime reflection API
11. [ ] Add Medium Priority requirements
12. [ ] Performance benchmark bridge overhead

### Long-Term (Future)

13. [ ] Modding API export
14. [ ] Full bidirectional metadata sync
15. [ ] JIT-compiled validation rules
