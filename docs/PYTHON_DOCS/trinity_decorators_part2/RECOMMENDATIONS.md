# RECOMMENDATIONS: trinity/decorators (Part 2)

## Rust Bridge Requirements

### High Priority

| Requirement | Rationale | Complexity |
|-------------|-----------|------------|
| PyO3 bindings for Op enum | Core building block for all decorator serialization | Medium |
| PyO3 bindings for Step struct | Required to serialize decorator effects to Rust | Medium |
| Registry bridge | Allows Rust systems to query decorated components | High |
| TAG op handler in Rust | Most common op, translates to ECS component metadata | Medium |

### Medium Priority

| Requirement | Rationale | Complexity |
|-------------|-----------|------------|
| REGISTER op handler in Rust | Enables native registry lookups | Medium |
| Validation offload | Performance-critical decorators can validate in Rust | Medium |
| Config dataclass serialization | Frozen dataclasses map to Rust structs | Low |
| Introspection FFI | Rust tooling needs to query decorator metadata | Medium |

### Low Priority

| Requirement | Rationale | Complexity |
|-------------|-----------|------------|
| HOOK op handler | Lifecycle hooks may remain Python-side | High |
| Stack validation in Rust | Compile-time anti-pattern detection | Medium |
| DESCRIBE op for docs | Documentation generation can stay Python | Low |

## Integration Strategy

### Phase 1: Core Types

1. Define Rust equivalents for Op enum
2. Define Rust Step struct with serde serialization
3. Create PyO3 bindings for Python-to-Rust conversion
4. Test roundtrip serialization

### Phase 2: Registry Bridge

1. Expose registry lookups via PyO3
2. Implement native registry storage in Rust
3. Create sync mechanism for Python-Rust registries
4. Test concurrent access patterns

### Phase 3: Op Handlers

1. Implement TAG handler (ECS metadata)
2. Implement REGISTER handler (native registry)
3. Create fallback to Python for unhandled ops
4. Benchmark Python vs Rust op execution

### Phase 4: Optimization

1. Move validation to Rust for hot paths
2. Implement stack validation at compile-time
3. Add native introspection API
4. Profile and optimize critical paths

## Testing Strategy

### Unit Tests
- Op serialization roundtrip
- Step builder output validation
- Validation function edge cases
- Config dataclass serialization

### Integration Tests
- Python decorator to Rust component
- Registry sync correctness
- Lifecycle hook propagation
- Stack composition effects

### Performance Tests
- Op execution latency (Python vs Rust)
- Registry lookup performance
- Validation overhead
- Memory usage comparison

### Compatibility Tests
- All 110+ decorators serialize correctly
- Stack combinations work across bridge
- Introspection API parity

## Risk Assessment Table

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Op enum mismatch | Low | High | Version both sides, validate on bridge |
| Registry sync race | Medium | Medium | Use atomic operations, clear ownership |
| Validation divergence | Low | Medium | Single source of truth, generate from spec |
| Performance regression | Low | Low | Benchmark before migration, fallback path |
| Python attribute access | Medium | Low | Cache attribute lookups, batch operations |
| Lifecycle hook timing | Medium | Medium | Clear hook ordering contract, tests |
| Stack combination explosion | Low | Low | Existing anti-pattern detection sufficient |
| Config dataclass versioning | Low | Medium | Schema versioning, migration helpers |

## Summary

The trinity/decorators Part 2 implementation is production-ready Python code. The ops-based architecture was designed with cross-language bridging in mind. The primary effort for GRANDPHASE2 is implementing the Rust-side op handlers and registry bridge, which can be done incrementally without breaking existing Python usage.
