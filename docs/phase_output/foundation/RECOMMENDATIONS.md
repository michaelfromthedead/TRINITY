# RECOMMENDATIONS: foundation/

## Rust Bridge Requirements

### High Priority

None. Foundation/ is a Python-only infrastructure layer with no GPU operations or hot-path concerns.

### Medium Priority

| Component | Rationale | Approach |
|-----------|-----------|----------|
| content_store.py SHA-256 | Crypto operations could benefit from Rust | Use PyO3 bindings to ring/rustcrypto |
| query.py execution | Large entity counts may bottleneck | Profile first; consider Rust iterator if >10K entities |

### Low Priority

| Component | Rationale | Approach |
|-----------|-----------|----------|
| delta_sync.py diffing | Network sync optimization | Only if network layer implemented |
| migrations.py BFS | Schema migration is infrequent | Keep Python |
| provenance.py tracking | Debug-only feature | Keep Python |

---

## Integration Strategy

### Phase 1: Validation (1-2 days)

1. Create integration tests for bridge.py TrinityWorldAdapter
2. Validate get_trinity_registry() returns correct component metadata
3. Test bidirectional sync between Trinity instances and ShellLang entities

### Phase 2: AI Interface Hardening (1-2 days)

1. Define JSON schema spec for shelllang/ai.py commands
2. Validate against actual AI agent requirements
3. Add input validation and error handling for malformed commands

### Phase 3: Security Audit (2-3 days)

1. Audit capabilities.py for capability escalation vulnerabilities
2. Audit secure_shell.py for sandbox escape risks
3. Review @require_capability decorator coverage
4. Add security test suite

---

## Testing Strategy

### Unit Tests (Existing Layer)

| Layer | Priority | Coverage Target |
|-------|----------|-----------------|
| Layer 0 (Essential) | HIGH | 90% |
| Layer 1 (Structural) | HIGH | 90% |
| Layer 2 (Reactive) | HIGH | 85% |
| Layer 3 (Interactive) | MEDIUM | 75% |
| Layer 4 (Integration) | HIGH | 80% |

### Integration Tests (New)

| Test Suite | Priority | Description |
|------------|----------|-------------|
| bridge_trinity_test.py | HIGH | TrinityWorldAdapter with real components |
| query_reactive_test.py | HIGH | Subscription callbacks on entity changes |
| provenance_chain_test.py | MEDIUM | Full derivation tree construction |
| capability_sandbox_test.py | HIGH | Capability enforcement edge cases |
| shelllang_ai_test.py | HIGH | All AI commands with validation |

### Performance Tests (New)

| Test | Trigger | Threshold |
|------|---------|-----------|
| query_large_world | >1000 entities | <100ms |
| content_store_hash | >1MB object | <50ms |
| delta_sync_patch | 100 changes | <10ms |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Capability escalation | LOW | HIGH | Security audit + test suite |
| Query cache memory leak | MEDIUM | MEDIUM | Weak references already used; add monitoring |
| Provenance context leak | LOW | LOW | Context variables auto-clear |
| AI command injection | MEDIUM | HIGH | Input validation + schema enforcement |
| Thread safety race | LOW | MEDIUM | RLock coverage review |
| Migration path missing | LOW | MEDIUM | BFS algorithm handles gracefully |

---

## Action Items

### Immediate (This Sprint)

1. [ ] Create bridge_trinity_test.py integration tests
2. [ ] Define AI command JSON schema
3. [ ] Review capability decorator coverage

### Short-term (Next Sprint)

1. [ ] Security audit of capabilities.py and secure_shell.py
2. [ ] Performance profiling with large entity counts
3. [ ] Add query cache monitoring/metrics

### Long-term (Backlog)

1. [ ] Consider Rust acceleration if profiling shows need
2. [ ] Network transport layer for delta_sync.py
3. [ ] Production file organization for FileBackend
4. [ ] External API documentation
