# PHASE 3: Test Mapping — TODO

**Duration:** 2-3 days

---

## Tasks

### T-MAP-3.1: Auto-mapping implementation
- [ ] Implement convention-based mapping for Rust blackbox tests
- [ ] Implement convention-based mapping for Rust unit tests
- [ ] Implement convention-based mapping for Python tests

### T-MAP-3.2: Manual mapping file
- [ ] Define TOML format for explicit mappings
- [ ] Implement parser for `test_mappings.toml`
- [ ] Handle glob patterns in targets

### T-MAP-3.3: Map Rust tests
- [ ] Scan `crates/*/tests/*.rs`
- [ ] Apply auto-mapping rules
- [ ] Create Tests edges

### T-MAP-3.4: Map Python tests
- [ ] Scan `tests/unit/`, `tests/integration/`, `tests/e2e/`
- [ ] Apply auto-mapping rules
- [ ] Create Tests edges

### T-MAP-3.5: Map inline tests
- [ ] Find `#[test]` in source files
- [ ] Map to containing module
- [ ] Create Tests edges

### T-MAP-3.6: Handle unmapped tests
- [ ] Identify tests without clear targets
- [ ] Log for manual review
- [ ] Create placeholder mappings or mark as orphan

### T-MAP-3.7: Coverage report
- [ ] Query: code nodes with at least one test
- [ ] Query: code nodes with no tests
- [ ] Generate coverage summary

### T-MAP-3.8: Validation
- [ ] Verify all 12,743 tests have at least one target
- [ ] Check for circular test dependencies
- [ ] Review orphan tests

---

## Estimates

| Task | Optimistic | Realistic | Pessimistic |
|------|------------|-----------|-------------|
| T-MAP-3.1 | 4h | 8h | 16h |
| T-MAP-3.2 | 2h | 4h | 8h |
| T-MAP-3.3 | 4h | 8h | 16h |
| T-MAP-3.4 | 4h | 8h | 16h |
| T-MAP-3.5 | 2h | 4h | 8h |
| T-MAP-3.6 | 2h | 4h | 8h |
| T-MAP-3.7 | 2h | 4h | 8h |
| T-MAP-3.8 | 2h | 4h | 8h |
| **Total** | **22h** | **44h** | **88h** |
