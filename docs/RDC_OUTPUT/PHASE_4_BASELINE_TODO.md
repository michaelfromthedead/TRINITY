# PHASE 4: Baseline Run — TODO

**Duration:** 1 day

---

## Tasks

### T-BASE-4.1: Cargo test integration
- [ ] Implement `run_cargo_test()` with JSON output
- [ ] Parse cargo test JSON format
- [ ] Extract test name, duration, result

### T-BASE-4.2: Pytest integration
- [ ] Implement `run_pytest()` with JSON report
- [ ] Parse pytest-json-report format
- [ ] Extract test name, duration, result

### T-BASE-4.3: Result mapping
- [ ] Look up test node in graph
- [ ] Get target nodes via Tests edges
- [ ] Aggregate results per target

### T-BASE-4.4: State transitions
- [ ] Implement TestsPassed event handling
- [ ] Implement TestsFailed event handling
- [ ] Update current_state in code_nodes

### T-BASE-4.5: Run all tests
- [ ] Execute cargo test (may take 10+ minutes)
- [ ] Execute pytest (may take 30+ minutes)
- [ ] Handle timeouts and failures

### T-BASE-4.6: Record baseline
- [ ] Store baseline timestamp
- [ ] Store per-node state
- [ ] Store any test failures for triage

### T-BASE-4.7: Validation
- [ ] Verify all nodes have state != UNKNOWN
- [ ] Count GREEN vs RED vs UNTESTED
- [ ] Report summary

---

## Estimates

| Task | Optimistic | Realistic | Pessimistic |
|------|------------|-----------|-------------|
| T-BASE-4.1 | 2h | 4h | 8h |
| T-BASE-4.2 | 2h | 4h | 8h |
| T-BASE-4.3 | 2h | 4h | 8h |
| T-BASE-4.4 | 2h | 4h | 8h |
| T-BASE-4.5 | 1h | 2h | 4h (+ test runtime) |
| T-BASE-4.6 | 1h | 2h | 4h |
| T-BASE-4.7 | 1h | 2h | 4h |
| **Total** | **11h** | **22h** | **44h** |
