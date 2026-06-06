# PHASE 4: Baseline Run — TODO

**Duration:** 1 day

---

## Tasks

### T-BASE-4.1: Cargo test integration ✓
- [x] Implement `run_cargo_test()` with JSON output
- [x] Parse cargo test JSON format
- [x] Extract test name, duration, result

### T-BASE-4.2: Pytest integration ✓
- [x] Implement `run_pytest()` with JSON report
- [x] Parse pytest-json-report format
- [x] Extract test name, duration, result

### T-BASE-4.3: Result mapping ✓
- [x] Look up test node in graph
- [x] Get target nodes via Tests edges
- [x] Aggregate results per target

### T-BASE-4.4: State transitions ✓
- [x] Implement TestsPassed event handling
- [x] Implement TestsFailed event handling
- [x] Update current_state in code_nodes

### T-BASE-4.5: Run all tests ✓
- [x] Execute cargo test (may take 10+ minutes)
- [x] Execute pytest (may take 30+ minutes)
- [x] Handle timeouts and failures

### T-BASE-4.6: Record baseline ✓
- [x] Store baseline timestamp
- [x] Store per-node state
- [x] Store any test failures for triage

### T-BASE-4.7: Validation ✓
- [x] Verify all nodes have state != UNKNOWN
- [x] Count GREEN vs RED vs UNTESTED
- [x] Report summary

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
