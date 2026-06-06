# PHASE 5: Workflow Activation — TODO

**Duration:** 1 day

---

## Tasks

### T-WORK-5.1: HarnessDaemon implementation ✓
- [x] Create `daemon.rs`
- [x] Implement main loop
- [x] Handle graceful shutdown

### T-WORK-5.2: File watcher integration ✓
- [x] Start watcher in separate thread
- [x] Send events to main loop
- [x] Debounce rapid file changes

### T-WORK-5.3: Event processor ✓
- [x] Process file change events
- [x] Trigger state transitions
- [x] Propagate staleness

### T-WORK-5.4: CLI commands ✓
- [x] `trinity-harness daemon` — start daemon
- [x] `trinity-harness query needs-testing` — list stale nodes
- [x] `trinity-harness run-stale` — run only stale tests
- [x] `trinity-harness update-from-results` — process test results

### T-WORK-5.5: CI workflow ✓
- [x] Create `.github/workflows/harness.yml`
- [x] Query stale tests step
- [x] Run stale tests step
- [x] Update state step

### T-WORK-5.6: Notification service ✓
- [x] Implement basic pub/sub
- [x] Add webhook support (optional)
- [x] Log state transitions

### T-WORK-5.7: Documentation ✓
- [x] Document daemon operation
- [x] Document CI integration
- [x] Document CLI commands

---

## Estimates

| Task | Optimistic | Realistic | Pessimistic |
|------|------------|-----------|-------------|
| T-WORK-5.1 | 2h | 4h | 8h |
| T-WORK-5.2 | 2h | 4h | 8h |
| T-WORK-5.3 | 2h | 4h | 8h |
| T-WORK-5.4 | 4h | 8h | 16h |
| T-WORK-5.5 | 2h | 4h | 8h |
| T-WORK-5.6 | 2h | 4h | 8h |
| T-WORK-5.7 | 2h | 4h | 8h |
| **Total** | **16h** | **32h** | **64h** |
