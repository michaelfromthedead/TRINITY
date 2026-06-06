# PHASE 5: Workflow Activation — TODO

**Duration:** 1 day

---

## Tasks

### T-WORK-5.1: HarnessDaemon implementation ✓
- [x] Create `daemon.rs`
- [x] Implement main loop
- [x] Handle graceful shutdown

### T-WORK-5.2: File watcher integration
- [ ] Start watcher in separate thread
- [ ] Send events to main loop
- [ ] Debounce rapid file changes

### T-WORK-5.3: Event processor
- [ ] Process file change events
- [ ] Trigger state transitions
- [ ] Propagate staleness

### T-WORK-5.4: CLI commands
- [ ] `trinity-harness daemon` — start daemon
- [ ] `trinity-harness query needs-testing` — list stale nodes
- [ ] `trinity-harness run-stale` — run only stale tests
- [ ] `trinity-harness update-from-results` — process test results

### T-WORK-5.5: CI workflow
- [ ] Create `.github/workflows/harness.yml`
- [ ] Query stale tests step
- [ ] Run stale tests step
- [ ] Update state step

### T-WORK-5.6: Notification service
- [ ] Implement basic pub/sub
- [ ] Add webhook support (optional)
- [ ] Log state transitions

### T-WORK-5.7: Documentation
- [ ] Document daemon operation
- [ ] Document CI integration
- [ ] Document CLI commands

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
