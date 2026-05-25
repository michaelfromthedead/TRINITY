# GAPS_SDLC_TODO — Master SDLC Worklist

**Purpose:** Track SDLC progress across all 20 gapsets. Each gapset is worked sequentially (1→2→...→20). Tasks within a gapset follow priority order: unblocked `[ ]` and `[-]` items first.

**Last updated:** 2026-05-24

---

## Pipeline State

| # | Gapset | Tasks | [x] | [~] | [-] | [ ] | Progress | Status |
|---|--------|-------|-----|-----|-----|-----|----------|--------|
| 1 | CORE | ~37* | 18 | 11 | 8 | 0 | 49% | ⬅ **IN PROGRESS** |
| 2 | FRAME_GRAPH | ~57* | 16 | 17 | 24 | 0 | 28% | ⏳ Queued |
| 3 | BRIDGE | 39 | 39 | 0 | 0 | 0 | **100%** | ✅ DONE + VERIFIED |
| 4 | MATERIALS | ~67* | 4 | 17 | 46 | 0 | 6% | ⏳ Queued |
| 5 | LIGHTING | ~33* | 1 | 4 | 28 | 0 | 3% | ⏳ Queued |
| 6 | GI_REFLECTIONS | ~44* | 0 | 8 | 36 | 0 | 0% | ⏳ Queued |
| 7 | POST_PROCESS | ~70* | 20 | 19 | 31 | 0 | 29% | ⏳ Queued |
| 8 | GPU_COMPUTE | ~35* | 12 | 11 | 12 | 0 | 34% | ⏳ Queued |
| 9 | RAY_TRACING | ~35* | 3 | 4 | 28 | 0 | 9% | ⏳ Queued |
| 10 | ENVIRONMENT | ~38* | 0 | 0 | 38 | 0 | 0% | ⏳ Queued |
| 11 | DEMOSCENE | ~46* | 20 | 14 | 12 | 0 | 43% | ⏳ Queued |
| 12 | ASSETS | ~40* | 0 | 1 | 6 | 33 | 0% | ⏳ Queued |
| 13 | TOOLING | ~62* | 24 | 18 | 20 | 0 | 39% | ⏳ Queued |
| 14 | ANIMATION | ~68* | 44 | 5 | 19 | 0 | 65% | ⏳ Queued |
| 15 | AUDIO | ~129* | 92 | 19 | 18 | 0 | 71% | ⏳ Queued |
| 16 | NETWORKING | ~65* | 45 | 9 | 11 | 0 | 69% | ⏳ Queued |
| 17 | GAMEPLAY | ~130* | 115 | 6 | 9 | 0 | 88% | ⏳ Queued |
| 18 | UI_XR | ~68* | ~55 | ~8 | ~5 | 0 | 80% | ⏳ Queued |
| 19 | PHYSICS | ~54* | 35 | 2 | 17 | 0 | 65% | ⏳ Queued |
| 20 | CROSS_CUTTING | ~10* | 5 | 4 | 1 | 0 | 50% | ⏳ Queued |

> \* = RDC-verified task counts from corrected PHASE_N_TODO.md. Checkmark counts represent true task status after source-code verification. GAP 18 counts are estimated (TODO not fully corrected).

### Legend

| Mark | Meaning | SDLC Action |
|------|---------|-------------|
| `[x]` | **DONE** — verified real, GREEN_LIGHT | None |
| `[~]` | **PARTIAL** — exists but incomplete or inactive | DEV needed to complete wiring |
| `[-]` | **ABSENT** — does not exist at all | DEV needed from scratch |
| `[ ]` | **NOT STARTED** — original plan, never begun | DEV needed from scratch |

### Overall Progress

```
████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~18% of gapset tasks GREEN_LIGHT
████████████░░░░░░░░░░░░░░░░░░░░░░░░  ~38% including partials
```

---

## Current Work Unit

**Active gapset:** GAPSET_1_CORE (sequential position: 1 of 20)

**Remaining work for CORE (8 ABSENT tasks):**

| ID | Task | Status | Effort |
|----|------|--------|--------|
| T-CORE-3.1 | ThreadPool with work-stealing | [-] | 1.5 days |
| T-CORE-3.2 | JobGraph and dependencies | [-] | 1.5 days |
| T-CORE-3.3 | parallel_for | [-] | 1 day |
| T-CORE-2.5a | HierarchicalChecksum | [-] | 1 day |
| T-CORE-2.5b | SystemPhase and SystemContext | [-] | 1 day |
| T-CORE-5.5 | Scheduler Bridge and Frame Loop | [-] | 1.5 days |
| T-CORE-1.3 | RingBuffer staging allocator | [~] | 1 day |
| T-CORE-1.4 | EntityId generational index | [~] | 0.5 day |

**11 PARTIAL tasks also need completion work.**

---

---

## Verification Log

### GAPSET_3_BRIDGE — Verified 2026-05-24

**Bridge Build:**
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p omega --features pyo3
cp target/debug/libomega.so _omega.so
```

**Bridge Test:**
```python
>>> import _omega
>>> _omega.frame_graph_execute('{"passes":[], "resources":[]}')
'{"success":true, "num_passes":0, ...}'
```

**Files Fixed:**
- `omega/Cargo.toml` — Added PyO3, renderer-backend, serde_json deps
- `omega/src/bridge.rs` — PyO3 0.20 API compatibility

**Result:** Python can now compile frame graphs via Rust backend.

---

## THREAD POOL (floating swarm state)

**Target: 6 active threads.** Cron fills to target each cycle.

| Slot | Gapset | Task | Stage | Agent IDs | Spawned |
|------|--------|------|-------|-----------|---------|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |
| 6 | — | — | — | — | — |
| 7 | — | — | — | — | — |
| 8 | — | — | — | — | — |

**Active threads:** 0/8

---

## Rules of Engagement

1. **Floating swarm:** Always maintain 4-8 active threads. Cron checks and fills.
2. **Sequential gapsets:** Work gapset 1 until fully GREEN, then gapset 2, etc.
3. **Priority within gapset:** `[-]` (absent) before `[~]` (partial). Dependencies `[x]` first.
4. **Full pipeline per task:** DEV → TEST_UNIT → QA_UNIT → VERDICT. No shortcuts.
5. **GREEN_LIGHT = toggle [x], free the slot, fill immediately.**
6. **All spawns in ONE message per cycle.** Batch every agent across all threads.
7. **Threads persist across cron cycles.** State lives in this table.
