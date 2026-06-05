# RDC Batch Processor — Auto-advance all gapsets to SDLC-READY

**Trigger:** Cron every 10 minutes (`3,13,23,33,43,53 * * * *`)
**Goal:** Process THREE gapsets IN PARALLEL per invocation. Each gapset gets its own RDC worker agent.
**Stop condition:** All 20 gapsets at SDLC-READY (PROJECT.md present in each).

---

## WORKFLOW PER INVOCATION

### Step 1 — Audit
Run `for d in /home/user/dev/USER/PROJECTS_VOID/TRINITY/docs/gap_sets/*/; do name=$(basename "$d"); has_proj=$([ -f "$d/PROJECT.md" ] && echo "READY" || echo "NEEDS"); echo "$name $has_proj"; done`

### Step 2 — Select 3 targets
Pick the FIRST 3 gapsets with NEEDS status (no PROJECT.md), in numerical order: 2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20. Skip 1 and 3 (already READY). If fewer than 3 remain, process all remaining.

### Step 3 — Spawn 3 parallel RDC workers
Spawn 3 Agent tool calls in ONE message, all with `run_in_background: true`. Each agent gets a self-contained prompt for ONE gapset:

```
RDC WORKER for GAPSET_N_NAME.

Read the existing doc at docs/gap_sets/<GAPSET>/PHASE_N_TODO.md — this is the original plan.

TASK: Run full RDC on this gapset:
1. Read PHASE_N_TODO.md thoroughly. Understand the tasks, phases, and what's claimed.
2. Do source-code investigation — read actual files in the TRINITY codebase to verify claims.
   - Rust: omega/src/, crates/renderer-backend/src/
   - Python: engine/, trinity/
   - WGSL: crates/renderer-backend/shaders/, engine/rendering/
3. Create GAP_N_SUMMARY.md — per-task verification: REAL [x], PARTIAL [~], ABSENT [-].
4. Edit PHASE_N_TODO.md in place — toggle checkmarks, add Reality: annotations, add corrected summary.
5. Create PROJECT.md — scope, goals, phase overview table, status.
6. Create CLARIFICATION.md — architectural philosophy, divergence analysis, deferred items.
7. Create PHASE_N_ARCH.md files — one per phase, with component breakdowns.
8. Report: "GAPSET_N: RDC DONE — X tasks, Y [x], Z [~], W [-]. K docs produced."

Discipline: never fabricate. Verify against actual files. Use cross-references to GAPSET_3_BRIDGE where shared infrastructure (omega, ComponentStore, wgpu, WGSL shaders) is relevant.
```

### Step 4 — Wait for all 3 agents to complete
Agents return automatically. Review their summaries briefly — if any agent failed or produced obviously wrong output, re-spawn it for that gapset.

### Step 5 — Update tracker
After all 3 complete, update the WORK UNIT TRACKER table below with results.

### Step 6 — If all 20 READY, stop
If every gapset has PROJECT.md, report "ALL DONE — 20/20 SDLC-READY" and stop spawning.

---

## RDC METHODOLOGY

Code-facing gapsets use the **codebase as source documents**:
- PHASE_N_TODO.md = structured problem statement ("here's what we planned")
- Actual source files = ground truth ("here's what exists")
- Source-code inspection = SCRIBE step
- Document production = TAXONOMY step
- Checkmarks: `[x]` = verified real, `[~]` = partial, `[-]` = absent
- Every task gets a `Reality:` annotation with file paths and line counts

Cross-reference: GAPSET_3_BRIDGE infrastructure (omega math, ComponentStore, wgpu renderer, WGSL shaders) is shared across all gapsets.

---

## WORK UNIT TRACKER

| # | Gapset | Status | Docs | Tasks [x]/[~]/[-] |
|---|--------|--------|------|---------------------|
| 1 | GAPSET_1_CORE | ✅ SDLC-READY | 10 | 18/11/8 |
| 2 | GAPSET_2_FRAME_GRAPH | ✅ SDLC-READY | — | 16/17/24 |
| 3 | GAPSET_3_BRIDGE | ✅ SDLC-READY | 17 | 39/0/0 |
| 4 | GAPSET_4_MATERIALS | ✅ SDLC-READY | — | 4/17/46 |
| 5 | GAPSET_5_LIGHTING | ✅ SDLC-READY | — | 1/4/28 |
| 6 | GAPSET_6_GI_REFLECTIONS | ✅ SDLC-READY | — | 0/8/36 |
| 7 | GAPSET_7_POST_PROCESS | ✅ SDLC-READY | — | 20/19/31 |
| 8 | GAPSET_8_GPU_COMPUTE | ✅ SDLC-READY | — | 12/11/12 |
| 9 | GAPSET_9_RAY_TRACING | ✅ SDLC-READY | — | 3/4/28 |
| 10 | GAPSET_10_ENVIRONMENT | ✅ SDLC-READY | — | 0/0/38 |
| 11 | GAPSET_11_DEMOSCENE | ✅ SDLC-READY | — | 20/14/12 |
| 12 | GAPSET_12_ASSETS | ✅ SDLC-READY | — | 0/1/6 |
| 13 | GAPSET_13_TOOLING | ✅ SDLC-READY | — | 24/18/20 |
| 14 | GAPSET_14_ANIMATION | ✅ SDLC-READY | — | 44/5/19 |
| 15 | GAPSET_15_AUDIO | ✅ SDLC-READY | — | 92/19/18 |
| 16 | GAPSET_16_NETWORKING | ✅ SDLC-READY | — | 45/9/11 |
| 17 | GAPSET_17_GAMEPLAY | ✅ SDLC-READY | — | 115/6/9 |
| 18 | GAPSET_18_UI_XR | ✅ SDLC-READY | — | ~55/~8/~5 |
| 19 | GAPSET_19_PHYSICS | ✅ SDLC-READY | — | 35/2/17 |
| 20 | GAPSET_20_CROSS_CUTTING | ✅ SDLC-READY | — | 5/4/1 |

**Progress:** 20/20 SDLC-READY ✅ **ALL COMPLETE**
**Completed:** 2026-05-24
