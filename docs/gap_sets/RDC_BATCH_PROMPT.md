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
| 2 | GAPSET_2_FRAME_GRAPH | ⏳ NEXT | — | — |
| 3 | GAPSET_3_BRIDGE | ✅ SDLC-READY | 17 | 39/0/0 |
| 4 | GAPSET_4_MATERIALS | ⏳ | — | — |
| 5 | GAPSET_5_LIGHTING | ⏳ | — | — |
| 6 | GAPSET_6_GI_REFLECTIONS | ⏳ | — | — |
| 7 | GAPSET_7_POST_PROCESS | ⏳ | — | — |
| 8 | GAPSET_8_GPU_COMPUTE | ⏳ | — | — |
| 9 | GAPSET_9_RAY_TRACING | ⏳ | — | — |
| 10 | GAPSET_10_ENVIRONMENT | ⏳ | — | — |
| 11 | GAPSET_11_DEMOSCENE | ⏳ | — | — |
| 12 | GAPSET_12_ASSETS | ⏳ | — | — |
| 13 | GAPSET_13_TOOLING | ⏳ | — | — |
| 14 | GAPSET_14_ANIMATION | ⏳ | — | — |
| 15 | GAPSET_15_AUDIO | ⏳ | — | — |
| 16 | GAPSET_16_NETWORKING | ⏳ | — | — |
| 17 | GAPSET_17_GAMEPLAY | ⏳ | — | — |
| 18 | GAPSET_18_UI_XR | ⏳ | — | — |
| 19 | GAPSET_19_PHYSICS | ⏳ | — | — |
| 20 | GAPSET_20_CROSS_CUTTING | ⏳ | — | — |

**Progress:** 2/20 SDLC-READY · 18 remaining · 3 per cycle = 6 cycles (~1 hour)
**Next targets:** GAPSET_2_FRAME_GRAPH, GAPSET_4_MATERIALS, GAPSET_5_LIGHTING
