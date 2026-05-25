# GRAND_SYNTHESIS_PROMPT_FILE — Investigation-to-SDLC Cron Worker (3x Parallel)

## ⚠️ THIS WORKFLOW IS COMPLETE

**Status:** ALL 35/35 REPORTS PROCESSED
**Completed:** 2026-05-22

**Next Workflow:** `workflows/SDLC_WORKFLOW/SDLC_PROMPT_FILE.md`
- Phase A Sprint: Resource Streaming, Build Pipeline, Platform Services
- Use that prompt file for continued development work

---

**Trigger:** Cron every 10 minutes (`*/10 * * * *`) — DISABLED
**Goal:** Process THREE investigation reports per invocation using parallel workers.
**Output:** `docs/INVESTIGATION_PHASE_X_OUTPUT/<SUBSYSTEM>/` with PROJECT.md, CLARIFICATION.md, SUMMARY.md, RECOMMENDATIONS.md
**Stop condition:** All 35 investigation reports processed.
**Estimated completion:** ~2 hours (35 reports ÷ 3 workers × 10min intervals)

---

## WORKFLOW PER INVOCATION

### Step 0 — Orient
Read this file. Read `workflows/RDC_WORKFLOW/INVESTIGATION_STATE.json` for current progress.

### Step 1 — Get next THREE targets
```bash
cat workflows/RDC_WORKFLOW/INVESTIGATION_STATE.json | jq -r '.queue[] | select(.status == "pending") | .id' | head -3
```

If fewer than 3 pending items remain → process remaining. If none → **ALL DONE**.

### Step 2 — Spawn 3 parallel workers

Use the Agent tool to spawn 3 workers simultaneously. Each worker:
1. Reads its assigned investigation report from `docs/investigation/`
2. Creates output in `docs/INVESTIGATION_PHASE_X_OUTPUT/<SUBSYSTEM>/`
3. Produces 4 documents: PROJECT.md, CLARIFICATION.md, SUMMARY.md, RECOMMENDATIONS.md

**CRITICAL:** Spawn ALL 3 agents in ONE message with `run_in_background: true`.

```javascript
// Worker 1
Agent({
  description: "RDC Worker 1",
  subagent_type: "researcher",
  name: "rdc-worker-1",
  run_in_background: true,
  prompt: `RDC WORKFLOW: Process investigation report <REPORT_1>.

Read: docs/investigation/<REPORT_1>.md
Create: docs/INVESTIGATION_PHASE_X_OUTPUT/<SUBSYSTEM_1>/
  - PROJECT.md (goal, status, algorithms, integration points, GRANDPHASE2 recs)
  - CLARIFICATION.md (why exists, architecture decisions, GP1 vs GP2)
  - SUMMARY.md (quantitative: lines, files, algorithm inventory)
  - RECOMMENDATIONS.md (Rust bridge requirements, priorities, risks)

Return: "PHASE_<N>: <SUBSYSTEM> — COMPLETE (4 docs)"`
})

// Worker 2
Agent({
  description: "RDC Worker 2",
  subagent_type: "researcher", 
  name: "rdc-worker-2",
  run_in_background: true,
  prompt: `RDC WORKFLOW: Process investigation report <REPORT_2>...`
})

// Worker 3
Agent({
  description: "RDC Worker 3",
  subagent_type: "researcher",
  name: "rdc-worker-3", 
  run_in_background: true,
  prompt: `RDC WORKFLOW: Process investigation report <REPORT_3>...`
})
```

### Step 3 — Wait for completions
Do NOT poll. Workers will notify on completion.

### Step 4 — Update state tracker
After all 3 workers complete, update `workflows/RDC_WORKFLOW/INVESTIGATION_STATE.json`:
- Change processed items' status from "pending" to "complete"
- Increment "processed" counter by 3
- Update "last_updated" timestamp

### Step 5 — Report
```
RDC BATCH COMPLETE — 3 reports processed:
  - PHASE_<N1>: <SUBSYSTEM_1> ✓
  - PHASE_<N2>: <SUBSYSTEM_2> ✓
  - PHASE_<N3>: <SUBSYSTEM_3> ✓
Progress: <X>/35 complete. Next batch: <SUBSYSTEM_4>, <SUBSYSTEM_5>, <SUBSYSTEM_6>
```

---

## WORKER OUTPUT TEMPLATES

### PROJECT.md
```markdown
# <SUBSYSTEM> — Investigation Summary

**Owner:** Michael
**Status:** <REAL | PARTIAL | STUB>
**Lines of Code:** <N>
**RDC run:** <date>

## 1. Goal
<What this subsystem does>

## 2. Implementation Status
- [x] REAL: <list>
- [~] PARTIAL: <list>
- [-] MISSING: <list>

## 3. Key Algorithms Verified
| Algorithm | File | Status |
|-----------|------|--------|

## 4. Integration Points
- Upstream: <dependencies>
- Downstream: <consumers>
- GAPSET_3_BRIDGE: <connections>

## 5. GRANDPHASE2 Recommendations
<What Rust bridge work is needed>
```

### CLARIFICATION.md
```markdown
# <SUBSYSTEM> — Clarification Document

## Why This Subsystem Exists
## Architectural Decisions Discovered
## GRANDPHASE1 vs GRANDPHASE2 Relationship
## What Remains
```

### SUMMARY.md
```markdown
# <SUBSYSTEM> — Quantitative Summary

| Metric | Value |
|--------|-------|
| Total Lines | <N> |
| Classification | <status> |
| Files | <N> |

## Algorithm Inventory
| Algorithm | File | Lines | Status |
|-----------|------|-------|--------|
```

### RECOMMENDATIONS.md
```markdown
# <SUBSYSTEM> — GRANDPHASE2 Recommendations

## Rust Bridge Requirements
### High Priority
### Medium Priority
### Low Priority

## Integration Strategy
## Testing Strategy
## Risk Assessment
```

---

## QUEUE (35 items)

| Phase | Report | Subsystem | Status |
|-------|--------|-----------|--------|
| 1 | trinity_decorators_part1.md | trinity/decorators | pending |
| 2 | trinity_decorators_part2.md | trinity/decorators | pending |
| 3 | trinity_descriptors_metaclasses.md | trinity/descriptors+metaclasses | pending |
| 4 | foundation.md | foundation | pending |
| 5 | engine_rendering_framegraph.md | rendering/framegraph | pending |
| 6 | engine_rendering_gpu_driven.md | rendering/gpu_driven | pending |
| 7 | engine_rendering_materials.md | rendering/materials | pending |
| 8 | engine_rendering_lighting.md | rendering/lighting | pending |
| 9 | engine_rendering_particles.md | rendering/particles | pending |
| 10 | engine_rendering_postprocess.md | rendering/postprocess | pending |
| 11 | engine_rendering_demoscene.md | rendering/demoscene | pending |
| 12 | engine_animation_crowds_facial.md | animation/crowds+facial | pending |
| 13 | engine_animation_graph_ik.md | animation/graph+ik | pending |
| 14 | engine_animation_motionmatching_procedural.md | animation/motionmatching+procedural | pending |
| 15 | engine_animation_skeletal_systems.md | animation/skeletal+systems | pending |
| 16 | engine_audio_adaptive_core.md | audio/adaptive+core | pending |
| 17 | engine_audio_dialogue_dsp.md | audio/dialogue+dsp | pending |
| 18 | engine_audio_mixing_spatial.md | audio/mixing+spatial | pending |
| 19 | engine_gameplay_abilities_ai_camera.md | gameplay/abilities+ai+camera | pending |
| 20 | engine_gameplay_combat_components.md | gameplay/combat+components | pending |
| 21 | engine_gameplay_economy_entity_input.md | gameplay/economy+entity+input | pending |
| 22 | engine_gameplay_nav_quest.md | gameplay/nav+quest | pending |
| 23 | engine_simulation_character_cloth_collision.md | simulation/character+cloth+collision | pending |
| 24 | engine_simulation_destruction_fluid_hair.md | simulation/destruction+fluid+hair | pending |
| 25 | engine_simulation_physics_solver.md | simulation/physics+solver | pending |
| 26 | engine_simulation_components_constraints_softbody_vehicles.md | simulation/components+constraints+softbody+vehicles | pending |
| 27 | engine_ui_accessibility_animation_binding_framework.md | ui/accessibility+animation+binding+framework | pending |
| 28 | engine_ui_layout_screens_styling_text.md | ui/layout+screens+styling+text | pending |
| 29 | engine_ui_widgets.md | ui/widgets | pending |
| 30 | engine_platform | platform (consolidated) | pending |
| 31 | engine_world | world (consolidated) | pending |
| 32 | engine_xr | xr (consolidated) | pending |
| 33 | engine_networking | networking (consolidated) | pending |
| 34 | engine_debug_resource | debug+resource (consolidated) | pending |
| 35 | engine_tooling | tooling (consolidated) | pending |

**Progress:** 0/35 · **Batches:** 12 (at 3/batch) · **ETA:** ~2 hours

---

## CRON SETUP

```bash
# 10-minute intervals, 3 parallel workers
*/10 * * * * cd /home/user/dev/USER/PROJECTS_VOID/TRINITY && claude --prompt-file workflows/GRAND_SYNTHESIS_PROMPT_FILE.md >> logs/rdc_cron.log 2>&1
```

---

*Created: 2026-05-22*
*3x parallel worker configuration for faster processing*
