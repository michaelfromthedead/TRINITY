# KNOWN_LIMITATIONS — LANG_DEV_V2 v2.0.0-DRAFT Buildout

**Purpose:** Honest disclosure of what v2.0.0-DRAFT does NOT yet do, what is deferred, and what would need to happen for a first live engagement.

**Buildout status:** All 9 parts of `LANG_DEV_V2_BUILDOUT_TODO.md` complete (32 tasks). State machine, contracts, workers, walkthroughs, validators, external integration all in place.

**Engagement status:** **Not yet engageable on a real target.** Requires first-engagement prep work documented below.

---

## 1. Reference library content not built

The reference library `pandas_mini` and its 8 nexus reports specified in `spec/REFERENCE_LIBRARY.md` are NOT built. Only the directory structure (`test_target/`) and the validators are committed.

**Why:** Building `pandas_mini` is ~300-500 LOC of Python + ~200KB of structured nexus reports. Creating it without first running the workflow against it produces "expected" outputs that may not match what the workflow actually generates — risking `expected_workspace_manifest.json` mismatches.

**Fix path:** First engagement on a small synthetic library (or a real one with existing nexus reports). Once one successful run completes, `pandas_mini` can be authored to match the now-known workflow behavior.

**Workaround:** Engage v2 on a real target with already-existing nexus reports (e.g., the original `pandas` from CREATOR_NOTES). Skip the reference-library validation path; rely on production-target compression threshold (18:1).

---

## 2. Validators do not yet exercise full schema coverage

`test_target/validators/validate_*.py` cover the major schema invariants (workspace manifest structure, decisions schema referential integrity, output existence). They do NOT validate:

- Per-task `output_contract` adherence at fine granularity (each contract's "Required outputs" table)
- Acceptance command exit codes (each contract's "Acceptance command" block)
- Source-doc citation accuracy (PHASE_QA's spot-check fabrication discipline)

**Why deferred:** These checks are role of PHASE_QA per spawn, not of standalone validators. The validators are bundled tools that PHASE_QA may invoke; they don't replace PHASE_QA's full verification.

**Fix path:** As workflow runs accumulate, add per-task validators (e.g., `validate_T_01_1_outputs.py`) for repeatable structural checks. Don't pre-build them speculatively.

---

## 3. No first-engagement smoke test executed

The workflow has not been engaged in any conversation. PRESTEP validation logic (per `spec/INPUT_CONTRACT.md` §6) has not been exercised against actual inputs.

**Why deferred:** v2.0.0-DRAFT is a buildout, not a first-run. First engagement is a separate work item.

**Risk:** Subtle bugs in PRESTEP path resolution, file-existence checks, or workspace_manifest.json initialization will only surface on first engagement.

**Fix path:** First engagement should be done on a small target with engaged-eyes review. Treat the first PRESTEP report as the smoke test.

---

## 4. PARALLEL_BATCH_UNIT semantics not exercised

Phase 4's parallel sibling pattern (T-04.1 + T-04.2 after T-04.3) is specified in `LANG_DEV_V2_WORKFLOW.json` and walked through in `walkthroughs/PHASE_4_PARALLEL_WALKTHROUGH.md`, but the QUEEN orchestration logic has not been tested against Claude's actual `run_in_background: true` task spawning.

**Risk:** Aggregate verdict logic may be subtly wrong; QUEEN may poll instead of awaiting completion notifications.

**Fix path:** First engagement on a target where Phase 4 reaches the parallel-batch step. Adjust QUEEN behavior based on observed orchestration patterns.

---

## 5. Recovery counter accounting needs first-run validation

`spec/RECOVERY_MODEL.md` specifies that T-01.1.1 spawns increment `recovery_attempts` (max 3) and that T-01.1's TASK_FAIL_RETRY counter is independent. The interaction between the two counters is specified, but PHASE_QA verdict cascades involving recovery have not been observed in practice.

**Risk:** Counters may double-increment or fail to reset; recovery loops may exceed bounds.

**Fix path:** First engagement that triggers recovery. Verify counter behavior in `workspace_manifest.json` matches spec.

---

## 6. METHODOLOGY_INTEGRATOR test infrastructure is paper-only

The integrator's three gates are fully specified (`spec/SHUFFLE_TEST_SPEC.md`, `spec/COMPRESSION_SPEC.md`, `spec/E2E_DEMO_SPEC.md`) but no actual integrator code exists. METHODOLOGY_INTEGRATOR is described as "Claude in main conversation invoked as the worker" — Claude executes the gates manually using read/run tools.

**Why:** Building integrator code without a real workspace to test against would just be more paper. Better to have Claude execute the spec procedurally on first run.

**Risk:** Manual procedural execution is more error-prone than tested code. Counts may be miscounted, comparison may be inexact, edge cases may be missed.

**Fix path:** After first integrator run completes, codify the procedure into Python test scripts (`test_target/integrator/run_shuffle_test.py`, etc.). Add to validators directory.

---

## 7. v1's WORKER_STEP_EXECUTOR / WORKER_STEP_QA still exist

The two v1 worker docs at `workflows/LANG_DEV/WORKER_STEP_EXECUTOR.md` and `WORKER_STEP_QA.md` are preserved as historical artifacts. They reference the v0.1.0 JSON which is now SUPERSEDED.

**Risk:** A future Claude session might read v1 worker docs without realizing they're deprecated. The DEPRECATED.md notice is in `workflows/LANG_DEV/`, not in the worker doc files themselves.

**Fix path:** Add a `**DEPRECATED — see ../LANG_DEV_V2/**` banner to the top of v1 worker docs. (Not done in this buildout to preserve the historical content; consider for a v2.0.1 cleanup pass.)

---

## 8. No cross-workflow conflict tests

v2's `no_workflow_mixing` rule says LANG_DEV_V2 cannot interleave with other active workflows. But QUEEN's enforcement of this is implicit — QUEEN simply doesn't spawn other workflows' workers. There is no explicit check at PRESTEP for "is another workflow currently active."

**Risk:** If a user types `LANG_DEV_V2_WORKFLOW` while in the middle of an SDLC run (e.g., between two task spawns), the engagement protocol may proceed without flagging the conflict.

**Fix path:** Add to PRESTEP §6: "Check INPROGRESS.md head entry for an active workflow marker; if active workflow != LANG_DEV_V2 and != none, escalate with 'workflow_mixing_attempted'." Defer to v2.1.

---

## 9. Open Questions from buildout TODO unresolved

Per `LANG_DEV_V2_BUILDOUT_TODO.md` §"Open questions to resolve during buildout":

1. **Whether v2 should be a workflow at all vs. handing RDC TODOs to SDLC.** Decision deferred — v2 keeps specialized workflow because of multi-doc reads, on-demand recovery, methodology-level acceptance. Revisit after first run informs the trade-off concretely.
2. **Sub-task spawning model** (one PHASE_EXECUTOR per parent task vs. one per sub-task). v2 chose: one per sub-task (T-02.4.1 and T-02.4.2 each get their own PHASE_TASK_UNIT). Document but don't litigate.
3. **Per-phase parallelism worth complexity?** Phase 4 yes (specified); Phase 2 sub-tasks marginal — kept sequential in v2.0.0.
4. **PCFG online learning timing** — explicitly deferred to post-methodology step; not in v2 scope.
5. **Multi-target executor** — out of scope for v2.0.0; one target per run.
6. **`<library>_decisions.json` schema versioning** — currently hardcoded to 1.0.0; defer schema-evolution policy to v2.1.
7. **Boss_level worker subdivision** — kept as single PHASE_EXECUTOR with task-specific source docs; no subdivision.

---

## 10. Documentation only goes one direction

v2 docs reference the RDC corpus (`workflows/LANG_DEV/LANGS_DEV_RDC/`) extensively but the RDC corpus does NOT reference v2. The RDC's `PHASE_*_TODO.md` headers say "Consumed by: `LANG_DEV_WORKFLOW` v0.2.0+" — they don't explicitly call out `LANG_DEV_V2_WORKFLOW`.

**Why not updated:** The RDC corpus is the AUTHORITATIVE methodology source; editing it to point to v2 would muddy the "RDC is the source, v2 implements it" relationship.

**Fix path:** None recommended. The cross-reference flows correctly: humans encountering the RDC corpus will find v2 via `workflows/README.md` table; humans encountering v2 will find the RDC via `WHY_V2.md` and worker docs.

---

## First-engagement preparation checklist

Before invoking `LANG_DEV_V2_WORKFLOW` for real:

- [ ] Choose target_library (recommend small, well-known: `requests`, `click`, or a domain-specific library you know)
- [ ] Generate or locate nexus reports for the target (8 reports: GRAVITY, GRAMMAR, VERBS, TIERS, CLASSIFICATION, GENESIS, GENERATOR, COMPRESSION) — out of scope for this workflow
- [ ] Place 22 source methodology docs at `LANG_DEV_STEPS/` (or set `step_source_dir` parameter)
- [ ] Choose empty workspace_dir (will be created)
- [ ] Verify `INPROGRESS.md` is in a clean state (no active workflow)
- [ ] Engage with `LANG_DEV_V2_WORKFLOW`
- [ ] Treat first PRESTEP report as smoke test — review carefully before proceeding
- [ ] Be prepared for any phase to escalate (this is a first run)
- [ ] Document any workflow-design issues you discover for v2.0.1

---

## Buildout deliverables index

For reference, the 32 tasks across 9 parts produced:

**Spec layer (Part 1):**
- `WHY_V2.md`, `spec/PHASE_MODEL.md`, `spec/INPUT_CONTRACT.md`, `spec/STEP_DOC_INVENTORY.md`

**Contract layer (Part 2):**
- `contracts/PHASE_<01-04>_CONTRACT.md`, `contracts/ARTIFACT_CATALOG.md`

**State machine (Part 3):**
- `LANG_DEV_V2_WORKFLOW.json`

**Workers (Part 4):**
- `WORKER_PHASE_EXECUTOR.md`, `WORKER_PHASE_QA.md`, `WORKER_METHODOLOGY_INTEGRATOR.md`, `spec/RECOVERY_MODEL.md`

**Acceptance specs (Part 5):**
- `spec/REFERENCE_LIBRARY.md`, `spec/SHUFFLE_TEST_SPEC.md`, `spec/COMPRESSION_SPEC.md`, `spec/E2E_DEMO_SPEC.md`

**Walkthroughs (Part 6):**
- `walkthroughs/{HAPPY_PATH,RECOVERY,SHUFFLE_FAIL,PHASE_4_PARALLEL}_WALKTHROUGH.md`

**Test scaffolding (Part 7):**
- `test_target/README.md`, `test_target/sample_data/small.csv`, `test_target/validators/{validate_workspace_manifest,validate_decisions_schema,validate_phase_outputs}.py`

**External integration (Part 8):**
- `LANG_DEV_V2/README.md` (final)
- `workflows/README.md` updated (directory + summaries + versioning)
- `workflows/CLAUDE_APPENDIX.md` updated (trigger table + engagement entry)
- `CLAUDE.md` updated (workflow table + 5 new hard rules + artifacts table + references)
- `workflows/SHARED/WORKER.md` updated (LANG_DEV_V2 role section)
- `workflows/LANG_DEV/DEPRECATED.md` (v1 supersession notice)
- `workflows/LANG_DEV/LANG_DEV_WORKFLOW.json` version bumped to `0.1.0-SUPERSEDED`

**Verification (Part 9):**
- `LANG_DEV_V2_BUILDOUT_TODO.md` — all checkboxes marked complete
- This file (`KNOWN_LIMITATIONS.md`)

---

## Final verdict on buildout

- ✓ All 32 buildout tasks delivered
- ✓ JSON state machine valid; all `output_contract` references resolve
- ✓ Validators run cleanly on synthetic test data
- ✓ External integration complete (workflows/README, CLAUDE.md, CLAUDE_APPENDIX, SHARED/WORKER)
- ⚠ Reference library content + first engagement deferred (Limitations §1, §3)
- ⚠ METHODOLOGY_INTEGRATOR is procedural, not coded (Limitations §6)
- ⚠ Workflow not exercised; subtle bugs may surface on first run (Limitations §3, §4, §5)

**Verdict: BUILDOUT COMPLETE; FIRST ENGAGEMENT REQUIRED before declaring v2.0.0 fit for production use.**

---

*End of KNOWN_LIMITATIONS.*
