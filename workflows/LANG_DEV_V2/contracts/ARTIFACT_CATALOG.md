# ARTIFACT_CATALOG — Cross-Phase Artifact Index

**Purpose:** Single index of every artifact produced/consumed across LANG_DEV_V2's 4 phases. Source of truth for `workspace_manifest.json` schema.

**Convention:**
- `Type: workspace` = transient/per-phase artifact (lives in `workspace_dir/STEP_NN/` or `workspace_dir/BOSS_LEVEL_N/`)
- `Type: authoritative` = cross-phase, lives at `workspace_dir/` root, consumed by multiple downstream tasks
- `Type: methodology` = produced by METHODOLOGY_INTEGRATOR; terminal output

---

## Master artifact index

| Path | Producer | Consumers | Type | Format |
|---|---|---|---|---|
| `workspace_manifest.json` | QUEEN (initial); each PHASE_EXECUTOR (append) | All tasks (read for prior outputs) | authoritative | JSON |
| **Phase 1 — DECONSTRUCTION** | | | | |
| `STEP_01/primitives_catalog.json` | T-01.1 | T-01.2, T-01.3, T-02.1, METHODOLOGY_INTEGRATOR | workspace | JSON |
| `STEP_01/tier_compression.md` | T-01.1 | T-01.1 (acceptance), METHODOLOGY_INTEGRATOR (compression check) | workspace | Markdown |
| `STEP_01/deconstruction_notes.md` | T-01.1 | T-01.1.1 (if recovery) | workspace | Markdown |
| `STEP_01/recovery_log.md` | T-01.1.1 (conditional) | T-01.1 retry | workspace | Markdown |
| `STEP_02/object_hierarchy.json` | T-01.2 | T-01.3, T-02.1 | workspace | JSON |
| `STEP_02/object_operation_matrix.md` | T-01.2 | T-02.1 (atom design context) | workspace | Markdown |
| `STEP_03/type_signatures.json` | T-01.3 | T-02.1, T-02.3 | workspace | JSON |
| `STEP_03/composition_graph.dot` | T-01.3 | (visual reference) | workspace | Graphviz DOT |
| `STEP_03/type_algebra.md` | T-01.3 | T-02.1 | workspace | Markdown |
| **Phase 2 — DESIGN** | | | | |
| `STEP_04/atoms_draft.json` | T-02.1 | T-02.2 | workspace | JSON |
| `STEP_04/port_types_draft.json` | T-02.1 | T-02.2 | workspace | JSON |
| `STEP_04/phases_draft.json` | T-02.1 | T-02.2 | workspace | JSON |
| `STEP_04/pcfg_weights_initial.json` | T-02.1 | T-03.5 (Solver) | workspace | JSON |
| `<library>_decisions.json` | T-02.2 | T-02.3, T-02.4, T-03.1.1, T-03.3, T-04.1, METHODOLOGY_INTEGRATOR | **authoritative** | JSON |
| `STEP_05B/bag_grammar_spec.md` | T-02.3 | T-02.4, T-03.1, T-03.2 | workspace | Markdown |
| `STEP_06/ruleset_spec.md` | T-02.4 | T-02.4.1, T-03.4 | workspace | Markdown |
| `STEP_06_1/conundrum_resolution.md` | T-02.4.1 | (philosophical reference) | workspace | Markdown |
| `STEP_06_2/token_inventory.md` | T-02.4.2 | T-03.1 (Lexer) | workspace | Markdown |
| **Phase 3 — IMPLEMENTATION** | | | | |
| `STEP_07/{lexer,tokens,test_lexer}.py` | T-03.1 | T-03.1.1 (validator), T-03.2 (parser) | workspace | Python |
| `STEP_07_1/{validator,vocabulary,levenshtein,test_validator}.py` | T-03.1.1 | T-03.2, T-04.3 (shares Levenshtein) | workspace | Python |
| `STEP_08/{parser,cst,test_parser}.py` | T-03.2 | T-03.3 (typer) | workspace | Python |
| `STEP_09/{typer,types,atom_catalog,test_typer}.py` | T-03.3 | T-03.4 (classifier) | workspace | Python |
| `STEP_10/{classifier,ast,column_analysis,test_classifier}.py` | T-03.4 | T-03.5 (solver), T-04.2 (optimizer) | workspace | Python |
| `STEP_11/{solver,constraints,execution_plan,test_solver}.py` | T-03.5 | T-04.1, T-04.2, T-04.4, METHODOLOGY_INTEGRATOR | workspace | Python |
| **Phase 4 — RUNTIME** | | | | |
| `BOSS_LEVEL_1/{executor,execution_context,atom_executors,trace,test_executor,e2e_test}.py` | T-04.1 | T-04.4 (debugger), METHODOLOGY_INTEGRATOR | workspace | Python |
| `BOSS_LEVEL_2/{optimizer,optimization_rules,cost_model,test_optimizer,correctness_suite}.py` | T-04.2 | METHODOLOGY_INTEGRATOR | workspace | Python |
| `BOSS_LEVEL_3/{error_reporter,unified_error,rendering,fuzzy_match,test_error_reporter}.py` | T-04.3 | T-04.1, T-04.2, T-04.4 (all stages emit unified Error) | workspace | Python |
| `BOSS_LEVEL_4/{debugger,debug_session,snapshot,visualization,commands,test_debugger,demo_session}.py` | T-04.4 | METHODOLOGY_INTEGRATOR (optional) | workspace | Python |
| **Methodology Integration** | | | | |
| `METHODOLOGY_REPORT.md` | METHODOLOGY_INTEGRATOR | (terminal output to human) | methodology | Markdown |
| `SHUFFLE_TEST_RESULTS.json` | METHODOLOGY_INTEGRATOR | (terminal output) | methodology | JSON |
| `COMPRESSION_REPORT.md` | METHODOLOGY_INTEGRATOR | (terminal output) | methodology | Markdown |
| `E2E_DEMO_OUTPUT.md` | METHODOLOGY_INTEGRATOR | (terminal output) | methodology | Markdown |

---

## `workspace_manifest.json` schema

Single source of truth for what-is-where in the workspace. Append-only after task PASS.

```json
{
  "schema_version": "1.0.0",
  "workflow": "LANG_DEV_V2",
  "workflow_version": "2.0.0-DRAFT",
  "engagement": {
    "started_at": "ISO 8601 timestamp",
    "target_library": "absolute path",
    "nexus_reports_dir": "absolute path",
    "step_source_dir": "absolute path",
    "workspace_dir": "absolute path"
  },
  "phases": [
    {
      "phase": 1,
      "name": "DECONSTRUCTION",
      "status": "complete | in_progress | pending | hold",
      "started_at": "ISO timestamp",
      "completed_at": "ISO timestamp (when complete)",
      "tasks": [
        {
          "task_id": "T-01.1",
          "status": "pass | fail_retry | fail_escalate | skip_by_design",
          "attempts": 1,
          "started_at": "ISO timestamp",
          "completed_at": "ISO timestamp",
          "outputs": [
            {"path": "STEP_01/primitives_catalog.json", "sha256": "..."},
            {"path": "STEP_01/tier_compression.md", "sha256": "..."}
          ],
          "qa_verdict_log": ["TASK_PASS at <ISO> by PHASE_QA"]
        }
      ]
    }
  ],
  "methodology_integration": {
    "status": "pending | running | green_light | incomplete",
    "started_at": "ISO timestamp (when run)",
    "completed_at": "ISO timestamp",
    "results": {
      "shuffle_test": {"iterations": 0, "passed": 0, "failed": 0},
      "compression_ratio": {"main_tier_count": 0, "library_api_count": 0, "ratio": 0.0},
      "e2e_demo": {"input": "<source string>", "output_summary": "...", "status": "pass | fail"}
    }
  }
}
```

**Update discipline:**
- Initialized empty by QUEEN at PRESTEP
- Each PHASE_EXECUTOR appends its outputs to the corresponding task entry on TASK_PASS
- QUEEN updates phase status when all phase tasks reach TASK_PASS
- METHODOLOGY_INTEGRATOR populates `methodology_integration` section
- Never rewrites prior entries; append + status-flip only

---

## Cross-phase consumers (the dependency graph)

```
Phase 1 outputs ──→ Phase 2 (T-02.1)
                         │
                         ▼
                  <library>_decisions.json [AUTHORITATIVE]
                         │
                         ├──→ Phase 3 every stage
                         │
                         └──→ Phase 4 (T-04.1 Executor registry)

Phase 1 outputs ──→ METHODOLOGY_INTEGRATOR (compression check from primitives_catalog)
Phase 2 outputs ──→ METHODOLOGY_INTEGRATOR (decisions.json validity)
Phase 3 outputs ──→ METHODOLOGY_INTEGRATOR (full pipeline; shuffle test)
Phase 4 outputs ──→ METHODOLOGY_INTEGRATOR (Executor for e2e demo, Optimizer for golden-rule check)
```

---

## File-naming conventions (binding)

- Workspace subdirs use `STEP_NN` or `STEP_NN_M` for per-step artifacts (`STEP_01`, `STEP_06_1`)
- Workspace subdirs use `BOSS_LEVEL_N` for runtime stages (`BOSS_LEVEL_1` through `BOSS_LEVEL_4`)
- The authoritative decisions file is `<library>_decisions.json` at workspace ROOT (not under `STEP_05A/`) because every Phase 3 stage consumes it directly
- Test files use `test_<stage>.py` (pytest discovery)
- E2E test scripts are `e2e_test.py` per stage that has one (T-04.1)
- Demos are `demo_session.py` (T-04.4)

---

## Path resolution rules (for `LANG_DEV_V2_WORKFLOW.json`)

Every `output_contract` reference in the workflow JSON resolves to a file under `workflows/LANG_DEV_V2/contracts/PHASE_<N>_CONTRACT.md`.

Every `source_docs` reference resolves to `<step_source_dir>/<filename>`.

Every artifact path in this catalog resolves to `<workspace_dir>/<path>` (paths are workspace-relative).

---

*End of ARTIFACT_CATALOG.*
