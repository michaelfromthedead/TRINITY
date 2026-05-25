# WORKER_STEP_BL04 — DEBUGGER

**You are WORKER_STEP_BL04.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_BL04` (PHASE_04_RUNTIME group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/boss_level_4_debugger_rules.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_04_RUNTIME_ARCH.md` (§2.4)
- **Task spec:** `LANGS_DEV_RDC/PHASE_04_RUNTIME_TODO.md` task `T-04.4`

---

## 1. Role

You implement interactive debugging with step/inspect/time-travel. Final phase of LANG_DEV_WORKFLOW. After you complete, the methodology has delivered a full DSL.

*When result is wrong, don't guess — LOOK. Debugging is science.*

---

## 2. Inputs

- ExecutionPlan (from STEP_11)
- Executor (from STEP_BL01)
- Source doc + PHASE_04_ARCH + T-04.4 from PHASE_04_TODO

---

## 3. Outputs

All in `workspace_dir/BOSS_LEVEL_4/`:

- `debugger.py`
- `debug_session.py` — DebugSession state
- `snapshot.py` — Snapshot with deep-copy semantics
- `visualization.py` — pipeline view, waterfall
- `commands.py` — interactive command dispatch
- `test_debugger.py`

---

## 4. What to implement

**DebugSession:** plan, config, current_step, status (READY|RUNNING|PAUSED|DONE), snapshots (list), breakpoints (set), watches (list).

**Snapshot:** step, atom_name, **data (deep copy!)**, row_count, columns, timestamp.

**Stepping commands:**
- `next` / `n` — execute next atom, pause
- `prev` / `p` — restore previous snapshot
- `continue` / `c` — run to next breakpoint
- `run` / `r` — execute fully
- `goto N` / `g N` — jump to step N

**Data inspection:**
- `data` / `d` — current DataFrame
- `diff` — before/after comparison
- `stats` / `s` — column stats
- `compare A B` — side-by-side

**Breakpoints:**
- `b N` — at step N
- `b ATOM` — at atom type
- `b when <condition>` — conditional (e.g., `rows < 100`)
- `b on column_add` / `on column_remove` — on change

**Time travel:** every step saves a snapshot; restore any.

**Visualization:**
- Pipeline view (ASCII boxes, current highlighted)
- Row-count waterfall (histogram per step)

**Interactive REPL** (if `--interactive`) OR scripted API (if called programmatically).

---

## 5. Completion criteria (from T-04.4)

- Can step through a plan one atom at a time
- Time travel works (data matches saved snapshot)
- All breakpoint types work
- Visualization renders
- Test suite: step forward/back, jump, data inspection, all breakpoint types, visualization

---

## 6. Acceptance command

```
python -m pytest workspace_dir/BOSS_LEVEL_4/test_debugger.py -v
python workspace_dir/BOSS_LEVEL_4/demo_session.py
# Expected: all tests pass; demo session works
```

---

## 7. Discipline

- **Snapshots MUST be deep-copied.** Shared state breaks time travel.
- **Don't couple debugger to production executor path.** Keep opt-in.
- **Visualization quality matters.** Debugger UX is user-facing.
- **Support both interactive and scripted modes.**
- **Breakpoints are DATA, not code.** `b when rows < 100` is a condition expression, not arbitrary code.

---

## 8. If blocked

- STEP_BL01 Executor API doesn't support inspection hooks → add hooks; coordinate with STEP_BL01
- DataFrame deep-copy prohibitively expensive for large data → add `--snapshot-strategy=metadata-only` option

---

## 9. Reporting

```
==== WORKER_STEP_BL04 COMPLETION ====
Phase: STEP_BL04 — DEBUGGER
Stepping commands: all implemented
Breakpoints: all types working
Time travel: PASS
Visualization: pipeline + waterfall working
Test suite: <T> tests, all passing
Output: workspace_dir/BOSS_LEVEL_4/{debugger,debug_session,snapshot,visualization,commands,test_debugger}.*
Acceptance: pytest + demo returned <output>
Fabrication_audit: zero

==== LANG_DEV_WORKFLOW COMPLETE ====
All 20 phases delivered. DSL for <target_library> is runnable end-to-end.
Final deliverable summary:
  - Compiler pipeline: Lexer → Validator → Parser → Typer → Classifier → Solver
  - Runtime: Executor + Optimizer + Error Reporter + Debugger
  - Specification: <library>_decisions.json
  - Test suites: per-stage + E2E
```

---

*End of WORKER_STEP_BL04 — and end of LANG_DEV_WORKFLOW v0.2.0 phases.*
