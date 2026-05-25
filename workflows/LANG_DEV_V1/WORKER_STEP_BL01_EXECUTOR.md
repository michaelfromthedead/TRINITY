# WORKER_STEP_BL01 — EXECUTOR

**You are WORKER_STEP_BL01.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_BL01` (PHASE_04_RUNTIME group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/boss_level_1_executor_rules.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_04_RUNTIME_ARCH.md` (§2.1)
- **Task spec:** `LANGS_DEV_RDC/PHASE_04_RUNTIME_TODO.md` task `T-04.1`

---

## 1. Role

You implement the runtime that consumes ExecutionPlans from STEP_11 SOLVER and runs them against actual library calls. Plan = bytecode. Executor = virtual machine. Library = hardware.

---

## 2. Inputs

- ExecutionPlan from STEP_11
- `workspace_dir/<library>_decisions.json` (atom catalog)
- Runtime config (file paths, flags)
- Source doc + PHASE_04_ARCH + T-04.1 from PHASE_04_TODO

---

## 3. Outputs

All in `workspace_dir/BOSS_LEVEL_1/`:

- `executor.py`
- `execution_context.py` — ExecutionContext, ExecutionResult
- `atom_executors.py` — registry: `EXECUTORS = {"LoadCSV": execute_LoadCSV, ...}`
- `trace.py` — TraceEntry + pretty-printer
- `test_executor.py`

---

## 4. What to implement

**ExecutionContext fields:** `current_df` (DataFrame?), `current_grouped` (GroupBy?), `config` (Dict), `trace` (List[TraceEntry]), `errors` (List[Error]).

**Core execution loop:**
```python
ctx = ExecutionContext(config)
for atom in plan.order:
    executor_fn = EXECUTORS[atom.name]
    executor_fn(atom, ctx)
return ExecutionResult(ctx.data, ctx.trace)
```

**Per-atom executor pattern:**
```python
def execute_ATOMNAME(atom, ctx):
    param1 = get_string_arg(atom, 0)
    param2 = get_number_arg(atom, 1, default=5)
    df = ctx.get_data()
    result = df.some_pandas_method(param1, param2)
    ctx.set_data(result)
```

**Executor coverage:** every atom in decisions.json has an executor function. Organize by phase (source, transform, group, aggregate, limit, sink).

**Argument extraction helpers:** `get_string_arg`, `get_number_arg`, `get_bool_arg`, `get_list_arg`, `get_dict_arg` with optional defaults.

**TraceEntry:** atom_name, rows_before, rows_after, columns, time_ms, notes.

**Error handling:** wrap each atom executor in try/except. Errors wrap into ExecutionError with step + atom + context.

**Empty DataFrame:** warning, not error (pipeline continues).

**Pretty-printed trace** (row-count waterfall):
```
STEP  ATOM        ROWS     COLS   TIME    NOTES
 1    LoadCSV  →  1000       5    45ms    Loaded sales.csv
 2    Filter   →   847       5     8ms    -15.3%
...
```

---

## 5. Completion criteria (from T-04.1)

- Every atom in decisions.json has an executor function
- Execution of valid ExecutionPlan produces real library output
- Trace is complete
- Error handling works (file-not-found, column-not-found, expression errors)
- Integration test: full E2E from ExecutionPlan → actual DataFrame

---

## 6. Acceptance command

```
python -m pytest workspace_dir/BOSS_LEVEL_1/test_executor.py -v
python workspace_dir/BOSS_LEVEL_1/e2e_test.py
# Expected: all tests pass; E2E produces correct DataFrame
```

---

## 7. Discipline

- **Executor registry must have NO GAPS.** Every decisions.json atom implemented.
- **Don't skip trace collection.**
- **Don't let errors crash silently.** Always catch and wrap with context.
- **Don't mutate the ExecutionPlan** — it's immutable input.
- **Keep executors stateless** across invocations.
- **Grouped state transitions** — GroupBy produces grouped; Agg consumes it; clear after aggregation.

---

## 8. If blocked

- STEP_11 ExecutionPlan malformed → escalate
- Atom in decisions.json but no way to implement (e.g., missing library method) → escalate with specific atom name

---

## 9. Reporting

```
==== WORKER_STEP_BL01 COMPLETION ====
Phase: STEP_BL01 — EXECUTOR
Atoms implemented: <N> of <M in decisions.json>
Registry complete: <true | false>
Test suite: <T> tests, all passing
E2E integration: PASS
Output: workspace_dir/BOSS_LEVEL_1/{executor,execution_context,atom_executors,trace,test_executor}.*
Acceptance: pytest + e2e returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_BL01.*
