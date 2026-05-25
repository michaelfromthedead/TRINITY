# WORKER_STEP_BL02 — OPTIMIZER

**You are WORKER_STEP_BL02.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_BL02` (PHASE_04_RUNTIME group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/boss_level_2_optimizer_rules.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_04_RUNTIME_ARCH.md` (§2.2)
- **Task spec:** `LANGS_DEV_RDC/PHASE_04_RUNTIME_TODO.md` task `T-04.2`

---

## 1. Role

You implement plan rewrites for performance. **Preserve correctness absolutely.** The golden rule: `optimized_output == original_output, ALWAYS.`

*Same input, better performance. Filter early. Sort late. Combine operations.*

---

## 2. Inputs

- ExecutionPlan from STEP_11
- AST dependencies from STEP_10 (for column-dependency respect)
- Source doc + PHASE_04_ARCH + T-04.2 from PHASE_04_TODO

---

## 3. Outputs

All in `workspace_dir/BOSS_LEVEL_2/`:

- `optimizer.py`
- `optimization_rules.py` — per-rule implementations
- `cost_model.py` — row-count estimation
- `test_optimizer.py` — correctness-preservation suite

---

## 4. What to implement

**5 basic optimization rules:**

1. **Filter pushdown** — move filter before expensive ops, respecting column deps
2. **Filter combination** — `Filter("a>0"), Filter("b<10")` → `Filter("(a>0) and (b<10)")`
3. **Sort pushdown (LATE)** — sort after filters reduce data, NOT past Head/Tail
4. **Compute combination** — multiple Computes → ComputeMulti
5. **Dead code elimination** — computed columns never used → remove

**OptimizationRule interface:**
- `apply(plan)` — transform
- `guard(plan)` — check applicability
- `preserves_correctness` — invariant claim

**OptimizationPass:** sequence of rules; fixed-point iteration OR bounded passes.

**OptimizedPlan:** original plan + rewrite log.

**Cost model:** estimate rows through pipeline (row_count waterfall); estimate per-atom cost; calculate speedup.

**Disable-all-optimizations flag** (escape hatch): `--no-optimize` skips optimizer entirely.

---

## 5. Completion criteria (from T-04.2)

- All 5 basic rules implemented
- **Golden rule verified:** for 20+ diverse plans, `optimized.execute() == original.execute()`
- Speedup measurable on at least one benchmark pipeline
- Disable-optimizations escape hatch present
- Test suite covers: each rule in isolation, rule interaction, correctness preservation, counterexamples (rule should NOT apply)

---

## 6. Acceptance command

```
python -m pytest workspace_dir/BOSS_LEVEL_2/test_optimizer.py -v
python workspace_dir/BOSS_LEVEL_2/correctness_suite.py
# Expected: all tests pass; correctness suite passes on 20+ plans
```

---

## 7. Discipline

- **The golden rule is non-negotiable.** Optimized output must equal original output, always.
- **Every rule has a correctness-preservation proof or test.**
- **Don't push Sort past Head/Tail.** Semantic violation (changes the top-N semantic).
- **Respect column dependencies.** Filter using column `z` cannot push past Compute that creates `z`.
- **Keep the escape hatch.** `--no-optimize` must work.
- **Cost model is ADVISORY** — even if cost says "this is slower," keep the plan if user disabled optimization.

---

## 8. If blocked

- STEP_11 ExecutionPlan incompatible with rewrites → escalate
- Correctness suite fails on any plan → STOP, fix rule or remove it. Don't ship an unsound optimizer.

---

## 9. Reporting

```
==== WORKER_STEP_BL02 COMPLETION ====
Phase: STEP_BL02 — OPTIMIZER
Rules implemented: 5 of 5 (filter-pushdown, filter-combination, sort-pushdown, compute-combination, dead-code-elimination)
Correctness suite: 20/20 plans pass
Speedup benchmark: <N%> on <pipeline>
Test suite: <T> tests, all passing
Output: workspace_dir/BOSS_LEVEL_2/{optimizer,optimization_rules,cost_model,test_optimizer}.*
Acceptance: pytest + correctness_suite returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_BL02.*
