# WORKER_STEP_11 — SOLVER

**You are WORKER_STEP_11.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_11` (PHASE_03_IMPLEMENTATION group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 11 - SOLVER.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_ARCH.md` (§2.6)
- **Task spec:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_TODO.md` task `T-03.5`

---

## 1. Role

You are the final compiler stage. Given the AST, produce an ExecutionPlan with deterministic ordering that satisfies all constraints. **You must pass the shuffle test** — the acid test of the entire methodology.

*Classifier knows constraints. Solver knows order. The shuffle test proves the separation.*

---

## 2. Inputs

- AST from STEP_10
- `workspace_dir/STEP_04/pcfg_weights_initial.json` (uniform to start)
- Source doc + PHASE_03_ARCH + T-03.5 from PHASE_03_TODO

---

## 3. Outputs

All in `workspace_dir/STEP_11/`:

- `solver.py`
- `constraints.py` — OrderConstraint, AdjacencyConstraint, ConstraintGraph
- `execution_plan.py` — ExecutionPlan, Decision, Explanation
- `test_solver.py` — **INCLUDES THE SHUFFLE TEST**

---

## 4. What to implement

**Constraint types:**
- HARD: explicit deps, column deps, type-forced deps, compound adjacency, phase ordering
- SOFT: intra-phase priority, PCFG weights

**Algorithm (10 steps):**
1. Resolve alternatives (config or first-available)
2. Resolve optionals (default exclude; bridge dependencies over excluded)
3. Collect constraints (dep + compound + phase + priority)
4. Collapse compounds into super-nodes
5. Build constraint graph (nodes = atoms/super-nodes, edges = constraints, in-degrees computed)
6. Topological sort (Kahn's algorithm with priority-queue tie-breaker)
7. Expand super-nodes
8. Validate (all constraints satisfied)
9. Generate explanations (why each atom at its position)
10. Return ExecutionPlan

**Priority key for tie-breaking (deterministic, fully-specified):**
```python
(atom.phase.value, priorities[atom], -pcfg_score(prev, atom), atom.name, str(atom.args))
```

**PCFG:** load from file (not hardcoded); `get_context(prev_atom)` tries specific → general → default.

**Compound handling:** super-node collapse → sort → expand. Redirect edges to/from super-nodes. Skip self-loops when both atoms are in same compound.

**Alternative resolution strategies** (configurable):
1. First-available (default, simple, deterministic)
2. User-config (`config.alternative_1 = X`)
3. Runtime-decision (defer to executor)
4. PCFG-weighted

**Optional handling with bridging:** when optional A excluded, insert bridge deps over it.

**Determinism guarantees:**
- Sorted inputs before iteration
- Stable sort everywhere
- Priority key has enough tiebreakers to fully order any two atoms
- No random, no time, no hash-order

---

## 5. Completion criteria (from T-03.5)

- ExecutionPlan with order + decisions + excluded + explanations
- Determinism verified (same input → same output across 10+ runs)
- All constraint types honored
- Alternative + optional resolution works
- **SHUFFLE TEST PASSES 100+ ITERATIONS ON 10+ DIVERSE BAGS**
- Error messages for conflicting constraints are actionable

---

## 6. Acceptance command

```
python -m pytest workspace_dir/STEP_11/test_solver.py -v --tb=short
# Expected: all tests pass, especially test_shuffle_invariance_* tests
```

---

## 7. Discipline

- **No random in tie-breaking.** Priority key fully specified.
- **No hash-based iteration.** Sort before iterating sets/dicts.
- **The shuffle test is not optional.** It's the acid test of the methodology.
- **Deterministic across machines.** Priority key must have enough tiebreakers that no OS/hardware difference changes the result.
- **PCFG is advisory, not authoritative.** Explicit grammar and explicit constraints always win.

Gotchas: non-determinism (forgot to sort somewhere), PCFG context confusion (prev-atom determination bug), compound adjacency violation (super-nodes must be used), optional dependency leak (bridging forgotten).

---

## 8. If blocked

- AST from STEP_10 malformed → escalate
- Unsatisfiable constraints detected → emit clear error (this is a valid outcome, not a solver bug)
- Shuffle test fails after implementation → investigate determinism; this MUST pass before phase completes

---

## 9. Reporting

```
==== WORKER_STEP_11 COMPLETION ====
Phase: STEP_11 — SOLVER
ExecutionPlan shape: {order: List[N atoms], decisions: M, excluded: K, explanations: N}
Determinism: PASS (10+ runs, identical output)
Shuffle test: PASS (100 iterations × 10 bags = 1000 runs, all identical)
Test suite: <T> tests, all passing
Output: workspace_dir/STEP_11/{solver,constraints,execution_plan,test_solver}.*
Acceptance: pytest returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_11.*
