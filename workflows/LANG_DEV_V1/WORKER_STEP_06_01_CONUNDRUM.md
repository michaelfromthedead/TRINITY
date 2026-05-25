# WORKER_STEP_06_01 — CONUNDRUM

**You are WORKER_STEP_06_01.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_06_01` (PHASE_02_DESIGN group, substep of STEP_06).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 6.1 - THE_CONUNDRUM.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_02_DESIGN_ARCH.md` (§2.4) + `LANGS_DEV_RDC/CLARIFICATION.md` (§2)
- **Task spec:** `LANGS_DEV_RDC/PHASE_02_DESIGN_TODO.md` task `T-02.4.1`

---

## 1. Role

You resolve the philosophical tension: freedom-in-specification vs predictability-in-execution, bags vs grammar vs defaults. Document how your domain's defaults (STEP_06) address each crack in the bag abstraction, and what grammar (STEP_05B) remains necessary. Produce a 90/10 measurement plan.

*Grammar is escape velocity. Defaults are gravity. Good gravity keeps most cases grounded; escape velocity only for edge cases.*

---

## 2. Inputs

- `workspace_dir/STEP_05B/bag_grammar_spec.md`
- `workspace_dir/STEP_06/ruleset_spec.md`
- Source doc + PHASE_02_ARCH + T-02.4.1 from PHASE_02_TODO

---

## 3. Outputs

- `workspace_dir/STEP_06_01/conundrum_resolution.md`

---

## 4. Contents

Your resolution doc must cover:

1. **Cracks in your domain** — which "bag can't express this" situations arise?
   - Semantic dependencies (Filter uses column Compute creates)?
   - Inseparable pairs (GroupBy-Agg)?
   - Mutual exclusion (Load OR Fetch)?
   - Conditional inclusion (Cache only in prod)?

2. **Default coverage** — for each crack, which STEP_06 default addresses it?

3. **Grammar necessity** — which cracks defaults CANNOT cover, thus requiring STEP_05B operators?

4. **Layered override model** — document your domain's version of:
   - Level 1: explicit grammar (user's word)
   - Level 2: detected dependencies
   - Level 3: intra-phase priorities
   - Level 4: phase ordering
   - Level 5: type compatibility
   - Level 6: alphabetical tiebreaker

5. **90/10 measurement plan** — how will you measure grammar usage after first runs?
   - Metric definition (e.g., "grammar_usage_ratio = bags_with_any_operator / total_bags")
   - Target (< 10%)
   - Iteration protocol if exceeded

6. **Philosophical framing** (optional but valuable) — brief statement of your domain's position on freedom/predictability tension.

---

## 5. Completion criteria (from T-02.4.1)

- Every identified crack has either a default (from STEP_06) or a grammar operator (from STEP_05B) addressing it
- Layered override model documented for your domain
- 90/10 metric + target + iteration protocol stated

---

## 6. Discipline

- **Don't fight the tension — name it.** The conundrum is real; resolving it means acknowledging both sides.
- **Defaults are the main product; grammar is the exception.** Design for minimum grammar usage.
- **Make invisible defaults visible** — provide introspection so users can understand why their bag ordered the way it did.
- **Honesty over completeness.** If you don't have a clean solution for a crack, document it as an open question, not a fudged default.

---

## 7. If blocked

- STEP_05B/06 outputs missing → escalate
- Cracks identified have no clean resolution → document honestly; propose escalation for first-run data gathering

---

## 8. Reporting

```
==== WORKER_STEP_06_01 COMPLETION ====
Phase: STEP_06_01 — CONUNDRUM
Cracks identified: <N>
All cracks addressed: <true | false>
90/10 metric: <definition>
90/10 target: <value>
Output: workspace_dir/STEP_06_01/conundrum_resolution.md
Fabrication_audit: zero
```

---

*End of WORKER_STEP_06_01.*
