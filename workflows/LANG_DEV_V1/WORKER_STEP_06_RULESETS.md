# WORKER_STEP_06 — RULESETS

**You are WORKER_STEP_06.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_06` (PHASE_02_DESIGN group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 6 - RULESETS AND DEFAULTS.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_02_DESIGN_ARCH.md` (§2.3)
- **Task spec:** `LANGS_DEV_RDC/PHASE_02_DESIGN_TODO.md` task `T-02.4`

---

## 1. Role

You fill in the 10 default categories that encode domain expertise so users don't have to write grammar. Defaults are gravity; grammar (from STEP_05B) is escape velocity. Good defaults handle 90% of cases invisibly.

---

## 2. Inputs

- `workspace_dir/<library>_decisions.json` (atoms, phases, port types)
- `workspace_dir/STEP_05B/bag_grammar_spec.md` (operators + precedence)
- Source doc + PHASE_02_ARCH + T-02.4 from PHASE_02_TODO

---

## 3. Outputs

- `workspace_dir/STEP_06/ruleset_spec.md`

---

## 4. The 10 categories to fill

For each: pick recommendation + alternatives-considered + rationale + override mechanism.

1. **Phase assignment** — how do new atoms get assigned phases? (explicit / name-based / type-based / registration-with-default)
2. **Intra-phase ordering** — what determines order within a phase? (declaration / alphabetical / learned-PCFG / specificity / dependency-analysis)
3. **Implicit compounds** — which atoms auto-group? (none / registered-list / type-exclusive-pairs)
4. **Implicit sequences** — which orderings are auto-enforced? (phase-based / dependency-based / registered / none)
5. **Type coercion** — near-matches: strict / auto / subtype-hierarchy / explicit-coercion-points
6. **Failure behavior** — hard-fail / best-effort / interactive / fail-with-suggestions
7. **Ambiguity resolution** — first-valid / learned / canonical / configurable
8. **Optional inclusion** — default-exclude / default-include / context-dependent / config-controlled
9. **Repetition bounds** — unbounded / practical-bounds / domain-specific
10. **Parallel execution** — never / always / explicit-only / phase-based

**Default hierarchy** (priority, top = highest):
1. Explicit grammar (user's word)
2. Detected dependencies (column / type-forced)
3. Intra-phase priorities
4. Phase ordering
5. Type compatibility
6. Alphabetical (tiebreaker)

**90/10 rule:** defaults should cover ≥90% of cases. State your domain's target.

---

## 5. Completion criteria (from T-02.4)

- All 10 categories filled with recommendation + rationale
- Alternatives documented per category
- Override mechanism specified per category
- Default hierarchy priority order documented
- 90/10 target stated

---

## 6. Discipline

- **Predictable > optimal.** A default users can predict beats a default that's optimal-but-surprising.
- **Strong opinions, weakly held.** Pick one, stick to it, document it. Override available when needed.
- **Avoid overly-clever defaults.** "Our system auto-detects optimal ordering with ML!" is unpredictable.
- **Avoid too many defaults.** ~10 rules, not 47.
- **Resolve conflicts via priority order.** Never allow two defaults to conflict ambiguously.
- **Document EVERY default so users can form mental models.**

Gotchas: overly-clever-defaults, too-many-defaults, conflicting-defaults, hidden-defaults, unchangeable-defaults.

Sub-tasks: T-02.4.1 (CONUNDRUM, next phase STEP_06_01) and T-02.4.2 (PRE_LEXER, next phase STEP_06_02). Those continue design; your job is the core 10 categories.

---

## 7. If blocked

- STEP_05A/05B outputs missing → escalate
- Category genuinely has no good default for this domain → document as "DEFERRED — user must specify" with override mechanism

---

## 8. Reporting

```
==== WORKER_STEP_06 COMPLETION ====
Phase: STEP_06 — RULESETS
Categories filled: 10 of 10
Default hierarchy: 6 levels documented
90/10 target: <your domain's estimate>
Output: workspace_dir/STEP_06/ruleset_spec.md
Fabrication_audit: zero
```

---

*End of WORKER_STEP_06.*
