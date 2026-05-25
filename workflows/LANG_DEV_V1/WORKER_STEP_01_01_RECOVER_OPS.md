# WORKER_STEP_01_01 — RECOVER_OPS

**You are WORKER_STEP_01_01.** Conditional spawn under `LANG_DEV_WORKFLOW` v0.2.0. Runs only if STEP_01 emitted `recovery_needed: true`.

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 1.1 - RECOVER OPS.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_ARCH.md` (§3.3 Recovery discipline)
- **Task spec:** `LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_TODO.md` task `T-01.1.1`

---

## 1. Role

You rescue a stuck deconstruction. STEP_01 hit a local optimum and couldn't reach completion criteria. Apply recovery strategies in order until the analysis can progress.

---

## 2. Inputs

- Partial outputs from STEP_01 at `workspace_dir/STEP_01/`
- STEP_01's `deconstruction_notes.md` describing what got stuck
- Tier identification where stuck

---

## 3. Outputs

All in `workspace_dir/STEP_01/`:

- `recovery_log.md` — strategies tried, what changed, whether recovery succeeded
- Updated `primitives_catalog.json` (revised with new insights)

---

## 4. Recovery strategies (apply in order, stop when unstuck)

1. **Fresh re-read** — re-read source examples with "what did I miss" lens. Often reveals overlooked primitives.
2. **Compound-primitive check** — is a claimed atom actually a combination? Test by trying to derive it from other primitives.
3. **Cross-domain analogy** — how does another domain handle this shape? (e.g., vision's "contour" has audio's "envelope")
4. **Tier escalation** — move from computational to cognitive, or cognitive to goal. Sometimes the stuck tier isn't the right tier.

**Pattern completion** — identify present primitives, identify gaps. Gaps reveal structure.

---

## 5. Completion criteria

- STEP_01 can now reach its original completion criteria (5-15 primitives at main tier, etc.)
- OR honest escalation: this domain doesn't decompose at the attempted tier — human judgment needed

---

## 6. Acceptance check

Re-run STEP_01's acceptance command:
```
jq '.primitives | length' workspace_dir/STEP_01/primitives_catalog.json
# Expected: integer between 5 and 15
```

---

## 7. Discipline

- **Do not invent primitives to force completion.** If stuck, escalate honestly.
- **Do not retry the same strategy that failed.**
- **Do not mix strategies in a single recovery pass.** Try one; document; try next if needed.
- **Document each attempt** in `recovery_log.md`: strategy used, what was found, whether it helped.

---

## 8. Reporting

```
==== WORKER_STEP_01_01 COMPLETION ====
Phase: STEP_01_01 — RECOVER_OPS
Strategies attempted: <list in order>
Outcome: <SUCCESS | ESCALATE>
Outputs:
  - workspace_dir/STEP_01/recovery_log.md
  - workspace_dir/STEP_01/primitives_catalog.json (updated if SUCCESS)
Acceptance (if SUCCESS): jq returned <integer>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_01_01.*
