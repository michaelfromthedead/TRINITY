# SHORT_RDC_SPEC — Compressed Relevance-Check Sub-Procedure

**Version:** v0.1.0
**Status:** DEFERRED — not implemented in v0.1.0
**Resolves:** TBD §16.1 (SHORT_RDC sub-procedure shape)
**Flagged for follow-up:** T3.3-FOLLOWUP
**Authoritative spec:** `ORGANIZE_WORKFLOW.json` §`relationship_to_other_workflows.rdc_coupling`

---

## 1. Context

TRIAGE classifies files in a MAINTENANCE circuit using the project's active rules, the file's content, and its structural position. For most file types — source code, scripts, configs — mechanical rule-matching and content reading are sufficient for a confident verdict.

For **prose files** (primarily `.md`), a different class of question arises: not "does this file match a rule?" but "is this content still relevant to the project?" A markdown file might be:

- An active architecture decision record (KEEP_IN_PLACE)
- A superseded design note (QUARANTINE:.archive)
- A historical thread that has been absorbed into MASTER.md (QUARANTINE:.archive)
- An active working draft that belongs in `docs/` (MOVE_TO)

The problem is that TRIAGE's full-read judgment may not have enough project context to distinguish "superseded but intellectually valuable" from "still active but temporarily misplaced." A compressed RDC-style relevance check — SHORT_RDC — would give TRIAGE a structured way to answer this question.

---

## 2. v0.1.0 Decision: DEFERRED

**SHORT_RDC is not implemented in v0.1.0.**

### 2.1 Rationale for deferral

Adding SHORT_RDC now would introduce coupling to RDC_WORKFLOW infrastructure (or a standalone relevance-checking sub-procedure) that is not empirically justified. Specifically:

1. **We do not yet know how often TRIAGE gets prose-archival decisions wrong.** The problem SHORT_RDC solves — false-positive archival of still-relevant prose files — may be rare in practice if TRIAGE's full-read judgment and ASK_USER escalation work well.

2. **The cost of getting it wrong is low.** Unlike code moves, prose quarantine to `.archive/` is reversible and the file is preserved. The user can restore a false-positive archival at any time.

3. **Building SHORT_RDC before observing failures would be speculative infrastructure.** It adds complexity and coupling (to RDC or a standalone worker) before we know it is needed.

4. **The ASK_USER escape hatch is a valid interim solution.** When TRIAGE cannot confidently classify a prose file, it emits `ASK_USER`. This puts the relevance judgment where it belongs — with the user — without requiring a sub-procedure.

Per `ORGANIZE_WORKFLOW.json` §`relationship_to_other_workflows.rdc_coupling`:
> "SHORT_RDC is NOT full RDC_WORKFLOW — it is a compressed per-file relevance check (TBD: exact shape; v0.1.0-DRAFT defers this to the TRIAGE worker's own judgment, with full RDC as an escape hatch the user can invoke manually)."

### 2.2 v0.1.0 Current Approach

TRIAGE handles prose file classification through two mechanisms:

1. **Full-read judgment:** TRIAGE reads the prose file fully and uses its content plus the project's active rules to emit a verdict. For prose with clear markers (e.g., "DEPRECATED", "superseded by", dated headers with stale dates), this is sufficient.

2. **ASK_USER for ambiguity:** When TRIAGE cannot confidently distinguish KEEP_IN_PLACE from QUARANTINE:.archive (e.g., a design note with no staleness markers but also no active references), it emits `ASK_USER`. QUEEN surfaces this to the user during ratification. The user makes the relevance judgment.

This approach is conservative by design: it prefers asking over guessing.

---

## 3. Future Activation Criteria

SHORT_RDC should be introduced when observational data satisfies **both** of the following conditions:

**Condition A — Sufficient observation volume:**  
≥ 10 completed MAINTENANCE runs (across any combination of projects) have been recorded in `.organize.json` run logs. This provides a meaningful sample size.

**Condition B — False-positive archival rate:**  
≥ 5 prose files that were QUARANTINE:.archive verdicts have subsequently been restored by the user (moved back out of `.archive/` to their canonical location). This indicates that TRIAGE's judgment on prose relevance is producing actionable errors at a rate that justifies investment.

If both conditions are met, SHORT_RDC should be designed and implemented as a dedicated lightweight worker (see §4).

**Monitoring note:** There is currently no automated tracking of `.archive/` restoration events. Until such tracking is implemented, the T3.3-FOLLOWUP flag serves as a reminder to check manually after significant usage.

---

## 4. Proposed Future Design (for implementation when criteria are met)

The following is a design sketch for SHORT_RDC. It is NOT implemented in v0.1.0 and may change before implementation.

### 4.1 Role

SHORT_RDC is a lightweight relevance-checking worker spawned by QUEEN when TRIAGE has flagged a prose file as a candidate for archival but cannot confidently distinguish archival from preservation.

### 4.2 Invocation condition

TRIAGE flags a file as `candidate-for-SHORT_RDC` instead of directly emitting `QUARANTINE:.archive`, when:
- The file is a prose file (`.md` or similar)
- The file has no mechanical staleness markers (no "DEPRECATED", no superseded-by reference, no dated header with a stale date)
- TRIAGE's confidence on the archival verdict would be MEDIUM or LOW

### 4.3 Inputs

- `file_path` — absolute path to the prose file candidate
- `project_context` — one of:
  - Path to `MASTER.md` (if RDC_WORKFLOW has been run on this project)
  - Fallback: path to `README.md` + any `ARCHITECTURE.md` or `ARCH.md` at project root

### 4.4 Outputs

SHORT_RDC returns a structured judgment:

```json
{
  "file": "<absolute path>",
  "relevance_score": 0.0,
  "relevance_rationale": "<2-3 sentences explaining the score>",
  "recommendation": "KEEP_IN_PLACE | QUARANTINE:.archive | ASK_USER",
  "evidence": "<specific passage in file or project_context that supports the recommendation>"
}
```

`relevance_score` is a continuous value from 0.0 (entirely irrelevant) to 1.0 (clearly active and referenced). It is not presented to the user directly — it is used internally to determine the recommendation:

- ≥ 0.7 → KEEP_IN_PLACE
- 0.3 to 0.7 → ASK_USER (borderline; present to user with both options)
- < 0.3 → QUARANTINE:.archive

### 4.5 Integration into TRIAGE_WAVE

When SHORT_RDC is implemented:
1. TRIAGE emits `candidate-for-SHORT_RDC` verdict (a new verdict type).
2. QUEEN collects all `candidate-for-SHORT_RDC` verdicts after the wave returns.
3. QUEEN spawns SHORT_RDC workers for each candidate (parallel, one per file).
4. SHORT_RDC workers return; QUEEN substitutes their recommendations into the aggregate plan.
5. The ratification dialog shows SHORT_RDC's recommendation with its rationale, not the raw TRIAGE flag.

### 4.6 Limitations of proposed design

- SHORT_RDC is only as good as `MASTER.md` or `README.md` as a project-context anchor. Projects without orientation docs will produce low-quality SHORT_RDC judgments.
- Relevance is inherently subjective. SHORT_RDC's score is an approximation; ASK_USER remains available for borderline cases.
- SHORT_RDC adds latency to the MAINTENANCE circuit for prose-heavy projects.

---

## 5. Interim Workaround

Until SHORT_RDC is implemented, users who want rigorous prose-relevance assessment have two options:

1. **Accept TRIAGE's ASK_USER escalation.** For each prose file that TRIAGE flags as ambiguous, QUEEN presents the file and its rationale during ratification. The user makes the call. This is the default v0.1.0 experience.

2. **Run RDC_WORKFLOW manually on the prose subset.** If the project has a significant body of `.md` files and the user suspects many of them are stale, they can invoke `RDC_WORKFLOW` separately on the project's prose directory. RDC's full consolidation process will surface relevance judgments as a side effect. ORGANIZE does not auto-trigger RDC; the user invokes it manually.

Per `ORGANIZE_WORKFLOW.json` §`hard_rules.no_auto_workflow_trigger`:
> "ORGANIZE never auto-triggers RDC, RECON, SDLC, or any other workflow. The user invokes those manually if ORGANIZE surfaces a need."

---

## 6. Cross-Reference

- `ORGANIZE_WORKFLOW.json` §`relationship_to_other_workflows.rdc_coupling` — the TBD this document resolves
- `ORGANIZE_WORKFLOW.json` §`known_tbds` item 1 — "SHORT_RDC sub-procedure shape"
- `WORKER_TRIAGE.md` §4 — full-read-or-skip rule; how TRIAGE handles unreadable/ambiguous files
- `WORKER_TRIAGE.md` §5, Step 2d — fallback behavior when no rule matches (ASK_USER default)
- `RATIFICATION_UI_SPEC.md` — how ASK_USER verdicts are presented during ratification

---

*End of SHORT_RDC_SPEC.md*  
*Flag: T3.3-FOLLOWUP — revisit after ≥10 maintenance runs with false-positive archival data*
