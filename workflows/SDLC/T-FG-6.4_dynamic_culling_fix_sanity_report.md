# T-FG-6.4 Dynamic Culling FIX Re-Review -- SENIOR_QA_SANITY Report

**Reviewer:** SENIOR_QA_SANITY
**Source:** `workflows/SDLC/T-FG-6.4_dynamic_culling_fix_findings_junior.md`
**Scope:** Verify JUNIOR_QA findings for correctness; flag any overzealous or missed items.

---

## Assessment Methodology

Each JUNIOR_QA finding is classified as:
- **REAL** -- The finding is legitimate, correctly assessed, and the verdict is sound.
- **OVERZEALOUS** -- The finding is exaggerated, the verdict is stricter than warranted, or it flags something acceptable without justification.

---

## REQUIRED ITEMS

### R1. `compute_live_output_set` (C-01)

**JUNIOR Verdict: PASS**

**Sanity: REAL**

The JUNIOR correctly verified:
- The function was a dead stub returning `Vec::new()`.
- The DEV fix implements full filtering for history, imported, and debug resources.
- Call site at line 3345 passes `config.debug_outputs_enabled` correctly.
- Two unit tests cover the positive cases (history/imported inclusions, debug pass writes).

**SENIOR observations:**
- The assessment is thorough and accurate. No overzealous claims.
- A potential gap (not flagged by JUNIOR, but not a finding in itself): no negative test verifying that non-history, non-imported, non-debug resources are *excluded*. This would strengthen coverage but is not required for the fix.

---

### R2. `compile_with_config()` (C-03 / H-01)

**JUNIOR Verdict: PASS**

**Sanity: REAL**

The JUNIOR correctly verified:
- New `compile_with_config()` at line 3306 with clean signature accepting `&CompilerConfig`.
- Original `compile()` at line 3285 preserved and delegates with `CompilerConfig::default()`.
- `apply_runtime_culling()` at line 3376 correctly filters based on runtime features and debug config.
- `CompilerConfig` doc at lines 2976-2980 updated to reference both `compute_live_output_set` and `apply_runtime_culling`.

**SENIOR observations:**
- Assessment is accurate and complete. No overzealous claims.
- The backward-compatible delegation pattern is correct.

---

### R3. Remove orphan comment (H-02)

**JUNIOR Verdict: PASS**

**Sanity: REAL**

The JUNIOR correctly verified:
- The stale `/// Stub: eliminates dead passes. Currently a no-op.` comment was deleted (confirmed at diff line 278).
- The ancillary `// Missing function stubs (recovered from merge, need implementation)` comment was also cleaned up.
- Documentation at the `compile` / `compile_with_config` area (lines 3285-3306) is now clean.
- Correctly distinguishes between the deleted stale stub comment and the legitimate "orphan" resource coloring documentation at lines 5341-5349.

**SENIOR observations:**
- Accurate and well-reasoned. The note about `greedy_color_resources` references being legitimate documentation (not orphan comments) is a correct distinction.
- Not overzealous.

---

### R4. Add doc warning to `compile()` about required post-step (H-03)

**JUNIOR Verdict: PARTIAL -- Doc added, assertion not added**

**Sanity: REAL** (but note on classification below)

The JUNIOR correctly identified:
- The REQUIRED doc warning was added: `compile_with_config()` doc at lines 3303-3305 warns about the required `apply_runtime_culling` post-step. The `compile()` doc at lines 3292-3305 similarly references it.
- The optional `#[must_use]` / `debug_assert!` recommendation from the Final Report was NOT implemented.
- The JUNIOR explicitly states this is "acceptable for the FIX cycle" and the assertion was "not marked as REQUIRED."

**SENIOR observations:**
- The REQUIRED component (doc warning) was fully implemented. The PARTIAL label in the section header derives from an optional recommendation, not from failing the requirement.
- The JUNIOR's own summary table correctly lists this as `PASS (doc)` -- the section-header `PARTIAL` is stricter than the substance of the review warrants. **This is a minor labeling inconsistency, not an overzealous finding** -- the JUNIOR's written assessment correctly explains the distinction.
- The optional recommendation is properly tracked as a follow-up note rather than a blocker.
- **Verdict on JUNIOR's assessment: REAL.** The content is accurate and fair, even if the section-header label could be read as slightly stricter than needed.

---

## RECOMMENDED ITEMS

### S1. `ALL_DEBUG` unit test (L-01)

**JUNIOR Verdict: NOT DONE**

**Sanity: REAL**

Factual observation. `FeatureSet::ALL_DEBUG` at line 1283 remains untested. This was a RECOMMENDED item; deferral is acceptable. Not overzealous.

---

### S2. Blackbox integration test for full pipeline (L-02)

**JUNIOR Verdict: NOT DONE**

**Sanity: REAL**

Factual observation. `blackbox_dynamic_culling.rs` still only tests `is_pass_live()`. The note that `CompilerConfig` is already available in test imports is additional helpful context. Not overzealous.

---

### S3. Shared test harness (L-03)

**JUNIOR Verdict: NOT DONE**

**Sanity: REAL**

Factual observation. Helper duplication between `#[cfg(test)]` module and blackbox tests persists. Not overzealous.

---

## Items the JUNIOR Did Not Flag (SENIOR perspective)

These are not failures in the JUNIOR review but contextual observations for the record:

1. **Negative test gap (R1):** No test verifies that non-qualifying resources are *excluded* from the live output set. The current tests only verify inclusions. Adding a negative assertion (e.g., "resources with `is_history: false` and `ResourceLifetime::Frame` are excluded") would close the coverage loop.

2. **Error path coverage (R2):** `compile_with_config` returns `Result<Self, String>` but no test exercises the error path (e.g., invalid pass-resource linkage triggering a Phase constraint failure). This is outside the scope of the culling fix but notable for general coverage.

Neither item was part of the REQUIRED or RECOMMENDED list for this FIX, so the JUNIOR was not remiss in omitting them.

---

## Summary

| Item | JUNIOR Verdict | Sanity | Detail |
|------|----------------|--------|--------|
| R1. `compute_live_output_set` | PASS | **REAL** | Correctly implemented and verified |
| R2. `compile_with_config()` | PASS | **REAL** | Clean API, backward compat, correct |
| R3. Orphan comment deletion | PASS | **REAL** | Stale comments removed, correct distinction |
| R4. Post-step doc warning | PASS (doc) | **REAL** | Doc added; optional assertion deferred -- acceptable |
| S1. `ALL_DEBUG` test | NOT DONE | **REAL** | Factual, deferred as expected |
| S2. Blackbox integration test | NOT DONE | **REAL** | Factual, deferred as expected |
| S3. Shared test harness | NOT DONE | **REAL** | Factual, deferred as expected |

**Overall SANITY verdict: All 7 findings are REAL. Zero OVERZEALOUS findings.**

The JUNIOR_QA review is thorough, accurate, and fair. Required vs. recommended distinctions are properly maintained. The PARTIAL label on R4 is semantically a minor strictness on the section header (the written assessment correctly explains it is acceptable), but does not rise to the level of overzealous. The three RECOMMENDED items are factually reported as deferred with no inappropriate pressure to include them.
