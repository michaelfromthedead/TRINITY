# EDITORIAL_BACK_WALKTHROUGH — Back-Half Pipeline Walkthrough

**Purpose:** Continues from EDITORIAL_FRONT_WALKTHROUGH.md's 17-finding integrated report. Traces SENIOR_SANITY, SENIOR_FINAL, REVISION, and the convergence to GREEN_LIGHT across two REVISE cycles. Uses the same Spin of Gravity 3-chapter mock.

**Continuity:** all finding IDs, passages, and chapter slugs are carried over from EDITORIAL_FRONT_WALKTHROUGH.md unchanged.

---

## Setup: Incoming State

From EDITORIAL_FRONT_WALKTHROUGH.md §6, EDITORIAL_SYNTHESIS produced an integrated report of 17 findings:

| Severity | Count | IDs |
|---|---|---|
| Critical | 2 | JV-001, SYN-001 |
| High | 9 | JV-002, JV-003, JC-001, JC-002, JS-001, JF-001, JF-002, SYN-002, SYN-003 |
| Medium | 5 | JV-004, JV-005, JC-003, JS-002, JS-003 |
| Low | 0 | (none — JF-003 was upgraded to Medium by SYNTHESIS compound in SYN-003) |

`qa_cycle_counter`: 0 (this is the first cycle)

Templates: BUNDLE_SPIN_OF_GRAVITY (VOICE_SOCRATIC + PERSONA_PHYSICIST_TEACHER + STYLE_ACADEMIC_EXPLORATORY + PROSE_MEDIUM_ACCESSIBLE)

---

## Stage 5: SENIOR_SANITY Filters the 17 Findings

QUEEN spawns SENIOR_SANITY with the integrated report + all chapter files + all templates + STORYBOARD.md.

### 5.1 SENIOR_SANITY procedure

SENIOR_SANITY reads the full integrated report. Reviews each finding in severity order: Critical → High → Medium. Verifies passage quotes against actual manuscript chapters. Checks template audit checklist item references against loaded templates.

### 5.2 Rulings — Critical tier (2 findings)

**JV-001** (Critical — VOICE — CH_01 §1.1 paragraph 1)

```
sanity_verdict: real
sanity_rationale: >
  real — [VOICE:observation_chapter_open] confirmed violated: passage opens with topic
  enumeration and a learning-objective statement; VOICE_SOCRATIC Audit Checklist item
  exists verbatim in loaded template; passage quote verified at CH_01 §1.1 para 1.
```

**SYN-001** (Critical — CROSS_AXIS: VOICE+STYLE — CH_01 §1.1 paragraph 1)

SYN-001 compounds JV-001 and JS-001. SENIOR_SANITY checks:
- JV-001 is real (confirmed above).
- JS-001 must also be real for the compound to hold.
- JS-001 flags [STYLE:no_topic_list_opening] on the same passage. SENIOR_SANITY confirms: STYLE_ACADEMIC_EXPLORATORY Audit Checklist item exists and applies.
- Both axes genuinely contribute (VOICE: observation-first requirement violated; STYLE: anti-topic-list requirement violated). The compound is genuine.

```
sanity_verdict: real
sanity_rationale: >
  real — both constituent violations confirmed: JV-001 real (VOICE:observation_chapter_open)
  and JS-001 real (STYLE:no_topic_list_opening); CROSS_AXIS compound is genuine; same
  passage simultaneously fails pedagogical contract and genre convention.
```

### 5.3 Rulings — High tier (9 findings)

**JV-002** (High — VOICE — CH_02 §2.1 paragraph 1)

SENIOR_SANITY locates the passage: "The phenomenon of spin precession is explained by Larmor's theorem..." — present verbatim. [VOICE:no_top_down_declaration] exists in template. Theorem is stated before any observation or question.

```
sanity_verdict: real
sanity_rationale: >
  real — passage states Larmor's theorem as a conclusion before any motivating observation;
  [VOICE:no_top_down_declaration] confirmed violated; STORYBOARD.md CH_02 key move 1
  requires opening with the concrete spinning-top observation, which is absent here.
```

**JV-003** (High — VOICE — CH_03 §3.1 paragraph 1)

SENIOR_SANITY locates: "Spin behaves like angular momentum. It is a quantum number..." — confirmed. However, SENIOR_SANITY checks whether this is a Characteristic Pattern of the template being misidentified as a violation (a common false positive for short declarative series). VOICE_SOCRATIC's Characteristic Pattern 4 (permitted restatement before exploration) — does this apply? SENIOR_SANITY reads the Characteristic Pattern 4 description: "restatement of a concept already established in a prior section, used to orient the reader before deepening." The passage does not restate an established concept; it introduces new declarations. Pattern 4 does not apply. Also checks [BUNDLE:show_compare_ask_complete] — the passage has no phenomenon shown, no comparison built, no question asked.

```
sanity_verdict: real
sanity_rationale: >
  real — four consecutive declarative sentences with no observation, comparison, or
  question; [VOICE:no_top_down_declaration] and [BUNDLE:show_compare_ask_complete] both
  violated; VOICE_SOCRATIC Characteristic Pattern 4 (restatement) does not apply because
  angular momentum has not been established in any prior chapter.
```

**JC-001** (High — CONCEPT — CH_03 §3.2 paragraph 3)

Invalid cross-reference to Chapter 4 which does not exist. SENIOR_SANITY confirms STRUCTURE.md lists only 3 chapters. Cross-reference is invalid.

```
sanity_verdict: real
sanity_rationale: >
  real — "See Chapter 4 for the formal derivation" is an invalid cross-reference;
  STRUCTURE.md confirms CH_04 does not exist; forward reference is broken; no
  interpretive framing excuses a reference to a non-existent chapter.
```

**JC-002** (High — CONCEPT — CH_03 §3.1 paragraph 2)

"Angular momentum" used as assumed knowledge in CH_03, but CH_03 is the chapter that introduces it (per STORYBOARD.md prerequisite chain). SENIOR_SANITY checks STORYBOARD.md: [ANGULAR_MOMENTUM_GENERAL] appears in CH_03 Concepts Introduced, not CH_01 or CH_02. The concept is treated as known before it is established in this chapter.

```
sanity_verdict: real
sanity_rationale: >
  real — [ANGULAR_MOMENTUM_GENERAL] is in CH_03 Concepts Introduced per STORYBOARD.md;
  using it as assumed knowledge in §3.1 before introducing it in §3.2+ violates
  definition-before-use; concept cannot be assumed known prior to this chapter.
```

**JS-001** (High — STYLE — CH_01 §1.1 paragraph 1)

Already verified as real when evaluating SYN-001.

```
sanity_verdict: real
sanity_rationale: >
  real — [STYLE:no_topic_list_opening] confirmed; topic enumeration and learning-objective
  in paragraph 1 is Anti-Pattern 1 of STYLE_ACADEMIC_EXPLORATORY; same passage confirmed
  in cross-axis ruling SYN-001.
```

**JF-001** (High — FLOW — CH_03 §3.1 paragraph 1)

SENIOR_SANITY reads STORYBOARD.md CH_02 closing state: "The reader understands spin precession... holds the question: why does angular momentum generalize this?" Then reads CH_03 actual opening: "Angular momentum is the rotational analogue of linear momentum." The opening does not acknowledge or address the reader's held question.

```
sanity_verdict: real
sanity_rationale: >
  real — STORYBOARD.md CH_02 closing state specifies reader holds question "why does
  angular momentum generalize this?"; CH_03 actual opening does not address this question;
  storyboard-specified transition absent; JUNIOR_FLOW evidence accurate.
```

**JF-002** (High — FLOW — CH_02, null section, null paragraph)

SENIOR_SANITY reads STORYBOARD.md CH_02 key move 4: explicitly requires the chapter to state the spin-first ordering rationale. Then reads the actual CH_02 closing (as described in the walkthrough). The closing discusses precession results without the pedagogical positioning statement.

```
sanity_verdict: real
sanity_rationale: >
  real — STORYBOARD.md key move 4 requires explicit spin-first ordering rationale at
  CH_02's close; actual close discusses precession without this pedagogical positioning;
  finding evidence accurate; storyboard fidelity failure confirmed.
```

**SYN-002** (High — CROSS_AXIS: VOICE+CONCEPT — CH_03 §3.1 paragraphs 1-2)

Compounds JV-003, JV-004, JC-002. SENIOR_SANITY confirms all three constituent findings are real (JV-003: real; JV-004: check below; JC-002: real). Both axes genuinely interact — declarative mode (VOICE) applied to a comparison with an undefined concept (CONCEPT).

```
sanity_verdict: real
sanity_rationale: >
  real — constituent findings JV-003 (real) and JC-002 (real) compound as described;
  declarative register used to assert comparison with concept not yet established; fixing
  either axis independently is insufficient — both must be addressed together; cross-axis
  interaction is genuine.
```

**SYN-003** (High — CROSS_AXIS: FLOW+CONCEPT — CH_03 §3.3 paragraph 2)

Compounds JC-001 and JF-003. Both constituent findings real (JC-001: real). JF-003 is checked now.

JF-003 (Medium — FLOW — CH_03 §3.3 paragraph 2): SENIOR_SANITY reads the passage and STORYBOARD.md. CH_03's chapter function: "complete the formal argument" and "show why the spin-first ordering was pedagogically necessary." The "See Chapter 4" deferral prevents CH_03 from fulfilling this function. JF-003 is real (verified as part of SYN-003 evaluation).

```
sanity_verdict: real
sanity_rationale: >
  real — constituent findings JC-001 (invalid cross-reference, real) and JF-003 (storyboard
  fidelity failure, real) compound as described; manuscript ends for this reader on an
  unresolved forward reference; CH_03 fails its storyboard chapter function; cross-axis
  interaction is genuine.
```

### 5.4 Rulings — Medium tier (5 findings)

**JV-004** (Medium — VOICE — CH_03 §3.1 paragraph 1)

Flags [BUNDLE:definition_substantive_not_gestural] — "Spin behaves like angular momentum" is gestural. SENIOR_SANITY checks: does the BUNDLE_SPIN_OF_GRAVITY Synergy 3 item exist with this exact key? Confirmed. Does "Spin behaves like angular momentum" violate it? The item requires an equation, operational criterion, or measurable consequence. The passage has none. However, SENIOR_SANITY also notices this finding overlaps with JV-003 at the same passage. Both findings address the same four-sentence block. They are independent audit items; JV-004 is real on its own merits.

```
sanity_verdict: real
sanity_rationale: >
  real — [BUNDLE:definition_substantive_not_gestural] confirmed in BUNDLE_SPIN_OF_GRAVITY
  Synergy 3; "behaves like" provides no equation, operational criterion, or measurable
  consequence; passage overlaps with JV-003 but cites a distinct audit item; both real.
```

**JV-005** (Medium — VOICE — CH_01, null section, null paragraph)

Flags [VOICE:no_i_will_explain] on "By the end of this chapter, the reader will understand." SENIOR_SANITY checks: does this audit item exist verbatim in VOICE_SOCRATIC? Confirmed. Is the passage a learning-objective formulation? Yes. However, SENIOR_SANITY notes that the severity may be slightly overstated: this is the same passage as JV-001 (partially overlapping — JV-001 covers the full paragraph, JV-005 covers one sentence within it). Both are real; the findings are on distinct audit items. JV-005 is real at Medium (the learning-objective is the second half of the Anti-Pattern, already captured at Critical in JV-001; this Medium finding is the specific [VOICE:no_i_will_explain] item distinct from the opening pattern).

```
sanity_verdict: real
sanity_rationale: >
  real — [VOICE:no_i_will_explain] confirmed in VOICE_SOCRATIC; "By the end of this
  chapter, the reader will understand" is a learning-objective formulation; separate
  audit item from JV-001's observation_chapter_open; both findings valid on the same
  passage but against distinct items.
```

**JC-003** (Medium — CONCEPT — CH_02 §2.3 paragraph 4)

Flags "Noether's theorem" introduced parenthetically without definition. SENIOR_SANITY reads the passage. It is true that Noether's theorem appears in a parenthetical without prior definition. However, SENIOR_SANITY reads the BOOK_MANIFEST.json — the target audience is described as "readers with undergraduate physics background (QM I)." SENIOR_SANITY checks whether Noether's theorem is a standard undergraduate physics prerequisite. In the mock BOOK_MANIFEST, the declared prerequisite knowledge includes "classical mechanics, Lagrangian formulation, canonical transformations." Noether's theorem is a standard result in Lagrangian mechanics and would be known to the stated audience. The finding may be overzealous — the concept may be a legitimate reader prerequisite.

```
sanity_verdict: overzealous
sanity_rationale: >
  overzealous — BOOK_MANIFEST declares target audience has undergraduate physics background
  including Lagrangian mechanics; Noether's theorem is standard curriculum for this audience
  level; parenthetical use without definition is appropriate for a reader prerequisite;
  JUNIOR_CONCEPT did not check manifest-declared reader prerequisites before flagging.
```

**JS-002** (Medium — STYLE/PROSE — CH_02 §2.2 paragraph 6)

Flags a 62-word sentence violating [PROSE:sentence_length_ceiling] (40 words). SENIOR_SANITY counts the words in the quoted passage: "The electron's spin — a quantum property that behaves mathematically like angular momentum (a vector quantity defined by its transformation properties under rotations, which is itself connected to the structure of the rotation group SO(3) through Noether's theorem, which in its full generality applies to any continuous symmetry of a Lagrangian system) — does not correspond to any physical rotation." This is indeed approximately 62 words (exact count: 60). Also violates [PROSE:subordination_depth_max_two] with three parenthetical levels.

```
sanity_verdict: real
sanity_rationale: >
  real — sentence count is 60 words (above 40-word ceiling); three levels of parenthetical
  subordination confirmed ([PROSE:subordination_depth_max_two] violated); both audit items
  exist in PROSE_MEDIUM_ACCESSIBLE and apply to this passage as quoted.
```

**JS-003** (Medium — STYLE/PROSE — CH_03 §3.1 paragraphs 1-4)

Flags four consecutive one-sentence paragraphs violating [PROSE:one_sentence_paragraph_limit]. SENIOR_SANITY reads the passage — "Spin behaves like angular momentum. It is a quantum number. Different particles have different spins. This distinction has important consequences." — four very short sentences. SENIOR_SANITY checks PROSE_MEDIUM_ACCESSIBLE: [PROSE:one_sentence_paragraph_limit] forbids more than 2 consecutive one-sentence paragraphs. This passage is 4. The [PROSE:paragraph_length_three_to_six] item also applies. However, SENIOR_SANITY notices this passage is also flagged by JV-003, JV-004, JC-002, and SYN-002 — an extremely dense cluster of findings on the same 4-sentence block. All are real. JS-003 is a real finding on a distinct PROSE audit item.

```
sanity_verdict: real
sanity_rationale: >
  real — four consecutive one-sentence "paragraphs" confirmed; [PROSE:one_sentence_paragraph_limit]
  (max 2 consecutive) violated; [PROSE:paragraph_length_three_to_six] violated; JUNIOR_STYLE
  evidence accurate; this is the same passage as JV-003/JV-004/JC-002 — multiple real
  findings cluster here.
```

### 5.5 JF-003 ruling

JF-003 was evaluated during the SYN-003 ruling above.

```
JF-003:
sanity_verdict: real
sanity_rationale: >
  real — "See Chapter 4 for the formal derivation" creates dead-end forward reference;
  STORYBOARD.md CH_03 chapter function requires completing the formal argument, not
  deferring it; confirmed during SYN-003 evaluation; JUNIOR_FLOW evidence accurate.
```

### 5.6 SANITY summary

| Verdict | Count | IDs |
|---|---|---|
| real | 16 | JV-001, JV-002, JV-003, JV-004, JV-005, JC-001, JC-002, JS-001, JS-002, JS-003, JF-001, JF-002, JF-003, SYN-001, SYN-002, SYN-003 |
| overzealous | 1 | JC-003 |
| Severity downgrades | 0 | (none needed) |

```
==== WORKER REPORT ====
Role: SENIOR_SANITY
QA cycle: 1

Integrated findings received: 17
  Critical: 2
  High: 9
  Medium: 5
  Low: 0

Rulings:
  real: 16
  overzealous: 1 (JC-003 — Noether's theorem is a manifest-declared reader prerequisite)
  severity downgrades: 0

Notable patterns:
  JUNIOR_CONCEPT over-flagged one parenthetical concept use (JC-003) without checking
  BOOK_MANIFEST.json declared reader prerequisites. Recommend JC worker calibration for
  next cycle: check manifest.prerequisites before flagging parenthetical uses of concepts.

Outstanding (for SENIOR_FINAL's attention):
  CH_03 §3.1 paragraphs 1-4 is flagged by 5 separate findings (JV-003, JV-004, JC-002,
  JS-003, SYN-002) — this passage is the most concentrated failure in CH_03. SENIOR_FINAL
  should verify that REVISION's compound fix of this passage addresses all 5 simultaneously.
  The findings are on distinct audit items but the passage is the same. Coordinated revision
  is needed.

Passing 16 real findings to SENIOR_FINAL.
```

---

## Stage 6: SENIOR_FINAL Independent Pass

QUEEN spawns SENIOR_FINAL with: 16 real findings from SANITY + full integrated report + all chapters + all templates + STORYBOARD.md. `qa_cycle_counter`: 0.

### 6.1 SENIOR_FINAL's independent pass

SENIOR_FINAL reads the full manuscript (CH_01, CH_02, CH_03). Performs §4 checks:

**Pacing at work-scale (3-chapter subset):** In a 3-chapter manuscript, pacing issues are limited. SENIOR_FINAL notes that CH_01 and CH_03 both have dense violation clusters; CH_02 is relatively clean except for the §2.1 and §2.2 issues. Pacing check: clean at this scale.

**Coherence at arc-scale:** The 16 real findings identify two Critical violations in CH_01 and multiple High violations in CH_03. The central pedagogical contract established in CH_01 (observation-first, Socratic co-discovery) is systematically violated in CH_03's §3.1 opening. This arc-scale coherence issue is partly captured by existing findings, but SENIOR_FINAL notes a new holistic finding:

**NEW FINDING — SF-001:** The manuscript establishes its pedagogical contract in CH_01 (a Critical violation in the opening itself), then repeats the same pattern (declarative opening) in CH_03. The recurrence is not captured by any existing finding as a *pattern* — it is captured only as two separate findings. The pattern suggests the author may have a habitual chapter-opening mode that conflicts with the Socratic contract, which is relevant for calibrating the REVISION's correction approach (correction guidance should note this pattern, not just fix each instance independently).

However, on inspection, this is an emergent *context* for REVISION rather than a new finding. Both instances are already captured (JV-001 for CH_01, JV-002/JV-003 for CH_02/CH_03). SENIOR_FINAL decides not to surface this as a new finding but will note it in correction_guidance for the relevant findings.

**Register drift across chapters:** SENIOR_FINAL reads voice register across all three chapters. CH_02 mid-chapter is notably more formal (the 62-word parenthetical sentence, JS-002) than the rest. But this is already captured. No unreported register drift.

**Holistic cross-axis interactions:** SENIOR_FINAL reads EDITORIAL_SYNTHESIS's missed-interaction check. Two potential holistic findings:

**NEW FINDING — SF-001 (revised identification):** The STORYBOARD.md specifies that CH_01's closing state should leave the reader "holding a question — why is spin non-local?" and that the pedagogical contract is established. But CH_01's actual opening (which REVISION will fix) sets up a topic-list mode. Even after REVISION fixes the opening, the closing state of CH_01 must be checked to ensure it actually delivers the reader to the storyboard-specified state. SENIOR_FINAL checks the CH_01 closing passage (not flagged by any finding): "...and this is why non-locality represents a genuine puzzle." This closing appears consistent with the storyboard closing state. No new finding needed here.

**NEW FINDING — SF-001 (final):** SENIOR_FINAL reads the entire STORYBOARD.md Reader Journey section. The Storyboard specifies Stage 1 closes with "The reader holds a question and the pedagogical contract." But the reader cannot properly receive the pedagogical contract in CH_01's opening if that opening is a topic-list — the contract is supposed to be *demonstrated* (Socratic in action) from the first words. This is a holistic finding that no junior captured: the effectiveness of the pedagogical contract establishment depends on the CH_01 opening being repaired. If REVISION fixes CH_01's opening but the repair is insufficient to actually demonstrate the contract, the entire reader journey Stage 1 fails. This is an emergent holistic concern about REVISION's adequacy.

However, this is not a new finding in the FINDING_FORMAT sense — it is correction_guidance for JV-001 and SYN-001 (the correction guidance must specify that the repair must *demonstrate* the Socratic contract, not just avoid the topic-list). SENIOR_FINAL incorporates this into correction_guidance.

**NEW FINDING — SF-001 (genuine new finding):** SENIOR_FINAL reads CH_02's closing passage carefully. The storyboard says CH_02 must explicitly state "the spin-first ordering rationale." JF-002 is real and already captured. But SENIOR_FINAL notices a second flow issue in CH_02 not captured by JF-002: the storyboard specifies CH_02 key move 3 is "Derives the Larmor frequency from the equations of motion." SENIOR_FINAL reads CH_02 §2.1 paragraph 1 — which JV-002 flags for top-down declaration — and notices that the Larmor derivation appears to be skipped: the chapter *states* the result without the derivation. The storyboard key move 3 requires the derivation. This is a new finding.

```yaml
- finding:
    id: SF-001
    axis: FLOW
    severity: High
    chapter: CH_02_SPIN_PRECESSION
    section: "2.1"
    paragraph_range: null
    passage_quote: |
      [STORYBOARD.md CH_02 key move 3]: "Derives the Larmor frequency from the
      equations of motion."
      [CH_02 actual]: "This frequency is proportional to the field strength."
      [Derivation absent from chapter — no equations of motion passage exists.]
    audit_checklist_item_ref: null
    violation_description: >
      STORYBOARD.md key move 3 for CH_02 specifies that the chapter derives the Larmor
      frequency from the equations of motion. The actual chapter states the proportionality
      result without the derivation. The key move is unperformed. This was not caught by
      JUNIOR_FLOW (which focused on the spin-first ordering rationale in key move 4) or
      by JUNIOR_CONCEPT (which focused on concept ordering). It is visible only from a
      holistic reading of the storyboard key moves against the chapter content.
    correction_guidance: >
      Add a brief derivation of the Larmor frequency (2-4 paragraphs) in CH_02 §2.2 or
      §2.3, positioned after the concrete spinning-top observation that REVISION will add
      (per JV-002 correction guidance). The derivation should proceed from the torque equation
      for a magnetic dipole in an external field to the precession frequency. This is a
      new content addition, not a revision of existing text — REVISION operates in
      drafter-mode for this insertion (passage-scale, author-owned chapter context:
      note that this is a new passage, not a flagged existing one). See STORYBOARD.md
      CH_02 key move 3. Prioritize storyboard adherence over minimality for this finding
      (the key move must be performed; the prose cannot be minimal if the content is absent).
    axes_interacting: null
    originating_worker: null
    synthesis_id: null
    source: independent_pass
```

**NEW FINDING — SF-002:** SENIOR_FINAL checks the pacing of the reader effort across chapters. CH_03 has the highest concentration of violations (5 findings in §3.1 alone, plus SYN findings). SENIOR_FINAL re-reads §3.1 in context: the passage is the densest failure point. Additionally, SENIOR_FINAL notices that after the §3.1 block, CH_03's remainder (§3.2-3.3) has fewer violations but contains the JC-001 invalid cross-reference and JF-003 flow failure. This density pattern in CH_03 §3.1 is already covered by existing findings — but SENIOR_FINAL notices that JS-003 (four one-sentence paragraphs) and JV-003 (declarative top-down opening) both fire on the same passage, and SENIOR_FINAL's correction_guidance for JV-003 must explicitly address the paragraph structure (Constraint 1: fixing voice posture alone won't fix the one-sentence paragraph problem). This is a correction_guidance concern, not a new finding. No SF-002 needed.

**Summary of independent pass new findings:** 1 new finding: SF-001 (High — FLOW — CH_02 §2.1 — missing Larmor derivation per storyboard key move 3).

### 6.2 Verdict determination

Real findings from SANITY: 16
New findings from independent pass: 1 (SF-001, High)
Total: 17 findings for consideration

Severity distribution:
- Critical: 2 (JV-001, SYN-001)
- High: 10 (JV-002, JV-003, JC-001, JC-002, JS-001, JF-001, JF-002, SYN-002, SYN-003, SF-001)
- Medium: 4 (JV-004, JV-005, JS-002, JS-003)
  (JC-003 dropped as overzealous; JF-003 is real at Medium)
- Low: 0

Verdict-emission rule §3.2: Critical findings exist (JV-001, SYN-001) → REVISE.

Are there ESCALATE conditions? `qa_cycle_counter` = 0 (well below limit of 3). No intentional-voice-shift concern — the declarative openings are systematic errors, not deliberate departures. No storyboard invalidation. No template compatibility conflict.

**Verdict: REVISE.**

### 6.3 REVISE list construction

Total findings for REVISE: 17 (16 sanity-real + 1 new SF-001)
Budget: 17 of 20 — within cap. No prioritization cut needed; all findings included.

Note: CH_03 §3.1 paragraphs 1-4 is addressed by 5 findings (JV-003, JV-004, JC-002, JS-003, SYN-002). SENIOR_FINAL treats these as **1 coordinated passage** in budget terms, noting this in the REVISE list. Budget with this consolidation: 17 findings covering approximately 13-14 distinct passage locations. Well within cap.

### 6.4 Example correction_guidance entries

**JV-001 + SYN-001 (coordinated — same passage):**

```
"Rewrite CH_01 §1.1 paragraph 1 to open with a concrete, vivid description of the
EPR correlation phenomenon — what happens when you measure the spin of one entangled
particle and immediately find the other's spin determined, regardless of separation.
Do not name what this is called in paragraph 1. Let the reader experience the puzzle.
The current paragraph's topic-list ('This chapter will cover...') and learning-objective
('the reader will understand...') should be eliminated entirely. The observation should
carry the reader into the chapter's motivation naturally. The pedagogical contract is
established by demonstrating it, not announcing it. See VOICE_SOCRATIC Characteristic
Pattern 1 (observation-first opening) and STYLE_ACADEMIC_EXPLORATORY Anti-Pattern 1
(no topic-list opening). The fix must demonstrate the Socratic contract in action from
the first sentence — this is the BUNDLE_SPIN_OF_GRAVITY's emergent show-compare-ask
sequence. REVISION is writing in author-owned mode: sentence-scale minimum within
paragraph 1 scope, but the full paragraph must change."
```

**JV-002:**

```
"Rewrite CH_02 §2.1 paragraph 1 to begin with the concrete observation of a spinning
top precessing under gravity — describe the phenomenon (the axis tracing a cone, the
rate of precession) before introducing Larmor's theorem as the explanation. Paragraph 1
should show the phenomenon; paragraph 2 (the beginning of unflagged text) can then
introduce the theorem. If paragraph 2 begins with 'This is explained by Larmor's
theorem,' no change is needed to paragraph 2. See VOICE_SOCRATIC Characteristic Pattern
1. Author-owned chapter: sentence-scale minimum; the entire current paragraph 1 must
be replaced with the observation paragraph."
```

**JV-003 + JV-004 + JC-002 + JS-003 + SYN-002 (coordinated — same passage, all 5 findings):**

```
"Rewrite CH_03 §3.1 paragraphs 1-2 as a coordinated revision addressing five simultaneous
findings. The revised paragraphs must:
(1) Open with the spinning-top or magnetic precession observation established in CH_02,
recalled as a concrete phenomenon the reader already holds [VOICE: observation-first;
SYN-002 voice-concept compound].
(2) Pose the question: 'What general framework encompasses this?' [VOICE:
show_compare_ask_complete].
(3) Introduce [ANGULAR_MOMENTUM_GENERAL] as the answer to that question — define it
formally (an equation, an operational definition, or both) before using it in the
spin comparison [CONCEPT: definition-before-use; JC-002].
(4) Build the spin-as-special-case comparison from the definition, using it substantively
(equation or operational criterion) rather than gesturally [BUNDLE:
definition_substantive_not_gestural; JV-004].
(5) The rewritten paragraphs should have full paragraph development (3-6 sentences each)
rather than the current one-sentence fragments [PROSE: paragraph_length_three_to_six;
JS-003].
STORYBOARD.md CH_03 key move 1 (defines angular momentum as the general case
encompassing spin) must be performed by this passage. Prioritize concept introduction
order (CONCEPT) over voice posture (VOICE) if a tradeoff is required — the definition
must appear before the comparison regardless of narrative structure. Author-owned chapter:
this is a sentence-to-paragraph-level rewrite of paragraphs 1-2 only."
```

**SF-001 (new — missing derivation):**

```
"Add 2-4 paragraphs deriving the Larmor frequency from the torque equation for a
magnetic dipole in an external field, positioned in CH_02 §2.2 or §2.3 after the
concrete spinning-top observation. The derivation is new content (not a revision of
existing text). Proceed from: torque on magnetic dipole in field → angular momentum
change → precession rate = Larmor frequency. Write in VOICE_SOCRATIC mode: the
derivation should follow the equations of motion as a discovery, not a statement.
This insertion is the performance of STORYBOARD.md key move 3. Storyboard adherence
(Constraint 2) takes priority over minimality (Constraint 5) for this finding — the
key move must be performed; no minimal paraphrase suffices. Author-owned chapter;
treat this as a passage insertion."
```

### 6.5 SENIOR_FINAL report header

```
==== WORKER REPORT ====
Role: SENIOR_FINAL
QA cycle: 1
qa_cycle_counter: 0 / 3

VERDICT: REVISE

Verdict rationale:
  Two Critical findings (JV-001, SYN-001) confirmed real — CH_01 §1.1 fails both the
  VOICE_SOCRATIC pedagogical contract and STYLE_ACADEMIC_EXPLORATORY's genre opening
  requirement. Additionally, 10 High findings span all three chapters including missing
  Larmor derivation (SF-001, new). Manuscript requires surgical REVISION before
  GREEN_LIGHT can be considered.

Independent pass summary:
  Chapters reviewed: 3
  Pacing at work-scale: clean (3-chapter subset insufficient for book-scale pacing issues)
  Coherence at arc-scale: clean (pedagogical contract issues captured by existing findings)
  Register drift across chapters: clean (JS-002 captures the one formal-register outlier)
  Holistic cross-axis: clean (no unreported interactions)
  New findings surfaced:
    SF-001 (High, FLOW, CH_02): Larmor derivation missing per STORYBOARD.md key move 3

Real findings from SANITY: 16
New from independent pass: 1 (SF-001)
Total for REVISION: 17
Budget: 17 / 20 (within cap; CH_03 §3.1 cluster consolidated as 1 passage)

Summary for QUEEN:
  Verdict: REVISE
  Action: QUEEN spawns REVISION with 17-finding actionable list; qa_cycle_counter → 1
  Context for INPROGRESS: First REVISE cycle. JC-003 dropped (overzealous). 17 real findings
    including new SF-001. CH_03 §3.1 is densest correction point (5 findings, 1 passage).
```

---

## Stage 7: REVISION — Cycle 1

QUEEN spawns REVISION with the 17-finding actionable list. `qa_cycle_counter` increments to 1.

### 7.1 Budget and chapter assignment

Findings: 17, Budget: 20, Budget used: 17. No cap issue.

Chapter files to modify: CH_01, CH_02, CH_03 (all three have flagged passages).

### 7.2 Revision execution by finding cluster

**CH_01 cluster (JV-001, SYN-001, JV-005):**

Three findings on CH_01 §1.1 paragraph 1. REVISION addresses as coordinated passage (all three cite the same paragraph).

- Original: "This chapter will cover the non-locality problem in quantum mechanics, the EPR thought experiment, Bell's theorem, and the field-theoretic resolution thesis. By the end of this chapter, the reader will understand why non-locality is a puzzle."
- Constraint check: template (VOICE_SOCRATIC observation-first + STYLE no-topic-list), storyboard (CH_01 key move 1: introduce EPR as entry point for non-locality), concept (no new terms introduced), local context (following paragraphs not quoted — REVISION reads them to verify connector), minimality (full paragraph 1 must change; sentence-level insufficient given how the entire paragraph fails).

REVISION writes:

```
Imagine two particles — an electron and a positron — created together and then
separated across a laboratory. Measure the spin of the electron. You find it is up.
Measure the positron's spin. You find it is down, every time, regardless of how far
apart they were when you measured. This happens instantaneously — or at least, faster
than any signal could carry the information from one detector to the other.
```

Constraints verified: VOICE (opens with concrete observation, no theorem stated); STYLE (no topic list, no learning objective); STORYBOARD (key move 1: introduces EPR thought experiment as non-locality entry point); CONCEPT (no new undefined terms — "spin" is a reader prerequisite); CONTEXT (following paragraph begins "Standard quantum mechanics describes this result..." — connector works); MINIMALITY (full paragraph 1 replaced; paragraphs 2+ unchanged).

Revision entry:
```yaml
entry:
  finding_id: JV-001 (coordinated with SYN-001, JV-005)
  original_passage: |
    "This chapter will cover the non-locality problem in quantum mechanics, the EPR thought
    experiment, Bell's theorem, and the field-theoretic resolution thesis. By the end of this
    chapter, the reader will understand why non-locality is a puzzle."
  revised_passage: |
    "Imagine two particles — an electron and a positron — created together and then
    separated across a laboratory. Measure the spin of the electron. You find it is up.
    Measure the positron's spin. You find it is down, every time, regardless of how far
    apart they were when you measured. This happens instantaneously — or at least, faster
    than any signal could carry the information from one detector to the other."
  change_rationale: >
    Replaces topic-list + learning-objective opener with concrete EPR correlation
    observation. Satisfies [VOICE:observation_chapter_open], [STYLE:no_topic_list_opening],
    [VOICE:no_i_will_explain], and [BUNDLE:show_compare_ask_complete] (observation phase).
    Performs STORYBOARD.md CH_01 key move 1.
  constraints_satisfied: [1, 2, 3, 4, 5]
  status: addressed
```

**CH_02 cluster (JV-002, JS-002, SF-001):**

JV-002: CH_02 §2.1 paragraph 1 — top-down Larmor statement. REVISION rewrites to add observation-first paragraph.
JS-002: CH_02 §2.2 paragraph 6 — 60-word sentence. REVISION splits into two conformant sentences.
SF-001: New content addition — Larmor derivation (2-4 paragraphs in §2.2 or §2.3).

All three are addressed. JF-002 (spin-first ordering rationale missing at CH_02 close) also addressed by adding 2-3 sentences to the CH_02 closing passage.

**CH_03 cluster (JV-003, JV-004, JC-002, JS-003, SYN-002, JC-001, JF-001, JF-003, SYN-003):**

CH_03 §3.1 paragraphs 1-2: coordinated rewrite per the 5-finding consolidation in correction_guidance. REVISION writes paragraphs that open with the spinning-top observation recalled from CH_02, pose the generalizing question, define angular momentum formally, and then construct the spin comparison substantively.

CH_03 §3.2 paragraph 3 and §3.3 paragraph 2: invalid cross-reference "See Chapter 4 for the formal derivation" — REVISION removes the cross-reference and closes the argument using what is established in the chapter, consistent with STORYBOARD.md CH_03 chapter function (complete the formal argument).

JF-001 (transition from CH_02 close to CH_03 open): REVISION adds a brief opening line to CH_03 §3.1 that acknowledges the reader's held question from CH_02. This is the connector that the storyboard specifies must link the two chapters.

### 7.3 Conflict check

REVISION checks: any of the 17 findings produce a constraint conflict?

For the 5-finding CH_03 §3.1 cluster: REVISION checks whether Constraints 1-5 can all be satisfied simultaneously. The correction_guidance explicitly specifies that CONCEPT (definition before use) takes priority over VOICE (narrative structure) if a tradeoff is required. REVISION constructs the rewrite: observation recalled → question posed → definition provided (formal) → comparison built. All five constraints satisfied. No conflict.

**No conflicts in Cycle 1.**

### 7.4 REVISION report summary

```
==== WORKER REPORT ====
Role: REVISION
QA cycle: 1

Findings received: 17 (Critical: 2, High: 10, Medium: 4, Low: 1)
Budget: 17 / 20 (cap not reached)

Addressed: 17
Conflicted: 0
Deferred (budget): 0

Chapter files modified: CH_01, CH_02, CH_03
Chapter files untouched: (none — all three had flagged passages)

Drafter-origin chapters: none (all three chapters are author-owned)
DRAFTER_GAP resolutions: 0

Constraint conflict summary: none

Notable coordinated revisions:
  CH_01 §1.1 para 1: JV-001, SYN-001, JV-005 coordinated (same passage, 3 findings)
  CH_03 §3.1 paras 1-2: JV-003, JV-004, JC-002, JS-003, SYN-002 coordinated
    (same passage, 5 findings)
```

---

## Stage 8: Re-entry — Full Pipeline Re-run (Cycle 2)

QUEEN increments `qa_cycle_counter` to 1. QUEEN writes REVISION completion to INPROGRESS.md. QUEEN re-enters JUNIOR_EDITORIAL from the top.

The 4 junior workers audit the revised manuscript. Brief description (not full retrace):

**JUNIOR_VOICE (Cycle 2):** Reads revised CH_01 — new observation-first opening passes [VOICE:observation_chapter_open]. No declaration-before-question. Reads revised CH_02 §2.1 — spinning-top observation now precedes Larmor introduction. Reads revised CH_03 §3.1 — observation recalled, question posed, definition built. Voice significantly improved. Finds: 0 Critical, 0 High. Finds 1 Medium: a sentence in the newly added Larmor derivation (SF-001 fill) where the register is slightly more formal than PROSE_MEDIUM_ACCESSIBLE permits. Finds 1 Low: a minor word choice in CH_01 paragraph 3 (unchanged passage — this was pre-existing; note it as Low).

**JUNIOR_CONCEPT (Cycle 2):** Reads revised CH_03 §3.1 — angular momentum now defined before use. Reads revised CH_03 §3.3 — "See Chapter 4" removed, argument closed in-chapter. Prerequisite chain verified. Finds: 0 Critical, 0 High. 0 Medium. The JC-003 overzealous finding does not re-appear (Noether's theorem parenthetical was not changed — it was ruled overzealous).

**JUNIOR_STYLE (Cycle 2):** Reads revised CH_02 §2.2 paragraph 6 — sentence split into two conformant sentences; [PROSE:sentence_length_ceiling] satisfied. Reads revised CH_01 — no topic list. Finds: 0 Critical, 0 High. 1 Low: minor notation inconsistency in the new Larmor derivation (uses ω_L before defining it — Low, not Critical since the definition follows immediately after).

**JUNIOR_FLOW (Cycle 2):** Reads CH_02 closing — spin-first ordering rationale now present. Reads CH_03 opening — transition from held question confirmed. "See Chapter 4" removed. Reads STORYBOARD.md key moves — Larmor derivation now present (SF-001 addressed). Finds: 0 Critical, 0 High. 0 Medium. 0 Low.

**EDITORIAL_SYNTHESIS (Cycle 2):** Passes through: 1 Medium (voice register in Larmor derivation), 1 Low (pre-existing word choice), 1 Low (notation order). Total pass-through: 3 findings. Cross-axis analysis: 3 chapters checked for interactions. No significant new cross-axis interactions detected. Total integrated findings: 3 (all Medium or Low).

**SENIOR_SANITY (Cycle 2):**

- Medium finding (voice register in Larmor derivation): real — the new derivation paragraph has sentences at C1 register; PROSE_MEDIUM_ACCESSIBLE specifies B2/C1 range; this is borderline but real at Medium.
- Low finding (pre-existing word choice in CH_01 para 3): overzealous — this is a pre-existing passage not touched by REVISION; noting pre-existing unflagged text is out of scope.
- Low finding (notation order in Larmor derivation): overzealous — ω_L defined in the next sentence (same paragraph); definition-before-use is satisfied within the paragraph; JUNIOR_STYLE miscounted the introduction ordering.

SANITY ruling: 1 real (Medium), 2 overzealous. Total real: 1 at Medium.

**SENIOR_FINAL (Cycle 2):**

Receives: 1 real finding at Medium. Performs independent pass.

Independent pass: reads all three chapters. Pacing, coherence, register drift all clean after Cycle 1 revisions. No new findings surfaced.

Verdict-emission rule §3.1: zero Critical findings, zero High findings. The one Medium finding is the Larmor derivation prose register — borderline; meets the GREEN_LIGHT threshold ("High findings Low-severity-or-below OR resolved"). The register finding is Medium — acceptable for GREEN_LIGHT given all Critical and High issues are resolved.

**Verdict: GREEN_LIGHT.**

```
==== WORKER REPORT ====
Role: SENIOR_FINAL
QA cycle: 2
qa_cycle_counter: 1 / 3

VERDICT: GREEN_LIGHT

Verdict rationale:
  All 17 findings from Cycle 1 addressed by REVISION. Cycle 2 pipeline surfaced 1 Medium
  finding (Larmor derivation prose register, borderline B2/C1 boundary) and 2 overzealous
  Low findings. Zero Critical. Zero High. The single Medium finding does not block
  GREEN_LIGHT — manuscript is publishable with this minor register deviation, and the
  derivation's pedagogical value outweighs the register exactitude.

Independent pass: clean across all three chapters. No new Critical or High findings.

Summary for QUEEN:
  Verdict: GREEN_LIGHT
  Action: QUEEN writes completion to INPROGRESS.md; polished chapter files committed;
    ready for BOOK_PRODUCTION.
  Context for INPROGRESS: GREEN_LIGHT at cycle 2 (qa_cycle_counter = 1).
    One Medium finding (Medium — CH_02 Larmor derivation register) noted for author
    awareness; not blocking.
```

---

## Stage 9: Pipeline Convergence Verification

| Cycle | qa_cycle_counter | Real findings | Critical | High | Medium/Low | Verdict |
|---|---|---|---|---|---|---|
| 1 | 0 | 17 | 2 | 10 | 5 | REVISE |
| 2 | 1 | 1 | 0 | 0 | 1 | GREEN_LIGHT |

**Convergence: confirmed. Pipeline converges in 2 REVISE cycles.**

The 17-finding first cycle reduces to 1-finding second cycle after surgical REVISION. The remaining Medium finding is a low-stakes prose register issue in newly-added content (the Larmor derivation), which does not warrant an additional REVISE cycle.

`qa_cycle_counter` at GREEN_LIGHT: 1 (well below the limit of 3).

### 9.1 Schema continuity verification

All findings in Cycle 1 and Cycle 2 conform to FINDING_FORMAT.md schema. Key checks:
- All `passage_quote` values are verbatim from manuscript
- All `audit_checklist_item_ref` values reference items that exist in loaded templates
- All `correction_guidance` fields populated for REVISE-list items (SENIOR_FINAL populated them)
- All `sanity_verdict` and `sanity_rationale` fields added by SENIOR_SANITY
- New SENIOR_FINAL finding (SF-001) uses SF prefix as specified in FINDING_FORMAT.md §2.1
- Constraint conflict fields (in REVISION report) used as specified in WORKER_REVISION.md §5

---

## Stage 10: INPROGRESS.md Entry (Final)

```
[2026-04-18] BOOK_EDITORIAL — GREEN_LIGHT
  QUEEN: Spin of Gravity editorial pipeline complete.
  Cycles: 2 (qa_cycle_counter = 1 at GREEN_LIGHT)
  Findings: 17 addressed in Cycle 1; 1 residual Medium at Cycle 2 (below blocking threshold)
  Polished chapter files: CH_01, CH_02, CH_03 committed.
  Notable: JC-003 ruled overzealous (Noether's theorem is reader prerequisite per manifest).
  Notable: SF-001 (new finding, missing Larmor derivation) successfully addressed by REVISION.
  Status: READY FOR BOOK_PRODUCTION.
```

---

*End of EDITORIAL_BACK_WALKTHROUGH.md.*
