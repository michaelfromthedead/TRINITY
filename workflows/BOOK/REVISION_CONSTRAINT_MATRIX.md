# REVISION_CONSTRAINT_MATRIX — Finding Type × Constraint Interaction Reference

**Version:** 1.0.0
**Status:** Active
**Consumed by:** WORKER_REVISION.md — REVISION consults this when planning edits
**Produced by:** T7.5 (Part 7, BOOK Editorial Buildout)

This matrix maps common editorial finding types to the five REVISION constraints most likely to interact when addressing that finding type. Consult it when planning an edit — it tells you which constraints to check most carefully, and what pitfalls to expect.

---

## 1. The Five Constraints (column headers)

| # | Constraint | Short name |
|---|---|---|
| 1 | Template adherence — all four axes must be satisfied after revision | TEMPLATE |
| 2 | Storyboard adherence — don't break logical structure in STORYBOARD.md | STORYBOARD |
| 3 | Concept consistency — no new terms; don't contradict existing definitions | CONCEPT |
| 4 | Local context — surrounding unflagged text must flow naturally after the edit | CONTEXT |
| 5 | Minimality — smallest valid edit that addresses the finding | MINIMALITY |

---

## 2. The Finding Types (row headers)

| Code | Finding type | Typical axis | Typical severity |
|---|---|---|---|
| VB | Voice break — passage uses the wrong voice register or posture | VOICE | Critical / High |
| CM | Concept mismatch — term used before definition, or inconsistent definition | CONCEPT | High |
| FB | Flow break — chapter transition or arc does not match storyboard | FLOW | High |
| SV | Style violation — genre convention not followed, prose density out of range | STYLE | Medium / High |
| PP | Prose pattern — sentence length, subordination depth, paragraph structure violation | STYLE / PROSE | Medium |
| DG | DRAFTER_GAP — gap marker in drafter-origin chapter flagged by JUNIOR_CONCEPT | CONCEPT | Critical |
| CA | Cross-axis compound — violation involves two or more axes simultaneously | CROSS_AXIS | High / Critical |

---

## 3. The Matrix

Impact levels: **high** (this constraint is likely to be in tension; check carefully), **medium** (check but usually manageable), **low** (rarely an issue for this finding type).

### 3.1 Voice Break (VB)

| Constraint | Likely impact | Hint for REVISION |
|---|---|---|
| TEMPLATE | **high** | Fixing a voice violation is the primary template constraint interaction. Verify the revised passage passes the specific [VOICE:X] audit item cited AND does not now fail adjacent VOICE or PROSE items. Watch for: fixing top-down declaration by introducing an observation may introduce a sentence that is too long [PROSE:sentence_length_ceiling]. |
| STORYBOARD | **high** | Voice fixes often require adding an observation or question before a statement of principle. The observation you add must not introduce a concept the storyboard reserves for a later chapter. Check STORYBOARD.md Concepts Introduced and Concepts Required for the affected chapter. |
| CONCEPT | **medium** | Voice rewrites typically do not introduce new terminology, but watch for: the observation you add to fix observation-first may name a concept that needs to be defined before it is observed. If the concept is already established in the prerequisite chain, this is fine. If not, flag. |
| CONTEXT | **high** | Voice breaks often occur at passage openings (observation-first rule). If you rewrite paragraph 1, the following unflagged paragraph 2 may now have a dangling connector ("This tells us that..." with no clear antecedent in your revised paragraph 1). Read ahead. |
| MINIMALITY | **medium** | Voice violations at chapter or section openings often require rewriting the full opening paragraph — this is typically the minimum, not excess. Word-substitution rarely suffices for observation-first violations. Sentence rewrite before paragraph rewrite when possible. |

**Common VB conflict:** template requires adding an observation before the principle statement (Constraint 1), but the concept to be observed has not yet been established at this point in the storyboard (Constraint 2). Flag as conflict — SENIOR_FINAL must decide whether to reorder storyboard or adjust the template application.

---

### 3.2 Concept Mismatch (CM)

| Constraint | Likely impact | Hint for REVISION |
|---|---|---|
| TEMPLATE | **medium** | Concept fixes (adding a definition before use) may change the passage's voice posture. Adding a formal definition in a VOICE_SOCRATIC chapter must be done in the Socratic mode — not as a textbook-style definition. Plan the definition as a discovery rather than a declaration. |
| STORYBOARD | **high** | The storyboard's Concepts Introduced field for the affected chapter tells you which concepts this chapter is responsible for introducing. A concept mismatch finding may target a passage that is *supposed* to establish the concept — verify whether this is a "define before use" issue (Constraint 2 may require the definition to appear earlier in this chapter) or a "wrong chapter for this concept" issue (which would require structural changes outside REVISION's scope). |
| CONCEPT | **high** | This is the primary axis for CM findings. Adding a definition for a concept that is already partially defined elsewhere requires reconciling the two definitions. Check whether a definition for this concept appears in any prior chapter. If yes: your definition must be consistent with it. If no: your definition sets the canonical form — use it consistently. |
| CONTEXT | **medium** | Adding a definition often requires inserting text before the first use. The text following the insertion (currently unflagged) may contain back-references to the "previous" paragraph that now refers to the inserted definition paragraph instead. Check that the insertion integrates cleanly. |
| MINIMALITY | **low** | Concept mismatch fixes are often straightforward: insert a definition sentence or rewrite the first-use passage to include a parenthetical definition. These are inherently minimal. Watch for: definition insertions that balloon into explanation that is better deferred. |

**Common CM conflict:** the concept must be defined before use (Constraint 3 requirement), but the storyboard says this chapter should present the concept as "already established" (implying it was established elsewhere, but it was not). This requires ESCALATE — the storyboard may need revision.

---

### 3.3 Flow Break (FB)

| Constraint | Likely impact | Hint for REVISION |
|---|---|---|
| TEMPLATE | **medium** | Flow breaks often require adding transition text between chapters or sections. The transition must be written in the correct voice and prose register. The Socratic voice's approach to transitions (posing a question that connects the previous chapter's closing state to this chapter's opening) must be applied. |
| STORYBOARD | **high** | Flow breaks are defined by reference to STORYBOARD.md. Your fix must achieve the specific storyboard transition or key move that the original passage missed. Read the full storyboard entry for both the closing and opening chapters involved in the break. The fix is measured against the storyboard description, not against your general sense of good transitions. |
| CONCEPT | **medium** | Transition text often references concepts established in the prior chapter. Verify that any concept named in your transition appears in the prior chapter's Concepts Introduced list. Do not use a concept in the transition that the reader does not yet have. |
| CONTEXT | **high** | Flow break fixes are inherently about context — making the ending of one chapter or section connect to the beginning of the next. Your fix is likely to touch the very boundary of flagged and unflagged text. Be precise about what constitutes the flagged passage and what is the minimal connector scope. |
| MINIMALITY | **medium** | Flow break fixes rarely require minimal edits — a missing transition is a missing paragraph, not a missing sentence. But do not add more transition content than the storyboard's key move description requires. The transition should achieve the storyboard-specified connection, not provide an overview of everything that follows. |

**Common FB conflict:** the correction_guidance says to add a transition paragraph connecting CH_02's closing state to CH_03's opening. But the natural place to add the transition (the opening of CH_03) is immediately adjacent to a passage flagged by a different finding (JV-003). Two findings address overlapping passage ranges. Resolve by addressing both findings in a single coordinated revision entry, noting both finding IDs.

---

### 3.4 Style Violation (SV)

| Constraint | Likely impact | Hint for REVISION |
|---|---|---|
| TEMPLATE | **high** | Style violations are template violations by definition. The fix must bring the passage into conformance with the cited [STYLE:X] or [PROSE:X] audit item. Simultaneously check that the fix does not violate the VOICE axis — a passage that becomes stylistically conformant but voice-violating has not been fully fixed. |
| STORYBOARD | **low** | Style violations rarely interact with the storyboard. The exception: if the style violation is a section structure issue (e.g., a chapter opens with a topic list instead of a conceptual hook), fixing it may require reordering what the opening section does — which can interact with storyboard key moves. |
| CONCEPT | **low** | Style fixes (citation format, argument structure, prose density) rarely affect concept consistency. The exception: if the fix involves restructuring an argument, verify that the restructured argument still introduces/uses concepts in the right order. |
| CONTEXT | **medium** | Prose register fixes (sentence length, subordination depth) affect individual sentences and should not break local context. Section structure fixes (opening pattern, citation format) may require larger edits that interact with surrounding unflagged text. |
| MINIMALITY | **high** | Style violations often invite over-correction. [PROSE:sentence_length_ceiling] violations: break the long sentence into two. [PROSE:subordination_depth_max_two]: collapse nested parentheticals. Do not restructure the entire paragraph because one sentence is too long. Minimum edit that satisfies the cited audit item. |

**Common SV conflict (prose density):** The flagged passage has a sentence that is 55 words long (violates [PROSE:sentence_length_ceiling] = 40 words). Splitting the sentence into two shorter ones requires restructuring the content — but one part of the sentence establishes a concept and the other uses it, and splitting disrupts this Characteristic Pattern of the VOICE template (explanation-within-use). This is a Constraint 1 / Constraint 5 tension: minimum edit (split the sentence) versus template adherence (the nested structure is a legitimate VOICE_FEYNMAN pattern). Resolution: check whether the VOICE template Characteristic Patterns explicitly permit parenthetical mid-sentence definitions. If yes: the sentence is correct per VOICE and the PROSE finding is overzealous (flag in revision report). If no: split the sentence and rewrite to achieve the same content in two conformant sentences.

---

### 3.5 Prose Pattern Violation (PP)

| Constraint | Likely impact | Hint for REVISION |
|---|---|---|
| TEMPLATE | **high** | Prose pattern violations (sentence length, subordination depth, paragraph structure) are PROSE template violations. Fix by bringing the passage into conformance with the specific PROSE audit item. Check that the fix does not introduce a VOICE violation — shorter sentences may reduce the precision of a concept's expression. |
| STORYBOARD | **low** | Prose pattern fixes rarely interact with the storyboard. Exception: a paragraph structure fix may change how many sentences are devoted to a concept, which can affect pacing without changing the logical structure. Usually acceptable. |
| CONCEPT | **low** | Watch for: splitting a sentence that defined a concept can separate the definition from the term it defines, creating a momentary ambiguity. Keep definition and definiendum in the same sentence or adjacent sentences after the split. |
| CONTEXT | **low** | Prose pattern fixes are typically local to the flagged sentence or paragraph. The surrounding text rarely depends on the specific prose structure of the flagged passage. Exception: if the flagged "paragraph" is actually the topic sentence of a paragraph that the following sentences develop — changing the topic sentence structure affects the paragraph's coherence. |
| MINIMALITY | **high** | Prose pattern violations invite over-correction. A four-sentence run of one-sentence paragraphs [PROSE:one_sentence_paragraph_limit violation]: merge the four short sentences into two properly-developed paragraphs. Do not restructure the surrounding sections. |

**Common PP pitfall:** A deeply nested parenthetical [PROSE:subordination_depth_max_two violation] exists because the original author wanted to pack three related pieces of information into one passage. Collapsing the nesting may require splitting into two separate passages, which in turn changes how the information is introduced and may break the local-context flow. Approach: restructure the parenthetical content as an adjacent sentence, not an omission. "X (which is Y, and Y relates to Z through Noether's theorem)" → "X, which is Y. This connects to Noether's theorem: any continuous symmetry..."

---

### 3.6 DRAFTER_GAP Marker (DG)

| Constraint | Likely impact | Hint for REVISION |
|---|---|---|
| TEMPLATE | **medium** | When filling a gap, the prose you write must conform to all four template axes. The gap marker says where material is missing; it does not say what template violations are permitted to fill it. Apply the same template discipline as any other revised passage. |
| STORYBOARD | **high** | The gap marker appears in a drafter-origin chapter. Read the storyboard entry for this chapter's section. The filled content must serve the same storyboard function as the gap marker implied was missing. Do not fill the gap with content that changes the chapter's storyboard role. |
| CONCEPT | **high** | Gap filling is the highest-risk operation for concept consistency. The gap marker names what was missing — you must fill it with content consistent with how that concept, fact, or argument is handled elsewhere in the manuscript. If no prior usage exists (MISSING state chapter), establish the concept in a way consistent with the scope entry. Do not invent facts. |
| CONTEXT | **medium** | The gap marker is typically surrounded by prose that expects the gap to be filled. The prose before the marker sets up for content that follows; the prose after the marker resumes from where the gap ends. Read both sides. Your fill must bridge them naturally. |
| MINIMALITY | **low** | Gap filling is inherently not minimal — you are adding content that did not exist. The minimality constraint applies to scope: fill the gap per the gap marker's `reason` field. Do not use the gap as an opportunity to add related content not specified by the gap marker. |

**DG resolution rule (Constraint 5 does not dominate here):** When filling a DRAFTER_GAP, content completeness (Constraint 3 — concept consistency and sufficiency) takes priority over strict minimality. A gap marker is a Critical finding precisely because the missing content is load-bearing. Fill it adequately. Use DRAFTER_GAP_ACK (Option B per §7.4 in WORKER_REVISION.md) if material is insufficient to fill it without fabrication.

---

### 3.7 Cross-Axis Compound (CA)

| Constraint | Likely impact | Hint for REVISION |
|---|---|---|
| TEMPLATE | **high** | Cross-axis findings involve multiple template axes. A CA fix that addresses the VOICE axis must not compromise the STYLE axis. You are simultaneously satisfying multiple audit items from multiple templates. Run the Audit Checklist from all involved axes against your revised passage. |
| STORYBOARD | **high** | Cross-axis compounds often involve a combination of voice register and concept ordering — which is precisely what the storyboard constrains. The storyboard's key moves and prerequisite chain are the reference for resolving concept-ordering elements of the compound. |
| CONCEPT | **high** | Cross-axis compounds that involve the CONCEPT axis (voice-concept conflicts, flow-concept conflicts) require particular care: fixing the voice while preserving the correct concept introduction order. These two requirements can pull against each other — concept correction_guidance specifies which axis to prioritize. |
| CONTEXT | **high** | Cross-axis fixes typically require more extensive text changes than single-axis fixes. More extensive changes increase the risk of local context disruption. Budget extra context-checking time for CA findings. |
| MINIMALITY | **medium** | CA findings typically require paragraph-scale edits because the compound is a property of the passage as a whole, not of a single sentence. The minimum for a CA fix is often a full paragraph rewrite. This is expected — not an over-edit. |

**Common CA conflict:** a voice-concept compound requires rewriting paragraph 1 to (a) open with an observation [VOICE], (b) define a concept before using it [CONCEPT], and (c) maintain a specific storyboard key move. These three requirements may specify a different ordering of content in paragraph 1. When all three cannot be simultaneously achieved, correction_guidance specifies which axis to prioritize — follow that specification, and if no specification was given, flag the conflict.

---

## 4. Quick-reference summary table

| Finding type | Most critical constraints to check | Typical minimum edit scope | Conflict risk |
|---|---|---|---|
| VB — Voice break | TEMPLATE + STORYBOARD | Paragraph (observation-first requires structural reordering) | Medium — concept-storyboard conflict common |
| CM — Concept mismatch | STORYBOARD + CONCEPT | Sentence insertion (definition-before-use) to paragraph (if ordering changes required) | High — storyboard prerequisite chain conflicts |
| FB — Flow break | STORYBOARD + CONTEXT | Paragraph (transition) to multi-paragraph (opening restructure) | Medium — boundary with unflagged text |
| SV — Style violation | TEMPLATE + MINIMALITY | Sentence (prose density) to section structure (argument organization) | Low overall; medium for section-structure fixes |
| PP — Prose pattern | TEMPLATE + MINIMALITY | Sentence (split/merge) to paragraph (structure) | Low — usually straightforward |
| DG — DRAFTER_GAP | CONCEPT + STORYBOARD | Variable — the gap size determines scope | High — fabrication risk if material thin |
| CA — Cross-axis | ALL FIVE | Paragraph (compound by nature) | High — multiple axes pulling against each other |

---

## 5. How to use this matrix

1. Look up the finding type from the finding's `axis` and `violation_description`.
2. Read the row for that finding type.
3. Before drafting your revision: identify the high-impact constraints and note what specific risk each presents for this particular edit.
4. After drafting your revision: run the five-constraint checklist in the order most likely to catch problems (high-impact constraints first).
5. If a conflict arises: produce a conflict flag per WORKER_REVISION.md §5.2.

This matrix is a reference, not a substitute for judgment. The matrix tells you where to look most carefully; your reading of the actual passage, template, and storyboard tells you what you find.

---

*End of REVISION_CONSTRAINT_MATRIX.md.*
