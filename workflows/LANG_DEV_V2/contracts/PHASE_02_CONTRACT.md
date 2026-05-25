# PHASE_02_CONTRACT — DESIGN

**Purpose:** Bind every Phase 2 task to concrete output files + acceptance commands. PHASE_QA's authority is bounded by this contract.

**Authoritative methodology source:** `workflows/LANG_DEV/LANGS_DEV_RDC/PHASE_02_DESIGN_TODO.md` and `..._ARCH.md`.

**CRITICAL — STEP 5 ordering (per COURT #1 SYNTHESIS in `LANGS_DEV_RDC/INPROGRESS.md`):**
- `STEP_05A` = `STEP 5 - DECISIONS SCHEMA.md` (format layer; T-02.2)
- `STEP_05B` = `STEP 5 - BAG GRAMMAR.md` (operator layer; T-02.3)
- v1 had these reversed. v2 enforces this ordering as a workflow hard rule (`step_5_ordering`).

**Phase verdict:** PHASE_GREEN_LIGHT iff T-02.1, T-02.2, T-02.3, T-02.4, T-02.4.1, T-02.4.2 all reach TASK_PASS.

---

## T-02.1 — Atomics

**Source docs (PHASE_EXECUTOR full-reads):**
- `STEP 4 - ATOMICS.md`
- `context.md` (re-read for primitive classification context — atom design must accommodate all 5 primitive types)

**Inputs:**
- `STEP_01/primitives_catalog.json`
- `STEP_02/object_hierarchy.json`
- `STEP_03/type_signatures.json`

**Required outputs:**

| File | Format | Required fields |
|---|---|---|
| `STEP_04/atoms_draft.json` | JSON | `atoms` array; each: `{name, description, inputs (typed ports), outputs (typed ports), phase, execution_stub}` |
| `STEP_04/port_types_draft.json` | JSON | `port_types` array; each: `{name (UPPER_SNAKE), description, examples: [...]}` |
| `STEP_04/phases_draft.json` | JSON | `phases` array; each: `{name (UPPER_SNAKE), order (unique int), description}` |
| `STEP_04/pcfg_weights_initial.json` | JSON | Uniform-distribution weights file (must exist even if empty patterns) |

**Completion criteria:**

1. Every primitive in `primitives_catalog.json` has at least one corresponding atom (1:1 or 1:many)
2. 5-10 port types defined (5 ≤ count ≤ 10; if outside range, escalate with rationale)
3. 3-8 phases defined; orders are unique non-negative integers
4. Every atom's `phase` field references a defined phase
5. Every atom's `inputs[]`/`outputs[]` reference defined port types
6. `pcfg_weights_initial.json` exists with valid JSON (uniform distribution scheme: equal weight on every (prev_atom, next_atom) pair, OR `{}` indicating "uniform fallback")
7. Self-contained discipline: no atom has hidden state (no `requires_global_state`, no `mutates_external` flags)

**Acceptance command:**
```bash
python -c "
import json
atoms = json.load(open('workspace_dir/STEP_04/atoms_draft.json'))['atoms']
ports = {p['name'] for p in json.load(open('workspace_dir/STEP_04/port_types_draft.json'))['port_types']}
phases = json.load(open('workspace_dir/STEP_04/phases_draft.json'))['phases']
phase_names = {p['name'] for p in phases}
phase_orders = [p['order'] for p in phases]
assert 5 <= len(ports) <= 10, f'Port count {len(ports)} out of [5,10]'
assert 3 <= len(phases) <= 8, f'Phase count {len(phases)} out of [3,8]'
assert len(set(phase_orders)) == len(phase_orders), 'Duplicate phase orders'
for a in atoms:
    assert a['phase'] in phase_names, f'Atom {a[\"name\"]} bad phase'
    for p in a.get('inputs', []) + a.get('outputs', []):
        assert p in ports, f'Atom {a[\"name\"]} bad port {p}'
json.load(open('workspace_dir/STEP_04/pcfg_weights_initial.json'))  # must parse
print(f'PASS: {len(atoms)} atoms, {len(ports)} ports, {len(phases)} phases, weights file present')
"
```

**Do NOT:**
- Do not create atoms with hidden state
- Do not let port types explode (>15 is a smell that indicates type confusion)
- Do not skip PCFG setup (file must exist even if uniform)
- Do not hardcode priorities in atom definitions — those go in T-02.4 ruleset
- Do not name atoms after libraries (`PandasFilter`); name after operations (`Filter`)

**Verdict:**
- All criteria + acceptance pass → `TASK_PASS`
- Hidden state in any atom → `TASK_FAIL_RETRY` (Critical)
- Port count or phase count outside range → `TASK_FAIL_RETRY` with guidance

---

## T-02.2 — Decisions Schema (STEP 5A)

**Source docs:**
- `STEP 5 - DECISIONS SCHEMA.md` (this is `STEP_05A` per COURT #1 SYNTHESIS)

**Inputs:**
- `STEP_04/atoms_draft.json`
- `STEP_04/port_types_draft.json`
- `STEP_04/phases_draft.json`

**Required outputs:**

| File | Format | Required fields |
|---|---|---|
| `<library>_decisions.json` (at `workspace_dir/` root, NOT in STEP_05A subdir) | JSON | `{meta: {library, domain, version, created (ISO timestamp)}, port_types: [...], phases: [...], atoms: [...]}` |

**Schema (binding):**
```json
{
  "meta": {
    "library": "string (e.g. 'pandas')",
    "domain": "string (e.g. 'tabular data manipulation')",
    "version": "semver string",
    "created": "ISO 8601 timestamp"
  },
  "port_types": [
    {"name": "UPPER_SNAKE", "description": "string", "examples": ["string", ...]}
  ],
  "phases": [
    {"name": "UPPER_SNAKE", "order": "non-negative int (unique)", "description": "string"}
  ],
  "atoms": [
    {
      "name": "string",
      "phase": "string (must match a phases[].name)",
      "inputs": ["port_type_name", ...],
      "outputs": ["port_type_name", ...],
      "description": "string",
      "derived_from_nexus": "optional string (nexus signal ref)"
    }
  ]
}
```

**Completion criteria:**

1. `<library>_decisions.json` is valid JSON at `workspace_dir/<library>_decisions.json`
2. `meta` block complete (all 4 required fields)
3. Referential integrity: every `atoms[].phase` exists in `phases[].name`; every entry of `atoms[].inputs[]`/`outputs[]` exists in `port_types[].name`
4. Uniqueness: no duplicate names within `port_types`, `phases`, or `atoms`
5. Phase orders unique
6. Minimum cardinality: ≥3 port_types, ≥3 phases, ≥5 atoms
7. Schema version field set to `1.0.0`

**Acceptance command:**
```bash
python -c "
import json, os
# Find the decisions.json (named per library)
ws = 'workspace_dir'
candidates = [f for f in os.listdir(ws) if f.endswith('_decisions.json')]
assert len(candidates) == 1, f'Expected exactly one *_decisions.json at workspace root; found {candidates}'
d = json.load(open(os.path.join(ws, candidates[0])))
assert all(k in d['meta'] for k in ['library','domain','version','created']), 'meta incomplete'
assert len(d['port_types']) >= 3 and len(d['phases']) >= 3 and len(d['atoms']) >= 5, 'cardinality too low'
port_names = {p['name'] for p in d['port_types']}
phase_names = {p['name'] for p in d['phases']}
phase_orders = [p['order'] for p in d['phases']]
assert len(set(phase_orders)) == len(phase_orders), 'duplicate phase order'
for a in d['atoms']:
    assert a['phase'] in phase_names, f'atom {a[\"name\"]} bad phase {a[\"phase\"]}'
    for p in a['inputs'] + a['outputs']:
        assert p in port_names, f'atom {a[\"name\"]} bad port {p}'
print(f'PASS: {candidates[0]} schema-valid')
"
```

**Do NOT:**
- Do not skip referential integrity validation
- Do not embed code in `decisions.json` — it is DECLARATIVE (no functions, no expressions)
- Do not hand-edit the file post-validation — re-run T-02.2 if changes needed
- Do not name the file generically (`decisions.json`) — must include library name

**Verdict:**
- All criteria + acceptance pass → `TASK_PASS`
- Referential integrity violation → `TASK_FAIL_RETRY` (Critical)

---

## T-02.3 — Bag Grammar (STEP 5B)

**Source docs:**
- `STEP 5 - BAG GRAMMAR.md` (this is `STEP_05B` per COURT #1 SYNTHESIS)

**Inputs:**
- `<library>_decisions.json` from T-02.2
- T-01.3's `type_signatures.json` (for dependency analysis insight)

**Required outputs:**

| File | Format | Required fields |
|---|---|---|
| `STEP_05B/bag_grammar_spec.md` | Markdown | DOMAIN header, NEEDED OPERATORS section (per-op yes/no + rationale), GRAMMAR section (BNF/EBNF), PRECEDENCE section, EXAMPLES section (3+ per operator) |

**Per-operator decision template:**
```
SEQUENCE (→):  yes/no — because <reason>
COMPOUND (-):  yes/no — because <reason>
SCOPE (()):    yes/no — because <reason>
ALTERNATIVE (|): yes/no — because <reason>
PARALLEL (||): yes/no — because <reason>
OPTIONAL (?):  yes/no — because <reason>
REPETITION (*+): yes/no — because <reason>
```

**Completion criteria:**

1. All 7 universal bag operators decided (yes/no) with rationale
2. Grammar section is unambiguous (no parse-rule overlap)
3. Precedence documented (highest-binding to lowest-binding)
4. Each `yes` operator has 3+ example expressions parsable under the grammar
5. Notation does not conflict with `STEP_06_2/token_inventory.md` (T-02.4.2 — but T-02.3 runs before; documented anticipation suffices)

**Acceptance command:**
```bash
# Spec-format check (count operators decided + count examples)
grep -E '^(SEQUENCE|COMPOUND|SCOPE|ALTERNATIVE|PARALLEL|OPTIONAL|REPETITION) ' workspace_dir/STEP_05B/bag_grammar_spec.md | wc -l | xargs -I{} test {} -eq 7
echo $?  # Expected: 0
grep -c '^EXAMPLE' workspace_dir/STEP_05B/bag_grammar_spec.md
# Expected: count >= (3 * number of YES operators)
```

**Do NOT:**
- Do not implement all 7 operators if your domain doesn't need them
- Do not leave precedence unspecified
- Do not use notation conflicting with the token inventory (T-02.4.2)

**Verdict:**
- All criteria + acceptance pass → `TASK_PASS`
- Missing operator decision → `TASK_FAIL_RETRY`

---

## T-02.4 — Rulesets and Defaults

**Source docs:**
- `STEP 6 - RULESETS AND DEFAULTS.md`

**Inputs:**
- `<library>_decisions.json` from T-02.2
- `STEP_05B/bag_grammar_spec.md` from T-02.3

**Required outputs:**

| File | Format | Required fields |
|---|---|---|
| `STEP_06/ruleset_spec.md` | Markdown | All 10 default categories filled with recommendation + rationale + override mechanism + alternatives considered; default-hierarchy priority list; 90/10 target stated |

**10 categories (binding — all must be addressed):**

1. Phase assignment (how do new atoms get assigned to phases?)
2. Intra-phase ordering (within a phase, what determines order?)
3. Implicit compounds (which atoms auto-group without `-` operator?)
4. Implicit sequences (which orderings are auto-enforced without `→` operator?)
5. Type coercion (when types almost-match, what happens?)
6. Failure behavior (when assembly fails, what happens?)
7. Ambiguity resolution (when multiple orders work, which wins?)
8. Optional inclusion (when is `?` included vs excluded?)
9. Repetition bounds (default min/max for `*` and `+`?)
10. Parallel execution (when can atoms run concurrently?)

**Default hierarchy (binding):**
1. Explicit grammar (highest)
2. Detected dependencies (column, type-forced)
3. Intra-phase priorities
4. Phase ordering
5. Type compatibility
6. Alphabetical (lowest tiebreaker)

**Completion criteria:**

1. All 10 categories addressed (recommendation + rationale + alternatives + override mechanism)
2. Default hierarchy documented with priority numbers
3. 90/10 target stated explicitly ("defaults handle 90% of cases, grammar 10%")
4. Each category cites a STEP-doc reference

**Acceptance command:**
```bash
# Verify all 10 categories present
for cat in "Phase assignment" "Intra-phase ordering" "Implicit compounds" "Implicit sequences" "Type coercion" "Failure behavior" "Ambiguity resolution" "Optional inclusion" "Repetition bounds" "Parallel execution"; do
  grep -q "$cat" workspace_dir/STEP_06/ruleset_spec.md || { echo "MISSING: $cat"; exit 1; }
done
echo "PASS: all 10 categories present"
```

**Do NOT:**
- Do not skip any category (silence on a category = workflow ESCALATE)
- Do not over-specify (>10 rules signals defaults that are too granular)

**Verdict:**
- All 10 categories + hierarchy + 90/10 → `TASK_PASS`
- Missing category → `TASK_FAIL_RETRY` listing the missing categories

---

## T-02.4.1 — The Conundrum (sub-task of T-02.4)

**Source docs:**
- `STEP 6.1 - THE_CONUNDRUM.md`

**Inputs:**
- All outputs of T-02.4

**Required outputs:**

| File | Format | Required fields |
|---|---|---|
| `STEP_06_1/conundrum_resolution.md` | Markdown | Cracks identified, defaults addressing each, residual grammar needed (the 10%), layered override model documented, 90/10 measurement plan |

**Completion criteria:**

1. Each "crack" the domain has (semantic dependency, inseparable pair, mutual exclusion, conditional inclusion) is named
2. For each crack: which default rule from T-02.4 addresses it, OR which grammar operator from T-02.3 addresses it
3. No crack left unaddressed (would mean broken self-assembly)
4. 90/10 measurement plan describes how grammar usage will be tracked after first DSL runs

**Acceptance command:**
```bash
test -s workspace_dir/STEP_06_1/conundrum_resolution.md && \
  grep -qi 'crack\|tension\|conundrum' workspace_dir/STEP_06_1/conundrum_resolution.md && \
  grep -qi '90/10\|measurement\|track' workspace_dir/STEP_06_1/conundrum_resolution.md
```

**Do NOT:**
- Do not skip — philosophical resolution is not optional
- Do not punt cracks to "future work"

**Verdict:**
- Doc internally consistent + every crack addressed → `TASK_PASS`
- Unaddressed crack → `TASK_FAIL_RETRY`

---

## T-02.4.2 — Pre-Lexer (sub-task of T-02.4)

**Source docs:**
- `STEP 6.2 - PRE LEXER.md`

**Inputs:**
- T-02.3 (grammar must agree with token notation)
- T-02.4 (no operator collisions)

**Required outputs:**

| File | Format | Required fields |
|---|---|---|
| `STEP_06_2/token_inventory.md` | Markdown | Per-token entries marked `^TOKEN`; structural tokens; keyword tokens; value-type tokens; explicitly absent list with rationale; regex per token type; shape-composition rules |

**Completion criteria:**

1. Total lexemes 10-15 (count `^TOKEN` lines; outside range escalates with rationale)
2. Every absent token (arithmetic, boolean, control flow, OOP, modules) explicitly listed with reason for absence
3. Regex pattern provided for each token type
4. No notation collision with T-02.3's grammar operators
5. NAME (`[A-Z][a-zA-Z0-9]*`) and IDENT (`[a-z_][a-zA-Z0-9_]*`) cleanly distinguished

**Acceptance command:**
```bash
count=$(grep -c '^TOKEN' workspace_dir/STEP_06_2/token_inventory.md)
[ "$count" -ge 10 ] && [ "$count" -le 15 ] || { echo "Token count $count out of [10,15]"; exit 1; }
grep -qi 'absent\|excluded\|not included' workspace_dir/STEP_06_2/token_inventory.md
echo "PASS: $count tokens, absence section present"
```

**Do NOT:**
- Do not add arithmetic / boolean / control-flow tokens (the language describes SHAPE, not LOGIC)
- Do not blur NAME and IDENT regexes
- Do not use notation that conflicts with grammar (T-02.3)

**Verdict:**
- 10-15 tokens + absence list + regex per token → `TASK_PASS`
- Token count out of range without rationale → `TASK_FAIL_RETRY`

---

## Phase 2 verdict gate

When T-02.1, T-02.2, T-02.3, T-02.4, T-02.4.1, T-02.4.2 all `TASK_PASS`:
- `<library>_decisions.json` is the authoritative DSL specification — every Phase 3 stage consumes it
- QUEEN appends Phase 2 summary to `workspace_manifest.json`
- QUEEN writes Phase 2 completion entry to `INPROGRESS.md`
- QUEEN emits `PHASE_GREEN_LIGHT`; advances to Phase 3 (T-03.1)

---

*End of PHASE_02_CONTRACT.*
