# COMPRESSION_SPEC — Compression Ratio Acceptance Gate

**Purpose:** Specify how METHODOLOGY_INTEGRATOR computes and verifies the compression ratio (Gate 2).

**Authoritative source:** `LANGS_DEV_RDC/PROJECT.md` §Success criteria #2 ("Achieves ~20:1 compression at the main semantic level"); `LANGS_DEV_RDC/PEDAGOGY.md` §3 ("Compression ratio — single → tiered").

---

## 1. What the compression ratio measures

The methodology's central claim: a small set of well-chosen primitives at the cognitive tier explains a large library API surface. Compression = `library_api_count / main_tier_primitive_count`.

Per `LANGS_DEV_RDC/CLARIFICATION.md` §1: "Reality is generated. Patterns in signals are not random statistical artifacts — they are signatures of generative processes." A high compression ratio is evidence the methodology found the generators, not just shallow patterns.

---

## 2. Definitions (binding)

### Main tier
Per `LANGS_DEV_RDC/PEDAGOGY.md` §2 and `PHASE_01_DECONSTRUCTION_ARCH.md` §2.3: **Tier 2 COGNITIVE** — what humans think in (norm, mean, solve). Most domains stabilize at Tier 2/Level 3 in the deconstruction ladder.

If `primitives_catalog.json` reports primitives at Tier 0 / Tier 1 / Tier 3 only (no Tier 2), the methodology has likely under-shot the cognitive level — flag as Gate 2 WARNING but allow PASS if deeper tier is justified for the domain.

### Main tier primitive count
`count of primitives in primitives_catalog.json where tier == 2`.

If 0 primitives at Tier 2, fall back to highest non-empty tier and document the substitution in `COMPRESSION_REPORT.md`.

### Library API count
Public API operations enumerated from `target_library`. Procedure depends on library type:

**Python library:**
1. Find all top-level modules (under `target_library/`)
2. For each module, enumerate names that:
   - Don't start with `_`
   - Are functions, methods, or class methods (not constants, not types)
3. Count unique names. Methods on the same class count separately (`DataFrame.head` and `DataFrame.tail` are 2 ops).

**Rust crate:**
1. Find `pub fn` declarations in `src/lib.rs` and re-exported modules
2. Count `pub` methods on `pub struct` types
3. Exclude trait impls (counted via the trait, not duplicated per impl)

**Other languages:** define enumeration procedure on first use; document in this file's revisions.

---

## 3. Threshold (parameterized)

| Target type | Threshold | Rationale |
|---|---|---|
| Production target library (real pandas, requests, etc.) | ≥ 18:1 (target ~20:1) | Per `LANGS_DEV_RDC/PROJECT.md` §Success criteria — the methodology's production criterion |
| Reference library (`pandas_mini` and similar small targets) | ≥ 1.5:1 | Reference libraries have small API surface; high ratios mathematically impossible without trivial primitives. Sanity check that compression occurs at all. |
| Custom (per-engagement override) | per-engagement spec | Allow human to set threshold for unusual targets; document in METHODOLOGY_REPORT |

**How threshold is selected per run:**
1. If `target_library` is `workflows/LANG_DEV_V2/test_target/*`, use reference threshold (1.5:1)
2. Else default to production threshold (18:1)
3. Override available via engagement-time parameter `compression_threshold` (not in v2.0.0 trigger params; deferred to v2.1)

This parameterization fixes the issue identified in `spec/REFERENCE_LIBRARY.md`: the reference library's small denominator makes 18:1 mathematically impossible without trivial primitives.

---

## 4. Procedure

```python
# Pseudocode for METHODOLOGY_INTEGRATOR's Gate 2
import json
from pathlib import Path

# Load primitives
catalog_path = workspace_dir / 'STEP_01' / 'primitives_catalog.json'
catalog = json.loads(catalog_path.read_text())
prims = catalog['primitives']

# Count main-tier primitives
main_tier_count = sum(1 for p in prims if p.get('tier') == 2)

# If empty, fall back to nearest non-empty tier
if main_tier_count == 0:
    for t in (3, 1, 0):  # try cognitive-adjacent first
        c = sum(1 for p in prims if p.get('tier') == t)
        if c > 0:
            main_tier_count = c
            fallback_tier = t
            break

# Count library API
api_count = enumerate_public_api(target_library_path)

# Compute ratio
ratio = api_count / main_tier_count if main_tier_count else float('inf')

# Determine threshold
if str(target_library_path).startswith(reference_target_prefix):
    threshold = 1.5
    target_type = "reference"
else:
    threshold = 18.0
    target_type = "production"

# Verdict
gate_2_status = "PASS" if ratio >= threshold else "FAIL"
```

---

## 5. Output: `COMPRESSION_REPORT.md`

```markdown
# Compression Report — LANG_DEV_V2 Gate 2

**Generated:** <ISO timestamp>
**Target library:** <path>
**Target type:** production | reference | custom
**Threshold applied:** <X>:1 (rationale: <reason>)

## Counts

| Metric | Value |
|---|---|
| Main-tier primitives | <N> |
| Tier used | 2 (cognitive) | <fallback tier with note> |
| Library public API count | <M> |
| Compression ratio | <M/N>:1 |

## Verdict
**Gate 2 status:** PASS | FAIL

Threshold: <X>:1 — <met / not met>

## API enumeration trace
<head + tail of the enumerated API list, ~30 lines max>
Total: <M> operations

## Primitive tier distribution
- Tier 0 (HARDWARE): <count> primitives
- Tier 1 (COMPUTATIONAL): <count> primitives
- Tier 2 (COGNITIVE): <count> primitives ← MAIN TIER
- Tier 3 (GOAL): <count> primitives

## Notes
- Tier-2 fallback (if applicable): <reason>
- Soft warnings: <e.g., "all primitives are UNIVERSAL — check if STRUCTURAL/BRIDGE primitives were missed">
```

---

## 6. Soft warnings (do not fail Gate 2 but appear in report)

These conditions trigger NOTES in `COMPRESSION_REPORT.md` but do NOT cause Gate 2 FAIL:

| Condition | Note |
|---|---|
| All primitives are UNIVERSAL type | "STRUCTURAL/BRIDGE/GOAL/PHILOSOPHICAL primitives may have been missed (per `context.md` §Outlier principle)" |
| 1 or 0 primitives at Tier 2 | "Cognitive tier is sparse; consider whether the domain truly has only 1-2 cognitive primitives, or whether deconstruction stopped early" |
| Ratio is extreme (>50:1) | "Very high ratio may indicate over-aggressive deconstruction (combining truly distinct operations into one primitive). Verify each primitive against `LANGS_DEV_RDC/CLARIFICATION.md` §5" |
| Ratio is just above threshold (1.0-1.2× threshold) | "Marginal pass — consider whether the methodology achieved its compression goal or barely met the floor" |

---

## 7. Edge cases

**target_library is a single Python file (no submodules):** enumerate `dir(module)` excluding underscore names + non-callables.

**target_library has internal helpers exposed publicly (anti-pattern):** count them anyway; the methodology should compress these into the same primitive class. If the ratio is then too high, the catalog is wrong.

**target_library has dynamic API (e.g., dispatched methods):** document the enumeration choice in `COMPRESSION_REPORT.md` notes; the count must be reproducible.

**primitives_catalog.json has no `tier` field on entries:** Gate 2 cannot run; flag as schema violation (should have been caught by PHASE_QA for T-01.1). METHODOLOGY_INTEGRATOR returns Gate 2 FAIL with `cause: "schema_invalid"`.

---

## 8. Relationship to other gates

- **Gate 1 (shuffle test)** verifies determinism — can pass independently of compression
- **Gate 2 (this)** verifies the methodology found the right primitive count — can fail even if pipeline is otherwise correct
- **Gate 3 (e2e demo)** verifies the pipeline runs — can pass even if compression ratio is bad

All three gates independent; all three must PASS for METHODOLOGY_GREEN_LIGHT.

---

*End of COMPRESSION_SPEC.*
