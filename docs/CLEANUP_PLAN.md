# Documentation Cleanup Plan

**Created:** 2026-05-25  
**Status:** PROPOSED  
**Goal:** Reduce confusion, archive completed work, keep critical docs accessible

---

## Current State

| Location | Files | Purpose | Status |
|----------|-------|---------|--------|
| `docs/*.md` | 32 | Root docs, specs, investigations | MIXED — some stale |
| `docs/gap_sets/` | 215 | Rust SDLC roadmap | ✅ COMPLETE — archive candidate |
| `docs/phase_output/` | 608 | Python RDC output | ✅ COMPLETE — archive candidate |
| `docs/evaluations/` | 32 | Python/Rust health reports | CURRENT |
| `workflows/` | 252 | Workflow definitions + reports | MIXED — reports are historical |
| `.claude-flow/sessions/` | 251 | Session logs | STALE — purge candidate |

---

## Phase 1: PURGE (Safe Deletions)

### 1.1 Session Logs (251 files)
```bash
rm -rf .claude-flow/sessions/*.json
```
**Risk:** None — these are ephemeral session state, not persistent knowledge.

### 1.2 Lock Files & Temp Files
```bash
rm -f docs/.~lock.*.md#
rm -f *.rlib  # Old Rust artifacts in root
rm -f check_output.txt
rm -f mod.rs_test_work
```
**Risk:** None — build artifacts and temp files.

---

## Phase 2: ARCHIVE (Move to archive/)

### 2.1 Create Archive Structure
```bash
mkdir -p docs/archive/gap_sets_completed
mkdir -p docs/archive/phase_output_completed
mkdir -p docs/archive/workflow_reports
mkdir -p docs/archive/historical
```

### 2.2 Archive Completed Work

| Source | Destination | Reason |
|--------|-------------|--------|
| `docs/gap_sets/` | `docs/archive/gap_sets_completed/` | 20/20 SDLC-READY, work done |
| `docs/phase_output/` | `docs/archive/phase_output_completed/` | 35/35 directories done |
| `workflows/SDLC/T-*.md` | `docs/archive/workflow_reports/` | Historical task reports |

**Risk:** LOW — These are completed work products, not active references.

### 2.3 Archive Historical Root Docs

| File | Action | Reason |
|------|--------|--------|
| `ARCHAEOLOGICAL_ANALYSIS.md` | ARCHIVE | Historical analysis of GRANDPHASE1/2 |
| `CONCERNING_PROGRESS.md` | ARCHIVE | Historical progress issues (resolved) |
| `FIX_BAD_RDC_FILES.md` | ARCHIVE | One-time fix instructions (done) |
| `CRON_JOB_PROMPT.md` | ARCHIVE | Cron setup instructions (done) |
| `GAME_ENGINE_INTEGRATION_TODO.md` | ARCHIVE | Original TODO (superseded) |
| `VIPERIDE_v2.md` | ARCHIVE | IDE spec (not active) |

---

## Phase 3: CONSOLIDATE (Merge Redundant Docs)

### 3.1 Status Documents → Single `STATUS.md`
Currently have:
- `STATUS.md` — current
- `REMAINING_WORK_ROADMAP.md` — overlaps
- `PYTHON_EVALUATION_TODO.md` — completed (24/24)

**Action:** Keep `STATUS.md` as single source, archive others.

### 3.2 Investigation Documents → Single Reference
Currently have:
- `ARCHITECTURE_INVESTIGATION_REPORT.md`
- `RUST_INVESTIGATIONS.md`
- `WGPU_INVESTIGATION.md`

**Action:** Keep all three — they cover different domains. Add cross-references.

### 3.3 Code Reviews → `docs/reviews/`
```bash
mkdir -p docs/reviews
mv docs/code_review_*.md docs/reviews/
mv docs/pcg_code_review.md docs/reviews/
```

---

## Phase 4: REORGANIZE (Clear Hierarchy)

### Final `docs/` Structure
```
docs/
├── STATUS.md                          # Single source of truth
├── README.md                          # Navigation guide (NEW)
├── RUST_VS_PYTHON.md                  # Language map
│
├── specs/                             # Authoritative specifications
│   ├── TRINITY_LATEST.md              # Trinity Pattern
│   ├── FORMULAS.md                    # Math
│   ├── INTERCONNECTIVITY.md           # Dependencies
│   └── PLATFORM_RHI_IMPLEMENTATION.md # RHI
│
├── architecture/                      # Architecture docs
│   ├── ARCHITECTURE_INVESTIGATION_REPORT.md
│   ├── RENDERER_ARCHITECTURE.md
│   └── RENDERER_BACKEND_CLEANUP.md
│
├── investigations/                    # Research & analysis
│   ├── RUST_INVESTIGATIONS.md
│   ├── WGPU_INVESTIGATION.md
│   └── WGPU_EXECUTION_PLAN.md
│
├── evaluations/                       # Current health reports
│   ├── SUMMARY.md
│   ├── *.md (Python)
│   └── rust/
│
├── reviews/                           # Code reviews
│   └── *.md
│
├── plans/                             # Active roadmaps
│   ├── SKIPPED_TESTS_DEBT.md
│   └── PYTHON_VERSION_PLAN.md
│
└── archive/                           # Completed/historical
    ├── gap_sets_completed/
    ├── phase_output_completed/
    ├── workflow_reports/
    └── historical/
```

---

## Phase 5: UPDATE (Fix Stale References)

### 5.1 Files Needing Updates

| File | Issue | Fix |
|------|-------|-----|
| `DESIGN_DOCS_INDEX.md` | May have broken paths | Re-scan and update |
| `TOC.md` / `TOC_FAST.md` | May reference moved files | Update or archive |
| `REMAINING_WORK_ROADMAP.md` | Overlaps STATUS.md | Archive |

### 5.2 Create New Navigation
- Create `docs/README.md` — explains the structure
- Update root `README.md` — point to `docs/STATUS.md`

---

## Execution Order

| Phase | Risk | Time | Reversible |
|-------|------|------|------------|
| 1. PURGE | None | 5 min | No (but nothing lost) |
| 2. ARCHIVE | Low | 15 min | Yes (just mv) |
| 3. CONSOLIDATE | Low | 10 min | Yes |
| 4. REORGANIZE | Medium | 20 min | Yes |
| 5. UPDATE | Low | 15 min | Yes |

**Total:** ~1 hour

---

## Validation Checklist

After cleanup:
- [ ] `docs/STATUS.md` is accurate and links work
- [ ] `docs/README.md` exists with navigation
- [ ] No broken links in active docs
- [ ] Archive is browsable for historical reference
- [ ] Session logs purged
- [ ] No stale trackers with wrong counts

---

## Decision Points (Need Your Input)

1. **Archive vs Delete gap_sets/phase_output?**
   - ARCHIVE (recommended): Preserves history, 215+608 files in archive/
   - DELETE: Saves space but loses RDC work products

2. **Keep TOC.md/TOC_FAST.md?**
   - They're 48K/34K lines of engine reference
   - Could archive or keep as specs

3. **Workflow reports (T-*.md)?**
   - 50+ files of SDLC task reports
   - Archive or delete?

4. **Aggressive or Conservative?**
   - AGGRESSIVE: Archive everything completed, minimal active docs
   - CONSERVATIVE: Keep more at root, smaller archive

---

## Quick Win (Do Now)

If you want immediate cleanup without the full plan:

```bash
# Purge session logs
rm -rf .claude-flow/sessions/*.json

# Remove temp files
rm -f docs/.~lock.*.md# *.rlib check_output.txt mod.rs_test_work

# Update stale tracker (already done)
```

This removes ~253 files with zero risk.

---

*Plan ready for review. Say "execute phase N" to run a specific phase, or "execute all" for full cleanup.*
