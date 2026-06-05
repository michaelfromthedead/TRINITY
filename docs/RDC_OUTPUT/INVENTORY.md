# RDC Inventory — V2 Testing Harness Document Set

**Created:** 2026-06-05
**Source Directory:** `docs/V*.md`
**Cluster Mode:** SINGLE (all documents share unified topic)
**Document Count:** 8

---

## Temporal Ordering

Documents sorted by: (1) explicit "Started" header, (2) dependency relationships, (3) file modification time.

| Order | Filename | Started | Size | Summary |
|-------|----------|---------|------|---------|
| 1 | V1_ADVERSARIAL_REVIEW.md | 2026-06-04 | 19KB | Adversarial QA review of existing tests — determines test quality |
| 2 | V1_IMPROVEMENT_CAMPAIGNS.md | 2026-06-04 | 14KB | Performance and code improvement campaigns |
| 3 | V1_TESTING_TOOLS.md | 2026-06-04 | 37KB | Current testing infrastructure and tooling |
| 4 | V2_SUPERSTATE_VISION.md | 2026-06-04 | 78KB | Core architecture — code as statechart, Harel model |
| 5 | V2_SUPERSQLITE_PERSISTENCE.md | 2026-06-04 | 52KB | Storage layer — supersqlite schema for code graph |
| 6 | V2_WORKFLOW_INTEGRATION.md | 2026-06-05 | 62KB | Engine — event sources, propagation, daemon |
| 7 | V2_CONTRACT_LANGUAGE.md | 2026-06-05 | 31KB | Specification — lightweight contract attributes |
| 8 | V2_MIGRATION_GUIDE.md | 2026-06-05 | 37KB | Execution — migrate 12,743 tests to V2 |

---

## Dependency Graph

```
V1 Layer (Current State):
  V1_ADVERSARIAL_REVIEW ─┐
  V1_IMPROVEMENT_CAMPAIGNS ├──> Current practice baseline
  V1_TESTING_TOOLS ────────┘

V2 Layer (Target State):
                    ┌─────────────────────────────────┐
                    │     V2_SUPERSTATE_VISION        │ ← Core architecture
                    │  (Harel statecharts, code graph)│
                    └────────────┬────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌────────────────────┐ ┌─────────────────────┐ ┌────────────────────┐
│ V2_SUPERSQLITE_    │ │ V2_WORKFLOW_        │ │ V2_CONTRACT_       │
│ PERSISTENCE        │ │ INTEGRATION         │ │ LANGUAGE           │
│ (storage layer)    │ │ (event engine)      │ │ (specification)    │
└────────────────────┘ └─────────────────────┘ └────────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────────┐
                    │     V2_MIGRATION_GUIDE          │
                    │  (how to get from V1 → V2)      │
                    └─────────────────────────────────┘
```

---

## Cluster Analysis

**Result:** SINGLE_CLUSTER

All 8 documents share:
- Same project context (TRINITY renderer-backend testing)
- Same author attribution (Michael + Claude collaboration)
- Same date range (2026-06-04 to 2026-06-05)
- Clear version progression (V1 → V2)
- Explicit cross-references between documents

No independent sub-projects detected. Standard sequential SCRIBE_LOOP applies.

---

## Reading Plan

SCRIBE will process documents in this exact order:

1. **Pass 1:** V1_ADVERSARIAL_REVIEW.md — establishes baseline concern (test quality)
2. **Pass 2:** V1_IMPROVEMENT_CAMPAIGNS.md — establishes improvement methodology
3. **Pass 3:** V1_TESTING_TOOLS.md — establishes current infrastructure
4. **Pass 4:** V2_SUPERSTATE_VISION.md — introduces target architecture (LARGEST DOC)
5. **Pass 5:** V2_SUPERSQLITE_PERSISTENCE.md — specifies storage for architecture
6. **Pass 6:** V2_WORKFLOW_INTEGRATION.md — specifies runtime engine
7. **Pass 7:** V2_CONTRACT_LANGUAGE.md — specifies contract syntax
8. **Pass 8:** V2_MIGRATION_GUIDE.md — synthesizes migration plan

---

## Expected MASTER.md Structure

Based on document content, MASTER will likely organize into:

1. **Problem Statement** — Why V2? (from V1 docs)
2. **Architecture** — Harel statecharts, code graph (from V2_SUPERSTATE_VISION)
3. **Storage** — SuperSQLite schema (from V2_SUPERSQLITE_PERSISTENCE)
4. **Engine** — Event processing, propagation (from V2_WORKFLOW_INTEGRATION)
5. **Contracts** — Attribute syntax, verification levels (from V2_CONTRACT_LANGUAGE)
6. **Migration** — Phases, timeline (from V2_MIGRATION_GUIDE)
7. **Tooling** — Current infrastructure to preserve (from V1_TESTING_TOOLS)

---

## QUEEN Notes

- Total source material: ~330KB
- Expect MASTER.md to be 40-60KB after consolidation (deduplication, synthesis)
- No conflicts expected — documents are additive, not contradictory
- If conflicts arise: likely V1→V2 supersession (later document wins)
