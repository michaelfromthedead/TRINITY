# INVENTORY - Engine Debug & Resource Subsystems

**Workflow:** RDC_WORKFLOW v1.2.0  
**Cluster:** engine_debug_resource  
**Generated:** 2026-05-23

## Source Documents (Temporal Order)

The following source documents were read in full during this RDC pass. Temporal ordering is based on investigation dates and content dependencies.

| # | File | Date | Lines | Summary |
|---|------|------|-------|---------|
| 1 | `engine_debug_profiling.md` | 2026-05-22 | 426 | CPU/GPU/memory/network profiling systems |
| 2 | `engine_debug_testing.md` | 2026-05-22 | 331 | Testing framework with assertions, fixtures, benchmarks, automation |
| 3 | `engine_resource_memory.md` | 2026-05-22 | 172 | Memory management: budgets, eviction, residency, pools |
| 4 | `engine_resource_streaming.md` | 2026-05-22 | 343 | Resource streaming: priority queue, asset-type managers |

**Total Source Lines:** 1,272

## Reading Order Rationale

All four documents share the same investigation date (2026-05-22). Ordering is based on logical dependencies:

1. **Profiling first** - Foundation for measuring performance across all systems
2. **Testing second** - Framework for validating implementations
3. **Memory third** - Core resource management (budgets, eviction)
4. **Streaming fourth** - Higher-level coordination that depends on memory/budget concepts

## Cluster Classification

**Single cluster** - All documents relate to engine infrastructure (debug + resource subsystems). No topical separation into independent sub-projects detected.
