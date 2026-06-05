# PEDAGOGY - Concept Evolution Log

**Workflow:** RDC_WORKFLOW v1.2.0  
**Cluster:** engine_debug_resource  
**Generated:** 2026-05-23

## Purpose

This document records the evolution of concepts as they were upserted into MASTER.md during sequential SCRIBE passes. Each entry captures what changed, from which source, and why.

---

## SCRIBE Pass 1: engine_debug_profiling.md

### New Concepts Introduced

| Concept | Value | Rationale |
|---------|-------|-----------|
| CPU Profiler | Complete implementation with `time.perf_counter_ns()` | First mention of profiling subsystem; establishes nanosecond timing foundation |
| GPU Profiler | Partial - CPU-side timing placeholder | Introduces render pass tracking with acknowledged GPU limitation |
| Memory Profiler | Complete with leak detection scoring | Introduces allocation tracking and confidence-based leak detection |
| Network Profiler | Complete with RTT/jitter/loss | Introduces packet tracking and network statistics |
| Statistics System | Complete with 4 stat types | Introduces generic statistics framework |
| CVar Configuration | 16 profiler CVars | Establishes runtime configuration pattern |
| Thread Safety Pattern | RLock for reentrant, Lock for single-owner | Establishes thread safety approach |
| Bounded Memory Growth | MAX_COMPLETED_SAMPLES pattern | Establishes memory management pattern |

### No Overwrites

First source document - all concepts are insertions.

---

## SCRIBE Pass 2: engine_debug_testing.md

### New Concepts Introduced

| Concept | Value | Rationale |
|---------|-------|-----------|
| Test Runner | Complete with discovery, filtering, fail-fast | First testing framework component |
| ExecutionMode | 4 modes: EDITOR, GAME, CLI, CI | Game engine specific execution contexts |
| Assertion Functions | 20 functions with TestFailure exception | Comprehensive assertion library |
| Benchmark System | BenchmarkResult with statistics, comparison | Performance testing framework |
| Automation Bot | 8 action types, InputSimulator | Gameplay automation testing |
| Fixture System | TestFixture, SharedFixture, CompositeFixture | Resource management for tests |
| Test Lifecycle | setUpClass -> setUp -> test -> tearDown -> tearDownClass | Standard xUnit pattern adapted for game engines |
| __test__ = False | Pytest compatibility marker | Framework coexistence pattern |

### No Overwrites

Testing is independent subsystem - no conflicts with profiling concepts.

---

## SCRIBE Pass 3: engine_resource_memory.md

### New Concepts Introduced

| Concept | Value | Rationale |
|---------|-------|-----------|
| Budget Manager | Per-category allocation with budget check | First resource management component |
| Asset Categories | 7 categories, 3 with default budgets | TEXTURE/MESH/AUDIO have defaults |
| Eviction Policies | 4 strategies: LRU, LFU, Size, Priority | Strategy pattern for cache management |
| EvictionCandidate | Metadata struct for eviction decisions | Standardized eviction data |
| Residency Manager | State machine: NON_RESIDENT -> LOADING -> RESIDENT -> EVICTING | Asset lifecycle coordination |
| Asset Pool | Generic object pool with slot-based allocation | O(1) allocation pattern |
| __slots__ usage | Memory efficiency for dataclasses | Reinforces bounded memory pattern |

### Concept Refinements

| Concept | Prior Value | New Value | Reason |
|---------|-------------|-----------|--------|
| Memory Management Pattern | "Bounded memory growth via maxlen/trim" | "Bounded memory + `__slots__` optimization" | Memory subsystem adds explicit slots usage |

---

## SCRIBE Pass 4: engine_resource_streaming.md

### New Concepts Introduced

| Concept | Value | Rationale |
|---------|-------|-----------|
| Stream Manager | heapq priority queue with concurrent limits | Central streaming coordinator |
| StreamState | 5 states: PENDING, ACTIVE, COMPLETE, CANCELLED, FAILED | Stream lifecycle |
| StreamType | 4 types: TEXTURE_MIP, MESH_LOD, AUDIO_CHUNK, WORLD_CHUNK | Asset type categorization |
| Priority Calculator | Weighted scoring (distance, screen_size, frequency) | Dynamic priority determination |
| Priority Buckets | 5 levels: CRITICAL to BACKGROUND | Threshold-based classification |
| Texture Streaming | Mip-level tracking (STUB) | Lower mip = higher priority |
| Mesh Streaming | LOD tracking (STUB) | Parallel to texture streaming |
| Audio Streaming | Ring buffer with chunk index (STUB) | Different pattern from texture/mesh |
| World Streaming | Camera-driven chunk loading (PARTIAL) | Spatial streaming pattern |
| I/O Limitation | "Synchronous simulation, no async I/O" | Explicit stub acknowledgment |

### Concept Refinements

| Concept | Prior Value | New Value | Reason |
|---------|-------------|-----------|--------|
| Classification Levels | "REAL, PARTIAL" | "REAL, PARTIAL, STUB" | Streaming introduces STUB as explicit level |
| Budget Integration | "Standalone budget manager" | "Budget defined but not connected to streaming" | Streaming reveals integration gap |

---

## Cross-Document Patterns

### Consistent Patterns Across All Sources

1. **`__slots__` optimization** - All performance-critical classes use `__slots__`
2. **Bounded memory** - All collections have explicit size limits
3. **Thread safety** - Locks used consistently where needed
4. **Strategy pattern** - Eviction policies, priority calculation
5. **State machine pattern** - Residency, streaming, world chunks

### Architectural Alignment

All four subsystems share:
- Clean separation of concerns
- Dataclass-based data transfer objects
- Explicit lifecycle management
- Testability considerations (injectable time_fn, __test__ = False)

---

## No Court Sessions Required

No contradictions detected between source documents. All concepts either:
1. Introduced new information (INSERT)
2. Refined existing concepts without conflict (OVERWRITE with temporal supersession)

The four subsystems address orthogonal concerns:
- Profiling measures performance
- Testing validates correctness
- Memory manages budgets
- Streaming coordinates loading

No cross-cutting conflicts arose.
