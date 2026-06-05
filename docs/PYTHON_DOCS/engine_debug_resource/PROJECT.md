# PROJECT - Engine Debug & Resource Subsystems

**Workflow:** RDC_WORKFLOW v1.2.0  
**Cluster:** engine_debug_resource  
**Generated:** 2026-05-23

---

## Scope

This project encompasses the **debug infrastructure** (profiling, testing) and **resource management** (memory budgeting, streaming) subsystems of the Trinity game engine. These are foundational engine services that support development, optimization, and runtime asset management.

### In Scope

1. **Debug Profiling** (`engine/debug/profiling/`)
   - CPU profiling with nanosecond timing
   - GPU profiling (CPU-side placeholder)
   - Memory profiling with leak detection
   - Network profiling with RTT/jitter/loss
   - Statistics system for visualization

2. **Debug Testing** (`engine/debug/testing/`)
   - Test runner with discovery and filtering
   - Assertion library (20 functions)
   - Benchmark framework with statistics
   - Automation bot for gameplay testing
   - Fixture system for resource management

3. **Resource Memory** (`engine/resource/memory/`)
   - Budget manager with per-category allocation
   - Eviction policies (LRU, LFU, Size, Priority)
   - Residency manager for asset lifecycle
   - Asset pool for object reuse

4. **Resource Streaming** (`engine/resource/streaming/`)
   - Stream manager with priority queue
   - Priority calculator with weighted scoring
   - Asset-type managers (texture, mesh, audio, world)

### Out of Scope

- Actual GPU timing via GPU timestamp queries (wgpu integration)
- Async I/O for streaming (currently synchronous simulation)
- Asset deserialization (handled by asset loader)
- GPU memory upload (handled by renderer)
- Archive/package streaming

---

## Goals

### Primary Goals

1. **Provide production-ready profiling** for CPU, memory, and network performance analysis
2. **Enable comprehensive testing** with runner, assertions, benchmarks, and automation
3. **Manage memory budgets** with per-category allocation and eviction policies
4. **Coordinate resource streaming** with priority-based scheduling

### Secondary Goals

1. **Thread safety** across all profiling components
2. **Bounded memory growth** to prevent profiler memory leaks
3. **Testability** via injectable dependencies (time_fn, etc.)
4. **Game engine integration** via execution modes and game commands

### Non-Goals

1. **External dependencies** - Pure Python implementation only
2. **Platform-specific code** - Portable across all supported platforms
3. **Release build overhead** - Debug descriptors stripped in release

---

## Constraints

### Technical Constraints

1. **Python 3.13** - Target runtime per project CLAUDE.md
2. **No external dependencies** - Standard library only for core modules
3. **`__slots__` required** - All performance-critical classes must use slots
4. **Thread-safe profilers** - Must support multi-threaded game code
5. **CVar configuration** - All profiler settings via CVar system

### Architectural Constraints

1. **Pure data structures** - Profiling/testing infrastructure, not business logic
2. **Strategy pattern for eviction** - Pluggable policies, not hardcoded
3. **State machine patterns** - Explicit lifecycle states for residency/streaming
4. **Bounded collections** - All histories/buffers have explicit max sizes

### Performance Constraints

1. **Nanosecond timing** - Use `time.perf_counter_ns()` for CPU profiling
2. **O(log n) priority queue** - Use `heapq` for streaming
3. **O(1) pool operations** - Stack-based free list allocation
4. **Minimal allocation overhead** - `__slots__` throughout

---

## Quality Attributes

### Implementation Quality

| Attribute | Assessment |
|-----------|------------|
| Code Completeness | High (REAL for most modules) |
| Documentation | Excellent (docstrings with examples) |
| Type Hints | Complete (all public APIs typed) |
| Error Handling | Comprehensive (try/finally patterns) |
| Testability | High (injectable dependencies) |
| Extensibility | High (inheritance, callbacks, registries) |

### Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| GPU timing is CPU-side | Inaccurate GPU measurements | Documented placeholder awaiting wgpu |
| Streaming I/O is simulated | No actual file loading | Architecture ready for async integration |
| Budget not connected to streaming | Memory limits not enforced | Budget manager exists, needs wiring |
| Screenshot action is placeholder | Logs only, no capture | Needs game rendering integration |
| Checkpoint/restore is placeholder | State tracking only | Needs game state serialization |

---

## Dependencies

### Internal Dependencies

| Module | Depends On |
|--------|------------|
| profiling | engine.debug.console.cvar |
| testing | (standalone) |
| memory | engine.resource.constants |
| streaming | engine.resource.constants |

### Standard Library

- `time` - High-resolution timing
- `threading` - Thread safety
- `traceback` - Stack trace capture
- `heapq` - Priority queue
- `itertools` - Request ID generation
- `dataclasses` - Data structures
- `enum` - Type enums
- `collections.deque` - Bounded histories

---

## Success Criteria

### Phase 1: Foundation (COMPLETE)

- [x] CPU profiler with nanosecond timing
- [x] Memory profiler with leak detection
- [x] Network profiler with RTT/jitter
- [x] Statistics system with 4 stat types
- [x] Test runner with discovery
- [x] Assertion library (20 functions)
- [x] Benchmark framework with statistics
- [x] Budget manager with categories
- [x] 4 eviction policies
- [x] Residency state machine
- [x] Asset pool with O(1) allocation
- [x] Priority queue streaming
- [x] Priority calculator with weights

### Phase 2: Integration (PENDING)

- [ ] GPU timestamp queries via wgpu
- [ ] Async I/O for streaming
- [ ] Budget enforcement in streaming
- [ ] Screenshot capture integration
- [ ] Checkpoint/restore game state
- [ ] Tracy profiler integration (optional)
- [ ] Chrome Tracing export (optional)

---

## Stakeholders

| Role | Interest |
|------|----------|
| Engine Developer | Profiling tools for optimization |
| Gameplay Developer | Testing framework for validation |
| Technical Artist | Budget tracking for assets |
| Build Engineer | CI execution mode for automation |
| QA | Automation bot for regression testing |
