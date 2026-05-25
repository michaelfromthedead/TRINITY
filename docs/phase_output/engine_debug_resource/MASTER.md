# MASTER - Engine Debug & Resource Subsystems

**Workflow:** RDC_WORKFLOW v1.2.0  
**Status:** Consolidated from 4 source documents  
**Generated:** 2026-05-23

---

## 1. Debug Subsystem - Profiling

### 1.1 CPU Profiler

**Module:** `engine/debug/profiling/cpu.py` (401 lines)  
**Classification:** REAL

**Core Types:**
- `ProfileSample`: Single sample with `start_ns`, `end_ns`, `parent`, `children`
- `FlatProfileEntry`: Aggregated entry with `total_time_ns`, `self_time_ns`, `call_count`, `min_time_ns`, `max_time_ns`
- `CPUProfiler`: Thread-safe profiler with per-thread stacks

**Key Features:**
- Nanosecond timing via `time.perf_counter_ns()`
- Hierarchical scopes with parent/child relationships
- Self-time calculation excludes child time
- `MAX_COMPLETED_SAMPLES = 10000` with trim to 5000
- Context manager: `with profiler.scope("render")`
- Decorator: `@profile(name="fn", warn_ms=16.67)`
- Thread-safe via `threading.RLock`

### 1.2 GPU Profiler

**Module:** `engine/debug/profiling/gpu.py` (366 lines)  
**Classification:** PARTIAL (CPU-side timing placeholder)

**Core Types:**
- `GPUPassType`: Enum - SHADOW, DEPTH_PREPASS, GBUFFER, LIGHTING, FORWARD, TRANSPARENT, POST_PROCESS, UI, COMPUTE, CUSTOM
- `GPUPassTiming`: Single pass with `name`, `pass_type`, `start_ns`, `end_ns`, `frame_index`
- `GPUFrameTiming`: Frame with `passes`, `total_gpu_time_ns`, `frame_time_ns`
- `GPUProfiler`: Frame/pass profiler with history

**Key Features:**
- Pass begin/end with automatic timing
- Frame history with configurable size
- Average pass times over N frames
- **Limitation:** Uses CPU-side `time.perf_counter_ns()` instead of GPU timestamp queries

### 1.3 Memory Profiler

**Module:** `engine/debug/profiling/memory.py` (523 lines)  
**Classification:** REAL

**Core Types:**
- `MemoryTag`: Enum - RENDERING, PHYSICS, AUDIO, GAMEPLAY, AI, NETWORK, UI, RESOURCES, SCRIPTING, DEBUG, SYSTEM, UNKNOWN
- `AllocationRecord`: Single allocation with `ptr`, `size`, `tag`, `timestamp`, `stack_trace`, `freed`, `freed_timestamp`
- `MemorySnapshot`: Point-in-time state with `total_allocated`, `allocation_count`, `usage_by_tag`, `allocations`
- `MemoryDiff`: Delta between snapshots
- `LeakCandidate`: Potential leak with `allocation`, `age_seconds`, `confidence`, `reason`

**Key Features:**
- Allocation tracking with optional stack traces via `traceback.extract_stack()`
- Named snapshots with diff capability
- Leak detection with confidence scoring:
  - Age factor (0.4 weight): `min(1.0, age / (min_age * 10))`
  - Size factor: >1MB adds 0.3, >100KB adds 0.15
  - Tag factor: RENDERING/GAMEPLAY add 0.2
  - Stack trace adds 0.1
- Bounded memory via `FreedHistoryMax/Trim` CVars

### 1.4 Network Profiler

**Module:** `engine/debug/profiling/network.py` (591 lines)  
**Classification:** REAL

**Core Types:**
- `PacketType`: Enum - RELIABLE, UNRELIABLE, ORDERED, SEQUENCED
- `PacketDirection`: Enum - SENT, RECEIVED
- `PacketRecord`: Single packet with `packet_id`, `size`, `direction`, `packet_type`, `timestamp`, `acknowledged`, `ack_timestamp`, `channel`
- `NetworkStats`: Period stats with `bytes_sent`, `bytes_received`, `packets_sent`, `packets_received`, `rtt_ms`, `loss_percent`, `jitter_ms`, `bandwidth_sent_kbps`, `bandwidth_received_kbps`
- `ConnectionStats`: Per-connection stats

**Key Features:**
- Packet tracking with RTT calculation from ack_timestamp
- Jitter calculation: average of consecutive RTT differences
- Packet loss detection: unacknowledged packets past timeout threshold
- Connection management with `register_connection()`, `unregister_connection()`
- Bandwidth history retrieval

### 1.5 Statistics System

**Module:** `engine/debug/profiling/stats.py` (658 lines)  
**Classification:** REAL

**Stat Types:**
- `CounterStat`: Incrementing/decrementing counter
- `TimerStat`: Time measurements with history, min/max/avg
- `GraphStat`: Time-series data for visualization
- `BarStat`: Categorical data (key-value pairs)

**Built-in Stat Groups:**
| Group | Stats |
|-------|-------|
| fps | fps.current, fps.frame_time, fps.frame_count |
| memory | memory.used_mb, memory.allocated_mb, memory.by_category |
| gpu | gpu.frame_time, gpu.pass_times, gpu.utilization |
| unit | unit.total, unit.active, unit.by_type, unit.update_time |

### 1.6 Profiler Configuration

**Module:** `engine/debug/profiling/config.py` (146 lines)

**CVars:**
| CVar | Default | Description |
|------|---------|-------------|
| profiler.gpu.FrameHistorySize | 120 | Frames to keep in GPU history |
| profiler.gpu.AverageFrames | 60 | Frames for GPU timing averages |
| profiler.memory.StackTraceDepth | 10 | Max stack trace depth |
| profiler.memory.LeakMinAgeSeconds | 60.0 | Min age for leak detection |
| profiler.memory.FreedHistoryMax | 10000 | Max freed allocations in history |
| profiler.memory.FreedHistoryTrim | 5000 | Trim freed history to this count |
| profiler.memory.LargeAllocationBytes | 1MB | Large allocation threshold |
| profiler.memory.MediumAllocationBytes | 100KB | Medium allocation threshold |
| profiler.network.StatsWindowSeconds | 1.0 | Time window for network stats |
| profiler.network.PacketHistorySize | 1000 | Packets to keep in history |
| profiler.network.RttSampleSize | 100 | RTT samples for averaging |
| profiler.network.PacketTimeoutMultiplier | 5.0 | Multiplier for loss timeout |
| profiler.network.BandwidthHistorySeconds | 10.0 | Default bandwidth history duration |
| profiler.stats.TimerHistorySize | 100 | Default timer history size |
| profiler.stats.GraphHistorySize | 120 | Default graph history size |
| profiler.cpu.WarnThresholdMs | 16.67 | Default CPU warning threshold |

**Total Profiling Lines:** 2,794

---

## 2. Debug Subsystem - Testing

### 2.1 Test Runner

**Module:** `engine/debug/testing/runner.py` (773 lines)  
**Classification:** REAL

**Core Types:**
- `ExecutionMode`: Enum - EDITOR, GAME, CLI, CI
- `TestResult`: Individual test execution result
- `SuiteResult`: Aggregated suite-level results
- `TestRunner`: Main runner with discovery, filtering, fail-fast, output capture
- `TestSuite`: Base class for test suites with automatic method discovery

**Decorators:**
- `@skip(reason)`: Unconditional skip
- `@skip_if(condition, reason)`: Conditional skip
- `@expected_failure(reason)`: Mark test as expected to fail (XFAIL)

**Discovery:**
- Files matching `test_*.py` or `*_test.py`
- Methods starting with `test_`
- Sorted by source line number

**Lifecycle:**
```
setUpClass() -> [for each test: setUp() -> test_method() -> tearDown()] -> tearDownClass()
```

### 2.2 Assertions

**Module:** `engine/debug/testing/assertions.py` (689 lines)  
**Classification:** REAL

**20 Assertion Functions:**
| Function | Purpose |
|----------|---------|
| expect_eq | Equality |
| expect_ne | Inequality |
| expect_true | Boolean true |
| expect_false | Boolean false |
| expect_near | Float comparison with epsilon |
| expect_throws | Exception assertion |
| expect_contains | Containment |
| expect_not_contains | Exclusion |
| expect_is | Identity (is) |
| expect_is_not | Non-identity |
| expect_none | None check |
| expect_not_none | Non-None check |
| expect_greater | > comparison |
| expect_greater_eq | >= comparison |
| expect_less | < comparison |
| expect_less_eq | <= comparison |
| expect_in_range | Range check |
| expect_type | Exact type check |
| expect_instance | isinstance check |

**TestFailure Exception:** Contains `message`, `expected`, `actual`, `assertion_type`

### 2.3 Benchmarks

**Module:** `engine/debug/testing/benchmarks.py` (637 lines)  
**Classification:** REAL

**Core Types:**
- `Benchmark`: Single benchmark with warmup, GC control, timing
- `BenchmarkSuite`: Collection of benchmarks
- `BenchmarkResult`: Statistics - avg, min, max, std_dev, median, percentiles (95th, 99th), ops/sec, coefficient of variation
- `BenchmarkComparison`: Compare results to baseline

**Decorators:**
- `@bench(iterations=1000, warmup=100)`
- `@suite.benchmark`

**Features:**
- GC collection tracking
- Comparison with speedup and percent change

### 2.4 Automation

**Module:** `engine/debug/testing/automation.py` (1030 lines)  
**Classification:** REAL

**Core Types:**
- `AutomationBot`: Executes test scenarios
- `TestScenario`: Sequence of steps
- `ScenarioStep`: Single step with action and expected result
- `Action`: Atomic operation
- `InputSimulator`: Simulates keyboard, mouse, gamepad

**Action Types (8):**
- INPUT, WAIT, EXECUTE, VERIFY, CHECKPOINT, RESTORE, LOG, SCREENSHOT

**Action Factory Methods:**
- `Action.input()`, `Action.click()`, `Action.key()`
- `Action.wait()`, `Action.delay()`
- `Action.execute()`, `Action.verify()`
- `Action.checkpoint()`, `Action.restore()`
- `Action.log()`, `Action.screenshot()`

**InputSimulator Methods:**
- `simulate_text()`, `simulate_key()`
- `simulate_mouse_click()`, `simulate_mouse_move()`
- `simulate_gamepad_button()`, `simulate_gamepad_axis()`

### 2.5 Fixtures

**Module:** `engine/debug/testing/fixtures.py` (616 lines)  
**Classification:** REAL

**Core Types:**
- `TestFixture`: Per-test fixtures
- `SharedFixture`: Singleton fixtures shared across suites
- `CompositeFixture`: Combines multiple fixtures
- `FixtureContext`: Metadata about current test execution

**Lifecycle:**
```
setUpClass(context) -> setUp(context) -> [test] -> tearDown(context) -> tearDownClass(context)
```

**Factory Functions:**
- `fixture(setup, teardown, name)`
- `shared_fixture(name, setup_class, teardown_class)`

### 2.6 Testing Constants

| Constant | Value | Location |
|----------|-------|----------|
| DEFAULT_TEST_TIMEOUT_MS | 30000 | runner.py |
| DEFAULT_ACTION_TIMEOUT_MS | 5000.0 | automation.py |
| DEFAULT_POLL_INTERVAL_MS | 100.0 | automation.py |
| DEFAULT_BENCHMARK_ITERATIONS | 1000 | benchmarks.py |
| DEFAULT_BENCHMARK_WARMUP | 100 | benchmarks.py |
| DEFAULT_SIGNIFICANCE_THRESHOLD | 0.05 | benchmarks.py |
| DEFAULT_FLOAT_EPSILON | 1e-6 | assertions.py |
| DEFAULT_VALUE_FORMAT_MAX_LENGTH | 100 | assertions.py |

**Total Testing Lines:** 3,918

---

## 3. Resource Subsystem - Memory

### 3.1 Budget Manager

**Module:** `engine/resource/memory/budget_manager.py` (120 lines)  
**Classification:** REAL

**Asset Categories:**
- TEXTURE (with default budget)
- MESH (with default budget)
- AUDIO (with default budget)
- ANIMATION (no default budget)
- SHADER (no default budget)
- MATERIAL (no default budget)
- OTHER (no default budget)

**Key Methods:**
- `allocate(category, size_bytes) -> bool`: Atomic allocation with budget check
- `free(category, size_bytes)`: Release allocated memory
- `is_over_budget(category) -> bool`: Budget violation detection
- `get_pressure() -> float`: Overall memory pressure ratio (0.0-1.0)

**Features:**
- Peak usage tracking per category
- `__slots__` for memory efficiency

### 3.2 Eviction Policies

**Module:** `engine/resource/memory/eviction.py` (133 lines)  
**Classification:** REAL

**Policies:**
| Policy | Strategy | Use Case |
|--------|----------|----------|
| LRUEviction | Oldest access time first | General purpose caching |
| LFUEviction | Lowest access count first | Frequency-based caching |
| SizeEviction | Largest assets first | Quick memory reclamation |
| PriorityEviction | Lowest priority first | Priority-aware streaming |

**Core Types:**
- `EvictionCandidate`: Metadata with asset_id, size_bytes, last_access_time, access_count, priority
- `EvictionPolicy`: Abstract base class
- `EvictionManager`: Coordinates candidate tracking and policy execution

**Implementation:** Strategy pattern with `_collect_until()` helper for consistent iteration.

### 3.3 Residency Manager

**Module:** `engine/resource/memory/residency_manager.py` (162 lines)  
**Classification:** REAL

**Residency States:**
```
NON_RESIDENT -> LOADING -> RESIDENT -> EVICTING -> NON_RESIDENT
```

**Key Methods:**
- `request_residency(asset_id, size_bytes, priority) -> bool`: Request with budget check
- `release_residency(asset_id)`: Explicit release
- `touch(asset_id)`: Update access time for LRU
- `update() -> list[int]`: Run eviction cycle, returns evicted IDs

**Integration:**
- Calls `BudgetManager.allocate()` before residency
- Calls `BudgetManager.free()` on release/eviction
- Maintains `EvictionCandidate` on state transitions
- Injectable `time_fn` for testability

### 3.4 Asset Pool

**Module:** `engine/resource/memory/asset_pool.py` (90 lines)  
**Classification:** REAL

**Features:**
- Type-generic via `Generic[T]`
- Pre-allocated `capacity` slots
- Free-list tracking with stack-based allocation (LIFO)
- `acquire(obj) -> (slot_id, obj)`: O(1) allocation
- `release(slot_id)`: O(1) deallocation
- `reset()`: Bulk release all slots

**Total Memory Lines:** 558

---

## 4. Resource Subsystem - Streaming

### 4.1 Stream Manager

**Module:** `engine/resource/streaming/stream_manager.py` (141 lines)  
**Classification:** PARTIAL (I/O simulation)

**Core Types:**
- `StreamType`: Enum - TEXTURE_MIP, MESH_LOD, AUDIO_CHUNK, WORLD_CHUNK
- `StreamState`: Enum - PENDING, ACTIVE, COMPLETE, CANCELLED, FAILED
- `StreamPriority`: IntEnum - CRITICAL(0) to BACKGROUND(4)
- `StreamRequest`: Request with auto-incrementing ID, priority-based ordering
- `StreamManager`: Priority queue coordinator

**Key Features:**
- `heapq`-based O(log n) priority queue
- Thread-safe auto-incrementing request IDs via `itertools.count()`
- `MAX_CONCURRENT_STREAMS = 8`
- **Limitation:** Progress simulation, no real I/O

### 4.2 Priority System

**Module:** `engine/resource/streaming/priority_system.py` (96 lines)  
**Classification:** REAL

**Core Types:**
- `PriorityBucket`: IntEnum - CRITICAL(0) to BACKGROUND(4)
- `PriorityWeights`: Configurable weights (distance=0.4, screen_size=0.35, frequency=0.25)
- `StreamPriorityCalculator`: Weighted scoring with bucket classification

**Priority Formula:**
```python
distance_score = 1.0 / (1.0 + distance)
priority = weight.distance * distance_score + weight.screen_size * screen_size + weight.frequency * frequency + base_priority
# Clamped to [0, 1]
```

**Bucket Thresholds:**
| Score | Bucket |
|-------|--------|
| >= 0.8 | CRITICAL |
| >= 0.6 | HIGH |
| >= 0.4 | NORMAL |
| >= 0.2 | LOW |
| < 0.2 | BACKGROUND |

### 4.3 Texture Streaming

**Module:** `engine/resource/streaming/texture_streaming.py` (59 lines)  
**Classification:** STUB

**Core Types:**
- `MipStreamRequest`: texture_id, target_mip_level, current_mip_level
- `TextureStreamManager`: Mip request queue with resident mip tracking

**Features:**
- Lower mip level = higher priority (more detail)
- Maintains `_resident_mips` dict
- **STUB:** No actual texture loading

### 4.4 Mesh Streaming

**Module:** `engine/resource/streaming/mesh_streaming.py` (43 lines)  
**Classification:** STUB

**Core Types:**
- `LODStreamRequest`: mesh_id, target_lod, current_lod
- `MeshStreamManager`: LOD request queue with resident LOD tracking

**Features:**
- Maintains `_resident_lods` dict
- **STUB:** No actual mesh loading

### 4.5 Audio Streaming

**Module:** `engine/resource/streaming/audio_streaming.py` (59 lines)  
**Classification:** STUB

**Core Types:**
- `AudioChunk`: chunk_index, sample_offset, sample_count, data (bytes)
- `AudioStreamManager`: Ring buffer per audio asset

**Features:**
- `AUDIO_CHUNK_SIZE = 4096` samples per chunk
- `_buffers`: dict[str, dict[int, AudioChunk]] ring buffer
- **STUB:** Fills chunks with `b"\x00" * AUDIO_CHUNK_SIZE`

### 4.6 World Streaming

**Module:** `engine/resource/streaming/world_streaming.py` (100 lines)  
**Classification:** PARTIAL

**Core Types:**
- `ChunkState`: Enum - UNLOADED, LOADING, LOADED, UNLOADING
- `WorldChunk`: chunk_x, chunk_y, state
- `WorldStreamManager`: Camera-driven chunk loading/unloading

**Features:**
- `CHUNK_SIZE = 64` units
- Default load radius: 3 chunks
- Camera position to chunk coordinates
- State machine: UNLOADED -> LOADING -> LOADED -> UNLOADING
- **Limitation:** LOADING -> LOADED is instant (no async)

### 4.7 Streaming Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| MAX_CONCURRENT_STREAMS | 8 | Maximum active streams |
| AUDIO_CHUNK_SIZE | 4096 | Samples per audio chunk |
| CHUNK_SIZE | 64 | World chunk grid size |
| DEFAULT_LOAD_RADIUS | 3 | Chunks to load around camera |
| CRITICAL_PRIORITY_THRESHOLD | 0.8 | Priority bucket threshold |
| HIGH_PRIORITY_THRESHOLD | 0.6 | Priority bucket threshold |
| NORMAL_PRIORITY_THRESHOLD | 0.4 | Priority bucket threshold |
| LOW_PRIORITY_THRESHOLD | 0.2 | Priority bucket threshold |
| DEFAULT_DISTANCE_WEIGHT | 0.4 | Priority calculation weight |
| DEFAULT_SCREEN_SIZE_WEIGHT | 0.35 | Priority calculation weight |
| DEFAULT_FREQUENCY_WEIGHT | 0.25 | Priority calculation weight |

**Unused Budget Constants:**
- DEFAULT_TEXTURE_BUDGET: 512 MB
- DEFAULT_MESH_BUDGET: 256 MB
- DEFAULT_AUDIO_BUDGET: 128 MB

**Total Streaming Lines:** 545

---

## 5. Cross-Cutting Concerns

### 5.1 Thread Safety

| Component | Lock Type | Pattern |
|-----------|-----------|---------|
| CPUProfiler | threading.RLock | Per-thread stacks, shared completed samples |
| GPUProfiler | threading.Lock | Single-threaded rendering assumption |
| MemoryProfiler | threading.RLock | Reentrant for nested tracking |
| NetworkProfiler | threading.Lock | Single lock for all state |
| Stats | threading.RLock | Reentrant for group operations |

### 5.2 Memory Management

All profilers implement bounded memory growth:
- CPUProfiler: `MAX_COMPLETED_SAMPLES = 10000`, trims to 5000
- GPUProfiler: `_frame_history` with configurable max size
- MemoryProfiler: `FreedHistoryMax/Trim` CVars
- NetworkProfiler: `deque(maxlen=history_size)`
- Stats: `deque(maxlen=history_size)`

### 5.3 Slots Optimization

All dataclasses and performance-critical classes use `__slots__` to eliminate `__dict__` overhead.

### 5.4 Global Default Instances

Each profiler module provides global default instances via `get_default_*()` / `set_default_*()` functions.

### 5.5 Pytest Compatibility

All test framework classes have `__test__ = False` to prevent pytest from collecting them.

---

## 6. Design Patterns

| Pattern | Usage |
|---------|-------|
| Strategy | Eviction policies, Priority calculation |
| Object Pool | AssetPool for reusable allocations |
| Coordinator | ResidencyManager, StreamManager |
| State Machine | StreamRequest, WorldChunk, Residency states |
| Registry | BenchmarkSuite._registry, SharedFixture._registry |
| Context Manager | Profiler scopes, Fixture.apply() |
| Factory | Action factory methods, fixture() factory |

---

## 7. Classification Summary

| Module | Lines | Status |
|--------|-------|--------|
| debug/profiling | 2,794 | REAL (GPU partial) |
| debug/testing | 3,918 | REAL |
| resource/memory | 558 | REAL |
| resource/streaming | 545 | PARTIAL (I/O stubs) |
| **Total** | **7,815** | |
