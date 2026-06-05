# Investigation: engine/debug/profiling

## Summary

The profiling system is a **REAL IMPLEMENTATION** with fully functional CPU timing using `time.perf_counter_ns()`, memory tracking with allocation/deallocation records and leak detection, network profiling with RTT/jitter/packet loss, and comprehensive statistics collection. The GPU profiler explicitly notes it uses CPU-side timing as a placeholder for actual GPU timestamp queries, but the infrastructure is complete and production-ready for integration with real GPU timing APIs.

## Classification

| Component | Status | Evidence |
|-----------|--------|----------|
| CPU Profiling | **REAL** | `time.perf_counter_ns()` nanosecond timing, hierarchical scopes |
| GPU Profiling | **PARTIAL** | CPU-side timing placeholder, full pass/frame structure ready |
| Memory Profiling | **REAL** | Allocation tracking, snapshots, leak detection with confidence scoring |
| Network Profiling | **REAL** | Packet tracking, RTT, jitter, loss detection, connection management |
| Statistics System | **REAL** | Counters, timers, graphs, bar charts with thread-safe implementation |

## Files Analyzed

| File | Lines | Status | Description |
|------|-------|--------|-------------|
| `__init__.py` | 109 | COMPLETE | Full module exports, imports all profiler components |
| `config.py` | 146 | COMPLETE | CVar-based runtime configuration for all profilers |
| `cpu.py` | 401 | COMPLETE | Real nanosecond timing, hierarchical profiling, decorators |
| `gpu.py` | 366 | PARTIAL | CPU-side timing placeholder, awaiting GPU timestamp queries |
| `memory.py` | 523 | COMPLETE | Full allocation tracking, snapshots, diff, leak detection |
| `network.py` | 591 | COMPLETE | Packet tracking, RTT, jitter, loss, connection management |
| `stats.py` | 658 | COMPLETE | Generic statistics: counters, timers, graphs, bar charts |

**Total: 2,794 lines**

---

## CPU Profiler (cpu.py)

### Key Classes

- **`ProfileSample`**: Single profiling sample with timing data
  - `start_ns`, `end_ns`: Nanosecond timestamps via `time.perf_counter_ns()`
  - `parent`, `children`: Hierarchical tree structure
  - `duration_ns`, `duration_ms`: Computed properties
  - `self_time_ns`: Excludes child time for accurate profiling

- **`FlatProfileEntry`**: Aggregated entry for flat view
  - `total_time_ns`, `self_time_ns`, `call_count`
  - `min_time_ns`, `max_time_ns` for outlier detection

- **`CPUProfiler`**: Thread-safe profiler class
  - Per-thread stacks and root samples via `Dict[int, List[ProfileSample]]`
  - `RLock` for thread safety
  - `MAX_COMPLETED_SAMPLES = 10000` to prevent unbounded memory growth

### API

```python
profiler = CPUProfiler()

# Scoped profiling (context manager)
with profiler.scope("render"):
    render_frame()

# Manual profiling
profiler.begin("physics")
physics.step()
profiler.end()

# Get results
hierarchy = profiler.get_hierarchy()  # Tree structure
flat = profiler.get_flat()            # Aggregated, sorted by total time

# Decorator
@profile(name="heavy_computation", warn_ms=16.67)
def process_frame():
    ...
```

### Hierarchical Timing

```python
# Self-time calculation excludes children
@property
def self_time_ns(self) -> int:
    child_time = sum(child.duration_ns for child in self.children)
    return max(0, self.duration_ns - child_time)
```

---

## GPU Profiler (gpu.py)

### Key Classes

- **`GPUPassType`**: Enum for render pass types
  - `SHADOW`, `DEPTH_PREPASS`, `GBUFFER`, `LIGHTING`, `FORWARD`
  - `TRANSPARENT`, `POST_PROCESS`, `UI`, `COMPUTE`, `CUSTOM`

- **`GPUPassTiming`**: Single render pass timing
  - `name`, `pass_type`, `start_ns`, `end_ns`, `frame_index`

- **`GPUFrameTiming`**: Aggregated frame timing
  - `passes: List[GPUPassTiming]`
  - `total_gpu_time_ns`: Sum of all pass durations
  - `frame_time_ns`: Wall-clock frame time

- **`GPUProfiler`**: Frame/pass profiler
  - History managed via `_frame_history` list with configurable size
  - Average calculations over N frames via CVar

### Limitation: CPU-Side Timing

```python
# gpu.py docstring (lines 2-4)
"""GPU Profiler for game engine rendering performance analysis.

Currently uses CPU-side timing as a placeholder for actual GPU timestamp queries.
"""

# Uses time.perf_counter_ns() instead of GPU timestamp queries
self._current_frame = GPUFrameTiming(
    frame_index=self._frame_index,
    frame_start_ns=time.perf_counter_ns()  # CPU-side, not GPU
)
```

### API

```python
profiler = GPUProfiler()

profiler.begin_frame()
profiler.begin_pass("shadow_pass", GPUPassType.SHADOW)
render_shadows()
profiler.end_pass()
profiler.end_frame()

# Get average pass times over recent frames
avg_times = profiler.get_average_pass_times(num_frames=60)
# Returns: {"shadow_pass": 2.5, "forward_pass": 8.3, ...}
```

---

## Memory Profiler (memory.py)

### Key Classes

- **`MemoryTag`**: Allocation categories
  - `RENDERING`, `PHYSICS`, `AUDIO`, `GAMEPLAY`, `AI`
  - `NETWORK`, `UI`, `RESOURCES`, `SCRIPTING`, `DEBUG`, `SYSTEM`, `UNKNOWN`

- **`AllocationRecord`**: Single allocation record
  - `ptr`: Unique ID (simulated pointer)
  - `size`, `tag`, `timestamp`
  - `stack_trace`: Optional, captured via `traceback.extract_stack()`
  - `freed`, `freed_timestamp`
  - `lifetime_seconds`: Computed property

- **`MemorySnapshot`**: Point-in-time memory state
  - `total_allocated`, `allocation_count`
  - `usage_by_tag: Dict[MemoryTag, int]`
  - `allocations: Dict[int, AllocationRecord]`

- **`MemoryDiff`**: Difference between snapshots
  - `delta_total`, `delta_count`
  - `delta_by_tag`, `new_allocations`, `freed_allocations`

- **`LeakCandidate`**: Potential memory leak with confidence scoring
  - `allocation`, `age_seconds`, `confidence`, `reason`

### Leak Detection Algorithm

```python
# Confidence scoring based on multiple factors:
# 1. Age factor (0.4 weight) - longer lived = higher confidence
age_factor = min(1.0, age / (min_age_seconds * 10))
confidence += age_factor * 0.4

# 2. Size factor - larger allocations are more concerning
if record.size > 1MB:
    confidence += 0.3  # Large allocation
elif record.size > 100KB:
    confidence += 0.15  # Medium allocation

# 3. Tag-based suspicion - RENDERING and GAMEPLAY are suspicious
if record.tag in {MemoryTag.RENDERING, MemoryTag.GAMEPLAY}:
    confidence += 0.2

# 4. Stack trace available adds 0.1 to confidence
```

### API

```python
profiler = MemoryProfiler(capture_stack_traces=True)

# Track allocations
ptr = profiler.track_allocation(1024, MemoryTag.RENDERING)
profiler.track_free(ptr)

# Snapshots and diff
profiler.snapshot("before_level_load")
load_level()
profiler.snapshot("after_level_load")
diff = profiler.diff("before_level_load", "after_level_load")

# Leak detection
leaks = profiler.detect_leaks(min_age_seconds=60.0)
for leak in leaks:
    print(f"{leak.allocation.size} bytes, confidence: {leak.confidence}")
```

---

## Network Profiler (network.py)

### Key Classes

- **`PacketType`**: `RELIABLE`, `UNRELIABLE`, `ORDERED`, `SEQUENCED`

- **`PacketDirection`**: `SENT`, `RECEIVED`

- **`PacketRecord`**: Single packet record
  - `packet_id`, `size`, `direction`, `packet_type`, `timestamp`
  - `acknowledged`, `ack_timestamp`, `channel`
  - `rtt_ms`: Computed from ack_timestamp - timestamp

- **`NetworkStats`**: Statistics for a time period
  - `bytes_sent`, `bytes_received`
  - `packets_sent`, `packets_received`
  - `rtt_ms`, `loss_percent`, `jitter_ms`
  - `bandwidth_sent_kbps`, `bandwidth_received_kbps`

- **`ConnectionStats`**: Per-connection statistics
  - `connection_id`, `remote_address`
  - `connected_since`, `last_activity`
  - `rtt_samples`, `average_rtt_ms`

### Jitter Calculation

```python
# Jitter = variation in RTT (average of consecutive differences)
if len(self._rtt_samples) >= 2:
    samples = list(self._rtt_samples)
    diffs = [abs(samples[i] - samples[i-1]) for i in range(1, len(samples))]
    jitter = sum(diffs) / len(diffs)
```

### Packet Loss Detection

```python
# Packets not acknowledged within timeout are considered lost
timeout_threshold = now - (window_seconds * timeout_multiplier)
lost_ids = [
    pid for pid, record in self._sent_packets.items()
    if record.timestamp < timeout_threshold
]
```

### API

```python
profiler = NetworkProfiler()

# Track packets
packet_id = profiler.track_packet_sent(256, PacketType.RELIABLE)
profiler.track_packet_received(128, PacketType.UNRELIABLE)
profiler.track_packet_ack(packet_id)  # Returns RTT in ms

# Connection tracking
conn = profiler.register_connection("player1", "192.168.1.100:7777")
stats = profiler.get_stats()
print(f"RTT: {stats.rtt_ms}ms, Loss: {stats.loss_percent}%")

# Bandwidth history
history = profiler.get_bandwidth_history(duration_seconds=10)
# Returns: [(timestamp, bytes_sent, bytes_received), ...]
```

---

## Statistics System (stats.py)

### Stat Types

- **`CounterStat`**: Incrementing/decrementing counter
- **`TimerStat`**: Time measurements with history, min/max/avg
- **`GraphStat`**: Time-series data for visualization
- **`BarStat`**: Categorical data (key-value pairs)

### Built-in Stat Groups

| Group | Stats |
|-------|-------|
| `fps` | `fps.current`, `fps.frame_time`, `fps.frame_count` |
| `memory` | `memory.used_mb`, `memory.allocated_mb`, `memory.by_category` |
| `gpu` | `gpu.frame_time`, `gpu.pass_times`, `gpu.utilization` |
| `unit` | `unit.total`, `unit.active`, `unit.by_type`, `unit.update_time` |

### API

```python
stats = Stats()

# Counter
stats.counter("entities_spawned", 1)  # Returns new value

# Timer
stats.timer("frame_time", 16.7)

# Graph (time series)
stats.graph("fps", 60.0)

# Bar chart (categorical)
stats.bar("enemies_by_type", "zombie", 15)
stats.bar_increment("enemies_by_type", "skeleton", 1)

# Access stats
fps_stat = stats.get_stat("fps.current")
fps_values = stats.get_value("fps.current")  # List[float]

# Group operations
stats.reset_group("fps")
print(stats.format_group("gpu"))
```

---

## Configuration (config.py)

All profilers use CVars for runtime configuration:

| CVar | Default | Description |
|------|---------|-------------|
| `profiler.gpu.FrameHistorySize` | 120 | Frames to keep in GPU history |
| `profiler.gpu.AverageFrames` | 60 | Frames for GPU timing averages |
| `profiler.memory.StackTraceDepth` | 10 | Max stack trace depth |
| `profiler.memory.LeakMinAgeSeconds` | 60.0 | Min age for leak detection |
| `profiler.memory.FreedHistoryMax` | 10000 | Max freed allocations in history |
| `profiler.memory.FreedHistoryTrim` | 5000 | Trim freed history to this count |
| `profiler.memory.LargeAllocationBytes` | 1MB | Large allocation threshold |
| `profiler.memory.MediumAllocationBytes` | 100KB | Medium allocation threshold |
| `profiler.network.StatsWindowSeconds` | 1.0 | Time window for network stats |
| `profiler.network.PacketHistorySize` | 1000 | Packets to keep in history |
| `profiler.network.RttSampleSize` | 100 | RTT samples for averaging |
| `profiler.network.PacketTimeoutMultiplier` | 5.0 | Multiplier for loss timeout |
| `profiler.network.BandwidthHistorySeconds` | 10.0 | Default bandwidth history duration |
| `profiler.stats.TimerHistorySize` | 100 | Default timer history size |
| `profiler.stats.GraphHistorySize` | 120 | Default graph history size |
| `profiler.cpu.WarnThresholdMs` | 16.67 | Default CPU warning threshold |

---

## Global Default Instances

Each profiler module provides global default instances:

```python
from engine.debug.profiling import (
    get_default_profiler, set_default_profiler,        # CPU
    get_default_gpu_profiler, set_default_gpu_profiler,
    get_default_memory_profiler, set_default_memory_profiler,
    get_default_network_profiler, set_default_network_profiler,
    get_default_stats, set_default_stats,
)
```

---

## Thread Safety

All profilers use thread-safe implementations:

| Profiler | Lock Type | Pattern |
|----------|-----------|---------|
| CPUProfiler | `threading.RLock` | Per-thread stacks, shared completed samples |
| GPUProfiler | `threading.Lock` | Single-threaded rendering assumption |
| MemoryProfiler | `threading.RLock` | Reentrant for nested tracking |
| NetworkProfiler | `threading.Lock` | Single lock for all state |
| Stats | `threading.RLock` | Reentrant for group operations |

---

## Memory Management

All profilers implement bounded memory growth:

- **CPUProfiler**: `MAX_COMPLETED_SAMPLES = 10000`, trims to 5000
- **GPUProfiler**: `_frame_history` with configurable max size
- **MemoryProfiler**: `FreedHistoryMax/Trim` CVars for freed allocations
- **NetworkProfiler**: `deque(maxlen=history_size)` for packets and RTT
- **Stats**: `deque(maxlen=history_size)` for all time-series data

---

## Verdict

**REAL IMPLEMENTATION (PARTIAL GPU)**

The profiling system is production-quality code with:

1. Real nanosecond-precision CPU timing via `time.perf_counter_ns()`
2. Complete memory allocation tracking with leak detection using confidence scoring
3. Full network profiling with RTT, jitter, and packet loss detection
4. Flexible statistics system supporting multiple visualization types
5. Thread-safe implementations throughout
6. CVar-based runtime configuration
7. Bounded memory growth with explicit trimming

The only limitation is GPU timing, which explicitly uses CPU-side timing as a documented placeholder awaiting GPU timestamp query integration. The infrastructure (pass types, frame timing, history management) is fully ready for real GPU timing.

---

## Integration Points

The profiling system integrates with:

1. **CVar system** (`engine.debug.console.cvar`): Runtime configuration
2. **Standard library**: `time.perf_counter_ns()`, `traceback`, `threading`
3. **No external dependencies**: Pure Python implementation

## Recommendations

1. **GPU Timing**: Integrate with wgpu GPU timestamp queries when available
2. **Flame Graph Export**: Add Chrome Tracing JSON export for `chrome://tracing`
3. **Tracy Integration**: Consider Tracy profiler integration for native timing
4. **Real Memory Hooks**: Hook into Python's `tracemalloc` for automatic tracking
