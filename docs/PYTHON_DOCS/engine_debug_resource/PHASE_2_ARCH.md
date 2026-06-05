# PHASE 2 ARCHITECTURE - Integration (PENDING)

**Workflow:** RDC_WORKFLOW v1.2.0  
**Cluster:** engine_debug_resource  
**Status:** PENDING  
**Generated:** 2026-05-23

---

## Overview

Phase 2 addresses the integration gaps identified in the source investigations. These are not architectural changes - the foundations are sound - but connections to external systems and async I/O that require implementation.

---

## 1. GPU Timing Integration

### 1.1 Current State

The GPU profiler uses CPU-side timing as a placeholder:

```python
# Current (cpu.py)
self._current_frame = GPUFrameTiming(
    frame_index=self._frame_index,
    frame_start_ns=time.perf_counter_ns()  # CPU-side
)
```

### 1.2 Target Architecture

GPU timestamp queries via wgpu:

```python
# Target
class GPUProfiler:
    def __init__(self, device: wgpu.GPUDevice):
        self._timestamp_query_set = device.create_query_set(
            type="timestamp",
            count=256  # Max passes per frame * 2 (start + end)
        )
        self._timestamp_buffer = device.create_buffer(...)
    
    def begin_pass(self, name: str, pass_type: GPUPassType, encoder: wgpu.GPUCommandEncoder):
        encoder.write_timestamp(self._timestamp_query_set, self._next_query_index)
        self._next_query_index += 1
    
    def end_pass(self, encoder: wgpu.GPUCommandEncoder):
        encoder.write_timestamp(self._timestamp_query_set, self._next_query_index)
        self._next_query_index += 1
    
    def resolve_frame(self) -> None:
        # Resolve timestamp queries to buffer
        # Read buffer (async or sync)
        # Convert GPU ticks to nanoseconds using timestamp period
```

### 1.3 Integration Points

| Component | Integration |
|-----------|-------------|
| wgpu.GPUDevice | Create timestamp query set |
| wgpu.GPUCommandEncoder | Write timestamps |
| wgpu.GPUBuffer | Resolve and read timestamps |
| Renderer | Pass encoder to profiler |

### 1.4 Fallback

If GPU timestamps are unavailable (feature not supported), fall back to current CPU-side timing with a warning.

---

## 2. Async I/O for Streaming

### 2.1 Current State

Stream manager simulates progress synchronously:

```python
# Current (stream_manager.py)
for rid, req in self._active.items():
    req.bytes_loaded = min(req.bytes_loaded + req.bytes_total, req.bytes_total)
    if req.bytes_loaded >= req.bytes_total:
        req.state = StreamState.COMPLETE
```

### 2.2 Target Architecture

Async file I/O with callback completion:

```python
# Target
class AsyncStreamManager:
    def __init__(self, executor: concurrent.futures.ThreadPoolExecutor):
        self._executor = executor
        self._futures: dict[int, concurrent.futures.Future] = {}
    
    def request_stream(
        self,
        asset_id: str,
        priority: StreamPriority,
        stream_type: StreamType,
        path: Path,
        on_complete: Callable[[bytes], None]
    ) -> StreamRequest:
        request = StreamRequest(...)
        future = self._executor.submit(self._load_asset, path)
        future.add_done_callback(lambda f: self._on_load_complete(request, f, on_complete))
        self._futures[request.request_id] = future
        return request
    
    def _load_asset(self, path: Path) -> bytes:
        with open(path, "rb") as f:
            return f.read()
    
    def _on_load_complete(
        self,
        request: StreamRequest,
        future: concurrent.futures.Future,
        callback: Callable[[bytes], None]
    ) -> None:
        try:
            data = future.result()
            request.state = StreamState.COMPLETE
            callback(data)
        except Exception as e:
            request.state = StreamState.FAILED
```

### 2.3 Alternative: asyncio

```python
# Alternative with asyncio
class AsyncStreamManager:
    async def stream_asset(self, path: Path) -> bytes:
        async with aiofiles.open(path, "rb") as f:
            return await f.read()
```

### 2.4 Integration Points

| Component | Integration |
|-----------|-------------|
| ThreadPoolExecutor | Background I/O threads |
| Asset Loader | Deserialize loaded bytes |
| Budget Manager | Check budget before loading |
| Residency Manager | Update state on completion |

---

## 3. Budget Enforcement in Streaming

### 3.1 Current State

Budget constants are defined but not used by streaming:

```python
# constants.py
DEFAULT_TEXTURE_BUDGET: int = 512 * _BYTES_PER_MB   # 512 MB
DEFAULT_MESH_BUDGET: int = 256 * _BYTES_PER_MB       # 256 MB
DEFAULT_AUDIO_BUDGET: int = 128 * _BYTES_PER_MB      # 128 MB

# streaming code never references these
```

### 3.2 Target Architecture

Stream manager integrates with budget manager:

```python
# Target
class BudgetAwareStreamManager:
    def __init__(
        self,
        stream_manager: StreamManager,
        budget_manager: BudgetManager,
        residency_manager: ResidencyManager
    ):
        self._stream_manager = stream_manager
        self._budget_manager = budget_manager
        self._residency_manager = residency_manager
    
    def request_stream(
        self,
        asset_id: str,
        size_bytes: int,
        category: AssetCategory,
        priority: StreamPriority,
        stream_type: StreamType
    ) -> Optional[StreamRequest]:
        # Check budget
        if not self._budget_manager.allocate(category, size_bytes):
            # Trigger eviction
            pressure = self._budget_manager.get_pressure()
            if pressure > 0.9:
                evicted = self._residency_manager.update()
                # Retry after eviction
                if not self._budget_manager.allocate(category, size_bytes):
                    return None  # Still over budget
        
        # Request residency
        if not self._residency_manager.request_residency(asset_id, size_bytes, priority):
            self._budget_manager.free(category, size_bytes)
            return None
        
        # Create stream request
        return self._stream_manager.request_stream(asset_id, priority, stream_type)
```

### 3.3 Integration Points

| Component | Integration |
|-----------|-------------|
| BudgetManager | Pre-check allocation |
| ResidencyManager | Track residency state |
| EvictionManager | Evict when over budget |
| StreamManager | Queue stream request |

---

## 4. Testing Framework Integration

### 4.1 Screenshot Action

Current state - placeholder:

```python
# Current (automation.py)
def _do_screenshot(self, action: Action) -> None:
    # Just logs, no actual capture
    print(f"Screenshot: {action.value}")
```

Target:

```python
# Target
def _do_screenshot(self, action: Action) -> None:
    renderer = self._get_renderer()
    pixels = renderer.read_framebuffer()
    path = self._screenshot_dir / f"{action.value}.png"
    Image.frombytes("RGBA", renderer.size, pixels).save(path)
```

### 4.2 Checkpoint/Restore

Current state - placeholder:

```python
# Current (automation.py)
def _do_checkpoint(self, action: Action) -> None:
    # State tracking only
    self._checkpoints[action.value] = {}
```

Target:

```python
# Target
def _do_checkpoint(self, action: Action) -> None:
    state = self._game_state_serializer.serialize()
    self._checkpoints[action.value] = state

def _do_restore(self, action: Action) -> None:
    state = self._checkpoints[action.value]
    self._game_state_serializer.deserialize(state)
```

### 4.3 InputSimulator Game Integration

Current state - interface only:

```python
# Current
def simulate_key(self, key: str, modifiers: list[str], press: bool, release: bool) -> None:
    # No actual input injection
    pass
```

Target:

```python
# Target
def simulate_key(self, key: str, modifiers: list[str], press: bool, release: bool) -> None:
    input_system = self._get_input_system()
    if press:
        for mod in modifiers:
            input_system.inject_key_down(mod)
        input_system.inject_key_down(key)
    if release:
        input_system.inject_key_up(key)
        for mod in reversed(modifiers):
            input_system.inject_key_up(mod)
```

---

## 5. Optional Enhancements

### 5.1 Tracy Profiler Integration

For native C++/Rust profiling integration:

```python
class TracyProfiler:
    def __init__(self):
        self._tracy = ctypes.CDLL("tracy_client.so")
    
    def begin_zone(self, name: str) -> int:
        return self._tracy.___tracy_emit_zone_begin_callstack(name.encode(), 0, 1)
    
    def end_zone(self, zone_id: int) -> None:
        self._tracy.___tracy_emit_zone_end(zone_id)
```

### 5.2 Chrome Tracing Export

For `chrome://tracing` visualization:

```python
def export_chrome_tracing(profiler: CPUProfiler, path: Path) -> None:
    events = []
    for sample in profiler.get_hierarchy():
        events.append({
            "name": sample.name,
            "cat": "cpu",
            "ph": "X",  # Complete event
            "ts": sample.start_ns / 1000,  # Microseconds
            "dur": sample.duration_ns / 1000,
            "pid": 0,
            "tid": sample.thread_id
        })
    with open(path, "w") as f:
        json.dump({"traceEvents": events}, f)
```

### 5.3 Composite Eviction Policies

For combined LRU + Priority eviction:

```python
class CompositeEviction(EvictionPolicy):
    def __init__(self, policies: list[tuple[EvictionPolicy, float]]):
        self._policies = policies  # Policy + weight
    
    def select(self, candidates: list[EvictionCandidate], bytes_needed: int) -> list[EvictionCandidate]:
        # Score each candidate by weighted policy scores
        scored = []
        for candidate in candidates:
            score = 0.0
            for policy, weight in self._policies:
                policy_score = self._get_policy_score(policy, candidate, candidates)
                score += policy_score * weight
            scored.append((score, candidate))
        
        # Return highest-scoring until bytes_needed satisfied
        scored.sort(reverse=True, key=lambda x: x[0])
        return self._collect_until(scored, bytes_needed)
```

---

## Dependencies

### New Dependencies for Phase 2

| Dependency | Purpose | Module |
|------------|---------|--------|
| wgpu | GPU timestamp queries | GPU profiler |
| aiofiles (optional) | Async file I/O | Streaming |
| concurrent.futures | Thread pool I/O | Streaming |
| PIL/Pillow | Screenshot capture | Automation |

### Integration Requirements

| System | Required Interface |
|--------|-------------------|
| Renderer | `read_framebuffer()`, `create_query_set()` |
| Game State | `serialize()`, `deserialize()` |
| Input System | `inject_key_down()`, `inject_key_up()` |
| Asset Loader | `deserialize(bytes, asset_type)` |

---

## Verification Criteria

Phase 2 is complete when:

1. GPU profiler uses real GPU timestamps (with CPU fallback)
2. Stream manager performs async file I/O
3. Streaming checks budget before loading
4. Eviction triggers when over budget
5. Screenshot action captures framebuffer
6. Checkpoint/restore serializes game state
7. InputSimulator injects into game input system
