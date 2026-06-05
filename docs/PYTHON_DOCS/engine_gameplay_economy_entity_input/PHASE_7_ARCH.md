# PHASE 7 ARCHITECTURE: Performance Optimization

## Overview

Optimize memory patterns for high-frequency operations through object pooling and reference analysis.

## Performance Targets

### High-Churn Objects

| Object | Creation Rate | Lifetime | Pool Priority |
|--------|---------------|----------|---------------|
| ItemInstance | High (loot drops, stack splits) | Variable | **HIGH** |
| InputEvent | Very high (every frame) | Single frame | **HIGH** |
| Transform | Medium (actor spawns) | Long | Low |
| ActionResult | High (every action) | Single frame | Medium |

### Memory Patterns

| Pattern | Current State | Target State |
|---------|--------------|--------------|
| ItemInstance allocation | New allocation per create | Pool-allocated |
| InputEvent allocation | New allocation per event | Pool-allocated |
| Actor parent/child refs | Weak references (good) | Verify no leaks |
| ID generation | Lock per ID | Batch ID reservation |

## Architecture Decisions

### ADR-PERF-1: Object Pool Design

Generic object pool with type-specific reset:
```python
class ObjectPool[T]:
    def __init__(self, factory: Callable[[], T], reset: Callable[[T], None], initial_size: int = 32):
        self._pool: list[T] = [factory() for _ in range(initial_size)]
        self._factory = factory
        self._reset = reset
    
    def acquire(self) -> T:
        if self._pool:
            return self._pool.pop()
        return self._factory()
    
    def release(self, obj: T) -> None:
        self._reset(obj)
        self._pool.append(obj)
```

### ADR-PERF-2: ItemInstance Pool

ItemInstance pool with quantity/definition reset:
```python
def reset_item_instance(item: ItemInstance) -> None:
    item.definition = None
    item.quantity = 0
    item._id = None  # Reassigned on acquire

item_pool = ObjectPool(ItemInstance, reset_item_instance)
```

### ADR-PERF-3: InputEvent Pool

InputEvent pool with per-frame bulk release:
```python
class InputEventPool:
    def __init__(self):
        self._pool = ObjectPool(InputEvent, reset_input_event)
        self._in_use: list[InputEvent] = []
    
    def acquire(self) -> InputEvent:
        event = self._pool.acquire()
        self._in_use.append(event)
        return event
    
    def release_frame(self) -> None:
        for event in self._in_use:
            self._pool.release(event)
        self._in_use.clear()
```

### ADR-PERF-4: ID Generation Optimization

Batch ID reservation to reduce lock contention:
```python
class IdGenerator:
    def __init__(self, batch_size: int = 1000):
        self._lock = Lock()
        self._batch_size = batch_size
        self._current_batch: list[int] = []
        self._next_batch_start = 0
    
    def next_id(self) -> int:
        if not self._current_batch:
            with self._lock:
                start = self._next_batch_start
                self._next_batch_start += self._batch_size
            self._current_batch = list(range(start, start + self._batch_size))
        return self._current_batch.pop()
```

### ADR-PERF-5: Weak Reference Audit

Verify weak reference patterns are correct:
- Parent -> children: strong references (parent owns children)
- Child -> parent: weak reference (prevents cycle)
- Controller -> pawn: strong reference (controller owns possession)
- Pawn -> controller: weak reference (prevents cycle)

## Profiling Strategy

### Benchmarks to Create

| Benchmark | Metric | Target |
|-----------|--------|--------|
| ItemInstance creation (1000x) | Time (ms) | < 1ms |
| ItemInstance pooled (1000x) | Time (ms) | < 0.1ms |
| InputEvent creation (10000x) | Time (ms) | < 10ms |
| InputEvent pooled (10000x) | Time (ms) | < 1ms |
| ID generation (10000x) | Time (ms) | < 5ms |
| ID generation batched (10000x) | Time (ms) | < 0.5ms |

### Memory Metrics

| Metric | Tool | Target |
|--------|------|--------|
| Peak memory during stress test | tracemalloc | No growth |
| Object count after stress test | gc.get_objects | Return to baseline |
| Reference cycles | gc.garbage | Zero |

## Files to Create/Modify

```
engine/
  core/
    pool.py                 # Generic ObjectPool
    id_generator.py         # Batched IdGenerator
  gameplay/
    economy/
      pool.py               # ItemInstance pool
    input/
      pool.py               # InputEvent pool

tests/
  performance/
    test_pool_bench.py      # Pool benchmarks
    test_memory_audit.py    # Memory pattern tests
```

## Risks

| Risk | Mitigation |
|------|------------|
| Pool exhaustion | Auto-grow with allocation |
| Stale references in pooled objects | Mandatory reset function |
| Thread safety | Pool-per-thread or explicit locking |
| Premature optimization | Benchmark before and after |
