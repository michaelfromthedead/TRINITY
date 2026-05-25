# PHASE 7 TODO: Performance Optimization

## Summary

Object pooling and memory optimization for high-frequency operations.

---

## T-PERF-7.1: Generic Object Pool

**File**: `engine/core/pool.py`

### Tasks

- [ ] Implement ObjectPool[T] generic class
- [ ] Implement acquire() method with auto-grow
- [ ] Implement release(obj) method with reset callback
- [ ] Implement clear() method for pool teardown
- [ ] Implement stats() method for monitoring
- [ ] Add initial_size and max_size parameters
- [ ] Add thread-safety option (optional lock)

### Acceptance Criteria

- Pool correctly recycles objects
- Auto-grow when pool empty
- Reset function called on release
- Stats report available/in-use counts

---

## T-PERF-7.2: ItemInstance Pool

**File**: `engine/gameplay/economy/pool.py`

### Tasks

- [ ] Create ItemInstancePool using generic pool
- [ ] Implement reset_item_instance() clearing all state
- [ ] Implement acquire_item(definition, quantity) convenience method
- [ ] Integrate pool with InventoryContainer.add_item()
- [ ] Integrate pool with ItemInstance destructor/cleanup
- [ ] Add pool metrics to economy module

### Acceptance Criteria

- 1000 item creates/destroys with no new allocations after warmup
- Stack splits reuse pooled instances
- Merge operations return instances to pool

---

## T-PERF-7.3: InputEvent Pool

**File**: `engine/gameplay/input/pool.py`

### Tasks

- [ ] Create InputEventPool with frame-based bulk release
- [ ] Implement reset_input_event() clearing all fields
- [ ] Implement acquire_event(device, type) convenience method
- [ ] Implement release_frame() for end-of-frame cleanup
- [ ] Integrate with DeviceManager event dispatch
- [ ] Add pool metrics to input module

### Acceptance Criteria

- 10000 events per frame with no allocations after warmup
- All events returned to pool at frame end
- No stale event references after frame

---

## T-PERF-7.4: Batched ID Generator

**File**: `engine/core/id_generator.py`

### Tasks

- [ ] Implement IdGenerator with batch reservation
- [ ] Implement next_id() with local batch consumption
- [ ] Implement batch_size parameter (default 1000)
- [ ] Implement thread-local batches for multi-threaded access
- [ ] Add metrics for batch reservation frequency
- [ ] Integrate with Actor ID generation
- [ ] Integrate with Item ID generation

### Acceptance Criteria

- Lock acquired once per batch_size IDs
- 10000 ID generations < 0.5ms
- Thread-safe for concurrent actor spawning

---

## T-PERF-7.5: Weak Reference Audit

**File**: `tests/performance/test_weak_refs.py`

### Tasks

- [ ] Test parent destruction leaves no child reference to parent
- [ ] Test child destruction removes from parent.children
- [ ] Test controller destruction unpossesses pawn
- [ ] Test pawn destruction clears controller.pawn
- [ ] Test no reference cycles in actor hierarchy (gc.garbage empty)
- [ ] Test no reference cycles in possession system
- [ ] Test weakref.ref() used correctly (not weakref.proxy())

### Acceptance Criteria

- All tests pass
- gc.collect() returns to baseline object count
- No objects in gc.garbage after test

---

## T-PERF-7.6: Memory Stress Tests

**File**: `tests/performance/test_memory_stress.py`

### Tasks

- [ ] Test 10000 actor spawn/destroy cycle
- [ ] Test 10000 item create/destroy cycle
- [ ] Test 10000 input event create/release cycle
- [ ] Measure peak memory with tracemalloc
- [ ] Verify return to baseline after cycle
- [ ] Verify no growth over repeated cycles

### Acceptance Criteria

- Peak memory < 2x baseline during stress
- Return to baseline after stress (within 10%)
- No growth over 10 stress cycles

---

## T-PERF-7.7: Pool Benchmarks

**File**: `tests/performance/test_pool_bench.py`

### Tasks

- [ ] Benchmark ItemInstance unpooled creation (1000x)
- [ ] Benchmark ItemInstance pooled creation (1000x)
- [ ] Benchmark InputEvent unpooled creation (10000x)
- [ ] Benchmark InputEvent pooled creation (10000x)
- [ ] Benchmark ID generation unpooled (10000x)
- [ ] Benchmark ID generation batched (10000x)
- [ ] Generate comparison report

### Acceptance Criteria

- Pooled operations at least 10x faster
- Batched ID generation at least 10x faster
- Benchmarks reproducible (low variance)

---

## T-PERF-7.8: Lock Contention Analysis

**File**: `tests/performance/test_lock_contention.py`

### Tasks

- [ ] Test concurrent ID generation from multiple threads
- [ ] Measure lock wait time
- [ ] Compare batched vs unbatched under contention
- [ ] Test pool access under contention
- [ ] Verify no deadlocks under stress

### Acceptance Criteria

- Batched ID reduces lock acquisitions by 1000x
- No deadlocks under 100-thread stress test
- Lock wait time < 1% of total time
