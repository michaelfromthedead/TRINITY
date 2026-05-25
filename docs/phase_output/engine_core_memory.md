# Investigation: engine/core/memory

## Summary

The memory subsystem contains **fully functional Python implementations** of all 6 documented allocator types plus memory tracking and object pooling. Each allocator is backed by real `bytearray` storage with proper allocation logic, bounds checking, and O(1) operations where promised. This is production-quality code, not stubs.

**Classification: 10/10 REAL implementations, 0 STUBS**

## Files

| File | Lines | Status | Classification |
|------|-------|--------|----------------|
| `__init__.py` | 26 | Complete | REAL - Proper module exports |
| `allocator.py` | 65 | Complete | REAL - ABC + MemoryTag enum + AllocationInfo dataclass |
| `linear.py` | 62 | Complete | REAL - Bump-pointer allocator |
| `stack.py` | 75 | Complete | REAL - LIFO allocator with marker API |
| `pool.py` | 89 | Complete | REAL - Fixed-size slot pool |
| `ring.py` | 66 | Complete | REAL - Circular buffer allocator |
| `slab.py` | 112 | Complete | REAL - Multi-pool with size classes |
| `tlsf.py` | 207 | Complete | REAL - Two-Level Segregated Fit |
| `tracker.py` | 75 | Complete | REAL - Per-tag statistics |
| `object_pool.py` | 56 | Complete | REAL - Generic object recycling |

**Total: 833 lines**

## Allocator Types Analysis

### 1. Linear (Bump) Allocator - `linear.py`

**Classification: REAL**

| Criteria | Status |
|----------|--------|
| Real backing storage | Yes - `bytearray(capacity)` |
| Working allocation | Yes - offset increment with bounds check |
| Free semantics | Correct - no-op (reset-only as documented) |
| Error handling | Yes - `MemoryError` on OOM, `ValueError` on invalid size |
| Complexity | O(1) allocation as documented |

```python
def allocate(self, size: int) -> int:
    if self._offset + size > self._capacity:
        raise MemoryError(...)
    offset = self._offset
    self._offset += size
    return offset
```

### 2. Stack (LIFO) Allocator - `stack.py`

**Classification: REAL**

| Criteria | Status |
|----------|--------|
| Real backing storage | Yes - `bytearray(capacity)` |
| Marker API | Yes - `get_marker()` / `free_to_marker()` |
| LIFO semantics | Yes - free unwinds to offset |
| Error handling | Yes - bounds validation |

```python
def get_marker(self) -> int:
    return self._offset

def free_to_marker(self, marker: int) -> None:
    if marker < 0 or marker > self._offset:
        raise ValueError(...)
    self._offset = marker
```

### 3. Pool (Fixed-size) Allocator - `pool.py`

**Classification: REAL**

| Criteria | Status |
|----------|--------|
| Real backing storage | Yes - `bytearray(element_size * count)` |
| Free-list | Yes - LIFO list + set for O(1) lookup |
| Double-free detection | Yes - checks `_free_set` before free |
| Slot-based allocation | Yes - returns index, not byte offset |

```python
def free(self, offset: int) -> None:
    if offset in self._free_set:
        raise ValueError(f"double free on slot {offset}")
    self._free_list.append(offset)
    self._free_set.add(offset)
```

### 4. Ring (Circular) Allocator - `ring.py`

**Classification: REAL**

| Criteria | Status |
|----------|--------|
| Real backing storage | Yes - `bytearray(capacity)` |
| Wrap-around | Yes - modulo arithmetic on head |
| Implicit free | Yes - documented no-op free |
| Overflow handling | Yes - prevents single allocation > capacity |

```python
def allocate(self, size: int) -> int:
    offset = self._head
    self._head = (self._head + size) % self._capacity
    self._used = min(self._used + size, self._capacity)
    return offset
```

### 5. Slab (Size-class) Allocator - `slab.py`

**Classification: REAL**

| Criteria | Status |
|----------|--------|
| Multi-pool composition | Yes - Dict[int, PoolAllocator] per size class |
| Size routing | Yes - picks smallest fitting class |
| Encoded offset | Yes - `(sc_idx << 32) | index` |
| Legacy API | Yes - `allocate_slab()` / `free_slab()` tuple interface |

```python
def allocate(self, size: int) -> int:
    sc = self._pick_class(size)
    index = self._pools[sc].allocate()
    sc_idx = self._size_classes.index(sc)
    return (sc_idx << 32) | index
```

### 6. TLSF (Two-Level Segregated Fit) - `tlsf.py`

**Classification: REAL**

This is the most sophisticated allocator in the module.

| Criteria | Status |
|----------|--------|
| Real backing storage | Yes - `bytearray(capacity)` |
| Two-level indexing | Yes - first-level (power-of-two) + second-level (linear subdivisions) |
| O(1) bitmap search | Yes - `_fl_bitmap` and `_sl_bitmaps` with bit manipulation |
| Block coalescing | Yes - `_coalesce()` merges adjacent free blocks |
| Size tracking | Yes - `_alloc_sizes` dict for size-less free |

```python
def _find_suitable(self, fl: int, sl: int) -> Optional[_Block]:
    # O(1) search via bitmap manipulation
    sl_map = self._sl_bitmaps.get(fl, 0) & (~0 << sl)
    if sl_map:
        found_sl = (sl_map & -sl_map).bit_length() - 1
        blocks = self._free_lists.get((fl, found_sl))
        if blocks:
            return blocks[0]
```

**Coalescing implementation:**
```python
def _coalesce(self, block: _Block) -> None:
    # Merges with adjacent free blocks in both directions
    if block.offset + block.size == other.offset:
        # Forward merge
    if other.offset + other.size == block.offset:
        # Backward merge
```

### 7. Memory Tracker - `tracker.py`

**Classification: REAL**

| Criteria | Status |
|----------|--------|
| Per-tag statistics | Yes - MemoryStats dataclass per MemoryTag |
| Live allocation tracking | Yes - `_live: Dict[int, AllocationInfo]` |
| Peak tracking | Yes - updates peak on each allocation |
| Leak detection | Yes - `get_live_allocations()` returns unfreed |

### 8. Object Pool - `object_pool.py`

**Classification: REAL**

| Criteria | Status |
|----------|--------|
| Generic typing | Yes - `Generic[T]` with TypeVar |
| Factory pattern | Yes - `Callable[[], T]` for object creation |
| Reset callback | Yes - optional `reset_func` on release |
| Max size limit | Yes - discards excess released objects |

## Dependencies

External imports from `engine.core.constants`:
- `DEFAULT_SLAB_SIZE_CLASSES`
- `DEFAULT_SLAB_SLOTS_PER_CLASS`
- `TLSF_SL_COUNT`
- `TLSF_SL_BITS`
- `TLSF_MIN_BLOCK`

## Architecture

```
Allocator (ABC)
    |
    +-- LinearAllocator    (bump pointer, reset-only)
    +-- StackAllocator     (LIFO with markers)
    +-- PoolAllocator      (fixed-size slots)
    +-- RingAllocator      (circular buffer)
    +-- SlabAllocator      (composes PoolAllocators)
    +-- TLSFAllocator      (general-purpose, O(1))

MemoryTracker (orthogonal, tracks any allocator)
ObjectPool[T] (generic, separate from byte allocators)
```

## Key Design Patterns

1. **ABC Compliance**: All byte allocators implement `Allocator` interface
2. **Composition**: SlabAllocator delegates to PoolAllocator instances
3. **Separation of Concerns**: MemoryTracker is independent of allocator type
4. **Type Safety**: Full type hints, `slots=True` for dataclasses, Generic typing

## Test Coverage Indicators

- Double-free detection suggests defensive coding for tests
- Logging at DEBUG level enables test observability
- Clear error messages with context (sizes, offsets)

## Verdict

**ALL REAL IMPLEMENTATIONS**

This module is production-quality code with:
- Real `bytearray` backing for all byte allocators
- Correct algorithmic implementations (TLSF bitmap search, coalescing)
- Proper error handling and validation
- Clean architecture with composition and inheritance
- Full type hints and documentation

No stubs, placeholders, or TODO markers found.
