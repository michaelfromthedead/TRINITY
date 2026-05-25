# Investigation: engine/resource/memory/

**Date:** 2026-05-22  
**Classification:** REAL (fully implemented production code)

## Summary

The `engine/resource/memory/` module provides a complete, production-ready memory management system for game assets. All five files contain real implementations with no stubs, placeholders, or TODO markers. The architecture follows a clean separation of concerns with four distinct subsystems working together.

## Files Analyzed

| File | Lines | Classification | Purpose |
|------|-------|----------------|---------|
| `residency_manager.py` | 162 | REAL | Orchestrates asset lifecycle and coordinates budget/eviction |
| `eviction.py` | 133 | REAL | Pluggable eviction policies (LRU, LFU, Size, Priority) |
| `budget_manager.py` | 120 | REAL | Per-category memory budget tracking |
| `asset_pool.py` | 90 | REAL | Generic object pooling with slot-based allocation |
| `__init__.py` | 53 | REAL | Public API exports |

**Total:** 558 lines

## Architecture

```
                    ┌─────────────────────┐
                    │  ResidencyManager   │
                    │  (Orchestrator)     │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
    ┌─────────────────┐ ┌──────────────┐ ┌───────────────┐
    │  BudgetManager  │ │EvictionMgr  │ │  AssetPool    │
    │  (Per-Category  │ │(LRU/LFU/    │ │  (Slot-Based  │
    │   Allocation)   │ │ Size/Prio)  │ │   Pooling)    │
    └─────────────────┘ └──────────────┘ └───────────────┘
```

## Subsystem Details

### 1. Budget Manager (`budget_manager.py`)

Per-category memory budgeting with allocation tracking.

**Asset Categories:**
- `TEXTURE` - with default budget from constants
- `MESH` - with default budget from constants
- `AUDIO` - with default budget from constants
- `ANIMATION` - no default budget
- `SHADER` - no default budget
- `MATERIAL` - no default budget
- `OTHER` - no default budget

**Key Features:**
- `allocate(category, size_bytes) -> bool` - Atomic allocation with budget check
- `free(category, size_bytes)` - Release allocated memory
- `is_over_budget(category) -> bool` - Budget violation detection
- `get_pressure() -> float` - Overall memory pressure ratio (0.0-1.0)
- Peak usage tracking per category

**Implementation Quality:** Complete with proper error handling. Uses `__slots__` for memory efficiency. Tracks peak bytes for profiling.

### 2. Eviction Policies (`eviction.py`)

Strategy pattern implementation with four pluggable policies.

| Policy | Selection Strategy | Use Case |
|--------|-------------------|----------|
| `LRUEviction` | Oldest access time first | General purpose caching |
| `LFUEviction` | Lowest access count first | Frequency-based caching |
| `SizeEviction` | Largest assets first | Quick memory reclamation |
| `PriorityEviction` | Lowest priority first | Priority-aware streaming |

**Key Types:**
- `EvictionCandidate` - Metadata for eviction decisions (asset_id, size_bytes, last_access_time, access_count, priority)
- `EvictionPolicy` - Abstract base class
- `EvictionManager` - Coordinates candidate tracking and policy execution

**Implementation Quality:** Clean strategy pattern. Helper `_collect_until()` avoids code duplication. All policies iterate sorted candidates until `bytes_needed` is satisfied.

### 3. Residency Manager (`residency_manager.py`)

Central coordinator that integrates budget management and eviction.

**Residency States:**
```
NON_RESIDENT ──► LOADING ──► RESIDENT
                              │
                              ▼
                          EVICTING ──► NON_RESIDENT
```

**Key Operations:**
- `request_residency(asset_id, size_bytes, priority) -> bool` - Request asset loading with budget check
- `release_residency(asset_id)` - Explicit release
- `touch(asset_id)` - Update access time for LRU
- `update() -> list[int]` - Run eviction cycle, returns evicted IDs

**Integration Points:**
- Calls `BudgetManager.allocate()` before allowing residency
- Calls `BudgetManager.free()` on release/eviction
- Adds/removes `EvictionCandidate` on state transitions
- Uses injectable `time_fn` for testability

**Implementation Quality:** Well-integrated. State machine is implicit but transitions are correct. `__slots__` used throughout for memory efficiency.

### 4. Asset Pool (`asset_pool.py`)

Generic object pool with pre-allocated slots.

**Features:**
- Type-generic via `Generic[T]`
- Pre-allocates `capacity` slots on construction
- Free-list tracking with stack-based allocation (LIFO for cache locality)
- `acquire(obj) -> (slot_id, obj)` - O(1) allocation
- `release(slot_id)` - O(1) deallocation
- `reset()` - Bulk release all slots

**Implementation Quality:** Production-ready. Proper validation on acquire/release. Uses reversed range for free-list to give sequential allocation initially.

## External Dependencies

The module imports from `engine.resource.constants`:
- `DEFAULT_TEXTURE_BUDGET`
- `DEFAULT_MESH_BUDGET`
- `DEFAULT_AUDIO_BUDGET`
- `DEFAULT_POOL_CAPACITY`

These constants define the baseline budgets and pool sizing.

## Design Patterns Used

1. **Strategy Pattern** - `EvictionPolicy` with interchangeable implementations
2. **Object Pool** - `AssetPool` for reusable allocations
3. **Coordinator** - `ResidencyManager` orchestrates subsystems
4. **Data Transfer Objects** - `EvictionCandidate`, `ResidencyInfo`, `BudgetEntry`

## Memory Efficiency Techniques

- All dataclasses use `__slots__` to eliminate `__dict__` overhead
- Explicit `__init__` methods in dataclasses (compatible with slots)
- Stack-based free-list in `AssetPool`
- Peak tracking for memory profiling

## Testing Considerations

- `ResidencyManager` accepts `time_fn` for deterministic testing
- All classes have clear boundaries for unit testing
- Eviction policies are independent and easily testable
- Pool operations have explicit error conditions

## Gaps and Observations

1. **No async support** - All operations are synchronous. Loading/evicting states exist but no async machinery.
2. **No streaming integration** - The residency manager tracks state but doesn't trigger actual I/O.
3. **No metrics/telemetry** - Peak tracking exists but no hooks for external monitoring.
4. **ANIMATION/SHADER/MATERIAL categories** - Defined but have no default budgets set.
5. **No composite eviction** - Cannot combine policies (e.g., LRU with priority weighting).

## Classification Rationale

**REAL** - All code is complete implementation:
- No `pass` statements or `raise NotImplementedError`
- No TODO/FIXME comments
- No stub methods or placeholder logic
- Full error handling throughout
- Uses engine constants (not hardcoded test values)
- Proper `__all__` exports

This module is production-ready and follows Python best practices for game engine memory management.
