# Memory & ECS Module Evaluation

**Modules:** renderer-backend::memory, component_store, type_registry
**Location:** `/crates/renderer-backend/src/`
**Lines:** ~2,500
**Quality Grade:** A

---

## Purpose

Memory allocators for GPU operations and SoA (Structure of Arrays) ECS storage for cache-efficient component access.

---

## File Inventory

| File | Lines | Purpose | Quality |
|------|-------|---------|---------|
| memory.rs | 569 | GPU memory allocators | A |
| component_store.rs | 658 | SoA ECS storage | A |
| type_registry.rs | 298 | Component type info | A |

---

## Memory Allocators (memory.rs)

### FrameAllocator

Bump-pointer allocator for per-frame transient data.

```rust
pub struct FrameAllocator {
    buffer: Vec<u8>,
    offset: usize,
    capacity: usize,
}

impl FrameAllocator {
    pub fn new(capacity: usize) -> Self;
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Option<&mut [u8]>;
    pub fn reset(&mut self);
    pub fn high_water_mark(&self) -> usize;
}
```

**Properties:**
- O(1) allocation (pointer bump)
- Zero fragmentation
- Batch deallocation via reset()
- Alignment support (round up)

### PoolAllocator

Fixed-size block allocator for uniform allocations.

```rust
pub struct PoolAllocator {
    pools: [BlockPool; 4],  // 64KB, 256KB, 1MB, 4MB
}

impl PoolAllocator {
    pub fn acquire(&mut self, size: usize) -> Option<PoolHandle>;
    pub fn release(&mut self, handle: PoolHandle);
}
```

**Block sizes:** 64KB, 256KB, 1MB, 4MB

### StackAllocator

LIFO allocator for nested staging.

```rust
pub struct StackAllocator {
    buffer: Vec<u8>,
    stack: Vec<usize>,  // Saved offsets
}

impl StackAllocator {
    pub fn push(&mut self) -> usize;
    pub fn pop(&mut self);
    pub fn allocate(&mut self, size: usize) -> Option<&mut [u8]>;
}
```

### GpuBudget

Atomic budget tracker for capacity planning.

```rust
pub struct GpuBudget {
    total: AtomicUsize,
    used: AtomicUsize,
}

impl GpuBudget {
    pub fn try_allocate(&self, size: usize) -> bool;
    pub fn release(&self, size: usize);
    pub fn usage_percent(&self) -> f32;
}
```

---

## Component Store (component_store.rs)

### SoA Storage Architecture

```rust
pub struct ComponentStore {
    archetypes: HashMap<ArchetypeId, Archetype>,
    entity_to_archetype: HashMap<u64, (ArchetypeId, usize)>,
    type_registry: Arc<RwLock<TypeRegistry>>,
}

pub struct Archetype {
    pub id: ArchetypeId,
    pub component_ids: Vec<u32>,
    pub columns: Vec<Vec<u8>>,  // SoA: one column per component
    pub entities: Vec<u64>,
    pub row_count: usize,
    pub free_rows: VecDeque<usize>,
}
```

### API

```rust
impl ComponentStore {
    pub fn spawn(&mut self, entity_id: u64, components: &[(u32, &[u8])]);
    pub fn despawn(&mut self, entity_id: u64);
    pub fn read_field(&self, entity_id: u64, component_id: u32, offset: usize, size: usize) -> Option<Vec<u8>>;
    pub fn write_field(&mut self, entity_id: u64, component_id: u32, offset: usize, data: &[u8]) -> bool;
    pub fn query(&self, component_ids: &[u32]) -> Vec<u64>;
    pub fn column_slice(&self, archetype_id: ArchetypeId, component_id: u32) -> Option<&[u8]>;
}
```

**Features:**
- Archetype-based storage (entities with same components share archetype)
- SoA column layout (cache-friendly iteration)
- Free-row reuse (swap-remove pattern)
- Contiguous column slices for GPU upload

---

## Type Registry (type_registry.rs)

### Component Type Information

```rust
pub struct TypeRegistry {
    types: HashMap<u32, ComponentTypeInfo>,
}

pub struct ComponentTypeInfo {
    pub id: u32,
    pub name: String,
    pub size: usize,
    pub alignment: usize,
    pub fields: Vec<FieldLayout>,
    pub flags: u32,
    pub archetype_id: Option<ArchetypeId>,
}

pub struct FieldLayout {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub field_type: FieldType,
}
```

### Archetype ID Derivation

```rust
impl ArchetypeId {
    pub fn from_components(component_ids: &[u32]) -> Self {
        let mut sorted = component_ids.to_vec();
        sorted.sort();
        let hash = compute_hash(&sorted);
        Self(hash)
    }
}
```

---

## Test Coverage

| Test Category | Count | Coverage |
|---------------|-------|----------|
| FrameAllocator | 5 | Alloc, align, reset, overflow |
| PoolAllocator | 4 | Acquire, release, size classes |
| StackAllocator | 4 | Push, pop, nested |
| ComponentStore | 35 | Spawn, despawn, read, write, query |
| TypeRegistry | 8 | Register, lookup, archetype derivation |

**Status:** Tests comprehensive, **cannot compile** due to missing exports.

---

## Blocking Issues

### 1. Not exported from lib.rs

```rust
// Need:
pub mod memory;
pub mod component_store;
pub mod type_registry;
```

### 2. No PyO3 bridge

Python `trinity/descriptors/rust_storage.py` calls `_omega.component_read()` etc., which don't exist.

---

## Recommendations

1. **Export from lib.rs** - Immediate
2. **Add PyO3 bindings** - component_read, component_write, etc.
3. **Wire to Python ECS** - Replace fallback with Rust path

---

## Python Counterpart

| Rust | Python | Status |
|------|--------|--------|
| FrameAllocator | engine/core/memory/linear.py | Parallel |
| PoolAllocator | engine/core/memory/pool.py | Parallel |
| StackAllocator | engine/core/memory/stack.py | Parallel |
| ComponentStore | engine/core/ecs/world.py | Python primary |
| TypeRegistry | trinity/metaclasses/component_meta.py | Python primary |

---
