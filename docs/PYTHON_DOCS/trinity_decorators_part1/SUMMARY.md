# SUMMARY: trinity/decorators (Part 1)

## Quantitative Metrics

| Metric | Value |
|--------|-------|
| Total Lines Examined | ~10,900 |
| Files Analyzed | 20 |
| Classification | 100% REAL |
| Stubs Found | 0 |
| Decorators Documented | 124 |
| Tiers Defined | 54 |
| Primitive Ops | 7 |
| Re-exported Symbols | ~150 |

---

## File Inventory

| File | Lines | Decorators | Category |
|------|-------|------------|----------|
| gpu.py | 886 | 8 | GPU |
| __init__.py | 857 | N/A | Re-exports |
| registry.py | 704 | N/A | Infrastructure |
| compilation.py | 654 | 7 | Compilation |
| base.py | 644 | N/A | Infrastructure |
| dev.py | 643 | 9 | Development |
| memory.py | 640 | 12 | Memory |
| ops.py | 588 | N/A | Infrastructure |
| ecs_core.py | 580 | 9 | ECS |
| bridges_caching.py | 516 | 9 | Caching |
| destruction.py | 477 | 6 | Physics |
| audio_extended.py | 461 | 8 | Audio |
| data_flow.py | 455 | 4 | Serialization |
| scheduling.py | 445 | 12 | Scheduling |
| rendering.py | 443 | 6 | Rendering |
| modding.py | 421 | 9 | Modding |
| ai_generation.py | 411 | 7 | AI |
| physics_sim.py | 380 | 7 | Physics |
| crafting.py | 379 | 5 | Gameplay |
| particles_vfx.py | 360 | 6 | VFX |

---

## Algorithm Inventory

| Algorithm | Status | File | Evidence |
|-----------|--------|------|----------|
| WGSL struct layout | REAL | gpu.py | _compute_gpu_struct_layout() with alignment math |
| wgpu usage flags | REAL | gpu.py | _resolve_wgpu_usage_flags() with WebGPU spec compliance |
| Flyweight registry | REAL | memory.py | _after_flyweight() with ID assignment |
| Atomic operations | REAL | memory.py | _after_atomic() with RLock |
| Profile timing | REAL | dev.py | _after_profile() with perf_counter |
| Query extraction | REAL | ecs_core.py | _extract_queries() with type hints |
| Snapshot history | REAL | data_flow.py | Ring buffer implementation |
| Retry backoff | REAL | bridges_caching.py | Exponential backoff with jitter |
| Fixed timestep | REAL | scheduling.py | Hz to delta conversion |
| Component registration | REAL | ecs_core.py | ComponentMeta integration |

---

## Key Evidence Snippets

### GPU Layout (gpu.py)
```python
def _compute_gpu_struct_layout(schema: dict[str, Any]) -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    offset = 0
    struct_align = 4
    for field_name, field_type in schema.items():
        info = _get_gpu_type_info(field_type)
        field_align = info["align"]
        aligned_offset = _round_up(field_align, offset)
```

### Ops Grammar (ops.py)
```python
class Op(Enum):
    TAG = "tag"
    HOOK = "hook"
    REGISTER = "register"
    DESCRIBE = "describe"
    TRACK = "track"
    VALIDATE = "validate"
    INTERCEPT = "intercept"
```

### Flyweight Pattern (memory.py)
```python
def _after_flyweight(target: Any, params: dict[str, Any]) -> Any:
    target._flyweight_registry: dict[int, Any] = {}
    target._flyweight_next_id = 0
```

### Thread Safety (registry.py)
```python
_lock = threading.RLock()
with _lock:
    target._component_id = ComponentMeta._next_id
    ComponentMeta._next_id += 1
```

---

## Dependency Graph

```
ops.py (foundation)
    |
    v
registry.py (foundation)
    |
    v
base.py
    |
    +---> ecs_core.py ---> trinity.metaclasses
    |
    +---> gpu.py
    |
    +---> memory.py
    |
    +---> scheduling.py
    |
    +---> data_flow.py
    |
    +---> [all other domain decorators]
    |
    v
__init__.py (re-exports all)
```

---

## External Dependencies

| Module | Usage |
|--------|-------|
| threading | Registry lock, atomic operations |
| asyncio | Coroutine detection |
| functools | Wrapper preservation |
| time | Profiling |
| warnings | Deprecation |
| dataclasses | Config classes |
| typing | Type hints |

All dependencies are Python standard library. No external packages required.
