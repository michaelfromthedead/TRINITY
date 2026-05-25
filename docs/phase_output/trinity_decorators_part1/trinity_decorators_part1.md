# Trinity Decorators Archaeological Investigation - Part 1

**Date**: 2026-05-22
**Scope**: Top 20 files by size in `trinity/decorators/` (~10,900 lines examined)
**Classification**: REAL vs STUB analysis

## Executive Summary

**Result: 100% REAL code - no stubs detected**

All 20 files examined contain fully implemented, production-quality code built on a sophisticated Ops-based decorator system. The decorator architecture follows a Chomsky grammar with 7 fundamental operations (TAG, HOOK, REGISTER, DESCRIBE, TRACK, VALIDATE, INTERCEPT) that compose into approximately 100+ domain-specific decorators across 54 tiers.

## Classification Table

| File | Lines | Classification | Decorators | Evidence |
|------|-------|----------------|------------|----------|
| `gpu.py` | 886 | REAL | 8 | WGPU layout algorithms, buffer allocation, struct alignment math |
| `__init__.py` | 857 | REAL | N/A | Re-export orchestration of ~150 symbols |
| `registry.py` | 704 | REAL | N/A | Thread-safe singleton, tier system (54 tiers), validation logic |
| `compilation.py` | 654 | REAL | 7 | FFI binding, platform detection, unavailability stubs |
| `base.py` | 644 | REAL | N/A | Decorator tracking, attribute attachment, validation utilities |
| `dev.py` | 643 | REAL | 9 | Profile timing with stats, trace logging, deprecation warnings |
| `memory.py` | 640 | REAL | 12 | Pool allocation, flyweight registry, CoW semantics, atomic ops |
| `ops.py` | 588 | REAL | N/A | Core 7 Ops implementation, `make_decorator` factory, rule validation |
| `ecs_core.py` | 580 | REAL | 9 | ComponentMeta integration, query extraction, system registration |
| `bridges_caching.py` | 516 | REAL | 9 | Retry backoff, throttle validation, observable patterns |
| `destruction.py` | 477 | REAL | 6 | Destructible config, fracture patterns, joint physics |
| `audio_extended.py` | 461 | REAL | 8 | DSP nodes, voice priority, sidechain compression |
| `data_flow.py` | 455 | REAL | 4 | Serialization methods, network config, snapshot history |
| `scheduling.py` | 445 | REAL | 12 | Fixed timestep, async coroutine detection, chain linking |
| `rendering.py` | 443 | REAL | 6 | GI contribution, shadow casting, reflection probes |
| `modding.py` | 421 | REAL | 9 | Mod metadata, version validation, dependency tracking |
| `ai_generation.py` | 411 | REAL | 7 | Example accumulation, pattern categories, complexity annotations |
| `physics_sim.py` | 380 | REAL | 7 | Solver types, CCD modes, buoyancy, wind physics |
| `crafting.py` | 379 | REAL | 5 | Recipe config, ingredient properties, loot tables |
| `particles_vfx.py` | 360 | REAL | 6 | Particle simulation modes, GPU compute, trail rendering |

## Key Architectural Findings

### 1. Ops-Based Decorator System (`ops.py`)

The system implements a Chomsky grammar with exactly 7 primitive operations:

```python
class Op(Enum):
    TAG = "tag"        # Attach queryable metadata
    HOOK = "hook"      # Wire lifecycle callbacks
    REGISTER = "register"  # Add to named registry
    DESCRIBE = "describe"  # Extract schema from annotations
    TRACK = "track"    # Enable change monitoring
    VALIDATE = "validate"  # Enforce constraints
    INTERCEPT = "intercept"  # Wrap field access
```

Every decorator is composed from these primitives via `make_decorator()`:

```python
def make_decorator(
    name: str,
    steps: Union[list[Step], Callable[..., list[Step]]],
    doc: str = "",
    validate: Optional[Callable[..., None]] = None,
    after_steps: Optional[Callable[[Any, dict[str, Any]], Any]] = None,
)
```

### 2. Tier System (`registry.py`)

54 tiers organized from Foundation (0-9) to Advanced Systems (46-53):

```python
class Tier(IntEnum):
    COMPILATION = 0    # @native, @ffi, @target
    ECS_CORE = 1       # @component, @system, @query
    MEMORY = 2         # @pooled, @aligned, @atomic
    SCHEDULING = 3     # @parallel, @exclusive, @fixed
    DATA_FLOW = 4      # @serializable, @networked
    GPU = 5            # @gpu_buffer, @shader, @render_pass
    # ... through to ...
    BRIDGES_CACHING = 53  # @cached, @lazy, @retry
```

### 3. GPU Layout Algorithms (`gpu.py`)

Real WGSL-compliant struct layout computation:

```python
def _compute_gpu_struct_layout(schema: dict[str, Any]) -> dict[str, Any]:
    """Compute full WGSL-compatible struct layout from annotations."""
    fields: list[dict[str, Any]] = []
    offset = 0
    struct_align = 4

    for field_name, field_type in schema.items():
        info = _get_gpu_type_info(field_type)
        field_align = info["align"]
        aligned_offset = _round_up(field_align, offset)
        # ... WGSL alignment rules
```

wgpu usage flag resolution with WebGPU spec compliance:

```python
_WGPU_USAGE_FLAGS: dict[str, int] = {
    "vertex": 0x0020,
    "index": 0x0010,
    "uniform": 0x0040,
    "storage": 0x0080,
    # ...
}

def _resolve_wgpu_usage_flags(usage: frozenset[str]) -> int:
    # WebGPU spec: STORAGE and INDIRECT buffers require COPY_DST
    if "storage" in usage or "indirect" in usage:
        flags |= _WGPU_USAGE_FLAGS["copy_dst"]
```

### 4. Memory Decorators (`memory.py`)

Flyweight pattern with working registry:

```python
def _after_flyweight(target: Any, params: dict[str, Any]) -> Any:
    target._flyweight_registry: dict[int, Any] = {}
    target._flyweight_next_id = 0

    def flyweight_init(self: Any, *args: Any, **kwargs: Any) -> None:
        cls_type = type(self)
        flyweight_id = cls_type._flyweight_next_id
        cls_type._flyweight_next_id += 1
        self._flyweight_id = flyweight_id
        cls_type._flyweight_registry[flyweight_id] = self

    target.__init__ = flyweight_init
```

Atomic operations with threading.RLock:

```python
def _after_atomic(target: Any, params: dict[str, Any]) -> Any:
    target._atomic_lock = threading.RLock()

    def fetch_add(self: Any, delta: int) -> int:
        with type(self)._atomic_lock:
            old = getattr(self, "value", 0)
            self.value = old + delta
            return old
```

### 5. Scheduling System (`scheduling.py`)

Fixed timestep with Hz/delta calculation:

```python
def _after_fixed(target: Any, params: dict[str, Any]) -> Any:
    hz = params.get("hz", DEFAULT_PHYSICS_HZ)
    target._fixed = True
    target._fixed_hz = hz
    target._fixed_delta = 1.0 / hz
```

Async coroutine detection:

```python
def _after_async_system(target: Any, params: dict[str, Any]) -> Any:
    target._async_system = True
    target._is_coroutine = asyncio.iscoroutinefunction(target)
```

### 6. ECS Integration (`ecs_core.py`)

Direct metaclass integration:

```python
def _after_component(target: Any, params: dict[str, Any]) -> Any:
    if not hasattr(target, "_component_id"):
        from trinity.metaclasses import ComponentMeta
        ComponentMeta._process_fields(target)
        ComponentMeta._install_descriptors(target)
        with ComponentMeta._lock:
            target._component_id = ComponentMeta._next_id
            ComponentMeta._next_id += 1
            ComponentMeta._registry[target._component_id] = target
```

Query extraction from type hints:

```python
def _extract_queries(fn: Callable[..., Any]) -> list[Any]:
    hints = get_type_hints(fn)
    for param_name, param_type in hints.items():
        origin = get_origin(param_type)
        if origin.__name__ == "Query":
            queries.append(param_type)
```

### 7. Data Flow Serialization (`data_flow.py`)

Working serialize/deserialize implementation:

```python
@classmethod
def serialize(cls, obj: Any) -> dict:
    data = {"__version__": cls._serializable_version, "__type__": cls.__name__}
    for field in cls._serializable_fields:
        if hasattr(obj, field):
            data[field] = getattr(obj, field)
    return data

@classmethod
def deserialize(cls, data: dict) -> Any:
    obj = cls.__new__(cls)
    for field in cls._serializable_fields:
        if field in data:
            setattr(obj, field, data[field])
    return obj
```

Snapshot history with ring buffer:

```python
def snapshot_save(self) -> int:
    if len(self._snapshot_history) >= self._snapshot_history_frames:
        self._snapshot_history.pop(0)  # Ring buffer
    self._snapshot_history.append(state)
```

### 8. Profile Decorator (`dev.py`)

Real timing statistics:

```python
def _after_profile(target: Any, params: dict[str, Any]) -> Any:
    stats = {
        "call_count": 0,
        "total_ms": 0.0,
        "min_ms": float("inf"),
        "max_ms": 0.0,
    }

    @functools.wraps(target)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = target(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        stats["call_count"] += 1
        stats["total_ms"] += elapsed
        # ...
```

## Validation Logic Quality

All decorators include comprehensive parameter validation. Examples:

```python
# gpu.py
def _validate_render_pass(color_attachments: int = 1, msaa: int = 1, **_: Any) -> None:
    if color_attachments < 1:
        raise ValueError(f"@render_pass: color_attachments must be >= 1")
    if msaa not in VALID_MSAA_SAMPLES:
        raise ValueError(f"@render_pass: msaa must be power of 2 (1, 2, 4, 8, 16)")

# bridges_caching.py
def _validate_retry_params(**kwargs: Any) -> None:
    if base_delay_ms > max_delay_ms:
        raise ValueError(f"base_delay_ms ({base_delay_ms}) must be <= max_delay_ms")
```

## Decorator Count by Category

| Category | Decorators | Tier |
|----------|------------|------|
| GPU | 8 | 5 |
| Compilation | 7 | 0 |
| Memory | 12 | 2 |
| Scheduling | 12 | 3 |
| ECS Core | 9 | 1 |
| Data Flow | 4 | 4 |
| Dev | 9 | 6 |
| Bridges/Caching | 9 | 53 |
| Destruction | 6 | 43 |
| Audio Extended | 8 | 49 |
| Rendering | 6 | 42 |
| Modding | 9 | 30 |
| AI Generation | 7 | 9 |
| Physics Sim | 7 | 46 |
| Crafting | 5 | 52 |
| Particles/VFX | 6 | 45 |

**Total in Part 1: ~124 decorators examined**

## Dependency Analysis

The decorator system has clear internal dependencies:

1. `ops.py` - Foundation (no dependencies)
2. `registry.py` - Foundation (no dependencies)
3. `base.py` - Depends on `registry.py`
4. All domain decorators depend on `ops.py`, `registry.py`, `base.py`
5. `ecs_core.py` - Additional dependency on `trinity.metaclasses`
6. `__init__.py` - Re-exports all modules

External dependencies (standard library only):
- `threading` (registry, memory)
- `asyncio` (scheduling)
- `functools` (dev wrappers)
- `time` (profiling)
- `warnings` (deprecation)
- `dataclasses` (all configs)
- `typing` (all files)

## Conclusion

The `trinity/decorators/` package is a sophisticated, fully implemented decorator system with no stubs. The code quality is high, featuring:

1. **Clean architecture**: 7 primitive Ops compose into 100+ decorators
2. **Proper validation**: All parameters validated with descriptive errors
3. **Thread safety**: Where needed (registry, atomic operations)
4. **Real algorithms**: GPU layout, profile timing, snapshot history
5. **Metaclass integration**: ECS components properly registered

This is production-quality Python code implementing a declarative game engine configuration system.
