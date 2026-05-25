# Investigation: trinity/decorators

## Summary

The Trinity decorator system is a REAL, sophisticated implementation built on a 7-primitive Op system (TAG, HOOK, REGISTER, DESCRIBE, TRACK, VALIDATE, INTERCEPT). All ~275 decorators are compositions of these primitives, with the `make_decorator()` factory being the single authoritative way to create them. The system includes full metaclass integration, field descriptors, GPU struct layout computation, and actual serialization/networking logic.

## Files

| File | Lines | Category | Notes |
|------|-------|----------|-------|
| `__init__.py` | 857 | Index | Exports ~275 decorators across 53 tiers |
| `ops.py` | 589 | Core | The 7 Ops + Step + make_decorator factory |
| `registry.py` | ~600 | Core | DecoratorRegistry singleton, 53 Tiers |
| `base.py` | 645 | Core | Tracking, validation, introspection |
| `ecs_core.py` | 581 | ECS | component, tag, resource, event, system, query, bundle, relation, derived |
| `gpu.py` | 887 | GPU | gpu_buffer, gpu_kernel, gpu_struct, shader, render_pass + WGSL layout |
| `data_flow.py` | 456 | Networking | serializable, networked, snapshot, versioned |
| `memory.py` | ~500 | Memory | pooled, packed, aligned, arena, flyweight, budget, allocator |
| `scheduling.py` | ~400 | Scheduling | phase, parallel, exclusive, after, before, run_if, fixed, job |
| `introspection.py` | 100 | API | primitives(), composites(), chain(), find_decorators() |
| 50+ other files | varies | Domain | Audio, AI, VFX, physics, gameplay, etc. |

## Decorator Categories (53 Tiers)

| Tier | Name | Decorators |
|------|------|------------|
| 0 | COMPILATION | native, ffi, target, unsafe, backend, capability, platform |
| 1 | ECS_CORE | component, tag, resource, event, system, query, bundle, relation, derived |
| 2 | MEMORY | pooled, packed, aligned, arena, flyweight, intern, generations, copy_on_write, inline_array, budget, allocator, atomic |
| 3 | SCHEDULING | phase, parallel, exclusive, after, before, run_if, fixed, job, async_system, throttle, deferred, chain |
| 4 | DATA_FLOW | serializable, networked, snapshot, versioned |
| 5 | GPU | gpu_buffer, gpu_kernel, gpu_struct, bind_group, dispatch, shader, render_pass, async_compute |
| 6 | DEV | profile, gpu_profile, trace, reloadable, editor, test, bench, invariant, deprecated |
| 7 | LIFECYCLE | on_add, on_remove, on_change, on_spawn, on_despawn |
| 8+ | Extended | 45 more tiers covering audio, AI, VFX, physics, gameplay, networking, etc. |

## How Decorators Work

The system is NOT stubs. Decorators are built from 7 primitive operations:

1. **TAG** - Attach queryable metadata to `target._tags` dict
2. **HOOK** - Wire lifecycle callbacks to `target._hooks` dict
3. **REGISTER** - Add target to named registries in `target._registries`
4. **DESCRIBE** - Extract schema from type annotations to `target._schema`
5. **TRACK** - Enable change monitoring via `target._tracked_fields`
6. **VALIDATE** - Enforce constraints via `target._constraints`
7. **INTERCEPT** - Wrap field get/set/delete via `target._intercepts`

Every decorator is created via `make_decorator(name, steps, after_steps)`:
- `steps`: A function returning `list[Step]` based on params
- `after_steps`: Domain-specific post-processing (register with metaclass, compute layouts)

## Key Decorators

### @component

```python
# Steps: TAG(component=True), TAG(component_name), REGISTER(ecs_core)
# After: Calls ComponentMeta._process_fields(), assigns _component_id, registers in ComponentMeta._registry
```

What it does: Registers a class as an ECS component. Integrates with ComponentMeta metaclass to assign unique IDs, process field types, and install descriptors.

### @system

```python
# Steps: TAG(system=True), TAG(system_phase=phase), REGISTER(ecs_core)
# After: Extracts Query/Res types from signature, sets _system_queries, _system_resources, registers with SystemMeta
```

What it does: Marks a function/class as an ECS system. Extracts parameter dependencies and registers with the scheduler.

### @gpu_buffer

```python
# Steps: TAG(gpu_buffer=True), TAG(gpu_buffer_config=GpuBufferConfig), REGISTER(gpu), DESCRIBE
# After: Computes WGSL struct layout (_gpu_buffer_size, _gpu_buffer_alignment), resolves wgpu usage bitmask
```

What it does: Marks a class as a GPU buffer. Computes WGSL-compatible memory layout with proper alignment (vec3 -> align 16, mat4 -> 64 bytes). Provides `allocate_wgpu_buffer()` and `create_wgpu_buffer()` to create actual wgpu buffers.

### @networked

```python
# Steps: TAG(networked=True), TAG(networked_config=NetworkedConfig), REGISTER(data_flow)
# After: Attaches _serialize_net/_deserialize_net methods, sets relevance/authority/priority attrs
```

What it does: Configures network replication. Adds serialization methods and marks fields for delta compression, interpolation, and prediction.

### @serializable

```python
# Steps: TAG(serializable=True), TAG(serializable_config), REGISTER(data_flow), DESCRIBE
# After: Adds serialize()/deserialize() classmethods, extracts _serializable_fields from annotations
```

What it does: Adds binary/JSON/msgpack serialization. Creates actual serialize/deserialize methods that iterate over annotated fields.

### @pooled

```python
# Steps: TAG(pool={initial_size, grow_factor, max_size}), HOOK(on_create), HOOK(on_destroy), REGISTER(PoolManager)
```

What it does: Pre-allocates memory pools. Wires lifecycle hooks for object reuse.

## Verdict

**REAL IMPLEMENTATION**

This is a production-grade decorator system with:
- Full metaclass integration (ComponentMeta, SystemMeta, EventMeta, ResourceMeta)
- Working GPU struct layout computation with WGSL alignment rules
- Actual serialization/deserialization methods
- Network replication configuration with delta/interpolation support
- Introspection API (primitives(), composites(), decompose())
- 53-tier ordering system with composition rules and validation

The only "stub" aspects are the runtime execution environments (actual wgpu device, actual network transport) which are external dependencies.

## Evidence

### Op-based Architecture (ops.py:32-42)

```python
class Op(Enum):
    """The 7 operations. Everything a decorator can do."""
    TAG = "tag"
    HOOK = "hook"
    REGISTER = "register"
    DESCRIBE = "describe"
    TRACK = "track"
    VALIDATE = "validate"
    INTERCEPT = "intercept"
```

### make_decorator Factory (ops.py:450-553)

```python
def make_decorator(
    name: str,
    steps: Union[list[Step], Callable[..., list[Step]]],
    doc: str = "",
    validate: Optional[Callable[..., None]] = None,
    after_steps: Optional[Callable[[Any, dict[str, Any]], Any]] = None,
):
    """Create a decorator from a list of Steps."""
    # ...builds decorator factory, tracks metadata, supports @decorator and @decorator()
```

### GPU Struct Layout (gpu.py:519-568)

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
        # ...computes offsets, sizes, alignment per WGSL rules
```

### ComponentMeta Integration (ecs_core.py:204-223)

```python
def _after_component(target: Any, params: dict[str, Any]) -> Any:
    name = params.get("name") or target.__name__
    target._component = True
    target._component_name = name
    if not hasattr(target, "_component_id"):
        from trinity.metaclasses import ComponentMeta
        ComponentMeta._process_fields(target)
        ComponentMeta._install_descriptors(target)
        with ComponentMeta._lock:
            target._component_id = ComponentMeta._next_id
            ComponentMeta._next_id += 1
            ComponentMeta._registry[target._component_id] = target
```

### Introspection API (introspection.py)

```python
def primitives(cls: type, field: Optional[str] = None) -> list[Step]:
    """Return primitive Steps for a class or field."""
    return decompose(cls)

def composites(cls: type, field: Optional[str] = None) -> list[str]:
    """Return composite decorator/descriptor names for a class or field."""
    return getattr(cls, "_applied_decorators", []).copy()
```
