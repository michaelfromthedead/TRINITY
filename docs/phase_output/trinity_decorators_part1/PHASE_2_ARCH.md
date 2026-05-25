# PHASE 2 ARCHITECTURE: GPU and Memory Decorators

## Phase Scope

Low-level resource decorators: `gpu.py` (Tier 5), `memory.py` (Tier 2)

## Architecture Decisions

### ADR-DEC-005: WGSL-Compliant Struct Layout

**Context**: GPU buffers require precise alignment per WebGPU spec.

**Decision**: Implement `_compute_gpu_struct_layout()` with:
- Type info lookup via `_get_gpu_type_info()`
- Alignment rounding via `_round_up()`
- Struct alignment = max of all field alignments
- Explicit padding bytes between fields

**Consequences**:
- Cross-platform GPU buffer creation works correctly
- vec3 aligns to 16 bytes (not 12) per WGSL spec
- Layout metadata is queryable for debugging

### ADR-DEC-006: wgpu Usage Flag Resolution

**Context**: Different buffer types require different usage flags.

**Decision**: Bitflag resolution with spec-mandated additions:
```python
_WGPU_USAGE_FLAGS = {
    "vertex": 0x0020,
    "index": 0x0010,
    "uniform": 0x0040,
    "storage": 0x0080,
    "indirect": 0x0100,
    "copy_src": 0x0004,
    "copy_dst": 0x0008,
}
```

**Consequences**:
- STORAGE and INDIRECT automatically get COPY_DST
- Flag combination via bitwise OR
- Invalid flag combinations caught at validation time

### ADR-DEC-007: Flyweight Pattern

**Context**: Many small objects need efficient memory and stable IDs.

**Decision**: Per-class flyweight registry:
- `_flyweight_registry: dict[int, Any]` on class
- `_flyweight_next_id: int` monotonic counter
- Instance gets `_flyweight_id` at init

**Consequences**:
- Dense integer keys for array storage
- ID survives serialization
- Registry enables iteration of all instances

### ADR-DEC-008: Atomic Operations via RLock

**Context**: Thread-safe primitive operations needed.

**Decision**: Attach `_atomic_lock: threading.RLock` to class, provide:
- `fetch_add(delta) -> old_value`
- `fetch_sub(delta) -> old_value`
- `compare_exchange(expected, desired) -> bool`

**Consequences**:
- Python-level atomicity (not CPU-level)
- Sufficient for Python threading model
- RLock allows recursive acquisition

## Component Diagram

```
+-------------+
|   gpu.py    |  @gpu_buffer, @shader, @render_pass, @compute_pass
+------+------+
       |
       +-- _compute_gpu_struct_layout() --> WGSL alignment
       |
       +-- _resolve_wgpu_usage_flags() --> WebGPU usage bits
       |
       +-- _validate_render_pass() --> MSAA power-of-2 check

+-------------+
|  memory.py  |  @pooled, @aligned, @flyweight, @atomic, @cow
+------+------+
       |
       +-- _after_flyweight() --> Registry/ID setup
       |
       +-- _after_atomic() --> RLock + fetch_add/sub
       |
       +-- Pool allocation hooks
```

## GPU Type Info Table

| WGSL Type | Size | Align | Notes |
|-----------|------|-------|-------|
| f32 | 4 | 4 | |
| i32 | 4 | 4 | |
| u32 | 4 | 4 | |
| vec2<f32> | 8 | 8 | |
| vec3<f32> | 12 | 16 | **Align 16, not 12** |
| vec4<f32> | 16 | 16 | |
| mat4x4<f32> | 64 | 16 | |

## Memory Decorator Patterns

### Pool Allocation
```python
@pooled(pool="physics", max_instances=1000)
class RigidBody:
    ...
```
- Pre-allocates pool of instances
- Allocation returns from pool
- Deallocation returns to pool

### Copy-on-Write
```python
@cow
class SharedState:
    ...
```
- Tracks reference count
- Clone on first mutation when refs > 1
- Zero-copy when single owner

## Validation Rules

| Decorator | Rule |
|-----------|------|
| @gpu_buffer | usage must be valid frozenset |
| @render_pass | MSAA in {1, 2, 4, 8, 16} |
| @render_pass | color_attachments >= 1 |
| @aligned | alignment must be power of 2 |
| @pooled | max_instances > 0 |
