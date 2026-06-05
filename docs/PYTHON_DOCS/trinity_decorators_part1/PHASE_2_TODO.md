# PHASE 2 TODO: GPU and Memory Decorators

## Summary
Implement GPU buffer/shader decorators and memory management decorators.

---

## T-DEC-2.1: Implement GPU Type Info Table

**File**: `trinity/decorators/gpu.py`

**Task**: Map WGSL types to size/alignment info.

**Acceptance Criteria**:
- [ ] `_get_gpu_type_info(type_name) -> {"size": int, "align": int}`
- [ ] f32, i32, u32: size=4, align=4
- [ ] vec2<f32>: size=8, align=8
- [ ] vec3<f32>: size=12, align=16 (WGSL spec)
- [ ] vec4<f32>: size=16, align=16
- [ ] mat4x4<f32>: size=64, align=16
- [ ] Unknown types raise ValueError

---

## T-DEC-2.2: Implement GPU Struct Layout Computation

**File**: `trinity/decorators/gpu.py`

**Task**: Compute WGSL-compliant struct layout from schema dict.

**Acceptance Criteria**:
- [ ] `_compute_gpu_struct_layout(schema) -> {"fields": [...], "size": int, "align": int}`
- [ ] Each field has: name, type, offset, size, align
- [ ] Offset computed with `_round_up(align, current_offset)`
- [ ] Struct alignment = max of all field alignments
- [ ] Total size padded to struct alignment

---

## T-DEC-2.3: Implement wgpu Usage Flag Resolution

**File**: `trinity/decorators/gpu.py`

**Task**: Resolve usage strings to WebGPU bitflags.

**Acceptance Criteria**:
- [ ] `_resolve_wgpu_usage_flags(usage: frozenset[str]) -> int`
- [ ] Flags: vertex=0x0020, index=0x0010, uniform=0x0040, storage=0x0080, indirect=0x0100
- [ ] STORAGE and INDIRECT auto-add COPY_DST (0x0008)
- [ ] Unknown flags raise ValueError

---

## T-DEC-2.4: Implement @gpu_buffer Decorator

**File**: `trinity/decorators/gpu.py`

**Task**: Decorator for GPU buffer configuration.

**Acceptance Criteria**:
- [ ] `@gpu_buffer(usage=["storage", "vertex"])`
- [ ] Validates usage flags
- [ ] Computes struct layout from class annotations
- [ ] Attaches: `_gpu_buffer=True`, `_gpu_usage`, `_gpu_layout`

---

## T-DEC-2.5: Implement @render_pass Decorator

**File**: `trinity/decorators/gpu.py`

**Task**: Decorator for render pass configuration.

**Acceptance Criteria**:
- [ ] `@render_pass(color_attachments=1, msaa=4, depth_stencil=True)`
- [ ] Validates: color_attachments >= 1
- [ ] Validates: msaa in {1, 2, 4, 8, 16}
- [ ] Attaches: `_render_pass=True`, `_render_pass_config`

---

## T-DEC-2.6: Implement @shader, @compute_pass, @texture, etc.

**File**: `trinity/decorators/gpu.py`

**Task**: Remaining GPU decorators (8 total).

**Acceptance Criteria**:
- [ ] @shader - shader module configuration
- [ ] @compute_pass - compute dispatch configuration
- [ ] @texture - texture binding configuration
- [ ] @sampler - sampler configuration
- [ ] @bind_group - bind group layout
- [ ] @pipeline_layout - pipeline layout
- [ ] @vertex_buffer - vertex buffer layout
- [ ] Each has appropriate validation

---

## T-DEC-2.7: Implement @flyweight Decorator

**File**: `trinity/decorators/memory.py`

**Task**: Flyweight pattern with registry.

**Acceptance Criteria**:
- [ ] Attaches `_flyweight_registry: dict[int, Any]` to class
- [ ] Attaches `_flyweight_next_id: int = 0`
- [ ] Wraps `__init__` to assign `_flyweight_id` and register
- [ ] Instance lookup by ID works

---

## T-DEC-2.8: Implement @atomic Decorator

**File**: `trinity/decorators/memory.py`

**Task**: Atomic operations via RLock.

**Acceptance Criteria**:
- [ ] Attaches `_atomic_lock: threading.RLock` to class
- [ ] Adds `fetch_add(self, delta) -> int` method
- [ ] Adds `fetch_sub(self, delta) -> int` method
- [ ] Adds `compare_exchange(self, expected, desired) -> bool` method
- [ ] All operations acquire lock

---

## T-DEC-2.9: Implement @pooled Decorator

**File**: `trinity/decorators/memory.py`

**Task**: Pool allocation decorator.

**Acceptance Criteria**:
- [ ] `@pooled(pool="physics", max_instances=1000)`
- [ ] Pre-allocates instance pool
- [ ] Wraps `__new__` to return from pool
- [ ] Provides `release()` method to return to pool
- [ ] Validates max_instances > 0

---

## T-DEC-2.10: Implement @aligned, @cow, @pinned, etc.

**File**: `trinity/decorators/memory.py`

**Task**: Remaining memory decorators (12 total).

**Acceptance Criteria**:
- [ ] @aligned - alignment specification
- [ ] @cow - copy-on-write semantics
- [ ] @pinned - prevent GC movement
- [ ] @interned - string/value interning
- [ ] @arena - arena allocation
- [ ] @stack_alloc - stack-based allocation
- [ ] @ref_counted - reference counting
- [ ] Each with appropriate validation

---

## Dependencies

```
PHASE 1 ──> T-DEC-2.1 ──> T-DEC-2.2 ──> T-DEC-2.4
                │                         │
                v                         v
            T-DEC-2.3 ──────────────> T-DEC-2.5 ──> T-DEC-2.6

PHASE 1 ──> T-DEC-2.7 ──> T-DEC-2.8 ──> T-DEC-2.9 ──> T-DEC-2.10
```

## Estimated Effort

| Task | Lines | Complexity |
|------|-------|------------|
| T-DEC-2.1 | ~40 | Low |
| T-DEC-2.2 | ~60 | Medium |
| T-DEC-2.3 | ~30 | Low |
| T-DEC-2.4 | ~50 | Medium |
| T-DEC-2.5 | ~40 | Low |
| T-DEC-2.6 | ~200 | Medium |
| T-DEC-2.7 | ~50 | Medium |
| T-DEC-2.8 | ~60 | Medium |
| T-DEC-2.9 | ~80 | Medium |
| T-DEC-2.10 | ~200 | Medium |
