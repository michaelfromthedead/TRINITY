# CLARIFICATION: Trinity Decorators Design Rationale

## Philosophical Framing

The Trinity decorator system embodies the principle that **game engine configuration should be declarative, composable, and type-safe**. Rather than imperative setup code scattered across initialization routines, the system expresses intent through decorator composition.

## Why Chomsky Grammar?

The choice of exactly 7 primitive operations (TAG, HOOK, REGISTER, DESCRIBE, TRACK, VALIDATE, INTERCEPT) follows formal language theory:

1. **Completeness**: These 7 primitives can express any decorator behavior
2. **Orthogonality**: Each primitive has one responsibility
3. **Composability**: Primitives combine via `make_decorator()` without interference

This mirrors how programming languages define a minimal set of keywords that compose into infinite programs.

## Why 54 Tiers?

The tier system (0-53) serves two purposes:

1. **Dependency ordering**: Lower tiers load before higher tiers (Compilation=0 before GPU=5)
2. **Semantic grouping**: Related decorators share a tier (all physics at 46)

This prevents circular dependencies and enables the registry to reason about decorator application order.

## Design Decisions

### Ops as Foundation

```
@gpu_buffer(usage=["storage", "vertex"])
```

Internally becomes:

```python
make_decorator(
    "gpu_buffer",
    steps=[
        Step(Op.TAG, {"_gpu_buffer": True}),
        Step(Op.DESCRIBE, {"schema": compute_gpu_layout}),
        Step(Op.VALIDATE, {"validate_fn": _validate_gpu_buffer}),
    ]
)
```

This decomposition means new decorators require no new infrastructure - just new step combinations.

### Thread-Safe Registry

The registry uses a singleton pattern with `threading.RLock()` because:

1. Multiple threads may register components during engine init
2. ECS queries may iterate the registry while systems register
3. Hot-reload scenarios need atomic registration

### WGSL Layout Computation

GPU struct layout follows WebGPU spec strictly:

- `vec3<f32>` aligns to 16 bytes (not 12)
- Struct alignment = max member alignment
- STORAGE and INDIRECT buffers require COPY_DST flag

This is not approximation - it's spec-compliant computation for cross-platform GPU buffer creation.

### Flyweight Pattern for Memory

The flyweight registry assigns monotonic IDs because:

1. Dense arrays benefit from small integer keys
2. ID stability survives serialization round-trips
3. Weak references can expire without ID collision

### Fixed Timestep for Physics

```python
hz = params.get("hz", DEFAULT_PHYSICS_HZ)
target._fixed_delta = 1.0 / hz
```

This ensures physics simulation determinism - independent of frame rate, physics steps are constant-time.

## Trade-offs Accepted

1. **Decorator overhead**: Each decorated function has wrapper indirection. Accepted because hot paths use native code, not Python.

2. **Metaclass coupling**: `@component` tightly couples to `ComponentMeta`. Accepted because this integration is the entire point - decorators configure metaclass behavior.

3. **Global registry**: Singleton registry is global mutable state. Accepted because game engines have exactly one component registry per process.

## What This Is Not

- **Not a framework**: This is infrastructure for Trinity's declarative layer
- **Not optional**: Core engine systems depend on decorator metadata
- **Not runtime-modifiable**: Decorators configure at import time, not runtime

## Integration Points

- `trinity/metaclasses/ComponentMeta` - ECS component registration
- `trinity/gpu/` - GPU buffer creation consumes layout metadata
- `trinity/scheduling/` - System runner reads scheduling metadata
- `trinity/net/` - Serialization uses `@serializable` metadata
