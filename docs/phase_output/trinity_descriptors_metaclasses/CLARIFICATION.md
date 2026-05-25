# CLARIFICATION: Trinity Descriptors, Metaclasses, and Tools

## Philosophical Framing

### The Three-Layer Architecture

Trinity implements a three-layer metaprogramming architecture that separates concerns while enabling powerful composition:

1. **Decorator Layer** - User-facing annotations that declare intent
2. **Descriptor Layer** - Field-level behavior implementation
3. **Metaclass Layer** - Class-level registration and validation

This separation allows each layer to evolve independently while maintaining clear integration contracts.

### Why Composable Descriptors?

Traditional ECS implementations hardcode field behaviors. A networked component requires different code than a cached component. Trinity's descriptor composition pattern allows behaviors to be mixed:

```python
# A field can be networked AND tracked AND validated AND cached
# Each descriptor handles its concern without knowing about the others
Networked -> Tracked -> Validated -> Cached -> Storage
```

The composition order matters: outermost descriptors see the composed behavior of inner descriptors. This enables powerful stacking without code duplication.

### Why Metaclasses Instead of Base Classes?

Metaclasses operate at class definition time, enabling:

1. **Automatic Registration** - Components register themselves without explicit calls
2. **Validation Before Runtime** - Mutable defaults detected at definition, not first use
3. **Step Recording** - Every metaclass action is recorded for introspection
4. **Transparent Integration** - User classes don't need to know about infrastructure

Base class inheritance would require explicit super() calls and couldn't intercept class creation itself.

## Design Rationale

### Descriptor Protocol Extensions

Python's descriptor protocol (`__get__`, `__set__`, `__delete__`, `__set_name__`) is extended with:

- **Lifecycle hooks** (`pre_get`, `post_get`, `pre_set`, `post_set`) - Enable cross-cutting concerns like logging, validation, or transformation
- **Composition metadata** (`inner`, `accepts_inner`, `accepts_outer`, `excludes`) - Enable safe chaining with validation
- **Read tracking** via `ContextVar` - Enable incremental computation by knowing what was read

### The Step Pattern

Every layer records its actions as Steps:

```python
cls._metaclass_steps.append(Step(Op.REGISTER, {"registry": "component_registry"}))
cls._metaclass_steps.append(Step(Op.TAG, {"key": "component_id", "value": 42}))
```

This enables:
- **Introspection** - Tools can examine what happened during class creation
- **Debugging** - step_trace.py shows the complete action history
- **Validation** - lint.py can verify composition rules were followed
- **Coverage** - op_coverage.py tracks which operations are used

### Rust Integration Strategy

The `RustStorageDescriptor` bridges Python and Rust:

1. **Primary Path**: Routes reads/writes to Rust component store via `_omega` module
2. **Fallback Path**: Uses `__dict__` storage when Rust unavailable
3. **Type Mapping**: `float->f32`, `int->i32`, `bool->u8`, `str->string`

This allows the same Python code to run in pure-Python mode (development, testing) or with Rust acceleration (production).

### Metaclass Hierarchy

All engine metaclasses inherit from `EngineMeta`, which provides:

- Global type registry for debugging
- Step recording infrastructure
- Thread-safe registry operations
- Clean `__repr__` for engine types

Specialized metaclasses add their domain concerns:

| Metaclass | Domain | Key Capability |
|-----------|--------|----------------|
| ComponentMeta | ECS data | Unique IDs, pooling, budget, layout |
| SystemMeta | ECS logic | Dependencies, parallelization, hot reload |
| EventMeta | Messages | Data-only validation, pooling, channels |
| StateMeta | FSMs | Transitions, hierarchy, history |
| AssetMeta | Resources | Loading, hot reload, dependencies |
| ProtocolMeta | Network | Versioning, negotiation, migration |
| ResourceMeta | Singletons | Ordered initialization, lazy loading |

## Key Design Decisions

### D1: Mutable Default Rejection

ComponentMeta rejects mutable defaults at class definition time:

```python
class Bad(Component):
    items: list = []  # REJECTED - shared mutable default
```

Rationale: Shared mutable defaults are a notorious Python footgun. Catching this at definition time prevents subtle bugs that would only manifest at runtime with multiple instances.

### D2: Pool Management at Metaclass Level

Components manage their own object pools via metaclass infrastructure:

- `return_to_pool()` - Return instance for reuse
- `pool_stats()` - Introspect pool state
- `max_instances` - Budget enforcement

Rationale: ECS workloads create/destroy many components. Pool management reduces allocation pressure without polluting user code.

### D3: System Dependency Analysis

SystemMeta analyzes `@reads`/`@writes` declarations to:

1. Determine execution order via topological sort
2. Identify parallelizable groups
3. Detect resource conflicts

Rationale: Manual dependency management is error-prone. Declarative dependencies enable automatic optimization.

### D4: Tools as First-Class Citizens

Development tools (step_trace, lint, op_coverage, doctor) are integrated into the framework, not afterthoughts.

Rationale: Metaprogramming systems are notoriously hard to debug. Built-in introspection tools make the system approachable.

## Integration Points

### Cross-Directory Dependencies

1. **Descriptor -> Metaclass**: ComponentMeta reads field type hints and installs appropriate descriptors
2. **Metaclass -> Tools**: Tools introspect `_metaclass_steps` and `_field_descriptors` attributes
3. **Rust Bridge**: Both RustStorageDescriptor and ComponentMeta use `_omega` for SoA storage
4. **Foundation**: TrackedDescriptor and ComponentMeta integrate with Foundation's central tracker

### Foundation Integration

The Foundation module provides:
- Central tracker for change notification
- EventLog for audit trails
- Provenance tracking for debugging

Both descriptors and metaclasses integrate with these services when available.
